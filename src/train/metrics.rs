use std::sync::Arc;

use burn::{
    prelude::*,
    tensor::{ExecutionError, TensorReadError},
    train::metric::{
        Adaptor, ItemLazy, Metric, MetricAttributes, MetricMetadata, MetricName, Numeric,
        NumericAttributes, NumericEntry, SerializedEntry,
    },
};

/// Training/validation output carrying the scalar loss tensor and token count.
/// Used by `GPT::forward_loss` (which needs the autodiff tensor for `.backward()`).
pub struct LmLossOutput {
    pub loss: Tensor<1>,
    pub num_tokens: usize,
}

/// Plain host-side scalar loss output passed across Burn's async metric channel.
///
/// Reading the scalar on the worker thread before `optimizer_step()` has two major benefits:
/// 1. Cuts GPU→CPU scalar readbacks in half (one readback feeds both Loss and Perplexity,
///    with no redundant `.mean()` kernel from Burn's `LossMetric`).
/// 2. Breaks Burn 0.22.1's `DdpTrainEpoch` 3-way deadlock where `AsyncProcessorTraining`
///    called `cuStreamSynchronize` on a GPU stream already blocked inside NCCL `all_reduce`.
#[derive(Clone, Copy, Debug)]
pub struct ScalarLossOutput {
    pub loss: f64,
    pub num_tokens: usize,
}

impl ItemLazy for ScalarLossOutput {
    fn sync(self) -> Result<Self, ExecutionError> {
        Ok(self)
    }
}

impl Adaptor<ScalarLossOutput> for ScalarLossOutput {
    fn adapt(&self) -> ScalarLossOutput {
        *self
    }
}

/// Token-weighted mean cross-entropy loss metric reading directly from [`ScalarLossOutput`]
/// with zero GPU operations.
#[derive(Clone)]
pub struct ScalarLossMetric {
    name: MetricName,
    sum_nll: f64,
    total_tokens: usize,
    current_loss: f64,
    current_tokens: usize,
}

impl Default for ScalarLossMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl ScalarLossMetric {
    pub fn new() -> Self {
        Self {
            name: Arc::new("Loss".to_string()),
            sum_nll: 0.0,
            total_tokens: 0,
            current_loss: f64::NAN,
            current_tokens: 0,
        }
    }

    fn running_loss(&self) -> f64 {
        if self.total_tokens > 0 {
            self.sum_nll / self.total_tokens as f64
        } else {
            f64::NAN
        }
    }

    fn format_entry(&self, final_entry: bool) -> SerializedEntry {
        let epoch_loss = self.running_loss();
        let formatted = format!("epoch {epoch_loss:.4} - batch {:.4}", self.current_loss);
        let serialized = if final_entry {
            NumericEntry::Final(epoch_loss).serialize()
        } else {
            NumericEntry::Aggregated {
                aggregated_value: self.current_loss,
                count: self.current_tokens,
            }
            .serialize()
        };
        SerializedEntry::new(formatted, serialized)
    }
}

impl Metric for ScalarLossMetric {
    type Input = ScalarLossOutput;

    fn update(
        &mut self,
        item: &Self::Input,
        _metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        self.sum_nll += item.loss * item.num_tokens as f64;
        self.total_tokens += item.num_tokens;
        self.current_loss = item.loss;
        self.current_tokens = item.num_tokens;
        Ok(self.format_entry(false))
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        Ok(self.format_entry(true))
    }

    fn clear(&mut self) {
        self.sum_nll = 0.0;
        self.total_tokens = 0;
        self.current_loss = f64::NAN;
        self.current_tokens = 0;
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: false,
        }
        .into()
    }
}

impl Numeric for ScalarLossMetric {
    fn value(&self) -> Option<NumericEntry> {
        Some(NumericEntry::Aggregated {
            aggregated_value: self.current_loss,
            count: self.current_tokens,
        })
    }

    fn running_value(&self) -> Option<NumericEntry> {
        Some(NumericEntry::Aggregated {
            aggregated_value: self.running_loss(),
            count: self.total_tokens,
        })
    }

    fn final_value(&self) -> NumericEntry {
        NumericEntry::Final(self.running_loss())
    }
}

/// Computes perplexity directly from [`ScalarLossOutput`] (`exp(NLL / tokens)`) with zero
/// GPU operations, avoiding Burn's default `PerplexityMetric` (which recomputes `log_softmax`
/// over the full vocabulary) and avoiding a second GPU→CPU sync.
#[derive(Clone)]
pub struct LossPerplexityMetric {
    name: MetricName,
    sum_nll: f64,
    total_tokens: usize,
    current_ppl: f64,
    current_tokens: usize,
}

impl Default for LossPerplexityMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl LossPerplexityMetric {
    pub fn new() -> Self {
        Self {
            name: Arc::new("Perplexity".to_string()),
            sum_nll: 0.0,
            total_tokens: 0,
            current_ppl: f64::NAN,
            current_tokens: 0,
        }
    }

    fn running_ppl(&self) -> f64 {
        if self.total_tokens > 0 {
            (self.sum_nll / self.total_tokens as f64).exp()
        } else {
            f64::INFINITY
        }
    }

    fn format_entry(&self, final_entry: bool) -> SerializedEntry {
        let epoch_ppl = self.running_ppl();
        let formatted = format!("epoch {epoch_ppl:.2} - batch {:.2}", self.current_ppl);
        let serialized = if final_entry {
            NumericEntry::Final(epoch_ppl).serialize()
        } else {
            NumericEntry::Aggregated {
                aggregated_value: self.current_ppl,
                count: self.current_tokens,
            }
            .serialize()
        };
        SerializedEntry::new(formatted, serialized)
    }
}

impl Metric for LossPerplexityMetric {
    type Input = ScalarLossOutput;

    fn update(
        &mut self,
        item: &Self::Input,
        _metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        self.sum_nll += item.loss * item.num_tokens as f64;
        self.total_tokens += item.num_tokens;
        self.current_ppl = item.loss.exp();
        self.current_tokens = item.num_tokens;
        Ok(self.format_entry(false))
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        Ok(self.format_entry(true))
    }

    fn clear(&mut self) {
        self.sum_nll = 0.0;
        self.total_tokens = 0;
        self.current_ppl = f64::NAN;
        self.current_tokens = 0;
    }

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: false,
        }
        .into()
    }
}

impl Numeric for LossPerplexityMetric {
    fn value(&self) -> Option<NumericEntry> {
        Some(NumericEntry::Aggregated {
            aggregated_value: self.current_ppl,
            count: self.current_tokens,
        })
    }

    fn running_value(&self) -> Option<NumericEntry> {
        Some(NumericEntry::Aggregated {
            aggregated_value: self.running_ppl(),
            count: self.total_tokens,
        })
    }

    fn final_value(&self) -> NumericEntry {
        NumericEntry::Final(self.running_ppl())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loss_perplexity_metric_matches_exp_weighted_mean_loss() {
        let mut loss_metric = ScalarLossMetric::new();
        let mut ppl_metric = LossPerplexityMetric::new();
        let meta = MetricMetadata {
            progress: burn::data::dataloader::Progress::new(0, 1, None),
            iteration: Some(1),
            lr: None,
        };

        let in1 = ScalarLossOutput {
            loss: 2.0,
            num_tokens: 10,
        };
        let in2 = ScalarLossOutput {
            loss: 4.0,
            num_tokens: 30,
        };
        loss_metric.update(&in1, &meta).unwrap();
        loss_metric.update(&in2, &meta).unwrap();
        ppl_metric.update(&in1, &meta).unwrap();
        ppl_metric.update(&in2, &meta).unwrap();

        match loss_metric.final_value() {
            NumericEntry::Final(loss) => assert!((loss - 3.5).abs() < 1e-6),
            other => panic!("unexpected loss final_value: {other:?}"),
        }

        let expected = 3.5_f64.exp();
        match ppl_metric.final_value() {
            NumericEntry::Final(ppl) => {
                assert!((ppl - expected).abs() < 1e-5, "got {ppl}, expected {expected}");
            }
            other => panic!("unexpected ppl final_value: {other:?}"),
        }
    }
}
