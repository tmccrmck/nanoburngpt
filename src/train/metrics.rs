use std::sync::Arc;

use burn::{
    prelude::*,
    tensor::{ExecutionError, TensorReadError},
    train::metric::{
        Adaptor, ItemLazy, LossInput, Metric, MetricAttributes, MetricMetadata, MetricName,
        Numeric, NumericAttributes, NumericEntry, SerializedEntry,
    },
};

/// Training/validation output carrying only the scalar loss and token count so
/// the `[batch * seq, vocab]` logits tensor is freed immediately after loss
/// computation instead of being retained across Burn's async metric channel.
pub struct LmLossOutput {
    pub loss: Tensor<1>,
    pub num_tokens: usize,
}

impl ItemLazy for LmLossOutput {
    fn sync(self) -> Result<Self, ExecutionError> {
        self.loss.device().flush()?;
        Ok(Self {
            loss: self.loss.without_autodiff(),
            num_tokens: self.num_tokens,
        })
    }
}

impl Adaptor<LossInput> for LmLossOutput {
    fn adapt(&self) -> LossInput {
        LossInput::new(self.loss.clone())
    }
}

pub struct LossPerplexityInput {
    pub loss: Tensor<1>,
    pub num_tokens: usize,
}

impl Adaptor<LossPerplexityInput> for LmLossOutput {
    fn adapt(&self) -> LossPerplexityInput {
        LossPerplexityInput {
            loss: self.loss.clone(),
            num_tokens: self.num_tokens,
        }
    }
}

/// Computes perplexity directly from the scalar cross-entropy loss (`exp(NLL / tokens)`),
/// avoiding Burn's default `PerplexityMetric` which recomputes a second full `log_softmax`
/// over the vocabulary on every micro-batch.
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
    type Input = LossPerplexityInput;

    fn update(
        &mut self,
        item: &Self::Input,
        _metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        let mean_loss = item.loss.clone().try_into_scalar::<f64>()?;
        self.sum_nll += mean_loss * item.num_tokens as f64;
        self.total_tokens += item.num_tokens;
        self.current_ppl = mean_loss.exp();
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
        let device = Device::flex();
        let mut metric = LossPerplexityMetric::new();
        let meta = MetricMetadata {
            progress: burn::data::dataloader::Progress::new(0, 1, None),
            iteration: Some(1),
            lr: None,
        };

        let in1 = LossPerplexityInput {
            loss: Tensor::<1>::from_floats([2.0], &device),
            num_tokens: 10,
        };
        let in2 = LossPerplexityInput {
            loss: Tensor::<1>::from_floats([4.0], &device),
            num_tokens: 30,
        };
        metric.update(&in1, &meta).unwrap();
        metric.update(&in2, &meta).unwrap();

        let expected = 3.5_f64.exp();
        match metric.final_value() {
            NumericEntry::Final(ppl) => {
                assert!((ppl - expected).abs() < 1e-5, "got {ppl}, expected {expected}");
            }
            other => panic!("unexpected final_value: {other:?}"),
        }
    }
}
