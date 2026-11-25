use std::collections::HashMap;
use std::time::{Duration, Instant};

use burn::train::{
    logger::{EvaluationProgressLogger, TrainingProgressLogger},
    metric::{MetricDefinition, MetricId},
    renderer::{
        EvaluationName, MetricState, MetricsRenderer, MetricsRendererEvaluation,
        MetricsRendererTraining,
    },
};

/// Compact plain-text metrics renderer for headless runs (`--no-tui` / cloud).
/// Prints live loss, perplexity, and tokens/sec instead of raw `ProgressSnapshot` structs.
pub struct CompactCliRenderer {
    block_size: usize,
    metric_names: HashMap<MetricId, String>,
    epoch: usize,
    total_epochs: usize,
    split_name: String,
    split_total: usize,
    split_items: usize,
    split_start: Instant,
    last_print: Instant,
    printed_in_split: bool,
    train_loss: Option<f64>,
    train_ppl: Option<f64>,
    valid_loss: Option<f64>,
    valid_ppl: Option<f64>,
    best_valid_ppl: Option<f64>,
}

impl CompactCliRenderer {
    pub fn new(block_size: usize) -> Self {
        let now = Instant::now();
        Self {
            block_size,
            metric_names: HashMap::new(),
            epoch: 1,
            total_epochs: 0,
            split_name: String::new(),
            split_total: 0,
            split_items: 0,
            split_start: now,
            last_print: now,
            printed_in_split: false,
            train_loss: None,
            train_ppl: None,
            valid_loss: None,
            valid_ppl: None,
            best_valid_ppl: None,
        }
    }

    fn record_metric(&mut self, is_train: bool, state: &MetricState) {
        if let MetricState::Numeric(entry, Some(numeric)) = state {
            let val = numeric.current();
            let Some(name) = self.metric_names.get(&entry.metric_id) else {
                return;
            };
            match (is_train, name.as_str()) {
                (true, "Loss") => self.train_loss = Some(val),
                (true, "Perplexity") => self.train_ppl = Some(val),
                (false, "Loss") => self.valid_loss = Some(val),
                (false, "Perplexity") => self.valid_ppl = Some(val),
                _ => {}
            }
        }
    }
}

impl MetricsRendererTraining for CompactCliRenderer {
    fn update_train(&mut self, state: MetricState) {
        self.record_metric(true, &state);
    }

    fn update_valid(&mut self, state: MetricState) {
        self.record_metric(false, &state);
    }
}

impl TrainingProgressLogger for CompactCliRenderer {
    fn start(
        &mut self,
        total_epochs: usize,
        starting_epoch: usize,
        _total_items: Option<usize>,
        _label: Option<&str>,
    ) {
        self.total_epochs = total_epochs;
        self.epoch = starting_epoch.max(1);
    }

    fn start_split(&mut self, split_name: &str, total_items: usize) {
        self.split_name = split_name.to_string();
        self.split_total = total_items;
        self.split_items = 0;
        self.split_start = Instant::now();
        self.last_print = self.split_start;
        self.printed_in_split = false;
        if split_name == "train" {
            self.train_loss = None;
            self.train_ppl = None;
        } else {
            self.valid_loss = None;
            self.valid_ppl = None;
        }
    }

    fn update_split(&mut self, items_processed: usize) {
        self.split_items = items_processed;
        if self.split_name != "train" {
            return;
        }
        let now = Instant::now();
        let due = !self.printed_in_split
            || now.duration_since(self.last_print) >= Duration::from_secs(10);
        if due && items_processed < self.split_total {
            self.last_print = now;
            self.printed_in_split = true;
            let elapsed = now.duration_since(self.split_start).as_secs_f64().max(1e-6);
            let tok_s = (items_processed * self.block_size) as f64 / elapsed;
            let loss_s = self
                .train_loss
                .map_or_else(|| "-".into(), |v| format!("{v:.4}"));
            let ppl_s = self
                .train_ppl
                .map_or_else(|| "-".into(), |v| format!("{v:.2}"));
            println!(
                "[ep {:>2}/{:<2} | {:>5}/{:<5}] train_loss={loss_s}  ppl={ppl_s}  tok/s={tok_s:.0}",
                self.epoch, self.total_epochs, items_processed, self.split_total
            );
        }
    }

    fn update_epoch(&mut self, epoch: usize) {
        self.epoch = epoch + 1;
    }

    fn end_split(&mut self) {
        let elapsed = self.split_start.elapsed().as_secs_f64().max(1e-6);
        let tok_s = (self.split_items * self.block_size) as f64 / elapsed;
        if self.split_name == "train" {
            let loss_s = self
                .train_loss
                .map_or_else(|| "-".into(), |v| format!("{v:.4}"));
            let ppl_s = self
                .train_ppl
                .map_or_else(|| "-".into(), |v| format!("{v:.2}"));
            println!(
                "[ep {:>2}/{:<2} | train] loss={loss_s}  ppl={ppl_s}  tok/s={tok_s:.0}  ({elapsed:.1}s)",
                self.epoch, self.total_epochs
            );
        } else {
            let loss_s = self
                .valid_loss
                .map_or_else(|| "-".into(), |v| format!("{v:.4}"));
            let ppl_s = self
                .valid_ppl
                .map_or_else(|| "-".into(), |v| format!("{v:.2}"));
            let best_tag = match (self.valid_ppl, self.best_valid_ppl) {
                (Some(p), None) => {
                    self.best_valid_ppl = Some(p);
                    "  (new best)"
                }
                (Some(p), Some(best)) if p < best => {
                    self.best_valid_ppl = Some(p);
                    "  (new best)"
                }
                _ => "",
            };
            println!(
                "[ep {:>2}/{:<2} | valid] loss={loss_s}  ppl={ppl_s}{best_tag}  ({elapsed:.1}s)",
                self.epoch, self.total_epochs
            );
        }
    }

    fn end(&mut self) {}

    fn log_event_training(&mut self, _event: String) {}
}

impl EvaluationProgressLogger for CompactCliRenderer {
    fn start_global_progress(&mut self, _total_tests: usize) {}
    fn start_test(&mut self, _name: &str, _total_items: usize) {}
    fn update_test_progress(&mut self, _items_processed: usize) {}
    fn end_test(&mut self) {}
    fn end_global_progress(&mut self) {}
    fn log_event_evaluation(&mut self, _event: String) {}
}

impl MetricsRendererEvaluation for CompactCliRenderer {
    fn update_test(&mut self, _name: EvaluationName, _state: MetricState) {}
}

impl MetricsRenderer for CompactCliRenderer {
    fn manual_close(&mut self) {}

    fn register_metric(&mut self, definition: MetricDefinition) {
        self.metric_names
            .insert(definition.metric_id, definition.name.to_string());
    }
}
