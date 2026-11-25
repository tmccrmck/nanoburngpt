use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{
    data::{BpeTokenizer, TextGenerationBatch, TextGenerationBatcher},
    datasets::Dataset,
    model::{GPT, GPTConfig},
};
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    grad_clipping::GradientClippingConfig,
    module::{Module, ParamGroup},
    nn::loss::CrossEntropyLossConfig,
    optim::{
        AdamWConfig, AdjustLrFn, ModuleOptimizer, MuonConfig,
        decay::WeightDecayConfig,
        lr_scheduler::{LrScheduler, LrSchedulerRecord},
    },
    prelude::*,
    tensor::{
        ExecutionError, TensorReadError,
        distributed::{DistributedConfig, ReduceOperation},
    },
    train::{
        ExecutionStrategy, InferenceStep, Learner, MultiDeviceOptim, SupervisedTraining,
        TrainOutput, TrainStep, TrainingStrategy,
        checkpoint::{
            ComposedCheckpointingStrategy, KeepLastNCheckpoints, MetricCheckpointingStrategy,
        },
        logger::{EvaluationProgressLogger, TrainingProgressLogger},
        metric::{
            Adaptor, ItemLazy, LossInput, LossMetric, Metric, MetricAttributes, MetricDefinition,
            MetricId, MetricMetadata, MetricName, Numeric, NumericAttributes, NumericEntry,
            SerializedEntry,
            store::{Aggregate, Direction, Split},
        },
        renderer::{
            EvaluationName, MetricState, MetricsRenderer, MetricsRendererEvaluation,
            MetricsRendererTraining,
        },
    },
};

// ---------------------------------------------------------------------------
// Multi-GPU & Optimizer selection
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizerChoice {
    AdamW,
    Muon,
}

impl FromStr for OptimizerChoice {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "adamw" => Ok(Self::AdamW),
            "muon" => Ok(Self::Muon),
            other => anyhow::bail!("Unknown --optimizer '{other}'. Available: adamw, muon"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiGpuMode {
    Off,
    Auto,
    Sharded,
    Ddp,
}

impl FromStr for MultiGpuMode {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "auto" => Ok(Self::Auto),
            "sharded" => Ok(Self::Sharded),
            "ddp" => Ok(Self::Ddp),
            other => anyhow::bail!(
                "Unknown --multi-gpu mode '{other}'. Available: auto, sharded, ddp, off"
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Training configuration
// ---------------------------------------------------------------------------

#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    /// Peak learning rate (also the warmup target).
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    /// Linear warmup steps. 0 = no warmup.
    #[config(default = 100)]
    pub warmup_iters: usize,
    #[config(default = 10)]
    pub num_epochs: usize,
    /// 0 = full dataset; positive value caps training items (smoke tests).
    #[config(default = 0)]
    pub max_train_items: usize,
    /// 0 = full validation set; positive value caps validation items.
    #[config(default = 0)]
    pub max_val_items: usize,
    /// Directory where checkpoints, metrics, config.json, and model weights are saved.
    #[config(default = "String::from(\"artifacts\")")]
    pub artifact_dir: String,
    /// Resume training from a saved checkpoint epoch (e.g. Some(5)).
    pub resume_epoch: Option<usize>,
    /// Gradient accumulation steps (1 = no accumulation)
    #[config(default = 1)]
    pub grad_accum_steps: usize,
    /// Enable activation checkpointing to reduce training VRAM
    #[config(default = false)]
    pub grad_checkpointing: bool,
    /// Disable the interactive TUI and use plain CLI progress output
    #[config(default = false)]
    pub no_tui: bool,
}

// ---------------------------------------------------------------------------
// Warmup + cosine decay LR scheduler (matches nanoGPT's get_lr())
// ---------------------------------------------------------------------------

/// Linear warmup for `warmup_iters` steps, then cosine decay to `min_lr`
/// over the remaining `total_iters - warmup_iters` steps.
#[derive(Clone, Debug)]
pub struct WarmupCosineScheduler {
    warmup_iters: usize,
    total_iters: usize,
    max_lr: f64,
    min_lr: f64,
    current_iter: usize,
}

impl WarmupCosineScheduler {
    pub fn new(max_lr: f64, min_lr: f64, warmup_iters: usize, total_iters: usize) -> Self {
        Self {
            warmup_iters,
            total_iters,
            max_lr,
            min_lr,
            current_iter: usize::MAX,
        }
    }

    fn lr_at(&self, iter: usize) -> f64 {
        let effective_warmup = self.warmup_iters.min(self.total_iters.saturating_sub(1));
        if effective_warmup > 0 && iter < effective_warmup {
            // Linear ramp: 0 → max_lr over effective_warmup steps
            self.max_lr * (iter + 1) as f64 / effective_warmup as f64
        } else if iter >= self.total_iters {
            // Hold at min_lr after the decay is complete
            self.min_lr
        } else {
            // Cosine decay: max_lr → min_lr
            let decay_iters = (self.total_iters - effective_warmup).max(1);
            let progress = (iter - effective_warmup) as f64 / decay_iters as f64;
            let coeff = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            self.min_lr + coeff * (self.max_lr - self.min_lr)
        }
    }
}

impl LrScheduler for WarmupCosineScheduler {
    fn step(&mut self) -> f64 {
        // wrapping_add so first step() computes lr_at(0), matching
        // Burn's CosineAnnealing convention (current_iter starts at MAX).
        self.current_iter = self.current_iter.wrapping_add(1);
        self.lr_at(self.current_iter)
    }

    fn to_record(&self) -> LrSchedulerRecord {
        LrSchedulerRecord::new().with_scalar("current_iter", self.current_iter as u64)
    }

    fn load_record(&mut self, record: LrSchedulerRecord) {
        if let Some(current_iter) = record.scalar::<u64>("current_iter") {
            self.current_iter = current_iter as usize;
        }
    }
}

// ---------------------------------------------------------------------------
// Lightweight scalar-loss output & perplexity metric (avoids retaining logits)
// ---------------------------------------------------------------------------

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
/// over the 50,257-token vocabulary on every micro-batch.
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

// ---------------------------------------------------------------------------
// Model training/inference steps
// ---------------------------------------------------------------------------

impl GPT {
    pub fn forward_loss(&self, item: TextGenerationBatch) -> LmLossOutput {
        let targets = item.targets; // [batch, seq]
        let [batch_size, seq_len] = targets.dims();
        let num_tokens = batch_size * seq_len;

        let logits = self.forward(item.inputs); // [batch, seq, vocab]
        let [_, _, vocab_size] = logits.dims();

        let logits_flat = logits.reshape([num_tokens, vocab_size]);
        let targets_flat = targets.reshape([num_tokens]);

        let loss = CrossEntropyLossConfig::new()
            .init(&logits_flat.device())
            .forward(logits_flat, targets_flat);

        LmLossOutput { loss, num_tokens }
    }
}

impl TrainStep for GPT {
    type Input = TextGenerationBatch;
    type Output = LmLossOutput;

    fn step(&self, item: TextGenerationBatch) -> TrainOutput<LmLossOutput> {
        let loss_scale = item.loss_scale;
        let out = self.forward_loss(item);
        let grads = if (loss_scale - 1.0).abs() > 1e-9 {
            (out.loss.clone() * loss_scale).backward()
        } else {
            out.loss.backward()
        };
        TrainOutput::new(self, grads, out)
    }
}

impl InferenceStep for GPT {
    type Input = TextGenerationBatch;
    type Output = LmLossOutput;

    fn step(&self, item: TextGenerationBatch) -> LmLossOutput {
        self.forward_loss(item)
    }
}

// ---------------------------------------------------------------------------
// Compact headless CLI renderer (replaces noisy CliMetricsRenderer)
// ---------------------------------------------------------------------------

/// Compact plain-text metrics renderer for headless runs (`--no-tui` / cloud).
/// Prints live loss, perplexity, and tokens/sec instead of raw `ProgressSnapshot` structs.
struct CompactCliRenderer {
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
    fn new(block_size: usize) -> Self {
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
        let due = !self.printed_in_split || now.duration_since(self.last_print) >= Duration::from_secs(10);
        if due && items_processed < self.split_total {
            self.last_print = now;
            self.printed_in_split = true;
            let elapsed = now.duration_since(self.split_start).as_secs_f64().max(1e-6);
            let tok_s = (items_processed * self.block_size) as f64 / elapsed;
            let loss_s = self.train_loss.map_or_else(|| "-".into(), |v| format!("{v:.4}"));
            let ppl_s = self.train_ppl.map_or_else(|| "-".into(), |v| format!("{v:.2}"));
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
            let loss_s = self.train_loss.map_or_else(|| "-".into(), |v| format!("{v:.4}"));
            let ppl_s = self.train_ppl.map_or_else(|| "-".into(), |v| format!("{v:.2}"));
            println!(
                "[ep {:>2}/{:<2} | train] loss={loss_s}  ppl={ppl_s}  tok/s={tok_s:.0}  ({elapsed:.1}s)",
                self.epoch, self.total_epochs
            );
        } else {
            let loss_s = self.valid_loss.map_or_else(|| "-".into(), |v| format!("{v:.4}"));
            let ppl_s = self.valid_ppl.map_or_else(|| "-".into(), |v| format!("{v:.2}"));
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

// ---------------------------------------------------------------------------
// Training entry point
// ---------------------------------------------------------------------------

pub(crate) fn compute_total_optimizer_steps(
    dataset_len: usize,
    batch_size: usize,
    grad_accum_steps: usize,
    num_devices: usize,
    num_epochs: usize,
) -> usize {
    let batch_size = batch_size.max(1);
    let grad_accum = grad_accum_steps.max(1);
    let num_devices = num_devices.max(1);
    let per_device_len = dataset_len.div_ceil(num_devices);
    let micro_steps_per_epoch = per_device_len.div_ceil(batch_size);
    let opt_steps_per_epoch = micro_steps_per_epoch.div_ceil(grad_accum);
    opt_steps_per_epoch * num_epochs
}

pub(crate) fn build_optimizer(choice: OptimizerChoice) -> ModuleOptimizer {
    let grad_clip = GradientClippingConfig::Norm(1.0);
    let norm_group = ParamGroup::from_any_predicates(vec!["ln_1", "ln_2", "ln_f"]);
    let norm_adamw = AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.95)
        .with_weight_decay(0.0)
        .with_epsilon(1e-8)
        .build();

    let base_optim = AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.95)
        .with_weight_decay(0.1)
        .with_epsilon(1e-8)
        .with_grad_clipping(Some(grad_clip.clone()))
        .init()
        .with_group(norm_group, norm_adamw, Some(grad_clip.init()));

    match choice {
        OptimizerChoice::AdamW => base_optim,
        OptimizerChoice::Muon => {
            let hidden_2d_group = ParamGroup::from_any_predicates(vec!["c_attn", "c_proj", "c_fc"]);
            let muon = MuonConfig::new()
                .with_adjust_lr_fn(AdjustLrFn::MatchRmsAdamW)
                .with_weight_decay(Some(WeightDecayConfig::new(0.1)))
                .build();
            base_optim.with_group(hidden_2d_group, muon, Some(grad_clip.init()))
        }
    }
}

/// Find the epoch with the lowest final validation perplexity whose checkpoint
/// file exists under `<artifact_dir>/checkpoint/model-<epoch>.bpk` and copy it
/// to `<artifact_dir>/model_best.bpk`.
pub(crate) fn export_best_checkpoint(artifact_dir: &Path) -> anyhow::Result<Option<(usize, f64)>> {
    let valid_dir = artifact_dir.join("valid");
    if !valid_dir.exists() {
        return Ok(None);
    }

    let mut best: Option<(usize, f64)> = None;
    for entry in std::fs::read_dir(&valid_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let Some(epoch_str) = name.to_str().and_then(|s| s.strip_prefix("epoch-")) else {
            continue;
        };
        let Ok(epoch) = epoch_str.parse::<usize>() else {
            continue;
        };
        let ckpt_path = artifact_dir
            .join("checkpoint")
            .join(format!("model-{epoch}.bpk"));
        if !ckpt_path.exists() {
            continue;
        }
        let ppl_log = entry.path().join("Perplexity.log");
        if !ppl_log.exists() {
            continue;
        }
        let text = std::fs::read_to_string(&ppl_log)?;
        let final_ppl = text.lines().rev().find_map(|line| {
            let (val, tag) = line.trim().split_once(',')?;
            (tag == "final").then(|| val.parse::<f64>().ok()).flatten()
        });
        if let Some(ppl) = final_ppl
            && ppl.is_finite()
            && best.is_none_or(|(_, best_ppl)| ppl < best_ppl)
        {
            best = Some((epoch, ppl));
        }
    }

    if let Some((epoch, ppl)) = best {
        let src = artifact_dir
            .join("checkpoint")
            .join(format!("model-{epoch}.bpk"));
        let dst = artifact_dir.join("model_best.bpk");
        std::fs::copy(&src, &dst)?;
        return Ok(Some((epoch, ppl)));
    }

    Ok(None)
}

pub fn run_training(
    devices: Vec<Device>,
    multi_gpu: MultiGpuMode,
    optimizer_choice: OptimizerChoice,
    dataset: Dataset,
    mut gpt_config: GPTConfig,
    training_config: TrainingConfig,
) -> anyhow::Result<()> {
    anyhow::ensure!(!devices.is_empty(), "At least one training device is required");

    let use_multi = devices.len() > 1 && multi_gpu != MultiGpuMode::Off;
    let num_devices = if use_multi { devices.len() } else { 1 };
    let grad_accum = training_config.grad_accum_steps.max(1);

    // Burn's GradientsAccumulator sums across micro-batches, and MultiDeviceLearningStrategy
    // sums across devices (whereas DDP uses ncclAvg mean reduction across devices).
    let loss_scale = if use_multi && multi_gpu != MultiGpuMode::Ddp {
        1.0 / (grad_accum * num_devices) as f64
    } else {
        1.0 / grad_accum as f64
    };

    // --- Data ---
    let (train_dataset, val_dataset) = dataset.load(gpt_config.block_size)?;

    // Optionally cap train/validation sets
    let train_dataset = if training_config.max_train_items > 0 {
        train_dataset.with_max_len(training_config.max_train_items)
    } else {
        train_dataset
    };
    let val_dataset = if training_config.max_val_items > 0 {
        val_dataset.with_max_len(training_config.max_val_items)
    } else {
        val_dataset
    };

    gpt_config.vocab_size = BpeTokenizer::VOCAB_SIZE;
    println!("Vocab size: {} (BPE r50k_base)", gpt_config.vocab_size);

    let artifact_dir = Path::new(&training_config.artifact_dir);
    std::fs::create_dir_all(artifact_dir)?;
    gpt_config
        .save(artifact_dir.join("config.json"))
        .map_err(|e| anyhow::anyhow!("Failed to save config: {e:?}"))?;

    // Compute total optimizer steps for the LR schedule (Burn steps LR once per optimizer update)
    let per_device_len = train_dataset.len().div_ceil(num_devices);
    let micro_steps_per_epoch = per_device_len.div_ceil(training_config.batch_size.max(1));
    let total_iters = compute_total_optimizer_steps(
        train_dataset.len(),
        training_config.batch_size,
        grad_accum,
        num_devices,
        training_config.num_epochs,
    );
    println!(
        "Devices: {num_devices}  |  Optimizer: {optimizer_choice:?}  |  Micro-steps/epoch/dev: {micro_steps_per_epoch}  |  Optimizer steps: {total_iters} (accum={grad_accum}, loss_scale={loss_scale:.4})  |  Warmup: {}",
        training_config.warmup_iters
    );

    let batcher_train =
        TextGenerationBatcher::new(gpt_config.block_size).with_loss_scale(loss_scale);
    let batcher_val = TextGenerationBatcher::new(gpt_config.block_size);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(training_config.batch_size)
        .shuffle(1337)
        .num_workers(training_config.num_workers)
        .build(train_dataset);

    let dataloader_val = DataLoaderBuilder::new(batcher_val)
        .batch_size(training_config.batch_size)
        .num_workers(training_config.num_workers)
        .build(val_dataset);

    // --- Optimizer (0.0 WD on 1D norms; optional hybrid Muon for 2D block weights) ---
    let optim = build_optimizer(optimizer_choice);

    // --- LR scheduler: linear warmup → cosine decay ---
    let min_lr = training_config.learning_rate / 10.0;
    let lr_scheduler = WarmupCosineScheduler::new(
        training_config.learning_rate,
        min_lr,
        training_config.warmup_iters,
        total_iters.max(1),
    );

    let main_device = if training_config.grad_checkpointing {
        devices[0].clone().autodiff().gradient_checkpointing()
    } else {
        devices[0].clone().autodiff()
    };
    let model = GPT::new(&gpt_config, &main_device);
    let learner = Learner::new(model, optim, lr_scheduler);

    let strategy = ComposedCheckpointingStrategy::builder()
        .add(MetricCheckpointingStrategy::new(
            &LossPerplexityMetric::new(),
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
        ))
        .add(KeepLastNCheckpoints::new(1))
        .build();

    let exec_strategy = if !use_multi {
        ExecutionStrategy::single(devices[0].clone())
    } else {
        match multi_gpu {
            MultiGpuMode::Ddp => ExecutionStrategy::ddp(
                devices,
                DistributedConfig {
                    all_reduce_op: ReduceOperation::Mean,
                },
            ),
            MultiGpuMode::Auto | MultiGpuMode::Sharded => {
                ExecutionStrategy::multi(devices, MultiDeviceOptim::OptimSharded)
            }
            MultiGpuMode::Off => ExecutionStrategy::single(devices[0].clone()),
        }
    };

    let mut training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_val)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LossPerplexityMetric::new())
        .metric_valid_numeric(LossPerplexityMetric::new())
        .with_default_checkpointers()
        .with_checkpointing_strategy(strategy)
        .with_application_logger(None)
        .with_training_strategy(TrainingStrategy::Default(exec_strategy))
        .num_epochs(training_config.num_epochs)
        .summary();

    if let Some(epoch) = training_config.resume_epoch {
        println!("Resuming from checkpoint epoch {epoch} in {}", artifact_dir.display());
        training = training.checkpoint(epoch);
    }

    if training_config.grad_checkpointing {
        training = training.gradient_checkpointing();
    }

    if grad_accum > 1 {
        training = training.grads_accumulation(grad_accum);
    }

    if training_config.no_tui || cfg!(not(feature = "tui")) {
        training = training.renderer(Box::new(CompactCliRenderer::new(gpt_config.block_size)));
    }

    let result = training.launch(learner);

    if let Some(err) = &result.error {
        anyhow::bail!("Training failed: {err:?}");
    }

    let final_path = artifact_dir.join("model_final");
    result
        .model
        .save_file(&final_path)
        .map_err(|e| anyhow::anyhow!("Failed to save final model: {e:?}"))?;
    println!("Final model saved to {}", final_path.display());

    if let Some((best_epoch, best_ppl)) = export_best_checkpoint(artifact_dir)? {
        println!(
            "Best validation model (epoch {best_epoch}, valid ppl {best_ppl:.2}) exported to {}",
            artifact_dir.join("model_best.bpk").display()
        );
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TextDataset;
    use burn::data::dataset::Dataset as BurnDataset;
    use burn::data::dataloader::batcher::Batcher;

    fn scheduler(max_lr: f64, min_lr: f64, warmup: usize, total: usize) -> WarmupCosineScheduler {
        WarmupCosineScheduler::new(max_lr, min_lr, warmup, total)
    }

    fn steps(s: &mut WarmupCosineScheduler, n: usize) -> Vec<f64> {
        (0..n).map(|_| s.step()).collect()
    }

    #[test]
    fn warmup_ramps_linearly() {
        let mut s = scheduler(1.0, 0.1, 4, 10);
        let lrs = steps(&mut s, 4);
        // iter 0 → 1/4, iter 1 → 2/4, iter 2 → 3/4, iter 3 → 4/4
        assert!((lrs[0] - 0.25).abs() < 1e-10);
        assert!((lrs[1] - 0.50).abs() < 1e-10);
        assert!((lrs[2] - 0.75).abs() < 1e-10);
        assert!((lrs[3] - 1.00).abs() < 1e-10);
    }

    #[test]
    fn cosine_decay_midpoint() {
        // After warmup the cosine schedule at exactly halfway through the
        // decay window should give (max_lr + min_lr) / 2.
        let max_lr = 1.0_f64;
        let min_lr = 0.1_f64;
        let warmup = 0;
        let total = 100;
        let mid = total / 2;
        let mut s = scheduler(max_lr, min_lr, warmup, total);
        let lrs = steps(&mut s, mid + 1);
        let expected = (max_lr + min_lr) / 2.0;
        assert!((lrs[mid] - expected).abs() < 1e-10, "got {}", lrs[mid]);
    }

    #[test]
    fn holds_min_lr_after_total_iters() {
        let mut s = scheduler(1.0, 0.1, 0, 5);
        // Advance past total
        let lrs = steps(&mut s, 10);
        for lr in &lrs[5..] {
            assert!((*lr - 0.1).abs() < 1e-10, "expected min_lr, got {lr}");
        }
    }

    #[test]
    fn no_warmup_starts_at_cosine_start() {
        // warmup_iters = 0: first step should be the cosine value at progress=0 → max_lr
        let mut s = scheduler(1.0, 0.1, 0, 10);
        let lr = s.step();
        assert!((lr - 1.0).abs() < 1e-10, "got {lr}");
    }

    #[test]
    fn warmup_cosine_transition_is_continuous() {
        // The LR at the last warmup step (max_lr) and the first cosine step (progress=0 → max_lr)
        // should be identical — no discontinuity at the boundary.
        let max_lr = 1.0_f64;
        let min_lr = 0.1;
        let warmup = 5;
        let total = 20;
        let mut s = scheduler(max_lr, min_lr, warmup, total);
        let lrs = steps(&mut s, warmup + 1);
        // lrs[warmup-1] is the last warmup step (should be max_lr)
        // lrs[warmup]   is the first cosine step (progress=0 → max_lr)
        assert!(
            (lrs[warmup - 1] - max_lr).abs() < 1e-10,
            "last warmup step: {}",
            lrs[warmup - 1]
        );
        assert!(
            (lrs[warmup] - max_lr).abs() < 1e-10,
            "first cosine step: {}",
            lrs[warmup]
        );
    }

    #[test]
    fn cosine_decays_monotonically() {
        let mut s = scheduler(1.0, 0.1, 0, 20);
        let lrs = steps(&mut s, 20);
        for w in lrs.windows(2) {
            assert!(w[0] >= w[1], "LR increased: {} -> {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_training_convergence() {
        let device = Device::flex().autodiff();

        // 1. Create a tiny dataset: "ABCABCABC..."
        let block_size = 4;
        let data: Vec<usize> = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
        let dataset = TextDataset::new(data, block_size);

        let config = GPTConfig {
            vocab_size: 3,
            n_layer: 1,
            n_head: 1,
            n_embd: 8,
            n_kv_head: None,
            block_size,
            dropout: 0.0,
            rope_theta: 10000.0,
        };
        let mut model = GPT::new(&config, &device);

        let mut optim = AdamWConfig::new().init();
        let mut lr_scheduler = WarmupCosineScheduler::new(0.01, 0.01, 0, 100);
        let batcher = TextGenerationBatcher::new(block_size);

        // 2. Initial loss
        let item = BurnDataset::get(&dataset, 0).unwrap();
        let batch = batcher.batch(vec![item.clone()], &device);
        let output = model.forward_loss(batch);
        let initial_loss: f32 = output.loss.into_scalar();

        // 3. Train for 25 steps (deterministic convergence across random seeds)
        for _ in 0..25 {
            let batch = batcher.batch(vec![item.clone()], &device);
            let train_output = TrainStep::step(&model, batch);
            model = optim.step(lr_scheduler.step(), model, train_output.grads);
        }

        // 4. Final loss
        let batch = batcher.batch(vec![item.clone()], &device);
        let output = model.forward_loss(batch);
        let final_loss: f32 = output.loss.into_scalar();

        assert!(
            final_loss < initial_loss,
            "Loss did not decrease: {} -> {}",
            initial_loss,
            final_loss
        );
    }

    #[test]
    fn optimizer_steps_account_for_multi_gpu_and_grad_accum() {
        assert_eq!(compute_total_optimizer_steps(100, 10, 1, 1, 3), 30);
        // 100 items / batch 10 = 10 micro-steps; accum 4 -> ceil(10/4) = 3 opt steps/epoch
        assert_eq!(compute_total_optimizer_steps(100, 10, 4, 1, 2), 6);
        // 2 GPUs split 100 items -> 50 items/dev -> 5 micro-steps -> ceil(5/2) = 3 opt steps/epoch
        assert_eq!(compute_total_optimizer_steps(100, 10, 2, 2, 2), 6);
        // Partial final batch: 101 items / batch 10 = 11 micro-steps
        assert_eq!(compute_total_optimizer_steps(101, 10, 1, 1, 1), 11);
        // Zero grad_accum_steps / num_devices safely treated as 1
        assert_eq!(compute_total_optimizer_steps(10, 5, 0, 0, 2), 4);
    }

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

        // Weighted mean NLL = (2.0 * 10 + 4.0 * 30) / 40 = 3.5 -> exp(3.5)
        let expected = 3.5_f64.exp();
        match metric.final_value() {
            NumericEntry::Final(ppl) => {
                assert!((ppl - expected).abs() < 1e-5, "got {ppl}, expected {expected}");
            }
            other => panic!("unexpected final_value: {other:?}"),
        }
    }

    #[test]
    fn muon_and_adamw_param_groups_converge() {
        for choice in [OptimizerChoice::AdamW, OptimizerChoice::Muon] {
            let device = Device::flex().autodiff();
            let block_size = 4;
            let data: Vec<usize> = vec![0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
            let dataset = TextDataset::new(data, block_size);
            let config = GPTConfig {
                vocab_size: 3,
                n_layer: 1,
                n_head: 1,
                n_embd: 8,
                n_kv_head: None,
                block_size,
                dropout: 0.0,
                rope_theta: 10000.0,
            };
            let mut model = GPT::new(&config, &device);
            let mut optim = build_optimizer(choice);
            let batcher = TextGenerationBatcher::new(block_size);
            let item = BurnDataset::get(&dataset, 0).unwrap();

            let initial_loss: f32 = model
                .forward_loss(batcher.batch(vec![item.clone()], &device))
                .loss
                .into_scalar();
            for _ in 0..8 {
                let batch = batcher.batch(vec![item.clone()], &device);
                let out = TrainStep::step(&model, batch);
                model = optim.step(0.01, model, out.grads);
            }
            let final_loss: f32 = model
                .forward_loss(batcher.batch(vec![item], &device))
                .loss
                .into_scalar();
            assert!(
                final_loss < initial_loss,
                "{choice:?} did not decrease loss: {initial_loss} -> {final_loss}"
            );
        }
    }

    #[test]
    fn export_best_checkpoint_selects_lowest_valid_ppl() {
        let dir = std::env::temp_dir().join("nanoburngpt_test_best_ckpt");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("checkpoint")).unwrap();
        std::fs::create_dir_all(dir.join("valid/epoch-1")).unwrap();
        std::fs::create_dir_all(dir.join("valid/epoch-2")).unwrap();
        std::fs::create_dir_all(dir.join("valid/epoch-3")).unwrap();

        std::fs::write(dir.join("checkpoint/model-1.bpk"), b"ep1").unwrap();
        std::fs::write(dir.join("checkpoint/model-2.bpk"), b"ep2").unwrap();
        std::fs::write(dir.join("checkpoint/model-3.bpk"), b"ep3").unwrap();
        std::fs::write(dir.join("valid/epoch-1/Perplexity.log"), "200.0,final\n").unwrap();
        std::fs::write(dir.join("valid/epoch-2/Perplexity.log"), "95.5,final\n").unwrap();
        std::fs::write(dir.join("valid/epoch-3/Perplexity.log"), "140.2,final\n").unwrap();

        let best = export_best_checkpoint(&dir).unwrap();
        assert_eq!(best, Some((2, 95.5)));
        assert_eq!(std::fs::read(dir.join("model_best.bpk")).unwrap(), b"ep2");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
