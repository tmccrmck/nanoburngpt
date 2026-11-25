pub mod metrics;
pub mod renderer;
pub mod scheduler;

pub use self::metrics::{LmLossOutput, LossPerplexityMetric, ScalarLossMetric, ScalarLossOutput};
pub use self::renderer::CompactCliRenderer;
pub use self::scheduler::WarmupCosineScheduler;

use std::path::Path;
use std::str::FromStr;

use crate::{
    data::{BpeTokenizer, TextDataset, TextGenerationBatch, TextGenerationBatcher},
    datasets::Dataset,
    model::{GPT, GPTConfig},
};
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    grad_clipping::GradientClippingConfig,
    module::{Module, ParamGroup},
    nn::loss::CrossEntropyLossConfig,
    optim::{AdamWConfig, AdjustLrFn, ModuleOptimizer, MuonConfig, decay::WeightDecayConfig},
    prelude::*,
    tensor::distributed::{DistributedConfig, ReduceOperation},
    train::{
        ExecutionStrategy, InferenceStep, Learner, MultiDeviceOptim, SupervisedTraining,
        TrainOutput, TrainStep, TrainingStrategy,
        checkpoint::{
            ComposedCheckpointingStrategy, KeepLastNCheckpoints, MetricCheckpointingStrategy,
        },
        metric::store::{Aggregate, Direction, Split},
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
    type Output = ScalarLossOutput;

    fn step(&self, item: TextGenerationBatch) -> TrainOutput<ScalarLossOutput> {
        let loss_scale = item.loss_scale;
        let out = self.forward_loss(item);
        let grads = if (loss_scale - 1.0).abs() > 1e-9 {
            (out.loss.clone() * loss_scale).backward()
        } else {
            out.loss.clone().backward()
        };
        let loss = out.loss.try_into_scalar::<f64>().unwrap_or(f64::NAN);
        TrainOutput::new(
            self,
            grads,
            ScalarLossOutput {
                loss,
                num_tokens: out.num_tokens,
            },
        )
    }
}

impl InferenceStep for GPT {
    type Input = TextGenerationBatch;
    type Output = ScalarLossOutput;

    fn step(&self, item: TextGenerationBatch) -> ScalarLossOutput {
        let out = self.forward_loss(item);
        let loss = out.loss.try_into_scalar::<f64>().unwrap_or(f64::NAN);
        ScalarLossOutput {
            loss,
            num_tokens: out.num_tokens,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers & training entry points
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
    let norm_group = ParamGroup::from_any_predicates(vec!["ln_1", "ln_2", "ln_f", "q_norm", "k_norm"]);
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

/// Download/load a built-in [`Dataset`] and run training.
pub fn run_training(
    devices: Vec<Device>,
    multi_gpu: MultiGpuMode,
    optimizer_choice: OptimizerChoice,
    dataset: Dataset,
    mut gpt_config: GPTConfig,
    training_config: TrainingConfig,
) -> anyhow::Result<()> {
    if gpt_config.vocab_size < BpeTokenizer::VOCAB_SIZE {
        gpt_config.vocab_size = BpeTokenizer::PADDED_VOCAB_SIZE;
    }
    let (train_dataset, val_dataset) = dataset.load(gpt_config.block_size)?;
    run_training_with_datasets(
        devices,
        multi_gpu,
        optimizer_choice,
        train_dataset,
        val_dataset,
        gpt_config,
        training_config,
    )
}

/// Run training with caller-supplied `(train_dataset, val_dataset)` splits.
pub fn run_training_with_datasets(
    devices: Vec<Device>,
    multi_gpu: MultiGpuMode,
    optimizer_choice: OptimizerChoice,
    train_dataset: TextDataset,
    val_dataset: TextDataset,
    gpt_config: GPTConfig,
    training_config: TrainingConfig,
) -> anyhow::Result<()> {
    anyhow::ensure!(!devices.is_empty(), "At least one training device is required");
    gpt_config.validate()?;

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

    let train_dataset = train_dataset.with_chunk_jitter(true);
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

    println!("Vocab size: {} (BPE r50k_base)", gpt_config.vocab_size);

    let artifact_dir = Path::new(&training_config.artifact_dir);
    std::fs::create_dir_all(artifact_dir)?;
    gpt_config
        .save(artifact_dir.join("config.json"))
        .map_err(|e| anyhow::anyhow!("Failed to save config: {e:?}"))?;

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

    let optim = build_optimizer(optimizer_choice);

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
        .metric_train_numeric(ScalarLossMetric::new())
        .metric_valid_numeric(ScalarLossMetric::new())
        .metric_train_numeric(LossPerplexityMetric::new())
        .metric_valid_numeric(LossPerplexityMetric::new())
        .with_default_checkpointers()
        .with_checkpointing_strategy(strategy)
        .with_application_logger(None)
        .with_training_strategy(TrainingStrategy::Default(exec_strategy))
        .num_epochs(training_config.num_epochs)
        .summary();

    if let Some(epoch) = training_config.resume_epoch {
        println!(
            "Resuming from checkpoint epoch {epoch} in {}",
            artifact_dir.display()
        );
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
    use burn::data::dataloader::batcher::Batcher;
    use burn::data::dataset::Dataset as BurnDataset;
    use burn::optim::lr_scheduler::LrScheduler;

    #[test]
    fn test_training_convergence() {
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
            qk_norm: false,
            logit_softcap: 0.0,
        };
        let mut model = GPT::new(&config, &device);

        let mut optim = AdamWConfig::new().init();
        let mut lr_scheduler = WarmupCosineScheduler::new(0.01, 0.01, 0, 100);
        let batcher = TextGenerationBatcher::new(block_size);

        let item = BurnDataset::get(&dataset, 0).unwrap();
        let batch = batcher.batch(vec![item.clone()], &device);
        let initial_loss: f32 = model.forward_loss(batch).loss.into_scalar();

        for _ in 0..25 {
            let batch = batcher.batch(vec![item.clone()], &device);
            let train_output = TrainStep::step(&model, batch);
            model = optim.step(lr_scheduler.step(), model, train_output.grads);
        }

        let batch = batcher.batch(vec![item.clone()], &device);
        let final_loss: f32 = model.forward_loss(batch).loss.into_scalar();

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
        assert_eq!(compute_total_optimizer_steps(100, 10, 4, 1, 2), 6);
        assert_eq!(compute_total_optimizer_steps(100, 10, 2, 2, 2), 6);
        assert_eq!(compute_total_optimizer_steps(101, 10, 1, 1, 1), 11);
        assert_eq!(compute_total_optimizer_steps(10, 5, 0, 0, 2), 4);
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
            qk_norm: false,
            logit_softcap: 0.0,
            };
            let mut model = GPT::new(&config, &device);
            let mut optim = build_optimizer(choice);
            let batcher = TextGenerationBatcher::new(block_size);
            let item = BurnDataset::get(&dataset, 0).unwrap();

            let initial_loss: f32 = model
                .forward_loss(batcher.batch(vec![item.clone()], &device))
                .loss
                .into_scalar();
            for _ in 0..16 {
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
