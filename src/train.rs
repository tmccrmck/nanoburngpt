use crate::{
    data::{TextDataset, TextGenerationBatch, TextGenerationBatcher, load_shakespeare},
    model::{GPTConfig, GPT},
};
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    nn::loss::CrossEntropyLossConfig,
    grad_clipping::GradientClippingConfig,
    optim::AdamWConfig,
    optim::lr_scheduler::LrScheduler,
    record::CompactRecorder,
    tensor::backend::{AutodiffBackend, Backend},
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
    },
};
use std::path::Path;

// ---------------------------------------------------------------------------
// Training configuration
// ---------------------------------------------------------------------------

#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 1337)]
    pub seed: u64,
    /// Peak learning rate (also the warmup target).
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    /// Minimum LR at the end of cosine decay. nanoGPT default: lr/10.
    #[config(default = 1e-4)]
    pub min_lr: f64,
    /// Linear warmup steps. 0 = no warmup.
    #[config(default = 100)]
    pub warmup_iters: usize,
    #[config(default = 10)]
    pub num_epochs: usize,
    /// AdamW beta2. nanoGPT uses 0.95 (default Burn: 0.999).
    #[config(default = 0.95)]
    pub beta2: f64,
    /// AdamW weight decay. nanoGPT uses 0.1 (default Burn: 1e-4).
    #[config(default = 0.1)]
    pub weight_decay: f64,
    /// 0 = full dataset; positive value caps training items (smoke tests).
    #[config(default = 0)]
    pub max_train_items: usize,
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
            current_iter: 0,
        }
    }
}

impl LrScheduler for WarmupCosineScheduler {
    type Record<B: burn::tensor::backend::Backend> = usize;

    fn step(&mut self) -> f64 {
        let iter = self.current_iter;
        self.current_iter += 1;

        if self.warmup_iters > 0 && iter < self.warmup_iters {
            // Linear ramp: 0 → max_lr over warmup_iters steps
            self.max_lr * (iter + 1) as f64 / self.warmup_iters as f64
        } else if iter >= self.total_iters {
            // Hold at min_lr after the decay is complete
            self.min_lr
        } else {
            // Cosine decay: max_lr → min_lr
            let progress = (iter - self.warmup_iters) as f64
                / (self.total_iters - self.warmup_iters) as f64;
            let coeff = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            self.min_lr + coeff * (self.max_lr - self.min_lr)
        }
    }

    fn to_record<B: burn::tensor::backend::Backend>(&self) -> Self::Record<B> {
        self.current_iter
    }

    fn load_record<B: burn::tensor::backend::Backend>(mut self, record: Self::Record<B>) -> Self {
        self.current_iter = record;
        self
    }
}

// ---------------------------------------------------------------------------
// Model training/inference steps
// ---------------------------------------------------------------------------

impl<B: Backend> GPT<B> {
    pub fn forward_classification(&self, item: TextGenerationBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets; // [batch, seq]
        let [batch_size, seq_len] = targets.dims();

        let logits = self.forward(item.inputs); // [batch, seq, vocab]
        let [_, _, vocab_size] = logits.dims();

        let logits_flat = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets_flat = targets.reshape([batch_size * seq_len]);

        let loss = CrossEntropyLossConfig::new()
            .init(&logits_flat.device())
            .forward(logits_flat.clone(), targets_flat.clone());

        ClassificationOutput {
            loss,
            output: logits_flat,
            targets: targets_flat,
        }
    }
}

impl<B: AutodiffBackend> TrainStep for GPT<B> {
    type Input = TextGenerationBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: TextGenerationBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for GPT<B> {
    type Input = TextGenerationBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: TextGenerationBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item)
    }
}

// ---------------------------------------------------------------------------
// Training entry point
// ---------------------------------------------------------------------------

pub fn run_training<B: AutodiffBackend>(
    device: B::Device,
    mut gpt_config: GPTConfig,
    training_config: TrainingConfig,
) {
    // --- Data ---
    let (train_dataset, val_dataset, tokenizer) =
        load_shakespeare(Path::new("data/input.txt"), gpt_config.block_size).unwrap();

    // Optionally cap the training set for smoke tests
    let train_dataset = if training_config.max_train_items > 0 {
        let cap = training_config.max_train_items.min(train_dataset.data.len());
        TextDataset::new(train_dataset.data[..cap].to_vec(), gpt_config.block_size)
    } else {
        train_dataset
    };

    gpt_config.vocab_size = tokenizer.vocab_size;
    println!("Vocab size: {}", gpt_config.vocab_size);

    std::fs::create_dir_all("artifacts").ok();
    gpt_config.save("artifacts/config.json").expect("Config saved");

    // Compute total training steps for the LR schedule
    let steps_per_epoch = train_dataset.len() / training_config.batch_size;
    let total_iters = steps_per_epoch * training_config.num_epochs;
    println!(
        "Steps/epoch: {steps_per_epoch}  |  Total steps: {total_iters}  |  Warmup: {}",
        training_config.warmup_iters
    );

    let batcher_train = TextGenerationBatcher::<B>::new(gpt_config.block_size);
    let batcher_val = TextGenerationBatcher::<B::InnerBackend>::new(gpt_config.block_size);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .build(train_dataset);

    let dataloader_val = DataLoaderBuilder::new(batcher_val)
        .batch_size(training_config.batch_size)
        .shuffle(training_config.seed)
        .num_workers(training_config.num_workers)
        .build(val_dataset);

    // --- Optimizer (nanoGPT hyperparameters) ---
    let optim = AdamWConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(training_config.beta2 as f32)
        .with_weight_decay(training_config.weight_decay as f32)
        .with_epsilon(1e-8)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    // --- LR scheduler: linear warmup → cosine decay ---
    let lr_scheduler = WarmupCosineScheduler::new(
        training_config.learning_rate,
        training_config.min_lr,
        training_config.warmup_iters,
        total_iters.max(1),
    );

    let model = GPT::new(&gpt_config, &device);
    let learner = Learner::new(model, optim, lr_scheduler);

    let result = SupervisedTraining::new("artifacts", dataloader_train, dataloader_val)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(training_config.num_epochs)
        .summary()
        .launch(learner);

    result
        .model
        .save_file("artifacts/model_final", &CompactRecorder::new())
        .expect("Failed to save final model");
    println!("Model saved to artifacts/model_final");

    tokenizer
        .save(Path::new("artifacts/tokenizer.json"))
        .expect("Failed to save tokenizer");
    println!("Tokenizer saved to artifacts/tokenizer.json");
}
