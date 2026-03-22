use crate::{
    data::{TextDataset, TextGenerationBatch, TextGenerationBatcher, load_shakespeare},
    model::{GPTConfig, GPT},
};
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    nn::loss::CrossEntropyLossConfig,
    optim::AdamWConfig,
    record::CompactRecorder,
    tensor::backend::{AutodiffBackend, Backend},
    train::{
        metric::{AccuracyMetric, LossMetric},
        ClassificationOutput, TrainOutput, TrainStep, InferenceStep, Learner, SupervisedTraining,
    },
};
use std::path::Path;

#[derive(Config, Debug)]
pub struct TrainingConfig {
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 123)]
    pub seed: u64,
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    #[config(default = 5)]
    pub num_epochs: usize,
    /// 0 means use the full dataset.
    #[config(default = 0)]
    pub max_train_items: usize,
}

impl<B: Backend> GPT<B> {
    pub fn forward_classification(
        &self,
        item: TextGenerationBatch<B>,
    ) -> ClassificationOutput<B> {
        let targets = item.targets; // [batch, seq]
        let [batch_size, seq_len] = targets.dims();
        
        let logits = self.forward(item.inputs); // [batch, seq, vocab]

        // Flatten for loss calculation
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

pub fn run_training<B: AutodiffBackend>(
    device: B::Device,
    mut gpt_config: GPTConfig,
    training_config: TrainingConfig,
) {
    // Data
    let (train_dataset, val_dataset, tokenizer) = load_shakespeare(Path::new("data/input.txt"), gpt_config.block_size).unwrap();
    
    // Optionally cap the training set for quick smoke tests
    let train_dataset = if training_config.max_train_items > 0 {
        let capped = train_dataset.data[..training_config.max_train_items.min(train_dataset.data.len())].to_vec();
        TextDataset::new(capped, gpt_config.block_size)
    } else {
        train_dataset
    };

    // Update vocab size based on dataset
    gpt_config.vocab_size = tokenizer.vocab_size;
    println!("Vocab size: {}", gpt_config.vocab_size);
    
    // Save config
    std::fs::create_dir_all("artifacts").ok();
    gpt_config.save("artifacts/config.json").expect("Config saved");

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

    // Model
    let model = GPT::new(&gpt_config, &device);
    let optim = AdamWConfig::new().init();
    let learner = Learner::new(model, optim, training_config.learning_rate);

    let result = SupervisedTraining::new("artifacts", dataloader_train, dataloader_val)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(training_config.num_epochs)
        .summary()
        .launch(learner);

    // Save the final trained model to a stable path for inference
    result
        .model
        .save_file("artifacts/model_final", &CompactRecorder::new())
        .expect("Failed to save final model");
    println!("Model saved to artifacts/model_final");

    // Save the tokenizer alongside the model
    tokenizer
        .save(Path::new("artifacts/tokenizer.json"))
        .expect("Failed to save tokenizer");
    println!("Tokenizer saved to artifacts/tokenizer.json");
}

