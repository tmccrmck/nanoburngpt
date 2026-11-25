pub mod data;
pub mod datasets;
pub mod inference;
pub mod model;
pub mod presets;
pub mod train;

pub use crate::data::{BpeTokenizer, TextDataset};
pub use crate::datasets::Dataset;
pub use crate::inference::generate_text;
pub use crate::model::{GPT, GPTConfig, KVCache, RotaryCache, SamplingParams};
pub use crate::presets::ModelPreset;
pub use crate::train::{
    MultiGpuMode, OptimizerChoice, TrainingConfig, run_training, run_training_with_datasets,
};
