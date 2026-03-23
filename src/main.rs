#![recursion_limit = "512"]

#[cfg(not(any(feature = "wgpu", feature = "cuda")))]
compile_error!("Enable at least one backend feature: `wgpu` or `cuda`");

use std::str::FromStr;

use burn::backend::Autodiff;
use burn::tensor::backend::{AutodiffBackend, Backend};
use clap::{Parser, Subcommand};
use nanoburngpt::{
    datasets::Dataset,
    inference::generate_text,
    model::GPTConfig,
    presets::ModelPreset,
    train::{run_training, TrainingConfig},
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Train {
        /// Model size preset (nano, gpt2-small, gpt2-medium, gpt2-large, gpt2-xl)
        #[arg(long, default_value = "nano")]
        model: String,
        /// Dataset to train on (shakespeare, wikitext103)
        #[arg(long, default_value = "shakespeare")]
        dataset: String,
        /// Override preset: number of transformer layers
        #[arg(long)]
        n_layer: Option<usize>,
        /// Override preset: number of attention heads
        #[arg(long)]
        n_head: Option<usize>,
        /// Override preset: embedding dimension
        #[arg(long)]
        n_embd: Option<usize>,
        /// Override preset: context window size
        #[arg(long)]
        block_size: Option<usize>,
        /// Dropout probability
        #[arg(long, default_value_t = 0.2)]
        dropout: f64,
        /// Training batch size
        #[arg(long, default_value_t = 32)]
        batch_size: usize,
        /// Number of dataloader workers
        #[arg(long, default_value_t = 4)]
        num_workers: usize,
        /// Random seed
        #[arg(long, default_value_t = 1337)]
        seed: u64,
        /// Learning rate
        #[arg(long, default_value_t = 1e-3)]
        learning_rate: f64,
        /// Number of training epochs
        #[arg(long, default_value_t = 10)]
        num_epochs: usize,
        /// Minimum LR at end of cosine decay (default: lr/10)
        #[arg(long, default_value_t = 1e-4)]
        min_lr: f64,
        /// Warmup steps before cosine decay (0 = no warmup)
        #[arg(long, default_value_t = 100)]
        warmup_iters: usize,
        /// AdamW beta2 (nanoGPT: 0.95)
        #[arg(long, default_value_t = 0.95)]
        beta2: f64,
        /// AdamW weight decay (nanoGPT: 0.1)
        #[arg(long, default_value_t = 0.1)]
        weight_decay: f64,
        /// Cap training items (0 = use full dataset; useful for smoke tests)
        #[arg(long, default_value_t = 0)]
        max_train_items: usize,
    },
    Generate {
        #[arg(long, default_value = "artifacts")]
        artifact_dir: String,
        #[arg(long, default_value = "\n")]
        prompt: String,
        /// Number of tokens to generate
        #[arg(long, default_value_t = 500)]
        max_tokens: usize,
        /// Sampling temperature (0 = greedy)
        #[arg(long, default_value_t = 0.8)]
        temperature: f64,
    },
}

fn dispatch<AB, B>(cli: Cli, device: B::Device)
where
    AB: AutodiffBackend<InnerBackend = B>,
    AB: Backend<Device = B::Device>,
    B: Backend,
    B::Device: std::fmt::Debug + Clone,
{
    match cli.command {
        Commands::Train {
            model,
            dataset,
            n_layer,
            n_head,
            n_embd,
            block_size,
            dropout,
            batch_size,
            num_workers,
            seed,
            learning_rate,
            num_epochs,
            min_lr,
            warmup_iters,
            beta2,
            weight_decay,
            max_train_items,
        } => {
            let preset = ModelPreset::from_str(&model)
                .expect("Invalid --model. Use: nano, gpt2-small, gpt2-medium, gpt2-large, gpt2-xl");
            let preset_cfg = preset.config();

            let gpt_config = GPTConfig {
                vocab_size: preset_cfg.vocab_size,
                n_layer:    n_layer.unwrap_or(preset_cfg.n_layer),
                n_head:     n_head.unwrap_or(preset_cfg.n_head),
                n_embd:     n_embd.unwrap_or(preset_cfg.n_embd),
                block_size: block_size.unwrap_or(preset_cfg.block_size),
                dropout,
            };

            let dataset_enum = Dataset::from_str(&dataset)
                .expect("Invalid --dataset. Use: shakespeare, wikitext103");

            let training_config = TrainingConfig {
                batch_size,
                num_workers,
                seed,
                learning_rate,
                min_lr,
                warmup_iters,
                beta2,
                weight_decay,
                num_epochs,
                max_train_items,
            };

            println!("Model: {} ({} layers, {} heads, {} embd, ctx {})",
                model, gpt_config.n_layer, gpt_config.n_head,
                gpt_config.n_embd, gpt_config.block_size);
            println!("Dataset: {}", dataset);
            println!("Device: {:?}", device);

            run_training::<AB>(device, dataset_enum, gpt_config, training_config);
        }
        Commands::Generate {
            artifact_dir,
            prompt,
            max_tokens,
            temperature,
        } => {
            println!("Generating text on device: {:?}", device);
            generate_text::<B>(device, &artifact_dir, &prompt, max_tokens, temperature);
        }
    }
}

fn main() {
    let cli = Cli::parse();

    #[cfg(feature = "cuda")]
    {
        use burn::backend::cuda::{Cuda, CudaDevice};
        dispatch::<Autodiff<Cuda<f32, i32>>, Cuda<f32, i32>>(cli, CudaDevice::default());
    }

    #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    {
        use burn::backend::wgpu::{Wgpu, WgpuDevice};
        dispatch::<Autodiff<Wgpu<f32, i32>>, Wgpu<f32, i32>>(cli, WgpuDevice::default());
    }
}
