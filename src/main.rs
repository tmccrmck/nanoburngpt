#![recursion_limit = "512"]

#[cfg(not(any(feature = "wgpu", feature = "cuda")))]
compile_error!("Enable at least one backend feature: `wgpu` or `cuda`");

use burn::backend::Autodiff;
use burn::tensor::backend::{AutodiffBackend, Backend};
use clap::{Parser, Subcommand};
use nanoburngpt::{
    inference::generate_text,
    model::GPTConfig,
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
        /// Number of transformer layers
        #[arg(long, default_value_t = 6)]
        n_layer: usize,
        /// Number of attention heads
        #[arg(long, default_value_t = 6)]
        n_head: usize,
        /// Embedding dimension
        #[arg(long, default_value_t = 384)]
        n_embd: usize,
        /// Context window size
        #[arg(long, default_value_t = 128)]
        block_size: usize,
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
            let gpt_config = GPTConfig {
                vocab_size: 0, // Set from data
                n_layer,
                n_head,
                n_embd,
                block_size,
                dropout,
            };
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
            println!("Starting training on device: {:?}", device);
            run_training::<AB>(device, gpt_config, training_config);
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
