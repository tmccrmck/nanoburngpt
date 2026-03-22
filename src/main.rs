#![recursion_limit = "512"]

use burn::backend::{wgpu::Wgpu, Autodiff};
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

fn main() {
    let cli = Cli::parse();

    type Backend = Wgpu<f32, i32>;
    type AutodiffBackend = Autodiff<Backend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

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
                num_epochs,
                max_train_items,
            };
            println!("Starting training on device: {:?}", device);
            run_training::<AutodiffBackend>(device, gpt_config, training_config);
        }
        Commands::Generate {
            artifact_dir,
            prompt,
            max_tokens,
            temperature,
        } => {
            println!("Generating text on device: {:?}", device);
            generate_text::<Backend>(device, &artifact_dir, &prompt, max_tokens, temperature);
        }
    }
}
