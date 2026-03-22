use burn::backend::{
    wgpu::Wgpu,
    Autodiff,
};
use clap::{Parser, Subcommand};
use nanoburngpt::{
    model::GPTConfig,
    train::{run_training, TrainingConfig},
    inference::generate_text,
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
        #[arg(long, default_value = "data/input.txt")]
        data: String,
    },
    Generate {
        #[arg(long, default_value = "artifacts")]
        artifact_dir: String,
        #[arg(long, default_value = "\n")]
        prompt: String,
    },
}

fn main() {
    let cli = Cli::parse();

    type Backend = Wgpu<f32, i32>;
    type AutodiffBackend = Autodiff<Backend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    match cli.command {
        Commands::Train { data: _ } => {
            // Config for NanoGPT (Small)
            let gpt_config = GPTConfig {
                vocab_size: 0, // Will be set from data
                n_layer: 6,
                n_head: 6,
                n_embd: 384,
                block_size: 128, // Reduced for MacBook Air
                dropout: 0.2,
            };
            
            let training_config = TrainingConfig {
                batch_size: 32,
                num_workers: 4,
                seed: 1337,
                learning_rate: 1e-3,
                num_epochs: 10, // Small for testing
            };

            println!("Starting training on device: {:?}", device);
            run_training::<AutodiffBackend>(device, gpt_config, training_config);
        }
        Commands::Generate { artifact_dir, prompt } => {
            println!("Generating text on device: {:?}", device);
            generate_text::<Backend>(device, &artifact_dir, &prompt);
        }
    }
}
