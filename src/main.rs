#![recursion_limit = "512"]

use std::str::FromStr;

use burn::prelude::*;
use clap::{Parser, Subcommand};
use nanoburngpt::{
    datasets::Dataset,
    inference::generate_text,
    model::GPTConfig,
    presets::ModelPreset,
    train::{TrainingConfig, run_training},
};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Force use of CPU (flex backend)
    #[arg(long, global = true)]
    cpu: bool,
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
        /// Override preset: number of key/value heads for GQA (Grouped-Query Attention)
        #[arg(long)]
        n_kv_head: Option<usize>,
        /// Override preset: embedding dimension
        #[arg(long)]
        n_embd: Option<usize>,
        /// Override preset: context window size
        #[arg(long)]
        block_size: Option<usize>,
        /// Override preset: dropout probability
        #[arg(long)]
        dropout: Option<f64>,
        /// Training batch size
        #[arg(long, default_value_t = 32)]
        batch_size: usize,
        /// Number of dataloader workers
        #[arg(long, default_value_t = 4)]
        num_workers: usize,
        /// Learning rate
        #[arg(long, default_value_t = 1e-3)]
        learning_rate: f64,
        /// Number of training epochs
        #[arg(long, default_value_t = 10)]
        num_epochs: usize,
        /// Warmup steps before cosine decay (0 = no warmup)
        #[arg(long, default_value_t = 100)]
        warmup_iters: usize,
        /// Cap training items (0 = use full dataset; useful for smoke tests)
        #[arg(long, default_value_t = 0)]
        max_train_items: usize,
        /// RoPE frequency base (10000.0 = original paper; nanochat uses 100000.0)
        #[arg(long, default_value_t = 10000.0)]
        rope_theta: f64,
        /// Enable mixed precision (AMP) — currently a no-op on Burn 0.22
        /// (dtype defaults API not yet wired); training runs in f32.
        #[arg(long)]
        amp: bool,
        /// Gradient accumulation steps (simulates larger batch size with less memory)
        #[arg(long, default_value_t = 1)]
        grad_accum_steps: usize,
        /// Disable the interactive TUI and print plain CLI progress logs
        /// (recommended for notebooks like Google Colab and non-TTY log capture)
        #[arg(long)]
        no_tui: bool,
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
        /// Top-k sampling: keep only the k most probable tokens (0 = disabled)
        #[arg(long, default_value_t = 0)]
        top_k: usize,
        /// Top-p (nucleus) sampling: keep smallest set with cumulative prob >= p (1.0 = disabled)
        #[arg(long, default_value_t = 1.0)]
        top_p: f64,
        /// Repetition penalty (> 1.0 penalizes repeated tokens; 1.0 = disabled)
        #[arg(long, default_value_t = 1.0)]
        repetition_penalty: f64,
    },
}

fn select_device(cpu: bool) -> Device {
    if cpu {
        return Device::flex();
    }
    #[cfg(feature = "cuda")]
    {
        return Device::cuda(0);
    }
    #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    {
        Device::wgpu(DeviceKind::DefaultDevice)
    }
    #[cfg(not(any(feature = "wgpu", feature = "cuda")))]
    {
        Device::flex()
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let cpu = cli.cpu;
    let device = select_device(cpu);

    match cli.command {
        Commands::Train {
            model,
            dataset,
            n_layer,
            n_head,
            n_kv_head,
            n_embd,
            block_size,
            dropout,
            batch_size,
            num_workers,
            learning_rate,
            num_epochs,
            warmup_iters,
            max_train_items,
            rope_theta,
            amp,
            grad_accum_steps,
            no_tui,
        } => {
            let preset = model
                .parse::<ModelPreset>()
                .expect("Invalid --model. Use: nano, gpt2-small, gpt2-medium, gpt2-large, gpt2-xl");
            let preset_cfg = preset.config();

            let gpt_config = GPTConfig {
                vocab_size: preset_cfg.vocab_size,
                n_layer: n_layer.unwrap_or(preset_cfg.n_layer),
                n_head: n_head.unwrap_or(preset_cfg.n_head),
                n_kv_head: n_kv_head.or(preset_cfg.n_kv_head),
                n_embd: n_embd.unwrap_or(preset_cfg.n_embd),
                block_size: block_size.unwrap_or(preset_cfg.block_size),
                dropout: dropout.unwrap_or(preset_cfg.dropout),
                rope_theta,
            };

            let dataset_enum = Dataset::from_str(&dataset)
                .expect("Invalid --dataset. Use: shakespeare, wikitext103");

            let training_config = TrainingConfig {
                batch_size,
                num_workers,
                learning_rate,
                warmup_iters,
                num_epochs,
                max_train_items,
                amp,
                grad_accum_steps,
                no_tui,
            };

            println!(
                "Model: {} ({} layers, {} heads, {} embd, ctx {})",
                model,
                gpt_config.n_layer,
                gpt_config.n_head,
                gpt_config.n_embd,
                gpt_config.block_size
            );
            println!("Dataset: {}", dataset);
            println!("Device: {:?}", device);
            if amp {
                println!(
                    "Warning: --amp is a no-op on Burn 0.22 (training in f32)."
                );
            }

            run_training(device, dataset_enum, gpt_config, training_config)?;
        }
        Commands::Generate {
            artifact_dir,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
        } => {
            let sampling = nanoburngpt::model::SamplingParams {
                temperature,
                top_k,
                top_p,
                repetition_penalty,
            };
            println!("Generating text on device: {:?}", device);
            generate_text(device, &artifact_dir, &prompt, max_tokens, &sampling);
        }
    }

    Ok(())
}
