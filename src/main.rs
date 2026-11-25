#![recursion_limit = "512"]

use std::{path::Path, str::FromStr};

use burn::{config::Config, prelude::*, tensor::FloatDType};
use clap::{Args, Parser, Subcommand};
use nanoburngpt::{
    bench::run_bench,
    data::load_text_file,
    datasets::Dataset,
    eval::run_eval,
    inference::generate_text,
    model::GPTConfig,
    presets::ModelPreset,
    train::{MultiGpuMode, OptimizerChoice, TrainingConfig, run_training},
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

#[derive(Args)]
struct TrainArgs {
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
    /// Cap validation items (0 = use full validation set)
    #[arg(long, default_value_t = 0)]
    max_val_items: usize,
    /// Directory where checkpoints, metrics, and final weights are saved
    #[arg(long, default_value = "artifacts")]
    artifact_dir: String,
    /// Resume training from a saved checkpoint epoch
    #[arg(long)]
    resume: Option<usize>,
    /// Optimizer (adamw, muon). muon uses Muon for 2D block weights + AdamW for embeddings/norms.
    #[arg(long, default_value = "adamw")]
    optimizer: String,
    /// RoPE frequency base (10000.0 = original paper; nanochat uses 100000.0)
    #[arg(long, default_value_t = 10000.0)]
    rope_theta: f64,
    /// Enable learnable RMSNorm on Q and K after RoPE (stabilizes higher learning rates)
    #[arg(long)]
    qk_norm: bool,
    /// Output logit softcap (`cap * tanh(logits / cap)`). 0.0 = disabled (e.g. 15.0)
    #[arg(long, default_value_t = 0.0)]
    logit_softcap: f64,
    /// Floating-point precision (f32, flex32, bf16, f16).
    /// flex32 uses FP16 Tensor Cores with FP32 storage/accumulation (recommended on T4/wgpu);
    /// bf16 halves VRAM on Ampere+ GPUs (L4/A100/H100).
    #[arg(long)]
    dtype: Option<String>,
    /// Shorthand for --dtype flex32 (mixed precision via FP16 Tensor Cores + FP32 storage)
    #[arg(long)]
    amp: bool,
    /// Gradient accumulation steps (simulates larger batch size with less memory)
    #[arg(long, default_value_t = 1)]
    grad_accum_steps: usize,
    /// Enable activation checkpointing to reduce peak training VRAM
    #[arg(long)]
    grad_checkpointing: bool,
    /// Multi-GPU strategy on multi-CUDA machines (auto, sharded, ddp, off)
    #[arg(long, default_value = "auto")]
    multi_gpu: String,
    /// Disable the interactive TUI and print plain CLI progress logs
    /// (recommended for notebooks like Google Colab and non-TTY log capture)
    #[arg(long)]
    no_tui: bool,
}

#[derive(Subcommand)]
enum Commands {
    Train(Box<TrainArgs>),
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
        /// Floating-point precision for inference (f32, flex32, bf16, f16)
        #[arg(long, default_value = "f32")]
        dtype: String,
        /// Load best validation checkpoint (`model_best.bpk`) instead of `model_final.bpk`
        #[arg(long)]
        best: bool,
    },
    /// Evaluate a trained checkpoint on a dataset or text file (Loss, Perplexity, BPB)
    Eval {
        #[arg(long, default_value = "artifacts")]
        artifact_dir: String,
        /// Built-in dataset whose validation split to evaluate (shakespeare, wikitext103)
        #[arg(long, default_value = "shakespeare")]
        dataset: String,
        /// Optional path to a custom UTF-8 text file (overrides --dataset)
        #[arg(long)]
        file: Option<String>,
        /// Evaluation batch size
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        /// Cap evaluated chunks (0 = full validation split)
        #[arg(long, default_value_t = 0)]
        max_items: usize,
        /// Floating-point precision for evaluation (f32, flex32, bf16, f16)
        #[arg(long, default_value = "f32")]
        dtype: String,
        /// Load best validation checkpoint (`model_best.bpk`) instead of `model_final.bpk`
        #[arg(long)]
        best: bool,
    },
    /// Benchmark train, eval, and KV-cache decode throughput on your local GPU (wgpu/cuda) or --cpu
    Bench {
        /// Model size preset (nano, gpt2-small, gpt2-medium, gpt2-large, gpt2-xl)
        #[arg(long, default_value = "gpt2-small")]
        model: String,
        /// Override preset: context window size
        #[arg(long, default_value_t = 256)]
        block_size: usize,
        /// Batch size for eval and train steps
        #[arg(long, default_value_t = 4)]
        batch_size: usize,
        /// Optimizer (adamw, muon)
        #[arg(long, default_value = "adamw")]
        optimizer: String,
        /// Floating-point precision (f32, flex32, bf16, f16)
        #[arg(long, default_value = "f32")]
        dtype: String,
        /// Warmup iterations before timing
        #[arg(long, default_value_t = 3)]
        warmup: usize,
        /// Timed iterations for eval and train steps
        #[arg(long, default_value_t = 10)]
        steps: usize,
        /// Number of tokens to generate in KV-cache decode benchmark
        #[arg(long, default_value_t = 64)]
        decode_tokens: usize,
        /// Enable QK-Norm
        #[arg(long)]
        qk_norm: bool,
        /// Output logit softcap (0.0 = disabled)
        #[arg(long, default_value_t = 0.0)]
        logit_softcap: f64,
        /// Chunk size (in tokens) for LM-head + cross-entropy loss (0 = unchunked)
        #[arg(long, default_value_t = 0)]
        ce_chunk_size: usize,
    },
}

fn parse_float_dtype(s: &str) -> anyhow::Result<FloatDType> {
    match s.to_ascii_lowercase().as_str() {
        "f32" => Ok(FloatDType::F32),
        "flex32" => Ok(FloatDType::Flex32),
        "bf16" => Ok(FloatDType::BF16),
        "f16" => Ok(FloatDType::F16),
        other => anyhow::bail!("Invalid --dtype '{other}'. Use: f32, flex32, bf16, f16"),
    }
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

fn select_training_devices(cpu: bool, multi_gpu: MultiGpuMode) -> Vec<Device> {
    if cpu || multi_gpu == MultiGpuMode::Off {
        return vec![select_device(cpu)];
    }
    #[cfg(feature = "cuda")]
    {
        let enumerated = Device::enumerate(burn::tensor::DeviceType::Cuda).into_vec();
        if !enumerated.is_empty() {
            return enumerated;
        }
        return vec![Device::cuda(0)];
    }
    #[cfg(not(feature = "cuda"))]
    {
        vec![select_device(cpu)]
    }
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let cpu = cli.cpu;

    match cli.command {
        Commands::Train(args) => {
            let TrainArgs {
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
                max_val_items,
                artifact_dir,
                resume,
                optimizer,
                rope_theta,
                qk_norm,
                logit_softcap,
                dtype,
                amp,
                grad_accum_steps,
                grad_checkpointing,
                multi_gpu,
                no_tui,
            } = *args;
            let preset = model.parse::<ModelPreset>()?;
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
                qk_norm,
                logit_softcap,
            };
            gpt_config.validate()?;

            let dataset_enum = Dataset::from_str(&dataset)?;
            let multi_gpu_mode = MultiGpuMode::from_str(&multi_gpu)?;
            let optimizer_choice = OptimizerChoice::from_str(&optimizer)?;
            let float_dtype = match &dtype {
                Some(d) => parse_float_dtype(d)?,
                None if amp => FloatDType::Flex32,
                None => FloatDType::F32,
            };

            let mut devices = select_training_devices(cpu, multi_gpu_mode);
            if devices.len() > 1 && float_dtype == FloatDType::Flex32 {
                match multi_gpu_mode {
                    MultiGpuMode::Auto => {
                        println!(
                            "Note: Burn 0.22.1 CUDA multi-GPU collectives (NCCL) do not support flex32; using 1 GPU with flex32 Tensor Cores."
                        );
                        devices.truncate(1);
                    }
                    MultiGpuMode::Sharded | MultiGpuMode::Ddp => {
                        anyhow::bail!(
                            "Burn 0.22.1 CUDA multi-GPU collectives (NCCL) do not support flex32; use --multi-gpu off (or auto) with flex32, or --dtype f32/bf16 with multi-GPU."
                        );
                    }
                    MultiGpuMode::Off => {}
                }
            }

            if float_dtype == FloatDType::F16 {
                eprintln!(
                    "Warning: training in f16 without a GradScaler may underflow; consider --dtype flex32 (T4/wgpu) or --dtype bf16 (Ampere+)."
                );
            }

            // Configure default float dtype on every device BEFORE any tensor/model is created.
            if float_dtype != FloatDType::F32 {
                for device in &mut devices {
                    device.configure(float_dtype).map_err(|e| {
                        anyhow::anyhow!(
                            "Failed to configure dtype {:?} on {:?}: {:?}",
                            float_dtype,
                            device,
                            e
                        )
                    })?;
                }
            }

            let training_config = TrainingConfig {
                batch_size,
                num_workers,
                learning_rate,
                warmup_iters,
                num_epochs,
                max_train_items,
                max_val_items,
                artifact_dir,
                resume_epoch: resume,
                grad_accum_steps,
                grad_checkpointing,
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
            println!("Devices ({}): {:?}", devices.len(), devices);
            println!(
                "Precision: {:?}  |  Optimizer: {:?}  |  Grad checkpointing: {}  |  Multi-GPU: {:?}",
                float_dtype, optimizer_choice, grad_checkpointing, multi_gpu_mode
            );

            run_training(
                devices,
                multi_gpu_mode,
                optimizer_choice,
                dataset_enum,
                gpt_config,
                training_config,
            )?;
        }
        Commands::Generate {
            artifact_dir,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            dtype,
            best,
        } => {
            let mut device = select_device(cpu);
            let float_dtype = parse_float_dtype(&dtype)?;
            if float_dtype != FloatDType::F32 {
                device.configure(float_dtype).map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to configure dtype {:?} on {:?}: {:?}",
                        float_dtype,
                        device,
                        e
                    )
                })?;
            }
            let sampling = nanoburngpt::model::SamplingParams {
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                vocab_limit: None,
            };
            let model_name = if best { "model_best" } else { "model_final" };
            println!(
                "Generating text on device: {:?} (dtype: {:?}, model: {})",
                device, float_dtype, model_name
            );
            generate_text(
                device,
                &artifact_dir,
                model_name,
                &prompt,
                max_tokens,
                &sampling,
            )?;
        }
        Commands::Eval {
            artifact_dir,
            dataset,
            file,
            batch_size,
            max_items,
            dtype,
            best,
        } => {
            let mut device = select_device(cpu);
            let float_dtype = parse_float_dtype(&dtype)?;
            if float_dtype != FloatDType::F32 {
                device.configure(float_dtype).map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to configure dtype {:?} on {:?}: {:?}",
                        float_dtype,
                        device,
                        e
                    )
                })?;
            }
            let config_path = format!("{artifact_dir}/config.json");
            let config = GPTConfig::load(&config_path)?;
            let eval_dataset = if let Some(path) = file {
                load_text_file(Path::new(&path), config.block_size)?
            } else {
                let ds = Dataset::from_str(&dataset)?;
                let (_, val) = ds.load(config.block_size)?;
                val
            };
            let eval_dataset = if max_items > 0 {
                eval_dataset.with_max_len(max_items)
            } else {
                eval_dataset
            };

            let model_name = if best { "model_best" } else { "model_final" };
            println!(
                "Evaluating {model_name} on device: {:?} (dtype: {:?}, chunks: {})",
                device,
                float_dtype,
                eval_dataset.len()
            );
            let report = run_eval(device, &artifact_dir, model_name, &eval_dataset, batch_size)?;
            let tok_s = report.total_tokens as f64 / report.elapsed_secs;
            println!(
                "Eval: loss={:.4}  ppl={:.2}  bpb={:.4}  tokens={}  bytes={}  ({:.1}s, {:.0} tok/s)",
                report.loss,
                report.perplexity,
                report.bpb,
                report.total_tokens,
                report.total_bytes,
                report.elapsed_secs,
                tok_s
            );
        }
        Commands::Bench {
            model,
            block_size,
            batch_size,
            optimizer,
            dtype,
            warmup,
            steps,
            decode_tokens,
            qk_norm,
            logit_softcap,
            ce_chunk_size,
        } => {
            let preset = model.parse::<ModelPreset>()?;
            let mut gpt_config = preset.config();
            gpt_config.block_size = block_size;
            gpt_config.qk_norm = qk_norm;
            gpt_config.logit_softcap = logit_softcap;
            gpt_config.validate()?;

            let optimizer_choice = OptimizerChoice::from_str(&optimizer)?;
            let mut device = select_device(cpu);
            let float_dtype = parse_float_dtype(&dtype)?;
            if float_dtype != FloatDType::F32 {
                device.configure(float_dtype).map_err(|e| {
                    anyhow::anyhow!(
                        "Failed to configure dtype {:?} on {:?}: {:?}",
                        float_dtype,
                        device,
                        e
                    )
                })?;
            }

            println!(
                "Benchmarking {model} (ctx={block_size}, batch={batch_size}, opt={optimizer_choice:?}, dtype={float_dtype:?}) on {device:?} ..."
            );
            let report = run_bench(
                device,
                &gpt_config,
                optimizer_choice,
                batch_size,
                warmup,
                steps,
                decode_tokens,
                ce_chunk_size,
            )?;
            println!(
                "  Eval   : {:>7.1} ms/step  ({:>8.0} tok/s)",
                report.eval_ms_per_step, report.eval_tok_per_sec
            );
            println!(
                "  Train  : {:>7.1} ms/step  ({:>8.0} tok/s)",
                report.train_ms_per_step, report.train_tok_per_sec
            );
            println!(
                "  Decode : {:>7.1} ms/tok   ({:>8.1} tok/s, {decode_tokens} tokens)",
                report.decode_ms_per_tok, report.decode_tok_per_sec
            );
        }
    }

    Ok(())
}
