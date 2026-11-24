# NanoBurnGPT: Developer Notes

For AI agents and developers working on this codebase.

## Goal

**NanoGPT** (decoder-only Transformer) trained on Shakespeare or WikiText-103, implemented with the **Rust Burn 0.22** deep learning framework (`wgpu` Metal on macOS, `cuda` for NVIDIA GPUs, `flex` on CPU).

Based on [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT).

## Stack

| Component   | Choice                                      |
|-------------|---------------------------------------------|
| Language    | Rust (edition 2024)                         |
| ML framework| [Burn 0.22.1](https://burn.dev/)            |
| Backend     | `wgpu` (Metal), `cuda` (NVIDIA), `flex` (CPU) |
| Dataset     | Shakespeare / WikiText-103 (BPE tokenized)  |
| CLI         | `clap`                                      |
| Tokenizer   | `tiktoken-rs` (GPT-2 BPE r50k_base)         |
| HTTP        | `ureq` (dataset download)                   |

## Project Structure

| File               | Purpose                                              |
|--------------------|------------------------------------------------------|
| `src/main.rs`      | CLI entry point (`train` / `generate` subcommands)   |
| `src/lib.rs`       | Module declarations                                  |
| `src/data.rs`      | `BpeTokenizer`, `TextDataset`, `TextGenerationBatcher` |
| `src/model/mod.rs`      | `GPTConfig`, `GPT`, `Block`, `MLP` (`burn::nn::RmsNorm`)     |
| `src/model/attention.rs`| `CausalSelfAttention`, `KVCache`, `LayerKV`                  |
| `src/model/sampling.rs` | `SamplingParams`, `sample_token`, `filter_top_k_p`           |
| `src/train.rs`     | `TrainStep`, `InferenceStep`, `run_training`         |
| `src/inference.rs` | `generate_text`                                      |

## Status: Working

The full pipeline compiles and runs end-to-end:

- `cargo run -- train` downloads data, trains, saves model
- `cargo run -- generate` loads model and generates text

## Burn 0.22 API Notes

Several API changes in Burn 0.22 are reflected in this codebase:

**Generic-free `Device` API**
- `Tensor<D>`, `Module`, `GPT`, and `Device` no longer take a backend generic parameter `B`.
- Select devices at runtime via `Device::cuda(0)`, `Device::wgpu(...)`, or `Device::flex()`, and wrap with `.autodiff()` for training.

**Learner / training loop**
- No `LearnerBuilder`. Use `Learner::new(model, optim, lr)` + `SupervisedTraining::new(dir, dl_train, dl_val)...launch(learner)`.
- `f64` (a `LearningRate`) directly implements `LrScheduler` — no wrapper needed.
- `DataLoaderBuilder::build()` already returns `Arc<dyn DataLoader<...>>`.

**Tensor ops**
- `squeeze` takes zero arguments but requires a const-generic output rank: `squeeze::<D2>()`.
- `squeeze` removes **all** size-1 dimensions, not just one. If multiple dims are size 1 (e.g. `batch=1, seq=1`), use `reshape([...])` instead to be explicit.
- `unsqueeze` similarly requires a const-generic: `unsqueeze::<D2>()`.
- `TensorData` fields are `bytes`, `shape`, `dtype` — no `.value`. Use `.as_slice::<T>()` to read values.

**Recursion limit**
- `#![recursion_limit = "512"]` is required in `main.rs` due to deep `Sync` trait evaluation chains in `wgpu-core`.

**Dropout**
- `Dropout::forward` is a no-op when not in autodiff mode. Inference is automatically correct — no explicit eval mode needed.

## LR Schedule

`WarmupCosineScheduler` in `src/train.rs` implements nanoGPT's schedule:
1. **Linear warmup** — steps `0..warmup_iters`: ramps from 0 → `learning_rate`
2. **Cosine decay** — steps `warmup_iters..total_iters`: decays from `learning_rate` → `min_lr`
3. **Hold** — steps beyond `total_iters`: stays at `min_lr`

`total_iters` is computed at runtime as `(dataset_size / batch_size) * num_epochs`.

Burn's built-in `ComposedLrScheduler` combines schedulers in **parallel** (multiply/sum), not sequentially, so a custom scheduler was needed.

## AdamW Hyperparameters

NanoGPT defaults, hardcoded (not configurable via CLI):

| Param          | Value |
|----------------|-------|
| `beta1`        | 0.9   |
| `beta2`        | 0.95  |
| `epsilon`      | 1e-8  |
| `weight_decay` | 0.1   |
| Gradient clip  | norm 1.0 |

## Key Constraints

- `n_embd` must be divisible by `n_head` (determines `head_dim = n_embd / n_head`).
- `head_dim` must be even (RoPE split-half formulation requires it).
- `n_head` must be a multiple of `n_kv_head` when GQA is used.
- The `nano` preset (2 layers, 4 heads, 64 embd) trains in seconds per epoch in release mode on Metal.

## Artifacts

Training writes to `artifacts/` (gitignored):

```
artifacts/
  config.json          GPTConfig (vocab size, architecture)
  model_final.bpk      Final trained weights (Burnpack)
  checkpoint/          Per-epoch checkpoints (model, optim, scheduler)
  train/               Training metric CSVs
  valid/               Validation metric CSVs
```

## Running

```sh
# Smoke test (~30s)
cargo run -- train \
  --num-epochs 1 --n-layer 2 --n-head 4 --n-embd 64 \
  --block-size 32 --batch-size 16 --max-train-items 2000
cargo run -- generate --max-tokens 100

# Real training
cargo run --release -- train

# Generate
cargo run --release -- generate --prompt "HAMLET:" --temperature 0.8
```

## References

- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- [The Burn Book](https://burn.dev/book/)
- [Burn API docs](https://docs.rs/burn/0.22.1/burn/)
