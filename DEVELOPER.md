# NanoBurnGPT: Developer Notes

For AI agents and developers working on this codebase.

## Goal

Character-level **NanoGPT** (decoder-only Transformer) trained on Tiny Shakespeare, implemented with the **Rust Burn 0.20** deep learning framework and `wgpu` (Metal) backend.

Based on [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT).

## Stack

| Component   | Choice                                      |
|-------------|---------------------------------------------|
| Language    | Rust (edition 2024)                         |
| ML framework| [Burn 0.20.1](https://burn.dev/)            |
| Backend     | `wgpu` (cross-platform GPU, Metal on macOS) |
| Dataset     | Tiny Shakespeare (character-level)          |
| CLI         | `clap`                                      |
| Config/serde| `serde` + `serde_json`                      |
| HTTP        | `reqwest` (dataset download)                |

## Project Structure

| File               | Purpose                                              |
|--------------------|------------------------------------------------------|
| `src/main.rs`      | CLI entry point (`train` / `generate` subcommands)   |
| `src/lib.rs`       | Module declarations                                  |
| `src/data.rs`      | `CharTokenizer`, `TextDataset`, `TextGenerationBatcher` |
| `src/model.rs`     | `CausalSelfAttention`, `MLP`, `Block`, `GPT`         |
| `src/train.rs`     | `TrainStep`, `InferenceStep`, `run_training`         |
| `src/inference.rs` | `generate_text`                                      |

## Status: Working

The full pipeline compiles and runs end-to-end:

- `cargo run -- train` downloads data, trains, saves model + tokenizer
- `cargo run -- generate` loads model and generates text

## Burn 0.20 API Notes

Several API changes from older Burn versions were resolved during implementation. Keep these in mind when updating:

**Learner / training loop**
- No `LearnerBuilder`. Use `Learner::new(model, optim, lr)` + `SupervisedTraining::new(dir, dl_train, dl_val)...launch(learner)`.
- `f64` (a `LearningRate`) directly implements `LrScheduler` — no wrapper needed.
- `DataLoaderBuilder::build()` already returns `Arc<dyn DataLoader<...>>`.

**Tensor ops**
- `squeeze` takes zero arguments but requires a const-generic output rank: `squeeze::<D2>()`.
- `squeeze` removes **all** size-1 dimensions, not just one. If multiple dims are size 1 (e.g. `batch=1, seq=1`), use `reshape([...])` instead to be explicit.
- `unsqueeze` similarly requires a const-generic: `unsqueeze::<D2>()`.
- `TensorData` fields are `bytes`, `shape`, `dtype` — no `.value`. Use `.as_slice::<T>()` to read values.

**Wgpu backend**
- `Wgpu<f32, i32>` — the `AutoGraphicsApi` type parameter was removed.
- `#![recursion_limit = "512"]` is required in `main.rs` due to deep `Sync` trait evaluation chains in `wgpu-core 26.x`.

**Dropout**
- `Dropout::forward` is a no-op when `B::ad_enabled() == false` (i.e. on non-autodiff backends). Inference is automatically correct — no explicit eval mode needed.

## Key Constraints

- `n_embd` must be divisible by `n_head` (determines `head_dim = n_embd / n_head`).
- Default config (6 layers, 6 heads, 384 embd) trains in ~1–3 min/epoch in release mode on Metal.

## Artifacts

Training writes to `artifacts/` (gitignored):

```
artifacts/
  config.json          GPTConfig (vocab size, architecture)
  tokenizer.json       CharTokenizer (char↔index mappings)
  model_final.mpk      Final trained weights (CompactRecorder)
  checkpoint/          Per-epoch checkpoints (model, optim, scheduler)
  experiment.log       Training log
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
- [Burn API docs](https://docs.rs/burn/0.20.1/burn/)
