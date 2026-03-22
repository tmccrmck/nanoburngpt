# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```sh
# Build
cargo build --release

# Smoke test (~30s, verifies full pipeline)
cargo run -- train \
  --num-epochs 1 --n-layer 2 --n-head 4 --n-embd 64 \
  --block-size 32 --batch-size 16 --max-train-items 2000
cargo run -- generate --max-tokens 100

# Full training
cargo run --release -- train

# Generate text
cargo run --release -- generate --prompt "HAMLET:" --temperature 0.8

# Check / lint
cargo check
cargo clippy
```

There are no automated tests. Verification is done by running the smoke test above.

## Architecture

Decoder-only Transformer (nanoGPT style) implemented with [Burn 0.20](https://burn.dev/) and the `wgpu` (Metal) backend.

**Data flow:**
1. `data.rs` — `load_shakespeare` downloads/reads text, builds a `CharTokenizer` (char↔index), splits into `TextDataset` (train/val), and `TextGenerationBatcher` collates samples into `TextGenerationBatch` (inputs/targets shifted by 1)
2. `model.rs` — `GPT::forward` runs token+position embeddings → N × `Block` (LayerNorm → `CausalSelfAttention` → LayerNorm → `MLP`) → final LayerNorm → `lm_head` Linear. `GPT::generate` auto-regressively appends tokens with temperature sampling
3. `train.rs` — `GPT::forward_classification` flattens `[batch, seq, vocab]` → `[batch*seq, vocab]` for cross-entropy. `run_training` wires together Burn's `Learner::new` + `SupervisedTraining::new(...).launch(learner)`. `WarmupCosineScheduler` implements the nanoGPT LR schedule (Burn's `ComposedLrScheduler` combines in parallel, not sequentially, so a custom impl was required)
4. `inference.rs` — loads `artifacts/config.json`, `artifacts/tokenizer.json`, `artifacts/model_final.mpk`, encodes prompt, calls `GPT::generate`, decodes output

**Artifacts** written to `artifacts/` (gitignored): `config.json`, `tokenizer.json`, `model_final.mpk`, per-epoch checkpoints, metric CSVs, `experiment.log`.

## Burn 0.20 API

Keep these in mind when modifying training or model code:

- **Training loop**: `Learner::new(model, optim, lr_scheduler)` + `SupervisedTraining::new(dir, dl_train, dl_val)...launch(learner)`. There is no `LearnerBuilder`.
- **`squeeze`**: Takes zero arguments but requires a const-generic output rank: `squeeze::<D>()`. It removes **all** size-1 dims simultaneously — use `reshape([...])` when multiple dims might be size 1 (e.g. batch=1 during inference).
- **`TensorData`**: No `.value` field. Use `.as_slice::<T>()` to read values.
- **`Wgpu` backend**: `Wgpu<f32, i32>` — `AutoGraphicsApi` type param was removed.
- **`#![recursion_limit = "512"]`** in `main.rs` is required due to deep `Sync` trait chains in `wgpu-core 26.x`.
- **Dropout**: `Dropout::forward` is a no-op when not in autodiff mode — no explicit eval mode needed for inference.

## Known gaps vs nanoGPT

- Causal mask is recomputed every forward pass instead of cached
- Multinomial sampling pulls probabilities to CPU (`into_data()`) — fine for batch=1 inference
- No Flash Attention (not available in Burn 0.20 wgpu)
- Character-level tokenizer only (nanoGPT uses BPE via tiktoken)

## Burn 0.20 API notes (additional)

- **`GradientClippingConfig`**: at `burn::grad_clipping::GradientClippingConfig`, not `burn::optim`
- **Weight tying**: `Embedding` exposes `weight: Param<Tensor<B, 2>>` as a public field; call `.val()` to get the tensor — this is Burn's standard pattern and participates in autograd correctly
- **Backend feature flags**: `--features cuda` selects CUDA backend; default is `wgpu` (Metal on macOS)
