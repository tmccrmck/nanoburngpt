# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```sh
# Build
cargo build --release

# Smoke test (~30s, verifies full pipeline)
cargo run -- train \
  --dataset shakespeare --model nano \
  --batch-size 16 --max-train-items 2000 --num-epochs 1
cargo run -- generate --max-tokens 100

# Full training (Shakespeare, nano)
cargo run --release -- train --dataset shakespeare --model nano

# Full training (Shakespeare, GPT-2 small)
cargo run --release -- train --dataset shakespeare --model gpt2-small

# Cloud GPU training (CUDA backend)
cargo run --release --features cuda -- train \
  --dataset wikitext103 --model gpt2-small

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
1. `data.rs` — `load_text` reads a pre-downloaded text file, tokenizes with `BpeTokenizer` (tiktoken-rs r50k_base, vocab_size=50257), splits 90/10 into `TextDataset` (train/val), and `TextGenerationBatcher` collates samples into `TextGenerationBatch` (inputs/targets shifted by 1)
2. `model.rs` — `GPT::forward` runs token+position embeddings → N × `Block` (LayerNorm → `CausalSelfAttention` → LayerNorm → `MLP`) → final LayerNorm → weight-tied output projection. `GPT::generate` auto-regressively appends tokens with temperature sampling
3. `train.rs` — `GPT::forward_classification` flattens `[batch, seq, vocab]` → `[batch*seq, vocab]` for cross-entropy. `run_training` wires together Burn's `Learner::new` + `SupervisedTraining::new(...).launch(learner)`. `WarmupCosineScheduler` implements the nanoGPT LR schedule (Burn's `ComposedLrScheduler` combines in parallel, not sequentially, so a custom impl was required)
4. `inference.rs` — loads `artifacts/config.json` and `artifacts/model_final.mpk`, creates `BpeTokenizer::new()`, encodes prompt, calls `GPT::generate`, decodes output
5. `datasets.rs` — `Dataset` enum (Shakespeare, WikiText103); `ensure_downloaded` fetches and preprocesses on first use, caches to `data/<name>/input.txt`
6. `presets.rs` — `ModelPreset` enum (nano, gpt2-small/medium/large/xl); `config()` returns a `GPTConfig` with all architectural parameters pre-filled

**Metrics:** Tracks loss, accuracy, and perplexity (exp(loss)) for both train and validation. Perplexity is the standard LM evaluation metric — lower is better. `MetricCheckpointingStrategy` saves only the best checkpoint by validation perplexity.

**Artifacts** written to `artifacts/` (gitignored): `config.json`, `model_final.mpk`, best-epoch checkpoint, metric CSVs, `experiment.log`.

## Burn 0.20 API

Keep these in mind when modifying training or model code:

- **Training loop**: `Learner::new(model, optim, lr_scheduler)` + `SupervisedTraining::new(dir, dl_train, dl_val)...launch(learner)`. There is no `LearnerBuilder`.
- **`squeeze`**: Takes zero arguments but requires a const-generic output rank: `squeeze::<D>()`. It removes **all** size-1 dims simultaneously — use `reshape([...])` when multiple dims might be size 1 (e.g. batch=1 during inference).
- **`TensorData`**: No `.value` field. Use `.as_slice::<T>()` to read values.
- **`Wgpu` backend**: `Wgpu<f32, i32>` — `AutoGraphicsApi` type param was removed.
- **`#![recursion_limit = "512"]`** in `main.rs` is required due to deep `Sync` trait chains in `wgpu-core 26.x`.
- **Dropout**: `Dropout::forward` is a no-op when not in autodiff mode — no explicit eval mode needed for inference.
- **`GradientClippingConfig`**: at `burn::grad_clipping::GradientClippingConfig`, not `burn::optim`
- **Weight tying**: `Embedding` exposes `weight: Param<Tensor<B, 2>>` as a public field; call `.val()` to get the tensor — this is Burn's standard pattern and participates in autograd correctly
- **Backend feature flags**: `--features cuda` selects CUDA backend; default is `wgpu` (Metal on macOS)
- **`PerplexityMetric`**: Burn internally converts metric outputs to `NdArray` backend — omit the type parameter on `PerplexityMetric::new()` in `.metric_train_numeric()`/`.metric_valid_numeric()` and let inference resolve it. For `MetricCheckpointingStrategy::new()`, use `PerplexityMetric::<B>::new()` since there's no context to infer from.

## Known gaps vs nanoGPT

- Causal mask is recomputed every forward pass instead of cached
- Multinomial sampling pulls probabilities to CPU (`into_data()`) — fine for batch=1 inference
- No Flash Attention (not available in Burn 0.20 wgpu)
- Tokenizer: always GPT-2 BPE (r50k_base, 50257 vocab) — no char-level fallback
