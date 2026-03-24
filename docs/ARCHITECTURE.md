# Architecture

NanoBurnGPT is a decoder-only Transformer (GPT-2 style) implemented in Rust using the [Burn 0.20](https://burn.dev/) deep learning framework.

---

## Overview

```
Text
 │
 ▼
BpeTokenizer          tiktoken-rs r50k_base, 50257-token vocab
 │
 ▼
TextDataset           sliding window sequences of length block_size
 │
 ▼
GPT (forward)
 ├── TokenEmbedding    [vocab_size, n_embd]
 ├── PositionEmbedding [block_size, n_embd]
 └── Block × n_layer
      ├── LayerNorm
      ├── CausalSelfAttention
      ├── LayerNorm
      └── MLP
 └── LayerNorm (final)
 └── Logits            x @ token_embedding.weight.T  (weight-tied)
 │
 ▼
CrossEntropyLoss      targets = inputs shifted by 1
```

---

## Tokenizer

**File:** `src/data.rs` — `BpeTokenizer`

Uses GPT-2's pretrained BPE vocabulary via `tiktoken-rs` (`r50k_base`). Fixed vocab size of **50,257 tokens**.

- `encode(text) -> Vec<usize>` — UTF-8 text to token IDs
- `decode(tokens) -> String` — token IDs back to text
- No training required — the vocabulary is GPT-2's, loaded at runtime

BPE encodes English text at roughly 0.25–0.33 tokens per character (e.g. Shakespeare's ~1.1M chars → ~320k tokens). This is far more efficient than character-level encoding, meaning the same context window covers much more semantic content.

---

## Dataset Pipeline

**Files:** `src/datasets.rs`, `src/data.rs`

### Dataset Registry (`datasets.rs`)

`Dataset` enum handles download and caching:

| Variant | Source | Local cache |
|---------|--------|-------------|
| `Shakespeare` | Karpathy's tinyshakespeare (~1MB) | `data/shakespeare/input.txt` |
| `WikiText103` | WikiText-103-raw from S3 (~500MB) | `data/wikitext103/input.txt` |

`ensure_downloaded()` checks for the local file first; downloads and preprocesses only on first use. WikiText-103 is extracted from a zip and stripped of markup tokens (`= Heading =`, ` @-@ `, etc.).

### TextDataset (`data.rs`)

After tokenization, the token sequence is split 90/10 into train and validation sets. A `TextDataset` wraps a flat `Vec<usize>` of token IDs and serves sliding-window samples:

```
tokens:  [t0, t1, t2, t3, t4, t5, ...]
         ├─ input  ─────────────────┤  indices [i .. i+block_size]
                  ├─ target ────────┤  indices [i+1 .. i+block_size+1]
```

Each training example is a pair `(input[0..block_size], target[1..block_size+1])` — the model learns to predict the next token at every position simultaneously.

`TextGenerationBatcher` stacks individual samples into batched `[batch, seq]` integer tensors for the GPU.

---

## Model

**File:** `src/model.rs`

### GPTConfig

```rust
pub struct GPTConfig {
    pub vocab_size: usize,   // always 50257 (BPE)
    pub n_layer:    usize,   // number of transformer blocks
    pub n_head:     usize,   // attention heads per block
    pub n_embd:     usize,   // embedding / residual stream dimension
    pub block_size: usize,   // context window (max sequence length)
    pub dropout:    f64,
}
```

### GPT

The top-level model. Holds:

- `token_embedding: Embedding` — maps token ID → `n_embd`-dim vector
- `position_embedding: Embedding` — learned position encoding for each slot 0..block_size
- `blocks: Vec<Block>` — the transformer stack
- `ln_f: LayerNorm` — final layer norm before output projection

**Weight tying:** there is no separate `lm_head` linear layer. The output projection reuses `token_embedding.weight` transposed:

```
logits = x_flat @ token_embedding.weight.T    shape: [batch*seq, vocab_size]
```

This halves the parameter count for the embedding/output pair and is standard practice in language models.

### Block

Each block is a Pre-LN (pre-normalization) transformer block:

```
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

Pre-LN (normalizing before the sublayer, not after) is more training-stable than the original "Post-LN" GPT-1 design.

### CausalSelfAttention

Multi-head scaled dot-product attention with a causal mask, using Burn's fused SDPA kernel (`burn::tensor::module::attention`).

```
Q, K, V = split(Linear(x, 3*n_embd))          # fused QKV projection
y = attention(Q, K, V, causal_mask)            # fused scaled dot-product attention
y = Linear(y, n_embd)                          # output projection
```

The `attention()` kernel handles `softmax(QK^T / sqrt(d_k)) * V` and causal masking internally. On CUDA backends this may use an optimized kernel; on wgpu it uses a standard fused implementation. Attention dropout is not supported by this kernel (residual dropout is still applied after the output projection).

**Causal mask:** a boolean tensor (`true` = masked) pre-computed once in `GPT::new` at the full `block_size × block_size` and sliced to `seq_len` in `forward`. This avoids recreating the mask on every step.

**Heads:** each of `n_head` heads operates on a `head_dim = n_embd / n_head` subspace independently. Results are concatenated before the output projection.

### MLP

A two-layer feed-forward network with GELU activation and a 4× hidden dimension expansion:

```
x → Linear(n_embd → 4*n_embd) → GELU → Linear(4*n_embd → n_embd) → Dropout
```

### Weight Initialization

Following nanoGPT:
- All linear weights: `Normal(mean=0, std=0.02)`
- Residual projections (`c_proj` in attention and MLP): `Normal(mean=0, std=0.02 / sqrt(2 * n_layer))`

The scaled-down residual init prevents the residual stream from growing too large in deep networks. The `1/sqrt(2*n_layer)` factor accounts for the two residual additions per block (attention + MLP).

---

## Training

**File:** `src/train.rs`

### Optimizer

AdamW with nanoGPT hyperparameters:

| Parameter | Value |
|-----------|-------|
| β₁ | 0.9 |
| β₂ | 0.95 |
| ε | 1e-8 |
| Weight decay | 0.1 |
| Gradient clipping | norm 1.0 |

### Learning Rate Schedule

Linear warmup for `warmup_iters` steps, then cosine decay to `min_lr`:

```
iter < warmup_iters:   lr = max_lr * (iter / warmup_iters)
iter >= total_iters:   lr = min_lr
otherwise:             lr = min_lr + 0.5*(max_lr - min_lr) * (1 + cos(π * progress))
```

Burn's built-in `ComposedLrScheduler` applies sub-schedulers in parallel rather than sequentially, so `WarmupCosineScheduler` implements `LrScheduler` directly.

### Loss

Cross-entropy over the flattened `[batch*seq, vocab]` logit tensor against the target token at each position. Every position in every sequence contributes to the loss equally.

### Training Loop

Uses Burn's `SupervisedTraining` runner:

```
Learner::new(model, AdamW, WarmupCosineScheduler)
SupervisedTraining::new("artifacts", train_dl, val_dl)
    .metric_train_numeric(LossMetric)
    .metric_valid_numeric(LossMetric)
    .metric_train_numeric(AccuracyMetric)
    .metric_valid_numeric(AccuracyMetric)
    .metric_train_numeric(PerplexityMetric)
    .metric_valid_numeric(PerplexityMetric)
    .with_file_checkpointer(CompactRecorder)
    .with_checkpointing_strategy(best_perplexity)
    .launch(learner)
```

Checkpoints are saved per epoch to `artifacts/checkpoint/`. The best checkpoint is selected by lowest validation perplexity (`MetricCheckpointingStrategy`). The final model is saved as `artifacts/model_final.mpk` (MessagePack via `CompactRecorder`).

---

## Inference

**File:** `src/inference.rs`, `src/model.rs` (`GPT::generate`, `GPT::forward_cached`)

### KV Cache

Auto-regressive generation uses a **KV cache** to avoid redundant computation. Instead of re-running the full sequence through every layer on each step, cached K and V tensors from previous steps are reused:

```
# Prefill: process entire prompt, build initial KV cache
logits, cache = model.forward_cached(prompt_tokens, cache=None)

# Decode: one token at a time, extending the cache
for _ in 0..max_new_tokens:
    next = sample(logits[:, -1, :])
    logits, cache = model.forward_cached(next, cache=Some(cache))
```

**Cache structure:** `KVCache` holds a `Vec<(K, V)>` per layer, where K and V have shape `[batch, n_head, cached_seq, head_dim]`. Each decode step concatenates the new token's K/V with the cached tensors along the sequence dimension.

**Masking:** during prefill (multi-token), the pre-computed causal mask is applied. During single-token decode steps, no mask is needed — the new token naturally attends to all cached positions plus itself.

**Position tracking:** the position embedding offset is derived from the cache length (`cache.layers[0].K.dims()[2]`), so new tokens receive the correct positional encoding without re-encoding the full sequence.

**Complexity:** reduces per-step cost from O(seq²) to O(seq) during generation, at the cost of O(n_layer × n_head × seq × head_dim) memory for the cache.

### Sampling

Temperature-scaled multinomial sampling. At `temperature=0`, greedy (argmax) is used instead. Probabilities are pulled to CPU (`into_data()`) for sampling since `rand::WeightedIndex` operates on CPU — acceptable overhead for batch=1 inference.

---

## Backends

**File:** `src/main.rs`, `Cargo.toml`

Selected at compile time via Cargo feature flags:

| Flag | Backend | Target |
|------|---------|--------|
| `--features wgpu` (default) | Burn wgpu | Metal on macOS, Vulkan/DX12 elsewhere |
| `--features cuda` | Burn CUDA (via cubecl) | NVIDIA GPUs |

A `compile_error!` fires if neither feature is enabled. CUDA takes precedence if both are specified.

```sh
# Default (wgpu/Metal on macOS)
cargo run --release -- train

# CUDA
cargo run --release --features cuda -- train
```

---

## Model Presets

**File:** `src/presets.rs`

| Preset | n_layer | n_head | n_embd | block_size | ~Params |
|--------|---------|--------|--------|------------|---------|
| `nano` | 2 | 4 | 64 | 32 | <1M |
| `gpt2-small` | 12 | 12 | 768 | 1024 | 117M |
| `gpt2-medium` | 24 | 16 | 1024 | 1024 | 345M |
| `gpt2-large` | 36 | 20 | 1280 | 1024 | 762M |
| `gpt2-xl` | 48 | 25 | 1600 | 1024 | 1.5B |

All presets use `vocab_size=50257`. Individual fields can be overridden via CLI flags (e.g. `--model gpt2-small --n-layer 8`).

---

## Artifacts

Written to `artifacts/` (gitignored):

| File | Contents |
|------|----------|
| `config.json` | `GPTConfig` serialized by Burn |
| `model_final.mpk` | Final model weights (MessagePack, `CompactRecorder`) |
| `checkpoint/` | Per-epoch weight checkpoints |
| `train/`, `valid/` | Metric CSVs (loss, accuracy per step) |
| `experiment.log` | Burn training log |

Inference loads `config.json` to reconstruct the architecture, then loads weights from `model_final.mpk`. The tokenizer needs no artifact — `BpeTokenizer::new()` loads GPT-2's vocabulary from the `tiktoken-rs` crate at runtime.

---

## Key Differences from nanoGPT

| nanoGPT (Python/PyTorch) | NanoBurnGPT (Rust/Burn) |
|--------------------------|------------------------|
| Python + PyTorch | Rust + Burn 0.20 |
| Flash Attention (PyTorch SDPA) | Burn's fused SDPA kernel (`burn::tensor::module::attention`) |
| KV cache for inference | KV cache (`GPT::forward_cached`, `KVCache`) |
| `torch.compile` | Burn's own graph fusion |
| CPU multinomial sampling | CPU multinomial via `rand::WeightedIndex` |
| Top-k / top-p sampling | Temperature-only sampling (no nucleus sampling yet) |
