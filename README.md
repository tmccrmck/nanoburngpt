# NanoBurnGPT

A character-level GPT implementation in Rust using the [Burn](https://burn.dev/) deep learning framework, based on [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT).

Trains on Tiny Shakespeare and generates text character by character. Targets MacBook Air via the `wgpu` (Metal) backend, but works on any GPU/CPU supported by wgpu.

## Requirements

- Rust (edition 2024)
- No Python, no CUDA — Metal on macOS or any wgpu-compatible GPU

## Quick Start

```sh
# Build
cargo build --release

# Train (downloads dataset automatically on first run)
cargo run --release -- train

# Generate text
cargo run --release -- generate --prompt "HAMLET:" --max-tokens 500
```

## Commands

### `train`

```
cargo run --release -- train [OPTIONS]

Options:
  --n-layer <N>           Transformer layers        [default: 6]
  --n-head <N>            Attention heads           [default: 6]
  --n-embd <N>            Embedding dimension       [default: 384]
  --block-size <N>        Context window            [default: 128]
  --dropout <F>           Dropout probability       [default: 0.2]
  --batch-size <N>        Batch size                [default: 32]
  --num-workers <N>       Dataloader workers        [default: 4]
  --learning-rate <F>     Learning rate             [default: 0.001]
  --num-epochs <N>        Training epochs           [default: 10]
  --max-train-items <N>   Cap dataset size (0=full) [default: 0]
```

> **Note:** `n_embd` must be divisible by `n_head`. Valid combinations: 64/4, 64/8, 128/4, 256/8, 384/6.

Saves `artifacts/model_final.mpk`, `artifacts/tokenizer.json`, and `artifacts/config.json` on completion.

### `generate`

```
cargo run --release -- generate [OPTIONS]

Options:
  --artifact-dir <PATH>   Where to load model from  [default: artifacts]
  --prompt <STR>          Seed text                 [default: "\n"]
  --max-tokens <N>        Tokens to generate        [default: 500]
  --temperature <F>       Sampling temperature      [default: 0.8]
                          (0 = greedy, higher = more random)
```

## Smoke Test

Verify the pipeline works end-to-end in ~30 seconds:

```sh
cargo run -- train \
  --num-epochs 1 --n-layer 2 --n-head 4 --n-embd 64 \
  --block-size 32 --batch-size 16 --max-train-items 2000

cargo run -- generate --max-tokens 100
```

## Training Time

On a MacBook Air (Metal/wgpu), the default 6-layer 384-dim model:

| Mode    | Time per epoch |
|---------|---------------|
| Debug   | ~12 min       |
| Release | ~1–3 min      |

Full 10-epoch run in release mode: roughly 10–30 minutes.

## Architecture

Standard decoder-only Transformer (nanoGPT style):

- **Tokenizer**: character-level, vocab built from training data (~65 chars for Shakespeare)
- **Embedding**: token + positional
- **Blocks**: `n_layer` × (causal self-attention + MLP), pre-norm with LayerNorm
- **Attention**: multi-head causal self-attention with dropout
- **Output**: linear head over vocab with cross-entropy loss
- **Generation**: temperature-scaled multinomial sampling (or greedy at temperature=0)

## Project Layout

```
src/
  main.rs       CLI entry point
  lib.rs        Module declarations
  data.rs       Tokenizer, dataset, batching
  model.rs      GPT, Transformer blocks, attention
  train.rs      Training loop (Burn SupervisedTraining)
  inference.rs  Text generation and model loading
```
