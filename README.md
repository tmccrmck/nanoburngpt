# NanoBurnGPT

A GPT implementation in Rust using the [Burn](https://burn.dev/) deep learning framework, based on [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT).

Trains on Shakespeare or WikiText-103 using GPT-2's BPE tokenizer. Targets MacBook Air via the `wgpu` (Metal) backend; supports CUDA for cloud GPU training.

## Requirements

- Rust (edition 2024)
- No Python — Metal on macOS (`wgpu` default) or CUDA (`--features cuda`)

## Quick Start

```sh
# Train nano model on Shakespeare (downloads dataset automatically)
cargo run --release -- train

# Generate text
cargo run --release -- generate --prompt "HAMLET:" --max-tokens 500
```

## Commands

### `train`

```
cargo run --release -- train [OPTIONS]

Options:
  --model <PRESET>        Model size preset             [default: nano]
                            nano, gpt2-small, gpt2-medium, gpt2-large, gpt2-xl
  --dataset <NAME>        Dataset to train on           [default: shakespeare]
                            shakespeare, wikitext103
  --n-layer <N>           Override preset: transformer layers
  --n-head <N>            Override preset: attention heads
  --n-embd <N>            Override preset: embedding dimension
  --block-size <N>        Override preset: context window size
  --dropout <F>           Dropout probability           [default: 0.2]
  --batch-size <N>        Batch size                    [default: 32]
  --num-workers <N>       Dataloader workers            [default: 4]
  --learning-rate <F>     Peak learning rate            [default: 0.001]
  --num-epochs <N>        Training epochs               [default: 10]
  --max-train-items <N>   Cap dataset size (0=full)     [default: 0]
```

Saves `artifacts/model_final.mpk` and `artifacts/config.json` on completion.

### `generate`

```
cargo run --release -- generate [OPTIONS]

Options:
  --artifact-dir <PATH>   Where to load model from      [default: artifacts]
  --prompt <STR>          Seed text                     [default: "\n"]
  --max-tokens <N>        Tokens to generate            [default: 500]
  --temperature <F>       Sampling temperature          [default: 0.8]
                          (0 = greedy, higher = more random)
```

## Model Presets

| `--model`     | Layers | Heads | Embd | Context | ~Params |
|---------------|--------|-------|------|---------|---------|
| `nano`        | 2      | 4     | 64   | 32      | <1M     |
| `gpt2-small`  | 12     | 12    | 768  | 1024    | 117M    |
| `gpt2-medium` | 24     | 16    | 1024 | 1024    | 345M    |
| `gpt2-large`  | 36     | 20    | 1280 | 1024    | 762M    |
| `gpt2-xl`     | 48     | 25    | 1600 | 1024    | 1.5B    |

Individual fields can be overridden: `--model gpt2-small --n-layer 8`

## Smoke Test

Verify the pipeline works end-to-end in ~30 seconds:

```sh
cargo run -- train \
  --dataset shakespeare --model nano \
  --batch-size 16 --max-train-items 2000 --num-epochs 1

cargo run -- generate --max-tokens 100
```

## Cloud GPU Training

```sh
# CUDA backend (e.g. RunPod, Lambda Labs)
cargo run --release --features cuda -- train \
  --dataset wikitext103 --model gpt2-small
```

## Architecture

Standard decoder-only Transformer (nanoGPT style):

- **Tokenizer**: GPT-2 BPE via tiktoken-rs (r50k_base, 50257 vocab)
- **Embedding**: token + positional, weight-tied to output projection
- **Blocks**: `n_layer` × (causal self-attention + MLP), pre-norm with LayerNorm
- **Attention**: multi-head causal self-attention with dropout
- **Init**: Normal(0, 0.02) weights; scaled residual init (0.02/√(2·n_layer))
- **Optimizer**: AdamW with gradient clipping (norm 1.0) and cosine LR decay
- **Generation**: temperature-scaled multinomial sampling (or greedy at temperature=0)

## Project Layout

```
src/
  main.rs       CLI entry point
  lib.rs        Module declarations
  data.rs       BPE tokenizer, dataset, batching
  model.rs      GPT, Transformer blocks, attention
  train.rs      Training loop (Burn SupervisedTraining)
  inference.rs  Text generation and model loading
  datasets.rs   Dataset registry (download + preprocessing)
  presets.rs    Model size presets (nano, GPT-2 family)
```
