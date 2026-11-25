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
  --n-kv-head <N>         Override preset: key/value heads (GQA)
  --n-embd <N>            Override preset: embedding dimension
  --block-size <N>        Override preset: context window size
  --dropout <F>           Override preset dropout       [default: preset]
  --batch-size <N>        Batch size                    [default: 32]
  --num-workers <N>       Dataloader workers            [default: 4]
  --learning-rate <F>     Peak learning rate            [default: 0.001]
  --num-epochs <N>        Training epochs               [default: 10]
  --warmup-iters <N>      Linear warmup steps           [default: 100]
  --max-train-items <N>   Cap dataset size (0=full)     [default: 0]
  --max-val-items <N>     Cap validation size (0=full)  [default: 0]
  --artifact-dir <PATH>   Output directory              [default: artifacts]
  --resume <EPOCH>        Resume from saved checkpoint epoch
  --optimizer <NAME>      Optimizer: adamw, muon        [default: adamw]
  --grad-accum-steps <N>  Gradient accumulation steps   [default: 1]
  --grad-checkpointing    Enable activation checkpointing to save VRAM
  --rope-theta <F>        RoPE frequency base           [default: 10000.0]
  --qk-norm               Enable learnable RMSNorm on Q and K after RoPE
  --logit-softcap <F>     Output logit softcap (0=off)  [default: 0.0]
  --dtype <DTYPE>         Precision: f32, flex32, bf16, f16 [default: f32]
  --amp                   Shorthand for --dtype flex32 (FP16 Tensor Cores + FP32 storage)
  --multi-gpu <MODE>      Multi-GPU mode: auto, sharded, ddp, off [default: auto]
  --no-tui                Disable interactive TUI; print compact CLI progress logs
```

Saves `artifacts/model_final.bpk`, `artifacts/model_best.bpk` (best validation perplexity), and `artifacts/config.json` on completion.

### `eval`

```
cargo run --release -- eval [OPTIONS]

Options:
  --artifact-dir <PATH>   Where to load model from             [default: artifacts]
  --dataset <NAME>        Built-in dataset val split           [default: shakespeare]
  --file <PATH>           Custom UTF-8 text file (overrides --dataset)
  --batch-size <N>        Evaluation batch size                [default: 8]
  --max-items <N>         Cap evaluated chunks (0 = full)      [default: 0]
  --dtype <DTYPE>         Precision: f32, flex32, bf16, f16    [default: f32]
  --best                  Load model_best.bpk instead of model_final.bpk
```

Reports **Loss**, **Perplexity**, and exact **Bits-Per-Byte (BPB)**.

### `bench`

```
cargo run --release -- bench [OPTIONS]

Options:
  --model <PRESET>        Model size preset                    [default: gpt2-small]
  --block-size <N>        Context window size                  [default: 256]
  --batch-size <N>        Batch size for eval/train            [default: 4]
  --optimizer <NAME>      Optimizer: adamw, muon               [default: adamw]
  --dtype <DTYPE>         Precision: f32, flex32, bf16, f16    [default: f32]
  --warmup <N>            Warmup steps before timing           [default: 3]
  --steps <N>             Timed steps for eval/train           [default: 10]
  --decode-tokens <N>     Tokens to generate in decode bench   [default: 64]
  --qk-norm               Enable QK-Norm
  --logit-softcap <F>     Output logit softcap (0 = off)       [default: 0.0]
```

Measures **Eval**, **Train**, and **KV-cache Decode** throughput on your local GPU (`wgpu`/Metal or CUDA) or `--cpu`.

### `generate`

```
cargo run --release -- generate [OPTIONS]

Options:
  --artifact-dir <PATH>      Where to load model from         [default: artifacts]
  --prompt <STR>             Seed text                        [default: "\n"]
  --max-tokens <N>           Tokens to generate               [default: 500]
  --temperature <F>          Sampling temperature             [default: 0.8]
                             (0 = greedy, higher = more random)
  --top-k <N>                Top-k sampling (0 = disabled)    [default: 0]
  --top-p <F>                Nucleus sampling (1.0 = all)     [default: 1.0]
  --repetition-penalty <F>   Repetition penalty (1.0 = off)   [default: 1.0]
  --dtype <DTYPE>            Precision: f32, flex32, bf16, f16 [default: f32]
  --best                     Load model_best.bpk instead of model_final.bpk
```

## Model Presets

| `--model`     | Layers | Heads | Embd | Context | ~Params |
|---------------|--------|-------|------|---------|---------|
| `nano`        | 2      | 4     | 64   | 32      | 3.3M    |
| `gpt2-small`  | 12     | 12    | 768  | 1024    | 124M    |
| `gpt2-medium` | 24     | 16    | 1024 | 1024    | 354M    |
| `gpt2-large`  | 36     | 20    | 1280 | 1024    | 772M    |
| `gpt2-xl`     | 48     | 25    | 1600 | 1024    | 1.56B   |

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

For high-performance training on NVIDIA GPUs (e.g. RunPod, Lambda Labs, A100/H100), use the provided `Dockerfile`.

### 1. Build and Push (Optional)
If you want to use your own registry:
```sh
docker build -t your-username/nanoburngpt:latest .
docker push your-username/nanoburngpt:latest
```

### 2. Run on RunPod
Deploy a Pod with an NVIDIA GPU (RTX 3090 or higher recommended) and use the image `your-username/nanoburngpt:latest` (or just run from source inside a standard PyTorch Pod).

**From Source inside a Pod:**
```sh
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# Train gpt2-small on WikiText-103
cargo run --release --features cuda -- train \
  --model gpt2-small --dataset wikitext103 --batch-size 64
```

**Using the Helper Script:**
The `run_cloud.sh` script is the default entry point for the Docker image. It defaults to training `gpt2-small` on `wikitext103`.

### 3. Run on Google Colab
Open [`colab.ipynb`](colab.ipynb) in Google Colab with a GPU runtime (**Runtime → Change runtime type → GPU**). The notebook installs Rust, builds the CUDA-only binary without `wgpu`/`tui`/`persistence` (`--no-default-features --features cuda`), overrides LTO settings (`CARGO_PROFILE_RELEASE_LTO=off` and `CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16`) for fast builds within Colab's RAM limit, and runs training with `--no-tui` for clean notebook output.

### Architecture

Standard decoder-only Transformer (nanoGPT / nanochat style):

- **Tokenizer**: GPT-2 BPE via `tiktoken-rs` (`r50k_base`, `50257` vocab, padded to `50304` for 64-aligned CUDA matmuls; BOS token prepended; padded slots masked during sampling)
- **Embedding**: token embedding only, weight-tied to output projection; shared `RotaryCache` (RoPE) on `GPT` replaces learned position embeddings
- **Blocks**: `n_layer` × (causal self-attention + MLP), pre-norm with RMSNorm (`burn::nn::RmsNorm`), optional QK-Norm (`--qk-norm`) and output logit softcap (`--logit-softcap`)
- **Attention**: multi-head causal self-attention with grouped-query attention (GQA), zero-copy `split_qkv`, and precomputed autodiff causal mask
- **MLP**: `n_embd → 4·n_embd → n_embd` with fused Squared ReLU (`clamp_min(0.0).square()`)
- **Init**: `Normal(0, 0.02)` weights; scaled residual init (`0.02/√(2·n_layer)`)
- **Optimizer**: AdamW or hybrid Muon + AdamW (`--optimizer muon`), 0.0 weight decay on 1D norms, gradient clipping (norm 1.0), cosine LR + linear warmup
- **Generation**: 100% on-device GPU top-k/top-p/repetition-penalty sampling; pre-allocated in-place KV cache with sliding-window roll-over past `block_size`

## Project Layout

```
src/
  main.rs          CLI entry point (train, eval, generate, bench)
  lib.rs           Public library exports
  data.rs          BPE tokenizer, zero-copy Arc<[i32]> dataset + chunk jitter, single-H2D batcher
  datasets.rs      Dataset registry (download + canonical train/valid splits)
  eval.rs          Checkpoint evaluation (Loss, Perplexity, exact Bits-Per-Byte)
  bench.rs         Local GPU/CPU throughput benchmark (eval, train, decode)
  inference.rs     UTF-8 streaming text generation and model loading
  presets.rs       Model size presets (nano, GPT-2 family)
  model/
    mod.rs         GPT, Block, MLP, RMSNorm, sliding-window generate
    attention.rs   RotaryCache/RotarySlice, CausalSelfAttention, in-place KVCache
    sampling.rs    SamplingParams, GPU top-k/top-p/repetition-penalty sampling
  train/
    mod.rs         Training loop, optimizer param groups, best-checkpoint export
    scheduler.rs   WarmupCosineScheduler
    metrics.rs     Deadlock-free ScalarLossOutput, ScalarLossMetric, LossPerplexityMetric
    renderer.rs    Compact headless CLI metrics renderer
```
