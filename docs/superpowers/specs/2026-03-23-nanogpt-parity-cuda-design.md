# Design: nanoGPT Parity + CUDA Backend Feature Flag

**Date:** 2026-03-23
**Status:** Approved

## Summary

Three nanoGPT parity improvements plus a CUDA/wgpu feature flag so the codebase can train on cloud GPU instances without source changes.

---

## 1. Weight Tying

**What:** Remove `lm_head: Linear<B>` from the `GPT` struct. Replace `self.lm_head.forward(x)` with a manual matrix multiply using `self.token_embedding.weight.val()` transposed.

**Why:** nanoGPT ties the output projection weight to the input embedding weight. This halves the parameter count for the embedding/lm_head pair (~50M params on GPT-2 medium), regularizes training by sharing the representation space, and is standard practice in language model implementations.

**How:**
```rust
// In GPT::forward, replace self.lm_head.forward(x) with:
let weight = self.token_embedding.weight.val(); // [vocab_size, n_embd]
let [vocab_size, n_embd] = weight.dims();
let [batch, seq, _] = x.dims();
x.reshape([batch * seq, n_embd])
    .matmul(weight.transpose())
    .reshape([batch, seq, vocab_size])
```

Burn's `Embedding` exposes `weight: Param<Tensor<B, 2>>` as a public field. The `val()` call returns the underlying tensor. This is Burn's standard pattern — `Linear::forward` itself calls `self.weight.val()` internally — so gradients flow back through both the embedding lookup and this projection correctly in autodiff mode.

**Impact:** Breaking change to `GPTRecord` serialization. Existing checkpoints in `artifacts/` are invalid after this change (acceptable — no real training has run yet).

---

## 2. Scaled Residual Init + Normal Weight Init

**What:** Initialize all linear layer weights with `Normal(mean=0, std=0.02)`. For residual projection layers (`c_proj` in both `CausalSelfAttention` and `MLP`), use `std = 0.02 / sqrt(2 * n_layer)` instead.

**Why:** nanoGPT explicitly initializes weights with `Normal(0, 0.02)` (vs Burn's default Kaiming uniform) and applies a scaled-down init to residual projections to prevent the residual stream from growing too large in deep networks. The `1/sqrt(2 * n_layer)` factor comes from the fact that each transformer block contributes two residual additions (attention + MLP).

**How:** Use `LinearConfig::with_initializer(Initializer::Normal { mean: 0.0, std })` at construction time in `CausalSelfAttention::new` and `MLP::new`. `Initializer` is imported from `burn::nn::Initializer`. The scaled std is computed from `config.n_layer`, which is already available through `&GPTConfig`.

**Fallback if `with_initializer` doesn't compile:** Scale `c_proj.weight` post-construction by reading the tensor via `.val()`, multiplying by the scale factor, and writing it back via `Param::initialized(id, scaled_tensor)`. This is more verbose but guaranteed to work regardless of `LinearConfig` API shape.

---

## 3. Gradient Clipping

**What:** Add `GradientClippingConfig::Norm(1.0)` to the `AdamWConfig`.

**Why:** nanoGPT clips gradient norm to 1.0 to prevent training instability from large gradient steps, especially early in training.

**How:**
```rust
use burn::optim::GradientClippingConfig;

AdamWConfig::new()
    .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
    // ... other settings
```

Burn's optimizer configs expose `with_grad_clipping` — this is the cleanest path, as the alternative (custom training loop) would require replacing `SupervisedTraining` entirely.

**Fallback if `with_grad_clipping` doesn't compile:** Skip gradient clipping for now and note it in CLAUDE.md as a remaining gap. The model will still train; instability from large gradients is more of a risk in long runs on large models.

---

## 4. CUDA / wgpu Feature Flag

**What:** Convert the hardcoded `wgpu` backend to a Cargo feature flag. `wgpu` remains the default (for local Metal development on macOS). Adding `--features cuda` selects the CUDA backend for cloud GPU training.

**Cargo.toml:**
```toml
[features]
default = ["wgpu"]
wgpu = ["burn/wgpu"]
cuda = ["burn/cuda"]

[dependencies]
burn = { version = "0.20.1", features = ["train", "tui"] }
```

**main.rs:** Extract a generic `dispatch<AB, B>()` function that holds the `match cli.command { ... }` logic, parameterized on autodiff backend `AB` and inner backend `B`. Select the concrete type in `main()` via `#[cfg(feature)]` blocks:

```rust
fn dispatch<AB, B>(cli: Cli, device: B::Device)
where
    AB: AutodiffBackend<InnerBackend = B>,
    AB: Backend<Device = B::Device>,
    B: Backend,
    B::Device: std::fmt::Debug + Clone,
{ ... }

fn main() {
    let cli = Cli::parse();
    #[cfg(feature = "cuda")]
    {
        use burn::backend::cuda::{Cuda, CudaDevice};
        dispatch::<Autodiff<Cuda<f32, i32>>, Cuda<f32, i32>>(cli, CudaDevice::default());
    }
    #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    {
        use burn::backend::wgpu::{Wgpu, WgpuDevice};
        dispatch::<Autodiff<Wgpu<f32, i32>>, Wgpu<f32, i32>>(cli, WgpuDevice::default());
    }
}

#[cfg(not(any(feature = "wgpu", feature = "cuda")))]
compile_error!("Enable at least one backend feature: `wgpu` or `cuda`");
```

CUDA takes precedence when both features are enabled. A `compile_error!` fires if neither feature is active.

**Training on a cloud GPU (after this change):**
```sh
cargo run --release --features cuda -- train
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/model.rs` | Remove `lm_head`; add `Initializer` import; update `CausalSelfAttention::new`, `MLP::new`, `GPT::new`, `GPT::forward` |
| `src/train.rs` | Add `GradientClippingConfig` import; add `.with_grad_clipping(...)` to `AdamWConfig` |
| `Cargo.toml` | Move `burn/wgpu` to feature; add `cuda` feature |
| `src/main.rs` | Replace type aliases with `dispatch` function + `#[cfg(feature)]` blocks |

## Out of Scope

- BPE tokenizer (character-level only)
- Flash Attention (not available in Burn 0.20 wgpu)
- Custom training loop (would unlock per-step grad norm logging but is a large change)
