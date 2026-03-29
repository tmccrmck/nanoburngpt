# Remove Bias from Linear Layers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `.with_bias(false)` to all 4 `LinearConfig` calls in the transformer, eliminating bias terms from every linear projection.

**Architecture:** Single-file change to `src/model.rs`. No config changes, no new fields, no new tests needed — existing tests cover the full forward path.

**Tech Stack:** Rust, Burn 0.20. VCS is jj (Jujutsu) — commit with `jj describe -m "..."` then `jj new`.

---

## File map

| File | What changes |
|------|-------------|
| `src/model.rs` | Add `.with_bias(false)` to 4 `LinearConfig` calls |

---

## Task 1: Remove bias from all linear layers

This is a single atomic change — all 4 calls must change together or compilation will still succeed but the work will be incomplete.

**Files:**
- Modify: `src/model.rs:151-156` (`CausalSelfAttention::c_attn`)
- Modify: `src/model.rs:157-162` (`CausalSelfAttention::c_proj`)
- Modify: `src/model.rs:274-279` (`MLP::c_fc`)
- Modify: `src/model.rs:280-285` (`MLP::c_proj`)

- [ ] **Step 1: Add `.with_bias(false)` to `c_attn` in `CausalSelfAttention::new`**

Find this block (around line 151):

```rust
c_attn: LinearConfig::new(config.n_embd, 3 * config.n_embd)
    .with_initializer(Initializer::Normal {
        mean: 0.0,
        std: 0.02,
    })
    .init(device),
```

Change to:

```rust
c_attn: LinearConfig::new(config.n_embd, 3 * config.n_embd)
    .with_bias(false)
    .with_initializer(Initializer::Normal {
        mean: 0.0,
        std: 0.02,
    })
    .init(device),
```

- [ ] **Step 2: Add `.with_bias(false)` to `c_proj` in `CausalSelfAttention::new`**

Find this block (around line 157):

```rust
c_proj: LinearConfig::new(config.n_embd, config.n_embd)
    .with_initializer(Initializer::Normal {
        mean: 0.0,
        std: proj_std,
    })
    .init(device),
```

Change to:

```rust
c_proj: LinearConfig::new(config.n_embd, config.n_embd)
    .with_bias(false)
    .with_initializer(Initializer::Normal {
        mean: 0.0,
        std: proj_std,
    })
    .init(device),
```

- [ ] **Step 3: Add `.with_bias(false)` to `c_fc` in `MLP::new`**

Find this block (around line 274):

```rust
c_fc: LinearConfig::new(config.n_embd, 4 * config.n_embd)
    .with_initializer(Initializer::Normal {
        mean: 0.0,
        std: 0.02,
    })
    .init(device),
```

Change to:

```rust
c_fc: LinearConfig::new(config.n_embd, 4 * config.n_embd)
    .with_bias(false)
    .with_initializer(Initializer::Normal {
        mean: 0.0,
        std: 0.02,
    })
    .init(device),
```

- [ ] **Step 4: Add `.with_bias(false)` to `c_proj` in `MLP::new`**

Find this block (around line 280):

```rust
c_proj: LinearConfig::new(4 * config.n_embd, config.n_embd)
    .with_initializer(Initializer::Normal {
        mean: 0.0,
        std: proj_std,
    })
    .init(device),
```

Change to:

```rust
c_proj: LinearConfig::new(4 * config.n_embd, config.n_embd)
    .with_bias(false)
    .with_initializer(Initializer::Normal {
        mean: 0.0,
        std: proj_std,
    })
    .init(device),
```

- [ ] **Step 5: Compile and run tests**

```bash
cargo check && cargo test
```

Expected: all 30 tests pass, no compilation errors.

- [ ] **Step 6: Commit**

```bash
jj describe -m "Remove bias from all linear layers"
jj new
```

---

## Done

All 4 linear projections (`c_attn`, `c_proj` in attention; `c_fc`, `c_proj` in MLP) now have no bias. Parameter count drops by `4 * n_embd` scalars per layer (e.g. ~256 params removed for nano, ~3072 for gpt2-small).
