# RoPE (Rotary Position Embeddings) Design

## Goal

Replace the learned absolute position embedding in `GPT` with Rotary Position Embeddings (RoPE). RoPE encodes position as a rotation applied to Q and K inside each attention layer, generalizes better to unseen sequence lengths, and adds no learnable parameters.

## Architecture

### What changes

**`src/model.rs`** — main changes: remove `position_embedding`, add `rope_cos`/`rope_sin`, add `apply_rope`, thread `(cos, sin)` through Block and CausalSelfAttention. Also update `create_test_config` helper.
**`src/presets.rs`** — add `rope_theta: 10000.0` to all `GPTConfig` constructions.
**`src/main.rs`** — add `--rope-theta` CLI flag to `Train` subcommand; wire it into the `GPTConfig { ... }` literal in `dispatch`.
**`src/train.rs`** — add `rope_theta: 10000.0` to the `GPTConfig` literal in `test_training_convergence`.

### Data structures

```rust
// GPTConfig gains one field with a Burn config default:
#[config(default = 10000.0)]
pub rope_theta: f64,
```

Using `#[config(default = 10000.0)]` (the established Burn pattern, used on every `TrainingConfig` field) rather than `#[serde(default)]`. This:
- Ensures old `config.json` files without the field deserialize correctly via Burn's `Config` machinery
- Keeps the pattern consistent with the rest of the codebase

Note: `#[config(default)]` only affects deserialization and the builder pattern — it does NOT make the field optional in Rust struct literal syntax. All `GPTConfig { ... }` construction sites must explicitly include `rope_theta: 10000.0`. The Files Modified table lists all four files with construction sites.

```rust
// GPT struct:
// REMOVE: position_embedding: Embedding<B>
// ADD (both stored as raw Tensor<B, 2>, not Param — same pattern as causal_mask):
rope_cos: Tensor<B, 2>,  // [block_size, head_dim/2]
rope_sin: Tensor<B, 2>,  // [block_size, head_dim/2]
```

### Precomputation in `GPT::new`

```
head_dim = n_embd / n_head
half     = head_dim / 2

assert!(head_dim % 2 == 0, "head_dim must be even for RoPE split-half formulation");

theta_i[i]       = 1.0 / (rope_theta ^ (2i / head_dim))   for i in 0..half
freq[pos, i]     = pos as f32 * theta_i[i]
rope_cos[pos, i] = cos(freq[pos, i])
rope_sin[pos, i] = sin(freq[pos, i])
```

The tables have shape `[block_size, head_dim/2]` and are precomputed once at init, exactly like `causal_mask`. All standard presets satisfy the assertion (nano: 64/4=16, gpt2-small: 768/12=64, etc.).

The tables are built as `f32` data and loaded onto the device using `Tensor::<B, 2>::from_data(data, device)` (same float dtype as the model weights) so element types match in `apply_rope`.

**Note on checkpoint compatibility:** Removing `position_embedding` from the struct means any checkpoint saved before this change cannot be resumed — Burn will fail to load the old key. All training runs must restart from scratch after this change.

### `apply_rope` free function

```rust
fn apply_rope<B: Backend>(
    x: Tensor<B, 4>,    // [batch, n_head, seq_len, head_dim]
    cos: Tensor<B, 2>,  // [seq_len, head_dim/2]
    sin: Tensor<B, 2>,  // [seq_len, head_dim/2]
) -> Tensor<B, 4>
```

Uses the **split-half formulation**:

```
half = head_dim / 2
x1   = x[..., :half]
x2   = x[..., half:]

cos  = cos.reshape([1, 1, seq_len, half])   // broadcast over batch, n_head
sin  = sin.reshape([1, 1, seq_len, half])

output = cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
```

Applied to Q and K only. V is unchanged.

The two new unit tests (`test_apply_rope_at_position_zero`, `test_apply_rope_shape`) call `apply_rope` directly with raw tensors — no `GPTConfig` construction needed.

### Call stack threading

`GPT::forward`:
- Remove `pos_emb` addition to `tok_emb`
- Slice `rope_cos[0..seq_len]`, `rope_sin[0..seq_len]`
- Pass `(cos_slice, sin_slice)` to each `block.forward(x, mask, cos, sin)`

`GPT::forward_cached`:
- Remove `pos_emb` addition to `tok_emb`
- `pos_offset = cache.layers[0].0.dims()[2]` (unchanged)
- Slice `rope_cos[pos_offset..pos_offset+seq_len]`, `rope_sin[pos_offset..pos_offset+seq_len]`
- Pass to each `block.forward_cached(x, cache, mask, cos, sin)`

`Block::forward` / `Block::forward_cached`:
- Add `cos: Tensor<B, 2>`, `sin: Tensor<B, 2>` parameters
- Pass straight through to `self.attn.forward(...)` / `self.attn.forward_cached(...)`

`CausalSelfAttention::forward` / `forward_cached`:
- Add `cos: Tensor<B, 2>`, `sin: Tensor<B, 2>` parameters
- After splitting Q, K, V from QKV projection:
  ```rust
  let q = apply_rope(q, cos.clone(), sin.clone());
  let k = apply_rope(k, cos, sin);
  ```
- Everything else unchanged

`main.rs` `dispatch` — add `rope_theta` to the `GPTConfig { ... }` literal:
```rust
rope_theta: rope_theta,  // from --rope-theta flag (f64, default 10000.0)
```
Parallel to how `n_layer`, `n_head`, etc. are handled (direct assignment from the CLI arg, not `unwrap_or` since it has a default value).

### KV cache correctness

RoPE is KV-cache compatible because rotation is applied before caching:
- **Prefill** (cache=None): slice `cos/sin` at `[0..seq_len]`. Q and K rotated at absolute positions 0..seq_len-1.
- **Decode** (cache=Some): slice `cos/sin` at `[pos_offset..pos_offset+seq_len]`. New Q and K rotated at their absolute positions. Cached K tensors were already rotated at their original positions when computed. Concatenation is position-correct throughout. (In practice `seq_len=1` during decode as `GPT::generate` processes one token at a time, but the general case is handled correctly.)

## Testing

| Test | What it verifies |
|------|-----------------|
| `test_apply_rope_at_position_zero` | At pos=0, freq=0 so cos=1, sin=0 — rotation is identity. Calls `apply_rope` directly. |
| `test_apply_rope_shape` | Output shape equals input shape. Calls `apply_rope` directly. |
| extend `head_dim_divides_evenly_for_all_presets` | Add `assert!(head_dim % 2 == 0)` to the existing test — the current check (`n_embd % n_head == 0`) does not verify even-ness. |
| `test_gpt_forward_shape` | Existing: I/O contract unchanged |
| `test_kv_cache_equivalence` | Existing: prefill + decode == full forward. Catches any RoPE position bugs in the cache path. |

## Files modified

| File | Change |
|------|--------|
| `src/model.rs` | Remove `position_embedding`; add `rope_cos`/`rope_sin` fields; add `apply_rope` free function; thread `(cos, sin)` through `Block` and `CausalSelfAttention`; update `create_test_config` helper with `rope_theta: 10000.0` |
| `src/presets.rs` | Add `rope_theta: 10000.0` to all `GPTConfig` constructions; extend `head_dim_divides_evenly_for_all_presets` to also assert `head_dim % 2 == 0` |
| `src/main.rs` | Add `--rope-theta: f64` (default 10000.0) to `Train` subcommand; add `rope_theta` field to `GPTConfig { ... }` literal in `dispatch` |
| `src/train.rs` | Add `rope_theta: 10000.0` to `GPTConfig` literal in `test_training_convergence` |
