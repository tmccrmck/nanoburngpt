# Remove Bias from Linear Layers Design

## Goal

Remove bias terms from all `Linear` layers in the transformer. Modern LLMs universally omit bias from linear projections — it adds parameters with negligible benefit in pre-norm architectures.

## What changes

**`src/model.rs`** — add `.with_bias(false)` to 4 `LinearConfig` calls:

| Location | Layer | Shape |
|----------|-------|-------|
| `CausalSelfAttention::new` | `c_attn` | `[n_embd, 3 * n_embd]` |
| `CausalSelfAttention::new` | `c_proj` | `[n_embd, n_embd]` |
| `MLP::new` | `c_fc` | `[n_embd, 4 * n_embd]` |
| `MLP::new` | `c_proj` | `[4 * n_embd, n_embd]` |

No other files change. `LayerNorm` bias is left intact — that is removed as part of the RMSNorm task (Tier 1.2).

## Why hardcode false

Bias is not a tunable hyperparameter for this architecture — there is no scenario where re-enabling it would be beneficial. A config knob would add noise to `GPTConfig` with no practical use. YAGNI.

## Checkpoint compatibility

Removing bias vectors changes the saved weight keys. Existing checkpoints cannot be resumed after this change.

## Testing

No new tests required. The 4 existing tests that exercise the full forward pass are sufficient:
- `test_gpt_forward_shape` — verifies I/O contract
- `test_kv_cache_equivalence` — exercises cached and uncached forward paths
- `test_training_convergence` — runs a short training loop end to end

## Files modified

| File | Change |
|------|--------|
| `src/model.rs` | Add `.with_bias(false)` to 4 `LinearConfig` calls |
