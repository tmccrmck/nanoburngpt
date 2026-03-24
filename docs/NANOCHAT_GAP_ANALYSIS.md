# Gap Analysis: nanoburngpt vs. nanochat

This document catalogs the architectural and implementation differences between nanoburngpt (Rust/Burn) and [Karpathy's nanochat](https://github.com/karpathy/nanochat), with implementation notes for closing each gap.

nanochat has evolved well beyond the original nanoGPT into a modern LLM training harness incorporating many 2024-2026 research advances. It covers the full pipeline: pretraining, SFT, RL, evaluation, inference, and a chat UI.

---

## Tier 1: Core Architecture (High Impact, Moderate Effort)

These changes affect model quality and are standard in modern LLMs.

### 1.1 RoPE (Rotary Position Embeddings)

| | nanochat | nanoburngpt |
|---|---|---|
| Encoding | RoPE, theta=100000 | Learned absolute position embeddings |

**What:** Replace `position_embedding: Embedding` with a precomputed rotation matrix applied to Q and K tensors. Position information is encoded as rotations in 2D subspaces of the head dimension.

**Why:** RoPE generalizes better to unseen sequence lengths, has no learnable parameters, and is the standard for all modern LLMs (LLaMA, Mistral, GPT-4, etc.).

**Implementation:**
- Remove `position_embedding` from `GPT`
- Precompute `cos` and `sin` tensors of shape `[max_seq_len, head_dim/2]` in `GPT::new`
- In `CausalSelfAttention::forward`, after computing Q and K, apply rotation: split each head into even/odd pairs, apply `(x * cos - y * sin, x * sin + y * cos)`
- Pass position offset for KV cache compatibility (already tracked via `pos_offset`)

**Burn considerations:** Pure tensor math, no new dependencies. The precomputed tables are constant tensors (same pattern as `causal_mask`).

### 1.2 RMSNorm

| | nanochat | nanoburngpt |
|---|---|---|
| Normalization | RMSNorm (no learnable params) | LayerNorm |

**What:** Replace `LayerNorm` with RMSNorm: `x / sqrt(mean(x^2) + eps)`. No mean subtraction, no learnable gamma/beta.

**Why:** Simpler, faster, fewer parameters. Used by LLaMA, Mistral, Gemma, and now nanochat.

**Implementation:**
- Create a `RMSNorm<B>` module (just stores `eps: f64`)
- Replace all `LayerNorm` usage in `Block` and `GPT` (5 instances: post-embedding, 2 per block, final)
- nanochat also applies RMSNorm inside attention (QK-norm) — see 2.1

**Burn considerations:** Burn doesn't have a built-in RMSNorm. Implement as: `x * rsqrt(x.powf(2.0).mean_dim(last) + eps)`. Straightforward.

### 1.3 Remove Bias from All Linear Layers

| | nanochat | nanoburngpt |
|---|---|---|
| Bias | None in any Linear | Standard (with bias) |

**What:** Set `bias=false` on all `LinearConfig` calls.

**Why:** Slightly fewer parameters, marginally faster, standard practice in modern LLMs. The bias terms provide negligible benefit when using pre-norm architecture.

**Implementation:**
- Add `.with_bias(false)` to every `LinearConfig` in `CausalSelfAttention::new` and `MLP::new`
- Verify Burn's `LinearConfig` supports this (it does: `LinearConfig::new(in, out).with_bias(false)`)

### 1.4 Untied Weights

| | nanochat | nanoburngpt |
|---|---|---|
| Weight tying | Untied (separate lm_head) | Tied (embedding reused) |

**What:** Add a separate `lm_head: Linear` output projection instead of reusing `token_embedding.weight`.

**Why:** nanochat (and most modern LLMs) untie these because the input embedding and output projection serve different roles. Untying roughly doubles the embedding parameter count but can improve quality.

**Implementation:**
- Add `lm_head: Linear<B>` to `GPT`
- Initialize with `Normal(0, 0.001)` (nanochat's init)
- Replace the manual `x.matmul(weight.transpose())` in `forward` with `self.lm_head.forward(x)`

**Trade-off:** Increases parameter count significantly for small models (nano: ~3.2M extra params for vocab_size=50257). Consider making this configurable.

### 1.5 Squared ReLU Activation

| | nanochat | nanoburngpt |
|---|---|---|
| MLP activation | relu^2 | GELU |

**What:** Replace `activation::gelu(x)` with `activation::relu(x).powf_scalar(2.0)`.

**Why:** Better training signal for sparse activations. Used in PaLM, nanochat, and others. Empirically outperforms GELU in recent scaling experiments.

**Implementation:** One-line change in `MLP::forward`.

---

## Tier 2: Attention Enhancements (High Impact, Higher Effort)

### 2.1 QK-Norm

| | nanochat | nanoburngpt |
|---|---|---|
| QK normalization | RMSNorm + 1.2x scale | None |

**What:** After computing Q and K (and applying RoPE), normalize each with RMSNorm then multiply by a constant scale factor of 1.2.

**Why:** Prevents attention logits from growing too large in deep networks. The 1.2x "sharpening" factor counteracts the softening effect of normalization.

**Implementation:**
- Apply RMSNorm to Q and K after RoPE
- Multiply by 1.2
- Replaces the manual `1/sqrt(head_dim)` scaling (which SDPA handles internally anyway)

### 2.2 Group-Query Attention (GQA)

| | nanochat | nanoburngpt |
|---|---|---|
| KV heads | Configurable (`n_kv_head`) | Same as `n_head` |

**What:** Use fewer K/V heads than Q heads. Q heads are grouped, and each group shares one K/V head.

**Why:** Reduces KV cache size proportionally (e.g., `n_kv_head = n_head/4` gives 4x smaller cache). Faster inference with minimal quality loss. Used by LLaMA 2 70B, Mistral, etc.

**Implementation:**
- Add `n_kv_head: usize` to `GPTConfig`
- In `CausalSelfAttention`, project K/V to `n_kv_head * head_dim` instead of `n_head * head_dim`
- Before attention, repeat K/V along the head dimension: `K.repeat([1, n_head/n_kv_head, 1, 1])`
- KV cache shrinks proportionally

**Note:** nanochat's auto-config currently sets `n_kv_head = n_head` (full MHA), so this is architecturally supported but not actively used for quality reasons at their scale.

### 2.3 Sliding Window Attention

| | nanochat | nanoburngpt |
|---|---|---|
| Attention span | Full context every layer | Full context every layer |

**What:** Some layers attend only to a short window (e.g., `sequence_len/4`), while others see the full context. Pattern like "SSSL" = 3 short layers, 1 long, repeating.

**Why:** Reduces memory and compute for attention in most layers while maintaining full-context understanding via the long layers. Used by Mistral, nanochat.

**Implementation:**
- Add `window_pattern: String` to config (e.g., "SSSL")
- Per-layer: if `S`, apply an additional mask limiting attention to the last `window_size` positions
- Requires modifying the causal mask per-layer (can't share a single mask across all layers)

**Burn considerations:** The pre-computed causal mask would need per-layer variants. Could store a vec of masks or compute the window mask dynamically for S layers.

---

## Tier 3: Training Enhancements (Medium-High Impact, High Effort)

### 3.1 Muon Optimizer

| | nanochat | nanoburngpt |
|---|---|---|
| Optimizer | Muon (matrix params) + AdamW (embeddings/scalars) | AdamW only |

**What:** A hybrid optimizer that uses Muon (orthogonalized momentum) for matrix-shaped parameters and AdamW for everything else.

**Why:** Muon provides faster convergence for transformer training. nanochat's training efficiency gains are largely attributed to this optimizer.

**Muon algorithm:**
1. Nesterov momentum on gradients
2. Orthogonalize the update via Polar Express (not Newton-Schulz): 5 precomputed coefficient triples approximate `U @ V^T` from SVD
3. NorMuon variance reduction: per-neuron adaptive scaling using factored second moments
4. Cautious update: mask out updates where `sign(update) != sign(gradient)`

**Implementation:** This is the single hardest gap to close. Requires:
- Implementing Polar Express orthogonalization (matrix operations: `G @ G^T @ G` chains)
- NorMuon second-moment tracking
- Parameter group splitting logic
- Per-group LR, momentum, and weight decay schedules

**Burn considerations:** Burn's `Optimizer` trait expects a single optimizer. May need a custom `HybridOptimizer` that dispatches to Muon or AdamW based on parameter shape. Polar Express requires efficient matrix multiply chains.

### 3.2 LR Schedule: Linear Warmdown

| | nanochat | nanoburngpt |
|---|---|---|
| LR schedule | Linear warmup + constant + linear warmdown | Warmup + cosine decay |

**What:** Replace cosine decay with a three-phase schedule: (1) linear warmup, (2) constant LR, (3) linear decay to 5% of peak over the last 65% of training.

**Why:** Simpler, and nanochat's experiments show it works as well or better than cosine at their scale.

**Implementation:** Modify `WarmupCosineScheduler` to support a constant phase and linear warmdown. Could rename to `WarmupConstantWarmdownScheduler`.

### 3.3 Gradient Accumulation

| | nanochat | nanoburngpt |
|---|---|---|
| Effective batch size | Up to 524k tokens via accumulation | Fixed batch_size |

**What:** Accumulate gradients over multiple micro-batches before stepping the optimizer.

**Why:** Enables large effective batch sizes on limited GPU memory. Essential for training gpt2-small+ on consumer hardware.

**Implementation:**
- Add `grad_accum_steps` to `TrainingConfig`
- Pass it to `SupervisedTraining::new(...).with_grad_accumulation(steps)` — Burn 0.20 natively supports this in all strategies (single-GPU, multi-GPU, DDP)
- No custom training loop required

### 3.4 Auto-Scaling Config from `--depth`

| | nanochat | nanoburngpt |
|---|---|---|
| Config | Manual presets | Manual presets |

**What:** A single `--depth` integer determines all hyperparameters: model_dim = depth * 64, num_heads = model_dim / 128, training horizon from scaling laws, optimal batch size from Power Lines paper, weight decay from T_epoch framework.

**Why:** Eliminates hyperparameter tuning. Produces near-optimal configs at any scale.

**Implementation:** Add a `from_depth(depth: usize)` method to `GPTConfig` that computes all fields. The formulas are documented in nanochat's `base_train.py`.

### 3.5 No Gradient Clipping

| | nanochat | nanoburngpt |
|---|---|---|
| Gradient clipping | None | Norm 1.0 |

**What:** nanochat does not clip gradients at all.

**Why:** With the Muon optimizer and QK-norm, gradient explosions are controlled architecturally rather than by clipping. Clipping can interfere with Muon's orthogonalization.

**Implementation:** Remove `GradientClippingConfig` from the optimizer setup. However, this should only be done after adopting Muon + QK-norm — with vanilla AdamW, gradient clipping is still important.

---

## Tier 4: Residual Stream Tricks (Medium Impact, Low-Medium Effort)

These are nanochat-specific innovations from modded-nanogpt.

### 4.1 Per-Layer Residual and x0 Lambdas

**What:** Before each block, scale the residual stream and blend in the original post-embedding representation:
```
x = resid_lambda[i] * x + x0_lambda[i] * x0
x = block(x)
```

**Init:** `resid_lambda` linearly decays from 1.15 to 1.05. `x0_lambda` decays from 0.20 to 0.05.

**Why:** Improves gradient flow and prevents the original signal from being completely washed out in deep networks.

**Implementation:** Add two `Vec<f32>` fields to `GPT`, initialized with the linear ramps. Apply in the forward loop before each block. These are learnable parameters in nanochat (stored as `nn.Parameter`).

### 4.2 Smear (Bigram Mixing)

**What:** Before the transformer blocks, mix each position with the previous position's embedding via a learned gate:
```
gate = smear_lambda * sigmoid(linear(x[:, :, :24]))
x[t] += gate * x[t-1]
```

**Why:** Injects local (bigram) context before self-attention, which helps with short-range dependencies.

**Implementation:** Small Linear(24, 1) + a scalar `smear_lambda`. Apply after embeddings, before blocks. Need to handle the KV cache case (cache the previous embedding).

### 4.3 Backout

**What:** After the last block, subtract a scaled copy of the mid-layer residual:
```
x = x - backout_lambda * x_mid   # x_mid captured at layer n_layer//2
```

**Why:** Removes low-level features that may confuse the output projection.

**Implementation:** Capture `x` at the midpoint of the block loop. Subtract `backout_lambda * x_mid` after the loop. `backout_lambda` initialized to 0.2.

### 4.4 Logit Softcap

**What:** `logits = 15 * tanh(logits / 15)` — smoothly caps logits to [-15, 15].

**Why:** Prevents extreme logit values that can destabilize softmax. Used by Gemma 2 and nanochat.

**Implementation:** One line after the output projection in `forward`.

---

## Tier 5: Value Embeddings (Medium Impact, Medium Effort)

### 5.1 ResFormer-Style Value Embeddings

**What:** On alternating layers, a separate embedding table maps token IDs to values that are gated and added to V:
```
ve = value_embed(token_ids)       # [batch, seq, kv_dim]
gate = 3 * sigmoid(ve_gate(x[:, :, :12]))  # small linear
v = v + gate * ve
```

**Why:** Provides a direct shortcut from token identity to attention values, bypassing the residual stream. The ResFormer paper shows this improves training efficiency.

**Implementation:**
- Add `value_embeddings: Vec<Option<Embedding<B>>>` to `GPT` (alternating layers get one)
- Add a small `ve_gate: Linear` per layer that has VE
- Pass token IDs through to `CausalSelfAttention::forward` so it can look up the value embedding
- Gate and add to V before attention

---

## Tier 6: Tokenizer & Data (Medium Impact, High Effort)

### 6.1 Custom BPE Tokenizer

| | nanochat | nanoburngpt |
|---|---|---|
| Tokenizer | Custom BPE, 32768 vocab | GPT-2 r50k_base, 50257 vocab |

**What:** Train a custom BPE tokenizer on the training data with GPT-4's split pattern and a smaller vocab.

**Why:** 32768 is a power of 2 (better for tensor cores), the vocab is tuned to the actual training distribution, and GPT-4's regex pattern handles whitespace and code better than GPT-2's.

**Implementation:** This is a large undertaking — requires a BPE training pipeline. For parity, could use the `tokenizers` crate (HuggingFace's Rust tokenizer library). Alternative: keep GPT-2 vocab but pad to nearest 64 for efficiency.

### 6.2 BOS-Aligned Best-Fit Packing

| | nanochat | nanoburngpt |
|---|---|---|
| Data packing | BOS-aligned best-fit, 100% utilization | Sequential sliding window |

**What:** Pack multiple documents into each training row, starting each document with BOS. Use a best-fit algorithm with a buffer of 1000 documents. When no document fits, crop the shortest to fill exactly.

**Why:** 100% token utilization (no padding waste), every row starts with BOS (proper document boundaries), and the model sees complete documents rather than arbitrary windows.

**Implementation:** Rewrite the data pipeline. The current `TextDataset` sliding window approach is simple but wastes tokens at document boundaries and doesn't respect BOS boundaries.

---

## Tier 7: Infrastructure (Low-Medium Impact, Variable Effort)

### 7.1 Mixed Precision (bf16/fp16)

| | nanochat | nanoburngpt |
|---|---|---|
| Precision | bf16 (default), fp16, fp8 | f32 |

**What:** Train in bfloat16 or float16 instead of float32.

**Why:** 2x memory reduction, ~2x speedup on modern hardware. Essential for larger models.

**Burn considerations:** Burn's wgpu backend uses f32. The CUDA backend supports f16. bf16 support depends on Burn version and backend. This may require waiting for Burn to add bf16 support.

### 7.2 Multi-GPU (DDP)

| | nanochat | nanoburngpt |
|---|---|---|
| Distribution | DDP via torchrun, ZeRO-2 optimizer sharding | Single GPU (not yet wired up) |

**What:** Burn 0.20 ships a full `DdpTrainingStrategy` in `burn-train` (enabled via `--features collective`). It spawns one worker thread per device, splits the dataloader across devices, and uses `burn-collective` for gradient all-reduce (tree or ring strategy). This is intra-node only (threads, not processes) — multi-node requires a separate global orchestrator.

**What nanoburngpt still needs:**
- Enable `collective` feature in `Cargo.toml`
- Replace `SupervisedTraining::new(...)` with `SupervisedTraining::new(...).with_strategy(DdpTrainingStrategy::new(devices, config))`
- Accept a `--num-gpus` CLI flag and enumerate `CudaDevice(i)` per GPU

**Remaining gaps vs nanochat:**
- nanochat uses NCCL-backed PyTorch DDP, which is highly optimized; `burn-collective` uses Rust threads and may have higher overhead
- ZeRO-2 optimizer sharding is not implemented (each worker holds a full copy of optimizer state)
- Multi-node (>1 machine) requires the global orchestrator and is marked beta

### 7.3 SFT and RL Pipeline

| | nanochat | nanoburngpt |
|---|---|---|
| Pipeline | Pretrain + SFT + RL | Pretrain only |

**What:** Supervised fine-tuning on conversation data (SmolTalk), then RL (GRPO) for reasoning tasks.

**Implementation:** Requires conversation data format, loss masking (only supervise assistant turns), RL reward computation. Large undertaking.

### 7.4 Evaluation Suite

| | nanochat | nanoburngpt |
|---|---|---|
| Eval | DCLM CORE, MMLU, ARC, GSM8K, HumanEval, BPB | Loss, accuracy, perplexity |

**What:** Standard LLM evaluation benchmarks.

**Implementation:** Integrate an eval harness. Could use the `lm-eval` crate or implement a subset directly.

---

## Recommended Implementation Order

For maximum impact with reasonable effort, ordered by priority:

1. **Remove bias** (Tier 1.3) — trivial, no downside
2. **RMSNorm** (Tier 1.2) — small module, touches few files
3. **Squared ReLU** (Tier 1.5) — one-line change
4. **Logit softcap** (Tier 4.4) — one-line change
5. **RoPE** (Tier 1.1) — moderate effort, high value, enables longer sequences
6. **QK-Norm** (Tier 2.1) — small change after RMSNorm exists
7. **Untied weights** (Tier 1.4) — small change, but increases params
8. **Linear warmdown schedule** (Tier 3.2) — modify existing scheduler
9. **Per-layer lambdas** (Tier 4.1) — small addition to forward loop
10. **GQA** (Tier 2.2) — moderate refactor of attention
11. **Gradient accumulation** (Tier 3.3) — Burn native; just wire `--grad-accum-steps` to `.with_grad_accumulation()`
12. **Smear + Backout** (Tiers 4.2, 4.3) — small additions
13. **Value embeddings** (Tier 5.1) — moderate, threading token IDs through layers
14. **Sliding window** (Tier 2.3) — moderate, per-layer mask variants
15. **Muon optimizer** (Tier 3.1) — hardest single item, biggest training impact
16. **Custom tokenizer** (Tier 6.1) — large effort, moderate impact
17. **Mixed precision** (Tier 7.1) — depends on Burn support
18. **Data packing** (Tier 6.2) — significant rewrite of data pipeline

Items 1-4 could be done in an afternoon. Items 5-9 in a few days. Items 10-15 are each multi-day efforts. Items 16-18 are week-scale projects.

---

## Appendix: Competing in OpenAI Parameter Golf

**Competition:** [openai/parameter-golf](https://github.com/openai/parameter-golf)
**Deadline:** April 30, 2026
**Prize:** $1,000,000 in RunPod compute credits
**Constraint:** 16 MB artifact (code + compressed weights) trained in under 10 minutes on 8×H100
**Metric:** Bits per byte (BPB) on FineWeb validation set. Baseline: 1.2244, current SOTA: ~1.119

### What the competition is actually optimizing

BPB is tokenizer-agnostic (bytes, not tokens), so a model with a small vocab (1024 tokens) and a model with a large vocab (50257 tokens) are compared on equal footing. The constraint is really: **how much model quality can you pack into 16 MB of compressed weights, trained in 10 minutes on 8 H100s?**

The competition rewards three things simultaneously:
1. Efficient architectures (quality per parameter)
2. Fast training (quality per GPU-second)
3. Aggressive compression (quality per byte after quantization)

### Can nanoburngpt compete?

The competition does not mandate PyTorch. The only hard requirements are that the artifact runs on 8×H100s, fits under 16 MB, and produces BPB scores on FineWeb. A Rust/Burn submission is technically eligible.

Burn 0.20 has one confirmed critical blocker and one item that needs wiring up:
- **No bf16 support on CUDA** — the competition baseline trains in bf16, giving ~2× speedup. Training in f32 wastes half the H100's compute budget. This is a Burn framework limitation.
- **Multi-GPU DDP exists but is not wired up** — Burn 0.20 ships `DdpTrainingStrategy` via `--features collective`, supporting intra-node multi-GPU via thread-per-device all-reduce. nanoburngpt needs to enable the feature and add a `--num-gpus` flag. Unknown how its performance compares to NCCL-backed PyTorch DDP.

The bf16 gap alone is significant. Whether `burn-collective`'s DDP matches NCCL throughput in practice is unverified.

### What must be implemented to compete (in priority order)

Here is what nanoburngpt needs to be competitive:

#### Must-have (table stakes)

| Item | Why required | Gap doc ref |
|------|-------------|-------------|
| **FineWeb data pipeline** | The metric is evaluated on FineWeb — Shakespeare/WikiText are wrong datasets | New |
| **BPB evaluation metric** | Must report BPB, not perplexity, to submit | New |
| **Small vocab (1024 BPE)** | Embedding tables are a huge fraction of 16 MB budget at vocab=50257 | Tier 6.1 |
| **int8 + zlib compression** | Must fit under 16 MB post-compression | New |
| **bf16 training** | ~2× speedup, required to hit SOTA training efficiency | Tier 7.1 |
| **Multi-GPU (8×H100)** | Time budget calibrated for 8 GPUs; Burn DDP exists but needs wiring up + performance validation | Tier 7.2 |
| **Gradient accumulation** | Burn 0.20 supports this natively; just needs a `--grad-accum-steps` CLI flag wired to `.with_grad_accumulation()` | Tier 3.3 |

#### High-impact architecture changes

These are what separates competitive submissions from the baseline. The competition's baseline already uses most of them:

| Item | BPB impact | Gap doc ref |
|------|-----------|-------------|
| **RoPE** | High — enables longer sequences | Tier 1.1 |
| **RMSNorm** | Medium — faster, less memory | Tier 1.2 |
| **No bias** | Low — fewer wasted parameters | Tier 1.3 |
| **ReLU²** | Medium — better than GELU at this scale | Tier 1.5 |
| **QK-Norm** | Medium — training stability | Tier 2.1 |
| **GQA** | Medium — reduces KV cache and attention params | Tier 2.2 |
| **Muon optimizer** | Very High — the single biggest training efficiency gain | Tier 3.1 |
| **Linear warmdown schedule** | Medium — matches competition baseline | Tier 3.2 |
| **Logit softcap** | Low-Medium — training stability | Tier 4.4 |
| **U-Net skip connections** | Medium — not in our gap doc, used by competition baseline |  New |
| **Per-layer residual scales** | Low-Medium | Tier 4.1 |

#### Compression-specific work (unique to this competition)

These have no equivalent in nanochat but are critical for fitting under 16 MB:

| Item | Description |
|------|-------------|
| **Quantization-Aware Training (QAT)** | Train with int8 quantization in the loop so the model learns to be robust to it. Current SOTA submissions use this. |
| **Vocab size optimization** | The parameter budget calculation: at vocab=50257 with dim=512, embeddings alone cost 50257 × 512 × 4 bytes = ~103 MB uncompressed. At vocab=1024: ~2 MB. The competition baseline uses vocab=1024 for this reason. |
| **Weight sharing / factorization** | Low-rank factorization of weight matrices to compress more aggressively |
| **Mixed-precision quantization** | int8 for most weights, int4 or int6 for less sensitive layers |
| **Post-training quantization (GPTQ-lite)** | Better calibration of quantization than naive rounding |
| **EMA/SWA** | Average weights over training trajectory — often produces better post-quantization quality |

#### Frontier techniques (current SOTA approaches)

These are what the top leaderboard entries are using, beyond what the baseline provides:

| Technique | Description |
|-----------|-------------|
| **Test-time training (TTT)** | Use LoRA to fine-tune at eval time on already-graded FineWeb tokens. Allowed by rules if done only on evaluated tokens. |
| **Sliding window evaluation** | Evaluate with overlapping context windows for better BPB estimation |
| **Longer sequences** | Current SOTA trains with seq_len=2048–4096 instead of 1024 |
| **Smear gate / value embeddings** | Residual stream tricks from nanochat (Tiers 4.2, 5.1) |
| **Ternary/1-bit weights** | Extreme compression; some non-record submissions experiment here |

### Realistic path to a competitive submission

**Phase 1 — Unblock the framework (prerequisite):**
Wait for or contribute bf16 + multi-GPU support to Burn. Without these, a Rust submission trains at ~16× disadvantage versus PyTorch baselines (8× GPUs × 2× precision). Alternatively, the competition could be entered with a single H100 and a longer training budget if OpenAI allows it.

**Phase 2 — Implement must-haves (~1 week of coding):**
1. FineWeb downloader + tokenized shard pipeline
2. BPB metric
3. Small vocab tokenizer (1024 BPE via `tokenizers` crate)
4. int8 quantization + zlib artifact packaging
5. Gradient accumulation

**Phase 3 — Architecture upgrades (~1 week):**
Close the architecture gaps in the order listed in the Recommended Implementation Order section. The most critical for this competition specifically are Muon (Tier 3.1), RoPE (Tier 1.1), and GQA (Tier 2.2).

**Phase 4 — Compression research (~ongoing):**
QAT is the main differentiator among current top submissions. Training the model to be quantization-robust is the highest-leverage remaining technique.

**Phase 5 — Tuning (~ongoing):**
The competition is won on details: vocab size, sequence length, batch size, learning rate schedule, layer count. The `--depth` auto-scaling approach from nanochat is a good starting point for principled hyperparameter search.

### Summary

The competition is currently not viable with nanoburngpt due to Burn framework limitations (no bf16 CUDA, no multi-GPU). If those were resolved, the main new work required is: FineWeb pipeline, BPB metric, small vocab tokenizer, int8+zlib compression, QAT, and the architecture changes already cataloged in this document (particularly Muon, RoPE, GQA). The competition has a strict April 30, 2026 deadline.
