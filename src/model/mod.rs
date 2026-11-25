pub mod attention;
pub mod sampling;

use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig,
        RmsNorm, RmsNormConfig,
    },
    prelude::*,
};

pub use self::attention::{CausalSelfAttention, KVCache, LayerKV, RotaryCache, RotarySlice};
use self::sampling::{mark_seen_tokens, sample_token};
pub use self::sampling::filter_top_k_p;
pub use self::sampling::SamplingParams;

#[derive(Config, Debug)]
pub struct GPTConfig {
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    /// Number of key/value heads (for Grouped Query Attention). If None, defaults to n_head.
    pub n_kv_head: Option<usize>,
    pub block_size: usize, // context window
    pub dropout: f64,
    /// RoPE frequency base (10000.0 = original paper; nanochat uses 100000.0)
    #[config(default = 10000.0)]
    pub rope_theta: f64,
    /// Apply learnable RMSNorm to Q and K after RoPE (stabilizes high learning rates).
    #[config(default = false)]
    pub qk_norm: bool,
    /// Output logit softcap (`cap * tanh(logits / cap)`). `0.0` = disabled (e.g. `15.0` for nanochat).
    #[config(default = 0.0)]
    pub logit_softcap: f64,
}

impl GPTConfig {
    /// Validate architectural invariants before initializing a model.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.vocab_size == 0 {
            anyhow::bail!("vocab_size must be > 0");
        }
        if self.n_layer == 0 {
            anyhow::bail!("n_layer must be > 0");
        }
        if self.n_head == 0 {
            anyhow::bail!("n_head must be > 0");
        }
        if self.n_embd == 0 || !self.n_embd.is_multiple_of(self.n_head) {
            anyhow::bail!(
                "n_embd ({}) must be a positive multiple of n_head ({})",
                self.n_embd,
                self.n_head
            );
        }
        let head_dim = self.n_embd / self.n_head;
        if head_dim == 0 || !head_dim.is_multiple_of(2) {
            anyhow::bail!("head_dim ({head_dim}) must be a positive even number for RoPE");
        }
        if let Some(n_kv_head) = self.n_kv_head
            && (n_kv_head == 0 || !self.n_head.is_multiple_of(n_kv_head))
        {
            anyhow::bail!(
                "n_head ({}) must be a multiple of n_kv_head ({n_kv_head})",
                self.n_head
            );
        }
        if self.block_size == 0 {
            anyhow::bail!("block_size must be > 0");
        }
        if !(0.0..1.0).contains(&self.dropout) {
            anyhow::bail!("dropout must be in [0.0, 1.0), got {}", self.dropout);
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            anyhow::bail!("rope_theta must be finite and > 0.0, got {}", self.rope_theta);
        }
        if !self.logit_softcap.is_finite() || self.logit_softcap < 0.0 {
            anyhow::bail!(
                "logit_softcap must be finite and >= 0.0, got {}",
                self.logit_softcap
            );
        }
        Ok(())
    }
}

#[derive(Module, Debug)]
pub struct MLP {
    c_fc: Linear,
    c_proj: Linear,
    dropout: Dropout,
}

impl MLP {
    pub fn new(config: &GPTConfig, device: &Device) -> Self {
        let proj_std = 0.02 / (2.0 * config.n_layer as f64).sqrt();
        Self {
            c_fc: LinearConfig::new(config.n_embd, 4 * config.n_embd)
                .with_bias(false)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                })
                .init(device),
            c_proj: LinearConfig::new(4 * config.n_embd, config.n_embd)
                .with_bias(false)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: proj_std,
                })
                .init(device),
            dropout: DropoutConfig::new(config.dropout).init(),
        }
    }

    #[inline]
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = self.c_fc.forward(x);
        // `clamp_min(0.0).square()` fuses into a single CubeCL kernel (unlike `relu`,
        // which falls back to `lower_equal_elem + mask_fill` with a bool mask).
        let x = x.clamp_min(0.0).square();
        let x = self.c_proj.forward(x);
        self.dropout.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct Block {
    ln_1: RmsNorm,
    attn: CausalSelfAttention,
    ln_2: RmsNorm,
    mlp: MLP,
}

impl Block {
    pub fn new(config: &GPTConfig, device: &Device) -> Self {
        Self {
            ln_1: RmsNormConfig::new(config.n_embd).init(device),
            attn: CausalSelfAttention::new(config, device),
            ln_2: RmsNormConfig::new(config.n_embd).init(device),
            mlp: MLP::new(config, device),
        }
    }

    #[inline]
    pub fn forward(
        &self,
        x: Tensor<3>,
        rope: &RotarySlice,
        causal_mask: Option<&Tensor<4, Bool>>,
    ) -> Tensor<3> {
        let x = x.clone() + self.attn.forward(self.ln_1.forward(x), rope, causal_mask);
        x.clone() + self.mlp.forward(self.ln_2.forward(x))
    }

    #[inline]
    pub fn forward_cached(
        &self,
        x: Tensor<3>,
        rope: &RotarySlice,
        cache: Option<LayerKV>,
        pos_offset: usize,
        max_len: usize,
    ) -> (Tensor<3>, LayerKV) {
        let (attn_out, new_cache) =
            self.attn
                .forward_cached(self.ln_1.forward(x.clone()), rope, cache, pos_offset, max_len);
        let x = x + attn_out;
        let x = x.clone() + self.mlp.forward(self.ln_2.forward(x));
        (x, new_cache)
    }
}

#[derive(Module, Debug)]
pub struct GPT {
    token_embedding: Embedding,
    rope: RotaryCache,
    causal_mask: Tensor<4, Bool>,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    logit_softcap: f64,
}

impl GPT {
    pub fn new(config: &GPTConfig, device: &Device) -> Self {
        let token_embedding = EmbeddingConfig::new(config.vocab_size, config.n_embd)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            })
            .init(device);

        let head_dim = config.n_embd / config.n_head;
        let rope = RotaryCache::new(config.block_size, head_dim, config.rope_theta, device);

        // Precompute strictly upper-triangular causal mask `[1, 1, block_size, block_size]`
        // once so autodiff training avoids rebuilding the mask on every layer (72 kernels/step).
        let n = config.block_size;
        let mut mask_vals = vec![false; n * n];
        for r in 0..n {
            for c in (r + 1)..n {
                mask_vals[r * n + c] = true;
            }
        }
        let causal_mask = Tensor::<4, Bool>::from_bool(TensorData::new(mask_vals, [1, 1, n, n]), device);

        let blocks = (0..config.n_layer)
            .map(|_| Block::new(config, device))
            .collect();

        let ln_f = RmsNormConfig::new(config.n_embd).init(device);

        Self {
            token_embedding,
            rope,
            causal_mask,
            blocks,
            ln_f,
            logit_softcap: config.logit_softcap,
        }
    }

    /// Run the transformer trunk (`token_embedding` → `blocks` → `ln_f`) without the final
    /// `lm_head` vocabulary projection. Returns `[batch, seq_len, n_embd]`.
    #[inline]
    pub fn forward_hidden(&self, idx: Tensor<2, Int>) -> Tensor<3> {
        let [_, seq_len] = idx.dims();
        let mut x = self.token_embedding.forward(idx);
        let rope = self.rope.slice(0, seq_len);

        // Pass precomputed mask under autodiff or Flex32 (which use attention_fallback);
        // keep `None` + `is_causal: true` for F32/BF16/F16 inference so FlashAttention runs natively.
        let use_precomputed_mask =
            x.device().is_autodiff() || x.dtype() == burn::tensor::DType::Flex32;
        let causal_mask = use_precomputed_mask
            .then(|| self.causal_mask.clone().slice([0..1, 0..1, 0..seq_len, 0..seq_len]));

        for block in &self.blocks {
            x = block.forward(x, &rope, causal_mask.as_ref());
        }

        self.ln_f.forward(x)
    }

    /// Project normalized 2D hidden states `[tokens, n_embd]` to logits `[tokens, vocab_size]`
    /// with optional output logit softcap.
    #[inline]
    pub(crate) fn project_hidden_2d(&self, hidden: Tensor<2>) -> Tensor<2> {
        let weight = self.token_embedding.weight.val();
        let logits = hidden.matmul(weight.transpose());
        if self.logit_softcap > 0.0 {
            let cap = self.logit_softcap;
            (logits / cap).tanh() * cap
        } else {
            logits
        }
    }

    #[inline]
    fn project_lm_head(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, seq_len, n_embd] = x.dims();
        let x = self.ln_f.forward(x);
        let logits = self.project_hidden_2d(x.reshape([batch * seq_len, n_embd]));
        let vocab_size = logits.dims()[1];
        logits.reshape([batch, seq_len, vocab_size])
    }

    #[inline]
    pub fn forward(&self, idx: Tensor<2, Int>) -> Tensor<3> {
        let hidden = self.forward_hidden(idx);
        let [batch, seq_len, n_embd] = hidden.dims();
        let logits = self.project_hidden_2d(hidden.reshape([batch * seq_len, n_embd]));
        let vocab_size = logits.dims()[1];
        logits.reshape([batch, seq_len, vocab_size])
    }

    /// Forward pass that builds/extends a KV cache for incremental decoding.
    #[inline]
    pub fn forward_cached(
        &self,
        idx: Tensor<2, Int>,
        cache: Option<KVCache>,
    ) -> (Tensor<3>, KVCache) {
        let [_, seq_len] = idx.dims();
        let max_len = cache
            .as_ref()
            .and_then(|c| c.layers.first())
            .map_or(seq_len.max(64), |l| l.k.dims()[2]);
        self.forward_cached_internal(idx, cache, max_len, false)
    }

    #[inline]
    fn forward_cached_internal(
        &self,
        idx: Tensor<2, Int>,
        cache: Option<KVCache>,
        max_len: usize,
        last_token_only: bool,
    ) -> (Tensor<3>, KVCache) {
        let [batch, seq_len] = idx.dims();
        let pos_offset = cache.as_ref().map_or(0, KVCache::len);

        let mut x = self.token_embedding.forward(idx);
        let rope = self.rope.slice(pos_offset, seq_len);

        // Move ownership of layer caches so `slice_assign` stays in-place (refcount == 1).
        let mut prev_layers = cache.map(|c| c.layers.into_iter());
        let mut new_layers = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let layer_cache = prev_layers.as_mut().and_then(Iterator::next);
            let (out, layer_kv) =
                block.forward_cached(x, &rope, layer_cache, pos_offset, max_len);
            x = out;
            new_layers.push(layer_kv);
        }

        // During prefill in `generate()`, only the final token's logits are needed;
        // slicing before `ln_f` + `lm_head` saves seq_len× head compute and VRAM.
        let x = if last_token_only && seq_len > 1 {
            let n_embd = x.dims()[2];
            x.slice([0..batch, (seq_len - 1)..seq_len, 0..n_embd])
        } else {
            x
        };

        let logits = self.project_lm_head(x);

        (
            logits,
            KVCache {
                layers: new_layers,
            },
        )
    }

    /// Auto-regressive generation without per-token GPU→CPU sync (best for batched/offline use).
    pub fn generate_tokens(
        &self,
        idx: Tensor<2, Int>,
        max_new_tokens: usize,
        sampling: &SamplingParams,
        block_size: usize,
    ) -> Tensor<2, Int> {
        self.generate_inner(idx, max_new_tokens, sampling, block_size, None::<fn(i32)>)
    }

    /// Auto-regressive generation with pre-allocated KV cache, sliding-window roll-over
    /// past `block_size`, and streaming token callback (for batch index 0).
    pub fn generate(
        &self,
        idx: Tensor<2, Int>,
        max_new_tokens: usize,
        sampling: &SamplingParams,
        block_size: usize,
        on_token: impl FnMut(i32),
    ) -> Tensor<2, Int> {
        self.generate_inner(idx, max_new_tokens, sampling, block_size, Some(on_token))
    }

    fn generate_inner(
        &self,
        idx: Tensor<2, Int>,
        max_new_tokens: usize,
        sampling: &SamplingParams,
        block_size: usize,
        mut on_token: Option<impl FnMut(i32)>,
    ) -> Tensor<2, Int> {
        let [batch, orig_prompt_len] = idx.dims();
        if orig_prompt_len == 0 || max_new_tokens == 0 || block_size == 0 {
            return idx;
        }
        let idx = if orig_prompt_len > block_size {
            idx.slice([0..batch, (orig_prompt_len - block_size)..orig_prompt_len])
        } else {
            idx
        };

        let (mut next_logits, mut cache) =
            self.forward_cached_internal(idx.clone(), None, block_size, true);

        let full_vocab = next_logits.dims()[2];
        let eff_vocab = sampling
            .vocab_limit
            .map_or(full_vocab, |limit| limit.min(full_vocab));

        let use_rep_penalty = sampling.repetition_penalty > 1.0 + 1e-6;
        let mut seen_counts = if use_rep_penalty {
            let device = idx.device();
            Some(mark_seen_tokens(
                Tensor::<2>::zeros([batch, eff_vocab], &device),
                &idx,
            ))
        } else {
            None
        };

        let mut history = idx.clone();
        let mut generated: Vec<Tensor<2, Int>> = Vec::with_capacity(max_new_tokens);
        let mut slide_start = 0usize;
        let slide_keep = (block_size / 2).clamp(1, block_size.saturating_sub(1).max(1));

        for step in 0..max_new_tokens {
            let [_, len, _] = next_logits.dims();
            let logits = next_logits
                .slice([0..batch, len - 1..len, 0..eff_vocab])
                .reshape([batch, eff_vocab]);

            let idx_next = sample_token(&logits, sampling, seen_counts.as_ref());

            if let Some(cb) = on_token.as_mut() {
                let token_id: i32 = idx_next.clone().slice([0..1, 0..1]).into_scalar();
                cb(token_id);
            }

            generated.push(idx_next.clone());
            if step + 1 == max_new_tokens {
                break;
            }

            if let Some(seen) = seen_counts.take() {
                seen_counts = Some(mark_seen_tokens(seen, &idx_next));
            }

            if cache.len() >= block_size {
                // Sliding window: re-prefill the most recent `slide_keep` tokens from pos 0
                // so RoPE positions always stay within `0..block_size`.
                let mut window_parts = Vec::with_capacity(1 + (generated.len() - slide_start));
                window_parts.push(history);
                window_parts.extend(generated[slide_start..].iter().cloned());
                let full_seq = Tensor::cat(window_parts, 1);
                let total_len = full_seq.dims()[1];
                let keep = slide_keep.min(total_len);
                let window = full_seq.slice([0..batch, (total_len - keep)..total_len]);
                history = window.clone();
                slide_start = generated.len();
                let (step_logits, new_cache) =
                    self.forward_cached_internal(window, None, block_size, true);
                next_logits = step_logits;
                cache = new_cache;
            } else {
                let (step_logits, new_cache) =
                    self.forward_cached_internal(idx_next, Some(cache), block_size, true);
                next_logits = step_logits;
                cache = new_cache;
            }
        }

        let mut all = Vec::with_capacity(1 + generated.len());
        all.push(idx);
        all.extend(generated);
        Tensor::cat(all, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> Device {
        Device::flex()
    }

    fn create_test_config() -> GPTConfig {
        GPTConfig {
            vocab_size: 100,
            n_layer: 2,
            n_head: 2,
            n_embd: 32,
            n_kv_head: None,
            block_size: 16,
            dropout: 0.0,
            rope_theta: 10000.0,
            qk_norm: false,
            logit_softcap: 0.0,
        }
    }

    #[test]
    fn test_gpt_forward_shape() {
        let device = test_device();
        let config = create_test_config();
        let gpt = GPT::new(&config, &device);

        let batch_size = 2;
        let seq_len = 10;
        let input = Tensor::<2, Int>::zeros([batch_size, seq_len], &device);

        let output = gpt.forward(input);
        let [b, s, v] = output.dims();

        assert_eq!(b, batch_size);
        assert_eq!(s, seq_len);
        assert_eq!(v, config.vocab_size);
    }

    #[test]
    fn test_kv_cache_equivalence() {
        let device = test_device();
        let config = create_test_config();
        let gpt = GPT::new(&config, &device);

        let batch_size = 1;
        let prompt_len = 5;
        let prompt =
            Tensor::<1, Int>::arange(0..prompt_len as i64, &device).reshape([batch_size, prompt_len]);

        let full_logits = gpt.forward(prompt.clone());
        let expected_logits = full_logits.slice([0..1, prompt_len - 1..prompt_len]);

        let context = prompt.clone().slice([0..1, 0..prompt_len - 1]);
        let (_, cache) = gpt.forward_cached(context, None);

        let last_token = prompt.clone().slice([0..1, prompt_len - 1..prompt_len]);
        let (cached_logits, _) = gpt.forward_cached(last_token, Some(cache));

        let diff = (expected_logits - cached_logits).abs().sum();
        let diff_val: f32 = diff.into_scalar();

        assert!(
            diff_val < 1e-4,
            "KV cache logits differ from full forward pass: {}",
            diff_val
        );
    }

    #[test]
    fn generate_produces_correct_number_of_tokens() {
        let device = test_device();
        let config = create_test_config();
        let gpt = GPT::new(&config, &device);

        let prompt_len = 3;
        let max_new = 5;
        let prompt = Tensor::<2, Int>::zeros([1, prompt_len], &device);
        let sampling = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            vocab_limit: None,
        };

        let output = gpt.generate(prompt, max_new, &sampling, config.block_size, |_| {});
        let [_, total_len] = output.dims();
        assert_eq!(total_len, prompt_len + max_new);
    }

    #[test]
    fn generate_slides_window_past_block_size() {
        let device = test_device();
        let config = create_test_config();
        let gpt = GPT::new(&config, &device);

        let prompt_len = 14;
        let max_new = 10;
        let prompt = Tensor::<2, Int>::zeros([1, prompt_len], &device);
        let sampling = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            vocab_limit: None,
        };

        let mut callback_count = 0usize;
        let output = gpt.generate(prompt, max_new, &sampling, config.block_size, |_| {
            callback_count += 1;
        });
        assert_eq!(output.dims(), [1, prompt_len + max_new]);
        assert_eq!(callback_count, max_new);
    }

    #[test]
    fn test_gqa_attention_shapes() {
        let device = test_device();
        let config = GPTConfig {
            vocab_size: 100,
            n_layer: 1,
            n_head: 4,
            n_kv_head: Some(2),
            n_embd: 32,
            block_size: 16,
            dropout: 0.0,
            rope_theta: 10000.0,
            qk_norm: false,
            logit_softcap: 0.0,
        };
        let gpt = GPT::new(&config, &device);

        let batch_size = 2;
        let seq_len = 10;
        let input = Tensor::<2, Int>::zeros([batch_size, seq_len], &device);

        let output = gpt.forward(input);
        let [b, s, v] = output.dims();

        assert_eq!(b, batch_size);
        assert_eq!(s, seq_len);
        assert_eq!(v, config.vocab_size);
    }

    #[test]
    fn test_gqa_kv_cache_equivalence() {
        let device = test_device();
        let config = GPTConfig {
            vocab_size: 100,
            n_layer: 1,
            n_head: 4,
            n_kv_head: Some(2),
            n_embd: 32,
            block_size: 16,
            dropout: 0.0,
            rope_theta: 10000.0,
            qk_norm: false,
            logit_softcap: 0.0,
        };
        let gpt = GPT::new(&config, &device);

        let batch_size = 1;
        let prompt_len = 5;
        let prompt =
            Tensor::<1, Int>::arange(0..prompt_len as i64, &device).reshape([batch_size, prompt_len]);

        let full_logits = gpt.forward(prompt.clone());
        let expected_logits = full_logits.slice([0..1, prompt_len - 1..prompt_len]);

        let context = prompt.clone().slice([0..1, 0..prompt_len - 1]);
        let (_, cache) = gpt.forward_cached(context, None);

        let [_, cached_heads, _, _] = cache.layers[0].k.dims();
        assert_eq!(cached_heads, 2, "KV cache should store n_kv_head heads");

        let last_token = prompt.clone().slice([0..1, prompt_len - 1..prompt_len]);
        let (cached_logits, _) = gpt.forward_cached(last_token, Some(cache));

        let diff = (expected_logits - cached_logits).abs().sum();
        let diff_val: f32 = diff.into_scalar();

        assert!(
            diff_val < 1e-4,
            "GQA KV cache logits differ from full forward pass: {}",
            diff_val
        );
    }

    #[test]
    fn initial_loss_matches_uniform_vocab() {
        use burn::nn::loss::CrossEntropyLossConfig;

        let device = test_device();
        let vocab_size = 1000;
        let seq_len = 16;
        let config = GPTConfig {
            vocab_size,
            n_layer: 2,
            n_head: 4,
            n_embd: 64,
            n_kv_head: None,
            block_size: seq_len,
            dropout: 0.0,
            rope_theta: 10000.0,
            qk_norm: false,
            logit_softcap: 0.0,
        };
        let gpt = GPT::new(&config, &device);

        let inputs = Tensor::<2, Int>::zeros([2, seq_len], &device);
        let targets = Tensor::<2, Int>::zeros([2, seq_len], &device);
        let logits = gpt.forward(inputs).reshape([2 * seq_len, vocab_size]);
        let loss: f32 = CrossEntropyLossConfig::new()
            .init(&device)
            .forward(logits, targets.reshape([2 * seq_len]))
            .into_scalar();

        // ln(1000) ~= 6.91; with std=0.02 init, loss should be near uniform entropy
        assert!(
            loss > 5.5 && loss < 8.5,
            "expected initial loss near ln(1000)=6.91, got {loss}"
        );
    }

    #[test]
    fn rmsnorm_weights_are_trainable_params() {
        let device = test_device();
        let config = create_test_config();
        let gpt = GPT::new(&config, &device);

        let emb = config.vocab_size * config.n_embd;
        let block_params = config.n_embd * (3 * config.n_embd)
            + config.n_embd * config.n_embd
            + config.n_embd * (4 * config.n_embd)
            + (4 * config.n_embd) * config.n_embd
            + 2 * config.n_embd;
        let expected = emb + config.n_layer * block_params + config.n_embd;
        assert_eq!(gpt.num_params(), expected);
    }

    #[test]
    fn generate_callback_matches_generated_count_past_block_size() {
        let device = test_device();
        let config = create_test_config();
        let gpt = GPT::new(&config, &device);

        let prompt_len = 14;
        let max_new = 25; // crosses block_size=16 multiple times (multiple slides)
        let prompt = Tensor::<2, Int>::zeros([1, prompt_len], &device);
        let sampling = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.1,
            vocab_limit: None,
        };

        let mut callback_count = 0usize;
        let output = gpt.generate(prompt, max_new, &sampling, config.block_size, |_| {
            callback_count += 1;
        });
        assert_eq!(output.dims(), [1, prompt_len + max_new]);
        assert_eq!(callback_count, max_new);
    }

    #[test]
    fn generate_crops_prompt_exceeding_block_size() {
        let device = test_device();
        let config = create_test_config();
        let gpt = GPT::new(&config, &device);

        let prompt = Tensor::<2, Int>::zeros([1, config.block_size + 4], &device);
        let sampling = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            vocab_limit: None,
        };

        let output = gpt.generate(prompt, 5, &sampling, config.block_size, |_| {});
        assert_eq!(output.dims(), [1, config.block_size + 5]);
    }

    #[test]
    fn generate_supports_batch_gt_one() {
        let device = test_device();
        let config = create_test_config();
        let gpt = GPT::new(&config, &device);

        let prompt = Tensor::<2, Int>::zeros([2, 3], &device);
        let sampling = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            vocab_limit: None,
        };

        let output = gpt.generate(prompt, 4, &sampling, config.block_size, |_| {});
        assert_eq!(output.dims(), [2, 7]);
    }

    #[test]
    fn generate_respects_vocab_limit_and_empty_prompt() {
        let device = test_device();
        let config = create_test_config(); // vocab_size = 100
        let gpt = GPT::new(&config, &device);

        // Empty prompt returns immediately without panicking.
        let empty = Tensor::<2, Int>::zeros([1, 0], &device);
        let sampling = SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            vocab_limit: Some(50),
        };
        let out_empty = gpt.generate_tokens(empty, 5, &sampling, config.block_size);
        assert_eq!(out_empty.dims(), [1, 0]);

        // Every generated token must stay strictly below vocab_limit=50.
        let prompt = Tensor::<2, Int>::zeros([1, 3], &device);
        let out = gpt.generate_tokens(prompt, 8, &sampling, config.block_size);
        assert_eq!(out.dims(), [1, 11]);
        let data = out.into_data();
        for &tok in data.as_slice::<i32>().unwrap() {
            assert!((0..50).contains(&tok), "sampled padded token {tok} >= 50");
        }
    }

    #[test]
    fn qk_norm_and_logit_softcap_bound_logits_and_preserve_kv_cache() {
        let device = test_device();
        let mut config = create_test_config();
        config.qk_norm = true;
        config.logit_softcap = 2.0;
        let gpt = GPT::new(&config, &device);

        let prompt = Tensor::<1, Int>::arange(0..6_i64, &device).reshape([1, 6]);
        let full = gpt.forward(prompt.clone());
        let max_abs: f32 = full.clone().abs().max().into_scalar();
        assert!(
            max_abs <= 2.0 + 1e-5,
            "logit softcap=2.0 violated: max |logit| = {max_abs}"
        );

        let expected_last = full.slice([0..1, 5..6]);
        let (_, cache) = gpt.forward_cached(prompt.clone().slice([0..1, 0..5]), None);
        let (cached_last, _) = gpt.forward_cached(prompt.slice([0..1, 5..6]), Some(cache));
        let diff: f32 = (expected_last - cached_last).abs().sum().into_scalar();
        assert!(diff < 1e-4, "QK-norm + softcap KV cache mismatch: {diff}");
    }
}
