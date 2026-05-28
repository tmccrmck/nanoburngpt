use burn::tensor::module::attention;
use burn::tensor::backend::ops::AttentionModuleOptions;
use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Initializer, LayerNorm,
        LayerNormConfig, Linear, LinearConfig, RotaryEncoding, RotaryEncodingConfig,
    },
    tensor::{activation, backend::Backend, Bool, Int, Tensor},
};
use burn::prelude::ToElement;
use rand::distr::{Distribution, weighted::WeightedIndex};

// ---------------------------------------------------------------------------
// KV Cache for inference
// ---------------------------------------------------------------------------

/// Sampling parameters for text generation.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f64,
    /// Keep only the top-k highest-probability tokens. 0 = disabled.
    pub top_k: usize,
    /// Nucleus sampling: keep the smallest set of tokens whose cumulative
    /// probability exceeds top_p. 1.0 = disabled.
    pub top_p: f64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

/// Sample next token from logits [batch, vocab], returns [batch, 1] int tensor.
fn sample_token<B: Backend>(
    logits: &Tensor<B, 2>,
    params: &SamplingParams,
    batch: usize,
    vocab: usize,
    device: &B::Device,
    rng: &mut impl rand::Rng,
) -> Tensor<B, 2, Int> {
    if params.temperature < 1e-6 {
        logits.clone().argmax(1).unsqueeze::<2>()
    } else {
        let logits = logits.clone() / params.temperature;

        // Optimization: if we have top_k but no top_p, do filtering on GPU
        if params.top_k > 0 && params.top_p >= 1.0 && params.top_k < vocab {
            let (values, indices) = logits.topk_with_indices(params.top_k, 1);
            let probs = activation::softmax(values, 1);

            let probs_data = probs.into_data().convert::<f32>();
            let indices_data = indices.into_data();

            let probs_f32 = probs_data.as_slice::<f32>().expect("f32 probs");
            // Convert indices to i32 for sampling logic
            let indices_i32 = indices_data.convert::<i32>();
            let indices_slice = indices_i32.as_slice::<i32>().expect("i32 indices");

            let tokens: Vec<i32> = (0..batch)
                .map(|b| {
                    let row_probs = &probs_f32[b * params.top_k..(b + 1) * params.top_k];
                    let row_indices = &indices_slice[b * params.top_k..(b + 1) * params.top_k];
                    let dist = WeightedIndex::new(row_probs).expect("valid weights");
                    row_indices[dist.sample(rng)]
                })
                .collect();

            return Tensor::<B, 1, Int>::from_ints(tokens.as_slice(), device).unsqueeze::<2>();
        }

        // Fallback for nucleus sampling (top_p) or full distribution
        let probs = activation::softmax(logits, 1);
        let probs_data = probs.into_data().convert::<f32>();
        let probs_f32 = probs_data.as_slice::<f32>().expect("f32 probs");

        let tokens: Vec<i32> = (0..batch)
            .map(|b| {
                let row = &probs_f32[b * vocab..(b + 1) * vocab];
                let filtered = filter_top_k_p(row, params.top_k, params.top_p);
                let dist = WeightedIndex::new(&filtered).expect("valid weights");
                dist.sample(rng) as i32
            })
            .collect();

        Tensor::<B, 1, Int>::from_ints(tokens.as_slice(), device).unsqueeze::<2>()
    }
}

/// Apply top-k and top-p filtering to a probability distribution, returning
/// a new distribution with excluded tokens zeroed out and probabilities renormalized.
pub fn filter_top_k_p(probs: &[f32], top_k: usize, top_p: f64) -> Vec<f32> {
    let vocab = probs.len();

    // Sort indices by descending probability
    let mut indices: Vec<usize> = (0..vocab).collect();
    indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    let mut cutoff = vocab;

    // Top-k: keep at most top_k tokens
    if top_k > 0 && top_k < vocab {
        cutoff = top_k;
    }

    // Top-p (nucleus): keep smallest set with cumulative prob >= top_p
    if top_p < 1.0 {
        let mut cumsum = 0.0_f64;
        for (i, &idx) in indices.iter().enumerate() {
            cumsum += probs[idx] as f64;
            if cumsum >= top_p {
                // Keep at least one token, and respect top_k bound
                cutoff = cutoff.min(i + 1);
                break;
            }
        }
    }

    // Zero out tokens outside the cutoff
    let mut result = vec![0.0f32; vocab];
    let mut total = 0.0f32;
    for &idx in &indices[..cutoff] {
        result[idx] = probs[idx];
        total += probs[idx];
    }

    // Renormalize
    if total > 0.0 {
        for p in &mut result {
            *p /= total;
        }
    }

    result
}

/// Per-layer cached K and V tensors: [batch, n_head, cached_seq, head_dim].
pub type LayerKV<B> = (Tensor<B, 4>, Tensor<B, 4>);

/// KV cache across all layers. `layers[i]` holds (K, V) for block `i`.
#[derive(Clone, Debug)]
pub struct KVCache<B: Backend> {
    pub layers: Vec<LayerKV<B>>,
}

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
}

#[derive(Module, Debug)]
pub struct CausalSelfAttention<B: Backend> {
    c_attn: Linear<B>,
    c_proj: Linear<B>,
    resid_dropout: Dropout,
    rope: RotaryEncoding<B>,
    n_head: usize,
    n_kv_head: usize,
    n_embd: usize,
}

impl<B: Backend> CausalSelfAttention<B> {
    pub fn new(config: &GPTConfig, device: &B::Device) -> Self {
        let n_kv_head = config.n_kv_head.unwrap_or(config.n_head);
        assert_eq!(
            config.n_head % n_kv_head,
            0,
            "n_head ({}) must be a multiple of n_kv_head ({})",
            config.n_head,
            n_kv_head
        );
        let head_dim = config.n_embd / config.n_head;
        let c_attn_out_dim = (config.n_head + 2 * n_kv_head) * head_dim;

        // c_proj is a residual projection — scale down by 1/sqrt(2*n_layer) per nanoGPT
        let proj_std = 0.02 / (2.0 * config.n_layer as f64).sqrt();
        Self {
            c_attn: LinearConfig::new(config.n_embd, c_attn_out_dim)
                .with_bias(false)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                })
                .init(device),
            c_proj: LinearConfig::new(config.n_embd, config.n_embd)
                .with_bias(false)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: proj_std,
                })
                .init(device),
            resid_dropout: DropoutConfig::new(config.dropout).init(),
            rope: RotaryEncodingConfig::new(config.block_size, head_dim)
                .with_theta(config.rope_theta as f32)
                .init(device),
            n_head: config.n_head,
            n_kv_head,
            n_embd: config.n_embd,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();
        let head_dim = self.n_embd / self.n_head;

        let qkv = self.c_attn.forward(x.clone());
        let q_dim = self.n_head * head_dim;
        let kv_dim = self.n_kv_head * head_dim;

        let q = qkv
            .clone()
            .slice([0..batch_size, 0..seq_len, 0..q_dim])
            .reshape([batch_size, seq_len, self.n_head, head_dim])
            .permute([0, 2, 1, 3]);
        let k = qkv
            .clone()
            .slice([0..batch_size, 0..seq_len, q_dim..(q_dim + kv_dim)])
            .reshape([batch_size, seq_len, self.n_kv_head, head_dim])
            .permute([0, 2, 1, 3]);
        let v = qkv
            .slice([
                0..batch_size,
                0..seq_len,
                (q_dim + kv_dim)..(q_dim + 2 * kv_dim),
            ])
            .reshape([batch_size, seq_len, self.n_kv_head, head_dim])
            .permute([0, 2, 1, 3]);

        // Apply RoPE to Q and K; V is unchanged
        let q = self.rope.forward(q);
        let k = self.rope.forward(k);

        // Repeat KV heads if n_kv_head < n_head
        let (k, v) = if self.n_kv_head < self.n_head {
            let reps = self.n_head / self.n_kv_head;
            (k.repeat_dim(1, reps), v.repeat_dim(1, reps))
        } else {
            (k, v)
        };

        let y = attention(q, k, v, mask, None, AttentionModuleOptions::default());

        let y = y
            .permute([0, 2, 1, 3])
            .reshape([batch_size, seq_len, self.n_embd]);

        let y = self.c_proj.forward(y);
        self.resid_dropout.forward(y)
    }

    /// Forward pass that returns and accepts cached K, V for incremental decoding.
    /// When `cache` is Some, new K/V are concatenated with the cached values.
    /// Returns (output, (full_k, full_v)) for storage in the KV cache.
    pub fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        cache: Option<LayerKV<B>>,
        mask: Option<Tensor<B, 4, Bool>>,
        pos_offset: usize,
    ) -> (Tensor<B, 3>, LayerKV<B>) {
        let [batch_size, seq_len, _] = x.dims();
        let head_dim = self.n_embd / self.n_head;

        let qkv = self.c_attn.forward(x.clone());
        let q_dim = self.n_head * head_dim;
        let kv_dim = self.n_kv_head * head_dim;

        let q = qkv
            .clone()
            .slice([0..batch_size, 0..seq_len, 0..q_dim])
            .reshape([batch_size, seq_len, self.n_head, head_dim])
            .permute([0, 2, 1, 3]);
        let mut k = qkv
            .clone()
            .slice([0..batch_size, 0..seq_len, q_dim..(q_dim + kv_dim)])
            .reshape([batch_size, seq_len, self.n_kv_head, head_dim])
            .permute([0, 2, 1, 3]);
        let mut v = qkv
            .slice([
                0..batch_size,
                0..seq_len,
                (q_dim + kv_dim)..(q_dim + 2 * kv_dim),
            ])
            .reshape([batch_size, seq_len, self.n_kv_head, head_dim])
            .permute([0, 2, 1, 3]);

        // Apply RoPE to Q and K before caching — positions are baked in
        let q = self.rope.apply(q, pos_offset);
        let k_new = self.rope.apply(k, pos_offset);
        k = k_new;

        // Concatenate with cached K, V from previous steps
        if let Some((cached_k, cached_v)) = cache {
            k = Tensor::cat(vec![cached_k, k], 2);
            v = Tensor::cat(vec![cached_v, v], 2);
        }

        let new_cache = (k.clone(), v.clone());

        // Repeat KV heads if n_kv_head < n_head before calling attention
        let (k_att, v_att) = if self.n_kv_head < self.n_head {
            let reps = self.n_head / self.n_kv_head;
            (k.repeat_dim(1, reps), v.repeat_dim(1, reps))
        } else {
            (k, v)
        };

        let y = attention(q, k_att, v_att, mask, None, AttentionModuleOptions::default());

        let y = y
            .permute([0, 2, 1, 3])
            .reshape([batch_size, seq_len, self.n_embd]);

        let y = self.c_proj.forward(y);
        (self.resid_dropout.forward(y), new_cache)
    }
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    c_fc: Linear<B>,
    c_proj: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> MLP<B> {
    pub fn new(config: &GPTConfig, device: &B::Device) -> Self {
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

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.c_fc.forward(x);
        let x = activation::gelu(x);
        let x = self.c_proj.forward(x);
        self.dropout.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    ln_1: LayerNorm<B>,
    attn: CausalSelfAttention<B>,
    ln_2: LayerNorm<B>,
    mlp: MLP<B>,
}

impl<B: Backend> Block<B> {
    pub fn new(config: &GPTConfig, device: &B::Device) -> Self {
        Self {
            ln_1: LayerNormConfig::new(config.n_embd).init(device),
            attn: CausalSelfAttention::new(config, device),
            ln_2: LayerNormConfig::new(config.n_embd).init(device),
            mlp: MLP::new(config, device),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.ln_1.forward(x.clone()), mask);
        x.clone() + self.mlp.forward(self.ln_2.forward(x))
    }

    pub fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        cache: Option<LayerKV<B>>,
        mask: Option<Tensor<B, 4, Bool>>,
        pos_offset: usize,
    ) -> (Tensor<B, 3>, LayerKV<B>) {
        let (attn_out, new_cache) =
            self.attn
                .forward_cached(self.ln_1.forward(x.clone()), cache, mask, pos_offset);
        let x = x + attn_out;
        let x = x.clone() + self.mlp.forward(self.ln_2.forward(x));
        (x, new_cache)
    }
}

#[derive(Module, Debug)]
pub struct GPT<B: Backend> {
    token_embedding: Embedding<B>,
    blocks: Vec<Block<B>>,
    ln_f: LayerNorm<B>,
    // Pre-computed causal mask [1, 1, block_size, block_size] — created once, sliced in forward.
    // Stored as a raw Tensor (not Param) so it's not a learnable parameter.
    causal_mask: Tensor<B, 4, Bool>,
}

impl<B: Backend> GPT<B> {
    pub fn new(config: &GPTConfig, device: &B::Device) -> Self {
        let token_embedding = EmbeddingConfig::new(config.vocab_size, config.n_embd).init(device);

        let blocks = (0..config.n_layer)
            .map(|_| Block::new(config, device))
            .collect();

        let ln_f = LayerNormConfig::new(config.n_embd).init(device);

        // Pre-compute the full causal mask once as a boolean tensor (true = masked).
        let bs = config.block_size;
        let causal_mask = Tensor::<B, 2>::ones([bs, bs], device)
            .tril(0)
            .equal_elem(0.0)
            .reshape([1, 1, bs, bs]);

        Self {
            token_embedding,
            blocks,
            ln_f,
            causal_mask,
        }
    }

    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq_len] = idx.dims();

        // Token embeddings only — position is handled by RoPE inside attention
        let mut x = self.token_embedding.forward(idx);

        // Slice the pre-computed causal mask to the current sequence length
        let mask = self
            .causal_mask
            .clone()
            .slice([0..1, 0..1, 0..seq_len, 0..seq_len]);

        for block in &self.blocks {
            x = block.forward(x, Some(mask.clone()));
        }

        let x = self.ln_f.forward(x);

        // Weight-tied output projection: x @ token_embedding.weight.T
        let weight = self.token_embedding.weight.val(); // [vocab_size, n_embd]
        let [vocab_size, n_embd] = weight.dims();
        x.reshape([batch * seq_len, n_embd])
            .matmul(weight.transpose())
            .reshape([batch, seq_len, vocab_size])
    }

    /// Forward pass that builds/extends a KV cache for incremental decoding.
    /// - `cache = None`: prefill (process full sequence with causal mask, return initial cache)
    /// - `cache = Some(...)`: decode (process new tokens, no mask needed for single-token steps)
    pub fn forward_cached(
        &self,
        idx: Tensor<B, 2, Int>,
        cache: Option<KVCache<B>>,
    ) -> (Tensor<B, 3>, KVCache<B>) {
        let [batch, seq_len] = idx.dims();

        // Position offset: when we have a cache, new tokens start at cached_len
        let pos_offset = cache.as_ref().map_or(0, |c| c.layers[0].0.dims()[2]);

        // Token embeddings only — position handled by RoPE
        let mut x = self.token_embedding.forward(idx);

        // Causal mask only needed during prefill (multi-token); single-token decode needs no mask
        let mask = if cache.is_none() && seq_len > 1 {
            Some(
                self.causal_mask
                    .clone()
                    .slice([0..1, 0..1, 0..seq_len, 0..seq_len]),
            )
        } else {
            None
        };

        let mut new_layers = Vec::with_capacity(self.blocks.len());
        for (i, block) in self.blocks.iter().enumerate() {
            let layer_cache = cache.as_ref().map(|c| c.layers[i].clone());
            let (out, layer_kv) =
                block.forward_cached(x, layer_cache, mask.clone(), pos_offset);
            x = out;
            new_layers.push(layer_kv);
        }

        let x = self.ln_f.forward(x);

        let weight = self.token_embedding.weight.val();
        let [vocab_size, n_embd] = weight.dims();
        let logits = x
            .reshape([batch * seq_len, n_embd])
            .matmul(weight.transpose())
            .reshape([batch, seq_len, vocab_size]);

        (logits, KVCache { layers: new_layers })
    }

    /// Auto-regressive generation with KV cache.
    /// Calls `on_token(token_id)` for each newly generated token, enabling
    /// streaming output. The callback receives the raw token ID (as i32).
    pub fn generate(
        &self,
        idx: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        sampling: &SamplingParams,
        block_size: usize,
        mut on_token: impl FnMut(i32),
    ) -> Tensor<B, 2, Int> {
        let mut rng = rand::rng();
        let [batch, _] = idx.dims();
        let device = idx.device();

        // Prefill: process the entire prompt, get initial KV cache
        let (logits, mut cache) = self.forward_cached(idx.clone(), None);
        let mut all_tokens = idx;
        let mut next_logits = logits;

        for _ in 0..max_new_tokens {
            let [_, len, vocab] = next_logits.dims();
            let logits = next_logits
                .slice([0..batch, len - 1..len, 0..vocab])
                .reshape([batch, vocab]);

            let idx_next = sample_token::<B>(&logits, sampling, batch, vocab, &device, &mut rng);

            // Stream the token to the caller
            let token_id = idx_next.clone().into_scalar().to_i32();
            on_token(token_id);

            all_tokens = Tensor::cat(vec![all_tokens, idx_next.clone()], 1);

            let [_, total_len] = all_tokens.dims();
            if total_len >= block_size {
                break;
            }

            // Decode step: single token, using KV cache
            let (step_logits, new_cache) = self.forward_cached(idx_next, Some(cache));
            next_logits = step_logits;
            cache = new_cache;
        }
        all_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_flex::Flex;

    // Use i32 integers to match the production Wgpu<f32, i32> backend; generate() reads
    // token ids as i32 and would panic with the NdArray<f32> default of i64.
    type B = Flex;

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
        }
    }

    #[test]
    fn test_gpt_forward_shape() {
        let device = Default::default();
        let config = create_test_config();
        let gpt = GPT::<B>::new(&config, &device);

        let batch_size = 2;
        let seq_len = 10;
        let input = Tensor::<B, 2, Int>::zeros([batch_size, seq_len], &device);

        let output = gpt.forward(input);
        let [b, s, v] = output.dims();

        assert_eq!(b, batch_size);
        assert_eq!(s, seq_len);
        assert_eq!(v, config.vocab_size);
    }

    #[test]
    fn test_kv_cache_equivalence() {
        let device = Default::default();
        let config = create_test_config();
        let gpt = GPT::<B>::new(&config, &device);

        let batch_size = 1;
        let prompt_len = 5;
        // Prompt: [0, 1, 2, 3, 4]
        let prompt = Tensor::<B, 1, Int>::arange(0..prompt_len as i64, &device)
            .reshape([batch_size, prompt_len]);

        // 1. Full forward pass
        let full_logits = gpt.forward(prompt.clone());
        // Last token's logits
        let expected_logits = full_logits.slice([0..1, prompt_len - 1..prompt_len]);

        // 2. Incremental KV cache pass
        // Process first 4 tokens to warm up cache
        let context = prompt.clone().slice([0..1, 0..prompt_len - 1]); // [0, 1, 2, 3]
        let (_, cache) = gpt.forward_cached(context, None);

        // Process 5th token (index 4) using cache
        let last_token = prompt.clone().slice([0..1, prompt_len - 1..prompt_len]); // [4]
        let (cached_logits, _) = gpt.forward_cached(last_token, Some(cache));

        // Compare logits
        let diff = (expected_logits - cached_logits).abs().sum();
        let diff_val = diff.into_scalar();

        assert!(
            diff_val < 1e-4,
            "KV cache logits differ from full forward pass: {}",
            diff_val
        );
    }

    #[test]
    fn top_k_keeps_only_k_tokens() {
        let probs = vec![0.1, 0.5, 0.2, 0.15, 0.05];
        let result = filter_top_k_p(&probs, 2, 1.0);
        // Only indices 1 (0.5) and 2 (0.2) should be non-zero
        assert_eq!(result[0], 0.0);
        assert!(result[1] > 0.0);
        assert!(result[2] > 0.0);
        assert_eq!(result[3], 0.0);
        assert_eq!(result[4], 0.0);
        // Should be renormalized
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    #[test]
    fn top_p_nucleus_filtering() {
        // Probabilities sorted desc: 0.5, 0.3, 0.15, 0.05
        let probs = vec![0.15, 0.5, 0.05, 0.3];
        let result = filter_top_k_p(&probs, 0, 0.8);
        // Cumulative: 0.5 → 0.8 → stop. Keep indices 1 (0.5) and 3 (0.3)
        assert_eq!(result[0], 0.0);
        assert!(result[1] > 0.0);
        assert_eq!(result[2], 0.0);
        assert!(result[3] > 0.0);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    #[test]
    fn top_k_and_top_p_combined() {
        // probs sorted desc: 0.5(idx1), 0.3(idx3), 0.15(idx0), 0.05(idx2)
        // top_k=3 keeps [0.5, 0.3, 0.15], top_p=0.6 then cuts:
        //   cumsum: 0.5 < 0.6, 0.5+0.3=0.8 >= 0.6 → keep 2 tokens
        let probs = vec![0.15, 0.5, 0.05, 0.3];
        let result = filter_top_k_p(&probs, 3, 0.6);
        assert_eq!(result[0], 0.0); // 0.15 excluded by top_p
        assert!(result[1] > 0.0); // 0.5 kept
        assert_eq!(result[2], 0.0); // 0.05 excluded by top_k
        assert!(result[3] > 0.0); // 0.3 kept
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn no_filtering_when_disabled() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let result = filter_top_k_p(&probs, 0, 1.0);
        for (i, &p) in result.iter().enumerate() {
            assert!((p - 0.25).abs() < 1e-5, "index {i}: {p}");
        }
    }

    #[test]
    fn top_k_1_keeps_only_argmax() {
        let probs = vec![0.1, 0.6, 0.2, 0.1];
        let result = filter_top_k_p(&probs, 1, 1.0);
        // Only the highest-prob token (index 1) survives, renormalized to 1.0
        assert_eq!(result[0], 0.0);
        assert!((result[1] - 1.0).abs() < 1e-5, "expected 1.0, got {}", result[1]);
        assert_eq!(result[2], 0.0);
        assert_eq!(result[3], 0.0);
    }


    #[test]
    fn generate_produces_correct_number_of_tokens() {
        let device = Default::default();
        let config = create_test_config();
        let gpt = GPT::<B>::new(&config, &device);

        let prompt_len = 3;
        let max_new = 5;
        let prompt = Tensor::<B, 2, Int>::zeros([1, prompt_len], &device);
        let sampling = SamplingParams { temperature: 0.0, top_k: 0, top_p: 1.0 };

        let output = gpt.generate(prompt, max_new, &sampling, config.block_size, |_| {});
        let [_, total_len] = output.dims();
        assert_eq!(total_len, prompt_len + max_new);
    }

    #[test]
    fn generate_respects_block_size_limit() {
        let device = Default::default();
        let config = create_test_config(); // block_size = 16
        let gpt = GPT::<B>::new(&config, &device);

        // Start with a prompt that's already close to block_size
        let prompt_len = 14;
        let max_new = 10; // Would exceed block_size if unchecked
        let prompt = Tensor::<B, 2, Int>::zeros([1, prompt_len], &device);
        let sampling = SamplingParams { temperature: 0.0, top_k: 0, top_p: 1.0 };

        let output = gpt.generate(prompt, max_new, &sampling, config.block_size, |_| {});
        let [_, total_len] = output.dims();
        // Should stop at block_size (16), not at 14 + 10 = 24
        assert!(total_len <= config.block_size, "output len {total_len} exceeds block_size {}", config.block_size);
    }

    #[test]
    fn test_gqa_attention_shapes() {
        let device = Default::default();
        let config = GPTConfig {
            vocab_size: 100,
            n_layer: 1,
            n_head: 4,
            n_kv_head: Some(2),
            n_embd: 32,
            block_size: 16,
            dropout: 0.0,
            rope_theta: 10000.0,
        };
        let gpt = GPT::<B>::new(&config, &device);

        let batch_size = 2;
        let seq_len = 10;
        let input = Tensor::<B, 2, Int>::zeros([batch_size, seq_len], &device);

        let output = gpt.forward(input);
        let [b, s, v] = output.dims();

        assert_eq!(b, batch_size);
        assert_eq!(s, seq_len);
        assert_eq!(v, config.vocab_size);
    }

    #[test]
    fn test_gqa_kv_cache_equivalence() {
        let device = Default::default();
        let config = GPTConfig {
            vocab_size: 100,
            n_layer: 1,
            n_head: 4,
            n_kv_head: Some(2),
            n_embd: 32,
            block_size: 16,
            dropout: 0.0,
            rope_theta: 10000.0,
        };
        let gpt = GPT::<B>::new(&config, &device);

        let batch_size = 1;
        let prompt_len = 5;
        let prompt = Tensor::<B, 1, Int>::arange(0..prompt_len as i64, &device)
            .reshape([batch_size, prompt_len]);

        // 1. Full forward pass
        let full_logits = gpt.forward(prompt.clone());
        let expected_logits = full_logits.slice([0..1, prompt_len - 1..prompt_len]);

        // 2. Incremental KV cache pass
        let context = prompt.clone().slice([0..1, 0..prompt_len - 1]);
        let (_, cache) = gpt.forward_cached(context, None);

        // Verify shape of KV cache: it must store exactly n_kv_head heads
        let (cached_k, _) = &cache.layers[0];
        let [_, cached_heads, _, _] = cached_k.dims();
        assert_eq!(cached_heads, 2, "KV cache should store n_kv_head heads");

        let last_token = prompt.clone().slice([0..1, prompt_len - 1..prompt_len]);
        let (cached_logits, _) = gpt.forward_cached(last_token, Some(cache));

        let diff = (expected_logits - cached_logits).abs().sum();
        let diff_val = diff.into_scalar();

        assert!(
            diff_val < 1e-4,
            "GQA KV cache logits differ from full forward pass: {}",
            diff_val
        );
    }
}
