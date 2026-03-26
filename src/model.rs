use burn::tensor::module::attention;
use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Initializer, LayerNorm,
        LayerNormConfig, Linear, LinearConfig,
    },
    tensor::{Bool, Int, Tensor, activation, backend::Backend},
};
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
        let probs = activation::softmax(logits, 1);
        let probs_data = probs.into_data();
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
fn filter_top_k_p(probs: &[f32], top_k: usize, top_p: f64) -> Vec<f32> {
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
    n_head: usize,
    n_embd: usize,
}

impl<B: Backend> CausalSelfAttention<B> {
    pub fn new(config: &GPTConfig, device: &B::Device) -> Self {
        // c_proj is a residual projection — scale down by 1/sqrt(2*n_layer) per nanoGPT
        let proj_std = 0.02 / (2.0 * config.n_layer as f64).sqrt();
        Self {
            c_attn: LinearConfig::new(config.n_embd, 3 * config.n_embd)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                })
                .init(device),
            c_proj: LinearConfig::new(config.n_embd, config.n_embd)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: proj_std,
                })
                .init(device),
            resid_dropout: DropoutConfig::new(config.dropout).init(),
            n_head: config.n_head,
            n_embd: config.n_embd,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 4, Bool>>,
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();
        let head_dim = self.n_embd / self.n_head;

        let qkv = self.c_attn.forward(x.clone());
        let qkv = qkv.reshape([batch_size, seq_len, 3, self.n_head, head_dim]);
        let qkv = qkv.permute([2, 0, 3, 1, 4]);
        let q = qkv
            .clone()
            .slice([0..1])
            .reshape([batch_size, self.n_head, seq_len, head_dim]);
        let k = qkv
            .clone()
            .slice([1..2])
            .reshape([batch_size, self.n_head, seq_len, head_dim]);
        let v = qkv
            .clone()
            .slice([2..3])
            .reshape([batch_size, self.n_head, seq_len, head_dim]);

        // Apply RoPE to Q and K; V is unchanged
        let q = apply_rope(q, cos.clone(), sin.clone());
        let k = apply_rope(k, cos, sin);

        let y = attention(q, k, v, mask);

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
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
    ) -> (Tensor<B, 3>, LayerKV<B>) {
        let [batch_size, seq_len, _] = x.dims();
        let head_dim = self.n_embd / self.n_head;

        let qkv = self.c_attn.forward(x.clone());
        let qkv = qkv.reshape([batch_size, seq_len, 3, self.n_head, head_dim]);
        let qkv = qkv.permute([2, 0, 3, 1, 4]);

        let q = qkv
            .clone()
            .slice([0..1])
            .reshape([batch_size, self.n_head, seq_len, head_dim]);
        let mut k = qkv
            .clone()
            .slice([1..2])
            .reshape([batch_size, self.n_head, seq_len, head_dim]);
        let mut v = qkv
            .clone()
            .slice([2..3])
            .reshape([batch_size, self.n_head, seq_len, head_dim]);

        // Apply RoPE to Q and K before caching — positions are baked in
        let q = apply_rope(q, cos.clone(), sin.clone());
        k = apply_rope(k, cos, sin);

        // Concatenate with cached K, V from previous steps
        if let Some((cached_k, cached_v)) = cache {
            k = Tensor::cat(vec![cached_k, k], 2);
            v = Tensor::cat(vec![cached_v, v], 2);
        }

        let new_cache = (k.clone(), v.clone());

        let y = attention(q, k, v, mask);

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
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                })
                .init(device),
            c_proj: LinearConfig::new(4 * config.n_embd, config.n_embd)
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
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.ln_1.forward(x.clone()), mask, cos, sin);
        let x = x.clone() + self.mlp.forward(self.ln_2.forward(x));
        x
    }

    pub fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        cache: Option<LayerKV<B>>,
        mask: Option<Tensor<B, 4, Bool>>,
        cos: Tensor<B, 4>,
        sin: Tensor<B, 4>,
    ) -> (Tensor<B, 3>, LayerKV<B>) {
        let (attn_out, new_cache) =
            self.attn
                .forward_cached(self.ln_1.forward(x.clone()), cache, mask, cos, sin);
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
    // RoPE frequency tables [1, 1, block_size, head_dim/2] — precomputed, constant (no grad).
    // Pre-shaped for broadcasting over batch and n_head dims; no reshape needed in forward.
    rope_cos: Tensor<B, 4>,
    rope_sin: Tensor<B, 4>,
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

        // Pre-compute RoPE frequency tables.
        // Split-half formulation: head_dim must be even.
        let head_dim = config.n_embd / config.n_head;
        assert!(
            head_dim % 2 == 0,
            "head_dim ({head_dim}) must be even for RoPE split-half formulation"
        );
        let half = head_dim / 2;
        let theta = config.rope_theta as f32;

        // inv_freqs[i] = 1 / (rope_theta ^ (2i / head_dim))
        let inv_freqs: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        // Build flat [bs * half] cos/sin arrays, then reshape
        let mut cos_data = vec![0.0f32; bs * half];
        let mut sin_data = vec![0.0f32; bs * half];
        for pos in 0..bs {
            for i in 0..half {
                let angle = pos as f32 * inv_freqs[i];
                cos_data[pos * half + i] = angle.cos();
                sin_data[pos * half + i] = angle.sin();
            }
        }

        let rope_cos = Tensor::<B, 1>::from_floats(cos_data.as_slice(), device)
            .reshape([1, 1, bs, half]);
        let rope_sin = Tensor::<B, 1>::from_floats(sin_data.as_slice(), device)
            .reshape([1, 1, bs, half]);

        Self {
            token_embedding,
            blocks,
            ln_f,
            causal_mask,
            rope_cos,
            rope_sin,
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

        // Slice RoPE tables for positions 0..seq_len
        let half = self.rope_cos.dims()[3];
        let cos = self.rope_cos.clone().slice([0..1, 0..1, 0..seq_len, 0..half]);
        let sin = self.rope_sin.clone().slice([0..1, 0..1, 0..seq_len, 0..half]);

        for block in &self.blocks {
            x = block.forward(x, Some(mask.clone()), cos.clone(), sin.clone());
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

        // Slice RoPE tables at the correct absolute positions
        let half = self.rope_cos.dims()[3];
        let cos = self
            .rope_cos
            .clone()
            .slice([0..1, 0..1, pos_offset..pos_offset + seq_len, 0..half]);
        let sin = self
            .rope_sin
            .clone()
            .slice([0..1, 0..1, pos_offset..pos_offset + seq_len, 0..half]);

        let mut new_layers = Vec::with_capacity(self.blocks.len());
        for (i, block) in self.blocks.iter().enumerate() {
            let layer_cache = cache.as_ref().map(|c| c.layers[i].clone());
            let (out, layer_kv) =
                block.forward_cached(x, layer_cache, mask.clone(), cos.clone(), sin.clone());
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
            let token_data = idx_next.clone().into_data();
            let token_id = token_data.as_slice::<i32>().expect("i32")[0];
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

/// Apply Rotary Position Embeddings to a query or key tensor.
///
/// Uses the split-half formulation: the first `head_dim/2` dims pair with
/// the second `head_dim/2` dims. Applied to Q and K only, never to V.
///
/// # Arguments
/// * `x`   — `[batch, n_head, seq_len, head_dim]`
/// * `cos` — `[1, 1, seq_len, head_dim/2]`  (precomputed, sliced to seq positions, pre-shaped for broadcast)
/// * `sin` — `[1, 1, seq_len, head_dim/2]`
fn apply_rope<B: Backend>(
    x: Tensor<B, 4>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [batch, n_head, seq_len, head_dim] = x.dims();
    let half = head_dim / 2;

    let x1 = x.clone().slice([0..batch, 0..n_head, 0..seq_len, 0..half]);
    let x2 = x.slice([0..batch, 0..n_head, 0..seq_len, half..head_dim]);

    // cos/sin are already shaped [1, 1, seq_len, half] — broadcast over batch and n_head
    // Split-half rotation:
    //   new_first_half  = x1 * cos - x2 * sin
    //   new_second_half = x1 * sin + x2 * cos
    Tensor::cat(
        vec![
            x1.clone() * cos.clone() - x2.clone() * sin.clone(),
            x1 * sin + x2 * cos,
        ],
        3,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_apply_rope_at_position_zero() {
        // At position 0: angle = 0 * freq = 0, so cos=1 and sin=0.
        // Rotation matrix is identity — output must equal input.
        let device = Default::default();
        // x: [batch=1, heads=1, seq=1, head_dim=4]
        let x = Tensor::<B, 4>::from_floats([[[[1.0_f32, 2.0, 3.0, 4.0]]]], &device);
        // cos=[1,1], sin=[0,0] — shape [1, 1, seq=1, half=2]
        let cos = Tensor::<B, 4>::from_floats([[[[1.0_f32, 1.0]]]], &device);
        let sin = Tensor::<B, 4>::from_floats([[[[0.0_f32, 0.0]]]], &device);
        let result = apply_rope(x.clone(), cos, sin);
        let diff = (result - x).abs().sum().into_scalar();
        assert!(diff < 1e-6, "RoPE at pos=0 should be identity, diff={diff}");
    }

    #[test]
    fn test_apply_rope_shape() {
        // Output shape must equal input shape regardless of values.
        let device = Default::default();
        let (batch, n_head, seq_len, head_dim) = (2, 4, 8, 16);
        let half = head_dim / 2;
        let x = Tensor::<B, 4>::zeros([batch, n_head, seq_len, head_dim], &device);
        let cos = Tensor::<B, 4>::ones([1, 1, seq_len, half], &device);
        let sin = Tensor::<B, 4>::zeros([1, 1, seq_len, half], &device);
        let result = apply_rope(x, cos, sin);
        assert_eq!(result.dims(), [batch, n_head, seq_len, head_dim]);
    }

    fn create_test_config() -> GPTConfig {
        GPTConfig {
            vocab_size: 100,
            n_layer: 2,
            n_head: 2,
            n_embd: 32,
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
            diff_val < 1e-5,
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
}
