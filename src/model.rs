use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Initializer, LayerNorm,
        LayerNormConfig, Linear, LinearConfig,
    },
    tensor::{backend::Backend, activation, Bool, Int, Tensor},
};
use burn::tensor::module::attention;
use rand::distr::{Distribution, weighted::WeightedIndex};

// ---------------------------------------------------------------------------
// KV Cache for inference
// ---------------------------------------------------------------------------

/// Sample next token from logits [batch, vocab], returns [batch, 1] int tensor.
fn sample_token<B: Backend>(
    logits: &Tensor<B, 2>,
    temperature: f64,
    batch: usize,
    vocab: usize,
    device: &B::Device,
    rng: &mut impl rand::Rng,
) -> Tensor<B, 2, Int> {
    if temperature < 1e-6 {
        logits.clone().argmax(1).unsqueeze::<2>()
    } else {
        let logits = logits.clone() / temperature;
        let probs = activation::softmax(logits, 1);
        let probs_data = probs.into_data();
        let probs_f32 = probs_data.as_slice::<f32>().expect("f32 probs");

        let tokens: Vec<i32> = (0..batch)
            .map(|b| {
                let row = &probs_f32[b * vocab..(b + 1) * vocab];
                let dist = WeightedIndex::new(row).expect("valid weights");
                dist.sample(rng) as i32
            })
            .collect();

        Tensor::<B, 1, Int>::from_ints(tokens.as_slice(), device)
            .unsqueeze::<2>()
    }
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
                .with_initializer(Initializer::Normal { mean: 0.0, std: 0.02 })
                .init(device),
            c_proj: LinearConfig::new(config.n_embd, config.n_embd)
                .with_initializer(Initializer::Normal { mean: 0.0, std: proj_std })
                .init(device),
            resid_dropout: DropoutConfig::new(config.dropout).init(),
            n_head: config.n_head,
            n_embd: config.n_embd,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4, Bool>>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = x.dims();
        let head_dim = self.n_embd / self.n_head;

        // Q, K, V
        let qkv = self.c_attn.forward(x.clone());
        let qkv = qkv.reshape([batch_size, seq_len, 3, self.n_head, head_dim]);

        let qkv = qkv.permute([2, 0, 3, 1, 4]);
        // Use reshape instead of squeeze to avoid ambiguity when batch or seq_len == 1
        let q = qkv.clone().slice([0..1]).reshape([batch_size, self.n_head, seq_len, head_dim]);
        let k = qkv.clone().slice([1..2]).reshape([batch_size, self.n_head, seq_len, head_dim]);
        let v = qkv.clone().slice([2..3]).reshape([batch_size, self.n_head, seq_len, head_dim]);

        // Burn's fused SDPA kernel: computes softmax(QK^T / sqrt(d_k)) * V.
        // May use an optimized kernel on CUDA; standard implementation on wgpu.
        // Does not support attention dropout (resid_dropout still applied after).
        let y = attention(q, k, v, mask);

        let y = y.permute([0, 2, 1, 3])
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
    ) -> (Tensor<B, 3>, LayerKV<B>) {
        let [batch_size, seq_len, _] = x.dims();
        let head_dim = self.n_embd / self.n_head;

        let qkv = self.c_attn.forward(x.clone());
        let qkv = qkv.reshape([batch_size, seq_len, 3, self.n_head, head_dim]);
        let qkv = qkv.permute([2, 0, 3, 1, 4]);

        let q = qkv.clone().slice([0..1]).reshape([batch_size, self.n_head, seq_len, head_dim]);
        let mut k = qkv.clone().slice([1..2]).reshape([batch_size, self.n_head, seq_len, head_dim]);
        let mut v = qkv.clone().slice([2..3]).reshape([batch_size, self.n_head, seq_len, head_dim]);

        // Concatenate with cached K, V from previous steps
        if let Some((cached_k, cached_v)) = cache {
            k = Tensor::cat(vec![cached_k, k], 2);
            v = Tensor::cat(vec![cached_v, v], 2);
        }

        let new_cache = (k.clone(), v.clone());

        let y = attention(q, k, v, mask);

        let y = y.permute([0, 2, 1, 3])
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
                .with_initializer(Initializer::Normal { mean: 0.0, std: 0.02 })
                .init(device),
            c_proj: LinearConfig::new(4 * config.n_embd, config.n_embd)
                .with_initializer(Initializer::Normal { mean: 0.0, std: proj_std })
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

    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<Tensor<B, 4, Bool>>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.ln_1.forward(x.clone()), mask);
        let x = x.clone() + self.mlp.forward(self.ln_2.forward(x));
        x
    }

    pub fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        cache: Option<LayerKV<B>>,
        mask: Option<Tensor<B, 4, Bool>>,
    ) -> (Tensor<B, 3>, LayerKV<B>) {
        let (attn_out, new_cache) = self.attn.forward_cached(self.ln_1.forward(x.clone()), cache, mask);
        let x = x + attn_out;
        let x = x.clone() + self.mlp.forward(self.ln_2.forward(x));
        (x, new_cache)
    }
}

#[derive(Module, Debug)]
pub struct GPT<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    blocks: Vec<Block<B>>,
    ln_f: LayerNorm<B>,
    // lm_head is weight-tied to token_embedding — no separate parameter
    // Full causal mask [1, 1, block_size, block_size] — created once, sliced in forward.
    // Stored as a raw Tensor (not Param) so it's not a learnable parameter.
    causal_mask: Tensor<B, 4, Bool>,
}

impl<B: Backend> GPT<B> {
    pub fn new(config: &GPTConfig, device: &B::Device) -> Self {
        let token_embedding = EmbeddingConfig::new(config.vocab_size, config.n_embd).init(device);
        let position_embedding = EmbeddingConfig::new(config.block_size, config.n_embd).init(device);

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
            position_embedding,
            blocks,
            ln_f,
            causal_mask,
        }
    }

    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq_len] = idx.dims();
        let device = idx.device();

        let pos = Tensor::arange(0..seq_len as i64, &device).unsqueeze::<2>();

        let tok_emb = self.token_embedding.forward(idx);
        let pos_emb = self.position_embedding.forward(pos);

        let mut x = tok_emb + pos_emb;

        // Slice the pre-computed causal mask to the current sequence length
        let mask = self.causal_mask.clone().slice([0..1, 0..1, 0..seq_len, 0..seq_len]);

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
        let device = idx.device();

        // Position offset: when we have a cache, new tokens start at cached_len
        let pos_offset = cache.as_ref().map_or(0, |c| c.layers[0].0.dims()[2]);
        let pos = Tensor::arange(pos_offset as i64..(pos_offset + seq_len) as i64, &device)
            .unsqueeze::<2>();

        let tok_emb = self.token_embedding.forward(idx);
        let pos_emb = self.position_embedding.forward(pos);
        let mut x = tok_emb + pos_emb;

        // Causal mask only needed during prefill (multi-token); single-token decode
        // needs no mask since it attends to all cached positions + itself.
        let mask = if cache.is_none() && seq_len > 1 {
            Some(self.causal_mask.clone().slice([0..1, 0..1, 0..seq_len, 0..seq_len]))
        } else {
            None
        };

        let mut new_layers = Vec::with_capacity(self.blocks.len());
        for (i, block) in self.blocks.iter().enumerate() {
            let layer_cache = cache.as_ref().map(|c| c.layers[i].clone());
            let (out, layer_kv) = block.forward_cached(x, layer_cache, mask.clone());
            x = out;
            new_layers.push(layer_kv);
        }

        let x = self.ln_f.forward(x);

        let weight = self.token_embedding.weight.val();
        let [vocab_size, n_embd] = weight.dims();
        let logits = x.reshape([batch * seq_len, n_embd])
            .matmul(weight.transpose())
            .reshape([batch, seq_len, vocab_size]);

        (logits, KVCache { layers: new_layers })
    }

    pub fn generate(
        &self,
        idx: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        temperature: f64,
        block_size: usize,
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
            // Take logits of the last position: [batch, vocab]
            let logits = next_logits
                .slice([0..batch, len - 1..len, 0..vocab])
                .reshape([batch, vocab]);

            let idx_next = sample_token::<B>(&logits, temperature, batch, vocab, &device, &mut rng);
            all_tokens = Tensor::cat(vec![all_tokens, idx_next.clone()], 1);

            // Check if we've hit the context window limit
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
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn create_test_config() -> GPTConfig {
        GPTConfig {
            vocab_size: 100,
            n_layer: 2,
            n_head: 2,
            n_embd: 32,
            block_size: 16,
            dropout: 0.0,
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
        let expected_logits = full_logits.slice([0..1, prompt_len-1..prompt_len]);

        // 2. Incremental KV cache pass
        // Process first 4 tokens to warm up cache
        let context = prompt.clone().slice([0..1, 0..prompt_len-1]); // [0, 1, 2, 3]
        let (_, mut cache) = gpt.forward_cached(context, None);

        // Process 5th token (index 4) using cache
        let last_token = prompt.clone().slice([0..1, prompt_len-1..prompt_len]); // [4]
        let (cached_logits, _) = gpt.forward_cached(last_token, Some(cache));

        // Compare logits
        let diff = (expected_logits - cached_logits).abs().sum();
        let diff_val = diff.into_scalar();
        
        assert!(diff_val < 1e-5, "KV cache logits differ from full forward pass: {}", diff_val);
    }
}
