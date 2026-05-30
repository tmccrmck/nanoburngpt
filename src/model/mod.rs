pub mod attention;
pub mod sampling;

use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Initializer, LayerNorm,
        LayerNormConfig, Linear, LinearConfig,
    },
    tensor::{activation, backend::Backend, Int, Tensor},
};
use burn::prelude::ToElement;

pub use self::attention::{CausalSelfAttention, KVCache, LayerKV};
use self::sampling::sample_token;
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
    /// Logit softcap applied before softmax: softcap * tanh(scores / softcap).
    /// Used by Gemma-2 and similar models. None = disabled.
    pub softcap: Option<f64>,
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
        let x = activation::relu(x).square();
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

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.ln_1.forward(x.clone()));
        x.clone() + self.mlp.forward(self.ln_2.forward(x))
    }

    pub fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        cache: Option<LayerKV<B>>,
        pos_offset: usize,
    ) -> (Tensor<B, 3>, LayerKV<B>) {
        let (attn_out, new_cache) =
            self.attn
                .forward_cached(self.ln_1.forward(x.clone()), cache, pos_offset);
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
}

impl<B: Backend> GPT<B> {
    pub fn new(config: &GPTConfig, device: &B::Device) -> Self {
        let token_embedding = EmbeddingConfig::new(config.vocab_size, config.n_embd).init(device);

        let blocks = (0..config.n_layer)
            .map(|_| Block::new(config, device))
            .collect();

        let ln_f = LayerNormConfig::new(config.n_embd).init(device);

        Self {
            token_embedding,
            blocks,
            ln_f,
        }
    }

    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq_len] = idx.dims();

        let mut x = self.token_embedding.forward(idx);

        for block in &self.blocks {
            x = block.forward(x);
        }

        let x = self.ln_f.forward(x);

        let weight = self.token_embedding.weight.val();
        let [vocab_size, n_embd] = weight.dims();
        x.reshape([batch * seq_len, n_embd])
            .matmul(weight.transpose())
            .reshape([batch, seq_len, vocab_size])
    }

    /// Forward pass that builds/extends a KV cache for incremental decoding.
    pub fn forward_cached(
        &self,
        idx: Tensor<B, 2, Int>,
        cache: Option<KVCache<B>>,
    ) -> (Tensor<B, 3>, KVCache<B>) {
        let [batch, seq_len] = idx.dims();

        let pos_offset = cache.as_ref().map_or(0, |c| c.layers[0].0.dims()[2]);

        let mut x = self.token_embedding.forward(idx);

        let mut new_layers = Vec::with_capacity(self.blocks.len());
        for (i, block) in self.blocks.iter().enumerate() {
            let layer_cache = cache.as_ref().map(|c| c.layers[i].clone());
            let (out, layer_kv) =
                block.forward_cached(x, layer_cache, pos_offset);
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
    pub fn generate(
        &self,
        idx: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        sampling: &SamplingParams,
        block_size: usize,
        mut on_token: impl FnMut(i32),
    ) -> Tensor<B, 2, Int> {
        let [batch, _] = idx.dims();

        let (logits, mut cache) = self.forward_cached(idx.clone(), None);
        let mut all_tokens = idx;
        let mut next_logits = logits;

        for _ in 0..max_new_tokens {
            let [_, len, vocab] = next_logits.dims();
            let logits = next_logits
                .slice([0..batch, len - 1..len, 0..vocab])
                .reshape([batch, vocab]);

            let idx_next = sample_token::<B>(&logits, sampling);

            let token_id = idx_next.clone().into_scalar().to_i32();
            on_token(token_id);

            all_tokens = Tensor::cat(vec![all_tokens, idx_next.clone()], 1);

            let [_, total_len] = all_tokens.dims();
            if total_len >= block_size {
                break;
            }

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
            softcap: None,
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
        let prompt = Tensor::<B, 1, Int>::arange(0..prompt_len as i64, &device)
            .reshape([batch_size, prompt_len]);

        let full_logits = gpt.forward(prompt.clone());
        let expected_logits = full_logits.slice([0..1, prompt_len - 1..prompt_len]);

        let context = prompt.clone().slice([0..1, 0..prompt_len - 1]);
        let (_, cache) = gpt.forward_cached(context, None);

        let last_token = prompt.clone().slice([0..1, prompt_len - 1..prompt_len]);
        let (cached_logits, _) = gpt.forward_cached(last_token, Some(cache));

        let diff = (expected_logits - cached_logits).abs().sum();
        let diff_val = diff.into_scalar();

        assert!(
            diff_val < 1e-4,
            "KV cache logits differ from full forward pass: {}",
            diff_val
        );
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
        let config = create_test_config();
        let gpt = GPT::<B>::new(&config, &device);

        let prompt_len = 14;
        let max_new = 10;
        let prompt = Tensor::<B, 2, Int>::zeros([1, prompt_len], &device);
        let sampling = SamplingParams { temperature: 0.0, top_k: 0, top_p: 1.0 };

        let output = gpt.generate(prompt, max_new, &sampling, config.block_size, |_| {});
        let [_, total_len] = output.dims();
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
            softcap: None,
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
            softcap: None,
        };
        let gpt = GPT::<B>::new(&config, &device);

        let batch_size = 1;
        let prompt_len = 5;
        let prompt = Tensor::<B, 1, Int>::arange(0..prompt_len as i64, &device)
            .reshape([batch_size, prompt_len]);

        let full_logits = gpt.forward(prompt.clone());
        let expected_logits = full_logits.slice([0..1, prompt_len - 1..prompt_len]);

        let context = prompt.clone().slice([0..1, 0..prompt_len - 1]);
        let (_, cache) = gpt.forward_cached(context, None);

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
