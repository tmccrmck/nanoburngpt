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
    tensor::activation,
};

pub use self::attention::{CausalSelfAttention, KVCache, LayerKV};
use self::sampling::sample_token;
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
        let x = activation::relu(x).square();
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
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = x.clone() + self.attn.forward(self.ln_1.forward(x.clone()));
        x.clone() + self.mlp.forward(self.ln_2.forward(x))
    }

    #[inline]
    pub fn forward_cached(
        &self,
        x: Tensor<3>,
        cache: Option<LayerKV>,
        pos_offset: usize,
    ) -> (Tensor<3>, LayerKV) {
        let (attn_out, new_cache) =
            self.attn
                .forward_cached(self.ln_1.forward(x.clone()), cache, pos_offset);
        let x = x + attn_out;
        let x = x.clone() + self.mlp.forward(self.ln_2.forward(x));
        (x, new_cache)
    }
}

#[derive(Module, Debug)]
pub struct GPT {
    token_embedding: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
}

impl GPT {
    pub fn new(config: &GPTConfig, device: &Device) -> Self {
        let token_embedding = EmbeddingConfig::new(config.vocab_size, config.n_embd)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            })
            .init(device);

        let blocks = (0..config.n_layer)
            .map(|_| Block::new(config, device))
            .collect();

        let ln_f = RmsNormConfig::new(config.n_embd).init(device);

        Self {
            token_embedding,
            blocks,
            ln_f,
        }
    }

    #[inline]
    pub fn forward(&self, idx: Tensor<2, Int>) -> Tensor<3> {
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
    #[inline]
    pub fn forward_cached(
        &self,
        idx: Tensor<2, Int>,
        cache: Option<KVCache>,
    ) -> (Tensor<3>, KVCache) {
        let [batch, seq_len] = idx.dims();

        let pos_offset = cache.as_ref().map_or(0, |c| c.layers[0].0.dims()[2]);

        let mut x = self.token_embedding.forward(idx);

        let mut new_layers = Vec::with_capacity(self.blocks.len());
        for (i, block) in self.blocks.iter().enumerate() {
            let layer_cache = cache.as_ref().map(|c| c.layers[i].clone());
            let (out, layer_kv) = block.forward_cached(x, layer_cache, pos_offset);
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

        (
            logits,
            KVCache {
                layers: new_layers,
            },
        )
    }

    /// Auto-regressive generation with KV cache.
    pub fn generate(
        &self,
        idx: Tensor<2, Int>,
        max_new_tokens: usize,
        sampling: &SamplingParams,
        block_size: usize,
        mut on_token: impl FnMut(i32),
    ) -> Tensor<2, Int> {
        let [batch, orig_prompt_len] = idx.dims();
        let idx = if orig_prompt_len > block_size {
            idx.slice([0..batch, (orig_prompt_len - block_size)..orig_prompt_len])
        } else {
            idx
        };
        let [_, prompt_len] = idx.dims();
        let steps = max_new_tokens.min(block_size.saturating_sub(prompt_len));
        if steps == 0 {
            return idx;
        }

        let (mut next_logits, mut cache) = self.forward_cached(idx.clone(), None);
        let mut prev_tokens = idx.clone();
        let mut generated: Vec<Tensor<2, Int>> = Vec::with_capacity(steps);

        for step in 0..steps {
            let [_, len, vocab] = next_logits.dims();
            let logits = next_logits
                .slice([0..batch, len - 1..len, 0..vocab])
                .reshape([batch, vocab]);

            let idx_next = sample_token(&logits, sampling, &prev_tokens);

            let token_id: i32 = idx_next.clone().slice([0..1, 0..1]).into_scalar();
            on_token(token_id);

            generated.push(idx_next.clone());
            if step + 1 == steps {
                break;
            }
            prev_tokens = Tensor::cat(vec![prev_tokens, idx_next.clone()], 1);
            let (step_logits, new_cache) = self.forward_cached(idx_next, Some(cache));
            next_logits = step_logits;
            cache = new_cache;
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
        };

        let output = gpt.generate(prompt, max_new, &sampling, config.block_size, |_| {});
        let [_, total_len] = output.dims();
        assert_eq!(total_len, prompt_len + max_new);
    }

    #[test]
    fn generate_respects_block_size_limit() {
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
        };

        let output = gpt.generate(prompt, max_new, &sampling, config.block_size, |_| {});
        let [_, total_len] = output.dims();
        assert!(
            total_len <= config.block_size,
            "output len {total_len} exceeds block_size {}",
            config.block_size
        );
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

        let (cached_k, _) = &cache.layers[0];
        let [_, cached_heads, _, _] = cached_k.dims();
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
    fn generate_callback_matches_generated_count_at_block_size() {
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
        };

        let mut callback_count = 0usize;
        let output = gpt.generate(prompt, max_new, &sampling, config.block_size, |_| {
            callback_count += 1;
        });
        assert_eq!(output.dims(), [1, config.block_size]);
        assert_eq!(callback_count, config.block_size - prompt_len);
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
        };

        let output = gpt.generate(prompt, 5, &sampling, config.block_size, |_| {});
        assert_eq!(output.dims(), [1, config.block_size]);
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
        };

        let output = gpt.generate(prompt, 4, &sampling, config.block_size, |_| {});
        assert_eq!(output.dims(), [2, 7]);
    }
}
