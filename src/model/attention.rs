use burn::tensor::module::attention;
use burn::tensor::backend::ops::AttentionModuleOptions;
use burn::{
    module::Module,
    nn::{
        Dropout, DropoutConfig, Initializer, Linear, LinearConfig,
        RotaryEncoding, RotaryEncodingConfig,
    },
    tensor::{backend::Backend, Tensor},
};
use super::GPTConfig;

/// Per-layer cached K and V tensors: [batch, n_head, cached_seq, head_dim].
pub type LayerKV<B> = (Tensor<B, 4>, Tensor<B, 4>);

/// KV cache across all layers. `layers[i]` holds (K, V) for block `i`.
#[derive(Clone, Debug)]
pub struct KVCache<B: Backend> {
    pub layers: Vec<LayerKV<B>>,
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
    softcap: Option<f64>,
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
            softcap: config.softcap,
        }
    }

    fn split_qkv(&self, x: Tensor<B, 3>) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>, usize, usize, usize) {
        let [batch_size, seq_len, _] = x.dims();
        let head_dim = self.n_embd / self.n_head;
        let q_dim = self.n_head * head_dim;
        let kv_dim = self.n_kv_head * head_dim;

        let qkv = self.c_attn.forward(x);
        let q = qkv.clone()
            .slice([0..batch_size, 0..seq_len, 0..q_dim])
            .reshape([batch_size, seq_len, self.n_head, head_dim])
            .permute([0, 2, 1, 3]);
        let k = qkv.clone()
            .slice([0..batch_size, 0..seq_len, q_dim..(q_dim + kv_dim)])
            .reshape([batch_size, seq_len, self.n_kv_head, head_dim])
            .permute([0, 2, 1, 3]);
        let v = qkv
            .slice([0..batch_size, 0..seq_len, (q_dim + kv_dim)..(q_dim + 2 * kv_dim)])
            .reshape([batch_size, seq_len, self.n_kv_head, head_dim])
            .permute([0, 2, 1, 3]);

        (q, k, v, batch_size, seq_len, head_dim)
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let (q, k, v, batch_size, seq_len, _) = self.split_qkv(x);

        let q = self.rope.forward(q);
        let k = self.rope.forward(k);

        let (k, v) = if self.n_kv_head < self.n_head {
            let reps = self.n_head / self.n_kv_head;
            (k.repeat_dim(1, reps), v.repeat_dim(1, reps))
        } else {
            (k, v)
        };

        let y = attention(
            q, k, v, None, None,
            AttentionModuleOptions {
                softcap: self.softcap,
                is_causal: true,
                ..Default::default()
            },
        );

        let y = y.permute([0, 2, 1, 3]).reshape([batch_size, seq_len, self.n_embd]);
        let y = self.c_proj.forward(y);
        self.resid_dropout.forward(y)
    }

    pub fn forward_cached(
        &self,
        x: Tensor<B, 3>,
        cache: Option<LayerKV<B>>,
        pos_offset: usize,
    ) -> (Tensor<B, 3>, LayerKV<B>) {
        let (q, mut k, mut v, batch_size, seq_len, _) = self.split_qkv(x);

        let q = self.rope.apply(q, pos_offset);
        k = self.rope.apply(k, pos_offset);

        if let Some((cached_k, cached_v)) = cache {
            k = Tensor::cat(vec![cached_k, k], 2);
            v = Tensor::cat(vec![cached_v, v], 2);
        }

        let new_cache = (k.clone(), v.clone());

        let (k_att, v_att) = if self.n_kv_head < self.n_head {
            let reps = self.n_head / self.n_kv_head;
            (k.repeat_dim(1, reps), v.repeat_dim(1, reps))
        } else {
            (k, v)
        };

        let y = attention(
            q, k_att, v_att, None, None,
            AttentionModuleOptions {
                softcap: self.softcap,
                is_causal: true,
                ..Default::default()
            },
        );

        let y = y.permute([0, 2, 1, 3]).reshape([batch_size, seq_len, self.n_embd]);
        let y = self.c_proj.forward(y);
        (self.resid_dropout.forward(y), new_cache)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_flex::Flex;

    type B = Flex;

    fn test_config() -> GPTConfig {
        GPTConfig {
            vocab_size: 100,
            n_layer: 2,
            n_head: 4,
            n_embd: 32,
            n_kv_head: None,
            block_size: 16,
            dropout: 0.0,
            rope_theta: 10000.0,
            softcap: None,
        }
    }

    fn gqa_config() -> GPTConfig {
        GPTConfig {
            vocab_size: 100,
            n_layer: 2,
            n_head: 4,
            n_kv_head: Some(2),
            n_embd: 32,
            block_size: 16,
            dropout: 0.0,
            rope_theta: 10000.0,
            softcap: None,
        }
    }

    #[test]
    fn forward_returns_correct_shape() {
        let device = Default::default();
        let config = test_config();
        let attn = CausalSelfAttention::<B>::new(&config, &device);

        let batch = 2;
        let seq = 8;
        let x = Tensor::<B, 3>::random([batch, seq, config.n_embd], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let y = attn.forward(x);

        assert_eq!(y.dims(), [batch, seq, config.n_embd]);
    }

    #[test]
    fn forward_gqa_returns_correct_shape() {
        let device = Default::default();
        let config = gqa_config();
        let attn = CausalSelfAttention::<B>::new(&config, &device);

        let batch = 2;
        let seq = 8;
        let x = Tensor::<B, 3>::random([batch, seq, config.n_embd], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let y = attn.forward(x);

        assert_eq!(y.dims(), [batch, seq, config.n_embd]);
    }

    #[test]
    fn forward_cached_equals_forward() {
        let device = Default::default();
        let config = test_config();
        let attn = CausalSelfAttention::<B>::new(&config, &device);

        let batch = 1;
        let seq = 8;
        let x = Tensor::<B, 3>::random([batch, seq, config.n_embd], burn::tensor::Distribution::Normal(0.0, 1.0), &device);

        let expected = attn.forward(x.clone());
        let expected_last = expected.slice([0..1, seq - 1..seq, 0..config.n_embd]);

        let prefix = x.clone().slice([0..1, 0..seq - 1, 0..config.n_embd]);
        let (_, cache) = attn.forward_cached(prefix, None, 0);

        let last = x.clone().slice([0..1, seq - 1..seq, 0..config.n_embd]);
        let (cached, _) = attn.forward_cached(last, Some(cache), seq - 1);

        let diff = (expected_last - cached).abs().sum().into_scalar();
        assert!(diff < 1e-4, "cached output differs from full forward: {diff}");
    }

    #[test]
    fn forward_cached_gqa_equals_forward() {
        let device = Default::default();
        let config = gqa_config();
        let attn = CausalSelfAttention::<B>::new(&config, &device);

        let batch = 1;
        let seq = 8;
        let x = Tensor::<B, 3>::random([batch, seq, config.n_embd], burn::tensor::Distribution::Normal(0.0, 1.0), &device);

        let expected = attn.forward(x.clone());
        let expected_last = expected.slice([0..1, seq - 1..seq, 0..config.n_embd]);

        let prefix = x.clone().slice([0..1, 0..seq - 1, 0..config.n_embd]);
        let (_, cache) = attn.forward_cached(prefix, None, 0);

        let last = x.clone().slice([0..1, seq - 1..seq, 0..config.n_embd]);
        let (cached, _) = attn.forward_cached(last, Some(cache), seq - 1);

        let diff = (expected_last - cached).abs().sum().into_scalar();
        assert!(diff < 1e-4, "GQA cached output differs from full forward: {diff}");
    }

    #[test]
    fn cache_grows_on_successive_calls() {
        let device = Default::default();
        let config = test_config();
        let attn = CausalSelfAttention::<B>::new(&config, &device);

        let batch = 1;
        let n_embd = config.n_embd;
        let head_dim = n_embd / config.n_head;

        let t0 = Tensor::<B, 3>::random([batch, 1, n_embd], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let (_, cache) = attn.forward_cached(t0, None, 0);
        assert_eq!(cache.0.dims(), [1, config.n_head, 1, head_dim], "cache K should have seq_len=1 after first call");

        let t1 = Tensor::<B, 3>::random([batch, 1, n_embd], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let (_, cache) = attn.forward_cached(t1, Some(cache), 1);
        assert_eq!(cache.0.dims(), [1, config.n_head, 2, head_dim], "cache K should have seq_len=2 after second call");
    }

    #[test]
    fn cache_grows_gqa_repeats_kv_heads() {
        let device = Default::default();
        let config = gqa_config();
        let attn = CausalSelfAttention::<B>::new(&config, &device);

        let batch = 1;
        let n_embd = config.n_embd;
        let head_dim = n_embd / config.n_head;

        let t0 = Tensor::<B, 3>::random([batch, 1, n_embd], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let (_, cache) = attn.forward_cached(t0, None, 0);
        let (cached_k, _) = cache;
        assert_eq!(cached_k.dims(), [1, 2, 1, head_dim], "GQA cache should store n_kv_head=2 heads");

        let t1 = Tensor::<B, 3>::random([batch, 1, n_embd], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let (_, cache) = attn.forward_cached(t1, Some((cached_k, Tensor::<B, 4>::zeros([1, 2, 1, head_dim], &device))), 1);
        assert_eq!(cache.0.dims(), [1, 2, 2, head_dim], "GQA cache should have seq_len=2 after two calls");
    }

    #[test]
    fn softcap_does_not_crash() {
        let device = Default::default();
        let mut config = test_config();
        config.softcap = Some(50.0);
        let attn = CausalSelfAttention::<B>::new(&config, &device);

        let batch = 2;
        let seq = 8;
        let x = Tensor::<B, 3>::random([batch, seq, config.n_embd], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let y = attn.forward(x);

        assert_eq!(y.dims(), [batch, seq, config.n_embd]);
    }
}
