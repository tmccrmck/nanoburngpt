use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::{
    module::Module,
    nn::{
        Dropout, DropoutConfig, Initializer, Linear, LinearConfig, RotaryEncoding,
        RotaryEncodingConfig,
    },
    prelude::*,
};
use super::GPTConfig;

/// Per-layer pre-allocated K and V cache buffers: `[batch, n_kv_head, max_len, head_dim]`.
/// Only positions `0..len` along dimension 2 contain populated entries.
#[derive(Clone, Debug)]
pub struct LayerKV {
    pub k: Tensor<4>,
    pub v: Tensor<4>,
    pub len: usize,
}

/// KV cache across all layers. `layers[i]` holds pre-allocated (K, V) buffers for block `i`.
#[derive(Clone, Debug)]
pub struct KVCache {
    pub layers: Vec<LayerKV>,
}

impl KVCache {
    /// Number of cached token positions currently populated.
    pub fn len(&self) -> usize {
        self.layers.first().map_or(0, |l| l.len)
    }

    /// Whether the cache has zero populated positions.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Module, Debug)]
pub struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    resid_dropout: Dropout,
    rope: RotaryEncoding,
    n_head: usize,
    n_kv_head: usize,
    n_embd: usize,
}

impl CausalSelfAttention {
    pub fn new(config: &GPTConfig, device: &Device) -> Self {
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
        }
    }

    #[inline]
    fn split_qkv(&self, x: Tensor<3>) -> (Tensor<4>, Tensor<4>, Tensor<4>, usize, usize, usize) {
        let [batch_size, seq_len, _] = x.dims();
        let head_dim = self.n_embd / self.n_head;
        let q_dim = self.n_head * head_dim;
        let kv_dim = self.n_kv_head * head_dim;

        let qkv = self.c_attn.forward(x);
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

        (q, k, v, batch_size, seq_len, head_dim)
    }

    #[inline]
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
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
            q,
            k,
            v,
            None,
            None,
            AttentionModuleOptions {
                is_causal: true,
                ..Default::default()
            },
        );

        let y = y
            .permute([0, 2, 1, 3])
            .reshape([batch_size, seq_len, self.n_embd]);
        let y = self.c_proj.forward(y);
        self.resid_dropout.forward(y)
    }

    #[inline]
    pub fn forward_cached(
        &self,
        x: Tensor<3>,
        cache: Option<LayerKV>,
        pos_offset: usize,
        max_len: usize,
    ) -> (Tensor<3>, LayerKV) {
        let (q, mut k, v, batch_size, seq_len, head_dim) = self.split_qkv(x);

        let q = self.rope.apply(q, pos_offset);
        k = self.rope.apply(k, pos_offset);

        let new_len = pos_offset + seq_len;
        let (mut k_buf, mut v_buf) = match cache {
            Some(layer_kv) if layer_kv.k.dims()[2] >= new_len => (layer_kv.k, layer_kv.v),
            Some(layer_kv) => {
                let cap = new_len.max(layer_kv.k.dims()[2] * 2);
                let pad = cap - layer_kv.k.dims()[2];
                let device = k.device();
                let zeros =
                    Tensor::<4>::zeros([batch_size, self.n_kv_head, pad, head_dim], &device);
                (
                    Tensor::cat(vec![layer_kv.k, zeros.clone()], 2),
                    Tensor::cat(vec![layer_kv.v, zeros], 2),
                )
            }
            None => {
                let cap = max_len.max(new_len);
                let device = k.device();
                (
                    Tensor::<4>::zeros([batch_size, self.n_kv_head, cap, head_dim], &device),
                    Tensor::<4>::zeros([batch_size, self.n_kv_head, cap, head_dim], &device),
                )
            }
        };

        // In-place update when uniquely owned (avoiding O(T^2) Tensor::cat per decode step).
        k_buf = k_buf.slice_assign(
            [
                0..batch_size,
                0..self.n_kv_head,
                pos_offset..new_len,
                0..head_dim,
            ],
            k,
        );
        v_buf = v_buf.slice_assign(
            [
                0..batch_size,
                0..self.n_kv_head,
                pos_offset..new_len,
                0..head_dim,
            ],
            v,
        );

        let k_valid = k_buf
            .clone()
            .slice([0..batch_size, 0..self.n_kv_head, 0..new_len, 0..head_dim]);
        let v_valid = v_buf
            .clone()
            .slice([0..batch_size, 0..self.n_kv_head, 0..new_len, 0..head_dim]);

        let (k_att, v_att) = if self.n_kv_head < self.n_head {
            let reps = self.n_head / self.n_kv_head;
            (k_valid.repeat_dim(1, reps), v_valid.repeat_dim(1, reps))
        } else {
            (k_valid, v_valid)
        };

        let y = attention(
            q,
            k_att,
            v_att,
            None,
            None,
            AttentionModuleOptions {
                // Single-token decode attends to all cached positions; skip causal mask kernels.
                is_causal: seq_len > 1,
                ..Default::default()
            },
        );

        let y = y
            .permute([0, 2, 1, 3])
            .reshape([batch_size, seq_len, self.n_embd]);
        let y = self.c_proj.forward(y);
        let new_cache = LayerKV {
            k: k_buf,
            v: v_buf,
            len: new_len,
        };
        (self.resid_dropout.forward(y), new_cache)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    fn test_device() -> Device {
        Device::flex()
    }

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
        }
    }

    #[test]
    fn forward_returns_correct_shape() {
        let device = test_device();
        let config = test_config();
        let attn = CausalSelfAttention::new(&config, &device);

        let batch = 2;
        let seq = 8;
        let x = Tensor::<3>::random(
            [batch, seq, config.n_embd],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let y = attn.forward(x);

        assert_eq!(y.dims(), [batch, seq, config.n_embd]);
    }

    #[test]
    fn forward_gqa_returns_correct_shape() {
        let device = test_device();
        let config = gqa_config();
        let attn = CausalSelfAttention::new(&config, &device);

        let batch = 2;
        let seq = 8;
        let x = Tensor::<3>::random(
            [batch, seq, config.n_embd],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let y = attn.forward(x);

        assert_eq!(y.dims(), [batch, seq, config.n_embd]);
    }

    #[test]
    fn forward_cached_equals_forward() {
        let device = test_device();
        let config = test_config();
        let attn = CausalSelfAttention::new(&config, &device);

        let batch = 1;
        let seq = 8;
        let x = Tensor::<3>::random(
            [batch, seq, config.n_embd],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let expected = attn.forward(x.clone());
        let expected_last = expected.slice([0..1, seq - 1..seq, 0..config.n_embd]);

        let prefix = x.clone().slice([0..1, 0..seq - 1, 0..config.n_embd]);
        let (_, cache) = attn.forward_cached(prefix, None, 0, config.block_size);

        let last = x.clone().slice([0..1, seq - 1..seq, 0..config.n_embd]);
        let (cached, _) = attn.forward_cached(last, Some(cache), seq - 1, config.block_size);

        let diff = (expected_last - cached).abs().sum().into_scalar::<f32>();
        assert!(
            diff < 1e-4,
            "cached output differs from full forward: {diff}"
        );
    }

    #[test]
    fn forward_cached_gqa_equals_forward() {
        let device = test_device();
        let config = gqa_config();
        let attn = CausalSelfAttention::new(&config, &device);

        let batch = 1;
        let seq = 8;
        let x = Tensor::<3>::random(
            [batch, seq, config.n_embd],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        let expected = attn.forward(x.clone());
        let expected_last = expected.slice([0..1, seq - 1..seq, 0..config.n_embd]);

        let prefix = x.clone().slice([0..1, 0..seq - 1, 0..config.n_embd]);
        let (_, cache) = attn.forward_cached(prefix, None, 0, config.block_size);

        let last = x.clone().slice([0..1, seq - 1..seq, 0..config.n_embd]);
        let (cached, _) = attn.forward_cached(last, Some(cache), seq - 1, config.block_size);

        let diff = (expected_last - cached).abs().sum().into_scalar::<f32>();
        assert!(
            diff < 1e-4,
            "GQA cached output differs from full forward: {diff}"
        );
    }

    #[test]
    fn cache_grows_on_successive_calls() {
        let device = test_device();
        let config = test_config();
        let attn = CausalSelfAttention::new(&config, &device);

        let batch = 1;
        let n_embd = config.n_embd;
        let head_dim = n_embd / config.n_head;

        let t0 = Tensor::<3>::random(
            [batch, 1, n_embd],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (_, cache) = attn.forward_cached(t0, None, 0, config.block_size);
        assert_eq!(cache.len, 1, "cache len should be 1 after first call");
        assert_eq!(
            cache.k.dims(),
            [1, config.n_head, config.block_size, head_dim],
            "cache K should be pre-allocated to block_size"
        );

        let t1 = Tensor::<3>::random(
            [batch, 1, n_embd],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (_, cache) = attn.forward_cached(t1, Some(cache), 1, config.block_size);
        assert_eq!(cache.len, 2, "cache len should be 2 after second call");
        assert_eq!(
            cache.k.dims(),
            [1, config.n_head, config.block_size, head_dim],
            "cache K capacity should remain block_size"
        );
    }

    #[test]
    fn cache_grows_gqa_repeats_kv_heads() {
        let device = test_device();
        let config = gqa_config();
        let attn = CausalSelfAttention::new(&config, &device);

        let batch = 1;
        let n_embd = config.n_embd;
        let head_dim = n_embd / config.n_head;

        let t0 = Tensor::<3>::random(
            [batch, 1, n_embd],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (_, cache) = attn.forward_cached(t0, None, 0, config.block_size);
        assert_eq!(cache.len, 1);
        assert_eq!(
            cache.k.dims(),
            [1, 2, config.block_size, head_dim],
            "GQA cache should store n_kv_head=2 heads"
        );

        let t1 = Tensor::<3>::random(
            [batch, 1, n_embd],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (_, cache) = attn.forward_cached(t1, Some(cache), 1, config.block_size);
        assert_eq!(cache.len, 2, "GQA cache should have len=2 after two calls");
        assert_eq!(cache.k.dims(), [1, 2, config.block_size, head_dim]);
    }
}
