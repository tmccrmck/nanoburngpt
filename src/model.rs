use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Initializer, LayerNorm,
        LayerNormConfig, Linear, LinearConfig,
    },
    tensor::{backend::Backend, activation, Int, Tensor},
};
use rand::distr::{Distribution, weighted::WeightedIndex};

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
    attn_dropout: Dropout,
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
            attn_dropout: DropoutConfig::new(config.dropout).init(),
            resid_dropout: DropoutConfig::new(config.dropout).init(),
            n_head: config.n_head,
            n_embd: config.n_embd,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
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

        // Scaled dot product attention
        let scale = (head_dim as f64).powf(-0.5);
        let mut att = q.matmul(k.swap_dims(2, 3)) * scale;

        // Causal mask
        let mask = Tensor::<B, 2>::ones([seq_len, seq_len], &x.device())
            .tril(0)
            .reshape([1, 1, seq_len, seq_len]);
        
        att = att.mask_fill(mask.equal_elem(0), f32::NEG_INFINITY);
        att = activation::softmax(att, 3);
        att = self.attn_dropout.forward(att);

        let y = att.matmul(v);
        
        let y = y.permute([0, 2, 1, 3])
            .reshape([batch_size, seq_len, self.n_embd]);
            
        let y = self.c_proj.forward(y);
        self.resid_dropout.forward(y)
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

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.ln_1.forward(x.clone()));
        let x = x.clone() + self.mlp.forward(self.ln_2.forward(x));
        x
    }
}

#[derive(Module, Debug)]
pub struct GPT<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    blocks: Vec<Block<B>>,
    ln_f: LayerNorm<B>,
    // lm_head is weight-tied to token_embedding — no separate parameter
}

impl<B: Backend> GPT<B> {
    pub fn new(config: &GPTConfig, device: &B::Device) -> Self {
        let token_embedding = EmbeddingConfig::new(config.vocab_size, config.n_embd).init(device);
        let position_embedding = EmbeddingConfig::new(config.block_size, config.n_embd).init(device);

        let blocks = (0..config.n_layer)
            .map(|_| Block::new(config, device))
            .collect();

        let ln_f = LayerNormConfig::new(config.n_embd).init(device);

        Self {
            token_embedding,
            position_embedding,
            blocks,
            ln_f,
        }
    }

    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq_len] = idx.dims();
        let device = idx.device();

        let pos = Tensor::arange(0..seq_len as i64, &device).unsqueeze::<2>();
        
        let tok_emb = self.token_embedding.forward(idx);
        let pos_emb = self.position_embedding.forward(pos);
        
        let mut x = tok_emb + pos_emb;
        
        for block in &self.blocks {
            x = block.forward(x);
        }
        
        let x = self.ln_f.forward(x);

        // Weight-tied output projection: x @ token_embedding.weight.T
        let weight = self.token_embedding.weight.val(); // [vocab_size, n_embd]
        let [vocab_size, n_embd] = weight.dims();
        x.reshape([batch * seq_len, n_embd])
            .matmul(weight.transpose())
            .reshape([batch, seq_len, vocab_size])
    }

    pub fn generate(
        &self,
        idx: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        temperature: f64,
        block_size: usize,
    ) -> Tensor<B, 2, Int> {
        let mut idx = idx;
        let mut rng = rand::rng();

        for _ in 0..max_new_tokens {
            let [batch, seq_len] = idx.dims();

            let idx_cond = if seq_len > block_size {
                idx.clone().slice([0..batch, seq_len - block_size..seq_len])
            } else {
                idx.clone()
            };

            let logits = self.forward(idx_cond);
            let [_, len, vocab] = logits.dims();
            // Take logits of the last position: [batch, vocab]
            let logits = logits
                .slice([0..batch, len - 1..len, 0..vocab])
                .reshape([batch, vocab]);

            // Scale by temperature then sample (or greedy when temp ~= 0)
            let idx_next = if temperature < 1e-6 {
                logits.argmax(1).unsqueeze::<2>()
            } else {
                let logits = logits / temperature;
                let probs = activation::softmax(logits, 1);
                let probs_data = probs.into_data();
                let probs_f32 = probs_data.as_slice::<f32>().expect("f32 probs");

                // Sample one token per batch element
                let tokens: Vec<i32> = (0..batch)
                    .map(|b| {
                        let row = &probs_f32[b * vocab..(b + 1) * vocab];
                        let dist = WeightedIndex::new(row).expect("valid weights");
                        dist.sample(&mut rng) as i32
                    })
                    .collect();

                Tensor::<B, 1, Int>::from_ints(tokens.as_slice(), &idx.device())
                    .unsqueeze::<2>()
            };

            idx = Tensor::cat(vec![idx, idx_next], 1);
        }
        idx
    }
}
