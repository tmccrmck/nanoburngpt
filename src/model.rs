use burn::{
    config::Config,
    module::Module,
    nn::{
        Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, Dropout, DropoutConfig,
    },
    tensor::{backend::Backend, activation, Int, Tensor},
};

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
        let linear_config = LinearConfig::new(config.n_embd, 3 * config.n_embd);
        let proj_config = LinearConfig::new(config.n_embd, config.n_embd);

        Self {
            c_attn: linear_config.init(device),
            c_proj: proj_config.init(device),
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
        let q = qkv.clone().slice([0..1]).squeeze::<4>();
        let k = qkv.clone().slice([1..2]).squeeze::<4>();
        let v = qkv.clone().slice([2..3]).squeeze::<4>();

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
        Self {
            c_fc: LinearConfig::new(config.n_embd, 4 * config.n_embd).init(device),
            c_proj: LinearConfig::new(4 * config.n_embd, config.n_embd).init(device),
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
    pub lm_head: Linear<B>,
}

impl<B: Backend> GPT<B> {
    pub fn new(config: &GPTConfig, device: &B::Device) -> Self {
        let token_embedding = EmbeddingConfig::new(config.vocab_size, config.n_embd).init(device);
        let position_embedding = EmbeddingConfig::new(config.block_size, config.n_embd).init(device);
        
        let blocks = (0..config.n_layer)
            .map(|_| Block::new(config, device))
            .collect();
            
        let ln_f = LayerNormConfig::new(config.n_embd).init(device);
        let lm_head = LinearConfig::new(config.n_embd, config.vocab_size)
            .with_bias(false)
            .init(device);

        Self {
            token_embedding,
            position_embedding,
            blocks,
            ln_f,
            lm_head,
        }
    }

    pub fn forward(&self, idx: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_, seq_len] = idx.dims();
        let device = idx.device();

        let pos = Tensor::arange(0..seq_len as i64, &device).unsqueeze::<2>();
        
        let tok_emb = self.token_embedding.forward(idx);
        let pos_emb = self.position_embedding.forward(pos);
        
        let mut x = tok_emb + pos_emb;
        
        for block in &self.blocks {
            x = block.forward(x);
        }
        
        let x = self.ln_f.forward(x);
        self.lm_head.forward(x)
    }

    pub fn generate(&self, idx: Tensor<B, 2, Int>, max_new_tokens: usize, temperature: f64, block_size: usize) -> Tensor<B, 2, Int> {
        let mut idx = idx;
        for _ in 0..max_new_tokens {
            let [_, seq_len] = idx.dims();
            
            let idx_cond = if seq_len > block_size {
                idx.clone().slice([0..idx.dims()[0], seq_len - block_size..seq_len])
            } else {
                idx.clone()
            };

            let logits = self.forward(idx_cond);
            let [batch, len, vocab] = logits.dims();
            let logits = logits.slice([0..batch, len-1..len, 0..vocab]).squeeze::<2>();
            
            let logits = logits / temperature;
            let idx_next = logits.argmax(1).unsqueeze::<2>();

            idx = Tensor::cat(vec![idx, idx_next], 1);
        }
        idx
    }
}
