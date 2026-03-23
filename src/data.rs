use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{backend::Backend, Int, Tensor},
};
use std::path::Path;
use tiktoken_rs::r50k_base;

pub struct TextDataset {
    pub data: Vec<usize>,
    block_size: usize,
}

impl TextDataset {
    pub fn new(data: Vec<usize>, block_size: usize) -> Self {
        Self { data, block_size }
    }

    pub fn len(&self) -> usize {
        if self.data.len() <= self.block_size {
            0
        } else {
            self.data.len() - self.block_size
        }
    }
}

impl Dataset<TextDatasetItem> for TextDataset {
    fn get(&self, index: usize) -> Option<TextDatasetItem> {
        if index + self.block_size + 1 > self.data.len() {
            return None;
        }

        let chunk = &self.data[index..index + self.block_size + 1];
        let input = chunk[..self.block_size].to_vec();
        let target = chunk[1..].to_vec();

        Some(TextDatasetItem { input, target })
    }

    fn len(&self) -> usize {
        self.len()
    }
}

#[derive(Clone, Debug)]
pub struct TextDatasetItem {
    pub input: Vec<usize>,
    pub target: Vec<usize>,
}

pub struct TextGenerationBatcher<B: Backend> {
    block_size: usize,
    _b: std::marker::PhantomData<B>,
}

impl<B: Backend> TextGenerationBatcher<B> {
    pub fn new(block_size: usize) -> Self {
        Self { block_size, _b: std::marker::PhantomData }
    }
}

#[derive(Clone, Debug)]
pub struct TextGenerationBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<B, TextDatasetItem, TextGenerationBatch<B>> for TextGenerationBatcher<B> {
    fn batch(&self, items: Vec<TextDatasetItem>, device: &B::Device) -> TextGenerationBatch<B> {
        let inputs = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_ints(item.input.as_slice(), device))
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_ints(item.target.as_slice(), device))
            .collect();

        let inputs = Tensor::cat(inputs, 0).reshape([items.len(), self.block_size]);
        let targets = Tensor::cat(targets, 0).reshape([items.len(), self.block_size]);

        TextGenerationBatch { inputs, targets }
    }
}

// ---------------------------------------------------------------------------
// BPE Tokenizer (GPT-2 / r50k_base vocabulary, 50257 tokens)
// ---------------------------------------------------------------------------

pub struct BpeTokenizer {
    bpe: tiktoken_rs::CoreBPE,
}

impl BpeTokenizer {
    pub fn new() -> Self {
        Self {
            bpe: r50k_base().expect("tiktoken r50k_base vocab should load"),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.bpe.encode_ordinary(text).into_iter().map(|t| t as usize).collect()
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        self.bpe
            .decode(tokens.iter().map(|&t| t as u32).collect())
            .unwrap_or_default()
    }

    pub const VOCAB_SIZE: usize = 50257;
}

// ---------------------------------------------------------------------------
// Generic text-file loader (dataset-agnostic)
// ---------------------------------------------------------------------------

/// Load a plain-text file, tokenize with BPE, split 90/10 train/val.
pub fn load_text(path: &Path, block_size: usize) -> anyhow::Result<(TextDataset, TextDataset)> {
    let text = std::fs::read_to_string(path)?;
    let tokenizer = BpeTokenizer::new();
    let data = tokenizer.encode(&text);

    let n = data.len();
    let split = (n as f64 * 0.9) as usize;

    Ok((
        TextDataset::new(data[..split].to_vec(), block_size),
        TextDataset::new(data[split..].to_vec(), block_size),
    ))
}
