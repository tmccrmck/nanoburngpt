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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dataset(n_tokens: usize, block_size: usize) -> TextDataset {
        let data: Vec<usize> = (0..n_tokens).collect();
        TextDataset::new(data, block_size)
    }

    #[test]
    fn len_is_tokens_minus_block_size() {
        let ds = make_dataset(10, 3);
        // 10 - 3 = 7 valid windows
        assert_eq!(ds.len(), 7);
    }

    #[test]
    fn len_zero_when_data_too_short() {
        let ds = make_dataset(3, 3);
        assert_eq!(ds.len(), 0);
    }

    #[test]
    fn get_returns_correct_input_and_target() {
        let ds = make_dataset(10, 4);
        let item = ds.get(0).expect("should exist");
        // tokens 0..4 → input, tokens 1..5 → target
        assert_eq!(item.input,  vec![0, 1, 2, 3]);
        assert_eq!(item.target, vec![1, 2, 3, 4]);
    }

    #[test]
    fn get_at_last_valid_index() {
        let block_size = 3;
        let n = 10;
        let ds = make_dataset(n, block_size);
        let last = ds.len() - 1; // index 6: tokens [6,7,8] / [7,8,9]
        let item = ds.get(last).expect("should exist");
        assert_eq!(item.input,  vec![6, 7, 8]);
        assert_eq!(item.target, vec![7, 8, 9]);
    }

    #[test]
    fn get_out_of_bounds_returns_none() {
        let ds = make_dataset(10, 3);
        assert!(ds.get(ds.len()).is_none());
        assert!(ds.get(100).is_none());
    }
}
