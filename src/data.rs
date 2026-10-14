use burn::{
    data::{dataloader::batcher::Batcher, dataset::{Dataset, DatasetError}},
    prelude::*,
};
use std::path::Path;
use tiktoken_rs::r50k_base;

pub struct TextDataset {
    pub data: Vec<usize>,
    block_size: usize,
    max_len: Option<usize>,
}

impl TextDataset {
    pub fn new(data: Vec<usize>, block_size: usize) -> Self {
        Self {
            data,
            block_size,
            max_len: None,
        }
    }

    pub fn with_max_len(mut self, max_len: usize) -> Self {
        self.max_len = Some(max_len);
        self
    }

    fn raw_len(&self) -> usize {
        if self.data.len() <= self.block_size {
            0
        } else {
            self.data.len() - self.block_size
        }
    }

    pub fn len(&self) -> usize {
        self.max_len.unwrap_or(usize::MAX).min(self.raw_len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Dataset<TextDatasetItem> for TextDataset {
    fn get(&self, index: usize) -> Result<TextDatasetItem, DatasetError> {
        if index >= self.len() {
            panic!("TextDataset index out of bounds: {index} >= {}", self.len());
        }

        let chunk = &self.data[index..index + self.block_size + 1];
        let input = chunk[..self.block_size].to_vec();
        let target = chunk[1..].to_vec();

        Ok(TextDatasetItem { input, target })
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

#[derive(Clone, Debug)]
pub struct TextGenerationBatcher {
    block_size: usize,
}

impl TextGenerationBatcher {
    pub fn new(block_size: usize) -> Self {
        Self { block_size }
    }
}

#[derive(Clone, Debug)]
pub struct TextGenerationBatch {
    pub inputs: Tensor<2, Int>,
    pub targets: Tensor<2, Int>,
}

impl Batcher<TextDatasetItem, TextGenerationBatch> for TextGenerationBatcher {
    fn batch(&self, items: Vec<TextDatasetItem>, device: &Device) -> TextGenerationBatch {
        let batch_size = items.len();
        let mut inputs_flat = Vec::with_capacity(batch_size * self.block_size);
        let mut targets_flat = Vec::with_capacity(batch_size * self.block_size);
        for item in &items {
            inputs_flat.extend_from_slice(&item.input);
            targets_flat.extend_from_slice(&item.target);
        }

        let inputs = Tensor::<1, Int>::from_ints(inputs_flat.as_slice(), device)
            .reshape([batch_size, self.block_size]);
        let targets = Tensor::<1, Int>::from_ints(targets_flat.as_slice(), device)
            .reshape([batch_size, self.block_size]);

        TextGenerationBatch { inputs, targets }
    }
}

// ---------------------------------------------------------------------------
// BPE Tokenizer (GPT-2 / r50k_base vocabulary, 50257 tokens)
// ---------------------------------------------------------------------------

pub struct BpeTokenizer {
    bpe: tiktoken_rs::CoreBPE,
}

impl Default for BpeTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl BpeTokenizer {
    pub fn new() -> Self {
        Self {
            bpe: r50k_base().expect("tiktoken r50k_base vocab should load"),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.bpe
            .encode_ordinary(text)
            .into_iter()
            .map(|t| t as usize)
            .collect()
    }

    pub fn decode(&self, tokens: &[usize]) -> String {
        self.bpe
            .decode(tokens.iter().map(|&t| t as u32).collect())
            .unwrap_or_default()
    }

    /// `<|endoftext|>` — GPT-2's BOS/EOS token.
    pub const BOS_TOKEN: usize = 50256;
    pub const VOCAB_SIZE: usize = 50257;
}

// ---------------------------------------------------------------------------
// Generic text-file loader (dataset-agnostic)
// ---------------------------------------------------------------------------

/// Tokenize text from a file, cache the token IDs as a `.bin` for fast reloads.
fn tokenize_and_cache(text_path: &Path, bin_path: &Path) -> anyhow::Result<Vec<usize>> {
    let text = std::fs::read_to_string(text_path)?;
    let tokenizer = BpeTokenizer::new();
    let mut tokens = tokenizer.encode(&text);
    tokens.insert(0, BpeTokenizer::BOS_TOKEN);

    let bytes: Vec<u8> = tokens
        .iter()
        .flat_map(|&t| (t as u32).to_le_bytes())
        .collect();
    std::fs::write(bin_path, bytes)?;

    Ok(tokens)
}

/// Load a pre-tokenized `.bin` file (raw u32 LE tokens).
fn load_cached(bin_path: &Path) -> anyhow::Result<Vec<usize>> {
    let bytes = std::fs::read(bin_path)?;
    let tokens = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as usize)
        .collect();
    Ok(tokens)
}

/// Load a text file (or its cached `.bin`), tokenize with BPE, prepend BOS,
/// split 90/10 train/val. Caches token IDs on first run — subsequent loads
/// skip BPE encoding entirely.
pub fn load_text(path: &Path, block_size: usize) -> anyhow::Result<(TextDataset, TextDataset)> {
    let bin_path = path.with_extension("bin");
    let data = if bin_path.exists() {
        load_cached(&bin_path)?
    } else {
        tokenize_and_cache(path, &bin_path)?
    };

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
        assert_eq!(item.input, vec![0, 1, 2, 3]);
        assert_eq!(item.target, vec![1, 2, 3, 4]);
    }

    #[test]
    fn get_at_last_valid_index() {
        let block_size = 3;
        let n = 10;
        let ds = make_dataset(n, block_size);
        let last = ds.len() - 1; // index 6: tokens [6,7,8] / [7,8,9]
        let item = ds.get(last).expect("should exist");
        assert_eq!(item.input, vec![6, 7, 8]);
        assert_eq!(item.target, vec![7, 8, 9]);
    }

    #[test]
    #[should_panic]
    fn get_out_of_bounds_panics() {
        let ds = make_dataset(10, 3);
        let _ = ds.get(ds.len());
    }

    #[test]
    fn is_empty_reflects_len() {
        assert!(make_dataset(3, 3).is_empty());
        assert!(!make_dataset(10, 3).is_empty());
    }

    #[test]
    fn batcher_output_shapes() {
        let block_size = 4;
        let ds = make_dataset(20, block_size);
        let batcher = TextGenerationBatcher::new(block_size);
        let items: Vec<_> = (0..3).map(|i| ds.get(i).unwrap()).collect();
        let batch = batcher.batch(items, &Device::flex());
        assert_eq!(batch.inputs.dims(), [3, block_size]);
        assert_eq!(batch.targets.dims(), [3, block_size]);
    }

    #[test]
    fn tokenizer_round_trips_ascii() {
        let tok = BpeTokenizer::new();
        let text = "Hello, world!";
        let encoded = tok.encode(text);
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, text);
    }

    #[test]
    fn tokenize_and_cache_round_trips() {
        let dir = std::env::temp_dir().join("nanoburngpt_test_cache");
        std::fs::create_dir_all(&dir).ok();
        let text_path = dir.join("test.txt");
        let bin_path = dir.join("test.bin");

        std::fs::write(&text_path, "Hello, world! This is a test.").ok();
        let tokens = super::tokenize_and_cache(&text_path, &bin_path).unwrap();
        let loaded = super::load_cached(&bin_path).unwrap();

        assert_eq!(tokens, loaded, "cached and loaded tokens should match");
        assert_eq!(tokens[0], BpeTokenizer::BOS_TOKEN, "BOS should be prepended");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn tokenizer_default_equals_new() {
        // Verifies Default impl is wired up correctly
        let tok = BpeTokenizer::default();
        let text = "test";
        assert_eq!(tok.encode(text), BpeTokenizer::new().encode(text));
    }
}
