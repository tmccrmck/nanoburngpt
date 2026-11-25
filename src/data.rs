use burn::{
    data::{dataloader::batcher::Batcher, dataset::{Dataset, DatasetError}},
    prelude::*,
};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    sync::Arc,
};
use tiktoken_rs::r50k_base;

pub struct TextDataset {
    pub data: Arc<[i32]>,
    start_offset: usize,
    token_len: usize,
    block_size: usize,
    max_len: Option<usize>,
}

impl TextDataset {
    pub fn new(data: Vec<usize>, block_size: usize) -> Self {
        let tokens: Arc<[i32]> = data.into_iter().map(|t| t as i32).collect();
        Self::from_tokens(tokens, block_size)
    }

    pub fn from_tokens(data: Arc<[i32]>, block_size: usize) -> Self {
        let token_len = data.len();
        Self::from_tokens_range(data, 0, token_len, block_size)
    }

    pub fn from_tokens_range(
        data: Arc<[i32]>,
        start_offset: usize,
        token_len: usize,
        block_size: usize,
    ) -> Self {
        debug_assert!(start_offset + token_len <= data.len());
        Self {
            data,
            start_offset,
            token_len,
            block_size,
            max_len: None,
        }
    }

    pub fn with_max_len(mut self, max_len: usize) -> Self {
        self.max_len = Some(max_len);
        self
    }

    fn raw_len(&self) -> usize {
        if self.block_size == 0 || self.token_len <= self.block_size {
            0
        } else {
            (self.token_len - 1) / self.block_size
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

        let start = self.start_offset + index * self.block_size;
        Ok(TextDatasetItem {
            data: Arc::clone(&self.data),
            start,
            len: self.block_size,
        })
    }

    fn len(&self) -> usize {
        self.len()
    }
}

/// Zero-copy slice handle into a shared token buffer (`Arc<[i32]>`).
#[derive(Clone, Debug)]
pub struct TextDatasetItem {
    data: Arc<[i32]>,
    start: usize,
    len: usize,
}

impl TextDatasetItem {
    #[inline]
    pub fn input(&self) -> &[i32] {
        &self.data[self.start..self.start + self.len]
    }

    #[inline]
    pub fn target(&self) -> &[i32] {
        &self.data[self.start + 1..self.start + self.len + 1]
    }
}

#[derive(Clone, Debug)]
pub struct TextGenerationBatcher {
    block_size: usize,
    loss_scale: f64,
}

impl TextGenerationBatcher {
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size,
            loss_scale: 1.0,
        }
    }

    pub fn with_loss_scale(mut self, loss_scale: f64) -> Self {
        self.loss_scale = loss_scale;
        self
    }
}

#[derive(Clone, Debug)]
pub struct TextGenerationBatch {
    pub inputs: Tensor<2, Int>,
    pub targets: Tensor<2, Int>,
    pub loss_scale: f64,
}

impl Batcher<TextDatasetItem, TextGenerationBatch> for TextGenerationBatcher {
    fn batch(&self, items: Vec<TextDatasetItem>, device: &Device) -> TextGenerationBatch {
        let batch_size = items.len();
        let mut inputs_flat = Vec::<i32>::with_capacity(batch_size * self.block_size);
        let mut targets_flat = Vec::<i32>::with_capacity(batch_size * self.block_size);
        for item in &items {
            inputs_flat.extend_from_slice(item.input());
            targets_flat.extend_from_slice(item.target());
        }

        let inputs = Tensor::<1, Int>::from_ints(inputs_flat.as_slice(), device)
            .reshape([batch_size, self.block_size]);
        let targets = Tensor::<1, Int>::from_ints(targets_flat.as_slice(), device)
            .reshape([batch_size, self.block_size]);

        TextGenerationBatch {
            inputs,
            targets,
            loss_scale: self.loss_scale,
        }
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

/// Tokenize text from a file, atomically cache the token IDs as a `.bin` for fast reloads.
fn tokenize_and_cache(text_path: &Path, bin_path: &Path) -> anyhow::Result<Vec<i32>> {
    let text = std::fs::read_to_string(text_path)?;
    let tokenizer = BpeTokenizer::new();
    let encoded = tokenizer.encode(&text);
    drop(text);

    let mut tokens = Vec::with_capacity(encoded.len() + 1);
    tokens.push(BpeTokenizer::BOS_TOKEN as i32);
    tokens.extend(encoded.into_iter().map(|t| t as i32));

    // Stream via BufWriter to a sibling temp file and rename atomically so an
    // interrupted run never leaves a corrupt `.bin` cache.
    let tmp_path = bin_path.with_extension("bin.tmp");
    {
        let mut writer = BufWriter::new(File::create(&tmp_path)?);
        for &t in &tokens {
            writer.write_all(&(t as u32).to_le_bytes())?;
        }
        writer.flush()?;
    }
    std::fs::rename(&tmp_path, bin_path)?;

    Ok(tokens)
}

/// Load a pre-tokenized `.bin` file (raw u32 LE tokens) directly into `Vec<i32>`.
fn load_cached(bin_path: &Path) -> anyhow::Result<Vec<i32>> {
    let bytes = std::fs::read(bin_path)?;
    if !bytes.len().is_multiple_of(4) {
        anyhow::bail!(
            "Corrupt token cache {} (byte length {} is not a multiple of 4); delete it to re-tokenize.",
            bin_path.display(),
            bytes.len()
        );
    }
    let tokens = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()) as i32)
        .collect();
    Ok(tokens)
}

fn load_or_tokenize(path: &Path) -> anyhow::Result<Arc<[i32]>> {
    let bin_path = path.with_extension("bin");
    let data = if bin_path.exists() {
        load_cached(&bin_path)?
    } else {
        tokenize_and_cache(path, &bin_path)?
    };
    Ok(Arc::from(data))
}

/// Load a single text file (or its cached `.bin`) into a `TextDataset`.
pub fn load_text_file(path: &Path, block_size: usize) -> anyhow::Result<TextDataset> {
    let data = load_or_tokenize(path)?;
    Ok(TextDataset::from_tokens(data, block_size))
}

/// Load a text file (or its cached `.bin`), tokenize with BPE, prepend BOS,
/// split 90/10 train/val sharing a single `Arc<[i32]>` buffer. Caches token IDs
/// on first run — subsequent loads skip BPE encoding entirely.
pub fn load_text(path: &Path, block_size: usize) -> anyhow::Result<(TextDataset, TextDataset)> {
    let data = load_or_tokenize(path)?;
    let n = data.len();
    let split = (n as f64 * 0.9) as usize;

    Ok((
        TextDataset::from_tokens_range(Arc::clone(&data), 0, split, block_size),
        TextDataset::from_tokens_range(data, split, n - split, block_size),
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
    fn len_is_non_overlapping_chunks() {
        let ds = make_dataset(10, 3);
        // 10 tokens with block_size=3 -> (10 - 1) / 3 = 3 chunks: [0..4], [3..7], [6..10]
        assert_eq!(ds.len(), 3);
        assert_eq!(make_dataset(4, 3).len(), 1);
    }

    #[test]
    fn chunks_are_non_overlapping() {
        let ds = make_dataset(10, 3);
        assert_eq!(ds.get(0).unwrap().input(), &[0, 1, 2]);
        assert_eq!(ds.get(0).unwrap().target(), &[1, 2, 3]);
        assert_eq!(ds.get(1).unwrap().input(), &[3, 4, 5]);
        assert_eq!(ds.get(1).unwrap().target(), &[4, 5, 6]);
        assert_eq!(ds.get(2).unwrap().input(), &[6, 7, 8]);
        assert_eq!(ds.get(2).unwrap().target(), &[7, 8, 9]);
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
        assert_eq!(item.input(), &[0, 1, 2, 3]);
        assert_eq!(item.target(), &[1, 2, 3, 4]);
    }

    #[test]
    fn get_at_last_valid_index() {
        let block_size = 3;
        let n = 10;
        let ds = make_dataset(n, block_size);
        let last = ds.len() - 1; // index 2: tokens [6,7,8] / [7,8,9]
        let item = ds.get(last).expect("should exist");
        assert_eq!(item.input(), &[6, 7, 8]);
        assert_eq!(item.target(), &[7, 8, 9]);
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
        assert_eq!(tokens[0], BpeTokenizer::BOS_TOKEN as i32, "BOS should be prepended");

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
