use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{backend::Backend, Int, Tensor},
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs, path::Path};

const DATA_URL: &str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CharTokenizer {
    pub char_to_idx: HashMap<char, usize>,
    pub idx_to_char: HashMap<usize, char>,
    pub vocab_size: usize,
}

impl CharTokenizer {
    pub fn new(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort();
        chars.dedup();
        
        let vocab_size = chars.len();
        let mut char_to_idx = HashMap::new();
        let mut idx_to_char = HashMap::new();

        for (i, c) in chars.into_iter().enumerate() {
            char_to_idx.insert(c, i);
            idx_to_char.insert(i, c);
        }

        Self {
            char_to_idx,
            idx_to_char,
            vocab_size,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| *self.char_to_idx.get(&c).unwrap_or(&0))
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|i| *self.idx_to_char.get(i).unwrap_or(&'?'))
            .collect()
    }
}

pub struct TextDataset {
    data: Vec<usize>,
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

pub fn load_shakespeare(path: &Path, block_size: usize) -> anyhow::Result<(TextDataset, TextDataset, CharTokenizer)> {
    let text = if path.exists() {
        fs::read_to_string(path)?
    } else {
        println!("Downloading dataset from {}...", DATA_URL);
        let text = reqwest::blocking::get(DATA_URL)?.text()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, &text)?;
        text
    };

    let tokenizer = CharTokenizer::new(&text);
    let data = tokenizer.encode(&text);
    
    // Split 90% train, 10% val
    let n = data.len();
    let split = (n as f64 * 0.9) as usize;
    let train_data = data[..split].to_vec();
    let val_data = data[split..].to_vec();

    Ok((
        TextDataset::new(train_data, block_size),
        TextDataset::new(val_data, block_size),
        tokenizer,
    ))
}

