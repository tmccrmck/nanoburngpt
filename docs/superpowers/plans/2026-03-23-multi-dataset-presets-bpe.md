# Multi-Dataset, Model Presets, BPE Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the character-level tokenizer with tiktoken-rs BPE, add a named dataset registry with auto-download, and expose GPT-2 family model size presets via CLI.

**Architecture:** Two new modules (`datasets.rs`, `presets.rs`) encapsulate dataset download logic and model configs respectively. `data.rs` loses `CharTokenizer`/`load_shakespeare` and gains `BpeTokenizer`/`load_text`. `train.rs` and `inference.rs` are updated to use the new types. `main.rs` gets `--dataset` and `--model` flags with `Option<usize>` overrides for arch params.

**Tech Stack:** Rust, Burn 0.20, tiktoken-rs 0.9.1 (BPE tokenizer), zip 2.x (WikiText-103 extraction), reqwest blocking (already present)

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `Cargo.toml` | Modify | Add tiktoken-rs, zip dependencies |
| `src/data.rs` | Modify | Replace `CharTokenizer` + `load_shakespeare` with `BpeTokenizer` + `load_text` |
| `src/datasets.rs` | Create | `Dataset` enum — names, URLs, download, preprocessing |
| `src/presets.rs` | Create | `ModelPreset` enum — GPT-2 family + nano configs |
| `src/lib.rs` | Modify | Export `pub mod datasets` and `pub mod presets` |
| `src/train.rs` | Modify | Accept `Dataset` param; use `load_text`; remove tokenizer save |
| `src/inference.rs` | Modify | Use `BpeTokenizer::new()` instead of loading from disk |
| `src/main.rs` | Modify | Add `--dataset`, `--model`; arch flags → `Option<usize>`; import new modules |
| `CLAUDE.md` | Modify | Update commands section |

---

## Task 1: Add dependencies

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Add tiktoken-rs and zip to Cargo.toml**

```toml
[dependencies]
# existing deps...
tiktoken-rs = "0.9.1"
zip = "2"
```

- [ ] **Step 2: Verify it compiles**

```sh
cargo check
```

Expected: compiles clean (new deps downloaded, nothing else changed yet).

- [ ] **Step 3: Commit**

```sh
git add Cargo.toml Cargo.lock
git commit -m "chore: add tiktoken-rs and zip dependencies"
```

---

## Task 2: Add BpeTokenizer to data.rs

**Files:**
- Modify: `src/data.rs`

The `CharTokenizer` and `load_shakespeare` will be removed in Task 8 after all callers are updated. For now, add the new code alongside.

- [ ] **Step 1: Add BpeTokenizer and load_text to data.rs**

Add after the existing imports at the top:

```rust
use tiktoken_rs::r50k_base;
```

Add the `BpeTokenizer` struct and `load_text` function at the bottom of `data.rs` (before the closing of the file):

```rust
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
        self.bpe.encode_ordinary(text)
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
```

Note: tiktoken-rs 0.9.x `decode` takes `Vec<u32>` not `Vec<usize>` — the `map(|&t| t as u32)` conversion is required. (The spec's code snippet omits this conversion; the plan's version above is correct.)

- [ ] **Step 2: Verify it compiles**

```sh
cargo check
```

Expected: clean compile. Both `CharTokenizer` and `BpeTokenizer` exist side-by-side at this point.

- [ ] **Step 3: Commit**

```sh
git add src/data.rs
git commit -m "feat: add BpeTokenizer and load_text alongside existing char tokenizer"
```

---

## Task 3: Create src/datasets.rs

**Files:**
- Create: `src/datasets.rs`

- [ ] **Step 1: Write datasets.rs**

```rust
use std::{
    fs,
    io::{Cursor, Read},
    path::PathBuf,
};

const SHAKESPEARE_URL: &str =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

const WIKITEXT103_URL: &str =
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip";

pub enum Dataset {
    Shakespeare,
    WikiText103,
}

impl Dataset {
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "shakespeare" => Ok(Self::Shakespeare),
            "wikitext103" => Ok(Self::WikiText103),
            other => anyhow::bail!(
                "Unknown dataset '{}'. Available: shakespeare, wikitext103",
                other
            ),
        }
    }

    /// Path where the preprocessed plain-text file is cached locally.
    pub fn local_path(&self) -> PathBuf {
        match self {
            Self::Shakespeare => PathBuf::from("data/shakespeare/input.txt"),
            Self::WikiText103 => PathBuf::from("data/wikitext103/input.txt"),
        }
    }

    /// Ensure the dataset is downloaded and preprocessed. Returns the local path.
    pub fn ensure_downloaded(&self) -> anyhow::Result<PathBuf> {
        let path = self.local_path();
        if path.exists() {
            return Ok(path);
        }

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        match self {
            Self::Shakespeare => self.download_shakespeare(&path),
            Self::WikiText103 => self.download_wikitext103(&path),
        }?;

        Ok(path)
    }

    fn download_shakespeare(&self, dest: &std::path::Path) -> anyhow::Result<()> {
        println!("Downloading Shakespeare from {}...", SHAKESPEARE_URL);
        let text = reqwest::blocking::get(SHAKESPEARE_URL)?.text()?;
        fs::write(dest, text)?;
        println!("Saved to {}", dest.display());
        Ok(())
    }

    fn download_wikitext103(&self, dest: &std::path::Path) -> anyhow::Result<()> {
        println!("Downloading WikiText-103 from {}...", WIKITEXT103_URL);
        let bytes = reqwest::blocking::get(WIKITEXT103_URL)?.bytes()?;
        println!("Extracting zip ({} MB)...", bytes.len() / 1_000_000);

        let cursor = Cursor::new(bytes);
        let mut archive = zip::ZipArchive::new(cursor)?;

        // Collect and concatenate the train + valid raw splits
        let target_files = [
            "wikitext-103-raw/wiki.train.raw",
            "wikitext-103-raw/wiki.valid.raw",
        ];

        let mut combined = String::new();
        for name in &target_files {
            match archive.by_name(name) {
                Ok(mut file) => {
                    file.read_to_string(&mut combined)?;
                    combined.push('\n');
                }
                Err(_) => {
                    // Zip entry names may vary — list available files for debugging
                    let names: Vec<String> = (0..archive.len())
                        .filter_map(|i| archive.by_index(i).ok().map(|f| f.name().to_string()))
                        .collect();
                    anyhow::bail!(
                        "Could not find '{}' in zip. Available files: {:?}",
                        name,
                        names
                    );
                }
            }
        }

        // Strip WikiText markup tokens
        let cleaned = strip_wikitext_markup(&combined);
        fs::write(dest, cleaned)?;
        println!("Saved preprocessed text to {}", dest.display());
        Ok(())
    }
}

/// Remove WikiText-103 markup artifacts from raw text.
fn strip_wikitext_markup(text: &str) -> String {
    text.lines()
        .filter(|line| {
            // Drop heading lines: = Title =, = = Section = =, etc.
            let trimmed = line.trim();
            !(trimmed.starts_with('=') && trimmed.ends_with('='))
        })
        .map(|line| {
            // Replace tokenization artifacts with their intended characters
            line.replace(" @-@ ", "-")
                .replace(" @.@ ", ".")
                .replace(" @,@ ", ",")
        })
        .collect::<Vec<_>>()
        .join("\n")
}
```

- [ ] **Step 2: Add reqwest import at top of datasets.rs**

The file already uses `reqwest` (added above). Ensure the import is at the top:

```rust
use std::{
    fs,
    io::{Cursor, Read},
    path::PathBuf,
};
```

`reqwest` is used as a path (`reqwest::blocking::get`) without a `use` statement, which is fine. No extra import needed.

- [ ] **Step 3: Create a presets.rs stub and export both new modules from lib.rs**

First create the stub so lib.rs export doesn't cause a compile error:

`src/presets.rs` (stub):
```rust
// stub — filled in Task 4
use crate::model::GPTConfig;
pub enum ModelPreset {}
impl ModelPreset {
    pub fn from_str(_s: &str) -> anyhow::Result<Self> { anyhow::bail!("stub") }
    pub fn config(&self) -> GPTConfig { unimplemented!() }
}
```

Then in `src/lib.rs`:
```rust
pub mod model;
pub mod data;
pub mod train;
pub mod inference;
pub mod datasets;
pub mod presets;
```

- [ ] **Step 4: Verify it compiles**

```sh
cargo check
```

Expected: clean compile (both new modules exist, even if presets is a stub).

- [ ] **Step 5: Commit**

```sh
git add src/datasets.rs src/lib.rs src/presets.rs
git commit -m "feat: add Dataset registry with Shakespeare and WikiText-103 download"
```

---

## Task 4: Create src/presets.rs

**Files:**
- Modify: `src/presets.rs` (replace stub)

- [ ] **Step 1: Write presets.rs**

```rust
use crate::model::GPTConfig;

pub enum ModelPreset {
    Nano,
    Gpt2Small,
    Gpt2Medium,
    Gpt2Large,
    Gpt2Xl,
}

impl ModelPreset {
    pub fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "nano"        => Ok(Self::Nano),
            "gpt2-small"  => Ok(Self::Gpt2Small),
            "gpt2-medium" => Ok(Self::Gpt2Medium),
            "gpt2-large"  => Ok(Self::Gpt2Large),
            "gpt2-xl"     => Ok(Self::Gpt2Xl),
            other => anyhow::bail!(
                "Unknown model preset '{}'. Available: nano, gpt2-small, gpt2-medium, gpt2-large, gpt2-xl",
                other
            ),
        }
    }

    /// Return the GPTConfig for this preset.
    /// vocab_size is always 50257 (BpeTokenizer::VOCAB_SIZE).
    pub fn config(&self) -> GPTConfig {
        match self {
            Self::Nano => GPTConfig {
                vocab_size: 50257,
                n_layer: 2,
                n_head: 4,
                n_embd: 64,
                block_size: 32,
                dropout: 0.0,
            },
            Self::Gpt2Small => GPTConfig {
                vocab_size: 50257,
                n_layer: 12,
                n_head: 12,
                n_embd: 768,
                block_size: 1024,
                dropout: 0.1,
            },
            Self::Gpt2Medium => GPTConfig {
                vocab_size: 50257,
                n_layer: 24,
                n_head: 16,
                n_embd: 1024,
                block_size: 1024,
                dropout: 0.1,
            },
            Self::Gpt2Large => GPTConfig {
                vocab_size: 50257,
                n_layer: 36,
                n_head: 20,
                n_embd: 1280,
                block_size: 1024,
                dropout: 0.1,
            },
            Self::Gpt2Xl => GPTConfig {
                vocab_size: 50257,
                n_layer: 48,
                n_head: 25,
                n_embd: 1600,
                block_size: 1024,
                dropout: 0.1,
            },
        }
    }
}
```

- [ ] **Step 2: Verify it compiles**

```sh
cargo check
```

Expected: clean compile.

- [ ] **Step 3: Commit**

```sh
git add src/presets.rs
git commit -m "feat: add ModelPreset enum with nano and GPT-2 family configs"
```

---

## Task 5: Update train.rs

**Files:**
- Modify: `src/train.rs`

- [ ] **Step 1: Add Dataset import**

At the top of `src/train.rs`, add:

```rust
use crate::{
    data::{load_text, BpeTokenizer, TextDataset, TextGenerationBatch, TextGenerationBatcher},
    datasets::Dataset,
    model::{GPTConfig, GPT},
};
```

Remove the `CharTokenizer` and `load_shakespeare` imports.

- [ ] **Step 2: Update run_training signature**

Change:

```rust
pub fn run_training<B: AutodiffBackend>(
    device: B::Device,
    mut gpt_config: GPTConfig,
    training_config: TrainingConfig,
) {
    let (train_dataset, val_dataset, tokenizer) =
        load_shakespeare(Path::new("data/input.txt"), gpt_config.block_size).unwrap();
```

To:

```rust
pub fn run_training<B: AutodiffBackend>(
    device: B::Device,
    dataset: Dataset,
    mut gpt_config: GPTConfig,
    training_config: TrainingConfig,
) {
    let data_path = dataset.ensure_downloaded().expect("Dataset download failed");
    let (train_dataset, val_dataset) =
        load_text(&data_path, gpt_config.block_size).expect("Failed to load dataset");
```

- [ ] **Step 3: Fix vocab_size and remove tokenizer save**

Replace:

```rust
gpt_config.vocab_size = tokenizer.vocab_size;
println!("Vocab size: {}", gpt_config.vocab_size);
```

With:

```rust
gpt_config.vocab_size = BpeTokenizer::VOCAB_SIZE;
println!("Vocab size: {} (BPE r50k_base)", gpt_config.vocab_size);
```

Remove these lines (tokenizer save, now unused):

```rust
tokenizer
    .save(Path::new("artifacts/tokenizer.json"))
    .expect("Failed to save tokenizer");
println!("Tokenizer saved to artifacts/tokenizer.json");
```

- [ ] **Step 4: Verify it compiles**

```sh
cargo check
```

Expected: error in `main.rs` because `run_training` signature changed. That's expected — fix in Task 7.

- [ ] **Step 5: Commit once main.rs is also updated (defer to Task 7)**

---

## Task 6: Update inference.rs

**Files:**
- Modify: `src/inference.rs`

- [ ] **Step 1: Replace tokenizer load with BpeTokenizer::new()**

Current inference.rs loads `CharTokenizer` from disk with a fallback. Replace entirely:

Change imports from:

```rust
use crate::data::{CharTokenizer, load_shakespeare};
```

To:

```rust
use crate::data::BpeTokenizer;
```

Remove the tokenizer loading block:

```rust
// 2. Load Tokenizer (saved during training, fallback to rebuilding from data)
let tokenizer_path = format!("{}/tokenizer.json", artifact_dir);
let tokenizer = CharTokenizer::load(Path::new(&tokenizer_path))
    .unwrap_or_else(|_| { ... });
```

Replace with:

```rust
// 2. Tokenizer — BPE vocab is fixed, no artifact needed
let tokenizer = BpeTokenizer::new();
```

- [ ] **Step 2: Remove unused import**

Remove `use std::path::Path;` if it's now only used for the artifact dir string (which is `&str`). Keep it if still needed for config loading (`GPTConfig::load` takes a path string, not `Path`).

- [ ] **Step 3: Verify it compiles**

```sh
cargo check
```

Expected: error in `main.rs` still (from Task 5). That's fine.

- [ ] **Step 4: Commit (defer to after main.rs fix in Task 7)**

---

## Task 7: Update main.rs

**Files:**
- Modify: `src/main.rs`

This is the largest change. Replace the entire file:

- [ ] **Step 1: Update imports**

Change:

```rust
use burn::backend::Autodiff;
use burn::tensor::backend::{AutodiffBackend, Backend};
use clap::{Parser, Subcommand};
use nanoburngpt::{
    inference::generate_text,
    model::GPTConfig,
    train::{run_training, TrainingConfig},
};
```

To:

```rust
use burn::backend::Autodiff;
use burn::tensor::backend::{AutodiffBackend, Backend};
use clap::{Parser, Subcommand};
use nanoburngpt::{
    datasets::Dataset,
    inference::generate_text,
    model::GPTConfig,
    presets::ModelPreset,
    train::{run_training, TrainingConfig},
};
```

- [ ] **Step 2: Update the Train variant in Commands**

Replace:

```rust
Train {
    #[arg(long, default_value_t = 6)]
    n_layer: usize,
    #[arg(long, default_value_t = 6)]
    n_head: usize,
    #[arg(long, default_value_t = 384)]
    n_embd: usize,
    #[arg(long, default_value_t = 128)]
    block_size: usize,
    // ...
}
```

With:

```rust
Train {
    /// Model size preset
    #[arg(long, default_value = "nano")]
    model: String,
    /// Dataset to train on
    #[arg(long, default_value = "shakespeare")]
    dataset: String,
    /// Override preset: number of transformer layers
    #[arg(long)]
    n_layer: Option<usize>,
    /// Override preset: number of attention heads
    #[arg(long)]
    n_head: Option<usize>,
    /// Override preset: embedding dimension
    #[arg(long)]
    n_embd: Option<usize>,
    /// Override preset: context window size
    #[arg(long)]
    block_size: Option<usize>,
    /// Dropout probability
    #[arg(long, default_value_t = 0.2)]
    dropout: f64,
    /// Training batch size
    #[arg(long, default_value_t = 32)]
    batch_size: usize,
    /// Number of dataloader workers
    #[arg(long, default_value_t = 4)]
    num_workers: usize,
    /// Random seed
    #[arg(long, default_value_t = 1337)]
    seed: u64,
    /// Peak learning rate
    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f64,
    /// Number of training epochs
    #[arg(long, default_value_t = 10)]
    num_epochs: usize,
    /// Minimum LR at end of cosine decay
    #[arg(long, default_value_t = 1e-4)]
    min_lr: f64,
    /// Warmup steps before cosine decay
    #[arg(long, default_value_t = 100)]
    warmup_iters: usize,
    /// AdamW beta2
    #[arg(long, default_value_t = 0.95)]
    beta2: f64,
    /// AdamW weight decay
    #[arg(long, default_value_t = 0.1)]
    weight_decay: f64,
    /// Cap training items (0 = full dataset)
    #[arg(long, default_value_t = 0)]
    max_train_items: usize,
},
```

- [ ] **Step 3: Update the dispatch function**

Replace the `Commands::Train { ... }` match arm in `dispatch`:

```rust
Commands::Train {
    model,
    dataset,
    n_layer,
    n_head,
    n_embd,
    block_size,
    dropout,
    batch_size,
    num_workers,
    seed,
    learning_rate,
    num_epochs,
    min_lr,
    warmup_iters,
    beta2,
    weight_decay,
    max_train_items,
} => {
    let preset = ModelPreset::from_str(&model)
        .expect("Invalid --model. Use: nano, gpt2-small, gpt2-medium, gpt2-large, gpt2-xl");
    let preset_cfg = preset.config();

    let gpt_config = GPTConfig {
        vocab_size: preset_cfg.vocab_size, // always 50257
        n_layer:    n_layer.unwrap_or(preset_cfg.n_layer),
        n_head:     n_head.unwrap_or(preset_cfg.n_head),
        n_embd:     n_embd.unwrap_or(preset_cfg.n_embd),
        block_size: block_size.unwrap_or(preset_cfg.block_size),
        dropout,
    };

    let dataset_enum = Dataset::from_str(&dataset)
        .expect("Invalid --dataset. Use: shakespeare, wikitext103");

    let training_config = TrainingConfig {
        batch_size,
        num_workers,
        seed,
        learning_rate,
        min_lr,
        warmup_iters,
        beta2,
        weight_decay,
        num_epochs,
        max_train_items,
    };

    println!("Model: {} ({} layers, {} heads, {} embd, ctx {})",
        model, gpt_config.n_layer, gpt_config.n_head,
        gpt_config.n_embd, gpt_config.block_size);
    println!("Dataset: {}", dataset);
    println!("Device: {:?}", device);

    run_training::<AB>(device, dataset_enum, gpt_config, training_config);
}
```

Note: `dataset` is used both as the field name and the `Dataset` value — Rust allows this (shadow binding). The `println!` for dataset name should use the string `model`/`dataset` before they are moved. Adjust if needed.

- [ ] **Step 4: Verify it compiles**

```sh
cargo check
```

Expected: clean compile. If there are minor name-shadowing or borrow issues in the match arm, adjust accordingly.

- [ ] **Step 5: Commit all pending changes together**

```sh
git add src/train.rs src/inference.rs src/main.rs
git commit -m "feat: wire Dataset and ModelPreset into train/inference/CLI"
```

---

## Task 8: Remove CharTokenizer and load_shakespeare

**Files:**
- Modify: `src/data.rs`

All callers have been updated. Now clean up the old code.

- [ ] **Step 1: Remove from data.rs**

Delete:
- The `use serde::{Deserialize, Serialize};` import (if only used by CharTokenizer)
- The `use std::collections::HashMap;` import (if only used by CharTokenizer)
- The `const DATA_URL` constant
- The entire `CharTokenizer` struct and `impl` block (including `save`/`load`)
- The `load_shakespeare` function

Keep:
- All `TextDataset`, `TextDatasetItem`, `TextGenerationBatcher`, `TextGenerationBatch` code (unchanged)
- `BpeTokenizer` and `load_text` (added in Task 2)

Also remove from the top:
```rust
use tiktoken_rs::r50k_base;
```
(Move the `r50k_base` call inline: `tiktoken_rs::r50k_base()` — or keep the use statement, either is fine.)

Remove `serde_json` from `Cargo.toml` if it's now only used by the removed `CharTokenizer`. Check if `serde_json` is used elsewhere first:

```sh
grep -r "serde_json" src/
```

If only in `data.rs` (now removed), delete from `Cargo.toml` too.

- [ ] **Step 2: Verify it compiles**

```sh
cargo check
```

Expected: clean compile.

- [ ] **Step 3: Commit**

```sh
git add src/data.rs Cargo.toml Cargo.lock
git commit -m "refactor: remove CharTokenizer and load_shakespeare, BPE-only now"
```

---

## Task 9: Smoke test

- [ ] **Step 1: Run the smoke test**

```sh
cargo run -- train \
  --dataset shakespeare \
  --model nano \
  --batch-size 16 \
  --max-train-items 2000 \
  --num-epochs 1
```

Expected output includes:
```
Model: nano (2 layers, 4 heads, 64 embd, ctx 32)
Vocab size: 50257 (BPE r50k_base)
Steps/epoch: ...
```
Training should complete in ~30s.

- [ ] **Step 2: Test generation**

```sh
cargo run -- generate --max-tokens 100
```

Expected: generates text (likely gibberish given 1 epoch on 2000 items — that's fine).

- [ ] **Step 3: Test unknown dataset/model error messages**

```sh
cargo run -- train --dataset foobar 2>&1 | grep "Unknown dataset"
cargo run -- train --model foobar 2>&1 | grep "Unknown model"
```

Expected: both print a clear error with available options.

- [ ] **Step 4: Test preset override**

```sh
cargo run -- train --model nano --n-layer 4 --max-train-items 500 --num-epochs 1 2>&1 | grep "layers"
```

Expected: prints `4 layers` (override applied).

- [ ] **Step 5: Commit**

```sh
git add -A
git commit -m "test: smoke test passes with BPE tokenizer, dataset registry, model presets"
```

---

## Task 10: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the Commands section**

Replace the smoke test command:

```sh
# Smoke test (~30s, verifies full pipeline)
cargo run -- train \
  --dataset shakespeare --model nano \
  --batch-size 16 --max-train-items 2000 --num-epochs 1
cargo run -- generate --max-tokens 100

# Full training (Shakespeare, nano)
cargo run --release -- train --dataset shakespeare --model nano

# Full training (Shakespeare, GPT-2 small)
cargo run --release -- train --dataset shakespeare --model gpt2-small

# Cloud GPU training (CUDA backend)
cargo run --release --features cuda -- train \
  --dataset wikitext103 --model gpt2-small

# Generate text
cargo run --release -- generate --prompt "HAMLET:" --temperature 0.8
```

- [ ] **Step 2: Update the Architecture section**

Update data flow step 1:
```
1. `data.rs` — `load_text` reads a pre-downloaded text file, tokenizes with `BpeTokenizer`
   (tiktoken-rs r50k_base, vocab_size=50257), splits into `TextDataset` (train/val),
   and `TextGenerationBatcher` collates samples into `TextGenerationBatch`
```

Add new architecture notes for datasets and presets:
```
5. `datasets.rs` — `Dataset` enum (Shakespeare, WikiText103); `ensure_downloaded` fetches
   and preprocesses on first use, caches to `data/<name>/input.txt`
6. `presets.rs` — `ModelPreset` enum (nano, gpt2-small/medium/large/xl); `config()` returns
   a `GPTConfig` with all architectural parameters pre-filled
```

- [ ] **Step 3: Update Known gaps section**

Remove the tokenizer artifact note. Add:
```
- Tokenizer: always GPT-2 BPE (r50k_base, 50257 vocab) — no char-level fallback
```

- [ ] **Step 4: Verify and commit**

```sh
cargo check
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for BPE tokenizer, dataset registry, model presets"
```

---

## Verification Checklist

After all tasks complete:

- [ ] `cargo check` passes clean
- [ ] Smoke test completes in ~30s and produces generated text
- [ ] `--dataset wikitext103` triggers download (or uses cache if already present)
- [ ] `--model gpt2-small` selects 12-layer config (visible in training output)
- [ ] `--model nano --n-layer 4` overrides just n_layer
- [ ] Unknown `--dataset` / `--model` values produce clear error messages
- [ ] `artifacts/tokenizer.json` is no longer written during training
