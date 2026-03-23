# Design: Multi-Dataset Support, Model Presets, BPE Tokenizer

**Date:** 2026-03-23
**Status:** Approved

## Summary

Three related features added together:
1. Replace the character-level tokenizer with tiktoken-rs BPE (GPT-2 vocab)
2. Named dataset registry with auto-download (`--dataset shakespeare|wikitext103`)
3. Named model size presets (`--model nano|gpt2-small|gpt2-medium|gpt2-large|gpt2-xl`)

---

## 1. BPE Tokenizer (tiktoken-rs)

**What:** Remove `CharTokenizer` from `data.rs`. Add a thin `BpeTokenizer` wrapper around `tiktoken_rs::CoreBPE` using the `r50k_base` (GPT-2) vocabulary.

**Why:** BPE tokenization compresses text more efficiently than character-level — the same 1024-token context window covers far more semantic content. Vocab size becomes 50257 (fixed), enabling direct comparison with published GPT-2 results. Also unblocks training on non-Shakespeare datasets where character-level degrades.

**Implementation:**

```rust
// src/data.rs
pub struct BpeTokenizer {
    bpe: tiktoken_rs::CoreBPE,
}

impl BpeTokenizer {
    pub fn new() -> Self {
        Self { bpe: tiktoken_rs::r50k_base().unwrap() }
    }
    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.bpe.encode_ordinary(text)
    }
    pub fn decode(&self, tokens: &[usize]) -> String {
        self.bpe.decode(tokens.to_vec()).unwrap_or_default()
    }
    pub const VOCAB_SIZE: usize = 50257;
}
```

**Artifact changes:**
- `artifacts/tokenizer.json` is no longer written or read — tiktoken-rs manages its own vocab cache
- `GPTConfig.vocab_size` is always set to `BpeTokenizer::VOCAB_SIZE` (50257); the `vocab_size: 0` placeholder is removed
- `inference.rs` constructs `BpeTokenizer::new()` directly instead of loading from disk

**Cargo.toml:** Add `tiktoken-rs = "0.5"` (verify latest version at implementation time).

---

## 2. Dataset Registry

**What:** New `src/datasets.rs` module. `Dataset` enum with variants for each supported dataset. Each variant knows its download URL, local cache path, and any preprocessing needed.

**Initial datasets:**
- `shakespeare` — Karpathy's raw Shakespeare text (existing download logic, moved here)
- `wikitext103` — WikiText-103-raw from S3; strip markup tokens (` @-@ `, `= Heading =` patterns) before use

**Interface:**

```rust
pub enum Dataset {
    Shakespeare,
    WikiText103,
}

impl Dataset {
    pub fn from_str(s: &str) -> anyhow::Result<Self>
    pub fn local_path(&self) -> PathBuf        // e.g. data/shakespeare/input.txt
    pub fn ensure_downloaded(&self) -> anyhow::Result<PathBuf>
}
```

`ensure_downloaded` checks if `local_path()` exists; if not, downloads and preprocesses. Uses `reqwest` blocking (already a dependency). WikiText-103 downloads a zip, extracts, concatenates train/valid splits, strips markup.

**`data.rs` changes:** `load_shakespeare` is replaced by `load_text(path: &Path, block_size: usize) -> (TextDataset, TextDataset)` — generic text file loader with no dataset-specific logic. It returns only the two datasets (no tokenizer), since vocab_size is now fixed. All dataset knowledge lives in `datasets.rs`.

**`train.rs` changes:** `run_training` signature gains a `dataset: Dataset` parameter and loses the tokenizer save. Updated signature:

```rust
pub fn run_training<B: AutodiffBackend>(
    device: B::Device,
    dataset: Dataset,
    mut gpt_config: GPTConfig,
    training_config: TrainingConfig,
)
```

Internally: call `dataset.ensure_downloaded()` to get the text path, then `load_text(path, block_size)`. Remove the `CharTokenizer::save()` call. `gpt_config.vocab_size` is set to `BpeTokenizer::VOCAB_SIZE` (50257) instead of deriving from the tokenizer.

---

## 3. Model Presets

**What:** New `src/presets.rs` module. `ModelPreset` enum mapping names to `GPTConfig` values.

**Presets:**

| `--model` | n_layer | n_head | n_embd | block_size | ~params |
|-----------|---------|--------|--------|------------|---------|
| `nano` | 2 | 4 | 64 | 32 | <1M |
| `gpt2-small` | 12 | 12 | 768 | 1024 | 117M |
| `gpt2-medium` | 24 | 16 | 1024 | 1024 | 345M |
| `gpt2-large` | 36 | 20 | 1280 | 1024 | 762M |
| `gpt2-xl` | 48 | 25 | 1600 | 1024 | 1.5B |

**Interface:**

```rust
pub enum ModelPreset {
    Nano,
    Gpt2Small,
    Gpt2Medium,
    Gpt2Large,
    Gpt2Xl,
}

impl ModelPreset {
    pub fn from_str(s: &str) -> anyhow::Result<Self>

    pub fn config(&self) -> GPTConfig {
        // vocab_size is always BpeTokenizer::VOCAB_SIZE (50257)
        match self {
            Self::Nano => GPTConfig { vocab_size: 50257, n_layer: 2,  n_head: 4,  n_embd: 64,   block_size: 32,   dropout: 0.0 },
            Self::Gpt2Small  => GPTConfig { vocab_size: 50257, n_layer: 12, n_head: 12, n_embd: 768,  block_size: 1024, dropout: 0.1 },
            Self::Gpt2Medium => GPTConfig { vocab_size: 50257, n_layer: 24, n_head: 16, n_embd: 1024, block_size: 1024, dropout: 0.1 },
            Self::Gpt2Large  => GPTConfig { vocab_size: 50257, n_layer: 36, n_head: 20, n_embd: 1280, block_size: 1024, dropout: 0.1 },
            Self::Gpt2Xl     => GPTConfig { vocab_size: 50257, n_layer: 48, n_head: 25, n_embd: 1600, block_size: 1024, dropout: 0.1 },
        }
    }
}
```

---

## 4. CLI Changes

**`--model`** (default: `nano`) and **`--dataset`** (default: `shakespeare`) added to the `Train` subcommand.

The four architectural flags become `Option<usize>` (no built-in default). The preset fills the baseline; an explicit flag overrides just that field.

Updated `Train` struct (relevant fields only):
```rust
Train {
    #[arg(long, default_value = "nano")]     model: String,
    #[arg(long, default_value = "shakespeare")] dataset: String,
    #[arg(long)] n_layer: Option<usize>,
    #[arg(long)] n_head:  Option<usize>,
    #[arg(long)] n_embd:  Option<usize>,
    #[arg(long)] block_size: Option<usize>,
    // all other flags unchanged with their existing defaults
}
```

In `dispatch`, add imports and resolve preset:
```rust
use nanoburngpt::{datasets::Dataset, presets::ModelPreset};

let preset = ModelPreset::from_str(&model)?.config();
let gpt_config = GPTConfig {
    vocab_size: preset.vocab_size,   // always 50257
    n_layer:    n_layer.unwrap_or(preset.n_layer),
    n_head:     n_head.unwrap_or(preset.n_head),
    n_embd:     n_embd.unwrap_or(preset.n_embd),
    block_size: block_size.unwrap_or(preset.block_size),
    dropout,
};
let dataset = Dataset::from_str(&dataset_str)?;
run_training::<AB>(device, dataset, gpt_config, training_config);
```

**Updated smoke test:**
```sh
cargo run -- train --dataset shakespeare --model nano \
  --batch-size 16 --max-train-items 2000 --num-epochs 1
cargo run -- generate --max-tokens 100
```

**Real training examples:**
```sh
cargo run --release -- train --model gpt2-small --dataset shakespeare
cargo run --release --features cuda -- train --model gpt2-small --dataset wikitext103
# Override one preset field:
cargo run --release -- train --model gpt2-small --n-layer 8
```

---

## Files Changed

| File | Change |
|------|--------|
| `src/data.rs` | Remove `CharTokenizer` (incl. `save`/`load`); add `BpeTokenizer`; replace `load_shakespeare` with `load_text` |
| `src/datasets.rs` *(new)* | `Dataset` enum with download + preprocessing logic |
| `src/presets.rs` *(new)* | `ModelPreset` enum with GPT-2 family configs |
| `src/lib.rs` | Export `pub mod datasets` and `pub mod presets` |
| `src/main.rs` | Add `--dataset`, `--model`; make 4 arch flags `Option`; import `Dataset`, `ModelPreset` |
| `src/train.rs` | Accept `Dataset` parameter; remove tokenizer save; call `ensure_downloaded` + `load_text` |
| `src/inference.rs` | Use `BpeTokenizer::new()` instead of loading from `artifacts/tokenizer.json` |
| `Cargo.toml` | Add `tiktoken-rs` (check crates.io for latest `0.5.x` at implementation time) |
| `CLAUDE.md` | Update commands, remove tokenizer artifact notes |

## Out of Scope

- Training the BPE vocabulary (we use GPT-2's pretrained vocab)
- Additional datasets beyond Shakespeare and WikiText-103
- Loading pretrained GPT-2 weights into the model
