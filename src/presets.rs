use crate::model::GPTConfig;
use std::str::FromStr;

#[derive(Debug)]
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

impl std::fmt::Display for ModelPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nano => write!(f, "nano"),
            Self::Gpt2Small => write!(f, "gpt2-small"),
            Self::Gpt2Medium => write!(f, "gpt2-medium"),
            Self::Gpt2Large => write!(f, "gpt2-large"),
            Self::Gpt2Xl => write!(f, "gpt2-xl"),
        }
    }
}

impl FromStr for ModelPreset {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str(s)
    }
}
