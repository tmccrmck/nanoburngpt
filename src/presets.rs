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
                rope_theta: 10000.0,
            },
            Self::Gpt2Small => GPTConfig {
                vocab_size: 50257,
                n_layer: 12,
                n_head: 12,
                n_embd: 768,
                block_size: 1024,
                dropout: 0.1,
                rope_theta: 10000.0,
            },
            Self::Gpt2Medium => GPTConfig {
                vocab_size: 50257,
                n_layer: 24,
                n_head: 16,
                n_embd: 1024,
                block_size: 1024,
                dropout: 0.1,
                rope_theta: 10000.0,
            },
            Self::Gpt2Large => GPTConfig {
                vocab_size: 50257,
                n_layer: 36,
                n_head: 20,
                n_embd: 1280,
                block_size: 1024,
                dropout: 0.1,
                rope_theta: 10000.0,
            },
            Self::Gpt2Xl => GPTConfig {
                vocab_size: 50257,
                n_layer: 48,
                n_head: 25,
                n_embd: 1600,
                block_size: 1024,
                dropout: 0.1,
                rope_theta: 10000.0,
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
        match s {
            "nano" => Ok(Self::Nano),
            "gpt2-small" => Ok(Self::Gpt2Small),
            "gpt2-medium" => Ok(Self::Gpt2Medium),
            "gpt2-large" => Ok(Self::Gpt2Large),
            "gpt2-xl" => Ok(Self::Gpt2Xl),
            other => anyhow::bail!(
                "Unknown model preset '{}'. Available: nano, gpt2-small, gpt2-medium, gpt2-large, gpt2-xl",
                other
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_str_all_variants() {
        assert!(matches!("nano".parse::<ModelPreset>().unwrap(), ModelPreset::Nano));
        assert!(matches!("gpt2-small".parse::<ModelPreset>().unwrap(), ModelPreset::Gpt2Small));
        assert!(matches!("gpt2-medium".parse::<ModelPreset>().unwrap(), ModelPreset::Gpt2Medium));
        assert!(matches!("gpt2-large".parse::<ModelPreset>().unwrap(), ModelPreset::Gpt2Large));
        assert!(matches!("gpt2-xl".parse::<ModelPreset>().unwrap(), ModelPreset::Gpt2Xl));
    }

    #[test]
    fn from_str_unknown_errors() {
        assert!("gpt3".parse::<ModelPreset>().is_err());
        assert!("".parse::<ModelPreset>().is_err());
    }

    #[test]
    fn nano_config_values() {
        let c = ModelPreset::Nano.config();
        assert_eq!(c.n_layer, 2);
        assert_eq!(c.n_head, 4);
        assert_eq!(c.n_embd, 64);
        assert_eq!(c.block_size, 32);
        assert_eq!(c.vocab_size, 50257);
    }

    #[test]
    fn gpt2_small_config_values() {
        let c = ModelPreset::Gpt2Small.config();
        assert_eq!(c.n_layer, 12);
        assert_eq!(c.n_head, 12);
        assert_eq!(c.n_embd, 768);
        assert_eq!(c.block_size, 1024);
    }

    #[test]
    fn head_dim_divides_evenly_for_all_presets() {
        for preset in [
            ModelPreset::Nano,
            ModelPreset::Gpt2Small,
            ModelPreset::Gpt2Medium,
            ModelPreset::Gpt2Large,
            ModelPreset::Gpt2Xl,
        ] {
            let c = preset.config();
            assert_eq!(
                c.n_embd % c.n_head,
                0,
                "{} n_embd={} n_head={} — not evenly divisible",
                c.n_embd,
                c.n_head,
                c.n_head
            );
            let head_dim = c.n_embd / c.n_head;
            assert_eq!(
                head_dim % 2,
                0,
                "n_embd={} n_head={} head_dim={head_dim} — must be even for RoPE",
                c.n_embd,
                c.n_head
            );
        }
    }
}
