use crate::data::BpeTokenizer;
use crate::model::{GPT, GPTConfig, SamplingParams};
use anyhow::Context;
use burn::{config::Config, module::Module, prelude::*};
use std::io::{self, Write};

pub fn generate_text(
    device: Device,
    artifact_dir: &str,
    model_name: &str,
    prompt: &str,
    max_tokens: usize,
    sampling: &SamplingParams,
) -> anyhow::Result<String> {
    sampling.validate()?;

    // 1. Load Config
    let config_path = format!("{}/config.json", artifact_dir);
    let config = GPTConfig::load(&config_path)
        .with_context(|| format!("Failed to load model config from {config_path}"))?;
    config.validate()?;

    // 2. Tokenizer — BPE vocab is fixed, no artifact needed
    let tokenizer = BpeTokenizer::new();

    // 3. Load Model (burnpack .bpk; extensionless path auto-resolves)
    let model_path = format!("{}/{}", artifact_dir, model_name);
    println!("Loading model from {model_path} ...");
    let model = GPT::new(&config, &device)
        .try_load_file(&model_path)
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to load model checkpoint '{model_path}': {e:?}. Run `cargo run -- train` first."
            )
        })?;

    // 4. Encode Prompt (prepend BOS if prompt is empty to avoid 0-length prefill)
    let mut tokens = tokenizer.encode(prompt);
    if tokens.is_empty() {
        tokens.push(BpeTokenizer::BOS_TOKEN);
    }
    let token_tensor =
        Tensor::<1, Int>::from_ints(tokens.as_slice(), &device).unsqueeze::<2>();

    // 5. Generate with UTF-8-safe streaming output
    let mut output = String::from(prompt);
    print!("{prompt}");
    io::stdout().flush().ok();

    // Never sample padded vocab slots beyond the real BPE vocabulary (50257..50304).
    let mut sampling = sampling.clone();
    if sampling.vocab_limit.is_none() {
        sampling.vocab_limit = Some(BpeTokenizer::VOCAB_SIZE);
    }

    let mut pending: Vec<usize> = Vec::new();
    model.generate(
        token_tensor,
        max_tokens,
        &sampling,
        config.block_size,
        |token_id| {
            pending.push(token_id as usize);
            let text = tokenizer.decode(&pending);
            if !text.is_empty() && !text.contains('\u{FFFD}') {
                output.push_str(&text);
                print!("{text}");
                io::stdout().flush().ok();
                pending.clear();
            }
        },
    );
    if !pending.is_empty() {
        let tail = tokenizer.decode(&pending);
        output.push_str(&tail);
        print!("{tail}");
        io::stdout().flush().ok();
    }

    println!();
    Ok(output)
}
