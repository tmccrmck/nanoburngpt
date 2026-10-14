use crate::data::BpeTokenizer;
use crate::model::{GPT, GPTConfig, SamplingParams};
use burn::{config::Config, module::Module, prelude::*};
use std::io::{self, Write};

pub fn generate_text(
    device: Device,
    artifact_dir: &str,
    prompt: &str,
    max_tokens: usize,
    sampling: &SamplingParams,
) {
    // 1. Load Config
    let config_path = format!("{}/config.json", artifact_dir);
    let config = GPTConfig::load(&config_path).expect("Config should exist");

    // 2. Tokenizer — BPE vocab is fixed, no artifact needed
    let tokenizer = BpeTokenizer::new();

    // 3. Load Model (burnpack .bpk; extensionless path auto-resolves)
    println!("Loading model from {artifact_dir}/model_final ...");
    let model = GPT::new(&config, &device)
        .try_load_file(format!("{}/model_final", artifact_dir))
        .expect("Model checkpoint not found. Run `cargo run -- train` first.");

    // 4. Encode Prompt
    let tokens = tokenizer.encode(prompt);
    let token_tensor =
        Tensor::<1, Int>::from_ints(tokens.as_slice(), &device).unsqueeze::<2>();

    // 5. Generate with streaming output
    print!("{prompt}");
    io::stdout().flush().ok();

    model.generate(
        token_tensor,
        max_tokens,
        sampling,
        config.block_size,
        |token_id| {
            let text = tokenizer.decode(&[token_id as usize]);
            print!("{text}");
            io::stdout().flush().ok();
        },
    );

    println!();
}
