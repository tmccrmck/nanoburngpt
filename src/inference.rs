use crate::data::BpeTokenizer;
use crate::model::{GPT, GPTConfig, GPTRecord, SamplingParams};
use burn::{
    config::Config,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{Int, Tensor, backend::Backend},
};
use std::io::{self, Write};

pub fn generate_text<B: Backend>(
    device: B::Device,
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

    // 3. Load Model
    println!("Loading model from {artifact_dir}/model_final ...");
    let record: GPTRecord<B> = CompactRecorder::new()
        .load(format!("{}/model_final", artifact_dir).into(), &device)
        .expect("Model checkpoint not found. Run `cargo run -- train` first.");

    let model = GPT::new(&config, &device).load_record(record);

    // 4. Encode Prompt
    let tokens = tokenizer.encode(prompt);
    let token_tensor = Tensor::<B, 1, Int>::from_ints(tokens.as_slice(), &device).unsqueeze::<2>();

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
