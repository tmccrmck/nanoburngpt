use crate::data::{CharTokenizer, load_shakespeare};
use crate::model::{GPTConfig, GPTRecord, GPT};
use burn::{
    config::Config,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{backend::Backend, Int, Tensor},
};
use std::path::Path;

pub fn generate_text<B: Backend>(
    device: B::Device,
    artifact_dir: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
) {
    // 1. Load Config
    let config_path = format!("{}/config.json", artifact_dir);
    let config = GPTConfig::load(&config_path).expect("Config should exist");

    // 2. Load Tokenizer (saved during training, fallback to rebuilding from data)
    let tokenizer_path = format!("{}/tokenizer.json", artifact_dir);
    let tokenizer = CharTokenizer::load(Path::new(&tokenizer_path))
        .unwrap_or_else(|_| {
            println!("Tokenizer not found at {tokenizer_path}, rebuilding from data...");
            let (_, _, t) = load_shakespeare(Path::new("data/input.txt"), config.block_size)
                .expect("Failed to load data for tokenizer");
            t
        });

    // 3. Load Model
    println!("Loading model from {artifact_dir}/model_final ...");
    let record: GPTRecord<B> = CompactRecorder::new()
        .load(format!("{}/model_final", artifact_dir).into(), &device)
        .expect("Model checkpoint not found. Run `cargo run -- train` first.");

    let model = GPT::new(&config, &device).load_record(record);

    // 4. Encode Prompt
    let tokens = tokenizer.encode(prompt);
    let token_tensor =
        Tensor::<B, 1, Int>::from_ints(tokens.as_slice(), &device).unsqueeze::<2>();

    // 5. Generate
    println!("Generating...\n");
    let generated = model.generate(token_tensor, max_tokens, temperature, config.block_size);

    // 6. Decode
    let data = generated.squeeze::<1>().into_data();
    let generated_tokens: Vec<usize> = data
        .as_slice::<i32>()
        .expect("i32 tensor data")
        .iter()
        .map(|&x| x as usize)
        .collect();

    let text = tokenizer.decode(&generated_tokens);
    println!(
        "Generated Text:\n----------------------------------------\n{}\n----------------------------------------",
        text
    );
}
