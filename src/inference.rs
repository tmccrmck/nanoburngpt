use crate::data::BpeTokenizer;
use crate::model::{GPTConfig, GPTRecord, GPT};
use burn::{
    config::Config,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{backend::Backend, Int, Tensor},
};

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
