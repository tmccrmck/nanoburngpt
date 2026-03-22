use crate::data::load_shakespeare;
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
) {
    // 1. Load Config
    let config_path = format!("{}/config.json", artifact_dir);
    let config = GPTConfig::load(&config_path).expect("Config should exist");
    
    // 2. Load Tokenizer
    // We rebuild it from data to ensure consistency. 
    // In production, you'd save the tokenizer.
    let (_, _, tokenizer) = load_shakespeare(Path::new("data/input.txt"), config.block_size).expect("Data loaded");
    
    // 3. Load Model
    println!("Loading model...");
    let record: GPTRecord<B> = CompactRecorder::new()
        .load(format!("{}/model", artifact_dir).into(), &device)
        .expect("Model checkpoint not found");
        
    let model = GPT::new(&config, &device).load_record(record);

    // 4. Encode Prompt
    let tokens = tokenizer.encode(prompt);
    let token_tensor = Tensor::<B, 1, Int>::from_ints(tokens.as_slice(), &device).unsqueeze::<2>(); // [1, seq_len]

    // 5. Generate
    println!("Generating...");
    let generated = model.generate(token_tensor, 500, 1.0, config.block_size); // 500 tokens
    
    // 6. Decode
    let data = generated.squeeze::<1>().into_data();
    let generated_tokens: Vec<usize> = data
        .as_slice::<i32>()
        .expect("i32 tensor data")
        .iter()
        .map(|&x| x as usize)
        .collect();
    
    let text = tokenizer.decode(&generated_tokens);
    println!("\nGenerated Text:\n----------------------------------------\n{}\n----------------------------------------", text);
}
