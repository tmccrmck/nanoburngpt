use std::time::Instant;

use anyhow::Context;
use burn::{
    config::Config,
    data::{dataloader::batcher::Batcher, dataset::Dataset as BurnDataset},
    module::Module,
    prelude::*,
};

use crate::{
    data::{BpeTokenizer, TextDataset, TextGenerationBatcher},
    model::{GPT, GPTConfig},
};

/// Summary metrics from evaluating a trained [`GPT`] checkpoint on a dataset.
#[derive(Clone, Copy, Debug)]
pub struct EvalReport {
    pub loss: f64,
    pub perplexity: f64,
    pub bpb: f64,
    pub total_tokens: usize,
    pub total_bytes: usize,
    pub elapsed_secs: f64,
}

/// Precompute UTF-8 byte lengths for every token ID in the BPE vocabulary.
fn build_token_byte_lengths(tokenizer: &BpeTokenizer) -> Vec<usize> {
    (0..BpeTokenizer::VOCAB_SIZE)
        .map(|id| tokenizer.decode(&[id]).len())
        .collect()
}

/// Evaluate an in-memory [`GPT`] model on `dataset` and compute Loss, Perplexity,
/// and exact Bits-Per-Byte (BPB).
pub fn evaluate_dataset(
    model: &GPT,
    dataset: &TextDataset,
    block_size: usize,
    batch_size: usize,
    device: &Device,
) -> anyhow::Result<EvalReport> {
    let n_items = dataset.len();
    anyhow::ensure!(
        n_items > 0,
        "Evaluation dataset is empty (need more than block_size={block_size} tokens)"
    );

    let batch_size = batch_size.max(1);
    let batcher = TextGenerationBatcher::new(block_size);
    let tokenizer = BpeTokenizer::new();
    let byte_lens = build_token_byte_lengths(&tokenizer);

    let t0 = Instant::now();
    let mut total_nll = 0.0_f64;
    let mut total_tokens = 0usize;
    let mut total_bytes = 0usize;

    let mut idx = 0usize;
    while idx < n_items {
        let end = (idx + batch_size).min(n_items);
        let mut items = Vec::with_capacity(end - idx);
        for i in idx..end {
            let item = BurnDataset::get(dataset, i)
                .map_err(|e| anyhow::anyhow!("Failed to read dataset item {i}: {e:?}"))?;
            for &tok in item.target() {
                if let Some(&len) = byte_lens.get(tok as usize) {
                    total_bytes += len;
                }
            }
            items.push(item);
        }

        let batch = batcher.batch(items, device);
        let out = model.forward_loss(batch);
        let batch_loss = out
            .loss
            .try_into_scalar::<f64>()
            .map_err(|e| anyhow::anyhow!("Failed to read eval loss scalar: {e:?}"))?;
        total_nll += batch_loss * out.num_tokens as f64;
        total_tokens += out.num_tokens;
        idx = end;
    }

    let elapsed_secs = t0.elapsed().as_secs_f64().max(1e-6);
    let loss = total_nll / total_tokens as f64;
    let perplexity = loss.exp();
    let bpb = if total_bytes > 0 {
        total_nll / (total_bytes as f64 * std::f64::consts::LN_2)
    } else {
        f64::NAN
    };

    Ok(EvalReport {
        loss,
        perplexity,
        bpb,
        total_tokens,
        total_bytes,
        elapsed_secs,
    })
}

/// Load a saved checkpoint from `artifact_dir` and evaluate it on `dataset`.
pub fn run_eval(
    device: Device,
    artifact_dir: &str,
    model_name: &str,
    dataset: &TextDataset,
    batch_size: usize,
) -> anyhow::Result<EvalReport> {
    let config_path = format!("{artifact_dir}/config.json");
    let config = GPTConfig::load(&config_path)
        .with_context(|| format!("Failed to load model config from {config_path}"))?;
    config.validate()?;

    let model_path = format!("{artifact_dir}/{model_name}");
    println!("Loading model from {model_path} ...");
    let model = GPT::new(&config, &device)
        .try_load_file(&model_path)
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to load model checkpoint '{model_path}': {e:?}. Run `cargo run -- train` first."
            )
        })?;

    evaluate_dataset(&model, dataset, config.block_size, batch_size, &device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_dataset_computes_consistent_loss_ppl_and_bpb() {
        let device = Device::flex();
        let tokenizer = BpeTokenizer::new();
        let tokens = tokenizer.encode("To be, or not to be, that is the question.\n".repeat(4).as_str());
        let block_size = 8;
        let dataset = TextDataset::new(tokens, block_size);

        let config = GPTConfig {
            vocab_size: BpeTokenizer::PADDED_VOCAB_SIZE,
            n_layer: 1,
            n_head: 2,
            n_embd: 16,
            n_kv_head: None,
            block_size,
            dropout: 0.0,
            rope_theta: 10000.0,
            qk_norm: false,
            logit_softcap: 0.0,
        };
        let model = GPT::new(&config, &device);
        let report = evaluate_dataset(&model, &dataset, block_size, 2, &device).unwrap();

        assert!(report.loss.is_finite() && report.loss > 0.0);
        assert!((report.perplexity - report.loss.exp()).abs() < 1e-4);
        assert!(report.total_tokens == dataset.len() * block_size);
        assert!(report.total_bytes > 0);
        let expected_bpb =
            (report.loss * report.total_tokens as f64) / (report.total_bytes as f64 * std::f64::consts::LN_2);
        assert!((report.bpb - expected_bpb).abs() < 1e-6);
    }
}
