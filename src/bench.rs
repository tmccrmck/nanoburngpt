use std::time::Instant;

use burn::{optim::GradientsParams, prelude::*};

use crate::{
    data::TextGenerationBatch,
    model::{GPT, GPTConfig, SamplingParams},
    train::{OptimizerChoice, build_optimizer},
};

/// Measured throughput results for a local GPU/CPU benchmark run.
#[derive(Clone, Copy, Debug)]
pub struct BenchReport {
    pub eval_ms_per_step: f64,
    pub eval_tok_per_sec: f64,
    pub train_ms_per_step: f64,
    pub train_tok_per_sec: f64,
    pub decode_ms_per_tok: f64,
    pub decode_tok_per_sec: f64,
}

/// Benchmark eval (forward + loss), train (`forward_loss + backward + optim.step`),
/// and KV-cache decode (`generate_tokens`) on `device`.
#[allow(clippy::too_many_arguments)]
pub fn run_bench(
    device: Device,
    gpt_config: &GPTConfig,
    optimizer_choice: OptimizerChoice,
    batch_size: usize,
    warmup_steps: usize,
    timed_steps: usize,
    decode_tokens: usize,
    ce_chunk_size: usize,
) -> anyhow::Result<BenchReport> {
    gpt_config.validate()?;
    let batch_size = batch_size.max(1);
    let timed_steps = timed_steps.max(1);
    let seq_len = gpt_config.block_size;
    let tokens_per_step = (batch_size * seq_len) as f64;

    // 1. Eval (forward + cross-entropy loss, no autodiff)
    let eval_model = GPT::new(gpt_config, &device);
    let inputs = Tensor::<2, Int>::zeros([batch_size, seq_len], &device);
    let targets = Tensor::<2, Int>::zeros([batch_size, seq_len], &device);
    let eval_batch = TextGenerationBatch {
        inputs,
        targets,
        loss_scale: 1.0,
    };

    for _ in 0..warmup_steps {
        let _: f32 = eval_model
            .forward_loss_chunked(eval_batch.clone(), ce_chunk_size)
            .loss
            .into_scalar();
    }
    let t0 = Instant::now();
    for _ in 0..timed_steps {
        let _: f32 = eval_model
            .forward_loss_chunked(eval_batch.clone(), ce_chunk_size)
            .loss
            .into_scalar();
    }
    let eval_secs = t0.elapsed().as_secs_f64().max(1e-9);
    let eval_ms_per_step = (eval_secs / timed_steps as f64) * 1000.0;
    let eval_tok_per_sec = (tokens_per_step * timed_steps as f64) / eval_secs;

    // 2. KV-cache decode (non-streaming, 100% on-device until final sync)
    let prompt_len = 16.min(seq_len.max(1));
    let decode_tokens = decode_tokens.max(1);
    let prompt = Tensor::<2, Int>::zeros([1, prompt_len], &device);
    let sampling = SamplingParams {
        temperature: 0.8,
        top_k: 40,
        top_p: 0.95,
        repetition_penalty: 1.1,
        vocab_limit: None,
    };
    // Warmup decode kernels
    let warm = eval_model.generate_tokens(prompt.clone(), 4, &sampling, seq_len);
    let _: i32 = warm.slice([0..1, 0..1]).into_scalar();

    let t0 = Instant::now();
    let out = eval_model.generate_tokens(prompt, decode_tokens, &sampling, seq_len);
    // Force full GPU sync on final generated token
    let out_len = out.dims()[1];
    let _: i32 = out.slice([0..1, (out_len - 1)..out_len]).into_scalar();
    let decode_secs = t0.elapsed().as_secs_f64().max(1e-9);
    let decode_ms_per_tok = (decode_secs / decode_tokens as f64) * 1000.0;
    let decode_tok_per_sec = decode_tokens as f64 / decode_secs;

    drop(eval_model);

    // 3. Train step (forward + backward + optimizer step)
    let ad_device = device.autodiff();
    let mut train_model = GPT::new(gpt_config, &ad_device);
    let mut optim = build_optimizer(optimizer_choice);
    let train_batch = TextGenerationBatch {
        inputs: Tensor::<2, Int>::zeros([batch_size, seq_len], &ad_device),
        targets: Tensor::<2, Int>::zeros([batch_size, seq_len], &ad_device),
        loss_scale: 1.0,
    };

    for _ in 0..warmup_steps {
        let out = train_model.forward_loss_chunked(train_batch.clone(), ce_chunk_size);
        let grads = GradientsParams::from_grads(out.loss.backward(), &train_model);
        train_model = optim.step(3e-4, train_model, grads);
    }
    let _: f32 = train_model
        .forward_loss_chunked(train_batch.clone(), ce_chunk_size)
        .loss
        .into_scalar();

    let t0 = Instant::now();
    for _ in 0..timed_steps {
        let out = train_model.forward_loss_chunked(train_batch.clone(), ce_chunk_size);
        let grads = GradientsParams::from_grads(out.loss.backward(), &train_model);
        train_model = optim.step(3e-4, train_model, grads);
    }
    // Sync final optimizer update before stopping timer
    let _: f32 = train_model
        .forward_loss_chunked(train_batch, ce_chunk_size)
        .loss
        .into_scalar();
    let train_secs = t0.elapsed().as_secs_f64().max(1e-9);
    // Subtract 1 eval pass amortized across timed_steps for clean train step timing
    let train_only_secs = (train_secs - (eval_secs / timed_steps as f64)).max(1e-9);
    let train_ms_per_step = (train_only_secs / timed_steps as f64) * 1000.0;
    let train_tok_per_sec = (tokens_per_step * timed_steps as f64) / train_only_secs;

    Ok(BenchReport {
        eval_ms_per_step,
        eval_tok_per_sec,
        train_ms_per_step,
        train_tok_per_sec,
        decode_ms_per_tok,
        decode_tok_per_sec,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_runs_on_cpu_and_returns_positive_throughput() {
        let config = GPTConfig {
            vocab_size: 64,
            n_layer: 1,
            n_head: 2,
            n_embd: 16,
            n_kv_head: None,
            block_size: 16,
            dropout: 0.0,
            rope_theta: 10000.0,
            qk_norm: false,
            logit_softcap: 0.0,
        };
        let report = run_bench(
            Device::flex(),
            &config,
            OptimizerChoice::AdamW,
            2,
            1,
            2,
            4,
            16,
        )
        .unwrap();
        assert!(report.eval_tok_per_sec > 0.0);
        assert!(report.train_tok_per_sec > 0.0);
        assert!(report.decode_tok_per_sec > 0.0);
    }
}
