/// Benchmarks for the hot paths in nanoburngpt.
///
/// Uses the Flex CPU backend so no GPU is required.  These benchmarks are
/// fast enough to run in CI and give stable numbers useful for spotting
/// regressions when changing model internals.
///
/// Run with:
///   cargo bench
///   cargo bench -- forward          # filter by name
///   cargo bench -- --save-baseline main
///   cargo bench -- --baseline main  # compare against saved baseline
use burn::prelude::*;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nanoburngpt::model::{GPT, GPTConfig, SamplingParams};

fn make_config(n_layer: usize, n_head: usize, n_embd: usize, block_size: usize) -> GPTConfig {
    GPTConfig {
        vocab_size: 256,
        n_layer,
        n_head,
        n_embd,
        n_kv_head: None,
        block_size,
        dropout: 0.0,
        rope_theta: 10000.0,
    }
}

fn test_device() -> Device {
    Device::flex()
}

/// Benchmark the full GPT forward pass across different sequence lengths.
/// Uses a tiny model so the per-step work is roughly fixed and seq_len is the variable.
fn bench_forward(c: &mut Criterion) {
    let device = test_device();
    let config = make_config(2, 4, 64, 256);
    let gpt = GPT::new(&config, &device);

    let mut group = c.benchmark_group("forward");
    for &seq_len in &[32usize, 64, 128, 256] {
        let input = Tensor::<2, Int>::zeros([1, seq_len], &device);
        group.bench_with_input(BenchmarkId::from_parameter(seq_len), &seq_len, |b, _| {
            b.iter(|| gpt.forward(input.clone()))
        });
    }
    group.finish();
}

/// Benchmark a single KV-cache decode step (prefill then one token decode).
/// This represents the steady-state cost per generated token during inference.
fn bench_decode_step(c: &mut Criterion) {
    let device = test_device();
    let config = make_config(2, 4, 64, 256);
    let gpt = GPT::new(&config, &device);

    let mut group = c.benchmark_group("decode_step");
    for &prompt_len in &[8usize, 32, 64, 128] {
        let prompt = Tensor::<2, Int>::zeros([1, prompt_len], &device);
        // Pre-warm: build the KV cache up to prompt_len so the benchmark only
        // measures the marginal cost of decoding one more token.
        let (_, cache) = gpt.forward_cached(prompt, None);
        let next_token = Tensor::<2, Int>::zeros([1, 1], &device);

        group.bench_with_input(
            BenchmarkId::new("prompt_len", prompt_len),
            &prompt_len,
            |b, _| {
                b.iter_batched(
                    || cache.clone(),
                    |unique_cache| {
                        let (logits, _new_cache) =
                            gpt.forward_cached(next_token.clone(), Some(unique_cache));
                        logits
                    },
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }
    group.finish();
}

/// Benchmark sampling filter (top-k / top-p) at realistic vocab sizes.
/// This runs on CPU regardless of backend, so it's also a pure-Rust benchmark.
fn bench_filter_sampling(c: &mut Criterion) {
    // Build a fake probability distribution over the real vocab size
    let vocab = 50257usize;
    let raw: Vec<f32> = (0..vocab).map(|i| (i as f32 + 1.0).recip()).collect();
    let total: f32 = raw.iter().sum();
    let probs: Vec<f32> = raw.into_iter().map(|p| p / total).collect();

    let mut group = c.benchmark_group("filter_top_k_p");
    // top-k only
    group.bench_function("top_k=50", |b| {
        b.iter(|| nanoburngpt::model::filter_top_k_p(&probs, 50, 1.0))
    });
    // top-p only
    group.bench_function("top_p=0.9", |b| {
        b.iter(|| nanoburngpt::model::filter_top_k_p(&probs, 0, 0.9))
    });
    // combined
    group.bench_function("top_k=50_top_p=0.9", |b| {
        b.iter(|| nanoburngpt::model::filter_top_k_p(&probs, 50, 0.9))
    });
    group.finish();
}

/// Benchmark generate() end-to-end (prefill + N decode steps, greedy sampling).
fn bench_generate(c: &mut Criterion) {
    let device = test_device();
    let config = make_config(2, 4, 64, 256);
    let gpt = GPT::new(&config, &device);
    let sampling = SamplingParams {
        temperature: 0.0, // greedy — deterministic, no randomness overhead
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
    };

    let mut group = c.benchmark_group("generate");
    for &max_new in &[8usize, 32, 64] {
        let prompt = Tensor::<2, Int>::zeros([1, 8], &device);
        group.bench_with_input(
            BenchmarkId::new("max_new_tokens", max_new),
            &max_new,
            |b, &n| {
                b.iter(|| {
                    gpt.generate(prompt.clone(), n, &sampling, config.block_size, |_| {})
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_forward,
    bench_decode_step,
    bench_filter_sampling,
    bench_generate,
);
criterion_main!(benches);
