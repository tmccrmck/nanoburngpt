use burn::{
    prelude::*,
    tensor::{activation, ops::IndexingUpdateOp},
};
use rand::distr::{Distribution, weighted::WeightedIndex};

/// Sampling parameters for text generation.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f64,
    /// Keep only the top-k highest-probability tokens. 0 = disabled.
    pub top_k: usize,
    /// Nucleus sampling: keep the smallest set of tokens whose cumulative
    /// probability exceeds top_p. 1.0 = disabled.
    pub top_p: f64,
    /// Repetition penalty: > 1.0 penalizes tokens that have already appeared.
    /// Applied as logit / penalty for each previous occurrence.
    /// 1.0 = disabled.
    pub repetition_penalty: f64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
        }
    }
}

/// Sample next token from logits [batch, vocab], returns [batch, 1] int tensor.
/// `prev_tokens` [batch, seq_len] contains all tokens generated so far (for repetition penalty).
#[inline]
pub(crate) fn sample_token(
    logits: &Tensor<2>,
    params: &SamplingParams,
    prev_tokens: &Tensor<2, Int>,
) -> Tensor<2, Int> {
    let [batch, vocab] = logits.dims();
    let device = logits.device();

    let logits = if params.repetition_penalty > 1.0 + 1e-6 {
        let [_, seq_len] = prev_tokens.dims();
        let ones = Tensor::<2>::ones([batch, seq_len], &device);
        let counts = Tensor::<2>::zeros([batch, vocab], &device)
            .scatter(1, prev_tokens.clone(), ones, IndexingUpdateOp::Add);
        let penalty = 1.0 / params.repetition_penalty;
        let mask = Tensor::<2>::ones([batch, vocab], &device)
            - (1.0 - penalty) * counts.clamp(0.0, 1.0);
        logits.clone() * mask
    } else {
        logits.clone()
    };

    if params.temperature < 1e-6 {
        logits.clone().argmax(1).unsqueeze::<2>()
    } else {
        let logits = logits.clone() / params.temperature;

        // Pure top-k (no top-p): fully on GPU
        if params.top_k > 0 && params.top_p >= 1.0 && params.top_k < vocab {
            let (values, indices) = logits.topk_with_indices(params.top_k, 1);
            let probs = activation::softmax(values, 1);
            let sampled = probs.categorical(1);
            return indices.gather(1, sampled);
        }

        // No filtering: fully on GPU
        if params.top_k == 0 && params.top_p >= 1.0 {
            let probs = activation::softmax(logits, 1);
            return probs.categorical(1);
        }

        // Top-p or combined filtering: keep CPU path
        let probs = activation::softmax(logits, 1);
        let probs_data = probs.into_data();
        let probs_f32 = probs_data.as_slice::<f32>().expect("f32 probs");

        let mut rng = rand::rng();
        let tokens: Vec<i32> = (0..batch)
            .map(|b| {
                let row = &probs_f32[b * vocab..(b + 1) * vocab];
                let filtered = filter_top_k_p(row, params.top_k, params.top_p);
                let dist = WeightedIndex::new(&filtered).expect("valid weights");
                dist.sample(&mut rng) as i32
            })
            .collect();

        Tensor::<1, Int>::from_ints(tokens.as_slice(), &device).unsqueeze::<2>()
    }
}

/// Apply top-k and top-p filtering to a probability distribution, returning
/// a new distribution with excluded tokens zeroed out and probabilities renormalized.
#[inline]
pub fn filter_top_k_p(probs: &[f32], top_k: usize, top_p: f64) -> Vec<f32> {
    let vocab = probs.len();

    let mut indices: Vec<usize> = (0..vocab).collect();

    // Determine an upper bound on how many tokens we need sorted.
    // If top-k is active, we never need more than top_k entries.
    let bound = if top_k > 0 && top_k < vocab { top_k } else { vocab };

    let cutoff = if top_p < 1.0 {
        // Partial sort up to bound, then sort just the first bound elements.
        if bound < vocab {
            indices.select_nth_unstable_by(bound, |&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            indices[..bound].sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        } else {
            indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        }

        let mut cumsum = 0.0_f64;
        let mut cutoff = bound;
        for (i, &idx) in indices.iter().enumerate() {
            cumsum += probs[idx] as f64;
            if cumsum >= top_p {
                cutoff = i + 1;
                break;
            }
        }
        cutoff
    } else {
        // Pure top-k: partial sort only.
        if bound < vocab {
            indices.select_nth_unstable_by(bound, |&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
            indices[..bound].sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        }
        bound
    };

    // Zero out tokens outside the cutoff
    let mut result = vec![0.0f32; vocab];
    let mut total = 0.0f32;
    for &idx in &indices[..cutoff] {
        result[idx] = probs[idx];
        total += probs[idx];
    }

    // Renormalize
    if total > 0.0 {
        for p in &mut result {
            *p /= total;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::filter_top_k_p;

    #[test]
    fn top_k_keeps_only_k_tokens() {
        let probs = vec![0.1, 0.5, 0.2, 0.15, 0.05];
        let result = filter_top_k_p(&probs, 2, 1.0);
        assert_eq!(result[0], 0.0);
        assert!(result[1] > 0.0);
        assert!(result[2] > 0.0);
        assert_eq!(result[3], 0.0);
        assert_eq!(result[4], 0.0);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    #[test]
    fn top_p_nucleus_filtering() {
        let probs = vec![0.15, 0.5, 0.05, 0.3];
        let result = filter_top_k_p(&probs, 0, 0.8);
        assert_eq!(result[0], 0.0);
        assert!(result[1] > 0.0);
        assert_eq!(result[2], 0.0);
        assert!(result[3] > 0.0);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    #[test]
    fn top_k_and_top_p_combined() {
        let probs = vec![0.15, 0.5, 0.05, 0.3];
        let result = filter_top_k_p(&probs, 3, 0.6);
        assert_eq!(result[0], 0.0);
        assert!(result[1] > 0.0);
        assert_eq!(result[2], 0.0);
        assert!(result[3] > 0.0);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn no_filtering_when_disabled() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let result = filter_top_k_p(&probs, 0, 1.0);
        for (i, &p) in result.iter().enumerate() {
            assert!((p - 0.25).abs() < 1e-5, "index {i}: {p}");
        }
    }

    #[test]
    fn top_k_1_keeps_only_argmax() {
        let probs = vec![0.1, 0.6, 0.2, 0.1];
        let result = filter_top_k_p(&probs, 1, 1.0);
        assert_eq!(result[0], 0.0);
        assert!((result[1] - 1.0).abs() < 1e-5, "expected 1.0, got {}", result[1]);
        assert_eq!(result[2], 0.0);
        assert_eq!(result[3], 0.0);
    }
}
