use burn::{
    prelude::*,
    tensor::{activation, ops::IndexingUpdateOp},
};

/// Sampling parameters for text generation.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f64,
    /// Keep only the top-k highest-probability tokens. 0 = disabled.
    pub top_k: usize,
    /// Nucleus sampling: keep the smallest set of tokens whose cumulative
    /// probability exceeds top_p. 1.0 = disabled.
    pub top_p: f64,
    /// Repetition penalty: > 1.0 penalizes tokens that have already appeared
    /// (divides positive logits by `penalty`, multiplies negative logits by `penalty`,
    /// once per seen token). 1.0 = disabled.
    pub repetition_penalty: f64,
    /// Optional upper bound on valid token IDs during sampling (e.g. `50257` when
    /// `vocab_size` is padded to `50304` for aligned CUDA matmuls). `None` = full vocab.
    pub vocab_limit: Option<usize>,
}

impl SamplingParams {
    /// Validate sampling hyperparameters before generation.
    pub fn validate(&self) -> anyhow::Result<()> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            anyhow::bail!("temperature must be finite and >= 0.0, got {}", self.temperature);
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            anyhow::bail!("top_p must be in (0.0, 1.0], got {}", self.top_p);
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty < 1.0 {
            anyhow::bail!(
                "repetition_penalty must be finite and >= 1.0, got {}",
                self.repetition_penalty
            );
        }
        Ok(())
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            vocab_limit: None,
        }
    }
}

/// Update an incremental `[batch, vocab]` seen-token counts tensor with new tokens `[batch, seq]`.
#[inline]
pub(crate) fn mark_seen_tokens(
    seen: Tensor<2>,
    tokens: &Tensor<2, Int>,
) -> Tensor<2> {
    let [batch, seq_len] = tokens.dims();
    if seq_len == 0 {
        return seen;
    }
    let ones = Tensor::<2>::ones([batch, seq_len], &seen.device());
    seen.scatter(1, tokens.clone(), ones, IndexingUpdateOp::Add)
}

/// Apply HuggingFace-style repetition penalty using a pre-accumulated `[batch, vocab]` seen counts tensor.
#[inline]
pub(crate) fn apply_repetition_penalty_seen(
    logits: &Tensor<2>,
    seen_counts: &Tensor<2>,
    penalty: f64,
) -> Tensor<2> {
    if penalty <= 1.0 + 1e-6 {
        return logits.clone();
    }
    let seen = seen_counts.clone().greater_elem(0.0);
    let pos_mask = logits.clone().greater_elem(0.0);
    // 0.0 * penalty == 0.0, so a single mask_where handles both positive and negative logits.
    let penalized = (logits.clone() * penalty).mask_where(pos_mask, logits.clone() / penalty);
    logits.clone().mask_where(seen, penalized)
}

/// Apply HuggingFace-style repetition penalty to logits for tokens present in `prev_tokens`:
/// positive logits are divided by `penalty`, negative logits are multiplied by `penalty`.
#[inline]
#[allow(dead_code)]
pub(crate) fn apply_repetition_penalty(
    logits: &Tensor<2>,
    prev_tokens: &Tensor<2, Int>,
    penalty: f64,
) -> Tensor<2> {
    if penalty <= 1.0 + 1e-6 {
        return logits.clone();
    }
    let [batch, vocab] = logits.dims();
    let [_, seq_len] = prev_tokens.dims();
    if seq_len == 0 {
        return logits.clone();
    }
    let seen_counts = mark_seen_tokens(
        Tensor::<2>::zeros([batch, vocab], &logits.device()),
        prev_tokens,
    );
    apply_repetition_penalty_seen(logits, &seen_counts, penalty)
}

/// Sample next token from logits `[batch, vocab]` entirely on the tensor device,
/// returning a `[batch, 1]` int tensor.
/// `seen_counts` is an optional incremental `[batch, vocab]` seen-token tensor for O(1) repetition penalty.
#[inline]
pub(crate) fn sample_token(
    logits: &Tensor<2>,
    params: &SamplingParams,
    seen_counts: Option<&Tensor<2>>,
) -> Tensor<2, Int> {
    let [_, vocab] = logits.dims();

    // Cast logits to F32 for numerically stable softmax/cumsum and to avoid
    // Burn 0.22.1's `sort_descending_with_indices` DTypeMismatch on Flex32.
    let logits = logits.clone().cast(burn::tensor::FloatDType::F32);
    let logits = match seen_counts {
        Some(seen) => apply_repetition_penalty_seen(&logits, seen, params.repetition_penalty),
        None => logits,
    };

    if params.temperature < 1e-6 {
        return logits.argmax(1);
    }

    let logits = logits / params.temperature;
    let use_top_k = params.top_k > 0 && params.top_k < vocab;
    let use_top_p = params.top_p < 1.0;

    let (candidate_logits, candidate_indices) = if use_top_k {
        let (values, indices) = logits.topk_with_indices(params.top_k, 1);
        if use_top_p {
            // Ensure descending order on every backend before nucleus cumsum.
            let (sorted_vals, sub_idx) = values.sort_descending_with_indices(1);
            let sorted_indices = indices.gather(1, sub_idx);
            (sorted_vals, Some(sorted_indices))
        } else {
            (values, Some(indices))
        }
    } else if use_top_p {
        let (values, indices) = logits.sort_descending_with_indices(1);
        (values, Some(indices))
    } else {
        (logits, None)
    };

    let mut probs = activation::softmax(candidate_logits, 1);

    if use_top_p {
        // Keep the top-1 token (prior_cum == 0.0 < top_p) and every token up through
        // the first token that crosses top_p. Clamp top_p > 0 so top-1 is never masked.
        let top_p = params.top_p.max(1e-9);
        let cumprobs = probs.clone().cumsum(1);
        let prior_cum = cumprobs - probs.clone();
        let exclude = prior_cum.greater_equal_elem(top_p);
        probs = probs.mask_fill(exclude, 0.0);
    }

    // Burn's `categorical` normalizes along `dim` internally.
    let sampled = probs.categorical(1);
    match candidate_indices {
        Some(indices) => indices.gather(1, sampled),
        None => sampled,
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
            indices.select_nth_unstable_by(bound, |&a, &b| probs[b].total_cmp(&probs[a]));
            indices[..bound].sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
        } else {
            indices.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
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
            indices.select_nth_unstable_by(bound, |&a, &b| probs[b].total_cmp(&probs[a]));
            indices[..bound].sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));
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

    #[test]
    fn repetition_penalty_lowers_both_positive_and_negative_logits() {
        use burn::prelude::*;

        let device = Device::flex();
        let logits = Tensor::<2>::from_data([[4.0_f32, -4.0, 2.0, -2.0]], &device);
        let prev_tokens = Tensor::<2, Int>::from_data([[0_i32, 1]], &device);
        let penalized = super::apply_repetition_penalty(&logits, &prev_tokens, 2.0);
        let data = penalized.to_data();
        let vals = data.as_slice::<f32>().unwrap();
        assert!((vals[0] - 2.0).abs() < 1e-5, "pos seen: {}", vals[0]);
        assert!((vals[1] - (-8.0)).abs() < 1e-5, "neg seen: {}", vals[1]);
        assert!((vals[2] - 2.0).abs() < 1e-5, "pos unseen: {}", vals[2]);
        assert!((vals[3] - (-2.0)).abs() < 1e-5, "neg unseen: {}", vals[3]);
    }

    #[test]
    fn filter_top_k_p_handles_nan_without_panicking() {
        let probs = vec![0.5, f32::NAN, 0.3, 0.2];
        let result = filter_top_k_p(&probs, 2, 0.9);
        assert_eq!(result.len(), probs.len());
    }

    #[test]
    fn sample_token_on_device_top_p_and_top_k_respect_nucleus() {
        use burn::prelude::*;

        let device = Device::flex();
        // Token 1 has overwhelming probability (> 0.999), so top_p=0.5 must always pick token 1.
        let logits = Tensor::<2>::from_data(
            [[0.0_f32, 20.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 20.0, 0.0]],
            &device,
        );
        for (top_k, top_p) in [(0, 0.5), (3, 0.5), (2, 1.0), (0, 0.0)] {
            let params = super::SamplingParams {
                temperature: 1.0,
                top_k,
                top_p,
                repetition_penalty: 1.0,
                vocab_limit: None,
            };
            let sampled = super::sample_token(&logits, &params, None);
            assert_eq!(sampled.dims(), [2, 1]);
            let data = sampled.into_data();
            let ids = data.as_slice::<i32>().unwrap();
            assert_eq!(ids, &[1, 3], "top_k={top_k}, top_p={top_p}");
        }
    }
}
