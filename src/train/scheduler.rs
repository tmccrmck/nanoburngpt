use burn::optim::lr_scheduler::{LrScheduler, LrSchedulerRecord};

/// Linear warmup for `warmup_iters` steps, then cosine decay to `min_lr`
/// over the remaining `total_iters - warmup_iters` steps (matches nanoGPT's `get_lr()`).
#[derive(Clone, Debug)]
pub struct WarmupCosineScheduler {
    warmup_iters: usize,
    total_iters: usize,
    max_lr: f64,
    min_lr: f64,
    current_iter: usize,
}

impl WarmupCosineScheduler {
    pub fn new(max_lr: f64, min_lr: f64, warmup_iters: usize, total_iters: usize) -> Self {
        Self {
            warmup_iters,
            total_iters,
            max_lr,
            min_lr,
            current_iter: usize::MAX,
        }
    }

    fn lr_at(&self, iter: usize) -> f64 {
        let effective_warmup = self.warmup_iters.min(self.total_iters.saturating_sub(1));
        if effective_warmup > 0 && iter < effective_warmup {
            // Linear ramp: 0 → max_lr over effective_warmup steps
            self.max_lr * (iter + 1) as f64 / effective_warmup as f64
        } else if iter >= self.total_iters {
            // Hold at min_lr after the decay is complete
            self.min_lr
        } else {
            // Cosine decay: max_lr → min_lr
            let decay_iters = (self.total_iters - effective_warmup).max(1);
            let progress = (iter - effective_warmup) as f64 / decay_iters as f64;
            let coeff = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
            self.min_lr + coeff * (self.max_lr - self.min_lr)
        }
    }
}

impl LrScheduler for WarmupCosineScheduler {
    fn step(&mut self) -> f64 {
        // wrapping_add so first step() computes lr_at(0), matching
        // Burn's CosineAnnealing convention (current_iter starts at MAX).
        self.current_iter = self.current_iter.wrapping_add(1);
        self.lr_at(self.current_iter)
    }

    fn to_record(&self) -> LrSchedulerRecord {
        LrSchedulerRecord::new().with_scalar("current_iter", self.current_iter as u64)
    }

    fn load_record(&mut self, record: LrSchedulerRecord) {
        if let Some(current_iter) = record.scalar::<u64>("current_iter") {
            self.current_iter = current_iter as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scheduler(max_lr: f64, min_lr: f64, warmup: usize, total: usize) -> WarmupCosineScheduler {
        WarmupCosineScheduler::new(max_lr, min_lr, warmup, total)
    }

    fn steps(s: &mut WarmupCosineScheduler, n: usize) -> Vec<f64> {
        (0..n).map(|_| s.step()).collect()
    }

    #[test]
    fn warmup_ramps_linearly() {
        let mut s = scheduler(1.0, 0.1, 4, 10);
        let lrs = steps(&mut s, 4);
        assert!((lrs[0] - 0.25).abs() < 1e-10);
        assert!((lrs[1] - 0.50).abs() < 1e-10);
        assert!((lrs[2] - 0.75).abs() < 1e-10);
        assert!((lrs[3] - 1.00).abs() < 1e-10);
    }

    #[test]
    fn cosine_decay_midpoint() {
        let max_lr = 1.0_f64;
        let min_lr = 0.1_f64;
        let total = 100;
        let mid = total / 2;
        let mut s = scheduler(max_lr, min_lr, 0, total);
        let lrs = steps(&mut s, mid + 1);
        let expected = (max_lr + min_lr) / 2.0;
        assert!((lrs[mid] - expected).abs() < 1e-10, "got {}", lrs[mid]);
    }

    #[test]
    fn holds_min_lr_after_total_iters() {
        let mut s = scheduler(1.0, 0.1, 0, 5);
        let lrs = steps(&mut s, 10);
        for lr in &lrs[5..] {
            assert!((*lr - 0.1).abs() < 1e-10, "expected min_lr, got {lr}");
        }
    }

    #[test]
    fn no_warmup_starts_at_cosine_start() {
        let mut s = scheduler(1.0, 0.1, 0, 10);
        let lr = s.step();
        assert!((lr - 1.0).abs() < 1e-10, "got {lr}");
    }

    #[test]
    fn warmup_cosine_transition_is_continuous() {
        let max_lr = 1.0_f64;
        let min_lr = 0.1;
        let warmup = 5;
        let total = 20;
        let mut s = scheduler(max_lr, min_lr, warmup, total);
        let lrs = steps(&mut s, warmup + 1);
        assert!((lrs[warmup - 1] - max_lr).abs() < 1e-10);
        assert!((lrs[warmup] - max_lr).abs() < 1e-10);
    }

    #[test]
    fn cosine_decays_monotonically() {
        let mut s = scheduler(1.0, 0.1, 0, 20);
        let lrs = steps(&mut s, 20);
        for w in lrs.windows(2) {
            assert!(w[0] >= w[1], "LR increased: {} -> {}", w[0], w[1]);
        }
    }

    #[test]
    fn warmup_exceeding_total_iters_is_clamped() {
        let mut s = scheduler(1.0, 0.1, 100, 5);
        let lrs = steps(&mut s, 6);
        // Effective warmup = 4: ramps to 1.0 at step 3, cosine starts at 1.0 at step 4, min_lr at step 5.
        assert!((lrs[3] - 1.0).abs() < 1e-10, "got {}", lrs[3]);
        assert!((lrs[4] - 1.0).abs() < 1e-10, "got {}", lrs[4]);
        assert!((lrs[5] - 0.1).abs() < 1e-10, "got {}", lrs[5]);
    }
}
