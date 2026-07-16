//! Token-level perplexity, accumulated the only way that is arithmetically
//! correct.
//!
//! # Why not burn's `PerplexityMetric`
//!
//! burn's version is correct -- it accumulates NLL and token counts rather than
//! averaging batch perplexities, which is the subtle part -- but its `Input`
//! carries the `[batch*seq, vocab]` logits and it recomputes `log_softmax` on
//! the CPU on every update. [`crate::train::output`] explains why that is
//! unaffordable here. This metric takes the two scalars the loss already
//! computed and does nothing but add them up.
//!
//! The accumulation rule is the part worth stating explicitly, because getting
//! it wrong produces a plausible number rather than an obvious error:
//! perplexity is `exp(total_nll / total_tokens)`, **not** the mean of per-batch
//! perplexities. Those agree only when every batch scores an equal number of
//! tokens, which fails for the last partial batch of an epoch and fails
//! throughout under strided evaluation. Averaging also biases the result upward
//! by Jensen's inequality, so the error flatters nothing -- it just misreports.

use core::marker::PhantomData;

use burn::{
    prelude::Backend,
    tensor::{ElementConversion, Tensor},
    train::metric::{
        format_float, state::FormatOptions, Metric, MetricAttributes, MetricMetadata, MetricName,
        Numeric, NumericAttributes, NumericEntry, SerializedEntry,
    },
};

/// What [`TokenPerplexityMetric`] consumes: one batch's NLL sum and its token
/// count, both already reduced to scalars on the device.
#[derive(Debug)]
pub struct TokenPerplexityInput<B: Backend> {
    sum_nll: Tensor<B, 1>,
    n_tokens: Tensor<B, 1>,
}

impl<B: Backend> TokenPerplexityInput<B> {
    pub fn new(sum_nll: Tensor<B, 1>, n_tokens: Tensor<B, 1>) -> Self {
        Self { sum_nll, n_tokens }
    }
}

/// What [`GradRmsMetric`] consumes: one batch's gradient RMS, already reduced to
/// a scalar on the device.
#[derive(Debug)]
pub struct GradRmsInput<B: Backend> {
    rms: Tensor<B, 1>,
}

impl<B: Backend> GradRmsInput<B> {
    pub fn new(rms: Tensor<B, 1>) -> Self {
        Self { rms }
    }
}

/// Root-mean-square of the gradient over every parameter, per batch.
///
/// # Why RMS and not the norm
///
/// Because the number this has to be compared against is AdamW's `epsilon`, and
/// epsilon is added to a *per-element* quantity: the update is
/// `m / (sqrt(v) + eps)`, and `sqrt(v)` is an exponential average of squared
/// per-element gradients -- i.e. an RMS. A global norm would grow with the
/// square root of the parameter count and be comparable to nothing.
///
/// Wortsman et al. 2023 (arXiv:2309.14322) §3.4 measure exactly this quantity
/// collapsing below epsilon partway through training, at which point epsilon,
/// not the gradient, sets the update size and learning stalls. That measurement
/// is the entire argument for this project's `epsilon = 1e-15`
/// ([`TrainConfig::epsilon`](crate::train::TrainConfig::epsilon)), and without
/// this metric that setting would be an article of faith: a claim about a
/// number nobody was recording. If `GradRms` never approaches 1e-8, the change
/// bought nothing and should be reverted.
///
/// # Why the per-batch value is what gets logged
///
/// A running mean is the wrong summary for this. The failure modes worth
/// catching -- a spike that precedes divergence, a collapse toward epsilon --
/// are both *departures* from the running mean, and averaging is precisely the
/// operation that hides them. So [`Metric::update`] serializes the batch's own
/// value; [`Numeric::running_value`] still offers the epoch mean for the
/// dashboard.
#[derive(Clone)]
pub struct GradRmsMetric<B: Backend> {
    name: MetricName,
    last: f64,
    sum: f64,
    batches: usize,
    _b: PhantomData<B>,
}

impl<B: Backend> Default for GradRmsMetric<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> GradRmsMetric<B> {
    pub fn new() -> Self {
        Self {
            name: MetricName::new("GradRms".to_string()),
            last: 0.0,
            sum: 0.0,
            batches: 0,
            _b: PhantomData,
        }
    }

    fn mean(&self) -> f64 {
        if self.batches > 0 {
            self.sum / self.batches as f64
        } else {
            0.0
        }
    }
}

impl<B: Backend> Metric for GradRmsMetric<B> {
    type Input = GradRmsInput<B>;

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn description(&self) -> Option<String> {
        Some("sqrt(mean(g^2)) over all parameter gradients, per batch".to_string())
    }

    fn attributes(&self) -> MetricAttributes {
        // Neither direction is "better": this is a diagnostic, not an objective.
        // `higher_is_better` has to say something, and false is the less
        // misleading of the two -- a gradient growing without bound is the
        // failure this exists to catch.
        NumericAttributes {
            unit: None,
            higher_is_better: false,
        }
        .into()
    }

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let rms = input.rms.clone().into_scalar().elem::<f64>();

        self.last = rms;
        self.sum += rms;
        self.batches += 1;

        // Scientific notation, and with reason: this number is expected to span
        // orders of magnitude and to be read against 1e-15. `format_float` with
        // any fixed precision renders a grad RMS of 3e-9 as "0.00".
        let formatted = format!("epoch {:.3e} - batch {rms:.3e}", self.mean());
        SerializedEntry::new(formatted, NumericEntry::Value(rms).serialize())
    }

    fn clear(&mut self) {
        self.last = 0.0;
        self.sum = 0.0;
        self.batches = 0;
    }
}

impl<B: Backend> Numeric for GradRmsMetric<B> {
    fn value(&self) -> NumericEntry {
        NumericEntry::Value(self.last)
    }

    fn running_value(&self) -> NumericEntry {
        NumericEntry::Value(self.mean())
    }
}

/// Perplexity per *token*, as `exp(sum_nll / n_tokens)`.
///
/// Per token, not per word or per byte: this is the training-time dashboard
/// number, and it is tokenizer-dependent, so it is **not** comparable against
/// GPT-2. The comparable metrics -- word-level perplexity and bits-per-byte --
/// are the evaluation suite's job, since they need the shard's `n_words` and
/// `n_bytes` and a strided pass the training loop does not perform. What this
/// metric is for is watching a run: it turns loss into a number whose scale is
/// familiar, and it bounds the model from above by `vocab_size` at
/// initialization.
#[derive(Clone)]
pub struct TokenPerplexityMetric<B: Backend> {
    name: MetricName,
    sum_nll: f64,
    total_tokens: usize,
    _b: PhantomData<B>,
}

impl<B: Backend> Default for TokenPerplexityMetric<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> TokenPerplexityMetric<B> {
    pub fn new() -> Self {
        Self {
            name: MetricName::new("Perplexity".to_string()),
            sum_nll: 0.0,
            total_tokens: 0,
            _b: PhantomData,
        }
    }

    fn perplexity(sum_nll: f64, tokens: usize) -> f64 {
        if tokens > 0 {
            (sum_nll / tokens as f64).exp()
        } else {
            f64::INFINITY
        }
    }

    fn entry(&self) -> NumericEntry {
        // Aggregated rather than Value so that burn's cross-epoch aggregation
        // weights each entry by its token count instead of treating a
        // ten-token batch as equal to a ten-thousand-token one.
        NumericEntry::Aggregated {
            aggregated_value: Self::perplexity(self.sum_nll, self.total_tokens),
            count: self.total_tokens,
        }
    }
}

impl<B: Backend> Metric for TokenPerplexityMetric<B> {
    type Input = TokenPerplexityInput<B>;

    fn name(&self) -> MetricName {
        self.name.clone()
    }

    fn description(&self) -> Option<String> {
        Some("exp(total NLL / total scored tokens), in this model's own token space".to_string())
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: false,
        }
        .into()
    }

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> SerializedEntry {
        let batch_nll = input.sum_nll.clone().into_scalar().elem::<f64>();
        let batch_tokens = input.n_tokens.clone().into_scalar().elem::<f64>() as usize;

        self.sum_nll += batch_nll;
        self.total_tokens += batch_tokens;

        let batch = Self::perplexity(batch_nll, batch_tokens);
        let epoch = Self::perplexity(self.sum_nll, self.total_tokens);

        let format = FormatOptions::new(self.name()).precision(2);
        let precision = format.precision_value().unwrap_or(2);
        let formatted = format!(
            "epoch {} - batch {}",
            format_float(epoch, precision),
            format_float(batch, precision)
        );

        SerializedEntry::new(formatted, self.entry().serialize())
    }

    fn clear(&mut self) {
        self.sum_nll = 0.0;
        self.total_tokens = 0;
    }
}

impl<B: Backend> Numeric for TokenPerplexityMetric<B> {
    fn value(&self) -> NumericEntry {
        self.entry()
    }

    fn running_value(&self) -> NumericEntry {
        self.entry()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::TestBackend;
    use burn::data::dataloader::Progress;

    fn input(sum_nll: f32, n_tokens: f32) -> TokenPerplexityInput<TestBackend> {
        let d = Default::default();
        TokenPerplexityInput::new(
            Tensor::from_data([sum_nll], &d),
            Tensor::from_data([n_tokens], &d),
        )
    }

    /// burn's own `MetricMetadata::fake()` is `#[cfg(test)]`, so it exists only
    /// within burn-train's own test builds. This metric ignores the metadata
    /// anyway.
    fn metadata() -> MetricMetadata {
        let progress = Progress {
            items_processed: 1,
            items_total: 1,
        };
        MetricMetadata {
            progress: progress.clone(),
            global_progress: progress,
            iteration: Some(0),
            lr: None,
        }
    }

    /// The definitional check: a model that assigns every one of `V` tokens
    /// probability `1/V` has NLL `ln V` per token and perplexity exactly `V`.
    #[test]
    fn a_uniform_model_over_v_tokens_has_perplexity_v() {
        let mut m = TokenPerplexityMetric::<TestBackend>::new();
        let vocab = 8192.0f32;
        m.update(&input(vocab.ln() * 100.0, 100.0), &metadata());
        assert!((m.value().current() - 8192.0).abs() < 1.0);
    }

    /// The reason this metric exists rather than a `NumericMetricState` average.
    /// Two batches of wildly different size must weight by token count; the
    /// naive mean of per-batch perplexities gives a different, wrong answer.
    #[test]
    fn batches_are_weighted_by_token_count_not_averaged() {
        let mut m = TokenPerplexityMetric::<TestBackend>::new();
        // 1000 tokens at NLL 1.0 each, then 10 tokens at NLL 5.0 each.
        m.update(&input(1000.0, 1000.0), &metadata());
        m.update(&input(50.0, 10.0), &metadata());

        let expected = ((1000.0 + 50.0) / 1010.0f64).exp();
        assert!((m.value().current() - expected).abs() < 1e-6);

        // What averaging per-batch perplexities would have produced. The gap is
        // large enough that a regression here cannot hide in the noise.
        let naive = ((1.0f64).exp() + (5.0f64).exp()) / 2.0;
        assert!(
            (naive - expected).abs() > 60.0,
            "the two rules must be distinguishable: {naive} vs {expected}"
        );
    }

    /// burn aggregates entries across epochs using `count`. If the count were
    /// batches rather than tokens, the aggregate would silently misweight.
    #[test]
    fn the_serialized_count_is_tokens() {
        let mut m = TokenPerplexityMetric::<TestBackend>::new();
        m.update(&input(10.0, 7.0), &metadata());
        m.update(&input(10.0, 3.0), &metadata());
        match m.value() {
            NumericEntry::Aggregated { count, .. } => assert_eq!(count, 10),
            other => panic!("perplexity must aggregate, got {other:?}"),
        }
    }

    /// `clear()` runs between splits and between epochs. State that survived it
    /// would fold the training set's NLL into the validation number.
    #[test]
    fn clear_resets_the_accumulators() {
        let mut m = TokenPerplexityMetric::<TestBackend>::new();
        m.update(&input(100.0, 10.0), &metadata());
        m.clear();
        m.update(&input(0.0, 10.0), &metadata());
        assert!((m.value().current() - 1.0).abs() < 1e-9, "exp(0) is 1");
    }
}
