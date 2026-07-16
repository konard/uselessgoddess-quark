//! The output of one training or validation step, and the masked loss that
//! produces it.
//!
//! # Why not burn's `SequenceOutput`
//!
//! burn ships `SequenceOutput<B>`, which carries the full `[batch, seq, vocab]`
//! logits and adapts to `PerplexityInput`. It is the obvious choice and it is
//! the wrong one here, for a reason that is structural rather than stylistic.
//!
//! Every metric input is produced from `<Output as ItemLazy>::ItemSync`, and
//! `sync()` registers its tensors in a [`Transaction`] -- i.e. it *reads them
//! back to the host*. `SequenceOutput::sync` reads back the logits. At this
//! project's shapes (batch 8 x seq 512 x vocab 8192, f32) that is 134 MB copied
//! off the GPU **per step**, after which `PerplexityMetric::update` runs a
//! 33.5M-element `log_softmax` on the CPU, also per step. The metric thread
//! would fall permanently behind the training loop and its queue would grow
//! without bound.
//!
//! So [`LmOutput`] carries three scalars instead. All three fall out of the
//! loss for free, `sync()` moves 12 bytes, and the perplexity metric becomes
//! two additions (see [`crate::train::metric`]). The logits never leave the
//! device.

use burn::{
    backend::Flex,
    prelude::Backend,
    tensor::{activation::log_softmax, Int, Tensor, Transaction},
    train::{
        metric::{Adaptor, LossInput},
        ItemLazy,
    },
};

use crate::train::metric::{GradRmsInput, TokenPerplexityInput};

/// What one step reports: the scalar the optimizer descends, plus the two
/// running totals perplexity is built from.
///
/// `sum_nll` and `n_tokens` are kept separately rather than folded into `loss`
/// because perplexity does not average. `exp(mean_of_batch_means)` is not
/// `exp(total_nll / total_tokens)` unless every batch scores exactly the same
/// number of tokens -- which is false for the final partial batch of an epoch,
/// and false throughout for strided evaluation.
#[derive(Debug)]
pub struct LmOutput<B: Backend> {
    /// Mean negative log-likelihood over the scored tokens of this batch.
    pub loss: Tensor<B, 1>,
    /// Negative log-likelihood summed over this batch's scored tokens, in nats.
    pub sum_nll: Tensor<B, 1>,
    /// How many tokens that sum covers.
    pub n_tokens: Tensor<B, 1>,
    /// RMS of the gradient over every parameter, or `None` where no gradient
    /// exists.
    ///
    /// `Option` rather than a zero, because "not measured" and "measured zero"
    /// are different claims and a validation pass has no gradient to report. It
    /// is `Some` only on the training path, which is why `GradRmsMetric` is
    /// registered on the train split alone -- see `crate::train::run`.
    pub grad_rms: Option<Tensor<B, 1>>,
}

/// `log P(target_t | context)` at every position: `[batch, seq]`.
///
/// The single definition of "how likely was this token", shared by the training
/// loss, the corpus evaluator, and BLiMP. Sharing it is the point -- a BLiMP
/// accuracy computed from a subtly different log-probability than the one the
/// model was trained on would be measuring a different model.
///
/// # Panics
/// If `targets` disagrees with `logits` on batch or sequence length.
pub fn token_log_probs<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 2> {
    let [batch, seq, vocab] = logits.dims();
    assert_eq!(targets.dims(), [batch, seq], "targets must match logits");

    let n = batch * seq;

    // log_softmax rather than log(softmax(..)): the former subtracts the row max
    // first, so a logit of 40 does not overflow the exponential. At 3M
    // parameters the logits stay small, but the loss is not the place to rely on
    // that.
    let log_probs = log_softmax(logits.reshape([n, vocab]), 1);
    log_probs
        .gather(1, targets.reshape([n, 1]))
        .reshape([batch, seq])
}

/// Cross-entropy over the positions `score_mask` marks.
///
/// Hand-rolled rather than `CrossEntropyLoss` for two reasons: the mask (burn's
/// loss takes a pad token, not a per-position mask), and `sum_nll`, which burn's
/// loss reduces away before returning.
///
/// # Panics
/// If the three tensors disagree on batch or sequence length.
pub fn masked_cross_entropy<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
    score_mask: Tensor<B, 2>,
) -> LmOutput<B> {
    let [batch, seq, _] = logits.dims();
    assert_eq!(score_mask.dims(), [batch, seq], "mask must match logits");

    let n = batch * seq;
    let picked = token_log_probs(logits, targets).reshape([n]);

    let mask = score_mask.reshape([n]);
    let sum_nll = picked.neg().mul(mask.clone()).sum();
    let n_tokens = mask.sum();

    // A batch with no scored positions cannot arise from the loaders in
    // `crate::train::run` (training windows are disjoint, so every position is
    // scored), but 0/0 is NaN and a single NaN gradient destroys the model
    // irrecoverably. The clamp costs one kernel and removes the failure mode.
    let loss = sum_nll.clone() / n_tokens.clone().clamp_min(1.0);

    // No gradient exists yet -- `backward` has not been called, and on the
    // validation path never will be. `TrainStep::step` fills this in.
    LmOutput {
        loss,
        sum_nll,
        n_tokens,
        grad_rms: None,
    }
}

/// The z-loss penalty: `mean(logsumexp(logits)^2)` over the scored positions.
///
/// Cross-entropy is invariant to a constant shift of a row of logits -- softmax
/// subtracts it away -- so nothing in the loss stops `logsumexp(logits)`, the
/// "z", from drifting far from zero. Wortsman et al. 2023 (arXiv:2309.14322)
/// §3.2 report that drift as an instability, and this penalty, from PaLM
/// (Chowdhery et al. 2022, §3) via Google's T5X, as the standard cure: it pins
/// the free constant near 0 without touching the distribution the model
/// expresses.
///
/// **Not added to [`LmOutput::loss`]**, on purpose. The reported and
/// checkpoint-selected loss stays pure cross-entropy, so a run with this on and
/// a run with it off are still comparable, and so that turning it on cannot
/// improve the metric by changing what the metric means. It reaches the
/// optimizer only through the gradient -- see `TrainStep::step`.
///
/// # Panics
/// If `score_mask` disagrees with `logits` on batch or sequence length.
pub fn masked_z_penalty<B: Backend>(
    logits: Tensor<B, 3>,
    score_mask: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let [batch, seq, vocab] = logits.dims();
    assert_eq!(score_mask.dims(), [batch, seq], "mask must match logits");
    let n = batch * seq;

    // Shift by the row max before exponentiating. The point of this penalty is
    // that z is unbounded, so the one place it must not be assumed small is the
    // computation of z itself. The shift is exact -- `log(sum(exp(x - m))) + m`
    // is `log(sum(exp(x)))` -- and it is what keeps a large z finite rather than
    // `inf`.
    let x = logits.reshape([n, vocab]);
    let max = x.clone().max_dim(1);
    let z = (x - max.clone()).exp().sum_dim(1).log() + max;

    let mask = score_mask.reshape([n, 1]);
    let sum = z.powi_scalar(2).mul(mask.clone()).sum();
    // Same clamp, same reason, as `masked_cross_entropy`: 0/0 is NaN and one
    // NaN gradient is unrecoverable.
    sum / mask.sum().clamp_min(1.0)
}

impl<B: Backend> ItemLazy for LmOutput<B> {
    // Flex is burn's own CPU backend, and burn-train depends on it
    // unconditionally -- see `burn-train/Cargo.toml`. Mirroring the choice burn
    // makes for `SequenceOutput` also lets `LossMetric::new()` infer its backend
    // parameter at the registration site, as it does in burn's examples.
    type ItemSync = LmOutput<Flex>;

    fn sync(self) -> Self::ItemSync {
        let device = &Default::default();

        // One transaction rather than an `into_data()` per tensor: it reads them
        // all back in a single round trip instead of stalling the queue once
        // each. `grad_rms` joins the same trip when it is present, so
        // instrumenting the training step costs four more bytes and no extra
        // synchronization -- and the validation path registers exactly what it
        // registered before.
        let mut tx = Transaction::default()
            .register(self.loss)
            .register(self.sum_nll)
            .register(self.n_tokens);
        if let Some(grad_rms) = self.grad_rms {
            tx = tx.register(grad_rms);
        }

        // Order is the registration order, so this is a queue, not an index into
        // something that might have moved.
        let mut data = tx.execute().into_iter();
        let mut next = || data.next().map(|d| Tensor::from_data(d, device));
        let expect = "the transaction yields one buffer per registered tensor";

        LmOutput {
            loss: next().expect(expect),
            sum_nll: next().expect(expect),
            n_tokens: next().expect(expect),
            // Absent iff nothing was registered for it, which is exactly the
            // condition it was `None` under on the way in.
            grad_rms: next(),
        }
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for LmOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: Backend> Adaptor<TokenPerplexityInput<B>> for LmOutput<B> {
    fn adapt(&self) -> TokenPerplexityInput<B> {
        TokenPerplexityInput::new(self.sum_nll.clone(), self.n_tokens.clone())
    }
}

impl<B: Backend> Adaptor<GradRmsInput<B>> for LmOutput<B> {
    /// # Panics
    /// If the item carries no gradient RMS. burn calls `adapt` once per
    /// registered metric per split, so this is reachable only by registering
    /// `GradRmsMetric` on the validation split -- where there is no gradient and
    /// the metric would be meaningless. `crate::train::run` registers it on
    /// train alone; panicking says so rather than inventing a zero.
    fn adapt(&self) -> GradRmsInput<B> {
        let rms = self
            .grad_rms
            .clone()
            .expect("GradRmsMetric belongs to the train split; validation computes no gradient");
        GradRmsInput::new(rms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::TestBackend;
    use burn::tensor::{ElementConversion, TensorData};

    fn scalar<B: Backend>(t: Tensor<B, 1>) -> f32 {
        t.into_scalar().elem::<f32>()
    }

    /// A uniform distribution over `V` classes assigns every token
    /// `log(1/V)`, so the loss must be exactly `ln V`. This pins the
    /// normalization: an implementation that forgot the softmax, or divided by
    /// the wrong count, would not land here.
    #[test]
    fn uniform_logits_cost_exactly_ln_vocab() {
        let d = Default::default();
        let vocab = 8;
        let logits = Tensor::<TestBackend, 3>::zeros([2, 3, vocab], &d);
        let targets = Tensor::<TestBackend, 2, Int>::zeros([2, 3], &d);
        let mask = Tensor::<TestBackend, 2>::ones([2, 3], &d);

        let out = masked_cross_entropy(logits, targets, mask);

        let expected = (vocab as f32).ln();
        assert!((scalar(out.loss) - expected).abs() < 1e-5);
        assert_eq!(scalar(out.n_tokens), 6.0);
        // Six tokens, each costing ln 8.
        assert!((scalar(out.sum_nll) - 6.0 * expected).abs() < 1e-4);
    }

    /// The mask is the whole reason this function exists rather than
    /// `CrossEntropyLoss`. A masked position must contribute to neither the
    /// numerator nor the denominator -- if it only left the numerator, the loss
    /// would be silently deflated by the fraction of masked tokens.
    #[test]
    fn masked_positions_contribute_to_neither_sum_nor_count() {
        let d = Default::default();
        let vocab = 4;
        // Position 0 of each row is confidently right; position 1 is confidently
        // wrong. Masking position 1 must leave only the cheap positions.
        let logits = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![
                    10.0f32, 0.0, 0.0, 0.0, // row 0, pos 0: target 0, right
                    0.0, 0.0, 0.0, 10.0, // row 0, pos 1: target 0, wrong
                ],
                [1, 2, vocab],
            ),
            &d,
        );
        let targets = Tensor::<TestBackend, 2, Int>::zeros([1, 2], &d);

        let all = masked_cross_entropy(
            logits.clone(),
            targets.clone(),
            Tensor::<TestBackend, 2>::ones([1, 2], &d),
        );
        let first_only = masked_cross_entropy(
            logits,
            targets,
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[1.0f32, 0.0]]), &d),
        );

        assert_eq!(scalar(all.n_tokens), 2.0);
        assert_eq!(scalar(first_only.n_tokens), 1.0);

        let (all_loss, masked_loss) = (scalar(all.loss), scalar(first_only.loss));
        // Dropping the expensive position must move the *mean*, not just the
        // sum: the denominator has to follow the mask.
        assert!(
            masked_loss < all_loss / 2.0,
            "masking the wrong prediction should leave only the cheap one: \
             {masked_loss} vs {all_loss}"
        );
        assert!(scalar(first_only.sum_nll) < 0.01);
    }

    /// `loss` and `sum_nll / n_tokens` must agree, because the live perplexity
    /// metric reads the latter while checkpoint selection reads the former. If
    /// they drift, the two numbers on the dashboard describe different models.
    #[test]
    fn loss_is_sum_nll_over_n_tokens() {
        let d = Default::default();
        let logits = Tensor::<TestBackend, 3>::random(
            [3, 5, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &d,
        );
        let targets = Tensor::<TestBackend, 2, Int>::from_data(
            TensorData::from([[1i32, 2, 3, 4, 5], [0, 1, 2, 3, 4], [5, 4, 3, 2, 1]]),
            &d,
        );
        let mask = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([
                [0.0f32, 0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0, 1.0],
            ]),
            &d,
        );

        let out = masked_cross_entropy(logits, targets, mask);
        let recomputed = scalar(out.sum_nll) / scalar(out.n_tokens.clone());
        assert_eq!(scalar(out.n_tokens), 12.0);
        assert!((scalar(out.loss) - recomputed).abs() < 1e-5);
    }

    /// `logsumexp` of `V` equal logits is `ln V`, so the penalty is exactly
    /// `(ln V)^2`. Pins the formula: a penalty that squared the wrong thing, or
    /// averaged before squaring, would not land here.
    #[test]
    fn the_z_penalty_of_uniform_logits_is_ln_vocab_squared() {
        let d = Default::default();
        let vocab = 8;
        let out = masked_z_penalty(
            Tensor::<TestBackend, 3>::zeros([2, 3, vocab], &d),
            Tensor::<TestBackend, 2>::ones([2, 3], &d),
        );
        let expected = (vocab as f32).ln().powi(2);
        assert!((scalar(out) - expected).abs() < 1e-4);
    }

    /// The reason this penalty exists, as an assertion.
    ///
    /// Adding a constant to every logit of a row leaves the softmax -- and so
    /// the cross-entropy -- bit-identical, which means nothing in the loss
    /// opposes that constant drifting. The penalty is the only term that can
    /// see it, so it must move when the loss does not.
    #[test]
    fn a_constant_logit_shift_is_invisible_to_loss_and_visible_to_the_z_penalty() {
        let d = Default::default();
        let logits = Tensor::<TestBackend, 3>::random(
            [2, 4, 16],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &d,
        );
        let targets = Tensor::<TestBackend, 2, Int>::zeros([2, 4], &d);
        let mask = Tensor::<TestBackend, 2>::ones([2, 4], &d);
        let shifted = logits.clone() + 5.0;

        let before = masked_cross_entropy(logits.clone(), targets.clone(), mask.clone());
        let after = masked_cross_entropy(shifted.clone(), targets, mask.clone());
        assert!(
            (scalar(before.loss) - scalar(after.loss)).abs() < 1e-4,
            "cross-entropy must be shift-invariant"
        );

        let z_before = scalar(masked_z_penalty(logits, mask.clone()));
        let z_after = scalar(masked_z_penalty(shifted, mask));
        assert!(
            z_after > z_before + 10.0,
            "the penalty must see the shift the loss cannot: {z_before} -> {z_after}"
        );
    }

    /// The penalty's own argument is that z is unbounded, so the one place that
    /// must not assume z is small is the code computing it. `exp(100)` overflows
    /// f32 (max ~3.4e38), so dropping the max-shift turns this into `inf` -- and
    /// an infinite penalty is a NaN gradient one step later.
    #[test]
    fn the_z_penalty_survives_logits_large_enough_to_overflow_exp() {
        let d = Default::default();
        let vocab = 8;
        let out = masked_z_penalty(
            Tensor::<TestBackend, 3>::full([1, 2, vocab], 100.0, &d),
            Tensor::<TestBackend, 2>::ones([1, 2], &d),
        );
        let got = scalar(out);
        assert!(got.is_finite(), "penalty overflowed to {got}");
        // logsumexp of V copies of 100 is 100 + ln V.
        let expected = (100.0 + (vocab as f32).ln()).powi(2);
        assert!((got - expected).abs() / expected < 1e-4, "got {got}");
    }

    /// Masked positions must not be averaged over, exactly as in the loss. If
    /// the denominator ignored the mask, the penalty would be diluted by the
    /// fraction of unscored positions.
    #[test]
    fn the_z_penalty_follows_the_mask() {
        let d = Default::default();
        // Position 0: logits 0 -> z = ln 4. Position 1: logits 10 -> z = 10+ln 4.
        let logits = Tensor::<TestBackend, 3>::from_data(
            TensorData::new(
                vec![0.0f32, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0],
                [1, 2, 4],
            ),
            &d,
        );
        let first_only = masked_z_penalty(
            logits.clone(),
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[1.0f32, 0.0]]), &d),
        );
        let expected = (4.0f32).ln().powi(2);
        assert!(
            (scalar(first_only) - expected).abs() < 1e-4,
            "masking the large-z position must leave only the small one"
        );
    }

    /// An all-masked batch is unreachable through the training loaders, but the
    /// clamp that makes it survivable is easy to delete by accident. NaN in a
    /// loss is not recoverable: it propagates into every parameter on the next
    /// optimizer step.
    #[test]
    fn an_entirely_masked_z_penalty_yields_zero_rather_than_nan() {
        let d = Default::default();
        let out = masked_z_penalty(
            Tensor::<TestBackend, 3>::zeros([1, 2, 4], &d),
            Tensor::<TestBackend, 2>::zeros([1, 2], &d),
        );
        assert_eq!(scalar(out), 0.0);
    }

    /// An all-masked batch is unreachable through the training loaders, but the
    /// clamp that makes it survivable is easy to delete by accident. NaN in a
    /// loss is not recoverable: it propagates into every parameter on the next
    /// optimizer step.
    #[test]
    fn an_entirely_masked_batch_yields_zero_rather_than_nan() {
        let d = Default::default();
        let out = masked_cross_entropy(
            Tensor::<TestBackend, 3>::zeros([1, 2, 4], &d),
            Tensor::<TestBackend, 2, Int>::zeros([1, 2], &d),
            Tensor::<TestBackend, 2>::zeros([1, 2], &d),
        );
        assert_eq!(scalar(out.loss), 0.0);
        assert_eq!(scalar(out.n_tokens), 0.0);
    }

    /// `sync` registers tensors in one order and reads the results back in
    /// another place; the two agreeing is an assumption, and a silent one. If
    /// they ever disagree the tensors do not vanish, they *swap* -- the loss
    /// metric would plot `n_tokens` and nothing would look broken.
    ///
    /// So: four values, all distinguishable, checked by identity rather than by
    /// shape.
    #[test]
    fn sync_round_trips_every_field_without_permuting_them() {
        let d = Default::default();
        let at = |x: f32| Tensor::<TestBackend, 1>::from_data(TensorData::from([x]), &d);
        let out = LmOutput {
            loss: at(1.0),
            sum_nll: at(2.0),
            n_tokens: at(3.0),
            grad_rms: Some(at(4.0)),
        };

        let synced = out.sync();

        assert_eq!(scalar(synced.loss), 1.0);
        assert_eq!(scalar(synced.sum_nll), 2.0);
        assert_eq!(scalar(synced.n_tokens), 3.0);
        assert_eq!(
            scalar(synced.grad_rms.expect("registered, so returned")),
            4.0
        );
    }

    /// The absent case is the one the validation path takes on every batch, and
    /// it is where an off-by-one in the read-back queue would show up: three
    /// registered tensors must yield three, and the fourth read must come back
    /// empty rather than stealing a value or panicking.
    #[test]
    fn sync_keeps_an_absent_grad_rms_absent() {
        let d = Default::default();
        let at = |x: f32| Tensor::<TestBackend, 1>::from_data(TensorData::from([x]), &d);
        let out = LmOutput {
            loss: at(1.0),
            sum_nll: at(2.0),
            n_tokens: at(3.0),
            grad_rms: None,
        };

        let synced = out.sync();

        assert_eq!(scalar(synced.loss), 1.0);
        assert_eq!(scalar(synced.n_tokens), 3.0);
        assert!(synced.grad_rms.is_none());
    }
}
