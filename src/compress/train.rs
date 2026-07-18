//! Training the compressor, on the harness the language model already has.
//!
//! This module is deliberately thin, and that is its point. Issue #12 asks for a
//! feature that does not bite into the existing codebase and does not duplicate
//! it either, and the way to have both is to notice how little of a training run
//! is actually architecture-specific. A compressor run needs the same shards,
//! the same windows, the same batcher, the same optimizer, the same schedule,
//! the same checkpoint pruning and the same best-epoch recovery. What differs is
//! three things:
//!
//!  * **the objective** -- reconstruct the span rather than predict the next
//!    token, which is [`TrainStep`] below and nothing else;
//!  * **the target** -- the span *is* its own target, so no new dataset, no new
//!    window type, no new batcher (see [`TokenBatch`]);
//!  * **the config** -- [`CompressTrainConfig`], which is a [`CompressConfig`]
//!    beside a [`TrainConfig`] rather than a re-declaration of either.
//!
//! Everything else is [`crate::train::launch`], called with a different closure.

use anyhow::{bail, Context, Result};
use burn::{
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Tensor},
    train::{InferenceStep, TrainOutput, TrainStep},
};
use serde::{Deserialize, Serialize};

use crate::{
    compress::{CompressConfig, Compressor},
    data::TokenBatch,
    train::{
        grad_rms, launch, open_datasets, output::masked_cross_entropy, output::LmOutput,
        refuse_to_merge_runs, TrainConfig,
    },
};

/// A compressor run: what to build, and how to train it.
///
/// Two configs side by side rather than one merged config with every field
/// copied out of [`TrainConfig`]. The redundancy that composition creates --
/// `train.model` and `compress.model`, `train.seq_len` and `compress.span_len`
/// -- is real, and [`Self::validate`] refuses a config where the two copies
/// disagree rather than silently preferring one. [`Self::sync`] is the way to
/// never have to think about it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompressTrainConfig {
    /// The compressor to build: shape, span, slots, bottleneck, regularizers.
    pub compress: CompressConfig,
    /// Everything about the *run*: shards, schedule, optimizer, artifacts.
    ///
    /// `train.model` and `train.seq_len` are not free -- they must equal
    /// `compress.model` and `compress.span_len`, because the harness reads them
    /// to check the shard's vocabulary and to cut the windows.
    pub train: TrainConfig,
}

impl Default for CompressTrainConfig {
    fn default() -> Self {
        Self::sync(
            CompressConfig::compressor_15m(),
            TrainConfig {
                artifact_dir: "artifacts/compress".into(),
                // A ratio, not the reference run's absolute 200 batches. This
                // schedule is new and has no run to stay byte-compatible with,
                // so the argument in `TrainConfig::warmup_ratio` applies with
                // nothing on the other side: 2% of the run is 2% of the run
                // however the epochs and the batch split are chosen.
                warmup_ratio: Some(0.02),
                // Inherited from the language model at the same width, because
                // there is no LR sweep for this objective and inventing a number
                // would be worse than reusing a measured one. It is the first
                // thing to tune -- see `docs/COMPRESSION.md`.
                ..TrainConfig::default()
            },
        )
    }
}

impl CompressTrainConfig {
    /// Pair a compressor with a run, making the run agree with it.
    ///
    /// `train.model` and `train.seq_len` are overwritten from `compress`, so the
    /// duplication that [`Self::validate`] guards against cannot be introduced
    /// by a caller that goes through here. The CLI does.
    pub fn sync(compress: CompressConfig, train: TrainConfig) -> Self {
        let train = TrainConfig {
            model: compress.model.clone(),
            seq_len: compress.span_len,
            ..train
        };
        Self { compress, train }
    }

    pub fn validate(&self) -> Result<()> {
        let mut errs = Vec::new();

        if let Err(e) = self.compress.validate() {
            errs.extend(e.into_iter().map(|e| format!("compress: {e}")));
        }
        if let Err(e) = self.train.validate() {
            errs.push(format!("train: {e}"));
        }

        // The harness reads `train.seq_len` to cut windows and `train.model` to
        // check the shard's vocabulary. If either disagrees with the compressor,
        // one of them is a lie -- and the failure would be a shape mismatch deep
        // inside the first forward pass rather than a message here.
        if self.train.seq_len != self.compress.span_len {
            errs.push(format!(
                "train.seq_len ({}) must equal compress.span_len ({}): the dataloader cuts \
                 windows to train.seq_len and the compressor asserts it received span_len",
                self.train.seq_len, self.compress.span_len
            ));
        }
        if self.train.model != self.compress.model {
            errs.push(
                "train.model must equal compress.model: the vocabulary check and the \
                 compressor's stacks are built from different copies of it"
                    .to_string(),
            );
        }

        // Not silently ignored. `z_loss` penalizes `logsumexp` on the LM head,
        // and the compressor has no field to carry the coefficient into its
        // step, so a run configured with one would train without it and never
        // say so.
        if self.train.z_loss != 0.0 {
            errs.push(format!(
                "train.z_loss is {} but the compressor's step does not apply it; set it to 0",
                self.train.z_loss
            ));
        }

        if errs.is_empty() {
            Ok(())
        } else {
            bail!("invalid CompressTrainConfig:\n  - {}", errs.join("\n  - "));
        }
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)
            .with_context(|| format!("writing compressor config to {}", path.display()))?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .with_context(|| format!("reading compressor config from {}", path.display()))?;
        serde_json::from_str(&json)
            .with_context(|| format!("parsing compressor config at {}", path.display()))
    }
}

/// One reconstruction pass and its loss.
///
/// The whole objective, and it is one line of substance: the span is encoded,
/// quantized, and decoded back, and the loss is cross-entropy against *the span
/// itself*. `batch.target` -- the next-token targets the same batcher also
/// produced -- is unused here, which is the concrete form of "the compressor
/// needs no new dataset".
///
/// The mask is ones rather than `batch.score_mask`. They coincide for training
/// windows (which are disjoint, so every position is scored), but they mean
/// different things: `score_mask` marks positions of `target` that a *strided*
/// language-model evaluation should not double-count, and a compressor
/// reconstructs every position of every span it is given. Borrowing the mask
/// would make the loss quietly depend on a striding decision that has nothing to
/// do with it.
fn reconstruction_step<B: Backend>(model: &Compressor<B>, batch: TokenBatch<B>) -> LmOutput<B> {
    let tokens = batch.input;
    let logits = model.forward(tokens.clone());
    let [batch_size, span] = tokens.dims();
    let mask = Tensor::ones([batch_size, span], &tokens.device());
    masked_cross_entropy(logits, tokens, mask)
}

impl<B: AutodiffBackend> TrainStep for Compressor<B> {
    type Input = TokenBatch<B>;
    type Output = LmOutput<B>;

    fn step(&self, batch: TokenBatch<B>) -> TrainOutput<LmOutput<B>> {
        let mut item = reconstruction_step(self, batch);

        // Divided for the same reason as in `QuarkLm`'s step, and it is worth
        // restating because it is the one place a copied-looking line is load
        // bearing: burn's `GradientsAccumulator` *sums* micro-batch gradients,
        // the loss is already a per-token mean, and gradient clipping is not
        // scale invariant. See `crate::train::TrainStep for QuarkLm`.
        let scaled = item
            .loss
            .clone()
            .div_scalar(self.grad_accumulation() as f64);
        let grads = scaled.backward();

        item.grad_rms = grad_rms(self, &grads, self.grad_accumulation() as f64);

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> InferenceStep for Compressor<B> {
    type Input = TokenBatch<B>;
    type Output = LmOutput<B>;

    /// Teacher-forced, exactly like the training step, and *not* the headline
    /// number.
    ///
    /// This is what checkpoints are selected on, so it has to be the quantity
    /// the optimizer descends -- selecting on one loss while descending another
    /// is how a run ends up returning a model that is not the one it reported.
    /// It is also an optimistic view of the compressor, since the decoder is fed
    /// the true prefix rather than its own output; the honest measurement is
    /// free-running reconstruction, which
    /// [`Compressor::reconstruct`](crate::compress::Compressor::reconstruct)
    /// performs and `docs/COMPRESSION.md` explains.
    fn step(&self, batch: TokenBatch<B>) -> LmOutput<B> {
        reconstruction_step(self, batch)
    }
}

/// Train a compressor, and return the best one by validation loss.
///
/// The body is short because it is meant to be: everything after the config is
/// [`crate::train::launch`], the same function the language model runs through.
pub fn run<B: AutodiffBackend>(
    mut config: CompressTrainConfig,
    device: B::Device,
) -> Result<Compressor<B::InnerBackend>> {
    config.validate()?;
    refuse_to_merge_runs(&config.train.artifact_dir, config.train.resume_from_epoch)?;

    let (train_dataset, valid_dataset, eos_id) = open_datasets(&config.train)?;

    // The decoder needs a token to start from and the tokenizer has no dedicated
    // BOS, so the shard's end-of-text id is used -- the convention already in
    // `eval/blimp.rs`. Taken from the shard rather than from the config because
    // the shard is the thing that knows: a config carrying a stale id would put
    // a token the corpus never uses at position 0 of every decode.
    if config.compress.bos_id != eos_id {
        tracing::info!(
            from = config.compress.bos_id,
            to = eos_id,
            "adopting the shard's end-of-text id as the decoder's start token"
        );
        config.compress.bos_id = eos_id;
    }
    // Re-checked after the overwrite: an id the shard reports but the model has
    // no embedding row for would index out of bounds on the first decode.
    config.validate()?;

    tracing::info!(
        params = config.compress.param_count(),
        span_len = config.compress.span_len,
        n_slots = config.compress.n_slots,
        token_ratio = config.compress.token_ratio(),
        bits_per_token = config.compress.rate_bits_per_token(),
        "starting compressor training"
    );

    // Written before the first step, so an interrupted run is still
    // reproducible -- and it is the *compressor* config that is written, not
    // just the `TrainConfig` half, or the artifact directory would not describe
    // the model it holds.
    config.save(&config.train.artifact_dir.join("config.json"))?;

    launch(
        &config.train,
        &device,
        train_dataset,
        valid_dataset,
        |device| {
            Compressor::<B>::new(config.compress.clone(), device)
                .with_grad_accumulation(config.train.grad_accumulation)
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        compress::CompressConfig,
        test_util::{TestAutodiffBackend, TestBackend},
    };
    use burn::{
        module::AutodiffModule,
        tensor::{Int, Tensor, TensorData},
    };

    type B = TestAutodiffBackend;

    /// The same batch on either backend, built by hand rather than sampled.
    ///
    /// Deterministic because two of the tests below compare a number computed on
    /// the autodiff backend against one computed on the inner backend, and a
    /// batch drawn twice from an RNG would differ for reasons that have nothing
    /// to do with what is being tested.
    fn batch<BB: Backend>(cfg: &CompressConfig, batch_size: usize) -> TokenBatch<BB> {
        let device = Default::default();
        let dims = [batch_size, cfg.span_len];
        let ids: Vec<i64> = (0..batch_size * cfg.span_len)
            .map(|i| ((i * 37 + 11) % cfg.model.vocab_size) as i64)
            .collect();
        TokenBatch {
            input: Tensor::<BB, 2, Int>::from_data(TensorData::new(ids, dims), &device),
            // Deliberately garbage: the compressor must not read it, and a test
            // that fed it the real next-token targets could not tell.
            target: Tensor::<BB, 2, Int>::zeros(dims, &device),
            score_mask: Tensor::zeros(dims, &device),
        }
    }

    fn tiny() -> CompressTrainConfig {
        CompressTrainConfig::sync(CompressConfig::tiny(), TrainConfig::default())
    }

    /// The default config is the one a user gets by typing nothing, so it had
    /// better be one the harness accepts.
    #[test]
    fn the_default_config_is_valid() {
        CompressTrainConfig::default().validate().unwrap();
    }

    /// `sync` exists to make the duplicated fields agree; if it did not, every
    /// CLI invocation would trip the validator.
    #[test]
    fn sync_makes_the_run_agree_with_the_compressor() {
        let cfg = CompressTrainConfig::sync(
            CompressConfig::tiny(),
            TrainConfig {
                seq_len: 999,
                ..TrainConfig::default()
            },
        );
        assert_eq!(cfg.train.seq_len, CompressConfig::tiny().span_len);
        assert_eq!(cfg.train.model, CompressConfig::tiny().model);
        cfg.validate().unwrap();
    }

    /// A hand-edited config where the window length and the span length have
    /// drifted apart must fail here, not inside the first forward pass.
    #[test]
    fn a_window_that_is_not_a_span_is_rejected() {
        let mut cfg = tiny();
        cfg.train.seq_len = cfg.compress.span_len - 1;
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("must equal compress.span_len"), "{err}");
    }

    /// A knob that does nothing is worse than a missing knob, because it looks
    /// like it worked.
    #[test]
    fn a_z_loss_the_step_would_ignore_is_rejected() {
        let mut cfg = tiny();
        cfg.train.z_loss = 1e-4;
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("does not apply it"), "{err}");
    }

    #[test]
    fn the_config_survives_a_round_trip_through_json() {
        let cfg = CompressTrainConfig::default();
        let json = serde_json::to_string_pretty(&cfg).unwrap();
        let back: CompressTrainConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, back);
    }

    /// The training step must produce a gradient for *every* parameter.
    ///
    /// This is the check that makes the run logically verifiable without running
    /// it, and it is not trivially true here: the encoder reaches the loss only
    /// through the quantizer, and a bottleneck whose straight-through estimator
    /// were wired up wrongly -- `round()` without the `detach()` trick -- would
    /// silently cut the encoder off. The loss would still go down, driven by the
    /// decoder alone, and the compressor would be learning nothing about
    /// compressing. So: count the parameters that received a gradient, and
    /// require all of them.
    #[test]
    fn every_parameter_gets_a_gradient() {
        use burn::{
            module::{Module, ModuleVisitor, Param},
            optim::GradientsParams,
        };

        /// Counts parameters the optimizer would and would not update.
        ///
        /// `GradientsParams` is what `TrainOutput` carries and what the
        /// optimizer consumes, so asking it -- rather than the raw autodiff
        /// gradients -- is asking the question that matters: a parameter absent
        /// from this map is a parameter that never moves.
        struct Count<'a, B: AutodiffBackend> {
            grads: &'a GradientsParams,
            with: usize,
            without: Vec<usize>,
            _backend: core::marker::PhantomData<B>,
        }
        impl<B: AutodiffBackend> ModuleVisitor<B> for Count<'_, B> {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                match self.grads.get::<B::InnerBackend, D>(param.id) {
                    Some(_) => self.with += 1,
                    None => self.without.push(param.val().shape().num_elements()),
                }
            }
        }

        let cfg = CompressConfig::tiny();
        let model = Compressor::<B>::new(cfg.clone(), &Default::default());
        let output = TrainStep::step(&model, batch::<B>(&cfg, 2));

        let mut count = Count::<B> {
            grads: &output.grads,
            with: 0,
            without: Vec::new(),
            _backend: core::marker::PhantomData,
        };
        model.visit(&mut count);

        assert!(
            count.without.is_empty(),
            "{} parameter tensors received no gradient (sizes {:?}); the encoder reaches the \
             loss only through the straight-through estimator, so this is what a broken \
             bottleneck looks like",
            count.without.len(),
            count.without
        );
        assert!(count.with > 0, "the model has no float parameters at all");

        // And the reported RMS is a real, finite number rather than the `None`
        // the validation path carries.
        let rms = output
            .item
            .grad_rms
            .expect("the training step measures the gradient")
            .into_scalar();
        assert!(rms.is_finite() && rms > 0.0, "grad_rms was {rms}");

        // The inference step reports no gradient, which is what keeps
        // `GradRmsMetric` registered on the train split only.
        let valid = model.valid().step(batch::<TestBackend>(&cfg, 2));
        assert!(valid.grad_rms.is_none());
    }

    /// Both steps must compute the *same* loss, or checkpoint selection would be
    /// ranking epochs by a quantity training never descended.
    ///
    /// Checked with the regularizers off, since token and latent dropout are
    /// deliberately active in training and absent in validation.
    #[test]
    fn training_and_validation_score_the_same_thing() {
        let cfg = CompressConfig {
            token_dropout: 0.0,
            latent_dropout: 0.0,
            ..CompressConfig::tiny()
        };
        let model = Compressor::<B>::new(cfg.clone(), &Default::default());

        let train = TrainStep::step(&model, batch::<B>(&cfg, 2))
            .item
            .loss
            .into_scalar();
        let valid = model
            .valid()
            .step(batch::<TestBackend>(&cfg, 2))
            .loss
            .into_scalar();

        assert!(
            (train - valid).abs() < 1e-4,
            "train {train} vs valid {valid}"
        );
    }
}
