//! The training runtime: config, steps, and the wiring into burn-train.
//!
//! The issue asks for a harness that someone else runs, on their own 16GB GPU,
//! for hours, without me present. That constraint shapes everything here:
//!
//!  * **Every number is in the config, and the config is written to disk.** A
//!    run that cannot be reproduced from its artifact directory is a run whose
//!    result means nothing.
//!  * **It fails before it starts, not after an hour.** [`TrainConfig::validate`]
//!    rejects every mistake it can name, and `run` calls it first.
//!  * **The schedule is computed, not guessed.** `num_iters` is derived from the
//!    dataset length, so a cosine decay actually reaches its floor at the end of
//!    training rather than somewhere arbitrary.
//!
//! See `docs/DESIGN.md` §7 for why these choices and not others.

pub mod metric;
pub mod output;

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{bail, Context, Result};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    grad_clipping::GradientClippingConfig,
    lr_scheduler::{
        composed::ComposedLrSchedulerConfig, cosine::CosineAnnealingLrSchedulerConfig,
        linear::LinearLrSchedulerConfig,
    },
    module::{Module, ModuleVisitor, Param},
    optim::AdamWConfig,
    prelude::Backend,
    record::CompactRecorder,
    tensor::{backend::AutodiffBackend, Tensor},
    train::{
        metric::{LearningRateMetric, LossMetric},
        InferenceStep, LearnerSummary, LearningResult, SupervisedTraining, TrainOutput, TrainStep,
    },
};
use serde::{Deserialize, Serialize};

use crate::{
    data::{Shard, TokenBatch, TokenBatcher, TokenDataset},
    model::QuarkLm,
    train::{
        metric::{GradRmsMetric, TokenPerplexityMetric},
        output::{masked_cross_entropy, masked_z_penalty, LmOutput},
    },
    ModelConfig,
};

/// Everything a run needs, in one serializable place.
///
/// Plain serde with public fields rather than burn's `Config` derive, matching
/// [`ModelConfig`]: the derive's generated `with_*` builders buy little here and
/// its `load`/`save` are less flexible than serde_json used directly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainConfig {
    /// The architecture to train. Defaults to the 3M reference model.
    pub model: ModelConfig,

    /// Shard produced by `quark prepare`, used for gradient steps.
    pub train_shard: PathBuf,
    /// A *disjoint* shard used only for the checkpoint-selection metric.
    pub valid_shard: PathBuf,
    /// Where checkpoints, logs and `config.json` go.
    pub artifact_dir: PathBuf,

    /// Context length. Must not exceed `model.max_seq_len`.
    pub seq_len: usize,
    /// Windows per forward pass. **The knob to turn when VRAM runs out.**
    pub batch_size: usize,
    /// Forward/backward passes per optimizer step. Raise this by the same factor
    /// you lower `batch_size` by, and the optimizer sees the same batch.
    pub grad_accumulation: usize,
    pub num_epochs: usize,

    /// Peak learning rate, reached at the end of warmup.
    pub lr: f64,
    /// Cosine floor, as a fraction of [`Self::lr`].
    pub min_lr_ratio: f64,
    /// Warmup length **in batches, not optimizer steps** -- see
    /// [`Self::total_batches`]. Ignored when [`Self::warmup_ratio`] is set.
    pub warmup_batches: usize,
    /// Warmup length as a *fraction of the whole run*, resolved against
    /// [`Self::total_batches`]. When set, it overrides [`Self::warmup_batches`].
    ///
    /// [`Self::warmup_batches`] is an absolute count, and so is a *different
    /// fraction* of every run: `200` is 1.2% of the 1-epoch reference but 0.3%
    /// of a 4-epoch sweep, and it shrinks again whenever `grad_accumulation`
    /// rises to hold a larger effective batch (docs/NEXT.md §3, §13). That drift
    /// is silent -- nothing recomputes the warmup when the run's shape changes --
    /// so two runs meant to be comparable warm the learning rate for different
    /// shares of training. A ratio pins the share instead: `Some(0.02)` warms
    /// over the first 2% of the schedule no matter how many epochs it runs or how
    /// the effective batch is split between `batch_size` and `grad_accumulation`.
    ///
    /// `#[serde(default)]` keeps configs written before this field loadable, and
    /// `None` reproduces the previous behaviour exactly -- the absolute
    /// `warmup_batches` count, unchanged -- so the reference run is untouched.
    #[serde(default)]
    pub warmup_ratio: Option<f64>,

    pub weight_decay: f32,
    pub beta_1: f32,
    pub beta_2: f32,
    pub epsilon: f32,
    /// Clip threshold applied to **each parameter tensor separately**, not to
    /// the global gradient norm.
    ///
    /// This is burn's semantics, not a choice, and it is worth stating because
    /// it is not what "clip_grad_norm 1.0" means anywhere else. GPT-2, PaLM and
    /// nanoGPT all clip the *global* norm: one coefficient, computed over the
    /// concatenation of every gradient, rescaling all of them together -- which
    /// shortens the update without rotating it. burn instead computes
    /// `sqrt(sum(g^2))` over one tensor and rescales that tensor alone
    /// (`GradientClipping::clip_by_norm`, whose signature takes a single
    /// `Tensor<B, D>`), and `OptimizerAdaptor` calls it once per parameter
    /// (`burn-optim/src/optim/simple/adaptor.rs`). Per-tensor clipping gives
    /// each tensor its own coefficient, so when it binds on some tensors and
    /// not others it changes the update's *direction*, not just its length.
    ///
    /// How much that matters here is damped but not nullified by AdamW: a
    /// gradient rescaled by a constant `c` leaves `m/sqrt(v)` unchanged, so a
    /// clip that binds equally at every step is nearly invisible. The clip
    /// coefficient is not constant, though -- it moves with the per-tensor norm
    /// -- and weight decay and `epsilon` both see the raw scale.
    ///
    /// Whether `1.0` binds at all at this scale is unmeasured; that is what
    /// `GradRms` in the metric log is for. Left at 1.0 rather than "fixed",
    /// because changing it is a change to the reference run and there is no
    /// evidence yet to justify one -- see docs/RESULTS.md §5.
    pub grad_clip_norm: f32,
    /// Coefficient on the z-loss penalty; `0.0` disables it.
    ///
    /// Penalizes `logsumexp(logits)^2`, the constant cross-entropy cannot see.
    /// See [`masked_z_penalty`] for the mechanism and the sources; 1e-4 is the
    /// PaLM/Wortsman value. It is **not** added to the reported loss, so
    /// enabling it does not make a run's numbers incomparable to the runs in
    /// issue #3 -- only its gradients differ.
    ///
    /// `#[serde(default)]` keeps configs written before this field loadable,
    /// and their meaning unchanged: zero, i.e. off.
    #[serde(default)]
    pub z_loss: f32,

    pub seed: u64,
    /// Dataloader worker threads.
    pub num_workers: usize,
    /// Epoch to resume from; `None` starts fresh. Requires checkpoints from a
    /// previous run in `artifact_dir`.
    pub resume_from_epoch: Option<usize>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::quark_3m(),

            train_shard: PathBuf::from("artifacts/train.bin"),
            valid_shard: PathBuf::from("artifacts/valid.bin"),
            artifact_dir: PathBuf::from("artifacts/run"),

            seq_len: 512,
            // 16 x 512 x 4 = 32,768 tokens per optimizer step. The model is 3M
            // parameters, so the activations, not the weights, are what fill a
            // 16GB card; batch_size is the dial that moves them.
            batch_size: 16,
            grad_accumulation: 4,
            num_epochs: 1,

            // Higher than GPT-2 small's 6e-4, deliberately. Optimal LR rises as
            // model size falls, and 3M parameters is two orders of magnitude
            // below 124M. This is the first thing to tune, and the first thing
            // to blame if loss diverges early.
            lr: 3e-3,
            // Decay to a tenth of peak rather than to zero, as in Hoffmann et
            // al. 2022 §3.
            min_lr_ratio: 0.1,
            warmup_batches: 200,
            // Absolute count by default: `None` keeps the reference run's
            // schedule byte-for-byte. Set `warmup_ratio` instead when a run
            // changes epochs or the batch split and the warmup should follow.
            warmup_ratio: None,

            // burn's AdamW defaults are epsilon 1e-5 and weight_decay 1e-4;
            // neither is a language-modelling default. 1e-5 is large enough to
            // damp the adaptive denominator for small gradients, and 1e-4 is
            // effectively no regularization at this scale.
            weight_decay: 0.1,
            beta_1: 0.9,
            // 0.95 rather than Adam's 0.999: the shorter second-moment window is
            // standard for transformer LMs (GPT-3 §B, Chinchilla) and copes
            // better with the gradient spikes a small model produces.
            beta_2: 0.95,
            // 1e-15, not the 1e-8 this used to be and not Adam's 1e-8 default.
            // Wortsman et al. 2023 (arXiv:2309.14322) §3.4 measure the gradient
            // RMS collapsing *below* epsilon during training, at which point the
            // constant, not the gradient, sets the update size and learning
            // stalls: "decreasing epsilon to 1e-15 improves loss and mitigates a
            // collapse in grad RMS". f32 holds 1e-15 with room to spare -- the
            // smallest normal is ~1.2e-38.
            //
            // This changes the reference run. It is the one hyperparameter here
            // that the runs in issue #3 did not use, and it is listed as such in
            // docs/RESULTS.md rather than slipped in.
            epsilon: 1e-15,
            grad_clip_norm: 1.0,
            // Off by default. Unlike epsilon, this one alters the objective, and
            // the case for it is a hypothesis about a divergence whose cause is
            // not established -- so it is a flag to turn on in an experiment,
            // not a change to the reference run.
            z_loss: 0.0,

            seed: 42,
            num_workers: 4,
            resume_from_epoch: None,
        }
    }
}

impl TrainConfig {
    /// Windows per epoch, and therefore [`LrScheduler`](burn::LrScheduler)
    /// steps per epoch.
    ///
    /// `div_ceil` because the final partial batch is still emitted:
    /// `PartialDataset::split_chunks` hands each worker a whole number of
    /// batches and `FixBatchStrategy` never emits an empty one, so this count is
    /// exact for any `num_workers`.
    pub fn batches_per_epoch(&self, dataset_len: usize) -> usize {
        dataset_len.div_ceil(self.batch_size)
    }

    /// The cosine schedule's period.
    ///
    /// **In batches, not optimizer steps.** burn's training loop calls
    /// `lr_step()` once per dataloader batch, *before* the branch that decides
    /// whether to apply gradients -- see
    /// `burn-train/src/learner/supervised/strategies/single/epoch.rs`. So with
    /// `grad_accumulation = 4` the scheduler advances four times per update. A
    /// schedule sized in optimizer steps would exhaust its decay in the first
    /// quarter of training and then sit at `min_lr` -- silently, since nothing
    /// checks.
    pub fn total_batches(&self, dataset_len: usize) -> usize {
        self.num_epochs * self.batches_per_epoch(dataset_len)
    }

    /// Warmup length in batches for a run of `total_batches`, resolving
    /// [`Self::warmup_ratio`] against it.
    ///
    /// With a ratio set, the warmup is a fixed share of the whole schedule --
    /// grad-accumulation- and epoch-invariant, since `total_batches` already
    /// folds both in. Rounded to the nearest batch, and never longer than the
    /// run: a `ratio >= 1.0` would leave no batches for the cosine decay, so it
    /// is clamped to `total_batches - 1` (the training-loop guard rejects the
    /// pathological case up front; this keeps the schedule well-formed for the
    /// callers, e.g. tests, that build it directly). With no ratio, the absolute
    /// [`Self::warmup_batches`] is returned unchanged.
    pub fn effective_warmup_batches(&self, total_batches: usize) -> usize {
        match self.warmup_ratio {
            Some(ratio) => {
                let batches = (ratio * total_batches as f64).round() as usize;
                batches.min(total_batches.saturating_sub(1))
            }
            None => self.warmup_batches,
        }
    }

    pub fn min_lr(&self) -> f64 {
        self.lr * self.min_lr_ratio
    }

    /// Reject what can be rejected now rather than after an hour of GPU time.
    pub fn validate(&self) -> Result<()> {
        let mut errs = Vec::new();

        if self.seq_len == 0 {
            errs.push("seq_len must be positive".to_string());
        }
        if self.seq_len > self.model.max_seq_len {
            errs.push(format!(
                "seq_len {} exceeds the model's max_seq_len {}: RoPE tables are built to \
                 max_seq_len, so longer windows would index out of bounds",
                self.seq_len, self.model.max_seq_len
            ));
        }
        if self.batch_size == 0 {
            errs.push("batch_size must be positive".to_string());
        }
        if self.grad_accumulation == 0 {
            errs.push("grad_accumulation must be at least 1".to_string());
        }
        if self.num_epochs == 0 {
            errs.push("num_epochs must be positive".to_string());
        }
        // burn's schedulers require initial_lr in (0, 1]; the check is here so
        // the message names the field rather than surfacing as a bare String
        // from init().
        if !(self.lr > 0.0 && self.lr <= 1.0) {
            errs.push(format!("lr must be in (0, 1], got {}", self.lr));
        }
        if !(0.0..=1.0).contains(&self.min_lr_ratio) {
            errs.push(format!(
                "min_lr_ratio must be in [0, 1], got {}",
                self.min_lr_ratio
            ));
        }
        // A ratio in [0, 1): 0 disables warmup, but 1.0 would spend the whole
        // run ramping and leave nothing for the cosine leg. The run-length guard
        // in `train` catches the derived count against the real dataset; this
        // catches a nonsense ratio before any data is loaded.
        if let Some(ratio) = self.warmup_ratio {
            if !(ratio.is_finite() && (0.0..1.0).contains(&ratio)) {
                errs.push(format!(
                    "warmup_ratio must be in [0, 1) when set (null falls back to \
                     warmup_batches), got {ratio}"
                ));
            }
        }
        if self.weight_decay < 0.0 {
            errs.push("weight_decay must not be negative".to_string());
        }
        if !(self.beta_1 > 0.0 && self.beta_1 < 1.0) {
            errs.push(format!("beta_1 must be in (0, 1), got {}", self.beta_1));
        }
        if !(self.beta_2 > 0.0 && self.beta_2 < 1.0) {
            errs.push(format!("beta_2 must be in (0, 1), got {}", self.beta_2));
        }
        if self.epsilon <= 0.0 {
            errs.push("epsilon must be positive".to_string());
        }
        if self.grad_clip_norm <= 0.0 {
            errs.push("grad_clip_norm must be positive".to_string());
        }
        if !(self.z_loss >= 0.0 && self.z_loss.is_finite()) {
            errs.push(format!(
                "z_loss must be finite and non-negative ({} disables it), got {}",
                0.0, self.z_loss
            ));
        }

        // `ModelConfig::validate` reports every violation rather than the first,
        // so flatten them in as peers instead of nesting one blob inside a
        // bullet.
        if let Err(model_errs) = self.model.validate() {
            errs.extend(model_errs.into_iter().map(|e| format!("model: {e}")));
        }

        if errs.is_empty() {
            Ok(())
        } else {
            bail!("invalid TrainConfig:\n  - {}", errs.join("\n  - "));
        }
    }

    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)
            .with_context(|| format!("writing training config to {}", path.display()))?;
        Ok(())
    }

    pub fn load(path: &std::path::Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .with_context(|| format!("reading training config from {}", path.display()))?;
        serde_json::from_str(&json)
            .with_context(|| format!("parsing training config at {}", path.display()))
    }
}

/// One forward pass and its loss. Shared by both steps so that training and
/// validation cannot drift apart -- if validation computed its loss differently,
/// checkpoint selection would be optimizing something other than what it reports.
///
/// The second return is the z-loss penalty, `None` unless `z_loss > 0`. It is
/// returned beside [`LmOutput`] rather than folded into it because it is not a
/// reported quantity: only the gradient sees it. `None` also means the extra
/// reduction over the `[batch, seq, vocab]` logits is not run at all, which is
/// what keeps the default path exactly as cheap as it was.
fn lm_step<B: Backend>(
    model: &QuarkLm<B>,
    batch: TokenBatch<B>,
    z_loss: f32,
) -> (LmOutput<B>, Option<Tensor<B, 1>>) {
    let logits = model.forward(batch.input);
    // From the same logits the loss is computed from -- the penalty is only
    // meaningful if it constrains the tensor that actually reaches the softmax.
    let penalty =
        (z_loss > 0.0).then(|| masked_z_penalty(logits.clone(), batch.score_mask.clone()));
    (
        masked_cross_entropy(logits, batch.target, batch.score_mask),
        penalty,
    )
}

/// Accumulates `sum(g^2)` and the element count across every parameter
/// gradient, staying on the device.
///
/// Visiting is the only way to reach gradients of every rank at once: the
/// `Tensor<B, D>` rank is a const generic, so a loop cannot hold them, but
/// `visit_float` is generic over `D` and burn calls it once per parameter. Each
/// gradient is reduced to a rank-1 scalar immediately and the scalars are summed
/// together, so nothing rank-shaped needs storing and nothing is read back to
/// the host here -- the single scalar rides the [`LmOutput`] transaction that
/// was already happening.
struct GradSumSquares<'a, B: AutodiffBackend> {
    grads: &'a B::Gradients,
    sum_sq: Option<Tensor<B::InnerBackend, 1>>,
    numel: usize,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for GradSumSquares<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        // `grad`, not `grad_remove`: the gradients are borrowed here and handed
        // to `TrainOutput::new` intact afterwards.
        let Some(grad) = param.val().grad(self.grads) else {
            // A parameter the loss does not depend on has no gradient. It
            // contributes no squares, and must contribute no elements either --
            // counting them would divide by a denominator the numerator never
            // saw and silently deflate the RMS.
            return;
        };
        self.numel += grad.shape().num_elements();
        let sum_sq = grad.powf_scalar(2.0).sum();
        self.sum_sq = Some(match self.sum_sq.take() {
            Some(acc) => acc + sum_sq,
            None => sum_sq,
        });
    }
}

impl<B: AutodiffBackend> QuarkLm<B> {
    /// RMS of this step's gradient over every parameter, as a device scalar.
    ///
    /// `unscale` undoes the `grad_accumulation` division that
    /// [`TrainStep::step`] applies before `backward`, so the number reported is
    /// the RMS of the gradient of the *unscaled* objective. That is the
    /// interpretable one: with `grad_accumulation = 1` it is exactly what the
    /// optimizer receives, and above 1 it approximates it, since the accumulated
    /// sum of `N` micro-gradients each divided by `N` is their mean. The
    /// approximation is only loose to the extent that micro-batch gradients
    /// disagree with each other.
    fn grad_rms(&self, grads: &B::Gradients, unscale: f64) -> Option<Tensor<B, 1>> {
        let mut visitor = GradSumSquares::<B> {
            grads,
            sum_sq: None,
            numel: 0,
        };
        self.visit(&mut visitor);

        let sum_sq = visitor.sum_sq?;
        debug_assert!(
            visitor.numel > 0,
            "a gradient exists but covers no elements"
        );
        let rms = sum_sq.div_scalar(visitor.numel as f64).sqrt();
        // Back to the autodiff backend, which is what `LmOutput<B>` holds. The
        // value is a measurement, not part of any graph -- nothing differentiates
        // it, and `from_inner` attaches no history.
        Some(Tensor::from_inner(rms.mul_scalar(unscale)))
    }
}

impl<B: AutodiffBackend> TrainStep for QuarkLm<B> {
    type Input = TokenBatch<B>;
    type Output = LmOutput<B>;

    fn step(&self, batch: TokenBatch<B>) -> TrainOutput<LmOutput<B>> {
        let (mut item, penalty) = lm_step(self, batch, self.z_loss());

        // The optimizer descends cross-entropy *plus* the z penalty; `item`
        // keeps the cross-entropy alone. Splitting them is what lets the
        // reported loss stay comparable to runs trained without the penalty,
        // and keeps `MetricCheckpointingStrategy` selecting on the quantity we
        // care about rather than on a regularizer.
        let objective = match penalty {
            Some(z) => item.loss.clone() + z.mul_scalar(self.z_loss()),
            None => item.loss.clone(),
        };

        // Divided, because burn's accumulator *sums*: `GradientsAccumulator`
        // combines with `grad.add(new)` (`burn-optim/src/optim/grad_accum.rs`),
        // and `masked_cross_entropy` already returns a per-token *mean*. Summing
        // N means gives N x the mean of the batch the N micro-batches make up,
        // so without this the optimizer sees a gradient N times too large.
        //
        // AdamW would largely absorb a uniform factor -- `m/sqrt(v)` is scale
        // invariant -- which is exactly what makes this worth being careful
        // about: gradient clipping is *not* invariant. burn clips inside
        // `OptimizerAdaptor::step`, on the accumulated sum, so leaving the
        // factor in would quietly reduce the effective `grad_clip_norm` to
        // `grad_clip_norm / grad_accumulation` and make trading `batch_size` for
        // `grad_accumulation` change the run instead of preserving it.
        //
        // Only the gradient is scaled: `item` keeps the unscaled loss, so the
        // reported metric stays the true per-token mean.
        let scaled = objective.div_scalar(self.grad_accumulation() as f64);
        let grads = scaled.backward();

        // Measured from the same gradients the optimizer is about to receive,
        // and before `TrainOutput::new` consumes them.
        item.grad_rms = self.grad_rms(&grads, self.grad_accumulation() as f64);

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> InferenceStep for QuarkLm<B> {
    type Input = TokenBatch<B>;
    type Output = LmOutput<B>;

    /// No penalty here, and not because it would be awkward: z-loss is a
    /// training-time regularizer, and validation reports the quantity
    /// checkpoints are selected on. Adding it would make the valid loss depend
    /// on a hyperparameter rather than on the model.
    fn step(&self, batch: TokenBatch<B>) -> LmOutput<B> {
        lm_step(self, batch, 0.0).0
    }
}

/// Warmup, then cosine decay, as a product of two schedulers.
///
/// The linear leg is a **multiplier**, not a learning rate: burn's default
/// `SchedulerReduction::Prod` multiplies the legs together, so ramping linearly
/// from ~0 to 1.0 over `warmup_batches` scales the cosine leg into a warmup. The
/// linear scheduler clamps at its `final_lr` forever afterwards, so past warmup
/// the product is exactly the cosine value.
///
/// Warmup matters more here than it would at 124M: Adam's second-moment estimate
/// is meaningless for its first few dozen steps, and at 3e-3 an unwarmed step
/// off a random initialization is large enough to leave the basin entirely.
fn lr_scheduler_config(config: &TrainConfig, total_batches: usize) -> ComposedLrSchedulerConfig {
    let cosine = CosineAnnealingLrSchedulerConfig::new(config.lr, total_batches)
        .with_min_lr(config.min_lr());

    let composed = ComposedLrSchedulerConfig::new().cosine(cosine);

    // Resolve `warmup_ratio` (if set) against the run length here, so every
    // caller -- the training loop and the schedule tests alike -- sees the same
    // warmup and the ratio is the single source of truth.
    let warmup_batches = config.effective_warmup_batches(total_batches);

    if warmup_batches == 0 {
        // `LinearLrSchedulerConfig::init` rejects num_iters == 0, so a
        // zero-length warmup has to be an absent leg rather than an empty one.
        return composed;
    }

    // The ramp must start above zero (burn requires initial_lr > 0); one
    // warmup-batch's worth of the ramp is the natural first rung.
    let start = 1.0 / warmup_batches as f64;
    composed.linear(LinearLrSchedulerConfig::new(start, 1.0, warmup_batches))
}

/// The name burn's `LossMetric` logs under, and the name its default
/// checkpointing strategy selects on. Both sides have to agree on this string or
/// the best checkpoint is chosen by a metric nobody recorded.
const LOSS_METRIC: &str = "Loss";

/// The epoch whose mean validation loss was lowest, read back from the metric
/// logs in `artifact_dir`.
///
/// This deliberately re-derives what burn's checkpointing strategy already
/// decided, because burn does not report it: `LearningResult` carries the model,
/// not the epoch it came from. The two agree by construction -- the default
/// strategy is
/// `MetricCheckpointingStrategy::new(&LossMetric, Aggregate::Mean, Direction::Lowest, Split::Valid)`
/// (`burn-train/src/learner/supervised/paradigm.rs`), and `LearnerSummary` reads
/// the same logs with the same `Aggregate::Mean` per epoch
/// (`burn-train/src/learner/summary.rs`).
///
/// `None` when no epoch can be named -- no logs, or every epoch diverged to NaN.
/// The caller has to cope rather than pick a wrong one.
fn best_valid_loss_epoch(artifact_dir: &Path) -> Option<usize> {
    let summary = LearnerSummary::new(artifact_dir, &[LOSS_METRIC]).ok()?;
    let loss = summary
        .metrics
        .valid
        .iter()
        .find(|m| m.name == LOSS_METRIC)?;

    loss.entries
        .iter()
        // A diverged epoch is not a candidate. Filtered rather than ordered,
        // because NaN has no order: `partial_cmp` returns `None` against it, and
        // a comparator that unwraps would panic on exactly the run that needs
        // this most.
        .filter(|e| !e.value.is_nan())
        .min_by(|a, b| {
            a.value
                .partial_cmp(&b.value)
                .expect("NaN entries are filtered out above")
        })
        .map(|e| e.step)
}

/// The highest epoch number already recorded in `artifact_dir`'s metric logs,
/// or `None` if it holds none.
///
/// This mirrors `FileMetricLogger::epochs` (`burn-train/src/logger/metric.rs`),
/// deliberately and exactly: one level of split directories (`train/`, `valid/`),
/// then `epoch-<n>` beneath them, and the *maximum* `n` found. What burn will
/// read is what this has to see, or the guard below would pass a directory burn
/// then reads more from.
fn recorded_epochs(artifact_dir: &Path) -> Option<usize> {
    let mut max_epoch = None;
    // A directory that cannot be listed records nothing, which is the same
    // answer as an empty one: this is a guard, and `run` fails on its own terms
    // a few lines later if the path is unusable.
    for split in std::fs::read_dir(artifact_dir).ok()?.flatten() {
        if !split.path().is_dir() {
            continue;
        }
        let Ok(epochs) = std::fs::read_dir(split.path()) else {
            continue;
        };
        for epoch in epochs.flatten() {
            if !epoch.path().is_dir() {
                continue;
            }
            let name = epoch.file_name();
            let Some(n) = name
                .to_str()
                .and_then(|n| n.strip_prefix("epoch-"))
                .and_then(|n| n.parse::<usize>().ok())
            else {
                continue;
            };
            max_epoch = Some(max_epoch.map_or(n, |m: usize| m.max(n)));
        }
    }
    max_epoch
}

/// Refuse to start a fresh run in a directory that already holds one.
///
/// This is not tidiness. burn's file metric logger truncates per epoch, so a
/// shorter run into a used directory overwrites the epochs it reaches and
/// *leaves the rest*, and then every consumer reads the union as one run:
///
///  * `FileMetricLogger::epochs` returns the maximum `epoch-<n>` on disk, so
///    `LearnerSummary` reports `Total Epochs: <n>` and prints the old run's
///    losses in the new run's summary table.
///  * `MetricCheckpointingStrategy` selects through
///    `NumericMetricsAggregate::find_epoch`, which walks epochs from 1 until one
///    is missing and takes the best. A stale epoch that beats the new run's
///    means the new model is never saved as best.
///  * [`best_valid_loss_epoch`] then names that stale epoch, and `run` loads
///    `checkpoint/model-<stale>` -- weights from *the previous architecture*.
///
/// This is not hypothetical: the `quark_22m` run reported in issue #6 printed
/// `Total Epochs: 10` for a `num_epochs: 1` config, with epochs 2-10 carrying
/// the `quark_3m_dense` numbers from the run before it (`docs/RESULTS.md` §5).
/// It loaded the right checkpoint only because its single epoch happened to beat
/// every stale one. That is luck, and luck is not a mechanism.
fn refuse_to_merge_runs(artifact_dir: &Path, resume_from_epoch: Option<usize>) -> Result<()> {
    // Resuming *wants* the previous run's logs: they are the same run.
    if resume_from_epoch.is_some() {
        return Ok(());
    }
    if let Some(epochs) = recorded_epochs(artifact_dir) {
        bail!(
            "{} already holds metric logs for {epochs} epoch(s) from an earlier run. Training \
             into it would merge the two: burn reads every `epoch-<n>` it finds, so the summary \
             would report both runs as one, and the best-checkpoint search could select -- and \
             load -- a checkpoint from the earlier model. Remove the directory, point \
             --artifact-dir somewhere else, or pass --resume-from-epoch to continue that run \
             deliberately.",
            artifact_dir.display()
        );
    }
    Ok(())
}

/// Train, and return the best model by validation loss.
///
/// Deliberately generic over the backend rather than hardcoding one: the issue
/// asks for wgpu as the primary backend, but the tests run this same function on
/// ndarray. A harness that only compiles against the GPU is a harness that only
/// gets tested by the person who has the GPU.
pub fn run<B: AutodiffBackend>(
    config: TrainConfig,
    device: B::Device,
) -> Result<QuarkLm<B::InnerBackend>> {
    config.validate()?;
    refuse_to_merge_runs(&config.artifact_dir, config.resume_from_epoch)?;

    let train_shard = Arc::new(
        Shard::open(&config.train_shard)
            .with_context(|| format!("opening train shard {}", config.train_shard.display()))?,
    );
    let valid_shard = Arc::new(
        Shard::open(&config.valid_shard)
            .with_context(|| format!("opening valid shard {}", config.valid_shard.display()))?,
    );

    // A shard tokenized against a different vocabulary would index the embedding
    // table out of bounds -- or worse, stay in bounds and train on nonsense.
    for (name, shard) in [("train", &train_shard), ("valid", &valid_shard)] {
        let got = shard.meta().vocab_size;
        if got != config.model.vocab_size {
            bail!(
                "{name} shard was tokenized with vocab_size {got} but the model has {}; \
                 re-run `quark prepare` with the matching tokenizer",
                config.model.vocab_size
            );
        }
    }

    let train_dataset = TokenDataset::train(train_shard, config.seq_len);
    let valid_dataset = TokenDataset::train(valid_shard, config.seq_len);
    if train_dataset.len() == 0 {
        bail!(
            "train shard holds fewer than {} tokens, so it yields no windows",
            config.seq_len + 1
        );
    }
    if valid_dataset.len() == 0 {
        bail!(
            "valid shard holds fewer than {} tokens, so it yields no windows",
            config.seq_len + 1
        );
    }

    let batches_per_epoch = config.batches_per_epoch(train_dataset.len());
    let total_batches = config.total_batches(train_dataset.len());
    let warmup_batches = config.effective_warmup_batches(total_batches);
    if warmup_batches >= total_batches {
        bail!(
            "warmup of {warmup_batches} batches must be less than the {total_batches} batches this \
             run will perform, or the learning rate never reaches its peak (warmup_ratio {:?}, \
             warmup_batches {})",
            config.warmup_ratio,
            config.warmup_batches
        );
    }

    tracing::info!(
        train_windows = train_dataset.len(),
        valid_windows = valid_dataset.len(),
        batches_per_epoch,
        total_batches,
        tokens_per_optimizer_step = config.batch_size * config.seq_len * config.grad_accumulation,
        params = config.model.param_count(),
        "starting training"
    );

    // Written before the first step, so an interrupted run is still reproducible.
    config.save(&config.artifact_dir.join("config.json"))?;

    let dataloader_train = DataLoaderBuilder::new(TokenBatcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .set_device(device.clone())
        .build(train_dataset);

    let dataloader_valid = DataLoaderBuilder::new(TokenBatcher)
        .batch_size(config.batch_size)
        // Not shuffled: validation is a sum over a fixed set, so the order
        // changes nothing except the reproducibility of the log.
        .num_workers(config.num_workers)
        .set_device(device.clone())
        .build(valid_dataset);

    // The model has to be told the accumulation factor and the z-loss
    // coefficient because burn implements `TrainStep` *for the model*, and hands
    // `step` nothing but the batch. See `QuarkLm::grad_accumulation`.
    let model = QuarkLm::<B>::new(config.model.clone(), &device)
        .with_grad_accumulation(config.grad_accumulation)
        .with_z_loss(config.z_loss);

    let optimizer = AdamWConfig::new()
        .with_beta_1(config.beta_1)
        .with_beta_2(config.beta_2)
        .with_epsilon(config.epsilon)
        .with_weight_decay(config.weight_decay)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(config.grad_clip_norm)))
        .init();

    let lr_scheduler = lr_scheduler_config(&config, total_batches)
        .init()
        .map_err(|e| anyhow::anyhow!("building the learning rate schedule: {e}"))?;

    let learner = burn::train::Learner::new(model, optimizer, lr_scheduler);

    let mut training = SupervisedTraining::new(
        config.artifact_dir.clone(),
        dataloader_train,
        dataloader_valid,
    )
    .num_epochs(config.num_epochs)
    .grads_accumulation(config.grad_accumulation)
    .with_file_checkpointer(CompactRecorder::new())
    // "Loss" on the *valid* split is not optional decoration: the default
    // checkpointing strategy selects on it by name
    // (`MetricCheckpointingStrategy::new(&LossMetric::<B>::new(), .., Split::Valid)`),
    // so without this registration nothing would ever be chosen as best.
    .metric_train_numeric(LossMetric::new())
    .metric_valid_numeric(LossMetric::new())
    .metric_train_numeric(TokenPerplexityMetric::new())
    .metric_valid_numeric(TokenPerplexityMetric::new())
    .metric_train_numeric(LearningRateMetric::new())
    // Train only, and necessarily: validation computes no gradient, so
    // `Adaptor<GradRmsInput>` has nothing to hand this and says so by panicking.
    // Registering it on the valid split would be a request for a number that
    // does not exist.
    .metric_train_numeric(GradRmsMetric::new())
    .summary();

    if let Some(epoch) = config.resume_from_epoch {
        training = training.checkpoint(epoch);
    }

    let LearningResult { model: last, .. } = training.launch(learner);

    // `LearningResult.model` is the model as it stood after the *last* epoch --
    // `strategy.rs` returns `learner.model()` -- which is not the best one
    // whenever the run overfits, and overfitting a 3M model on WikiText-103 is
    // the expected case rather than a remote one. burn does keep the best
    // checkpoint (the default `MetricCheckpointingStrategy` selects on mean
    // valid loss, and never deletes the epoch it selected), but it does not tell
    // us which epoch that was, so the epoch is read back from the same logs.
    let model = match best_valid_loss_epoch(&config.artifact_dir) {
        Some(epoch) => {
            // The path `FileCheckpointer` writes to: `<dir>/checkpoint/model-<epoch>`,
            // with the recorder supplying the extension.
            let path = config
                .artifact_dir
                .join("checkpoint")
                .join(format!("model-{epoch}"));

            let best = QuarkLm::<B::InnerBackend>::new(config.model.clone(), &device)
                .load_file(path.clone(), &CompactRecorder::new(), &device)
                .with_context(|| {
                    format!(
                        "loading the best checkpoint (epoch {epoch}) from {}",
                        path.display()
                    )
                })?;
            tracing::info!(epoch, "recovered the best epoch by validation loss");
            best
        }
        // Nothing to select on: no metric logs, or every epoch diverged. The
        // last model is then the only honest answer, and the log says which one
        // this is rather than claiming a best that was never chosen.
        None => {
            tracing::warn!(
                "no validation loss was recorded, so no best epoch could be chosen; \
                 falling back to the model from the final epoch"
            );
            last
        }
    };

    let final_path = config.artifact_dir.join("model");
    model
        .clone()
        .save_file(final_path.clone(), &CompactRecorder::new())
        .with_context(|| format!("saving trained model to {}", final_path.display()))?;
    tracing::info!(path = %final_path.display(), "saved the trained model");

    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{data::ShardWriter, test_util::TestAutodiffBackend};
    use burn::{
        lr_scheduler::LrScheduler, prelude::ElementConversion, tensor::Distribution, tensor::Int,
        tensor::Tensor,
    };

    type B = TestAutodiffBackend;

    fn tiny_config(dir: &std::path::Path) -> TrainConfig {
        let model = ModelConfig::tiny();
        TrainConfig {
            seq_len: 16,
            batch_size: 2,
            grad_accumulation: 1,
            num_epochs: 1,
            warmup_batches: 2,
            num_workers: 1,
            train_shard: dir.join("train.bin"),
            valid_shard: dir.join("valid.bin"),
            artifact_dir: dir.join("run"),
            model,
            ..Default::default()
        }
    }

    /// A shard of `n` pseudo-random tokens drawn from `vocab`.
    fn write_shard(path: &std::path::Path, n: usize, vocab: usize) {
        let mut w = ShardWriter::create(path, vocab, 0).unwrap();
        // Deterministic and cheap: a linear congruential walk, not randomness we
        // need to be good, only tokens we need to be varied.
        let mut x: u64 = 12345;
        let tokens: Vec<u32> = (0..n)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 33) as usize % vocab) as u32
            })
            .collect();
        w.push_document("word ".repeat(n).trim(), &tokens).unwrap();
        w.finish().unwrap();
    }

    #[test]
    fn the_reference_config_is_valid() {
        TrainConfig::default().validate().unwrap();
    }

    /// Pins what `grad_clip_norm` actually does, because it is not what the name
    /// means anywhere else and the difference is invisible until you look.
    ///
    /// Global-norm clipping -- GPT-2, PaLM, nanoGPT -- computes one coefficient
    /// over every gradient at once and rescales them all by it, which shortens
    /// the update without rotating it. burn's `GradientClipping::Norm` cannot do
    /// that: `clip_gradient` takes a single `Tensor<B, D>`, and `OptimizerAdaptor`
    /// hands it one parameter at a time. So each tensor gets its own coefficient.
    ///
    /// The two are told apart by what survives: with gradients of norm 10 and
    /// 0.1 and a threshold of 1.0, global clipping scales *both* by ~1/10 and the
    /// 100:1 ratio between them is preserved; per-tensor clipping pulls the first
    /// to 1.0, leaves the second alone at 0.1 (it is already under threshold),
    /// and the ratio becomes 10:1. The ratio is the direction of the combined
    /// update, so this asserts the thing that matters rather than a norm.
    ///
    /// If burn ever switches to global clipping, this test fails and the doc on
    /// [`TrainConfig::grad_clip_norm`] -- and docs/RESULTS.md -- need rewriting.
    #[test]
    fn burn_clips_each_parameter_tensor_separately_not_the_global_norm() {
        use burn::grad_clipping::GradientClipping;
        type Inner = <B as AutodiffBackend>::InnerBackend;

        let d = Default::default();
        let clip = GradientClipping::Norm(1.0);
        // Norm is `sqrt(sum(g^2))` over the whole tensor, so a constant tensor of
        // `k/sqrt(numel)` has norm exactly `k`. Two shapes, to make it clear the
        // rank is not what is being tested.
        let flat = |k: f32, n: usize| -> Tensor<Inner, 1> {
            Tensor::ones([n], &d) * (k / (n as f32).sqrt())
        };
        let norm = |t: Tensor<Inner, 1>| -> f32 { t.powi_scalar(2).sum().sqrt().into_scalar() };

        let big = clip.clip_gradient(flat(10.0, 16));
        let small = clip.clip_gradient(flat(0.1, 4));

        let (big, small) = (norm(big), norm(small));
        assert!(
            (big - 1.0).abs() < 1e-3,
            "the over-threshold tensor must be pulled to exactly the threshold, got {big}"
        );
        assert!(
            (small - 0.1).abs() < 1e-3,
            "the under-threshold tensor must be untouched -- a global clip would have shrunk it to ~0.01 -- got {small}"
        );
        assert!(
            (big / small - 10.0).abs() < 0.1,
            "per-tensor clipping must collapse the 100:1 ratio to 10:1, i.e. rotate the update; got {:.1}:1",
            big / small
        );
    }

    /// Writes the metric log burn's `FileMetricLogger` writes: one entry per
    /// line under `<split>/epoch-<n>/<Metric>.log`, each `"<value>,<count>"` --
    /// the serialization of `NumericEntry::Aggregated`, which is what a real run
    /// records (one entry per batch, `count` being that batch's size).
    fn write_valid_loss_log(
        artifact_dir: &std::path::Path,
        epoch: usize,
        batches: &[(f64, usize)],
    ) {
        let dir = artifact_dir.join("valid").join(format!("epoch-{epoch}"));
        std::fs::create_dir_all(&dir).unwrap();
        let body = batches
            .iter()
            .map(|(value, count)| format!("{value},{count}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.join("Loss.log"), body).unwrap();
    }

    /// The point of the whole exercise: the best epoch is the one with the
    /// lowest validation loss, which is *not* in general the last one. A run
    /// that overfits after epoch 2 must still hand back epoch 2.
    #[test]
    fn the_best_epoch_is_the_lowest_valid_loss_not_the_last() {
        let dir = tempfile::tempdir().unwrap();
        write_valid_loss_log(dir.path(), 1, &[(1.0, 4)]);
        write_valid_loss_log(dir.path(), 2, &[(0.5, 4)]);
        write_valid_loss_log(dir.path(), 3, &[(2.0, 4)]);

        assert_eq!(best_valid_loss_epoch(dir.path()), Some(2));
    }

    /// Epochs are compared by their mean over the epoch's batches, because that
    /// is the aggregate burn's checkpointing strategy selects on
    /// (`Aggregate::Mean`). Selecting on, say, the last batch of each epoch
    /// would pick epoch 1 here, and so would disagree with the checkpoint burn
    /// actually kept.
    #[test]
    fn epochs_are_compared_by_their_mean_loss() {
        let dir = tempfile::tempdir().unwrap();
        // Mean 2.0, but ends at 0.1.
        write_valid_loss_log(dir.path(), 1, &[(3.9, 1), (0.1, 1)]);
        // Mean 1.0, and ends higher.
        write_valid_loss_log(dir.path(), 2, &[(1.0, 1), (1.0, 1)]);

        assert_eq!(best_valid_loss_epoch(dir.path()), Some(2));
    }

    /// That mean is weighted by each batch's token count, not a mean of means:
    /// the last batch of an epoch is usually smaller, and letting it count as
    /// much as a full one would pick a different epoch than burn did.
    ///
    /// Unweighted, epoch 1 averages (0.1 + 1.9)/2 = 1.0 and would tie epoch 2 at
    /// 1.0; weighted by count it is (0.1*1 + 1.9*99)/100 = 1.882, so epoch 2
    /// wins. Asserting the winner is epoch 2 is therefore exactly the assertion
    /// that the weighting happens.
    #[test]
    fn the_mean_is_weighted_by_batch_size() {
        let dir = tempfile::tempdir().unwrap();
        write_valid_loss_log(dir.path(), 1, &[(0.1, 1), (1.9, 99)]);
        write_valid_loss_log(dir.path(), 2, &[(1.0, 100)]);

        assert_eq!(best_valid_loss_epoch(dir.path()), Some(2));
    }

    /// A diverged epoch must not win by being unordered. NaN compares `None`
    /// against everything, so a min that unwrapped `partial_cmp` would panic on
    /// precisely the run that most needs a checkpoint recovered.
    #[test]
    fn a_diverged_epoch_is_not_a_candidate() {
        let dir = tempfile::tempdir().unwrap();
        write_valid_loss_log(dir.path(), 1, &[(1.0, 4)]);
        write_valid_loss_log(dir.path(), 2, &[(f64::NAN, 4)]);

        assert_eq!(best_valid_loss_epoch(dir.path()), Some(1));
    }

    /// No logs means no answer -- not epoch 0, and not a panic. `run` falls back
    /// to the final model and says so.
    #[test]
    fn an_empty_artifact_dir_names_no_best_epoch() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(best_valid_loss_epoch(dir.path()), None);
        assert_eq!(
            best_valid_loss_epoch(&dir.path().join("does-not-exist")),
            None
        );
    }

    /// The failure `refuse_to_merge_runs` exists to prevent, demonstrated rather
    /// than asserted about: this is what a 1-epoch run into the directory of a
    /// finished 10-epoch run would select.
    ///
    /// Epoch 1 is the new run's -- burn's `FileLogger` opens with
    /// `.truncate(true)`, so it overwrote what was there. Epochs 2-10 are the old
    /// run's, untouched, and burn reads them as if they were this run's. The best
    /// epoch is then epoch 3, from a model that no longer exists, and `run` would
    /// load `checkpoint/model-3` into the new architecture.
    #[test]
    fn without_the_guard_a_stale_epoch_wins_the_best_checkpoint_search() {
        let dir = tempfile::tempdir().unwrap();
        // The new run: one epoch, and a good one.
        write_valid_loss_log(dir.path(), 1, &[(3.361, 512)]);
        // The old run's tail, still on disk.
        for (epoch, loss) in [(2, 3.9), (3, 3.2), (4, 4.1)] {
            write_valid_loss_log(dir.path(), epoch, &[(loss, 512)]);
        }

        assert_eq!(best_valid_loss_epoch(dir.path()), Some(3));
        assert_eq!(recorded_epochs(dir.path()), Some(4));
    }

    /// So a fresh run into a used directory is refused, and the message says what
    /// to do instead. The alternative is the run in issue #6: `num_epochs: 1`,
    /// `Total Epochs: 10`.
    #[test]
    fn a_fresh_run_refuses_a_used_artifact_dir() {
        let dir = tempfile::tempdir().unwrap();
        write_valid_loss_log(dir.path(), 1, &[(1.0, 4)]);

        let err = refuse_to_merge_runs(dir.path(), None)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("earlier run") && err.contains("--resume-from-epoch"),
            "the error must name the cause and the way out; got: {err}"
        );
    }

    /// Resuming reads the same logs on purpose -- it is the same run. The guard
    /// must not stand in front of the one case that wants what it forbids.
    #[test]
    fn resuming_is_not_refused() {
        let dir = tempfile::tempdir().unwrap();
        write_valid_loss_log(dir.path(), 1, &[(1.0, 4)]);

        refuse_to_merge_runs(dir.path(), Some(1)).unwrap();
    }

    /// And the ordinary case stays ordinary: a directory that does not exist, is
    /// empty, or holds only a previous run's `config.json` is not a merge.
    #[test]
    fn a_directory_without_epoch_logs_is_not_a_used_run() {
        let dir = tempfile::tempdir().unwrap();
        refuse_to_merge_runs(&dir.path().join("does-not-exist"), None).unwrap();
        refuse_to_merge_runs(dir.path(), None).unwrap();

        std::fs::write(dir.path().join("config.json"), "{}").unwrap();
        std::fs::create_dir_all(dir.path().join("checkpoint")).unwrap();
        refuse_to_merge_runs(dir.path(), None).unwrap();
    }

    /// Collects the ids of the 2-D float parameters, so a gradient can be looked
    /// up in [`GradientsParams`] by the id the module actually carries.
    #[derive(Default)]
    struct TwoDimParamIds {
        ids: Vec<burn::module::ParamId>,
    }

    impl<Bk: Backend> burn::module::ModuleVisitor<Bk> for TwoDimParamIds {
        fn visit_float<const D: usize>(&mut self, param: &burn::module::Param<Tensor<Bk, D>>) {
            if D == 2 {
                self.ids.push(param.id);
            }
        }
    }

    /// z-loss has to reach the optimizer and nothing else.
    ///
    /// Both halves are load-bearing. If the penalty did not change the
    /// gradient it would be doing nothing; if it changed the reported loss,
    /// every number a run produces would silently depend on a regularizer
    /// coefficient, and `MetricCheckpointingStrategy` -- which selects on mean
    /// valid `Loss` -- would be choosing checkpoints by it.
    #[test]
    fn z_loss_changes_the_gradient_but_not_the_reported_loss() {
        let device = Default::default();
        // Cloned rather than rebuilt, and forced first, for the reason spelled
        // out in `gradient_accumulation_divides_the_loss_so_the_optimizer_sees_one_batch`.
        let plain = QuarkLm::<B>::new(ModelConfig::tiny(), &device);
        let _ = plain.forward(Tensor::<B, 2, Int>::zeros([1, 4], &device));
        let penalized = plain.clone().with_z_loss(1e-1);

        let batch = || TokenBatch {
            input: Tensor::<B, 2, Int>::zeros([2, 8], &device),
            target: Tensor::<B, 2, Int>::ones([2, 8], &device),
            score_mask: Tensor::<B, 2>::ones([2, 8], &device),
        };

        let out_plain = TrainStep::step(&plain, batch());
        let out_penalized = TrainStep::step(&penalized, batch());

        let loss =
            |o: &TrainOutput<LmOutput<B>>| -> f32 { o.item.loss.clone().into_scalar().elem() };
        assert_eq!(
            loss(&out_plain),
            loss(&out_penalized),
            "the reported loss must be cross-entropy alone"
        );

        let mut visitor = TwoDimParamIds::default();
        plain.visit(&mut visitor);
        let id = *visitor.ids.first().expect("the model has 2-D parameters");

        let norm = |grads: &burn::optim::GradientsParams| -> f32 {
            let g = grads
                .get::<<B as AutodiffBackend>::InnerBackend, 2>(id)
                .expect("that parameter has a gradient");
            g.powf_scalar(2.0).sum().sqrt().into_scalar().elem()
        };
        let (a, b) = (norm(&out_plain.grads), norm(&out_penalized.grads));
        assert!(a > 0.0, "the test batch must produce a gradient");
        assert!(
            (a - b).abs() > a * 1e-6,
            "the penalty must reach the gradient: {a} vs {b}"
        );
    }

    /// `grad_accumulation` must divide the loss, because burn's accumulator
    /// *sums*: `burn-optim/src/optim/grad_accum.rs` accumulates with
    /// `grad.add(new)`. Without the division, N micro-batches hand the optimizer
    /// N x the gradient of the batch they are supposed to add up to.
    ///
    /// AdamW would mostly absorb a uniform factor -- `m/sqrt(v)` is scale
    /// invariant -- but gradient clipping does not. burn clips inside
    /// `OptimizerAdaptor::step`, on the already-accumulated sum, so an unscaled
    /// loss silently turns `grad_clip_norm` into `grad_clip_norm /
    /// grad_accumulation`. That falsifies the invariant `TrainConfig` documents:
    /// "raise this by the same factor you lower `batch_size` by, and the
    /// optimizer sees the same batch". Following that advice to escape an OOM
    /// would quietly clip harder, so the smaller-batch run is not the run it
    /// claims to be.
    #[test]
    fn gradient_accumulation_divides_the_loss_so_the_optimizer_sees_one_batch() {
        let device = Default::default();
        let config = ModelConfig::tiny();
        // Cloned, not rebuilt: the two models must share parameter *ids* for
        // their gradients to be comparable, and `new` would draw fresh ones.
        let plain = QuarkLm::<B>::new(config.clone(), &device);
        // burn's `Param` initializes lazily, and cloning an *uninitialized* one
        // draws a fresh random value rather than copying
        // (`burn-core/src/module/param/base.rs`). A forward pass forces every
        // parameter to exist first, so the clone is the same model rather than a
        // different one -- otherwise this compares two unrelated gradients and
        // the ratio it asserts means nothing.
        let _ = plain.forward(Tensor::<B, 2, Int>::zeros([1, 4], &device));
        let accumulating = plain.clone().with_grad_accumulation(4);

        let batch = || TokenBatch {
            input: Tensor::<B, 2, Int>::zeros([2, 8], &device),
            target: Tensor::<B, 2, Int>::ones([2, 8], &device),
            score_mask: Tensor::<B, 2>::ones([2, 8], &device),
        };

        let out_plain = TrainStep::step(&plain, batch());
        let out_accumulating = TrainStep::step(&accumulating, batch());

        let mut visitor = TwoDimParamIds::default();
        plain.visit(&mut visitor);
        let id = *visitor.ids.first().expect("the model has 2-D parameters");

        let norm = |grads: &burn::optim::GradientsParams| -> f32 {
            let g = grads
                .get::<<B as AutodiffBackend>::InnerBackend, 2>(id)
                .expect("that parameter has a gradient");
            g.powf_scalar(2.0).sum().sqrt().into_scalar().elem()
        };

        let plain_norm = norm(&out_plain.grads);
        let accumulating_norm = norm(&out_accumulating.grads);
        assert!(plain_norm > 0.0, "the test batch must produce a gradient");
        // One of four micro-batches carries a quarter of the step's gradient.
        let want = plain_norm / 4.0;
        assert!(
            (accumulating_norm - want).abs() <= want * 1e-4,
            "grad_accumulation = 4 must scale the gradient to a quarter: \
             got {accumulating_norm}, want {want} (unscaled would be {plain_norm})"
        );

        // The *reported* loss stays the true per-token mean: the division is a
        // property of the optimizer step, not of the number the metric shows.
        let loss =
            |o: &TrainOutput<LmOutput<B>>| -> f32 { o.item.loss.clone().into_scalar().elem() };
        assert_eq!(loss(&out_plain), loss(&out_accumulating));
    }

    /// `grad_rms` measures gradients that have already been divided by
    /// `grad_accumulation`, and multiplies the factor back out. That undo is the
    /// one place the number can come out wrong while still looking entirely
    /// reasonable -- a run at `grad_accumulation = 4` would simply report a grad
    /// RMS four times too small, with nothing to compare it against.
    ///
    /// The property is exact: dividing the objective by `N` divides every
    /// gradient by `N`, so a correctly unscaled RMS cannot depend on `N` at all.
    /// The test asserts the invariance and names the wrong answer, so a failure
    /// says which way it broke.
    #[test]
    fn grad_rms_is_reported_free_of_the_grad_accumulation_scaling() {
        let device = Default::default();
        // Cloned after a forward pass, for the reason spelled out in
        // `gradient_accumulation_divides_the_loss_so_the_optimizer_sees_one_batch`:
        // burn's params initialize lazily and cloning an uninitialized one draws
        // a fresh value.
        let plain = QuarkLm::<B>::new(ModelConfig::tiny(), &device);
        let _ = plain.forward(Tensor::<B, 2, Int>::zeros([1, 4], &device));
        let accumulating = plain.clone().with_grad_accumulation(4);

        let rms = |model: &QuarkLm<B>| -> f32 {
            let batch = TokenBatch {
                input: Tensor::<B, 2, Int>::zeros([2, 8], &device),
                target: Tensor::<B, 2, Int>::ones([2, 8], &device),
                score_mask: Tensor::<B, 2>::ones([2, 8], &device),
            };
            let out = TrainStep::step(model, batch);
            out.item
                .grad_rms
                .expect("the training step measures the gradient it just computed")
                .into_scalar()
                .elem()
        };

        let plain_rms = rms(&plain);
        let accumulating_rms = rms(&accumulating);

        assert!(
            plain_rms > 0.0 && plain_rms.is_finite(),
            "the test batch must produce a gradient, got {plain_rms}"
        );
        assert!(
            (accumulating_rms - plain_rms).abs() <= plain_rms * 1e-4,
            "grad_rms must not depend on grad_accumulation: got {accumulating_rms} at 4 \
             vs {plain_rms} at 1. Without the unscale it would read {}",
            plain_rms / 4.0
        );
    }

    /// Validation has no gradient, so it must report no gradient RMS rather than
    /// a zero -- "not measured" and "measured zero" are different claims, and the
    /// second one would plot as a grad-RMS collapse that never happened.
    #[test]
    fn the_validation_step_reports_no_grad_rms() {
        let device = Default::default();
        let model = QuarkLm::<B>::new(ModelConfig::tiny(), &device);
        let batch = TokenBatch {
            input: Tensor::<B, 2, Int>::zeros([2, 8], &device),
            target: Tensor::<B, 2, Int>::ones([2, 8], &device),
            score_mask: Tensor::<B, 2>::ones([2, 8], &device),
        };
        assert!(InferenceStep::step(&model, batch).grad_rms.is_none());
    }

    /// The RoPE tables are sized to `max_seq_len`; a longer window would index
    /// past them. Catching it in `validate` turns a panic mid-run into a message.
    #[test]
    fn a_window_longer_than_the_model_supports_is_rejected() {
        let mut c = TrainConfig::default();
        c.seq_len = c.model.max_seq_len + 1;
        let err = c.validate().unwrap_err().to_string();
        assert!(err.contains("max_seq_len"), "{err}");
    }

    #[test]
    fn config_survives_a_roundtrip_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        let c = TrainConfig::default();
        c.save(&path).unwrap();
        assert_eq!(TrainConfig::load(&path).unwrap(), c);
    }

    /// The schedule is sized in batches because burn steps it per batch. This
    /// pins the arithmetic that claim rests on: 10 windows at batch size 4 is 3
    /// batches (not 2), and 2 epochs of that is 6.
    #[test]
    fn the_schedule_is_measured_in_batches_including_the_partial_one() {
        let c = TrainConfig {
            batch_size: 4,
            num_epochs: 2,
            ..Default::default()
        };
        assert_eq!(c.batches_per_epoch(10), 3);
        assert_eq!(c.total_batches(10), 6);
        // Grad accumulation must not enter the count: the scheduler advances per
        // batch regardless of when gradients are applied.
        let c = TrainConfig {
            grad_accumulation: 8,
            ..c
        };
        assert_eq!(c.total_batches(10), 6);
    }

    /// The three properties the composed schedule must have, checked against the
    /// real scheduler rather than against my reading of it: it starts far below
    /// peak, it reaches peak at the end of warmup, and it lands on the floor at
    /// the end of training.
    #[test]
    fn the_learning_rate_warms_up_then_decays_to_the_floor() {
        let c = TrainConfig {
            lr: 1e-3,
            min_lr_ratio: 0.1,
            warmup_batches: 10,
            ..Default::default()
        };
        let total = 100;
        let mut s = lr_scheduler_config(&c, total).init().unwrap();

        let first = LrScheduler::step(&mut s);
        assert!(
            first < c.lr / 5.0,
            "warmup must start well below peak, got {first}"
        );

        let mut peak: f64 = first;
        for _ in 1..=c.warmup_batches {
            peak = peak.max(LrScheduler::step(&mut s));
        }
        assert!(
            (peak - c.lr).abs() < c.lr * 0.05,
            "the ramp should reach ~peak by the end of warmup: {peak} vs {}",
            c.lr
        );

        let mut last = peak;
        for _ in c.warmup_batches..total {
            last = LrScheduler::step(&mut s);
        }
        assert!(
            (last - c.min_lr()).abs() < c.min_lr() * 0.05,
            "cosine should land on min_lr at the final batch: {last} vs {}",
            c.min_lr()
        );
    }

    /// Warmup is a *multiplier* under `Prod`. If the linear leg were treated as
    /// a learning rate in its own right the product would be wrong everywhere,
    /// so check that past warmup the schedule is the bare cosine.
    #[test]
    fn past_warmup_the_schedule_is_exactly_the_cosine_leg() {
        let c = TrainConfig {
            lr: 1e-3,
            warmup_batches: 5,
            ..Default::default()
        };
        let total = 50;
        let mut composed = lr_scheduler_config(&c, total).init().unwrap();
        let mut cosine = CosineAnnealingLrSchedulerConfig::new(c.lr, total)
            .with_min_lr(c.min_lr())
            .init()
            .unwrap();

        for i in 0..total {
            let a = LrScheduler::step(&mut composed);
            let b = LrScheduler::step(&mut cosine);
            if i >= c.warmup_batches {
                assert!(
                    (a - b).abs() < 1e-12,
                    "batch {i}: composed {a} should equal cosine {b} once the ramp clamps at 1.0"
                );
            }
        }
    }

    /// A zero-length warmup must drop the linear leg rather than build one with
    /// `num_iters = 0`, which burn rejects.
    #[test]
    fn a_zero_length_warmup_is_allowed() {
        let c = TrainConfig {
            warmup_batches: 0,
            ..Default::default()
        };
        let mut s = lr_scheduler_config(&c, 10).init().unwrap();
        assert!((LrScheduler::step(&mut s) - c.lr).abs() < 1e-12);
    }

    /// The default, and every config written before `warmup_ratio` existed, must
    /// resolve to the absolute count -- unchanged behaviour, so the reference run
    /// is untouched. The serde half checks a config missing the field loads as
    /// `None` rather than failing.
    #[test]
    fn an_absent_warmup_ratio_keeps_the_absolute_count() {
        let c = TrainConfig::default();
        assert!(c.warmup_ratio.is_none());
        assert_eq!(c.effective_warmup_batches(10_000), c.warmup_batches);

        let mut v: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        v.as_object_mut().unwrap().remove("warmup_ratio");
        let back: TrainConfig = serde_json::from_value(v).unwrap();
        assert!(back.warmup_ratio.is_none());
        assert_eq!(back, c);
    }

    /// A ratio is just a run-relative way to name a batch count: 0.1 of a
    /// 100-batch run must drive the schedule identically to `warmup_batches = 10`,
    /// and it overrides the absolute field even when that field is nonsense.
    #[test]
    fn warmup_ratio_drives_the_schedule_like_the_equivalent_count() {
        let by_ratio = TrainConfig {
            lr: 1e-3,
            warmup_ratio: Some(0.1),
            warmup_batches: 999, // deliberately absurd; the ratio must win
            ..Default::default()
        };
        let by_count = TrainConfig {
            lr: 1e-3,
            warmup_batches: 10,
            ..Default::default()
        };
        let total = 100;
        assert_eq!(by_ratio.effective_warmup_batches(total), 10);

        let mut a = lr_scheduler_config(&by_ratio, total).init().unwrap();
        let mut b = lr_scheduler_config(&by_count, total).init().unwrap();
        for i in 0..total {
            let (x, y) = (LrScheduler::step(&mut a), LrScheduler::step(&mut b));
            assert!((x - y).abs() < 1e-12, "batch {i}: ratio {x} vs count {y}");
        }
    }

    /// The point of the ratio: it is the same *share* of every run, so it does
    /// not drift when the run's shape changes. Quadrupling the epochs quadruples
    /// the warmup batches (an absolute count would have stayed put and become a
    /// quarter of the share), and `grad_accumulation` -- which never enters
    /// `total_batches` -- never moves it.
    #[test]
    fn warmup_ratio_holds_its_share_across_epochs_and_grad_accumulation() {
        let n = 1000; // 100 batches/epoch at batch_size 10
        let short = TrainConfig {
            batch_size: 10,
            num_epochs: 1,
            warmup_ratio: Some(0.05),
            ..Default::default()
        };
        let long = TrainConfig {
            num_epochs: 4,
            ..short.clone()
        };
        assert_eq!(short.total_batches(n), 100);
        assert_eq!(long.total_batches(n), 400);
        assert_eq!(short.effective_warmup_batches(short.total_batches(n)), 5);
        assert_eq!(long.effective_warmup_batches(long.total_batches(n)), 20);

        let more_accum = TrainConfig {
            grad_accumulation: 8,
            ..short.clone()
        };
        assert_eq!(
            more_accum.effective_warmup_batches(more_accum.total_batches(n)),
            5,
            "grad_accumulation is absent from total_batches, so it cannot move the warmup"
        );
    }

    /// The range check fires before any data is touched: a ratio at or above 1.0
    /// would leave no room for the cosine decay, and a non-finite one is nonsense.
    #[test]
    fn validate_rejects_a_warmup_ratio_outside_the_unit_interval() {
        let ok = TrainConfig {
            warmup_ratio: Some(0.02),
            ..Default::default()
        };
        assert!(ok.validate().is_ok());
        for bad in [Some(1.0), Some(-0.1), Some(f64::NAN), Some(f64::INFINITY)] {
            let c = TrainConfig {
                warmup_ratio: bad,
                ..Default::default()
            };
            assert!(c.validate().is_err(), "ratio {bad:?} should be rejected");
        }
    }

    /// The training step must produce a gradient for every parameter. A dangling
    /// `detach`, or a head excluded from the graph, shows up here and nowhere
    /// else until the run silently fails to learn.
    #[test]
    fn a_training_step_produces_gradients_and_a_finite_loss() {
        let device = Default::default();
        let cfg = ModelConfig::tiny();
        let model = QuarkLm::<B>::new(cfg.clone(), &device);

        let batch = TokenBatch::<B> {
            input: Tensor::<B, 2, Int>::random(
                [2, 8],
                Distribution::Uniform(0.0, cfg.vocab_size as f64),
                &device,
            ),
            target: Tensor::<B, 2, Int>::random(
                [2, 8],
                Distribution::Uniform(0.0, cfg.vocab_size as f64),
                &device,
            ),
            score_mask: Tensor::<B, 2>::ones([2, 8], &device),
        };

        let out = TrainStep::step(&model, batch);
        let loss = out.item.loss.into_scalar();
        assert!(loss.is_finite(), "loss must be finite, got {loss}");

        let n_grads = out.grads.len();
        assert!(
            n_grads > 0,
            "the backward pass must reach the parameters; got {n_grads} gradient tensors"
        );
    }

    /// A fresh model predicts roughly uniformly, so its loss should sit near
    /// `ln(vocab_size)`. Far below means the head is leaking the target; far
    /// above means the initialization is broken.
    #[test]
    fn an_untrained_model_starts_near_uniform_loss() {
        let device = Default::default();
        let cfg = ModelConfig::tiny();
        let model = QuarkLm::<B>::new(cfg.clone(), &device);

        let batch = TokenBatch::<B> {
            input: Tensor::<B, 2, Int>::random(
                [4, 16],
                Distribution::Uniform(0.0, cfg.vocab_size as f64),
                &device,
            ),
            target: Tensor::<B, 2, Int>::random(
                [4, 16],
                Distribution::Uniform(0.0, cfg.vocab_size as f64),
                &device,
            ),
            score_mask: Tensor::<B, 2>::ones([4, 16], &device),
        };

        let loss: f64 = InferenceStep::step(&model, batch).loss.into_scalar().into();
        let uniform = (cfg.vocab_size as f64).ln();
        assert!(
            (loss - uniform).abs() < 0.5,
            "initial loss {loss} should be near ln({}) = {uniform:.3}",
            cfg.vocab_size
        );
    }

    /// The end-to-end check: shards on disk, a real `Learner`, real checkpoints.
    /// This is a microtest -- one epoch of a 2-layer toy on ~600 tokens -- not
    /// training. It exists so that the wiring is exercised on CI, where nobody
    /// has a GPU, rather than discovered to be broken on the user's machine an
    /// hour into a real run.
    #[test]
    fn a_full_run_completes_and_writes_the_artifacts_it_promises() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = tiny_config(dir.path());

        write_shard(&cfg.train_shard, 600, cfg.model.vocab_size);
        write_shard(&cfg.valid_shard, 200, cfg.model.vocab_size);

        let model = run::<B>(cfg.clone(), Default::default()).unwrap();
        assert_eq!(model.config().vocab_size, cfg.model.vocab_size);

        assert!(
            cfg.artifact_dir.join("config.json").exists(),
            "the run must record the config that produced it"
        );
        assert!(
            cfg.artifact_dir.join("model.mpk").exists(),
            "the run must leave a loadable model behind"
        );
        assert!(
            cfg.artifact_dir.join("checkpoint").exists(),
            "checkpointing must be wired up, or a long run cannot be resumed"
        );

        // The diagnostics have to survive to disk, not merely to the dashboard of
        // a run that has since exited. The epoch-6 divergence in docs/RESULTS.md
        // §5 is undiagnosed precisely because the artifact directory did not
        // carry the number that would have settled it, and a metric that is
        // registered but unlogged reproduces that failure exactly.
        let train_epoch_1 = cfg.artifact_dir.join("train").join("epoch-1");
        let grad_rms = train_epoch_1.join("GradRms.log");
        assert!(
            grad_rms.exists(),
            "the gradient RMS must reach the artifact directory: it is the evidence \
             for or against `epsilon = 1e-15`, and evidence only counts if it outlives the run"
        );
        // One serialized value per line, which is burn's format, not ours
        // (`FileMetricLogger::log_item` writes `NumericEntry::serialize`, and that
        // is `f64::to_string`). Worth pinning anyway: the number has to be
        // *readable*, and `experiments/run_analysis.py` is the reader. Rust's f64
        // Display round-trips exactly and never emits an exponent, so a grad RMS
        // of 1e-15 lands as "0.000000000000001" -- long, lossless, parseable. The
        // scientific notation in `GradRmsMetric::update` is the dashboard string,
        // a different field, and this asserts the log rather than assuming the two
        // agree.
        let logged: Vec<f64> = std::fs::read_to_string(&grad_rms)
            .unwrap()
            .lines()
            .map(|line| {
                line.parse().unwrap_or_else(|e| {
                    panic!("every line must parse as a number, got {line:?}: {e}")
                })
            })
            .collect();
        assert!(
            !logged.is_empty() && logged.iter().all(|&v| v > 0.0 && v.is_finite()),
            "every logged gradient RMS must be positive and finite, got {logged:?}"
        );
        assert!(
            !cfg.artifact_dir
                .join("valid")
                .join("epoch-1")
                .join("GradRms.log")
                .exists(),
            "validation computes no gradient, so it must log no gradient RMS rather \
             than a column of zeros that would plot as a collapse"
        );
    }

    /// The saved `model.mpk` must be the *best* epoch's weights, not the last
    /// epoch's -- `LearningResult.model` is the latter, so `run` has to go back
    /// to the checkpoint burn kept.
    ///
    /// The setup forces the two to differ, because otherwise the assertion is
    /// vacuous: train and valid are drawn from *different* random streams, so
    /// there is nothing to generalize and every epoch past the first can only
    /// overfit. The learning rate is high enough to make that happen within the
    /// few steps a microtest can afford. The test asserts nothing about *which*
    /// epoch wins -- it reads that from the logs -- only that the file on disk
    /// is that epoch's, and separately that a later epoch existed to be wrongly
    /// chosen instead.
    #[test]
    fn the_saved_model_is_the_best_epoch_not_the_final_one() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = TrainConfig {
            num_epochs: 4,
            lr: 3e-2,
            warmup_batches: 1,
            ..tiny_config(dir.path())
        };

        write_shard(&cfg.train_shard, 600, cfg.model.vocab_size);
        // A different stream from the train shard, so nothing learned on one
        // transfers to the other.
        let mut w = ShardWriter::create(&cfg.valid_shard, cfg.model.vocab_size, 0).unwrap();
        let mut x: u64 = 999;
        let tokens: Vec<u32> = (0..200)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((x >> 33) as usize % cfg.model.vocab_size) as u32
            })
            .collect();
        w.push_document(&"word ".repeat(200), &tokens).unwrap();
        w.finish().unwrap();

        run::<B>(cfg.clone(), Default::default()).unwrap();

        // Re-derived from the raw logs rather than from `best_valid_loss_epoch`,
        // so this checks `run`'s wiring instead of agreeing with itself. Each
        // line is `"<value>,<count>"`, and the mean is weighted by count.
        let mean = |epoch: usize| -> Option<f64> {
            let path = cfg
                .artifact_dir
                .join("valid")
                .join(format!("epoch-{epoch}"))
                .join("Loss.log");
            let body = std::fs::read_to_string(path).ok()?;
            let (sum, n) = body
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| {
                    let (v, c) = l
                        .trim()
                        .split_once(',')
                        .expect("`<value>,<count>` per line");
                    let (v, c): (f64, usize) = (v.parse().unwrap(), c.parse().unwrap());
                    (v * c as f64, c)
                })
                .fold((0.0, 0), |(sv, sn), (v, n)| (sv + v, sn + n));
            (n > 0).then(|| sum / n as f64)
        };
        let losses: Vec<(usize, f64)> = (1..=cfg.num_epochs)
            .filter_map(|e| Some((e, mean(e)?)))
            .collect();
        assert_eq!(
            losses.len(),
            cfg.num_epochs,
            "every epoch should have logged a validation loss: {losses:?}"
        );
        let best = losses
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;

        // Guards the assertion below against becoming vacuous: if the run stops
        // overfitting, the last epoch would be the best one and this test would
        // pass without testing anything.
        assert_ne!(
            best, cfg.num_epochs,
            "this test only means something when the best epoch is not the last; \
             validation losses were {losses:?}"
        );

        let saved = std::fs::read(cfg.artifact_dir.join("model.mpk")).unwrap();
        let best_ckpt = std::fs::read(
            cfg.artifact_dir
                .join("checkpoint")
                .join(format!("model-{best}.mpk")),
        )
        .unwrap_or_else(|e| panic!("the best epoch's checkpoint must survive: {e}"));
        assert!(
            saved == best_ckpt,
            "model.mpk must be epoch {best}'s weights (the lowest validation loss), \
             but it does not match that checkpoint; losses were {losses:?}"
        );
    }

    /// Training on a shard built with a different tokenizer is the kind of
    /// mistake that produces a plausible-looking loss curve for a model that
    /// learned nothing. It has to be caught at startup.
    #[test]
    fn a_shard_from_a_different_tokenizer_is_rejected_before_training() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = tiny_config(dir.path());
        write_shard(&cfg.train_shard, 600, cfg.model.vocab_size);

        // Same tokens, but the sidecar claims a vocabulary the model doesn't have.
        let mut w = ShardWriter::create(&cfg.valid_shard, cfg.model.vocab_size * 2, 0).unwrap();
        w.push_document("a b c", &[1, 2, 3]).unwrap();
        w.finish().unwrap();

        let err = run::<B>(cfg, Default::default()).unwrap_err().to_string();
        assert!(err.contains("vocab_size"), "{err}");
    }
}
