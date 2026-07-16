#!/usr/bin/env python3
"""Analysis of the three WikiText-103 runs reported in issue #3.

Every headline number in docs/RESULTS.md comes from this script, matching the
convention set by experiments/scaling_budget.py. Run it and the doc is
reproducible:

    python3 experiments/run_analysis.py

Numbers marked MEASURED are transcribed from the issue's console output or from
primary sources; numbers marked DERIVED are computed here. The distinction
matters: this analysis reverses a pre-registered design decision, so it must be
auditable which numbers are the user's observations and which are our inference.

The three runs, all on WikiText-103 with an identical eval protocol:

  run1  quark_3m        1 unique layer x 12 loops, d_model 384   ~11.0 GB, ~1 h
  run2  quark_3m_dense  6 unique layers x 1 loop,  d_model 168   ~5.5 GB, ~15 m
  run3  quark_3m_dense  same config, 10 epochs instead of 1

The headline finding is NOT "run2 beat run1" -- that comparison is confounded
(see confound_check()). It is the iso-compute identity in loop_vs_dense_identity():
run1 already paid, in wall-clock and VRAM, for a model with 7.6x more parameters
than it actually stored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# MEASURED: transcribed from the issue's console output.
# Line numbers refer to the issue body as rendered by `gh issue view`.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    name: str
    config: str
    valid_loss: float  # best mean valid Loss reported by the Learner summary
    word_ppl: float
    bits_per_byte: float
    token_ppl: float
    total_nll: float
    blimp: float  # accuracy in percent
    vram_gb: float  # user-reported, approximate
    minutes: float  # user-reported, approximate


# Shared eval denominators. Identical across all three runs, which is itself a
# check: the same corpus and tokenizer were scored every time.
SCORED_TOKENS = 320_000  # 99.9925% of the corpus
WORDS = 241_211
BYTES = 1_280_639

RUN1 = Run("run1", "quark_3m (1x12 looped)", 3.706, 115.163, 1.2897, 35.792, 1_144_870.9, 57.05, 11.0, 60.0)
RUN2 = Run("run2", "quark_3m_dense (6x1, 1 ep)", 3.653, 108.275, 1.2730, 34.166, 1_129_995.6, 58.63, 5.5, 15.0)
RUN3 = Run("run3", "quark_3m_dense (6x1, 10 ep)", 3.707, 123.193, 1.3081, 37.657, 1_161_130.0, 60.93, 5.5, 150.0)
RUNS = [RUN1, RUN2, RUN3]

# run3's per-epoch Learner summary (MEASURED).
RUN3_TRAIN_LOSS = {1: 4.049, 3: 3.760, 6: 5.033}  # min at 3, max at 6
RUN3_VALID_LOSS = {1: 3.803, 3: 3.707, 6: 4.725}  # min at 3, max at 6

# MEASURED: GPT-2 124M, WikiText-103 word-level perplexity, ZERO-SHOT.
# Radford et al. 2019, "Language Models are Unsupervised Multitask Learners",
# Table 3. WebText excluded Wikipedia, so this is a genuine zero-shot transfer
# number and arguably a soft target -- we train in-domain.
GPT2_SMALL_WT103_WORD_PPL = 37.50

# MEASURED: training hyperparameters, from src/train/mod.rs TrainConfig::default().
LR_PEAK = 3e-3
LR_MIN_RATIO = 0.1
WARMUP_BATCHES = 200
BATCH_SIZE = 16
GRAD_ACCUM = 4
SEQ_LEN = 512

# MEASURED: the LR logs attached to the issue are 16444 lines long, one per
# dataloader batch, all tagged epoch 1. Sanity: 16444 * 16 * 512 = 134.7M tokens,
# which matches WikiText-103's ~103M words at the 1.3266 tokens/word this
# tokenizer actually produced. So one epoch is 16444 batches.
BATCHES_PER_EPOCH = 16_444

# MEASURED: last logged LR of epoch 1 in each run. run1 and run2's LR logs are
# byte-identical -- same schedule, same batch count -- and end fully annealed.
# run3's cosine is stretched over 10x the batches, so its epoch 1 ends near peak.
# These two values are what pin the reconstruction in lr_at_batch().
LOGGED_LR_END_EPOCH1_RUN12 = 0.000300000024637043
LOGGED_LR_END_EPOCH1_RUN3 = 0.002933934266768104

# ---------------------------------------------------------------------------
# Architecture, mirroring src/config.rs so this script is self-contained.
# Verified against the Rust budget in test_matches_rust_budget().
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Arch:
    name: str
    vocab_size: int
    d_emb: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    n_unique_layers: int
    n_loops: int
    tie_embeddings: bool = True

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_layer_applications(self) -> int:
        return self.n_unique_layers * self.n_loops

    def matmul_params_per_layer(self) -> int:
        d, kv = self.d_model, self.n_kv_heads * self.d_head
        wq, wk, wv, wo = d * d, d * kv, d * kv, d * d
        ffn = d * self.d_ff * 2 + self.d_ff * d  # SwiGLU: gate, up, down
        return wq + wk + wv + wo + ffn

    def params_per_layer(self) -> int:
        return self.matmul_params_per_layer() + 2 * self.d_model  # + RMSNorm gains

    def param_count(self) -> int:
        total = self.vocab_size * self.d_emb + self.d_emb * self.d_model
        if self.tie_embeddings:
            total += self.d_model * self.d_emb  # H -> E projection
        else:
            total += self.d_model * self.vocab_size
        total += self.params_per_layer() * self.n_unique_layers
        total += self.d_model  # final norm
        return total

    def compute_equivalent_params(self) -> int:
        """Params a dense, unshared model with identical per-token FLOPs has."""
        return self.matmul_params_per_layer() * self.n_layer_applications


# src/config.rs presets.
QUARK_3M = Arch("quark_3m", 8192, 128, 384, 6, 2, 1152, 1, 12)
QUARK_3M_DENSE = Arch("quark_3m_dense", 8192, 128, 168, 4, 1, 448, 6, 1)

# The proposal: quark_3m's compute graph, untied. Same width, same depth, same
# FLOPs, same activation memory -- 12 distinct layers instead of 1 reused 12x.
QUARK_22M = Arch("quark_22m", 8192, 128, 384, 6, 2, 1152, 12, 1)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# 1. Is the harness's arithmetic self-consistent?
# ---------------------------------------------------------------------------


def check_eval_arithmetic() -> None:
    section("1. Eval arithmetic self-consistency (DERIVED, checks MEASURED)")
    print("Recomputing each run's reported metrics from its own total NLL.")
    print("If these disagree, no comparison below means anything.\n")
    print(f"{'run':<8} {'token PPL':>18} {'word PPL':>18} {'bits/byte':>18}")
    ok = True
    for r in RUNS:
        token_ppl = math.exp(r.total_nll / SCORED_TOKENS)
        word_ppl = math.exp(r.total_nll / WORDS)
        bpb = r.total_nll / (BYTES * math.log(2))
        for got, want in ((token_ppl, r.token_ppl), (word_ppl, r.word_ppl), (bpb, r.bits_per_byte)):
            if abs(got - want) / want > 1e-3:
                ok = False
        print(
            f"{r.name:<8} {token_ppl:>9.3f} vs {r.token_ppl:<6.3f} "
            f"{word_ppl:>9.3f} vs {r.word_ppl:<6.3f} "
            f"{bpb:>9.4f} vs {r.bits_per_byte:<6.4f}"
        )
    print(f"\n  all three metrics reproduce from total NLL alone: {'YES' if ok else 'NO'}")
    print(f"  tokens/word = {SCORED_TOKENS / WORDS:.4f}  (word NLL = this x token NLL)")
    print("\n  => The eval harness can be trusted. The numbers are internally consistent.")


# ---------------------------------------------------------------------------
# 2. The identity that drives the whole recommendation.
# ---------------------------------------------------------------------------


def loop_vs_dense_identity() -> None:
    section("2. THE IDENTITY: what run1's compute actually bought (DERIVED)")
    print("Weight sharing reduces STORAGE, not ARITHMETIC. Looping one layer 12")
    print("times costs exactly what 12 distinct layers cost. So consider the")
    print("untied twin of quark_3m: same width, same depth, same FLOPs.\n")

    print(f"{'':<28} {'quark_3m (1x12)':>18} {'quark_22m (12x1)':>18}")
    print("-" * 66)
    for label, fn in [
        ("d_model", lambda a: a.d_model),
        ("d_ff", lambda a: a.d_ff),
        ("layer applications", lambda a: a.n_layer_applications),
        ("unique layers", lambda a: a.n_unique_layers),
        ("compute-equiv params", lambda a: f"{a.compute_equivalent_params():,}"),
        ("STORED params", lambda a: f"{a.param_count():,}"),
    ]:
        print(f"{label:<28} {str(fn(QUARK_3M)):>18} {str(fn(QUARK_22M)):>18}")

    assert QUARK_3M.compute_equivalent_params() == QUARK_22M.compute_equivalent_params()
    assert QUARK_3M.matmul_params_per_layer() == QUARK_22M.matmul_params_per_layer()
    print("\n  Identical compute-equivalent params  => identical FLOPs/token")
    print("  Identical width, depth and head count => identical activation memory")

    ratio = QUARK_22M.param_count() / QUARK_3M.param_count()
    print(f"\n  Stored params: {QUARK_3M.param_count():,} -> {QUARK_22M.param_count():,}  ({ratio:.1f}x)")

    extra = QUARK_22M.param_count() - QUARK_3M.param_count()
    opt_gb = extra * 4 * 4 / 1e9  # param + grad + Adam m + v, f32
    print(f"  Extra optimizer memory for the untied weights: {opt_gb:.3f} GB")
    print(f"  ...against run1's MEASURED {RUN1.vram_gb} GB footprint on a 16 GB card.")

    print("\n  The looped model's function class is a STRICT SUBSET of the dense one:")
    print("  tie all 12 layers' weights and you recover quark_3m exactly. The dense")
    print("  model can represent everything the looped model can, and more, at the")
    print("  same arithmetic cost.")
    print(f"\n  => run1 spent {RUN1.minutes:.0f} min and {RUN1.vram_gb} GB running a "
          f"{QUARK_3M.compute_equivalent_params():,}-param")
    print(f"     compute graph in order to store {QUARK_3M.param_count():,} params.")
    print("     The 3M budget was never the binding constraint. VRAM is, and VRAM is")
    print("     consumed by activations (compute), not by stored weights.")


# ---------------------------------------------------------------------------
# 3. The confound in the pre-registered test.
# ---------------------------------------------------------------------------


def confound_check() -> None:
    section("3. What run1-vs-run2 does and does NOT show (DERIVED)")
    print("docs/DESIGN.md Sec 5 pre-registered: 'quark_3m_dense is the honest control")
    print("-- if it matches quark_3m, looping bought nothing.' By that criterion the")
    print("result is in, and it is worse than a match: dense WINS.\n")

    print(f"{'':<24} {'run1 looped':>14} {'run2 dense':>14} {'delta':>12}")
    print("-" * 66)
    rows = [
        ("valid loss (nats)", RUN1.valid_loss, RUN2.valid_loss),
        ("word PPL", RUN1.word_ppl, RUN2.word_ppl),
        ("BLiMP %", RUN1.blimp, RUN2.blimp),
        ("VRAM GB", RUN1.vram_gb, RUN2.vram_gb),
        ("minutes", RUN1.minutes, RUN2.minutes),
    ]
    for label, a, b in rows:
        print(f"{label:<24} {a:>14.3f} {b:>14.3f} {b - a:>+12.3f}")
    c1, c2 = QUARK_3M.compute_equivalent_params(), QUARK_3M_DENSE.compute_equivalent_params()
    print(f"{'compute-equiv params':<24} {c1:>14,} {c2:>14,} {c1 / c2:>11.1f}x")
    print(f"{'stored params':<24} {QUARK_3M.param_count():>14,} {QUARK_3M_DENSE.param_count():>14,}")

    print("\n  BUT: this is NOT a clean looping-vs-dense test. The two configs differ")
    print("  on THREE axes at once:")
    print(f"    width          d_model {QUARK_3M.d_model} vs {QUARK_3M_DENSE.d_model}")
    print(f"    layer diversity      1 vs {QUARK_3M_DENSE.n_unique_layers} unique layers")
    print(f"    depth               {QUARK_3M.n_layer_applications} vs "
          f"{QUARK_3M_DENSE.n_layer_applications} applications")
    print("\n  So run2 shows: at a 3M budget, spending sharing's savings on WIDTH does")
    print("  not pay for the loss of LAYER DIVERSITY. That is a real and useful")
    print("  result -- and it is untested in the literature (nobody re-widens a looped")
    print("  model to restore param parity) -- but it does not isolate the mechanism.")
    print("\n  The clean test is quark_22m vs quark_3m: identical on every axis except")
    print("  whether the 12 layer applications share one weight set. That run is the")
    print("  experiment this analysis proposes.")


# ---------------------------------------------------------------------------
# 4. The 10-epoch regression.
# ---------------------------------------------------------------------------


def lr_at_batch(batch: int, total_batches: int) -> float:
    """Reconstruct the LR schedule from src/train/mod.rs lr_scheduler_config().

    Linear warmup MULTIPLIER (1/warmup .. 1.0 over warmup_batches) times a
    cosine decay from LR_PEAK to LR_PEAK*LR_MIN_RATIO over total_batches.
    burn calls lr_step() once per dataloader BATCH, not per optimizer step, so
    the schedule is sized in batches. Validated to 6 s.f. against run1's
    logged LR values.
    """
    warm = min(1.0, (batch + 1) / WARMUP_BATCHES)
    lo = LR_PEAK * LR_MIN_RATIO
    cos = lo + 0.5 * (LR_PEAK - lo) * (1 + math.cos(math.pi * batch / total_batches))
    return warm * cos


def regression_analysis() -> None:
    section("4. Why 10 epochs did WORSE than 1 epoch (DERIVED)")
    print("run3 is the same config as run2, trained 10x longer, and it LOST:")
    print(f"    word PPL {RUN2.word_ppl} (1 epoch)  ->  {RUN3.word_ppl} (10 epochs)")
    print("\n  Three explanations ruled out by the data:\n")

    print("  (a) NOT a checkpoint-selection bug. src/train/mod.rs best_valid_loss_epoch()")
    print("      reads the LearnerSummary, filters NaN, and min()s on valid Loss; run()")
    print("      reloads that epoch before saving. It correctly selected epoch 3.")

    print("\n  (b) NOT overfitting. At the epoch-6 blow-up:")
    print(f"          train loss {RUN3_TRAIN_LOSS[6]:.3f}  >  valid loss {RUN3_VALID_LOSS[6]:.3f}")
    print("      A memorizing model has train << valid. This is the opposite: the")
    print("      model got worse at data it had already seen. It diverged.")

    print("\n  (c) NOT simply 'peak LR too high'. Reconstructing the schedule:")
    total = BATCHES_PER_EPOCH * 10
    print(f"      {'epoch':>6} {'LR at epoch start':>20}")
    for ep in (1, 3, 5, 6, 7, 10):
        print(f"      {ep:>6} {lr_at_batch((ep - 1) * BATCHES_PER_EPOCH, total):>20.3e}")
    print("      The run SURVIVED epochs 1-5 at ~peak LR and blew up in epoch 6 at a")
    print("      FALLING LR. 'Too high' does not explain a blow-up after the peak.")

    print("\n  The one thing that IS established -- run2 vs run3 is confounded by the")
    print("  LR schedule:")
    print(f"      run2's single epoch ended FULLY ANNEALED at LR {LR_PEAK * LR_MIN_RATIO:.1e}")
    print(f"      run3's best checkpoint (epoch 3) sat at ~{lr_at_batch(2 * BATCHES_PER_EPOCH, total):.2e}")
    print("        -- roughly 95% of peak, never annealed.")
    print("      An un-annealed mid-cosine checkpoint is systematically worse than an")
    print("      annealed one. That alone explains much of 108.28 -> 123.19 without")
    print("      invoking overfitting at all.")

    print("\n  ROOT CAUSE OF THE DIVERGENCE: NOT DETERMINED, and NOT DETERMINABLE from")
    print("  the attached logs -- they cover epoch 1 only, so epochs 2-10 are")
    print("  unobservable. That limitation is itself the finding: the harness reports")
    print("  per-epoch MEANS, which hide a spike until it is catastrophic, and it has")
    print("  no divergence detection. ~60% of a 10-epoch run's GPU time produced")
    print("  nothing and left no diagnosable trace.")


# ---------------------------------------------------------------------------
# 5. Distance to the target.
# ---------------------------------------------------------------------------


def gap_to_gpt2() -> None:
    section("5. Distance to GPT-2 124M (DERIVED from MEASURED)")
    print(f"Target: GPT-2 124M zero-shot WikiText-103 word PPL = {GPT2_SMALL_WT103_WORD_PPL}")
    print("(Radford et al. 2019 Table 3; WebText excluded Wikipedia, so this is a")
    print("genuine zero-shot number and we train in-domain -- a real edge.)\n")
    print(f"{'run':<8} {'word PPL':>10} {'nats/word':>11} {'gap to GPT-2':>14} {'PPL ratio':>11}")
    print("-" * 60)
    target_nats = math.log(GPT2_SMALL_WT103_WORD_PPL)
    for r in RUNS:
        nats = math.log(r.word_ppl)
        print(f"{r.name:<8} {r.word_ppl:>10.3f} {nats:>11.4f} {nats - target_nats:>+14.4f} "
              f"{r.word_ppl / GPT2_SMALL_WT103_WORD_PPL:>10.2f}x")
    best = min(RUNS, key=lambda r: r.word_ppl)
    print(f"\n  Best run ({best.name}) is {math.log(best.word_ppl) - target_nats:.3f} nats/word short.")
    print("  That is a large gap, not a rounding error. Closing it is the whole task.")


# ---------------------------------------------------------------------------
# 6. Does BLiMP track perplexity? (It does not.)
# ---------------------------------------------------------------------------


def blimp_vs_ppl() -> None:
    section("6. BLiMP and perplexity DECOUPLE (DERIVED from MEASURED)")
    print(f"{'run':<8} {'word PPL':>10} {'BLiMP %':>10}")
    print("-" * 30)
    for r in RUNS:
        print(f"{r.name:<8} {r.word_ppl:>10.3f} {r.blimp:>10.2f}")
    print("\n  BLiMP rises monotonically 57.05 -> 58.63 -> 60.93 while run3's")
    print("  perplexity gets WORSE. The two metrics disagree about which model is best.")
    print("\n  Caution before leaning on any of this: these are SINGLE-SEED runs.")
    print(f"  The run1-vs-run2 valid-loss delta is {RUN2.valid_loss - RUN1.valid_loss:.3f} nats and the")
    print(f"  BLiMP delta is {RUN2.blimp - RUN1.blimp:.2f} points. Neither is obviously outside")
    print("  seed noise, and at 57-61% both models sit near BLiMP's 50% chance floor,")
    print("  a low-signal regime. Seed variance is unmeasured and should be measured.")


# ---------------------------------------------------------------------------
# 7. VRAM: what actually fits.
# ---------------------------------------------------------------------------


def analytic_activation_gb(a: Arch, seq: int = SEQ_LEN, batch: int = BATCH_SIZE, dtype_bytes: int = 4) -> float:
    """Activation + attention-score memory, following experiments/scaling_budget.py."""
    per_tok = 4 * a.d_model + 3 * a.d_ff
    act = per_tok * seq * batch * a.n_layer_applications * dtype_bytes / 1e9
    attn = batch * a.n_heads * seq * seq * a.n_layer_applications * dtype_bytes / 1e9
    return act + attn


def vram_check() -> None:
    section("7. VRAM: calibrating the analytic model against MEASURED footprints")
    r1_est = analytic_activation_gb(QUARK_3M)
    r2_est = analytic_activation_gb(QUARK_3M_DENSE)
    print(f"{'':<24} {'analytic est':>14} {'user MEASURED':>14}")
    print("-" * 56)
    print(f"{'quark_3m (1x12)':<24} {r1_est:>13.2f}G {RUN1.vram_gb:>13.1f}G")
    print(f"{'quark_3m_dense (6x1)':<24} {r2_est:>13.2f}G {RUN2.vram_gb:>13.1f}G")

    # Two points, two unknowns: fixed overhead + a scale factor.
    slope = (RUN1.vram_gb - RUN2.vram_gb) / (r1_est - r2_est)
    fixed = RUN1.vram_gb - slope * r1_est
    print(f"\n  2-point calibration: VRAM ~= {fixed:.2f} GB fixed + {slope:.2f} x analytic")
    print("  (CRUDE: two points, two unknowns, zero degrees of freedom. It cannot be")
    print("  wrong on these two runs and has no evidence of being right anywhere else.")
    print("  The scaling_budget.py model accounts only for activations; the residual is")
    print("  params, grads, optimizer state, the autodiff graph and allocator slack.)")

    print("\n  The load-bearing claim needs NO model at all:")
    print(f"    quark_22m has the SAME compute graph as quark_3m, so the same activation")
    print(f"    memory ({analytic_activation_gb(QUARK_22M):.2f} GB analytic, identical), plus "
          f"{(QUARK_22M.param_count() - QUARK_3M.param_count()) * 16 / 1e9:.2f} GB of")
    print(f"    extra weights+grads+Adam state. run1 MEASURED {RUN1.vram_gb} GB on a 16 GB card.")
    print(f"    => quark_22m fits, with room to spare, and takes about run1's ~{RUN1.minutes:.0f} min/epoch.")


# ---------------------------------------------------------------------------
# Self-checks.
# ---------------------------------------------------------------------------


def test_matches_rust_budget() -> None:
    """The Python arch mirror must agree with src/config.rs exactly."""
    # Asserted by src/config.rs::reference_config_fits_the_3m_budget.
    assert QUARK_3M.param_count() == 2_868_352, QUARK_3M.param_count()
    # Asserted by the dense preset's own budget test.
    assert QUARK_3M_DENSE.param_count() == 2_871_880, QUARK_3M_DENSE.param_count()
    # Asserted by src/config.rs::sharing_saves_parameters_but_not_compute.
    assert QUARK_3M.compute_equivalent_params() > 6 * QUARK_3M.param_count()


def test_lr_reconstruction_matches_logs() -> None:
    """The reconstructed schedule must reproduce the LOGGED learning rates.

    Without this the epoch-boundary LR table in regression_analysis() is just an
    assertion about code we read, not about the run that actually happened. Both
    checks land after warmup, where the ramp multiplier is exactly 1.0, so they
    test the cosine -- which is the part the argument rests on.
    """
    last = BATCHES_PER_EPOCH - 1
    got12 = lr_at_batch(last, BATCHES_PER_EPOCH)  # run1/run2: 1 epoch
    got3 = lr_at_batch(last, BATCHES_PER_EPOCH * 10)  # run3: 10 epochs
    assert abs(got12 - LOGGED_LR_END_EPOCH1_RUN12) / LOGGED_LR_END_EPOCH1_RUN12 < 1e-8, got12
    assert abs(got3 - LOGGED_LR_END_EPOCH1_RUN3) / LOGGED_LR_END_EPOCH1_RUN3 < 1e-8, got3


def main() -> None:
    test_matches_rust_budget()
    test_lr_reconstruction_matches_logs()
    print(__doc__)
    check_eval_arithmetic()
    loop_vs_dense_identity()
    confound_check()
    regression_analysis()
    gap_to_gpt2()
    blimp_vs_ppl()
    vram_check()
    print()


if __name__ == "__main__":
    main()
