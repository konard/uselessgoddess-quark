#!/usr/bin/env python3
"""Analysis of the quark_22m run reported in issue #6, and what to do next.

Every headline number in docs/NEXT.md comes from this script, matching the
convention set by experiments/run_analysis.py. Run it and the doc is
reproducible:

    python3 experiments/next_steps.py     # -> experiments/out/next_steps.txt

Numbers marked MEASURED are transcribed from the issue's console output or from
a primary source that was read directly. Numbers marked DERIVED are computed
here. Numbers marked UNSUPPORTED are claims that circulate, that this project
went looking for, and that turned out to have no primary source behind them --
they are listed because *not* acting on them is a decision too, and it should
be auditable.

The distinction is doing real work in this file. Issue #6 asks for "максимально
классных техник" -- the coolest techniques. Most of the cool techniques in the
small-LM literature are measured at 100M-1.5B and extrapolated down to 22M by
people who did not run them at 22M. This script's job is to keep the fitted
range visible next to every recommendation, because quark sits outside almost
all of them.

The single most decision-relevant fact in this file needs no literature at all:
quark_22m was trained for ONE epoch at D/N = 6.5 tokens per parameter, which is
0.33x Chinchilla. Everything else is second-order.
"""

from __future__ import annotations

import math
import pathlib
import re

# ---------------------------------------------------------------------------
# MEASURED: quark_22m, transcribed from issue #6's console output.
# One honestly-annealed epoch on WikiText-103, same eval protocol as issue #3.
# ---------------------------------------------------------------------------

Q22M_VALID_LOSS = 3.361
Q22M_WORD_PPL = 74.965
Q22M_BITS_PER_BYTE = 1.1731
Q22M_TOKEN_PPL = 25.897
Q22M_BLIMP = 61.76
Q22M_VRAM_GB = 11.3  # user-reported; the card is 16 GB

# The two prior runs, for reference. Same protocol, same corpus.
# See experiments/run_analysis.py, which owns these.
RUN1_VALID_LOSS = 3.706  # quark_3m, 1x12 looped
RUN1_WORD_PPL = 115.163
RUN1_BLIMP = 57.05
RUN2_VALID_LOSS = 3.653  # quark_3m_dense, 6x1, 1 epoch
RUN2_WORD_PPL = 108.275
RUN2_BLIMP = 58.63

# ---------------------------------------------------------------------------
# MEASURED: the architecture, from src/config.rs. Pinned by the Rust tests, and
# re-asserted here by test_matches_rust_config() so this file cannot drift.
# ---------------------------------------------------------------------------

D_MODEL = 384
N_LAYERS = 12
D_FF = 1152  # SwiGLU
N_HEADS = 6
N_KV_HEADS = 2  # GQA
D_EMB = 128  # factorized embedding, tied to the head in quark_3m
VOCAB = 8192
SEQ_LEN = 512

PARAM_COUNT_3M = 2_868_352
PARAM_COUNT_22M = 21_800_320
COMPUTE_EQUIVALENT_PARAMS = 20_643_840

# MEASURED: the training config, from the issue body.
BATCH_SIZE = 16
GRAD_ACCUM = 4
BATCHES_PER_EPOCH = 16_444  # owned by run_analysis.py
LR_PEAK = 3e-3
WARMUP_BATCHES = 200


# ---------------------------------------------------------------------------
# Section 1. The budget: where quark actually sits
# ---------------------------------------------------------------------------


def tokens_per_epoch() -> int:
    """DERIVED: tokens the model sees in one pass."""
    return BATCHES_PER_EPOCH * BATCH_SIZE * SEQ_LEN


def optimizer_steps(epochs: int = 1) -> int:
    """DERIVED: burn steps the optimizer once per GRAD_ACCUM dataloader batches.

    This is not the same as BATCHES_PER_EPOCH, and the difference is what makes
    the warmup bug in warmup_is_a_rounding_error() a bug.
    """
    return (BATCHES_PER_EPOCH * epochs) // GRAD_ACCUM


def the_budget() -> None:
    d = tokens_per_epoch()
    n = COMPUTE_EQUIVALENT_PARAMS
    ratio = d / n
    print()
    print("1. THE BUDGET -- and this is the whole answer, before any technique")
    print("   " + "-" * 68)
    print(f"      DERIVED  tokens per epoch      {d:>15,}")
    print(f"      DERIVED  compute-equiv params  {n:>15,}")
    print(f"      DERIVED  D/N                   {ratio:>15.2f} tokens/param")
    print(f"      DERIVED  vs Chinchilla (20)    {ratio / 20:>15.2f}x")
    print(f"      DERIVED  optimizer steps       {optimizer_steps():>15,}")
    print()
    print("      quark_22m is not undertrained by a little. It saw 1/3 of the")
    print("      tokens Chinchilla calls compute-optimal, in 1 epoch, and")
    print("      Chinchilla-optimal is itself the *wrong* target here: it")
    print("      assumes unlimited data. WikiText-103 is fixed, so the binding")
    print("      question is repetition, not fresh tokens -- see section 3.")
    print()
    print("      Every architectural idea in issue #6 is a second-order")
    print("      correction to this first-order fact.")


def warmup_is_a_rounding_error() -> None:
    """DERIVED: burn calls lr_step() per dataloader batch, not per optimizer step.

    So `warmup_batches: 200` does not mean 200 warmup *steps*. It means 200
    batches = 50 optimizer steps. This is a genuine mismatch between what the
    config appears to say and what the optimizer experiences, and it is worth
    fixing before any LR sweep -- an LR sweep on top of a 1.2% warmup is
    measuring the warmup as much as the LR.
    """
    steps = optimizer_steps()
    warmup_steps = WARMUP_BATCHES // GRAD_ACCUM
    frac = warmup_steps / steps
    print()
    print("2. THE WARMUP IS 1.2% OF THE RUN")
    print("   " + "-" * 68)
    print(f"      MEASURED warmup_batches        {WARMUP_BATCHES:>8}")
    print(f"      DERIVED  = optimizer steps     {warmup_steps:>8}  (/{GRAD_ACCUM} grad accum)")
    print(f"      DERIVED  total steps           {steps:>8}")
    print(f"      DERIVED  warmup fraction       {frac:>8.2%}")
    print()
    print("      burn steps the LR scheduler once per dataloader batch, so the")
    print("      config's 200 is 200 batches, not 200 steps. Typical practice is")
    print("      1-2% of steps, so this lands in range by accident -- but it")
    print("      lands there for the wrong reason, and it will silently scale")
    print("      the wrong way the moment grad_accum changes.")


# ---------------------------------------------------------------------------
# Section 2. What quark_22m established
# ---------------------------------------------------------------------------


def untying_paid() -> None:
    """MEASURED + DERIVED: the iso-FLOP comparison RESULTS.md §3 pre-registered."""
    delta = RUN1_VALID_LOSS - Q22M_VALID_LOSS
    print()
    print("3. UNTYING WON 0.345 NATS AT IDENTICAL FLOPs -- prediction confirmed")
    print("   " + "-" * 68)
    print("      Both models run the SAME compute graph: 12 layer applications,")
    print("      d_model 384, d_ff 1152. They differ only in whether the 12")
    print("      applications share one set of weights.")
    print()
    print(f"      MEASURED quark_3m  (1x12 tied)   valid loss {RUN1_VALID_LOSS:.3f}"
          f"   word PPL {RUN1_WORD_PPL:>7.3f}   BLiMP {RUN1_BLIMP:.2f}")
    print(f"      MEASURED quark_22m (12x1 dense)  valid loss {Q22M_VALID_LOSS:.3f}"
          f"   word PPL {Q22M_WORD_PPL:>7.3f}   BLiMP {Q22M_BLIMP:.2f}")
    print(f"      DERIVED  delta                              {delta:+.3f} nats"
          f"   {RUN1_WORD_PPL / Q22M_WORD_PPL:>7.2f}x PPL   {Q22M_BLIMP - RUN1_BLIMP:+.2f}")
    print()
    print(f"      DERIVED  stored params  {PARAM_COUNT_3M:>10,} -> {PARAM_COUNT_22M:>10,}"
          f"  ({PARAM_COUNT_22M / PARAM_COUNT_3M:.1f}x)")
    print(f"      DERIVED  FLOPs/token             identical (compute-equiv params"
          f" = {COMPUTE_EQUIVALENT_PARAMS:,} both)")
    print(f"      MEASURED VRAM                    {Q22M_VRAM_GB} GB of 16")
    print()
    print("      RESULTS.md §3 predicted this and §9 pre-registered the")
    print("      falsification ('quark_22m fails to beat quark_3m'). It did not")
    print("      fire. The function-class argument survives: tying is a strict")
    print("      subset of the hypothesis class, and it cost 0.345 nats.")
    print()
    print("      This is iso-FLOP, and it agrees in direction with Saunshi et al.")
    print("      (arXiv:2502.17416). It is NOT the iso-parameter comparison that")
    print("      paper makes, and it does not speak to it.")


def softmax_bottleneck() -> None:
    """DERIVED: the rank cap the factorized+tied embedding imposes.

    Yang et al. (arXiv:1711.03953): a softmax over V words computed from a
    d-dimensional state is a rank-d factorization of a log-probability matrix
    the data may want at higher rank.
    """
    print()
    print("4. THE SOFTMAX RANK CAP IS 128, AND UNTYING DID NOT LIFT IT")
    print("   " + "-" * 68)
    print(f"      MEASURED d_emb                 {D_EMB:>6}   <- the cap")
    print(f"      MEASURED d_model               {D_MODEL:>6}")
    print(f"      MEASURED vocab                 {VOCAB:>6}")
    print()
    print("      quark's head factorizes as (d_model -> d_emb -> vocab), so the")
    print("      logit matrix has rank <= 128 no matter what the body computes.")
    print("      quark_22m untied the *layers*; the head is still a 128-rank")
    print("      bottleneck on a 384-dim state.")
    print()
    print("      This is the cheapest untested lever left, and it is untested")
    print("      HERE as well as in the literature -- see doc §4. ALBERT's own")
    print("      Table 3 not-shared row is monotone in E (81.3/81.7/81.8/82.3 at")
    print("      E=64/128/256/768); the famous 'E=128 is optimal' result holds")
    print("      only in the all-shared row, which quark_22m is no longer in.")


# ---------------------------------------------------------------------------
# Section 3. Epochs -- the recommendation whose fitted range contains quark
# ---------------------------------------------------------------------------

# MEASURED: Muennighoff et al., "Scaling Data-Constrained Language Models"
# (arXiv:2305.16264), Eq 17. R*_D is their fitted decay constant for repeated
# data. Their sweep covers 7M-9B params -- it CONTAINS quark -- and data
# budgets D_C in {100M, 400M, 1.5B}, which BRACKET quark's 135M. This is the
# only recommendation in this file whose fitted range contains quark on both
# axes, which is why it is first.
R_STAR_D = 15.387756


def effective_tokens(epochs: int) -> float:
    """DERIVED from Muennighoff Eq 17: value of D tokens formed by repeating U_D.

    Returns D'/U_D -- effective tokens as a multiple of the unique corpus.
    """
    repeats = epochs - 1  # R_D in their notation: extra passes beyond the first
    return 1.0 + R_STAR_D * (1.0 - math.exp(-repeats / R_STAR_D))


def epochs_table() -> None:
    unique = tokens_per_epoch()
    print()
    print("5. EPOCHS: THE ONE RECOMMENDATION FITTED AT quark's SCALE")
    print("   " + "-" * 68)
    print("      Muennighoff et al. (arXiv:2305.16264) Eq 17, R*_D = 15.387756.")
    print("      Their sweep spans 7M-9B params (contains quark, incl. a literal")
    print("      20M architecture) and D_C in {100M, 400M, 1.5B} (brackets")
    print("      quark's 135M). Nothing else in this file can say that.")
    print()
    print("      epochs   effective tokens   D'/D     gained vs the row above")
    for e in (1, 2, 4, 8, 16):
        d_prime = effective_tokens(e) * unique
        prev = effective_tokens(e // 2) * unique if e > 1 else 0.0
        print(f"      {e:>6}   {d_prime / 1e6:>13.1f}M   {d_prime / (e * unique):>5.1%}"
              f"   {(d_prime - prev) / 1e6:>+8.1f}M")
    ceiling = (1.0 + R_STAR_D) * unique
    print()
    print(f"      DERIVED  ceiling (infinite epochs)  {ceiling / 1e9:.2f}B effective tokens")
    print(f"      DERIVED  4 epochs retains           {effective_tokens(4) / 4:.1%} of nominal value")
    print()
    print("      Their §5, verbatim: 'best loss at around 20-60x more parameters")
    print("      and epochs [than one-epoch compute-optimal]... one-epoch models")
    print("      significantly under-utilize their training data.'")
    print()
    print("      THE CATCH, and it is not small (their App S/Q): every run behind")
    print("      this fit used dropout 0.1 AND weight decay 0.1. quark has NO")
    print("      dropout. The fitted curve is for a regularized model repeating")
    print("      data; an unregularized one is a different curve, and nobody has")
    print("      measured it. So the recommendation is epochs AND dropout, not")
    print("      epochs alone. They are one intervention, not two.")


def babylm_epochs() -> None:
    """MEASURED: LTG-BERT Table 3, via experiments/research/competitors.md §5."""
    print()
    print("6. INDEPENDENT CORROBORATION: nobody else trains for one epoch")
    print("   " + "-" * 68)
    print("      MEASURED  LTG-BERT Table 3, BLiMP vs training length:")
    print("           ~250 epochs -> 83.2      ~1000 epochs -> 83.4")
    print("           ~500 epochs -> 83.5      ~2000 epochs -> 83.5")
    print("      i.e. FLAT across an 8x compute range, no overfitting at 2000.")
    print()
    print("      MEASURED  2025.babylm-main.12 §2: 'most other participants")
    print("                reported training for roughly 20 epochs.'")
    print("      MEASURED  Wilcox et al. 2025 capped at 20 epochs with 'only a")
    print("                2-3 point drop' vs the original's >2000.")
    print()
    print("      Read together with section 5: returns saturate somewhere around")
    print("      20 epochs, and 1 -> 4 is where the cheap part of the curve is.")
    print("      quark has run ONE. This is the least exotic and best-supported")
    print("      change available.")


# ---------------------------------------------------------------------------
# Section 4. Batch size -- two independently fitted laws, one ordering
# ---------------------------------------------------------------------------


def flops_budget() -> float:
    """DERIVED: C = 6ND, the standard training-FLOP estimate."""
    return 6.0 * COMPUTE_EQUIVALENT_PARAMS * tokens_per_epoch()


def deepseek_batch_and_lr() -> None:
    """MEASURED: DeepSeek LLM (arXiv:2401.02954) Eq 1, their fitted scaling laws.

    B_opt = 0.2920 * C^0.3271 (tokens), eta_opt = 0.3118 * C^-0.1250.
    Fitted over C in roughly 1e17-2e19; quark's 1.7e16 is BELOW that range, so
    both numbers are extrapolations. Reported anyway because the *ordering* they
    produce is robust to a lot of extrapolation error -- see the note.
    """
    c = flops_budget()
    b_opt = 0.2920 * c**0.3271
    eta_opt = 0.3118 * c**-0.1250
    b_actual = BATCH_SIZE * GRAD_ACCUM * SEQ_LEN
    print()
    print("7. BATCH SIZE: quark is batch-STARVED, not batch-bloated")
    print("   " + "-" * 68)
    print(f"      DERIVED  C = 6ND                {c:>12.3e} FLOPs")
    print(f"      DERIVED  B_opt (DeepSeek Eq 1)  {b_opt:>12,.0f} tokens")
    print(f"      MEASURED B_actual (16x4x512)    {b_actual:>12,} tokens")
    print(f"      DERIVED  ratio                  {b_opt / b_actual:>12.2f}x")
    print()
    print(f"      DERIVED  eta_opt (DeepSeek Eq 1) {eta_opt:>11.2e}")
    print(f"      MEASURED lr_peak                 {LR_PEAK:>11.2e}"
          f"   ({LR_PEAK / eta_opt:.2f}x eta_opt)")
    print()
    print("      The LR is already right, to 3%. Do not touch it while changing")
    print("      the batch -- and note that changing the batch INVALIDATES this,")
    print("      since eta_opt and B_opt were fitted jointly.")
    print()
    print("      CAVEAT: C = 1.7e16 is below DeepSeek's fitted range. Treat the")
    print("      absolute numbers as soft. The ORDERING is what to act on:")
    print("      B_actual < B_opt, and a second, independently fitted law (Zhang")
    print("      et al.'s critical batch size) puts the CBS higher still. Two")
    print("      laws fitted by different groups on different data agree that")
    print("      32,768 is on the small side. That is the claim.")


# ---------------------------------------------------------------------------
# Section 5. Architecture arithmetic -- exact, no fitted constants
# ---------------------------------------------------------------------------


def per_layer_cost() -> tuple[float, float]:
    """DERIVED: attention and FFN matmul params per layer, in units of d_model^2."""
    d2 = D_MODEL * D_MODEL
    head_dim = D_MODEL // N_HEADS
    kv_dim = N_KV_HEADS * head_dim
    attn = (
        D_MODEL * D_MODEL  # q
        + D_MODEL * kv_dim  # k
        + D_MODEL * kv_dim  # v
        + D_MODEL * D_MODEL  # o
    )
    ffn = 3 * D_MODEL * D_FF  # SwiGLU: gate, up, down
    return attn / d2, ffn / d2


def kaplan_applies() -> None:
    """DERIVED: quark's per-layer cost is 11.67 d^2, so N ~= 12 L d^2 holds.

    This is a lucky coincidence and worth stating explicitly, because it means
    Kaplan's and Levine's fits -- both of which assume the vanilla 12 L d^2 --
    are directly usable on quark WITHOUT re-deriving them. GQA's cheap attention
    (2.67 d^2 vs the vanilla 4 d^2) very nearly cancels SwiGLU's expensive FFN
    (9 d^2 vs the vanilla 8 d^2).
    """
    attn, ffn = per_layer_cost()
    total = attn + ffn
    print()
    print("8. KAPLAN'S N ~= 12*L*d^2 APPLIES TO quark, BY COINCIDENCE")
    print("   " + "-" * 68)
    print(f"      DERIVED  attention (GQA, 2 kv heads)   {attn:>6.3f} d^2"
          f"   (vanilla MHA: 4.000)")
    print(f"      DERIVED  FFN (SwiGLU, d_ff=1152)       {ffn:>6.3f} d^2"
          f"   (vanilla MLP: 8.000)")
    print(f"      DERIVED  total per layer               {total:>6.3f} d^2"
          f"   (vanilla:     12.000)")
    print(f"      DERIVED  error vs Kaplan's 12          {abs(total - 12) / 12:>6.2%}")
    print()
    print(f"      DERIVED  12 layers x {total:.3f} d^2 = {N_LAYERS * total * D_MODEL**2:,}"
          f"  == compute-equivalent params ({COMPUTE_EQUIVALENT_PARAMS:,})")
    print()
    print("      Two deviations from vanilla, pointing opposite ways, cancelling")
    print("      to 2.8%. So Kaplan's and Levine's depth/width fits can be read")
    print("      off directly. This is not a designed property; it is luck, and")
    print("      it stops being true the moment GQA or d_ff changes.")


def aspect_ratio() -> None:
    """DERIVED: quark is already thin. The cheap depth win is already had."""
    print()
    print("9. quark IS ALREADY 2x THINNER THAN GPT-2 SMALL")
    print("   " + "-" * 68)
    print("      Aspect ratio = d_model / n_layers.")
    print()
    print(f"      DERIVED  quark              {D_MODEL / N_LAYERS:>6.1f}   ({D_MODEL}/{N_LAYERS})")
    print(f"      MEASURED GPT-2 small        {768 / 12:>6.1f}   (768/12)")
    print(f"      MEASURED MobileLLM-350M     {30.0:>6.1f}   -- its *final* AR, after their search")
    print(f"      MEASURED Kaplan (48,1600)   {1600 / 48:>6.1f}")
    print()
    print("      quark already sits where MobileLLM's architecture search")
    print("      *landed*. 'Go deep and thin' is advice quark has already taken.")
    print("      There is no large win left here -- the remaining question is")
    print("      whether 12 is slightly too shallow, which is doc §5.")


def relu2_is_not_free() -> None:
    """DERIVED: at equal params, ReLU^2 and SwiGLU cost identical matmul FLOPs.

    This kills the usual framing. SwiGLU uses 3 matmuls of d_model x d_ff;
    ReLU^2 uses 2. So equal params means d_ff' = 1.5 * d_ff -- which is exactly
    Shazeer's 2/3 rule, inverted. Equal params then forces equal matmul FLOPs,
    because both are 2 * params FLOPs/token. There is no FLOP advantage. The
    advantage, if any, is activation memory and one fewer kernel.
    """
    d_ff_relu2 = 3 * D_FF // 2
    swiglu_params = 3 * D_MODEL * D_FF
    relu2_params = 2 * D_MODEL * d_ff_relu2
    swiglu_act = 2 * D_FF  # gate and up are both retained for the backward pass
    relu2_act = d_ff_relu2
    print()
    print("10. ReLU^2 vs SwiGLU IS NOT A FLOP DECISION -- IT IS A MEMORY DECISION")
    print("    " + "-" * 67)
    print(f"      MEASURED SwiGLU d_ff                {D_FF:>8}   (3 matmuls)")
    print(f"      DERIVED  equal-param ReLU^2 d_ff'   {d_ff_relu2:>8}   (2 matmuls)")
    print()
    print(f"      DERIVED  SwiGLU FFN params/layer    {swiglu_params:>8,}")
    print(f"      DERIVED  ReLU^2 FFN params/layer    {relu2_params:>8,}"
          f"   -> identical")
    print(f"      DERIVED  matmul FLOPs/token/layer   {2 * swiglu_params:>8,}"
          f"   -> identical, both")
    print()
    print("      Equal params implies equal FLOPs. The '2x cheaper' framing is")
    print("      an artifact of comparing at equal d_ff, which is not a fair")
    print("      comparison. What DOES change:")
    print()
    print(f"      DERIVED  FFN activations retained/token/layer:"
          f"  {swiglu_act:>5} -> {relu2_act:>5}"
          f"   ({1 - relu2_act / swiglu_act:.0%} less)")
    print()
    print(f"      DERIVED  FFN share of quark_22m's params:"
          f"  {N_LAYERS * swiglu_params / PARAM_COUNT_22M:.0%}")
    print()
    print("      That last number is why this matters more here than in the")
    print("      papers: quark is mostly FFN. But the only numeric equal-param")
    print("      head-to-head that exists (arXiv:2402.03804 Table 2, at 1B) is a")
    print("      TIE -- SwiGLU 50.53 vs ReLU^2 50.48, with ReLU^2 given 1.59%")
    print("      MORE FFN params. Decide on engineering grounds; there is no")
    print("      quality argument in either direction.")


def sliding_window_is_a_noop() -> None:
    """DERIVED: a window >= the context is not a window. Exactly, not approximately."""
    head_dim = D_MODEL // N_HEADS
    ctx = 1024
    kv_bytes = 2 * N_LAYERS * N_KV_HEADS * head_dim * ctx * 2  # k+v, fp16
    print()
    print("11. SLIDING-WINDOW ATTENTION IS A PROVABLE NO-OP AT quark's CONTEXT")
    print("    " + "-" * 67)
    print(f"      MEASURED seq_len                    {SEQ_LEN:>8}")
    print("      A w=1024 window over a <=1024 causal sequence excludes nothing")
    print("      at any position. Not 'approximately nothing' -- the attention")
    print("      output is bit-for-bit identical to full causal attention.")
    print()
    print(f"      DERIVED  KV cache at {ctx} ctx, fp16  {kv_bytes / 1e6:>8.2f} MB"
          f"   = {kv_bytes / 16e9:.4%} of a 16 GB card")
    print()
    print("      The memory it saves is a rounding error, and it saves it on the")
    print("      one resource quark is not short of. Gemma 3's own Figure 6 shows")
    print("      the curves indistinguishable at 1K context. Skip it.")


def muon_eligibility() -> None:
    """DERIVED: how much of quark Muon can actually act on.

    Muon updates 2D parameters. quark's coverage is unusually good -- better
    than anything published -- because its embedding is factorized down to 128.
    This is the strongest structural argument FOR trying Muon here, and it is
    reported next to the argument against.
    """
    attn, ffn = per_layer_cost()
    body = N_LAYERS * (attn + ffn) * D_MODEL**2
    emb = VOCAB * D_EMB + D_EMB * D_MODEL
    eligible = body / (body + emb)
    ns_steps = 5
    overhead = ns_steps * D_MODEL / (BATCH_SIZE * GRAD_ACCUM * SEQ_LEN)
    print()
    print("12. MUON: BEST-IN-CLASS COVERAGE, WORST-CASE BATCH")
    print("    " + "-" * 67)
    print(f"      DERIVED  2D body params (Muon-eligible)  {body:>12,}")
    print(f"      DERIVED  embedding params (AdamW)        {emb:>12,}")
    print(f"      DERIVED  Muon-eligible fraction          {eligible:>12.2%}")
    print()
    print("      That is better coverage than any published Muon result --")
    print("      Keller's 124M NanoGPT is ~31% embedding. quark's factorized")
    print("      d_emb=128 is why. Structurally, this is the best case for Muon.")
    print()
    print("      And yet. Keller's own Muon docstring, verbatim: 'We believe it")
    print("      is unlikely to work well for training with small batch size.'")
    print(f"      His batch is 524,288 tokens. quark's is {BATCH_SIZE * GRAD_ACCUM * SEQ_LEN:,}"
          f" -- {524288 / (BATCH_SIZE * GRAD_ACCUM * SEQ_LEN):.0f}x smaller.")
    print()
    print(f"      DERIVED  Newton-Schulz overhead, by his own formula T*m/B:")
    print(f"               {ns_steps} * {D_MODEL} / {BATCH_SIZE * GRAD_ACCUM * SEQ_LEN:,}"
          f" = {overhead:.2%}  (his headline: 0.7%)")
    print()
    print("      ~8x his quoted overhead, and likely optimistic: a 384x384")
    print("      Newton-Schulz on one GPU is latency-bound, not FLOP-bound.")
    print()
    print("      The smallest model in the Muon paper is 399M -- 18x quark. Its")
    print("      own Table 3 fit encodes a SHRINKING advantage (1.92x at 399M")
    print("      degrading to 1.72x at 1.5B); the famous '2x' is the smallest-")
    print("      model endpoint of a decaying curve, not a floor.")


def what_to_tune_first() -> None:
    """MEASURED: arXiv:2509.02046 Fig 1 and Fig 3, at 100M-ish scale."""
    print()
    print("13. THE OPTIMIZER ANSWER IS 'TUNE THE LR YOU HAVE'")
    print("    " + "-" * 67)
    print("      MEASURED arXiv:2509.02046 Fig 1, verbatim: 'Up to a 2x speedup")
    print("               is achievable by tuning a single hyperparameter")
    print("               (learning rate) in the GPT-3 recipe for a 100M model.'")
    print("      MEASURED same paper Fig 3: 'The highest speedup is capped at")
    print("               1.4x' -- for any alternative optimizer.")
    print()
    print("      So the honest expected value of switching optimizers is ~1.3-1.4x,")
    print("      and it is SMALLER than the gain from tuning the LR of the one you")
    print("      already have. Three independent sources at 124M-190M converge on")
    print("      this. AlgoPerf's 2024 winner managed 1.28x under controlled")
    print("      conditions -- every '2x' in the literature is 60% larger than")
    print("      anything that has survived a competition.")
    print()
    print("      And the same paper's reconciliation cuts FOR AdEMAMix and")
    print("      AGAINST Muon at quark's batch, verbatim: 'Since Mars and")
    print("      AdEMAMix both perform gradient averaging and variance reduction,")
    print("      these methods are advantageous in their noise-dominated small-")
    print("      batch regime, whereas in our larger-batch setting these benefits")
    print("      diminish and matrix-level optimizers become more competitive.'")
    print()
    print("      quark IS the noise-dominated small-batch regime.")


# ---------------------------------------------------------------------------
# Section 6. The tokenizer question
# ---------------------------------------------------------------------------


def tokenizer_verdict() -> None:
    """MEASURED: Lester et al. (arXiv:2404.03626) ran exactly this experiment."""
    print()
    print("14. THE NEURAL TOKENIZER: SOMEONE ALREADY RAN THIS, AT 25M")
    print("    " + "-" * 67)
    print("      Issue #6 asks about replacing BPE with a learned compressor.")
    print("      Lester et al. (arXiv:2404.03626) ran that experiment at 25m --")
    print("      quark's size class -- and it lost on both axes at once:")
    print()
    print("      MEASURED  method          bits/byte   FLOPs/byte")
    print("      MEASURED  SentencePiece        1.12      11.69M")
    print("      MEASURED  EqualInfoAC          1.25      15.42M")
    print()
    print("      DERIVED   strict Pareto domination: worse loss AND more compute.")
    print("      MEASURED  §4, verbatim: 'Our SentencePiece baseline outperforms")
    print("                all other methods.'")
    print("      MEASURED  the gap WIDENS as scale falls: +0.070 bits/byte at 2b,")
    print("                +0.130 at 25m. Small models are hurt MORE, not less.")
    print()
    print("      And a harder problem than losing: a lossy latent cannot report")
    print("      perplexity at all. LCM §2.4.1: 'cannot produce the probability")
    print("      explicitly.' CALM (arXiv:2510.27688) §3.1: 'Standard evaluation")
    print("      metrics like Perplexity... can no longer be computed' -- even")
    print("      with a >99.9%-accurate codec. quark's entire evidence base is")
    print("      word PPL and bits/byte. This would delete it.")
    print()
    print("      UNSUPPORTED: 'ugly BPE fragments hurt quality'. This project")
    print("      went looking for the evidence and found the opposite: SuperBPE")
    print("      at 200k vocab was WORSE on bits/byte (0.7465 vs 0.7482), and")
    print("      Schmidt et al. (arXiv:2402.18376) tested fewer-tokens-implies-")
    print("      better directly and found it 'not to be the case'. The fragments")
    print("      are ugly to read. There is no measurement that they are ugly to")
    print("      learn from.")


def vocab_size() -> None:
    """DERIVED: three routes to V_opt at 22M, and they agree."""
    print()
    print("15. VOCAB 8192 IS DEFENSIBLE; THE ONLY EXPERIMENT WORTH RUNNING IS 4096")
    print("    " + "-" * 67)
    print("      MEASURED  arXiv:2407.13623 fits N_v = 0.20 * C^0.42 -- but note")
    print("                it fits N_v = V*d, NOT V. Its smallest IsoFLOP row is")
    print("                3B, 136x quark. This is an extrapolation, not a fit.")
    print("      DERIVED   three independent routes converge on V_opt ~ 3.6K-9K")
    print("                at 22M. 8192 sits at the top of that range.")
    print("      MEASURED  §5/Fig 7: the data-constrained optimum SHRINKS. quark")
    print("                is data-constrained.")
    print()
    print("      VERDICT: 8192 is fine, plausibly slightly large. The defensible")
    print("      experiment is 4096 vs 8192. Never larger.")
    print()
    print("      COUNTER-EVIDENCE, reported because it is real and it disagrees:")
    print("      arXiv:2311.01955 Table 5 measures vocab 8k->40k = +5.5 BLiMP.")
    print("      That is a MASKED model, and transfer to causal is UNVERIFIED.")
    print("      It is the strongest argument against the paragraph above.")


# ---------------------------------------------------------------------------
# Section 7. Calibration -- what BLiMP 61.76 actually means
# ---------------------------------------------------------------------------


def blimp_calibration() -> None:
    """MEASURED: why the 80-84 BabyLM numbers are not quark's league."""
    print()
    print("16. BLiMP 61.76 IS NOT AS BAD AS THE LEADERBOARD MAKES IT LOOK")
    print("    " + "-" * 67)
    print("      The 80-84 figures quark is implicitly measured against are")
    print("      MASKED models scored by pseudo-log-likelihood. That is not the")
    print("      same measurement quark makes, and the difference is large:")
    print()
    print("      MEASURED  Salazar et al. 2020 Table 7: BERT-base PLL 84.2 vs")
    print("                GPT-2-345M true-LL 82.6 -- 'despite using less than")
    print("                half the data and a third of the capacity.'")
    print()
    print("      The apples-to-apples target is a CAUSAL model on quark's OWN")
    print("      corpus, scored on full unfiltered BLiMP:")
    print()
    print("      MEASURED  Transformer-XL, causal, WikiText-103, 67k BLiMP: 68.7")
    print("                (BLiMP TACL 2020 Table 3)")
    print(f"      MEASURED  quark_22m:                                    {Q22M_BLIMP}")
    print("      MEASURED  OPT-125M, causal, 10M words:                   62.6")
    print("      MEASURED  5-gram, 3.1B words:                            60.5")
    print()
    print("      quark_22m sits at OPT-125M's number with 1/6 the parameters.")
    print("      The honest target is 68.7, not 80.")
    print()
    print("      Two caveats, cutting OPPOSITE ways -- report both or neither:")
    print("      MEASURED  BabyLM FILTERS BLiMP (13.7% removed) and warns results")
    print("                'cannot [be] directly compare[d]' to full-BLiMP runs.")
    print("      MEASURED  The BabyLM evaluator COUNTS TIES AS CORRECT. Its true")
    print("                random floor is 0.543, not 0.500, and an order-blind")
    print("                Zipf-frequency baseline scores 0.663 -- beating quark")
    print("                while ignoring word order entirely. A causal LM never")
    print("                ties, so it collects NONE of that credit.")
    print("      DERIVED   => quark's 61.76 is UNDERSTATED against published")
    print("                BabyLM figures, by an amount nobody has quantified.")


# ---------------------------------------------------------------------------
# Self-tests: this file must not drift from the code or from run_analysis.py.
# ---------------------------------------------------------------------------

HERE = pathlib.Path(__file__).resolve().parent


def test_matches_rust_config() -> None:
    """The architecture constants above must match src/config.rs."""
    src = (HERE.parent / "src" / "config.rs").read_text()
    for name, want in [
        ("param_count_3m", PARAM_COUNT_3M),
        ("param_count_22m", PARAM_COUNT_22M),
        ("compute_equivalent", COMPUTE_EQUIVALENT_PARAMS),
    ]:
        if f"{want:_}" not in src and str(want) not in src:
            raise AssertionError(
                f"{name} = {want:,} is not in src/config.rs. Either the config "
                f"changed or this script is stale; both are bugs."
            )


def test_matches_run_analysis() -> None:
    """Shared constants must match the script that owns them."""
    src = (HERE / "run_analysis.py").read_text()
    m = re.search(r"BATCHES_PER_EPOCH\s*=\s*([\d_]+)", src)
    if m is None or int(m.group(1).replace("_", "")) != BATCHES_PER_EPOCH:
        raise AssertionError("BATCHES_PER_EPOCH disagrees with run_analysis.py")


def test_muennighoff_eq17_reproduces_the_paper() -> None:
    """Eq 17 must return D' = U_D exactly at one epoch: no repeats, no decay."""
    assert abs(effective_tokens(1) - 1.0) < 1e-12, effective_tokens(1)
    # And it must be monotone increasing but sublinear in epochs -- that is the
    # entire content of the equation. If this fails, the sign is wrong.
    for e in range(1, 32):
        assert effective_tokens(e + 1) > effective_tokens(e)
        assert effective_tokens(e + 1) / (e + 1) < effective_tokens(e) / e + 1e-12
    # Asymptote: D'/U_D -> 1 + R*_D.
    assert abs(effective_tokens(10_000) - (1 + R_STAR_D)) < 1e-6


def test_relu2_equal_params_means_equal_flops() -> None:
    """The claim in relu2_is_not_free() is exact; pin it."""
    d_ff_relu2 = 3 * D_FF // 2
    assert 3 * D_MODEL * D_FF == 2 * D_MODEL * d_ff_relu2


def test_kaplan_coincidence() -> None:
    """The 11.67 d^2 identity must reproduce compute_equivalent_params exactly."""
    attn, ffn = per_layer_cost()
    got = int(N_LAYERS * (attn + ffn) * D_MODEL**2)
    assert got == COMPUTE_EQUIVALENT_PARAMS, (got, COMPUTE_EQUIVALENT_PARAMS)


def main() -> None:
    test_matches_rust_config()
    test_matches_run_analysis()
    test_muennighoff_eq17_reproduces_the_paper()
    test_relu2_equal_params_means_equal_flops()
    test_kaplan_coincidence()

    print(__doc__)
    the_budget()
    warmup_is_a_rounding_error()
    untying_paid()
    softmax_bottleneck()
    epochs_table()
    babylm_epochs()
    deepseek_batch_and_lr()
    kaplan_applies()
    aspect_ratio()
    relu2_is_not_free()
    sliding_window_is_a_noop()
    muon_eligibility()
    what_to_tune_first()
    tokenizer_verdict()
    vocab_size()
    blimp_calibration()
    print()


if __name__ == "__main__":
    main()
