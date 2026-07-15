#!/usr/bin/env python3
"""Feasibility math for a 3.0M-parameter LM aiming at GPT-2 124M quality.

Every headline number in docs/DESIGN.md comes from this script. Run it and the
doc is reproducible:

    python3 experiments/scaling_budget.py

The analysis splits the issue's goal into two targets that have OPPOSITE
verdicts, which is the whole point:

  1. "Match GPT-2 124M on OpenWebText perplexity."  RULED OUT.  A capacity
     argument, not a data argument: 3M params cannot reach GPT-2's loss even
     with infinite data, so neither more tokens nor distillation helps.

  2. "Match GPT-2 124M on WikiText-103 word-level perplexity."  PLAUSIBLE, and
     there is a published existence proof at 4.5M non-embedding params.  The
     edge is that GPT-2's 37.50 is ZERO-SHOT and WebText excluded Wikipedia,
     while we train in-domain.

Sources are cited inline. Numbers marked MEASURED are transcribed from primary
sources; numbers marked DERIVED are computed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants, all traced to primary sources.
# ---------------------------------------------------------------------------

# Chinchilla parametric fit (Hoffmann et al. 2022, arXiv:2203.15556).
# The coefficients live in Appendix D.2, Equation 10 -- NOT in section 3.3:
#     L(N, D) = E + A/N^alpha + B/D^beta
# Appendix F: "we also count embeddings matrices in the total parameter count",
# so N is TOTAL params, and their vocab is SentencePiece 32k.
CHINCHILLA_E, CHINCHILLA_A, CHINCHILLA_B = 1.69, 406.4, 410.7
CHINCHILLA_ALPHA, CHINCHILLA_BETA = 0.34, 0.28

# Besiroglu et al. 2024 (arXiv:2404.10102, Epoch AI) refit Hoffmann's own data
# and report that the published fit is wrong: it implies ~70 tokens/param while
# Chinchilla was actually trained at ~20, and it fails to fit the reconstructed
# data at p < 1e-235. They attribute it to an optimizer that "stopped before
# convergence due to a poor choice of loss scale". We carry BOTH fits and only
# make claims that hold under each -- a disputed constant must not be load
# bearing. Note B and beta differ by ~5x and ~28%.
BESIROGLU_E, BESIROGLU_A, BESIROGLU_B = 1.8172, 482.01, 2085.43
BESIROGLU_ALPHA, BESIROGLU_BETA = 0.3478, 0.3658

# Fit support. The abstract says 70M-16B, but Table A9 lists 50 models whose
# smallest is 44M. D spans 5B-500B tokens.
FIT_N_MIN, FIT_N_MAX = 4.4e7, 1.6e10

# MEASURED. nanoGPT (github.com/karpathy/nanoGPT) evaluates the OpenAI GPT-2
# 124M checkpoint on OpenWebText: val loss 3.12 nats/token, GPT-2 BPE vocab
# 50257.
GPT2_OWT_ZEROSHOT_LOSS = 3.12

# MEASURED. Same source, and the more honest baseline. Quote: "taking the GPT-2
# (124M) checkpoint and finetuning on OWT directly for a while reaches loss down
# to ~2.85. This then becomes the more appropriate baseline w.r.t. reproduction."
# The 3.12 above is inflated by WebText->OpenWebText distribution shift, since
# OpenWebText is a best-effort reproduction of the never-released WebText.
GPT2_OWT_FINETUNED_LOSS = 2.85

# MEASURED. Radford et al. 2019, Table 3, the "117M" row -- which IS the 124M
# model. Per the openai/gpt-2 README: "our original parameter counts were wrong
# due to an error... you may have seen small referred to as 117M". One model,
# two names. Word-level PPL, zero-shot, with an invertible de-tokenizer.
GPT2_WIKITEXT103_ZEROSHOT_PPL = 37.50

# MEASURED. Merity et al. 2016 (arXiv:1609.07843), Table 1. This paper
# introduced WikiText-103 but reported no perplexity on it.
WT103_VOCAB = 267_735
WT103_TRAIN_TOKENS = 103_227_021

BUDGET = 3_000_000
SEQ = 1024


def chinchilla_loss(n_params: float, n_tokens: float) -> float:
    """Predicted loss in nats/token, Hoffmann's published coefficients."""
    return (CHINCHILLA_E
            + CHINCHILLA_A / n_params ** CHINCHILLA_ALPHA
            + CHINCHILLA_B / n_tokens ** CHINCHILLA_BETA)


def besiroglu_loss(n_params: float, n_tokens: float) -> float:
    """Same, under the Epoch AI refit."""
    return (BESIROGLU_E
            + BESIROGLU_A / n_params ** BESIROGLU_ALPHA
            + BESIROGLU_B / n_tokens ** BESIROGLU_BETA)


def capacity_floor(n_params: float, besiroglu: bool = False) -> float:
    """Loss at infinite data: the best this parameter count can ever do.

    The D term vanishes as D -> infinity, leaving E + A/N^alpha. This is the
    number that matters, because it is the one no training recipe can beat.
    """
    if besiroglu:
        return BESIROGLU_E + BESIROGLU_A / n_params ** BESIROGLU_ALPHA
    return CHINCHILLA_E + CHINCHILLA_A / n_params ** CHINCHILLA_ALPHA


def hr(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1. Calibration
# ---------------------------------------------------------------------------
hr("1. Calibration: do the fits reproduce GPT-2 124M's MEASURED OWT loss?")

print("  WebText's size and epoch count were never published; ~40GB of text is")
print("  very roughly 9B BPE tokens. Sweep rather than pretend to know.")
print()
print(f"  {'D (tokens)':>12} {'Hoffmann':>10} {'Besiroglu':>10}   vs measured")
for d in (9e9, 3e10, 1e11):
    print(f"  {d:12.1e} {chinchilla_loss(124e6, d):10.3f} {besiroglu_loss(124e6, d):10.3f}"
          f"   (zero-shot {GPT2_OWT_ZEROSHOT_LOSS:.2f}, finetuned {GPT2_OWT_FINETUNED_LOSS:.2f})")
print()
print("  Both fits land within ~0.1-0.2 nats of the measured value at GPT-2")
print("  scale, across a 10x sweep of D. Good enough to trust directionally")
print("  HERE. That says nothing about trusting them at 3M -- see section 2.")


# ---------------------------------------------------------------------------
# 2. The capacity wall
# ---------------------------------------------------------------------------
hr("2. Capacity floor: best achievable loss at INFINITE data")

print(f"  {'params':>12} {'Hoffmann':>10} {'PPL':>8} {'Besiroglu':>10} {'PPL':>8}  {'in fit range?':>18}")
for n in (3e6, 5e6, 10e6, 3e7, 4.4e7, 8.5e7, 1.24e8):
    f_h, f_b = capacity_floor(n), capacity_floor(n, besiroglu=True)
    in_range = "yes" if FIT_N_MIN <= n <= FIT_N_MAX else "NO (extrapolated)"
    print(f"  {n:12.2e} {f_h:10.3f} {math.exp(f_h):8.1f} {f_b:10.3f} {math.exp(f_b):8.1f}"
          f"  {in_range:>18}")

floor_3m = capacity_floor(3e6)
gap_zs = floor_3m - GPT2_OWT_ZEROSHOT_LOSS
gap_ft = floor_3m - GPT2_OWT_FINETUNED_LOSS
print()
print(f"  3M capacity floor (Hoffmann)   : {floor_3m:.3f} nats  (PPL {math.exp(floor_3m):.1f})")
print(f"  GPT-2 124M on OWT, zero-shot   : {GPT2_OWT_ZEROSHOT_LOSS:.3f} nats  (PPL {math.exp(GPT2_OWT_ZEROSHOT_LOSS):.1f})")
print(f"  GPT-2 124M on OWT, finetuned   : {GPT2_OWT_FINETUNED_LOSS:.3f} nats  (PPL {math.exp(GPT2_OWT_FINETUNED_LOSS):.1f})")
print(f"  GAP vs zero-shot               : {gap_zs:+.3f} nats  =>  {math.exp(gap_zs):.2f}x worse PPL")
print(f"  GAP vs finetuned               : {gap_ft:+.3f} nats  =>  {math.exp(gap_ft):.2f}x worse PPL")
print()
print("  ---- HONESTY BLOCK: how much does this argument actually prove? ----")
print()
print(f"  N=3e6 is {FIT_N_MIN / 3e6:.0f}x BELOW the smallest fitted model (44M), i.e. ~1.2")
print("  decades outside a fit whose N-support spans only ~2.6 decades. Worse,")
print("  it is self-contradictory under Chinchilla's own definitions: N counts")
print("  embeddings, and with their 32k vocab a 3M-param TOTAL budget cannot")
print("  even hold the embedding matrix at any sane d_model. The formula would")
print("  be describing a model that is nearly all embedding and nothing like")
print("  anything they fitted. So: DO NOT cite 4.24 nats as a prediction.")
print()
print("  What survives the caveats is the DIRECTION and the ORDER OF MAGNITUDE:")
print("  both independent fits put the 3M floor ~1.1-1.4 nats above GPT-2's")
print("  measured loss, and the floor is monotone in N under any fit with")
print("  alpha > 0. The conclusion 'a 3M model cannot match a 124M model on")
print("  the same distribution' does not depend on the disputed constants.")
print()
print("  The mechanism is what matters, and it is not subtle: the gap is at")
print("  INFINITE data. So 'train on 15B tokens' cannot fix it, and neither")
print("  can distillation -- a student cannot exceed its own capacity floor no")
print("  matter how good the teacher is. Distillation changes WHICH function")
print("  inside the student's hypothesis class you converge to; it does not")
print("  enlarge the class.")


# ---------------------------------------------------------------------------
# 3. What would it take to close the gap?
# ---------------------------------------------------------------------------
hr("3. Required 'effective parameter' multiplier to close the OWT gap")

for label, a, e, alpha, target in (
    ("Hoffmann  vs zero-shot 3.12", CHINCHILLA_A, CHINCHILLA_E, CHINCHILLA_ALPHA, GPT2_OWT_ZEROSHOT_LOSS),
    ("Hoffmann  vs finetuned 2.85", CHINCHILLA_A, CHINCHILLA_E, CHINCHILLA_ALPHA, GPT2_OWT_FINETUNED_LOSS),
    ("Besiroglu vs zero-shot 3.12", BESIROGLU_A, BESIROGLU_E, BESIROGLU_ALPHA, GPT2_OWT_ZEROSHOT_LOSS),
):
    # Invert L = E + A/N^alpha  =>  N = (A / (L - E))^(1/alpha)
    needed = (a / (target - e)) ** (1 / alpha)
    print(f"  {label}: need N={needed:.2e}  ->  {needed / 3e6:5.1f}x our budget")

print()
print("  Modern architecture (SwiGLU, RoPE, RMSNorm, better optimizer and LR")
print("  schedule) is empirically worth roughly 1.2-2x in effective params.")
print("  Every row above demands more than that, and the Besiroglu row demands")
print("  vastly more. The verdict is robust to which fit you believe, which is")
print("  exactly why we retarget rather than argue about constants.")


# ---------------------------------------------------------------------------
# 4. Where the target IS contestable
# ---------------------------------------------------------------------------
hr("4. The contestable target: WikiText-103 word-level PPL")

print(f"""  GPT-2 124M scores {GPT2_WIKITEXT103_ZEROSHOT_PPL} word-level PPL on WikiText-103 ZERO-SHOT
  (Radford et al. 2019, Table 3, "117M" row), and WebText explicitly excluded
  Wikipedia. So 37.50 is an OUT-OF-DOMAIN transfer number. We train ON the
  WikiText-103 train split ({WT103_TRAIN_TOKENS:,} tokens): in-domain. That
  asymmetry -- not parameter efficiency -- is the edge, and it is worth a lot.

  This is not hand-waving. There is a published existence proof:

    DEQ-Transformer small (Bai et al. 2019, arXiv:1909.01377, Table 3)
      total params        : 138M
      NON-EMBEDDING params:  4.5M   <-- the actual transformer body
      WikiText-103 test   : 32.4 word-level PPL, in-domain
      (compare TrXL-small: 139M total -> 35.8)

  A 4.5M-parameter transformer BODY reaches 32.4, beating GPT-2 124M's 37.50.
  The 138M total is almost entirely vocabulary, which is the real lesson:
  word-level WikiText-103 is a vocabulary-STORAGE problem, not a modeling one.""")

print()
print("  Why we cannot copy DEQ's setup: our 3M budget is TOTAL, embeddings")
print("  included. A word-level output layer is arithmetically impossible.")
print()
print(f"  WikiText-103 vocab = {WT103_VOCAB:,} words. Tied embedding matrix alone:")
print(f"    {'d_model':>8} {'embedding params':>18}  {'vs 3.0M budget':>16}")
for d in (16, 32, 128, 256, 512):
    p = WT103_VOCAB * d
    print(f"    {d:8d} {p:18,}  {p / BUDGET:15.1f}x")
max_dim = BUDGET / WT103_VOCAB
print(f"    -> a 3.0M TOTAL budget caps word-level d_model at {max_dim:.1f}. Absurd.")
print()
print("  So we use a SUBWORD vocab (8192 BPE) and report word-level PPL by")
print("  renormalizing: PPL_word = exp(total_NLL / n_words). This is EXACTLY")
print("  the protocol GPT-2 used -- Radford et al: 'We evaluate the same")
print("  quantity by computing the log-probability of a dataset according to a")
print("  WebText LM and dividing by the number of canonical units.' GPT-2's own")
print("  vocab (50257 BPE) is not word-level either. The metric is therefore")
print("  tokenizer-independent and the comparison is legitimate.")
print()
print("  Per-token PPL across tokenizers is NOT legitimate: a smaller vocab")
print("  mechanically lowers it (fewer choices per step, more steps per word).")
print("  The harness must never report it as a cross-model comparison.")
print()
print("  ---- The one trap we cannot fully close ----")
print("  GPT-2's 37.50 uses an 'invertible de-tokenizer' that undoes WikiText's")
print("  <unk>/@-@/space-before-punctuation artifacts, worth 2.5-5 PPL by their")
print("  own account. OpenAI never released it. So 37.50 is not exactly")
print("  reproducible, and any number we compute without an equivalent")
print("  de-tokenizer is not comparable to it. Mitigation: we re-evaluate the")
print("  GPT-2 checkpoint OURSELVES through the same harness code path and")
print("  report THAT as the baseline alongside the published 37.50.")


# ---------------------------------------------------------------------------
# 5. Parameter budget
# ---------------------------------------------------------------------------
hr("5. Parameter budget (must match src/config.rs param_count())")


@dataclass
class Arch:
    name: str
    vocab: int
    d_emb: int      # factorized embedding rank
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    n_unique_layers: int
    n_loops: int    # each unique layer applied this many times
    tie_embeddings: bool = True

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    def breakdown(self) -> dict[str, int]:
        b: dict[str, int] = {}
        b["token_embedding (V*E)"] = self.vocab * self.d_emb
        b["embed_proj (E*H)"] = self.d_emb * self.d_model
        if self.tie_embeddings:
            # Reuse token_embedding transposed, plus a separate H->E unembed
            # projection: cheap, and lets the in/out spaces differ.
            b["unembed_proj (H*E)"] = self.d_model * self.d_emb
        else:
            b["lm_head (H*V)"] = self.d_model * self.vocab

        kv = self.n_kv_heads * self.d_head
        per_layer = (
            self.d_model * self.d_model      # Wq
            + self.d_model * kv              # Wk
            + self.d_model * kv              # Wv
            + self.d_model * self.d_model    # Wo
            + self.d_model * self.d_ff       # W_gate
            + self.d_model * self.d_ff       # W_up
            + self.d_ff * self.d_model       # W_down
            + 2 * self.d_model               # 2x RMSNorm
        )
        b[f"layers ({self.n_unique_layers} unique x {per_layer:,})"] = (
            self.n_unique_layers * per_layer
        )
        b["final_norm"] = self.d_model
        return b

    @property
    def total(self) -> int:
        return sum(self.breakdown().values())

    @property
    def compute_equiv_params(self) -> int:
        """Params a DENSE model with the same FLOPs/token would have.

        Weight sharing saves storage, not compute: looping one layer 12 times
        costs exactly what 12 distinct layers cost.
        """
        kv = self.n_kv_heads * self.d_head
        per_layer = (2 * self.d_model * self.d_model
                     + 2 * self.d_model * kv
                     + 3 * self.d_model * self.d_ff)
        return per_layer * self.n_unique_layers * self.n_loops


CANDIDATES = [
    # research.txt's proposal. E=32 caps the output log-prob matrix at rank 32
    # (Yang et al. 2018) -- a hard expressiveness ceiling it never mentions.
    Arch("research.txt (V=4096, E=32, H=256)", 4096, 32, 256, 4, 1, 512, 1, 12),
    # Ours: spend the freed budget on embedding RANK, not on width.
    Arch("quark-3m (V=8192, E=128, H=384)", 8192, 128, 384, 6, 2, 1152, 1, 12),
    # Same budget, different sharing structure. Holding the budget fixed is what
    # makes these a controlled A/B: any quality difference is attributable to the
    # sharing structure rather than to size. Widths are forced -- more unique
    # layers must be paid for with less width.
    Arch("quark-3m-deep (V=8192, E=128, H=288, 2x6)", 8192, 128, 288, 4, 1, 768, 2, 6),
    Arch("quark-3m-dense (V=8192, E=128, H=168, 6x1)", 8192, 128, 168, 4, 1, 448, 6, 1),
    Arch("quark-tiny (V=4096, E=96, H=256)", 4096, 96, 256, 4, 1, 704, 1, 8),
]

for arch in CANDIDATES:
    print(f"\n  {arch.name}")
    print(f"  {'-' * len(arch.name)}")
    for k, v in arch.breakdown().items():
        print(f"    {k:42s} {v:>10,}")
    status = "OK" if arch.total <= BUDGET else "OVER BUDGET"
    print(f"    {'TOTAL':42s} {arch.total:>10,}   [{status}, {BUDGET - arch.total:+,} vs 3.0M]")
    print(f"    {'compute-equivalent dense params':42s} {arch.compute_equiv_params:>10,}"
          f"   <- drives FLOPs, NOT the budget")
    print(f"    {'output softmax rank cap':42s} {arch.d_emb:>10,}"
          f"   <- Yang et al. softmax bottleneck")

print()
print("  Softmax bottleneck (Yang et al. 2018, arXiv:1711.03953, Corollary 1):")
print("  with tied+factorized embeddings the logit matrix factors through R^E,")
print("  so rank(logits) <= E and the model provably cannot express the true")
print("  distribution once rank(A) > E+1. Their Table 6 measures a d=400 softmax")
print("  saturating at rank exactly 400 -- the bound is ACTIVE, not slack.")
print("  research.txt's E=32 is a rank-32 cap on a real-language next-token")
print("  distribution. That is the single worst decision in that document, and")
print("  the reason quark-3m spends 1.05M of its 3.0M on embedding rank.")


# ---------------------------------------------------------------------------
# 6. Training cost
# ---------------------------------------------------------------------------
hr("6. Training cost on a single 16GB GPU (wgpu backend)")

arch = CANDIDATES[1]
n_layer_apps = arch.n_unique_layers * arch.n_loops

print(f"  Architecture: {arch.name}")
print(f"  Trainable params            : {arch.total:,}")
print(f"  Compute-equivalent params   : {arch.compute_equiv_params:,}"
      f"  ({arch.compute_equiv_params / arch.total:.1f}x the budget)")
print()
print("  Weight sharing means we PAY the FLOPs of a ~21M model while STORING a")
print("  3M model. Budget the run against the former. This is the cost of the")
print("  parameter trick, and it is not optional.")
print()

for tokens in (3e9, 10e9, 15e9):
    # 6*N*D: the standard fwd+bwd FLOP estimate for dense matmuls.
    flops_matmul = 6 * arch.compute_equiv_params * tokens
    # Attention score/context matmuls: 12 * L_eff * d_model * seq per token
    # (fwd+bwd). Non-negligible at seq=1024 for a model this narrow.
    flops_attn = 12 * n_layer_apps * arch.d_model * SEQ * tokens
    total_flops = flops_matmul + flops_attn
    print(f"  D={tokens:5.1e} tokens: {total_flops:.2e} FLOPs"
          f"  (matmul {flops_matmul:.1e} + attn {flops_attn:.1e})")
    for name, tflops in (("4060Ti-16GB @10% MFU", 2.2),
                         ("4080-16GB   @15% MFU", 7.0),
                         ("optimistic  @30% MFU", 15.0)):
        hours = total_flops / (tflops * 1e12) / 3600
        print(f"      {name}: {hours:7.1f} h  ({hours / 24:5.1f} days)")
    print()

print(f"  WikiText-103 train is only {WT103_TRAIN_TOKENS / 1e6:.0f}M words, so an in-domain run is")
print("  a few epochs = single-digit hours. The 3-15B token figures apply to")
print("  the OpenWebText pretraining leg, which is days -- affordable but not")
print("  free, and worth doing only if it demonstrably helps WT103.")

hr("7. Activation memory at seq=1024 (the real 16GB constraint)")

BYTES = 4  # f32; halve for f16 activations
print(f"  Layer applications needing stored activations: {n_layer_apps}")
print("  Sharing does NOT reduce this. Every loop iteration's activations must")
print("  be kept for backprop unless recomputed -- the shared layer sees")
print("  different inputs each loop, so there is nothing to reuse.")
print()
print(f"  {'batch':>6} {'act. GB':>9} {'attn GB':>9} {'total GB':>9}  {'fits 16GB?':>11}")
for batch in (4, 8, 16, 32):
    # Per layer application, per token: residual stream + qkv + attn out +
    # SwiGLU gate/up/down intermediates: ~4*d_model + 3*d_ff floats.
    per_tok = 4 * arch.d_model + 3 * arch.d_ff
    act_gb = per_tok * SEQ * batch * n_layer_apps * BYTES / 1e9
    # Materialized attention matrices (no fused/flash kernel assumed on wgpu).
    attn_gb = batch * arch.n_heads * SEQ * SEQ * n_layer_apps * BYTES / 1e9
    total = act_gb + attn_gb
    print(f"  {batch:6d} {act_gb:9.2f} {attn_gb:9.2f} {total:9.2f}"
          f"  {'yes' if total < 13 else 'NO':>11}")

print()
print("  The materialized attention matrix dominates and scales as seq^2.")
print("  Mitigations, in the order the harness applies them:")
print("    1. gradient checkpointing per loop iteration (n_layer_apps -> 1)")
print("    2. micro-batch + gradient accumulation to reach the target batch")
print("    3. shorter seq (512) early in training, extend later")
print("  Plan: micro-batch 8 @ seq 1024, accumulate to ~128-256 sequences.")
