# What to do with quark_22m, its config, and its size

Answers issue #6. Every number here is reproducible:

```
python3 experiments/next_steps.py     # -> experiments/out/next_steps.txt
```

Numbers marked MEASURED are from the issue's console output or a primary source
that was read directly. Numbers marked DERIVED are computed by that script.
Numbers marked **UNSUPPORTED** are claims that circulate widely, that this
document went looking for the evidence behind, and that turned out not to have
any. They are listed on purpose: *not* doing something is a decision too, and it
should be auditable.

Sources: [`experiments/research/competitors.md`](../experiments/research/competitors.md)
and [`experiments/research/techniques.md`](../experiments/research/techniques.md).

The issue asks for "максимально классных техник" — the coolest techniques. This
document mostly argues **against** them, and the reason is in §1. It is not
conservatism. It is that quark is at 22M and almost every cool technique was
measured at 100M–1.5B by people who did not run it at 22M, and this project has
already published a document that had to retract two claims for exactly that
reason. Where a technique's fitted range actually contains quark, it is
recommended enthusiastically. That happens once.

---

## 0. The decision, up front

| Question | Decision |
|---|---|
| **Model size** | **Keep 22M. Do not grow it.** The untying win is banked and it was free. At 22M the model has seen **6.5 tokens per parameter** (DERIVED) — 0.33× Chinchilla. Growing the model makes that ratio *worse*. There is no evidence a bigger model is the constraint; there is direct evidence the token budget is. |
| **The one change that matters** | **Train 4 epochs with dropout 0.1**, and sweep weight decay {0.1, 0.5, 1.0, 2.0}. This is the *only* recommendation in this document whose fitted range contains quark on both axes (Muennighoff et al., 7M–9B params, 100M–1.5B data). It is also the least exotic. Those two facts are related. |
| **Batch size** | **Raise 32,768 → ~64k tokens.** Two independently-fitted laws agree quark is batch-*starved*, not bloated. Keep `lr_peak = 3e-3`; it is within **3%** of DeepSeek's fitted η_opt (DERIVED) — but re-tune it after the batch changes, because the two were fitted jointly. |
| **Neural tokenizer instead of BPE** | **No.** Lester et al. ran this exact experiment at **25m** — quark's size class — and SentencePiece won on **both** axes at once (1.12 vs 1.25 bits/byte, at *less* compute). The gap **widens as scale falls**. And a lossy latent cannot report perplexity at all, which would delete this project's entire evidence base. |
| **Vocab 8192** | **Keep.** Three routes converge on V_opt ≈ 3.6K–9K at 22M. The only defensible experiment is **4096 vs 8192**. Never larger. |
| **Optimizer** | **Keep AdamW.** Tune its LR first — that is worth up to **2×** (measured at 100M); every alternative optimizer is capped at **1.4×** (same paper, same figure). Skip Sophia and Lion outright (§9). If you try one thing, try **AdEMAMix with β3=0.999**, not Muon — and §9 explains why that ordering is the opposite of the internet's. |
| **Dataset** | **Keep WikiText-103 as the comparability anchor; add a filtered corpus as a second track.** Do not switch. Every number this project owns is on WikiText-103, and switching corpora forfeits all of them at once for a gain nobody here has measured. |
| **ROCm** | **Done** — and `vulkan` too, which you should try *first*. burn's own matmul benchmarks put Vulkan ahead of ROCm on both AMD cards they measured (§11). |
| **The bug you should fix before any of this** | Your run reported `Total Epochs: 10` on a `num_epochs: 1` config. That is not cosmetic — it is a **stale-checkpoint hazard**, and it nearly loaded the wrong model's weights. Root cause confirmed against burn's source; guard shipped (§12). |

The rest of this document is why.

---

## 1. The budget, which is the whole answer

Before any technique, one arithmetic fact (DERIVED):

| | |
|---|---:|
| tokens per epoch | 134,709,248 |
| compute-equivalent params | 20,643,840 |
| **D/N** | **6.53 tokens/param** |
| vs Chinchilla's 20 | **0.33×** |
| optimizer steps | 4,111 |

quark_22m is not undertrained by a little. It saw **a third** of the tokens
Chinchilla calls compute-optimal, in **one** pass, in **4,111** optimizer steps.

And Chinchilla-optimal is itself the wrong target here, in the direction that
makes this worse rather than better. Chinchilla assumes unlimited fresh data.
WikiText-103 is fixed at ~135M tokens. So the binding question is not "get more
tokens" — it is "how much value is left in the tokens you have", which is §2.

Every architectural idea in issue #6 — ReLU², QK-norm, softcap, sliding window,
a learned tokenizer — is a **second-order correction to this first-order fact**.
That is the single most decision-relevant sentence in this document, and it needs
no literature at all to support it.

---

## 2. Epochs: the one recommendation fitted at quark's scale

MEASURED — Muennighoff et al., *Scaling Data-Constrained Language Models*
(arXiv:2305.16264), Eq 17, R\*_D = 15.387756.

This paper matters more than everything else cited here combined, for one
reason: **its fitted range contains quark on both axes.** Params 7M–9B (including
a literal 20M architecture). Data budgets D_C ∈ {100M, 400M, 1.5B}, which
*bracket* quark's 135M. Nothing else in this document can make that claim — not
Muon (smallest model 399M), not MobileLLM (125M), not the vocab law (3B), not
Sophia, not modded-nanogpt.

DERIVED from Eq 17 — the value of repeating WikiText-103:

| epochs | effective tokens | D′/D | gained vs row above |
|---:|---:|---:|---:|
| 1 | 134.7M | 100.0% | +134.7M |
| 2 | 265.1M | 98.4% | +130.4M |
| **4** | **501.9M** | **93.1%** | **+236.7M** |
| 8 | 892.3M | 82.8% | +390.4M |
| 16 | 1425.6M | 66.1% | +533.2M |

Ceiling at infinite epochs: **2.21B** effective tokens — 16× what quark has used.

At 4 epochs a repeated token is still worth **93.1%** of a fresh one. The second
pass is worth **97%** of the first. quark is throwing that away.

Their §5, verbatim:

> best loss at around **20-60× more parameters and epochs** [than one-epoch
> compute-optimal]... **one-epoch models significantly under-utilize their
> training data.**

Independent corroboration that this is not a quirk of one fit — MEASURED,
LTG-BERT Table 3, BLiMP vs training length:

| ~250 epochs | ~500 | ~1000 | ~2000 |
|---:|---:|---:|---:|
| 83.2 | 83.5 | 83.4 | 83.5 |

Flat across an **8× compute range**, no overfitting at 2000 epochs. And
2025.babylm-main.12 §2: "most other participants reported training for roughly
20 epochs." Wilcox et al. 2025 capped at 20 and took "only a 2-3 point drop."

**Nobody in this field trains for one epoch. quark trains for one epoch.**

### The catch, and it is not small

Their Appendix S/Q: every run behind that fit used **dropout 0.1 and weight decay
0.1**. quark has **no dropout at all**.

So the fitted curve describes a *regularized* model repeating data. An
unregularized one is a different curve and nobody has measured it. The
recommendation is therefore **epochs AND dropout, as one intervention, not two**.
Running 4 epochs without dropout is not "the cheap 93.1%" — it is an untested
extrapolation dressed up as a fitted one.

Corroborating datapoint (MEASURED): GPT-BERT uses dropout 0.1 at
**quark's exact architecture** — 12 layers, hidden 384, 6 heads, vocab 8192.
quark uses 0.0. That is not a considered difference; it looks like an oversight.

**Recommendation: 4 epochs, dropout 0.1, weight decay swept {0.1, 0.5, 1.0, 2.0}.**
Cost: ~4 hours on the 16GB card. Expected: the largest single win available.

---

## 3. Batch size: quark is starved, not bloated

DERIVED, from DeepSeek LLM (arXiv:2401.02954) Eq 1:

| | |
|---|---:|
| C = 6ND | 1.669e16 FLOPs |
| B_opt = 0.2920·C^0.3271 | **59,117 tokens** |
| B_actual (16 × 4 × 512) | **32,768 tokens** |
| ratio | 1.80× |
| η_opt = 0.3118·C^−0.1250 | **2.92e-3** |
| quark's `lr_peak` | 3.00e-3 — **1.03× η_opt** |

The LR is already right, to 3%. That is a genuinely good sign about the existing
config and it should not be disturbed casually.

C = 1.7e16 is *below* DeepSeek's fitted range, so treat the absolute numbers as
soft. The **ordering** is what to act on, and it is robust to a lot of
extrapolation error: B_actual < B_opt, and a second, independently-fitted law
(Zhang et al.'s critical batch size) puts the CBS higher still (~118k). Two laws,
different groups, different data, same direction.

**Recommendation: raise the batch to ~64k tokens** (grad_accum 4 → 8). Re-tune
the LR afterwards — B_opt and η_opt were fitted *jointly*, so changing one
invalidates the other's calibration.

### The warmup is 1.2% of the run, by accident

DERIVED: burn calls `lr_step()` once per **dataloader batch**, not per optimizer
step. So `warmup_batches: 200` is **50 optimizer steps** out of 4,111 = **1.22%**.

Typical practice is 1–2% of steps, so this lands in range — for the wrong reason,
and it will silently scale the wrong way the moment `grad_accum` changes. Which
§3 just recommended changing. Fix the unit before the sweep, or the sweep
measures the warmup as much as the LR.

---

## 4. Model size: keep 22M, and lift the rank cap instead

The untying result is confirmed and it was the right call — MEASURED:

| run | valid loss | word PPL | BLiMP |
|---|---:|---:|---:|
| `quark_3m` (1×12 tied) | 3.706 | 115.163 | 57.05 |
| `quark_22m` (12×1 dense) | **3.361** | **74.965** | **61.76** |
| DERIVED delta | **+0.345 nats** | 1.54× | +4.71 |

At **identical FLOPs**, +0.30 GB, 7.6× the stored parameters. RESULTS.md §3
predicted this and §9 pre-registered the falsification ("quark_22m fails to beat
quark_3m"). It did not fire.

Now: **do not grow the model further.** §1 is why — D/N is already 0.33×
Chinchilla and growing N makes it worse. The 4.7GB of headroom is better spent on
epochs (§2) and batch (§3), both of which improve the ratio instead of degrading
it.

### The cheapest untested lever: d_emb 128 → 256

DERIVED: quark's head factorizes as d_model(384) → d_emb(128) → vocab(8192). The
logit matrix therefore has **rank ≤ 128** regardless of what the 384-dim body
computes. That is Yang et al.'s softmax bottleneck (arXiv:1711.03953), and
**quark_22m untied the layers but left the head capped.**

MEASURED, ALBERT Table 3 — the **not-shared** row is monotone in E:

| E=64 | E=128 | E=256 | E=768 |
|---:|---:|---:|---:|
| 81.3 | 81.7 | 81.8 | 82.3 |

The famous "E=128 is optimal" result holds **only in the all-shared row** — which
`quark_3m` was in and `quark_22m` is not. Untying moved quark into the row where
bigger E is monotonically better.

Caveat worth stating plainly: quark's rank-128 factorized softmax at 22M is
territory nobody has measured. This is a cheap experiment, not a known win.

### Depth: quark is already where the search lands

DERIVED aspect ratio (d_model / n_layers):

| quark | GPT-2 small | MobileLLM-350M (final) | Kaplan (48,1600) |
|---:|---:|---:|---:|
| **32.0** | 64.0 | 30.0 | 33.3 |

quark is already **2× thinner than GPT-2** and sits essentially where MobileLLM's
architecture search *landed*. "Go deep and thin" is advice quark has already
taken. The big cheap win here is banked.

The remaining question is whether 12 layers is slightly too few. Levine et al.
(arXiv:2006.12467) is the single most relevant paper in this document after §2 —
decoder-only autoregressive LMs, 10⁶–6·10⁸ params (spanning quark), R²=0.998 —
and its fit gives **L_opt ≈ 14.5 at width ≈ 345**, with N_Transition(12) = 13.0M
< quark's 20.6M. Reading: quark is **not too deep; slightly deeper is indicated**.
Clean iso-param swaps are 17×320 (−0.5% params) and 14×352 (−2.9%).

This is a **third-order** effect and I am not recommending it before §2 and §3.
Two honesty notes: MobileLLM disagrees with Levine (L≈17–20 vs L≈14–15), and
Levine's §5.2.3 identifies a third regime — "a network can be too deep" — that
their own theory explicitly does not predict, and quark lives in it.

**UNSUPPORTED**, found while checking this: GPT-2's 1/√N residual init has **zero
ablation** — the string "ablat" appears 0 times in the paper, and the official
repo uses a flat 0.02. quark's "1/√(2N)" is *nanoGPT's* interpretation of GPT-2,
not GPT-2's. Depth-μP §10.3 calls 1/√L "brittle". Not a bug; just not the
pedigree it's assumed to have.

### Kaplan's law applies to quark, by luck

DERIVED, and worth recording because it licenses everything above:

| component | cost | vanilla |
|---|---:|---:|
| attention (GQA, 2 kv heads) | 2.667 d² | 4.000 |
| FFN (SwiGLU, d_ff=1152) | 9.000 d² | 8.000 |
| **total per layer** | **11.667 d²** | **12.000** |

Error vs Kaplan's `N ≈ 12·L·d²`: **2.8%**. And 12 × 11.667 × 384² =
**20,643,840** — exactly `compute_equivalent_params()`.

GQA's cheap attention very nearly cancels SwiGLU's expensive FFN. So Kaplan's and
Levine's depth/width fits read off directly, with no re-derivation. This is not a
designed property — it is a coincidence, and it stops holding the moment GQA or
d_ff changes.

---

## 5. The tokenizer: no, and somebody already ran the experiment

Issue #6 proposes replacing BPE with a learned ≤5M-param compressor, citing a
ChatGPT report and a Kimi report ("не надо слепо доверять" — don't trust
blindly). Good instinct. Here is what checking them turned up.

**Lester et al. (arXiv:2404.03626) ran this experiment at 25m** — quark's size
class — MEASURED:

| method | bits/byte | FLOPs/byte |
|---|---:|---:|
| SentencePiece | **1.12** | **11.69M** |
| EqualInfoAC | 1.25 | 15.42M |

DERIVED: **strict Pareto domination** — worse loss *and* more compute. Their §4,
verbatim: "**Our SentencePiece baseline outperforms all other methods.**"

And the gap **widens as scale falls**: +0.070 bits/byte at 2b, **+0.130 at 25m**.
Small models are hurt *more*, not less. This is the opposite of the proposal's
premise.

The other cited work does not say what the reports imply it says:

- **BLT** needs a separate **100M-param** entropy patcher — larger than quark
  entire — and states "BPE models perform better with small training budgets".
- **MEGABYTE**'s compute-controlled table has **no subword baseline**.
- **H-Net** (760M/1.3B) "start[s] off worse... but scale[s] better", crossing
  over only after 30B–200B bytes. WikiText-103 is ~0.5B.
- **ICAE, 500xCompressor, Gist, xRAG, AutoCompressor** are **context
  compressors, not tokenizer replacements**. Different problem.
- **FSQ** has never been tested on text.

### The fatal problem is measurement, not quality

A lossy latent **cannot report perplexity at all**. LCM §2.4.1: "cannot produce
the probability explicitly." CALM (arXiv:2510.27688) §3.1: "Standard evaluation
metrics like Perplexity... **can no longer be computed**" — and that is with a
>99.9%-accurate codec.

quark's entire evidence base is word PPL and bits/byte, frozen in
`experiments/protocol_fixture.json` and asserted by a test. This change would
delete it. That is not a cost worth paying for a technique that also loses.

### The premise itself is unsupported

**UNSUPPORTED**: "BPE collects ugly fragments and that hurts quality"
("неприятные некрасивые куски слов"). This project went looking and found the
opposite:

- SuperBPE at 200k vocab was **worse** on bits/byte (0.7465 vs 0.7482).
- Schmidt et al. (arXiv:2402.18376) tested "fewer tokens ⇒ better" directly and
  found it "**not to be the case**".

The fragments are ugly **to read**. There is no measurement that they are ugly
**to learn from**. Those are different claims and only the first one is true.

### Vocab size

MEASURED: arXiv:2407.13623 fits N_v = 0.20·C^0.42 — but note it fits
**N_v = V·d, not V**, and its smallest IsoFLOP row is **3B**, 136× quark. DERIVED:
three independent routes converge on **V_opt ≈ 3.6K–9K** at 22M. Its §5/Fig 7
also shows the data-constrained optimum *shrinks*, and quark is data-constrained.

**Verdict: 8192 is defensible, plausibly slightly large. The only experiment
worth running is 4096 vs 8192. Never larger.**

Counter-evidence, reported because it is real and it disagrees: arXiv:2311.01955
Table 5 measures vocab 8k→40k = **+5.5 BLiMP**. That is a **masked** model and
transfer to causal is **unverified**. It is the strongest argument against the
paragraph above and you should know it exists.

---

## 6. Dataset: add a track, don't switch

The issue is right that WikiText-103 is old and lightly filtered. But:

Every number this project owns — RESULTS.md's three runs, quark_22m, the frozen
protocol, the GPT-2 baseline — is on WikiText-103. Switching corpora forfeits all
of them **simultaneously**, and replaces a measured comparison with an unmeasured
one. The next run would have no baseline at all.

**Recommendation: keep WikiText-103 as the anchor and add a filtered corpus as a
second track**, reporting both. Candidates worth evaluating, in rough order of
fit: FineWeb-Edu (filtered, English, and the classifier is public), Cosmopedia,
TinyStories (as a diagnostic, not a target).

**This is the least-researched section in this document** and I want to be
explicit about that rather than pad it. I do not have a primary-source
recommendation between those candidates at 22M with a like-for-like eval. What I
can say from §2 is that the corpus size question is *downstream* of the epoch
question: at 4 epochs quark extracts 501.9M effective tokens from 135M unique
ones, so "more data" is not the constraint that a corpus swap would solve. Fix
the epochs first; the dataset choice gets easier to measure once there is a
4-epoch baseline to measure it against.

---

## 7. Calibration: BLiMP 61.76 is better than it looks

Worth fixing the target before optimizing against it. The 80–84 figures quark is
implicitly measured against are **masked models scored by pseudo-log-likelihood**.
That is not the measurement quark makes.

MEASURED, Salazar et al. 2020 Table 7: BERT-base PLL **84.2** vs GPT-2-345M
true-LL **82.6** — "despite using less than half the data and a third of the
capacity." The scoring method is worth ~+1.6 to +10.

The apples-to-apples target is a **causal** model on **quark's own corpus**,
scored on **full unfiltered BLiMP** — MEASURED:

| model | BLiMP |
|---|---:|
| Transformer-XL, causal, WikiText-103 (BLiMP TACL 2020 Table 3) | **68.7** |
| **quark_22m** | **61.76** |
| OPT-125M, causal, 10M words | 62.6 |
| 5-gram, 3.1B words | 60.5 |

quark_22m sits at OPT-125M's number with **1/6 the parameters**. **The honest
target is 68.7, not 80.**

Two caveats that cut in **opposite** directions — report both or neither:

1. BabyLM **filters** BLiMP (13.7% removed) and warns results "cannot [be]
   directly compare[d]".
2. The BabyLM evaluator **counts ties as correct** (2025.babylm-main.16 §4: "In
   **22 of the 67 BLiMP subtasks**, the two sentences in each minimal pair are
   permutations of the same multiset of words... ties are counted as correct,
   which inflates accuracy"). The true random floor is **0.543, not 0.500**, and
   an order-blind Zipf-frequency baseline scores **0.663** — beating quark while
   ignoring word order entirely.

A causal LM never ties, so it collects **none** of that credit. DERIVED:
**quark's 61.76 is understated against published BabyLM figures**, by an amount
nobody has quantified.

BLiMP's own §6.3, verbatim, and it is the most useful sentence for planning here:

> **increasing model size (number of parameters) is unlikely to improve
> performance**... All models have overall BLiMP accuracy of 0.84 ± .01%...
> **amount of training data has the biggest impact.**

That is §1's conclusion arrived at independently.

---

## 8. Architecture micro-techniques: the honest scorecard

| technique | verdict | why |
|---|---|---|
| **ReLU² instead of SwiGLU** | Neutral on quality; **25% less FFN activation memory** | See below — it is not a FLOP decision |
| **QK-norm** | Neutral on loss at a tuned LR | Wortsman tests **quark's exact scale** |
| **z-loss** | Skip — redundant | Redundant *given weight decay*, which quark has |
| **Attention softcap** | Contested; see below | Gemma 3 **removed** it; modded-nanogpt measured it helping |
| **Sliding-window attention** | **Provable no-op** | A window ≥ the context is not a window |
| **Flash attention** | **Do it** | Frees ~2.25 GB; pure engineering, no quality question |

### ReLU² is a memory decision, not a FLOP decision

DERIVED, and this kills the usual framing:

| | SwiGLU | equal-param ReLU² |
|---|---:|---:|
| d_ff | 1152 (3 matmuls) | **1728** (2 matmuls) |
| FFN params/layer | 1,327,104 | **1,327,104** |
| matmul FLOPs/token/layer | 2,654,208 | **2,654,208** |

**Equal params implies equal FLOPs, exactly.** The "2× cheaper" framing is an
artifact of comparing at equal d_ff, which is not a fair comparison.

What does change: FFN activations retained per token per layer drop **2304 →
1728 (25% less)**. And DERIVED: **the FFN is 73% of quark_22m's parameters**, so
this matters more here than in the papers it comes from.

The only numeric equal-param head-to-head that exists (arXiv:2402.03804 Table 2,
at 1B) is a **tie**: SwiGLU 50.53 vs ReLU² 50.48 — with ReLU² given 1.59% *more*
FFN params. Decide on engineering grounds; there is no quality argument either
way.

**UNSUPPORTED**, and this one is worth flagging loudly: the circulating claim
that ReLU² gets "95% of SwiGLU's quality with 15% faster training" is
**fabricated** — it has no source. Primer's ReLU²>SwiGLU claim is **figure-only**
(Fig 5); the Appendix A.7 table said to isolate it **does not exist**. Shazeer's
GLU paper contains **no ReLU² at all** — its real lesson is that *all GLU variants
beat all non-GLU variants*; the gain is the gate. And modded-nanogpt has **never**
compared ReLU² to SwiGLU (zero grep matches).

### QK-norm and z-loss are LR-stability tools, not free wins

Wortsman et al. (arXiv:2309.14322) tests **N=1.9e7 with seq_len 512** — quark's
exact scale, which almost nothing else does. Appendix B, verbatim:

> the LR vs. loss curves are **indistinguishable up to some critical learning
> rate** when using qk-layernorm (Figure 1), adding z-loss (Figure 3), or
> changing warm-up.

**At a well-tuned LR these are neutral on loss.** The "free win" framing is
UNSUPPORTED. What they buy is ~25× less LR sensitivity — valuable if you are
about to run an LR sweep (§3), worthless otherwise. If you add QK-norm, do it
**per-head with no biases** (their Fig E.8). z-loss's origin is the Mesh-TF
**codebase, not a paper**.

One gap worth naming: Wortsman's 1.9e7 model saw ~690 tokens/param; quark sees
6.5 — **~106× apart**. The scale matches; the regime does not.

### Softcap: two primary sources, opposite conclusions

- Gemma 2's 50.0/30.0 softcap has **zero ablation**, and **Gemma 3 removed it**,
  saying only "we replace the soft-capping of Gemma 2 with QK-norm".
- modded-nanogpt measured **logit** softcap over **80 runs, p=0.0001**, and
  reports it "improves performance **in the small-scale regime**".

These directly contradict each other. Reporting both. If you test one thing here,
test logit softcap — it is the one modded-nanogpt technique with a primary-source
argument for transferring *downward*, which is the direction quark needs.

### Sliding window is provably nothing

DERIVED: a w=1024 window over a ≤1024 causal sequence excludes **nothing** at any
position — the output is bit-for-bit identical to full causal attention. quark's
seq_len is 512. And the KV cache it would shrink is **6.29 MB = 0.037% of a 16 GB
card**. It saves a rounding error, on the one resource quark is not short of.

---

## 9. Optimizers: tune the one you have

MEASURED, arXiv:2509.02046 Fig 1, verbatim:

> **Up to a 2× speedup is achievable by tuning a single hyperparameter (learning
> rate)** in the GPT-3 recipe for a 100M model.

Same paper, Fig 3: for any *alternative* optimizer, "**The highest speedup is
capped at 1.4×**".

So the honest expected value of switching optimizers is **~1.3–1.4×**, and it is
*smaller* than tuning the LR of the one you already have. Three independent
sources at 124M–190M converge on this. AlgoPerf's 2024 winner managed **1.28×**
under controlled conditions — every "2×" in the wild is 60% larger than anything
that has survived a competition.

### Muon: best-in-class coverage, worst-case batch

DERIVED: **95.15% of quark is Muon-eligible** — better than any published Muon
result (Keller's 124M NanoGPT is ~31% embedding). quark's factorized d_emb=128 is
why. Structurally this is the *best* case for Muon.

And yet — Keller's own Muon docstring, verbatim:

> **We believe it is unlikely to work well for training with small batch size.**

His batch is 524,288 tokens. quark's is 32,768 — **16× smaller**.

DERIVED, Newton-Schulz overhead by **his own formula** T·m/B: 5 × 384 / 32,768 =
**5.86%**, vs his 0.7% headline — **8× worse**, and likely optimistic, since a
384×384 NS iteration on one GPU is latency-bound, not FLOP-bound.

The smallest model in the Muon paper is **399M — 18× quark**. Its own Table 3 fit
encodes a **shrinking** advantage (1.92× at 399M decaying to 1.72× at 1.5B): the
famous "2×" is the smallest-model endpoint of a decaying curve, not a floor. And
the cleanest primary ablation in Keller's own records (`102924_Optimizers`) is
Muon 3.2760 vs Adam 3.3406 at equal steps — **+0.065 nats for +2.45% wallclock**,
with SOAP marginally *better* than Muon. Muon's share of modded-nanogpt's 34×
speedup is **6.6% in log-space**.

If you implement it anyway, implement **Eq 4** (not Eq 7) so lr=3e-3 transfers.

### AdEMAMix is the sleeper, and here is why that's not a hunch

arXiv:2509.02046's own reconciliation, verbatim — note that it cuts **for**
AdEMAMix and **against** Muon at quark's batch specifically:

> Since Mars and AdEMAMix both perform gradient averaging and variance reduction,
> these methods are advantageous in their **noise-dominated small-batch regime**,
> whereas in our larger-batch setting these benefits diminish and matrix-level
> optimizers become more competitive.

**quark *is* the noise-dominated small-batch regime.** That is the same sentence
that explains why Muon is the wrong pick here.

Config detail that decides it: with β3=0.9999 over quark's 4,111 steps, only
**33.8%** of the asymptotic mass accumulates — the technique never engages.
**β3=0.999 → 98.4%.** And Fig 17, verbatim: "Using β3=0.999, the gap between
AdEMAMix and AdamW **increases as the number of iterations decreases**." Caveat:
their shortest LM run is 32,000 iterations at 531k tokens/step, smallest model
110M.

### Skip these outright

- **Sophia.** Sophia-H was **deleted from the repo one day after arXiv v1**
  (commit 195a786). A reviewer read **1.25×** off the authors' own figure. The
  author admitted "6e-4 learning rate fails with some random seeds and we
  selected the random seed with which it works". Kaddour et al. re-ran it **not
  counting Hessian steps** — "Surprisingly, we still do not observe any speedup."
  Semenov et al.: "Sophia **diverges in the small-batch setting**." (**UNSUPPORTED**,
  while checking: "sophia" appears **0 times** in all 102 pages of AlgoPerf.)
- **Lion.** Its own §6: "it is likely that Lion performs **no better than AdamW
  if the batch size is small (<64)**." quark is at exactly 64 sequences.
- **SOAP**, if you try it: its default `max_precond_dim=10000` lets quark's 8192
  vocab slip **under** the threshold → **537 MB** of preconditioner.
- **Adam-mini.** Its Table 8's smallest model is **39M with d_model=384** —
  quark's exact width, which is rare and worth noting — and Table 4 shows
  **+0.0096 nats**. But it is a *memory* optimizer, and the ~85MB it saves is
  irrelevant on a 16 GB card with 4.7 GB free.

---

## 10. Distillation: there is no evidence, not thin evidence

Worth stating precisely because it is the obvious next idea and it is a trap.

**For causal-LM pretraining distillation below 30M params, primary evidence is
ZERO — not thin, absent.** Only two papers qualify (<30M, pretraining-stage,
same-size control) and **both are masked**: MobileBERT (25.3M) and TinyBERT₄
(14.5M). Both have a catch that undoes the headline:

- MobileBERT Table 9's +3.6/+3.3/+2.7/+2.4 is **almost entirely Feature Map
  Transfer** — pure logit distillation alone is **+0.3 MNLI-m**. And Turc et al.
  (arXiv:1908.08962) **directly contradicts** the FMT result: "We experimented
  with related approaches [intermediate activations], but found only slight gains
  which were dominated by the gains from pre-training."
- TinyBERT₄'s 70.2 → 77.0 includes task distillation + data augmentation. The
  clean GD-only ablation is **−3.1**.

The nearest causal datapoints are all *above* quark: 44M (+1.2 BLiMP), **58M**
(Baby Llama, +3.73 BLiMP — and note it is 58M, not <30M), 345M (+1.1–1.3).

Three facts that settle it:

1. **Baby Llama did not win BabyLM 2023.** ELC-BERT — 24M, **zero
   distillation** — won both tracks. The strongest sub-30M BabyLM result in
   existence is a non-distillation result.
2. **The trajectory is down.** 2023: "common and often successful". 2024: four
   sentences, no KD winner (GPT-BERT, zero distillation, won both). 2025: not
   discussed, ~zero submissions. AntLM independently **failed to replicate**:
   "our own replication of the BabyLlama model through distillation did not
   achieve ideal results."
3. **Distillation Scaling Laws (arXiv:2502.08606)**: "**smaller models are more
   likely to benefit from supervised pretraining**, whereas larger models are
   more likely to benefit from distillation"; U-shaped student error;
   "distillation can not produce lower model cross-entropies than supervised
   learning when both learning processes are given enough data or compute."
   (Starts at 143M — so it too is an extrapolation down to quark.)

**Verdict: skip.** If it is ever revisited, the counterweight worth knowing is
that "a small teacher suffices" appears twice independently, and the Law of
Capacity Gap puts the optimal teacher at **~2.5× the student** — so a ~55M
teacher, not a 7B one.

---

## 11. ROCm and Vulkan

Issue #6: "можешь добавить rocm — так как у меня карта amd... не надо это
тестировать, просто добавь возможность." Done, and untested per instruction —
nobody here has an AMD card.

Two features, not one: `rocm` and **`vulkan`**. Vulkan is `burn-wgpu` with a
SPIR-V compiler instead of WGSL — the same GPU stack you already train on,
compiled for a backend AMD drivers handle better. It is **not** a third stack.

**Try `vulkan` first.** MEASURED, burn's own matmul benchmarks
(burn.dev/blog/sota-multiplatform-matmul): on the RX 7600 the autotune-favored
variant is **~4.6–5.0 TFLOPs on ROCm vs ~11.5–12 on Vulkan**. Vulkan led on both
AMD cards they measured. That is why `Default` ranks wgpu → cuda → **vulkan** →
rocm → ndarray.

```
cargo build --release --features vulkan     # try this first
cargo build --release --features rocm       # needs Linux + ROCm 6.4.x-7.1.1
```

Known ROCm hazards, all upstream, none fixable here: burn#4202 (segfault if `cpu`
and `rocm` are both enabled), cubecl#1365 (**RDNA2 effectively broken**),
cubecl#1147 (RX 7900 XTX + ROCm 7.1 BF16 conflicts), and an `assert_eq!` panic at
init on an unrecognized `gcnArchName`. `burn-rocm`'s README is stale —
`CUBECL_ROCM_PATH` and "ROCm 6.2.2" no longer apply.

CI checks all three backends (`wgpu`, `vulkan`, `rocm`) at `cargo check`. It
cannot go further: no runner has a GPU, and `cargo build` of the binary reaches
the linker and fails on undefined `hip*` symbols. Compiling is the guarantee on
offer. (`cuda` is deliberately absent — it needs a toolkit to build at all, so a
check job would test the runner image rather than this code.) Per the issue's main
rule — "не тестируй тяжёлые вычисления локально" — no heavy compute was run
locally for this document; it is arithmetic and reading.

---

## 12. The bug in your run, which is not cosmetic

Your Learner Summary says `Total Epochs: 10` on a `num_epochs: 1` config. You
wrote it off — "там просто runs конфиг мусорный но это честно одна эпоха". The
epoch count is honest; the summary is not, and the reason is a real hazard.

Root cause, confirmed against burn v0.21.0's source:

1. `FileLogger::new` opens with `.truncate(true)` → a new run overwrites the
   epochs it reaches and **leaves later epochs from the previous run on disk**.
2. `FileMetricLogger::epochs()` returns the **maximum** `epoch-<n>` it finds.
3. `LearnerSummary::new` sets `epochs = logger.epochs()` and iterates `1..=n` →
   your `Total Epochs: 10`, with epochs 2–10 carrying **run3's** numbers. (Train
   max 5.033 @ epoch 6, valid max 4.725 @ epoch 6 — byte-identical to
   RESULTS.md §5. It is run3's data, verbatim.)
4. `MetricCheckpointingStrategy` → `find_epoch` takes the **min over all epochs
   found**, including stale ones.
5. `best_valid_loss_epoch` then names a **stale** epoch, and `run()` loads
   `checkpoint/model-<stale>` — **weights from the previous architecture**.

Your run escaped step 5 **by luck**: 3.361 beat run3's best of 3.707. Had
quark_22m been slightly worse, it would have silently evaluated a `quark_3m_dense`
checkpoint and reported it as quark_22m.

Fixed in this PR: training into a directory that already holds `epoch-<n>` logs
now **refuses to start**, unless `--resume-from-epoch` says the merge is
deliberate. Four tests cover it, including one that reproduces the stale-checkpoint
selection with the guard removed.

---

## 13. Recommended order

Each step is gated on the one before it, and the ordering is the point — §1 says
the exotic stuff is second-order.

1. **Fix the warmup unit** (§3). Cheap, and everything downstream is an LR-adjacent
   sweep that the current unit quietly corrupts.
2. **4 epochs + dropout 0.1, wd swept {0.1, 0.5, 1.0, 2.0}** (§2). ~4 GPU-hours.
   The only recommendation here fitted at quark's scale, and the largest expected
   win. *Everything below is noise until this runs.*
3. **Batch → ~64k tokens, then re-tune the LR** (§3). Two independent laws say
   quark is starved.
4. **Flash attention** (§8). ~2.25 GB freed, no quality question, funds steps 2–3.
5. **d_emb 128 → 256** (§4). The cheapest untested lever, in the ALBERT row quark
   just moved into.
6. Only then, and only one at a time: logit softcap (§8), 14×352 depth (§4),
   vocab 4096 (§5), AdEMAMix β3=0.999 (§9).
7. **Not on this list:** neural tokenizer (§5), distillation (§10), Sophia/Lion
   (§9), sliding window (§8), a bigger model (§1).

### What would falsify this document

In RESULTS.md §9's spirit — the cheapest experiments that would prove sections of
this wrong:

- **§2 is wrong if** 4 epochs + dropout does not beat 1 epoch by more than seed
  noise. That would mean Muennighoff's fit does not survive removing his data
  scale, and it is the load-bearing claim here.
- **§4 is wrong if** d_emb 256 does not beat d_emb 128 at 22M untied. That would
  mean the rank-128 cap is not binding and ALBERT's not-shared row does not
  transfer to causal.
- **§3 is wrong if** batch 64k underperforms 32,768 at a re-tuned LR. That would
  mean both DeepSeek's and Zhang's laws break below their fitted range.
- **§9 is wrong if** a tuned-LR AdamW is beaten by >1.4× by any alternative.
- **§5 is wrong if** vocab 4096 beats 8192 by more than the 8k→40k masked result
  suggests it should lose by.

An open item RESULTS.md §9 still wants and issue #6 did not answer: `GradRms.log`
was not attached, so the cheapest falsification there — whether AdamW's
`eps = 1e-15` is justified — remains unanswered. (Your run used `eps = 1e-8` from
an older `config.json` regardless.)
