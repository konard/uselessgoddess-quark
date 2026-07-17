# Results: the three WikiText-103 runs, and what to do next

Answers issue #3. Every number here is reproducible:

```
python3 experiments/run_analysis.py     # -> experiments/out/run_analysis.txt
```

Numbers marked MEASURED are from the issue's console output or a primary source.
Numbers marked DERIVED are computed by that script. The distinction matters
because this document **reverses a pre-registered design decision**, so it must
be auditable which numbers are observations and which are inference.

Sources for every competitor claim: [`experiments/research/competitors.md`](../experiments/research/competitors.md).

---

## 0. The decision, up front

The issue asks for a decision, not a survey. Here it is.

| Question | Decision |
|---|---|
| **Parameter count** | **Raise it to 21.8M, by untying the loop.** Not by growing the model — `quark_3m` *already ran* this compute graph. It costs **0 extra FLOPs**, +0.30 GB, and buys **7.6× the parameters**. |
| **Next run** | **`quark_22m` vs `quark_3m`**, same everything else. It settles the question DESIGN.md §5 pre-registered and the reported runs cannot answer. |
| **Headline target** | **Drop "beat GPT-2 124M on WikiText-103 word PPL" as the headline.** It is a contest quark plays on *easy mode* (in-domain vs GPT-2's zero-shot) and still loses 2.89×, against models that are 97% embedding table. Keep it as a secondary metric — it is already implemented and tokenizer-independent. |
| **Niche (paper)** | **The compute-allocation study at 3–30M.** quark is uniquely positioned: two of the three runs already exist, the third is one config away, and the result so far *contradicts a 2025 ICLR paper* at a scale nobody has tested. |
| **Niche (artifact)** | **Defer one run.** The evidence says 3M is below the floor where this size class does anything useful. Whether 22M clears it is exactly what the free experiment measures. Do not pick a product niche before the measurement that decides if there is one. |

The rest of this document is why.

---

## 1. What the runs establish

Three runs, identical eval protocol, single seed each. MEASURED:

| run | config | valid loss | word PPL | BLiMP | VRAM | time |
|---|---|---:|---:|---:|---:|---:|
| run1 | `quark_3m` — 1 unique layer × 12 loops, d_model 384 | 3.706 | 115.163 | 57.05 | ~11.0 GB | ~60 min |
| run2 | `quark_3m_dense` — 6 unique × 1, d_model 168, 1 epoch | **3.653** | **108.275** | 58.63 | ~5.5 GB | ~15 min |
| run3 | `quark_3m_dense` — same, 10 epochs | 3.707 | 123.193 | **60.93** | ~5.5 GB | ~150 min |

**The eval harness is trustworthy.** DERIVED: every run's token PPL, word PPL and
bits/byte reproduce from its own total NLL alone, to the printed precision, at
tokens/word = 1.3266. Nothing below would mean anything if this failed.

**The pre-registered test fired, and the pre-registered answer was wrong.**
DESIGN.md §5 says: *"`quark_3m_dense` is the honest control — if it matches
`quark_3m`, looping bought nothing."* It did not match. It **won**, at half the
VRAM and a quarter of the time.

---

## 2. What the runs do *not* establish

**run1 vs run2 is confounded three ways.** The configs differ on width (384 →
168), layer diversity (1 → 6 unique) *and* depth (12 → 6 applications),
simultaneously. When run2 wins by 0.053 nats there is no way to attribute the
win. So the falsification test DESIGN.md pre-registered **cannot actually be
settled by these runs** — the control was not a control.

What run1-vs-run2 *does* show is narrower and still interesting: *at a 3M budget,
spending sharing's savings on **width** does not pay for the loss of **layer
diversity**.* That is a real result, and per §4 it appears to be untested in the
literature.

**It does *not* contradict the recent literature, and an earlier draft of this
section was wrong to imply it might.** Saunshi et al. (arXiv:2502.17416, ICLR
2025) find looping improves perplexity **at iso-parameter** — a looped model
beats a *dense model of the same stored size*. run1-vs-run2 is not that
comparison, and neither is anything else here. Contradicting Saunshi would take
`quark_3m` against a genuinely dense 2.87M model; no such run exists.

The comparison this project *can* make is iso-**FLOP**, and issue #6 has now
made it: see §3 and §10. It came out in Saunshi's direction, not against it.

**run2 vs run3 is confounded by the LR schedule.** DERIVED, by reconstructing the
schedule and validating it to 10 significant figures against the logged LRs:
run2's single epoch ended **fully annealed at 3.0e-4**; run3's best checkpoint
(epoch 3) sat at **2.74e-3, ~95% of peak, never annealed**. An un-annealed
mid-cosine checkpoint is systematically worse than an annealed one. That alone
explains much of 108.28 → 123.19 without invoking overfitting.

**Single seed.** The run1-vs-run2 delta is 0.053 nats; the BLiMP delta is 1.58
points. Neither is obviously outside seed noise, and seed variance here is
unmeasured. **Any claim resting on these deltas is provisional.**

---

## 3. The central finding: run1 paid for 22M and stored 3M

This is the one that changes the plan, and it needs no fitted constant and no
new experiment. It is arithmetic.

Weight sharing reduces **storage**, not **arithmetic**. Looping one layer 12
times costs exactly what 12 distinct layers cost. So compare `quark_3m` against
its untied twin — same width, same depth, same head count:

| | `quark_3m` (1×12) | `quark_22m` (12×1) |
|---|---:|---:|
| d_model / d_ff / heads | 384 / 1152 / 6 | 384 / 1152 / 6 |
| layer applications | 12 | 12 |
| **compute-equivalent params** | **20,643,840** | **20,643,840** |
| **stored params** | **2,868,352** | **21,800,320** |

DERIVED, and pinned by `src/config.rs::untying_quark_3m_is_free_in_arithmetic_and_buys_parameters`
so it cannot silently rot:

- Identical compute-equivalent params ⇒ **identical FLOPs/token**.
- Identical width and depth ⇒ **identical activation memory**.
- The only cost of untying is **0.30 GB** of extra weights, grads and Adam state
  — against run1's **MEASURED 11.0 GB** on a 16 GB card.

> **run1 spent 60 minutes and 11 GB running a 20.6M-parameter compute graph, in
> order to store 2.87M parameters.**

The 3.0M budget was never the binding constraint. **VRAM is** — and VRAM is
consumed by activations, which track *compute*, not stored weights. The
checkpoint is ~11 MB. Storage was abundant and free; the project spent its
scarce resource to conserve its abundant one.

And the function-class argument is exact: **tie all 12 layers of `quark_22m` and
you recover `quark_3m` exactly.** The looped model's hypothesis class is a
*strict subset*. At equal compute, the dense model cannot be worse except through
optimization or overfitting effects. If it *is* worse, that is a genuinely
interesting result and the first thing to report.

### 3.1 Is 7.6× the right size for the gap?

DERIVED. The measured gap from run2 to GPT-2 is **1.060 nats/word**. Pricing that
gap in parameters under *both* fits `scaling_budget.py` carries:

| fit | multiplier to buy 1.060 nats at N=2.87M |
|---|---:|
| Hoffmann | 4.7× |
| Besiroglu | 4.1× |

**Untying pays 7.6×, for free.** Two fits that disagree about the constants agree
the lever is scaled to the problem — same order of magnitude, on the correct side
of it.

This is **not** a prediction that `quark_22m` reaches 37.50, and DESIGN.md §1.1
explicitly forbids using these fits as one: quark is 15× below the smallest model
Chinchilla fitted, the fit is OWT nats/token, and the target is WikiText-103
nats/word. The claim is only that the free lever is scaled to the gap rather than
dwarfed by it. The argument that needs no fitted constant is the one above:
Chinchilla's `N` counts **stored** params, and sharing is precisely a reduction
in stored params at fixed FLOPs. It can only lower the capacity ceiling.

---

## 4. The competitive picture, and a correction to our own premise

Full sources: [`experiments/research/competitors.md`](../experiments/research/competitors.md).

### 4.1 The existence proof is weaker than README.md claims

README.md says WikiText-103 is *"**Plausible**, with a published existence proof:
a 4.5M-parameter transformer body already beats GPT-2's zero-shot 37.50."*

MEASURED (Bai et al. 2019, arXiv:1909.01377, Table 3): DEQ-Transformer small has
**4.5M non-embedding params, 32.4 test PPL — and 138M params in total.** DERIVED:
the embedding table is **133.5M, i.e. 97% of the model.**

DESIGN.md §2 is honest about this and says so. But the inference it draws —
*"word-level WikiText-103 is a vocabulary-storage problem, not a modeling one"* —
is doing more work than it can bear. That 267,735 × ~500 embedding is not inert
storage; it is the output softmax, and it is *modeling*. quark's tied `d_emb=128`
caps `rank(logits) ≤ 128` (the softmax bottleneck, Yang et al. 2018). DEQ's is
~4× wider over a 33× larger vocabulary. **Those are not the same model with
different storage. The existence proof shows the small *body* is not the
obstacle; it does not show a small *total* can do it.**

Confirming the gap: **no published transformer under 30M total params reports
word-level WikiText-103 perplexity at all.** The 267,735-word vocab makes it
arithmetically near-impossible — a plain embedding alone exceeds 30M at any
`d_embed > 112`, and the best-known compression (Baevski & Auli's adaptive
inputs) still costs ~44M. quark's sub-word route is legitimate — it is the one
Baevski & Auli sanction — but it means **quark has no peers on this benchmark,
and its "competitors" are models 48× larger that spend 97% of themselves on a
lookup table quark cannot afford and does not want.**

The gap is also *worse* than it looks in one respect, and this belongs in any
honest writeup. MEASURED (Radford et al. §2.1): *"We removed all Wikipedia
documents from WebText."* GPT-2's 37.50 is genuinely out-of-domain zero-shot;
quark trains in-domain. **quark is playing the easier game and still losing by
2.89×.** (It looks better in another respect: GPT-2's number includes 2.5–5 PPL
of de-tokenizer gains, per §3.1 of the same paper.)

### 4.2 The nominated competitor is not one

The issue calls cactus-compute/needle *"an interesting competitor"*. Checked
against the repo, docs, model card and HF API:

- MEASURED: 200B tokens on 16× TPU v6e in 27h. **30,427,676 params** per the HF
  API — the "26M" counts only embeddings + attention projections.
- **needle reports no perplexity, no WikiText-103, no BLiMP. It publishes no
  benchmark numbers of any kind.** Its README claim to beat FunctionGemma-270m
  et al. is supported by no table, number, or harness anywhere in the project.

needle is an encoder-decoder tool-calling model with no FFN, distilled from
Gemini. It shares a vocab size with quark and nothing else — different task,
different architecture class, no overlapping metric. Its one transferable idea is
the thesis behind dropping FFNs: *"At small scale, FFN parameters are wasted. ~2/3
of standard transformer parameters are FFN."* That is a claim about **where to
spend a fixed budget**, and it deserves testing on its own merits.

SmolLM2-135M likewise reports **neither WikiText-103 nor BLiMP**, and is 47×
quark's size trained on 2T tokens.

### 4.3 BabyLM: where this size class actually publishes, and it has a cliff

| model | params | BLiMP |
|---|---:|---:|
| GPT-BERT (2024 Strict winner) | 119M | 86.1 |
| **GPT-BERT (2024 Strict-Small winner)** | **30M** | **81.2** |
| ELC-BERT "Original" | 24M | 80.00 |
| WhatIf | 26M | 66.9 |
| BERTtime Stories | 24M | 63.2 |
| **quark_3m (run3)** | **2.87M** | **60.93** |
| Co4 | 8M | 53.55 |
| BitMar | 14M | 48.7 |

Three facts follow, and they are the most decision-relevant in this document:

1. **~24–30M is a demonstrated sweet spot.** The 2024 Strict-Small winner *is* a
   30M model, landing within 5 points of the 119M model.
2. **Below ~16M, BLiMP collapses toward chance.** 14M → 48.7. 8M → 53.55.
   **quark at 2.87M scores 57.05–60.93 — exactly where the curve says a model its
   size lands.** quark's BLiMP is not an anomaly to debug; it is the size class
   reporting in. The 4-point spread across the three runs is noise on top of a
   number the parameter count already fixed.
3. **Batch size is load-bearing at this scale.** Re-running ELC-BERT's 24M config
   at smaller batch sizes collapsed BLiMP **from 80.00 to 44.17–52.22 across all
   twelve re-runs** (aclanthology 2025.babylm-main.12 Table 2). quark's effective
   batch is 16 × 4 × 512 = 32,768 tokens.

Two things that won at this scale are worth stealing, and neither costs
parameters. GPT-BERT's entire trick: *"by shifting MLM predictions one position
to the right, the MLM predictions become aligned with next-token predictions from
CLM"* — no architectural change. And ELC-BERT trained *"over 2000 epochs"* for
Strict-Small, which independently corroborates Muennighoff et al. that quark's 10
epochs are nowhere near the repetition limit.

---

## 5. The epoch-6 divergence: what is known, and what is not

run3 is run2's config trained 10× longer, and it lost. Three explanations are
ruled out by the data:

- **Not a checkpoint bug.** `best_valid_loss_epoch()` reads the summary, filters
  NaN, min()s on valid Loss, and `run()` reloads it. It correctly selected epoch 3.
- **Not overfitting.** At the epoch-6 blow-up, **train 5.033 > valid 4.725**. A
  memorizing model has train ≪ valid. This is the opposite: the model got worse at
  data it had already seen. It **diverged**.
- **Not simply "peak LR too high".** The run *survived* epochs 1–5 at ~peak
  (2.93e-3 → 2.07e-3) and blew up in epoch 6 at a **falling** LR of 1.65e-3.

**Root cause: NOT DETERMINED, and not determinable from the attached logs** —
they cover epoch 1 only, so epochs 2–10 are unobservable.

That limitation is itself the finding. The harness reports per-epoch **means**,
which hide a spike until it is catastrophic, and it has no divergence detection.
**~60% of a 10-epoch run's GPU time produced nothing and left no diagnosable
trace.** I am not going to guess a mechanism I cannot evidence; the fix is to
make the next occurrence diagnosable, which is what Wortsman et al. (arXiv:2309.14322)
recommend: *"instabilities can be predicted before they emerge by examining the
scaling behavior of model activation and gradient norms."*

§6 item 2 builds the gradient half of that (`GradRms`, per batch, written to the
artifact directory and asserted by the end-to-end test). The activation half —
max attention logit — is **not built**, and §6 says so plainly rather than
letting QK-norm's existence imply the question is closed. **Nothing in this PR
diagnoses the epoch-6 divergence.** QK-norm and z-loss address a *hypothesis*
about it; they are not a diagnosis, and both ship off by default.

---

## 6. What to implement, in order

Each item is sourced and each is cheap. Nothing here is speculative.

1. **`quark_22m` — the controlled experiment.** ✅ *Done, this PR.* Changes
   exactly one variable versus `quark_3m`: `n_unique_layers` 1 → 12.
2. **Per-step instrumentation** — grad RMS, loss spikes, max attention logit.
   Sourced above. Without it, the next divergence is equally undiagnosable.
   **Partly done** — two of three:

   - ✅ **Grad RMS** *(this PR)*: `GradRmsMetric` logs `sqrt(mean(g^2))` over every
     parameter, per batch, to `<artifact_dir>/train/epoch-N/GradRms.log`. Per-batch
     and not a running mean, because a spike and a collapse are both *departures*
     from the mean, and averaging is the operation that hides them.
   - ✅ **Loss spikes** *(already existed)*: burn's `FileMetricLogger` already
     writes per-batch entries, which `experiments/run_analysis.py` reads. What was
     missing in run3 was not the logging but the *retention* — the attached logs
     cover epoch 1 only.
   - ❌ **Max attention logit** *(not done — scoped out)*: it needs an auxiliary
     output threaded through `QuarkLm::forward` → `Block` → `attention`, which no
     existing path carries. QK-norm bounds the logits by construction, but **only
     when enabled, and the reference config has it off** — so this is a real
     remaining gap, not a solved problem. If a divergence recurs with
     `qk_norm = false`, this is the first thing to build.

3. **QK-norm** — Wortsman et al. §3.1 validate it **down to 10M params**; it is
   the standard fix for attention-logit blow-up and enables higher LR.
   ✅ *Done, this PR*, **off by default**. Normalized **before** RoPE: RoPE is a
   rotation and preserves L2 norm, so it commutes with RMSNorm's scaling but not
   with its learned per-element gain (OLMo 2 and Gemma 2 both order it this way).

4. **z-loss** — same paper. quark had neither (grep found zero hits for both).
   ✅ *Done, this PR*, **off by default**. Penalizes `logsumexp(logits)^2`, which
   cross-entropy is invariant to: the reported loss is unchanged, the gradient is
   not.

5. **AdamW ε 1e-8 → 1e-15** — Wortsman §3.4: *"Decreasing ϵ to 1e-15 improves
   loss and mitigates a collapse in grad RMS."*
   ✅ *Done, this PR* — and this is the **only change to the reference run**, so it
   is the one that has to be able to lose. Item 2's `GradRms` is what makes it
   falsifiable: **if grad RMS never approaches 1e-8, epsilon bought nothing and
   should be reverted.** Until that number exists this is a bet on a paper written
   about larger models, not a result.

6. **Seed variance.** 3 seeds of `quark_3m_dense` at 1 epoch is ~45 min total and
   tells us whether 0.053 nats means anything. Every conclusion in §2 is
   provisional until this exists. ❌ **Not done — needs the 16GB GPU**, so it is
   yours to run; CI cannot fake it.

**Why 3, 4 and 5 default to off, and 5 does not.** 3 and 4 are new knobs, so
leaving them off keeps `quark_3m` byte-identical to the config that produced the
reported runs — the `quark_22m` comparison in §0 only means something if exactly
one variable moved. 5 is not a knob; it is an edit to the shared default, and it
is the one place this PR knowingly perturbs the baseline. It is called out here
so that a `quark_22m` result cannot later be quietly attributed to the epsilon
change, or vice versa.

### 6.1 A correction found while building item 2

`grad_clip_norm`'s doc claimed gradients are rescaled when their **global** norm
exceeds the threshold. That was false, and it is worth stating because it is what
the name means everywhere else. burn's `GradientClipping::clip_by_norm` takes a
single `Tensor<B, D>` and computes `sqrt(sum(g^2))` over that one tensor, and
`OptimizerAdaptor::step` calls it once per parameter
(`burn-optim/src/optim/simple/adaptor.rs:199-200`). GPT-2, PaLM and nanoGPT clip
the *global* norm: one coefficient over every gradient at once, which shortens the
update without rotating it. **burn gives each tensor its own coefficient, so it
rotates the update.** A test now pins this by the property that discriminates the
two: gradients of norm 10 and 0.1 clipped at 1.0 come out 1.0 and 0.1 — ratio
10:1, where a global clip would have preserved 100:1.

AdamW damps this (a gradient rescaled by a constant leaves `m/sqrt(v)` unchanged)
but does not nullify it: the coefficient moves with the per-tensor norm, and
weight decay and epsilon both see the raw scale.

`grad_clip_norm` is **left at 1.0 regardless** — changing it would be a second
change to the reference run, and whether 1.0 binds at all at this scale is
unmeasured. `GradRms` is what will answer that.

Explicitly **not** doing, and why:

- **Diffusion** — DESIGN.md §8 rejects it; the research confirms that was right.
- **Mixture-of-Recursion, gist tokens, learned context compression** — all
  interesting, all *depth/compute* allocation mechanisms. They optimize the
  resource quark has already proven it is not short of. §3 says quark is short of
  **stored capacity**, and MoR does not add any. Revisit after `quark_22m`.
- **QAT** — optimizes checkpoint size. The checkpoint is 11 MB. This is
  optimizing the abundant resource, again.

---

## 7. Niche (bonus): where a model this size can be useful

The honest frame comes from §4.3: at 22–30M, the literature says **grammatical
competence is achievable** (BLiMP 81.2) while world knowledge is not. So the niche
must be tasks that need **structure, not knowledge**:

- **Speculative-decoding draft model.** This is *literally* the issue's own
  Mixture-of-Recursion intuition — "think hard on hard tokens, emit easy ones in
  one pass" — but across two models instead of inside one, where it already works
  in production. It is the only framing under which "replace a bigger model with a
  smaller one" is *honest*: the small model handles easy tokens, the big one
  verifies, and correctness is preserved exactly. A Rust/burn draft model with low
  per-token overhead is a genuine engineering edge, and the metric (acceptance
  rate) rewards exactly what quark optimizes (PPL on the target's distribution).
- **Structured/constrained decoding and single-shot tool calls** — needle's bet,
  unbenchmarked, so the field is wide open for the first model to publish a number.
- **Grammatical error correction / reranking** — BLiMP-shaped tasks.

**Recommendation: do not commit to a product niche yet.** At 2.87M the evidence
says quark is below the floor where any of these work. Whether 22M clears it is
precisely what the free experiment in §6.1 measures. Picking the niche first would
be choosing a destination before knowing if the vehicle moves.

The **paper**, however, is already in reach and does not depend on that outcome.

---

## 8. The paper

quark is uniquely positioned for one contribution, and it is not "a language
model in Rust" — that is engineering, not a claim.

It is: **what does weight sharing actually buy at the 3–30M scale, when you spend
the savings?**

- All three points now exist: run1, run2, and `quark_22m` (issue #6, §10).
- The comparison run1-vs-run2 makes — *re-widening a looped model to restore
  parameter parity* — appears **untested in the literature**.
- The `quark_22m`-vs-run1 comparison is iso-FLOP and **agrees** with Saunshi et
  al. (arXiv:2502.17416, ICLR 2025) in direction, at a scale they did not test.
  The iso-*parameter* claim they actually make remains untested here.
- The identity in §3 is a clean, checkable, general result that the field
  routinely gets wrong: *sharing optimizes the resource that was never scarce.*

That is a real paper, it is honest, and the experiments are hours, not TPU-months.

---

## 9. What would falsify this

- **`quark_22m` fails to beat `quark_3m`.** The function-class argument says it
  cannot lose except through optimization or overfitting effects — so if it does
  lose, the argument in §3 is incomplete and that is the most interesting outcome
  available. Report it first.
- **Seed variance ≥ 0.05 nats.** Then run1-vs-run2 says nothing, and §2's
  "narrower result" evaporates. Measure before building on it.
- **`quark_22m` blows past 16 GB.** The VRAM claim needs no model — same graph as
  run1's MEASURED 11 GB, +0.30 GB — but a measurement beats an argument.
- **`GradRms` never approaches 1e-8.** Then AdamW `epsilon = 1e-15` — the only
  change this PR makes to the reference run — is cargo-culted from a paper about
  larger models and should be reverted to 1e-8. This is the cheapest falsification
  on the list: it needs no extra run, only a look at `train/epoch-N/GradRms.log`
  from the next one.
