# quark: design and feasibility analysis

Every number below is computed by `experiments/scaling_budget.py`; its output is
checked in at `experiments/out/scaling_budget.txt`. Every parameter count is
additionally asserted against the constructed burn module in
`src/model/lm.rs::analytic_budget_matches_the_real_module` — the analysis and the
code cannot silently disagree.

---

## 0. Summary

The issue sets one goal — 3.0M parameters matching GPT-2 124M — but that goal is
really two targets with **opposite verdicts**. This is the central finding, and
it was worth establishing before writing a training loop.

| target | verdict | why |
|---|---|---|
| Match GPT-2 124M on **OpenWebText** perplexity | **Not achievable.** | The 3M capacity floor sits ~1.1–1.4 nats above GPT-2's measured loss *at infinite data*. No amount of data or distillation closes a gap that exists at infinite data. |
| Match GPT-2 124M on **WikiText-103** word-level perplexity | **Plausible**, with a published existence proof. | GPT-2's 37.50 is *zero-shot, out-of-domain*. We train in-domain. A 4.5M-parameter transformer body already beats it. |

So quark targets WikiText-103 word-level perplexity, BLiMP, and a fixed
generation eval — and reports OpenWebText loss honestly as a number we expect to
lose on. Claiming otherwise would be the easy thing to do and it would be false.

**The edge is the in-domain/out-of-domain asymmetry, not parameter efficiency.**
Being straight about that is what makes the target credible rather than a
marketing claim.

---

## 1. Why OpenWebText is ruled out

Chinchilla's scaling law (Hoffmann et al. 2022, arXiv:2203.15556, Appendix D.2
Eq. 10) decomposes loss as

```
L(N, D) = E + A/N^alpha + B/D^beta
```

Set `D -> infinity` and the data term vanishes, leaving `E + A/N^alpha`: the
**capacity floor**, the best loss a model of `N` parameters can ever reach on
this distribution, with infinite data and perfect optimization.

```
    params   Hoffmann      PPL  Besiroglu      PPL
  3.00e+06      4.241     69.5      4.511     91.0
  1.24e+08      2.410     11.1      2.555     12.9

  GPT-2 124M on OWT, measured, zero-shot : 3.120 nats
  GPT-2 124M on OWT, measured, finetuned : 2.850 nats
  GAP at 3M vs zero-shot                 : +1.121 nats  => 3.07x worse PPL
```

The mechanism matters more than the constant: **the gap is at infinite data.**
That single fact kills both proposed rescues.

- *"Train on more tokens."* The floor is already the `D -> infinity` limit.
- *"Distill from GPT-2."* A student cannot exceed its own capacity floor no
  matter how good the teacher is. Distillation changes *which* function inside
  the student's hypothesis class you converge to; it does not enlarge the class.

### 1.1 How much this argument actually proves

Honesty about the caveats, because they are serious:

`N = 3e6` is **15× below the smallest model Chinchilla fitted** (44M, Table A9),
i.e. ~1.2 decades outside a fit whose support spans only ~2.6 decades. Worse, it
is self-contradictory under Chinchilla's own definitions: their `N` *includes*
embeddings, and with a 32k vocab a 3M-parameter total budget cannot even hold the
embedding matrix at any sane width. The formula would be describing a model
unlike anything they fitted.

**So 4.24 nats is not a prediction, and this document never uses it as one.**

What survives is the *direction* and the *order of magnitude*:

1. Two independently-fitted laws — Hoffmann, and the Besiroglu et al. refit
   (arXiv:2404.10102) that corrects it — agree the gap is ~1.1–1.4 nats.
2. Closing it needs **5.5×–10.2×** more effective parameters (computed, §3 of the
   script). Modern architecture (SwiGLU, RoPE, RMSNorm, better schedules) is
   empirically worth ~1.2–2×. Every row demands more than architecture can pay.
3. The floor is monotone in `N` for **any** fit with `alpha > 0`. The conclusion
   "a 3M model cannot match a 124M model on the same distribution" does not
   depend on the disputed constants at all.

That is why this document carries both fits: so no contested number is
load-bearing.

---

## 2. Why WikiText-103 is winnable

GPT-2's 37.50 on WikiText-103 (Radford et al. 2019, Table 3) is **zero-shot**,
and WebText **explicitly excluded Wikipedia**. It is an out-of-domain transfer
number. We train on the WikiText-103 train split: in-domain. Same distribution,
103M words of it.

This is not hand-waving. There is a published existence proof:

> **DEQ-Transformer small** (Bai et al. 2019, arXiv:1909.01377, Table 3)
> - non-embedding params: **4.5M**
> - WikiText-103 test: **32.4** word-level PPL, in-domain
> - (compare Transformer-XL small, 139M total: 35.8)

A 4.5M-parameter transformer *body* reaches 32.4, beating GPT-2 124M's 37.50. Its
138M total is almost entirely vocabulary — which is the real lesson:
**word-level WikiText-103 is a vocabulary-storage problem, not a modeling one.**

### 2.1 The connection to our architecture

The existence proof is not from an exotic family. A DEQ is a **single layer
applied repeatedly until it reaches a fixed point** — weight sharing taken to its
limit, infinite depth with one parameter set.

`quark_3m` is one unique layer applied 12 times: a finite unrolling of exactly
that idea. The architecture that provides the existence proof and the
architecture we are building are the same family; DEQ solves for the fixed point
analytically, we unroll a fixed number of steps. That is a principled reason to
expect looping to work here, rather than a generic appeal to "sharing saves
parameters."

### 2.2 What we cannot copy

DEQ's 138M total is unavailable to us: our 3.0M budget is **total, embeddings
included**. A word-level output layer is arithmetically impossible.

```
  WikiText-103 vocab = 267,735 words. Tied embedding matrix alone:
     d_model   embedding params    vs 3.0M budget
          16          4,283,760              1.4x
         128         34,270,080             11.4x
    -> a 3.0M TOTAL budget caps word-level d_model at 11.2. Absurd.
```

Hence a subword vocabulary, and hence §3.

---

## 3. Metric legitimacy

**Per-token perplexity is not comparable across tokenizers.** A smaller vocab
mechanically lowers it: fewer choices per step, more steps per word. Reporting
quark's per-token PPL against GPT-2's would be meaningless — and flattering. The
harness must never do it.

The fix is the protocol **GPT-2 itself used**:

```
PPL_word = exp(total_NLL / n_words)
```

where `total_NLL` sums over all subword tokens but the divisor counts *words*.
Radford et al.: *"We evaluate the same quantity by computing the log-probability
of a dataset according to a WebText LM and dividing by the number of canonical
units."* GPT-2's own 50257-entry BPE vocab is not word-level either — it faces
precisely the same problem and solves it precisely this way. The metric is
tokenizer-independent, so the comparison is legitimate. We also report **bits per
byte**, which is tokenizer-independent by construction.

### 3.1 The trap we cannot fully close

GPT-2's 37.50 uses an "invertible de-tokenizer" that undoes WikiText's `<unk>`,
`@-@` and space-before-punctuation artifacts — worth **2.5–5 PPL by OpenAI's own
account**. It was never released. So 37.50 is not exactly reproducible, and any
number computed without an equivalent de-tokenizer is not comparable to it.

Mitigation, and it is not optional: **we re-evaluate the GPT-2 checkpoint
ourselves** and report that number as the baseline, alongside the published
37.50. Only a self-measured baseline is controlled.

That measurement is `experiments/gpt2_baseline.py`, which runs the released
checkpoint over the same text through the same protocol. **The same protocol, not
the same code** — and the difference is worth being exact about, because an
earlier draft of this section promised the latter. quark is Rust on burn; GPT-2
is a HuggingFace checkpoint with a different architecture and a different
tokenizer. Porting it into burn would mean a large amount of weight-mapping code
whose correctness nothing would check, in service of one number. The honest trade
is to write the protocol twice and make the two halves prove they agree.

That proof is `experiments/protocol_fixture.json`: the protocol frozen as data,
asserted by both implementations — `cargo test the_frozen_protocol` on one side,
`python experiments/gpt2_baseline.py --self-test` on the other, which the script
also runs automatically before it will measure anything. It pins the parts that
could silently diverge and would then be misread as a difference between the
models:

| pinned | why it could drift |
|---|---|
| the document stream | which text each model is scored on: the article split, the per-document denominators, the EOS separators |
| window layout and striding | off-by-one in "which token does position `t` predict"; double-scoring an overlap; padding a partial tail |
| denominators (words, bytes) | the numerator is the model's, but the denominator is a property of the text alone and **must** be identical |
| the formulas | averaging per-batch perplexities instead of exponentiating a total |
| BLiMP decision and aggregation | ties; paradigm-weighting instead of pair-weighting |

The gap that remains is real, and rather than argue that it is small, here is the
one instance of it we have already hit. The first four rows were pinned and both
implementations passed; the first row was not, and did not exist. Both sides
counted words and bytes with identical functions — and then Rust summed them over
the *documents* it had split the file into, while Python counted them on the
*whole file*. Splitting trims each document, so the two agree on words and differ
on bytes forever. Every fixture case passed, because the fixture pinned **how to
count and not what to count**, and those are different questions.

That is the shape of the residual risk: not a subtle numerical disagreement, but
a question nobody thought to ask. Two things narrow it beyond the fixture, and
both are in CI:

- `experiments/check_shard_denominators.sh` runs `quark prepare` on a real file
  through the real binary and diffs the sidecar against what the Python computes
  independently. It also asserts that counting on the whole file *would* have
  given a different answer — a check that cannot distinguish the two would pass
  for free.
- `--shard` performs the same cross-check on your actual corpus at measurement
  time, and refuses to report a perplexity if it fails.

Neither closes the gap in principle. A fixture can only pin questions someone
asked, and this is the honest statement of what that is worth.

### 3.2 BLiMP

There is **no citable GPT-2-small BLiMP number**. The BLiMP paper's §6.3 (~84%)
contradicts its own Table 3 (GPT-2-*large*, 774M: 80.1), unreconciled. BabyLM's
74.88 is BLiMP-*filtered* and not comparable. So we run it ourselves, on both
models — quark in `src/eval/blimp.rs`, GPT-2 via `gpt2_baseline.py --blimp`,
under the protocol match described in §3.1 and pinned by the same fixture.

Protocol: unnormalized full-sentence log-probability, no length normalization —
minimal pairs are length-matched by construction, and normalizing would corrupt
the comparison. This is BLiMP's own `simple_LM_method`.

Two details are pinned in the fixture because they are silent when wrong:

- **Ties count as wrong.** A uniform model ties on every equal-length pair, so
  scoring ties as correct would report a completely broken model at ~100%. The
  rule that flatters a degenerate model is the wrong rule.
- **Accuracy is pair-weighted, not paradigm-weighted.** The full release has 1000
  pairs per paradigm, so the two agree there and the choice looks cosmetic; on any
  subset they diverge sharply. The fixture's own case makes the point — 54.46%
  pair-weighted against 80.00% paradigm-weighted, a 25-point gap created by a
  10-pair paradigm outvoting a 900-pair one.

Both sentences are scored with a `<|endoftext|>` prefix so the first token has a
context and gets scored. BLiMP pairs routinely differ at the very first word
("*Whose* hat should Tom wear" vs "*Who* should Tom wear the hat"), so a scorer
that skipped it would be blind to exactly the token under test.

---

## 4. Architecture

One `ModelConfig` (`src/config.rs`) drives the whole family. The architecture is
"floating" in the issue's sense: variants are *configs*, not forks.

`quark_3m`, the reference:

| field | value | reason |
|---|---|---|
| `vocab_size` | 8192 | BPE trained on the target corpus |
| `d_emb` | 128 | embedding rank — see §4.1 |
| `d_model` | 384 | residual width |
| `n_heads` / `n_kv_heads` | 6 / 2 | GQA: K/V cost 1/3 of full width |
| `d_ff` | 1152 | 3× width, SwiGLU |
| `n_unique_layers` × `n_loops` | 1 × 12 | one layer, twelve applications |
| `tie_embeddings` | true | the output layer is free |

Pre-norm RMSNorm, SwiGLU, RoPE, no biases.

### 4.1 The single most important decision: `d_emb = 128`

`research.txt` proposes `d_emb = 32`. That is the one decision in it that is
not merely suboptimal but *provably* wrong.

With tied, factorized embeddings the logits are produced by projecting the
residual down to `d_emb` and multiplying by the `V × E` table. The logit matrix
therefore **factors through `R^E`**, so `rank(logits) <= E`. Yang et al. 2018
(arXiv:1711.03953, Corollary 1) show a model provably cannot express the true
distribution once its rank exceeds `E+1` — the **softmax bottleneck**. Their
Table 6 measures a `d=400` softmax saturating at rank exactly 400: the bound is
**active, not slack**.

`E = 32` is a rank-32 cap on a real next-token distribution. No amount of depth,
data, or training fixes it; it is a property of the hypothesis class. quark
spends **1,048,576 of its 3.0M — 37% of the entire budget — on embedding rank**,
and that is the most defensible line item in the table.

### 4.2 Why attention is hand-rolled

burn 0.21's `nn::attention::MultiHeadAttention` cannot express what we need:

- no grouped/multi-query support — Q, K, V are all `Linear(d_model, d_model)`, so
  K and V cost full width (~0.2M we would rather spend on the FFN);
- no hook to apply RoPE — the Q/K projections happen inside its `forward`;
- its `MhaCache` requires re-feeding the whole sequence per decode step, and the
  incremental `TensorCache` API is `pub(crate)`;
- it applies dropout to the *scores*, pre-softmax, which is not standard.

`src/model/attention.rs` uses only `Linear`, `RotaryEncoding` and tensor ops.

> **Implementation note, learned the hard way.** burn's `triu_mask` returns
> `false` *inside* the upper triangle and `true` outside — the opposite polarity
> to the obvious reading. Used naively with `mask_fill(-inf)` it masks the *past*,
> leaves the *future* visible, and produces an all-`-inf` final row whose softmax
> is `NaN`. The correct primitive is `tril_mask(shape, pos_offset)`, which is
> `true` exactly where `j > i + pos_offset`. burn's own
> `generate_autoregressive_mask` uses `tril_mask` too. Caught by
> `attention_is_causal`; this is why that test exists.

---

## 5. Parameter budget

```
  quark-3m (V=8192, E=128, H=384)
    token_embedding (V*E)                       1,048,576
    embed_proj (E*H)                               49,152
    unembed_proj (H*E)                             49,152
    layers (1 unique x 1,721,088)               1,721,088
    final_norm                                        384
    TOTAL                                       2,868,352   [OK, +131,648 vs 3.0M]
    compute-equivalent dense params            20,643,840   <- drives FLOPs, NOT the budget
```

The family holds the budget **fixed** so that comparing members isolates the
sharing structure rather than size:

| preset | shape | params | compute-equiv | note |
|---|---|---|---|---|
| `quark_3m` | 1 unique × 12 loops, H=384 | 2,868,352 | 20.6M | reference |
| `quark_3m_deep` | 2 unique × 6 loops, H=288 | 2,865,568 | 10.5M | two parameter sets, narrower |
| `quark_3m_dense` | 6 unique × 1, H=168 | 2,871,880 | 1.78M | **control**: no sharing |

All three within 0.25%. The widths are *forced*: more unique layers must be paid
for with less width. `quark_3m_dense` is the honest control — if it matches
`quark_3m`, looping bought nothing and we should say so.

---

## 6. Weight sharing buys storage, not compute

The most common way to mis-plan a shared-layer run is to conflate two numbers
that sharing drives apart:

- **`param_count()` = 2,868,352** — what the 3.0M budget constrains.
- **`compute_equivalent_params()` = 20,643,840** — what FLOPs and wall-clock
  track. **7.2× larger.**

Looping one layer 12 times costs exactly what 12 distinct layers cost. We pay the
compute of a ~21M model to store a 3M model. The two are separate methods in
`src/config.rs`, deliberately.

Activation memory follows compute, not storage: every loop iteration's
activations must be kept for backprop, because the shared layer sees a different
input each time. **There is nothing to reuse.**

```
   batch   act. GB   attn GB  total GB   fits 16GB?
       8      1.96      2.42      4.38      yes
      16      3.93      4.83      8.76      yes
      32      7.85      9.66     17.52       NO
```

Plan: micro-batch 8 @ seq 1024, gradient accumulation to ~128–256 sequences,
gradient checkpointing per loop iteration.

---

## 7. Training

WikiText-103 train is 103M words, so an in-domain run is a few epochs —
single-digit hours. The OpenWebText pretraining leg is the expensive one:

```
  D=3.0e+09 tokens: 5.41e+17 FLOPs
      4060Ti-16GB @10% MFU:    68.4 h  (  2.8 days)
      4080-16GB   @15% MFU:    21.5 h  (  0.9 days)
```

Affordable but not free — and worth doing only if it demonstrably helps
WikiText-103. That is an experiment, not an assumption.

Optimizer AdamW (**override burn's defaults**: eps 1e-5 and weight_decay 1e-4 are
not what you want), warmup + cosine schedule, gradient clipping at norm 1.0.

### 7.1 Initialization: burn's embedding default is wrong for a tied table

burn initializes `Embedding` from `N(0, 1)`. For an untied model that is merely
unusual; for ours it is a bug, because the table *is* the output matrix.

`final_norm` leaves the residual at RMS 1. `unembed_proj` is Kaiming-uniform with
gain `1/sqrt(3)`, so `k = sqrt(1/fan_in)`, `var(W) = 1/(3*fan_in)`, and its output
has variance `1/3` regardless of width. The logits are that times the table:

```
  var(logit) = d_emb * (1/3) * std_emb^2
```

At `std_emb = 1` and `d_emb = 128` the logit sigma is 6.5, and the initial loss —
about `ln V + sigma^2/2` — is ~26 nats against a uniform model's 9.01. The first
thing training would buy is the undoing of the initialization.

We use GPT-2's `std = 0.02`, giving a logit sigma of 0.13: uniform to within a
hundredth of a nat. The constant is not tuned to one preset — solving
`d_emb * std^2/3 = 0.01` gives 0.031 at `d_emb = 32` and 0.015 at `d_emb = 128`,
so 0.02 sits inside the range the whole family wants. Everything else keeps
burn's Kaiming default, which is right for it.

`model::lm::a_fresh_model_predicts_near_uniformly` pins this, and the harness
test `an_untrained_model_starts_near_uniform_loss` — which is what caught it —
pins the consequence.

Not yet done: GPT-2 also scales residual projections by `1/sqrt(2N)` to stop the
residual stream growing with depth. With 12 applications of one shared layer that
concern is amplified, not reduced. Pre-norm plus `final_norm` makes it survivable,
so this is a tuning question for the first real run rather than a correctness one.

### 7.2 Distillation: does it make sense?

The issue asks for GPT-2 distillation "if it actually makes sense". Split by
target, the answer differs, which is why it is worth asking:

- **For OpenWebText: no.** §1. The gap is at infinite data; a teacher cannot
  raise a student's capacity floor.
- **For WikiText-103: unclear, and testable.** GPT-2 is *out-of-domain* here and
  scores 37.50 while a 4.5M body scores 32.4. Distilling from a teacher that is
  worse than your target on your target domain is not obviously useful. Its
  soft targets may still carry usable signal about general English.

So it is implemented behind a flag and treated as an ablation to be measured, not
a headline method. `KLDivLoss` exists in burn 0.21 for the logit path.

---

## 8. Diffusion language models: considered, rejected

The issue explicitly raises diffusion LMs. They were considered and rejected, for
a reason specific to *this* issue rather than a general judgement.

**The success metric here is perplexity, and diffusion LMs do not produce exact
likelihoods.** Masked/absorbing-state diffusion LMs are trained against a
variational bound (ELBO); the reported "perplexity" is an *upper bound* on the
true value, not the quantity GPT-2's 37.50 measures. Comparing an ELBO-derived
bound against an exact autoregressive likelihood is not a like-for-like
comparison, and the direction of the incomparability is unknown.

BLiMP makes it worse: minimal-pair grammaticality requires ranking two sentences
by log-probability. With bounds rather than exact values, the comparison inherits
the slack of the bound — for pairs whose true log-probabilities differ by less
than that slack, the ranking is not meaningful.

An autoregressive model computes exactly the quantity both evaluations are
defined in terms of. Choosing an architecture that cannot express the metric,
when the metric *is* the goal, would be a strategic error regardless of the
architecture's merits. If the goal were generation quality or controllability,
this analysis would come out differently.

---

## 9. What `research.txt` gets wrong

It is a useful starting point, and its core instinct — spend the budget on shared
depth — matches where we landed. But:

| claim | assessment |
|---|---|
| `E = 32` | **Provably wrong.** Rank-32 softmax bottleneck (§4.1). |
| "will easily hit 75–80% BLiMP" | Unsupported. There is no citable GPT-2-small BLiMP baseline to compare against (§3.2). |
| "guaranteed to beat GPT-2" | Unsupported, and false for OpenWebText (§1). |
| `V = 4096` | Defensible, but 8192 balances sequence length against budget better. |
| shared layer × 12 | **Right**, and better-founded than it argues — cf. DEQ (§2.1). |

---

## 10. Risks

1. **The de-tokenizer gap (§3.1).** Irreducible; mitigated by self-measuring the
   baseline.
2. **Looping may underperform its compute-equivalent depth.** 12 applications of
   one layer is not equivalent to 12 distinct layers in quality. `quark_3m_dense`
   is the control that measures this.
3. **8192 BPE on 103M words** may over- or under-segment. Vocab size is a config
   field; it is an ablation.
4. **wgpu backend maturity** for long training runs is unproven at this scale.
   `cuda` and `ndarray` are available as fallbacks behind features.
5. **The DEQ existence proof is in-domain, word-level, and 4.5M non-embedding.**
   Our body is smaller once the 1.05M embedding is subtracted. It proves the
   target is not absurd; it does not prove we reach it.

---

## 11. Status

| component | state |
|---|---|
| Analysis (`experiments/scaling_budget.py`) | done |
| Model family (`src/config.rs`, `src/model/`) | done, 29 tests |
| Data pipeline + BPE (`src/data/`) | done, 27 tests |
| Training harness (`src/train/`) | done, 19 tests |
| Evaluation (`src/eval/`: word PPL, BLiMP, generation) | done, 26 tests |
| GPT-2 baseline (`experiments/gpt2_baseline.py`) | done — protocol pinned by `protocol_fixture.json` (§3.1) |
| CI | done — fmt, clippy `-D warnings`, tests, wgpu build check, both halves of the eval protocol |

104 tests, all CPU. The end-to-end harness test trains a 2-layer toy on ~600
tokens through a real `Learner` and asserts the artifacts exist; it is wiring
verification, not training. The evaluation tests likewise assert protocol and
wiring, not quality — an untrained model has no quality to assert.

Per the issue: **no local training.** CI runs CPU microtests only; the reference
run is the user's, on 16GB.
