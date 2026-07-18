# Compression — issue #12: the survey, the decision, and the metric

This is the written answer to [issue #12](https://github.com/uselessgoddess/quark/issues/12).
It asks four things, and this document answers them in order:

1. **Which base — `quark_3m` or `quark_22m`?** → the untied one. §2.
2. **Survey 2019–2026, laid out systematically.** → §1, with the design that
   falls out of it in §3.
3. **Which metric?** → free-running reconstruction accuracy, with a measured
   bit-rate beside it. §4.
4. **Without biting into the codebase, and without duplicating it.** → §5 has
   the ledger: **one line in `lib.rs`, one enum on the attention module, three
   functions made `pub(crate)`**. Everything else is new files under
   `src/compress/`.

Same provenance rule as the rest of `docs/`: a number is **MEASURED** (a run
here, or a cited primary source), **DERIVED** (computed from measured inputs), or
**PROJECTED** (extrapolation). Where no run exists, this says so. Per the issue,
**nothing here was trained** — locally or in CI. §6 is about what "logically
verifiable" was made to mean instead.

---

## 0. TL;DR

- The end goal in the issue is a modular LM: `compressor encoder → tiny lm body →
  compressor decoder`. That goal, not reconstruction alone, decides the design.
  A latent that reconstructs perfectly but is unpredictable is useless to the
  body in the middle. This is why two regularizers that no reference
  implementation in the issue has (token dropout, latent dropout) are on by
  default, and why the bottleneck is **FSQ** rather than VQ.
- **Base: untied.** `CompressConfig::compressor_15m()` is `n_unique_layers: 4,
  n_loops: 1` per stack — the `quark_22m` recipe at a smaller depth. MEASURED
  (ANALYSIS.md §0): untying moved WikiText-103 word perplexity **108 → 74.965**
  and BLiMP **58.6 → 61.76** at **+0 FLOPs / +0.30 GB VRAM**. Weight sharing buys
  storage; storage was never the binding constraint (the checkpoint is ~11 MB).
  §2 argues the case is *stronger* for an autoencoder than for an LM.
- **Size: 14,993,920 parameters** (DERIVED, `quark compress --dry-run`), placed
  deliberately inside the 13–17M range the issue reports for the model this is
  being compared against. Like-for-like beats bigger.
- **Metric: free-running greedy exact-match reconstruction accuracy**, at a
  stated bits/token. Teacher-forced accuracy is reported only as an upper bound,
  because it is the number that makes a broken bottleneck look fine. §4.
- **Rate honesty.** The default is `256 → 64` slots: 4× on sequence length,
  **3.024 bits/token** against ~4.6 bits/token of source entropy (DERIVED from
  `quark_22m`'s measured perplexity). The compressor is therefore **lossy by
  construction**, and no amount of training makes it otherwise. Anybody quoting
  "500×" is quoting a token ratio, not a bit rate — §1.6.
- **What will not work at this scale:** 64× compression. PROJECTED from
  [2502.13063](https://arxiv.org/abs/2502.13063), which MEASURED capacity
  scaling with the *decoder*: 1568 tokens/vector at 8B, but only ~96 at 410M.
  At 15M the honest window is **4–8× at >95% reconstruction**; 16× is a stretch
  goal. §7.

---

## 1. The survey: what has been tried, 2019–2026

Sorted by *what the compressed thing is*, because that is the axis that decides
everything downstream — not by year.

### 1.1 Soft prompts: compress into continuous vectors

The dominant line. An encoder turns `N` tokens into `K ≪ N` continuous vectors
that a frozen or fine-tuned decoder reads as a prefix.

| Work | Year | Ratio claimed | What it actually established |
|---|---|---|---|
| **Gist Tokens** ([2304.08467](https://arxiv.org/abs/2304.08467)) | 2023 | up to 26× | Prompts can be compressed into activations by a *masking* trick alone — no new parameters. The idea that the compressor can be the model itself. |
| **AutoCompressors** ([2305.14788](https://arxiv.org/abs/2305.14788)) | 2023 | ~30× | Summary vectors can be *accumulated recursively* over segments, extending effective context. Long-range fan-in works. |
| **ICAE** ([2307.06945](https://arxiv.org/abs/2307.06945)) | 2023 | 4× | The most useful number in the whole literature for us: **BLEU 99.1 at 4×**, and *"unsatisfactory"* by 16× — with a **7B** decoder. Pretraining on autoencoding + language modeling before instruction tuning. |
| **500xCompressor** ([2408.03094](https://arxiv.org/abs/2408.03094)) | 2024 | 6×–480× | Compress into **KV values** rather than embeddings, which carries strictly more per slot. Also: quality degrades smoothly, and the headline ratio is a token ratio (§1.6). |
| **Cramming 1568 Tokens** ([2502.13063](https://arxiv.org/abs/2502.13063)) | 2025 | 1568 tok/vec | The capacity bound. Compression is limited by the **decoder's** size, not the encoder's: 1568 tokens/vector at 8B vs **~96 at 410M**. Also gives the information-theoretic ceiling `L ≤ d·b / log₂|V|`. **This paper is why the target here is 4–8× and not 100×.** |
| **CALM** ([2510.27688](https://arxiv.org/abs/2510.27688)) | 2025 | 4× | The closest published thing to the issue's end goal: an autoencoder whose latent an LM then *models*. Establishes that reconstruction is not enough — the latent must be **robust**, or "a small perturbation in the vector could decode into totally unrelated text". Token dropout 0.15 + latent dropout 0.15; MEASURED 3.99 → 4.70 on its downstream metric when the regularization stack is added. |
| **Compression is Routing** ([2512.16963](https://arxiv.org/abs/2512.16963)) | 2025 | — | Reframes the whole family as learned routing; useful as a lens, no new recipe at our scale. |
| **Optical Context Compression** ([2512.03643](https://arxiv.org/abs/2512.03643)) | 2025 | 10× | Renders text to an image and compresses that. Cited to be complete; irrelevant to a text-only Rust crate. |

**Taken from this line:** the encoder–slot-queries–prefix-decoder shape (ICAE,
Gist, 500xCompressor all converge on it), the 4× starting ratio (ICAE's measured
knee, scaled down), and CALM's two regularizers.

**Not taken:** the frozen-LLM assumption. All of these attach to a 7B–8B
pretrained decoder. We have 15M, and §2 explains what that changes.

### 1.2 Discrete bottlenecks: compress into codes

If the latent is to be *modeled* by a small LM body — the issue's end goal — a
continuous latent forces that body to do regression. A discrete latent lets it do
what a transformer is already good at: predict a symbol from a finite set.

| Work | Year | What it established |
|---|---|---|
| **VQ-VAE** ([1711.00937](https://arxiv.org/abs/1711.00937)) | 2017 | The original. A learned codebook, straight-through gradients, a commitment loss. |
| **SoundStream / RVQ** ([2107.03312](https://arxiv.org/abs/2107.03312)) | 2021 | Residual stacking to reach high rates. Standard in audio codecs. |
| **LFQ** ([2310.05737](https://arxiv.org/abs/2310.05737)) | 2023 | Drop the codebook entirely; binarize each dimension. Vocabulary can grow without collapse. |
| **FSQ** ([2309.15505](https://arxiv.org/abs/2309.15505)) | 2023 | Bound each dimension to `L_i` levels and round. **No codebook parameter, no commitment loss, no EMA, no dead codes.** MEASURED: matches VQ at equal rate, and VQ's codebook utilization collapses above ~2^11 entries while FSQ stays near 100% by construction. |
| **BSQ** ([2406.07548](https://arxiv.org/abs/2406.07548)) | 2024 | Spherical variant; similar story. |
| **Representation Collapsing in VQ** ([2411.16550](https://arxiv.org/abs/2411.16550)) | 2024 | Diagnoses *why* VQ collapses. Confirms the failure is worst exactly where a small model lives. |

**Chosen: FSQ.** Three reasons, in order of weight. (a) There is no codebook to
collapse — a 3–20M model cannot afford to spend capacity keeping one alive.
(b) There are no auxiliary losses, so the training step is one cross-entropy and
nothing to balance. (c) The rate is **exact and countable on paper**:
`K · Σ log₂ L_i` bits per span, no estimation. That last property is what makes
the honesty in §4 possible at all. Default levels `[7,5,5,5,5]` → codebook 4375
(`= 2^12.095`), which is above the ~2^10–2^11 region where FSQ is reported to
lose to VQ.

### 1.3 The information-theoretic line

- **Language Modeling Is Compression** ([2309.10668](https://arxiv.org/abs/2309.10668)),
  **LLMZip** ([2306.04050](https://arxiv.org/abs/2306.04050)) — an LM *is* a
  compressor; its cross-entropy in bits/token is its rate. This is the reason §4
  reports bits/token at all, and the reason `is_lossless_feasible()` exists in
  the config: a channel of 3.02 bits/token cannot losslessly carry a 4.6
  bits/token source, and no architecture argument changes that.

### 1.4 Denoising and robustness

- **DAAE** ([1905.12777](https://arxiv.org/abs/1905.12777)) — corrupting the
  input of a text autoencoder is what makes its latent space *smooth* rather than
  a lookup table with holes between the entries. 2019, and still the cleanest
  statement of why §3's `token_dropout` is not optional.
- **Exposure bias** (Ranzato et al., [1511.06732](https://arxiv.org/abs/1511.06732))
  — a teacher-forced decoder is scored on a task it will never face. This is
  the whole justification for §4's choice of headline metric.

### 1.5 The one that is a trap

- **No Mean Feat** ([2510.20797](https://arxiv.org/abs/2510.20797)) — mean-pooling
  a transformer's states is a *far* weaker summarizer than it looks, and the
  reported numbers of methods that use it are correspondingly inflated. Learned
  slot queries, not pooling.

### 1.6 Why published ratios are mostly not comparable

Nearly every headline ratio in §1.1 is a **token ratio**: `N tokens / K slots`.
It measures a saving in attention cost and KV-cache size. It says nothing about
information, because a slot is a `d`-dimensional float vector — at fp16 and
`d=384` that is 6144 bits, versus 13 bits for a token id. On that accounting, a
"4× compression" into continuous slots **expands** the data by ~118×.

This is not a rhetorical point; it is why the config exposes both numbers under
different names (`token_ratio()` and `rate_bits_per_token()`) and why the
discrete bottleneck matters. With FSQ the second number is exact, so the claim
"4.30× fewer bits" is checkable rather than a genre convention.

---

## 2. The base: `quark_3m` or `quark_22m`?

**Decision: the untied recipe (`quark_22m`'s), at four layers per stack.**

The issue says the hour may have come for `quark_3m` and asks me to check. It
has not, and here is the check.

**The measurement already exists.** ANALYSIS.md §0, MEASURED in PR #7: untying
the loop — `n_unique_layers` 1→12, `n_loops` 12→1, *everything else identical* —
moved WikiText-103 word perplexity **108 → 74.965** and BLiMP **58.6 → 61.76**,
at **+0 FLOPs and +0.30 GB VRAM**. Cross-layer sharing buys exactly one thing:
stored parameters. It costs quality, and it costs it at no compute saving
whatsoever.

**Storage is not the constraint.** `quark_22m`'s checkpoint is ~11 MB. The
project's binding constraint, per ANALYSIS.md §0, is the **token budget under a
16 GB VRAM ceiling** — not parameters and not the architecture.

**For an autoencoder the argument is stronger than for an LM**, for two reasons
the LM case does not have:

1. **Reconstruction is capacity-bound, not compute-bound.** The task is to hold a
   span and put it back. [2502.13063](https://arxiv.org/abs/2502.13063) MEASURED
   this scaling directly and found it tracks decoder *size*. A shared-weight loop
   is precisely the configuration that spends compute without buying capacity —
   the wrong trade for this objective.
2. **The two stacks compute different functions.** Summarizing and expanding are
   not the same operation. Sharing weights *within* a stack is already the wrong
   direction; the encoder and decoder here therefore also get **separate weight
   sets** while sharing one `ModelConfig` (same width, since they exchange
   latents and a mismatch would buy nothing but a projection).

**So why not literally `quark_22m`?** Because the comparison target in the issue
is 13–17M parameters, and two stacks of 12 layers would be ~43M. Four unique
layers per stack, at `quark_3m`'s width, is **14,993,920 parameters** — inside
the stated range, and comparable like-for-like. Depth per stack is the parameter
to raise if the reconstruction ceiling turns out to be capacity and not rate;
§7 says how to tell which.

**What is inherited from `quark_3m`** is only its *width*: vocab 8192, `d_emb`
128, `d_model` 384, 6 heads / 2 KV, `d_ff` 1152, RoPE, SwiGLU, RMSNorm, pre-norm,
factorized and tied embeddings. That part of the design was never in question.

---

## 3. The design that follows

```text
  x_1..x_N  ->  [ embed | K slot queries ]  bidirectional stack  ->  last K
                                                                       |
                                                             to_latent + FSQ
                                                                       |
  x^_1..x^_N <-  causal stack  <-  [ from_latent(z) | bos, x_1..x_{N-1} ]
```

Each choice, and the alternative it rejects:

- **Learned slot queries, not mean pooling** — §1.5.
- **Bidirectional encoder.** The encoder holds the span in full; it is
  summarizing, not predicting. Forbidding it to look right would cost
  information for nothing. This is the *only* change to shared model code: an
  `Attend` enum selecting a mask, no parameters, causal path bit-identical.
- **Prefix-conditioned decoder, not cross-attention.** ICAE, Gist and
  500xCompressor all do this — and it is also what keeps the feature out of the
  rest of the crate. A cross-attending decoder needs a second attention module
  inside `Block`, which is a new field, which means **every existing checkpoint
  stops loading**. A prefix needs nothing new: the decoder is an ordinary causal
  stack whose first `K` positions happen not to be tokens.
- **FSQ bottleneck** — §1.2.
- **Token dropout 0.15** (DAAE, CALM). The single most important regularizer
  here, and the one the reference implementations in the issue omit. A
  teacher-forced autoregressive decoder can reconstruct most of a span from its
  own prefix *without ever consulting the latent*; corrupting the prefix is what
  forces information through the bottleneck rather than around it.
- **Latent dropout 0.15** (CALM). A latent that survives perturbation is one a
  downstream LM body has some chance of predicting *into*. Without it the
  modular-LM end goal is unreachable however good reconstruction gets.
- **No learned positional table.** RoPE already extrapolates. A learned table is
  exactly what caps the reference implementation in the issue at 128 tokens —
  see §8.
- **No padding mask, no second embedding table.** Spans are dense fixed-length
  windows out of a token shard; the encoder, decoder and output head all read one
  table.

---

## 4. The metric

Compression papers report a metric that flatters them. This picks the one that
does not.

**H1 — headline: free-running greedy exact-match reconstruction accuracy.**
Decode from the latent alone, feeding the model its *own* outputs, and count
exactly-matching tokens. Reported at a stated bits/token.

Why not teacher-forced accuracy — the number everyone quotes? Because with
teacher forcing the decoder is handed the true prefix at every step, and a model
that ignores the bottleneck entirely still scores well. It measures the language
model, not the compressor. Free-running is also the only condition that matches
how the thing will actually be used in `encoder → body → decoder`: at inference
there is no true prefix to hand it. Exposure bias, Ranzato et al. 2015.
Teacher-forced accuracy *is* reported — as an **upper bound only**. The gap
between the two is itself the diagnostic: large gap ⇒ the decoder is leaning on
its prefix rather than the latent.

Why exact match rather than BLEU: at 15M, partial credit hides the failure mode
that matters (fluent text that is not the input). BLEU is the right metric when
you are already near-perfect; ICAE's 99.1 is a BLEU, and at that level the two
agree anyway.

**H2 — the rate: bits/token, measured not claimed.** `K · Σ log₂ L_i / N`, exact
because the bottleneck is discrete. Default `64 · 12.0951 / 256 = 3.024`
bits/token, versus 13.0 bits for a raw token id (4.30×). Any accuracy claim
without this number beside it is unfalsifiable — §1.6.

**H3 — the curve, not a point: `CR@99`.** The largest token ratio still holding
≥99% free-running accuracy. A single (ratio, accuracy) pair is cherry-pickable;
the ratio at a fixed quality bar is not. This is the number to compare against
the friend's model.

**H4 — downstream retention.** Perplexity of the existing `quark` LM on text
round-tripped through the compressor, against its perplexity on the original.
This is the only metric that speaks to "maximum preservation of meaning" as
opposed to preservation of *tokens*, and it is the one that predicts whether the
modular-LM goal is reachable.

**Not used:** reconstruction cross-entropy as a headline. It is the training
loss; reporting it as a result is grading your own homework.

---

## 5. The cost to the existing codebase

The issue's hardest constraint: must not bite deeply into the codebase (the
feature is optional), must not duplicate it either. The ledger:

**Changes to shared code — all of it:**

| File | Change | Effect on existing runs |
|---|---|---|
| `src/lib.rs` | `pub mod compress;` | none |
| `src/model/attention.rs` | `Attend` enum selecting causal vs bidirectional mask | none — no parameters, causal path bit-identical, checkpoints unaffected |
| `src/train/mod.rs` | `run` split into `open_datasets` + a model-generic `launch`; `QuarkLm::grad_rms` → free generic `grad_rms`; `refuse_to_merge_runs` made `pub(crate)` | none — pure extraction, no behaviour change |
| `src/bin/quark.rs` | one `compress` subcommand | none |

**Duplication avoided.** A compressor run needs the same shards, windows,
batcher, optimizer, schedule, checkpoint pruning and best-epoch recovery as an LM
run. Only three things differ: the **objective** (reconstruct, not predict), the
**target** (the span is its own target), and the **config**. So `src/compress/train.rs`
is ~470 lines of which the actual training loop is *zero*: it calls
`crate::train::launch` with a different closure.

**Zero new data code.** `TokenDataset::train(shard, span_len)` + `TokenBatcher` +
`TokenBatch` are reused as-is — a span *is* `batch.input`, and the reconstruction
target is that same tensor. `batch.target` simply goes unused, which is the
concrete form of "the compressor needs no new dataset".

**Configs compose rather than merge.** `CompressTrainConfig { compress, train }`,
with `sync()` as the only sanctioned constructor and `validate()` refusing any
config where the two halves disagree (`train.seq_len ≠ compress.span_len`,
`train.model ≠ compress.model`) rather than silently preferring one. A `z_loss`
the compressor's step cannot apply is **rejected**, not ignored.

---

## 6. "Logically verifiable" — what was done instead of training

The issue forbids training locally or in CI, and asks for code where it is
*obvious* the training would succeed. Concretely, that meant making the
non-obvious parts checkable without a GPU:

- **The parameter budget is analytic.** `CompressConfig::budget()` derives every
  parameter from the config, and a test asserts it equals the count of the
  constructed module. If the two disagree, the config is lying — caught in
  0.5 s, not after an hour of training.
- **The rate is analytic.** Same discipline, `rate_bits_per_token()`.
- **`every_parameter_gets_a_gradient`.** A `ModuleVisitor` walks the built
  compressor after one backward pass and asserts every float parameter appears in
  the `GradientsParams` map the optimizer consumes. A parameter absent there is a
  parameter that never moves — the classic silent failure of a multi-stack model,
  and normally invisible until a loss curve plateaus.
- **`training_and_validation_score_the_same_thing`.** The `TrainStep` and
  `InferenceStep` paths are asserted to compute the same loss on the same batch,
  so a validation metric cannot silently measure a different objective.
- **FSQ round-trip tests.** Quantize → dequantize → index → code, asserting the
  grid is exactly what the arithmetic says.
- **Validation before dispatch.** Every shape error that would otherwise surface
  as a tensor mismatch deep inside the first forward pass is a message from
  `validate()` instead, before the backend spins up. `quark compress --span-len
  4096 --dry-run` reports the RoPE bound; it does not allocate.
- **`--dry-run`** prints the resolved config, the budget table and the rate
  without touching the GPU.

27 tests, all passing on `ndarray`, none of them training anything.

---

## 7. What to expect, honestly

PROJECTED, from [2502.13063](https://arxiv.org/abs/2502.13063)'s measured
capacity-vs-decoder-size scaling (1568 tokens/vector at 8B; ~96 at 410M) and
ICAE's measured knee (BLEU 99.1 at 4×, unsatisfactory by 16× — with a 7B
decoder):

| Token ratio | At 15M | Basis |
|---|---|---|
| **4×** (default) | should reach >95% free-running | ICAE's 4× is comfortable at 7B; scaled down, still the conservative end |
| **8×** | plausible; the interesting experiment | between ICAE's knee and its failure, at 1/500th the decoder |
| **16×** | stretch; expect visible loss | ICAE calls this unsatisfactory *at 7B* |
| **64×+** | will not work | below the capacity bound by any reading of the scaling |

And the rate bound is independent of all of that: at 3.024 bits/token against
~4.6 bits/token of source entropy (DERIVED from `quark_22m`'s MEASURED
perplexity of 74.965), the default configuration **cannot** be lossless. It is
lossy by construction. `is_lossless_feasible()` exists so that a disappointing
number can be attributed to the *rate* rather than blamed on training — raise
`n_slots` or the FSQ levels, not the epoch count.

**How to tell capacity from rate**, when the first run disappoints: raise the
rate alone (more FSQ levels, same slots). If accuracy moves, it was rate-bound;
if not, it is capacity-bound and the answer is depth per stack.

---

## 8. Notes on the reference implementation in the issue

The issue attached three files and asked for weak points, not borrowing. Nothing
was copied. Two structural observations:

1. **The learned positional table caps the context.** It is what fixes that
   implementation at 128 tokens, and it does not extrapolate. Using RoPE here
   means `span_len` is a config field bounded only by `max_seq_len`, and
   `256 → 64` is a starting point rather than a ceiling.
2. **No input corruption.** With a teacher-forced autoregressive decoder and no
   token dropout, the training loss can fall while the bottleneck stays nearly
   unused — the decoder answers from its own prefix. This will look like success
   and fail at inference, which is exactly the gap §4's H1-vs-teacher-forced
   comparison is designed to expose.

The 8192 vocabulary matches ours, so `CR@99` is directly comparable. The training
corpus is not (fineweb vs WikiText-103) — per the issue, training here stays on
**wiki-103**, the same corpus the rest of the project is measured on.

---

## 9. Running it

```bash
# Same shards as the language model — nothing new to prepare.
quark compress --dry-run                      # budget, rate, resolved config; no GPU
quark compress --backend wgpu                 # the 15M reference, 256 -> 64
quark compress --span-len 512 --n-slots 64    # 8x, the interesting experiment
quark compress --preset tiny --backend ndarray  # a toy, for a smoke test
```

`--span-len` moves the run's window too (`sync()` carries it), so the two halves
of the config cannot drift apart from a flag.

---

## 10. Open questions

- **The learning rate is inherited, not swept.** It comes from the language model
  at the same width, because there is no sweep for this objective and inventing a
  number would be worse than reusing a measured one. It is the first thing to
  tune.
- **No run has happened.** Every quality number above is cited or projected, and
  labelled as such. The first real datum will be H1 at the default configuration.
- **The body in the middle is not built.** This lands the two ends of
  `encoder → body → decoder`. Whether a small LM can *model* this latent sequence
  is the next question, and CALM's regularizers are here specifically so that the
  answer is not foreclosed.
