# Decision: what to do next

This answers the questions in issue #3 that are not "what do the numbers say". For
what the numbers say, see [`RESULTS.md`](RESULTS.md); every claim there is rebuilt
from the run logs by a script in `experiments/`.

The issue asks me to doubt its own ideas ("подвергай сомнениям мои идеи, я могу
ошибаться"). I have, and most of them do not survive. That is the useful part of
this document, so the objections come first and the plan comes last.

---

## 0. The decision in one paragraph

**Do not change the architecture yet. The model is undertrained by roughly 8x, and
nothing measured so far is a property of the architecture — it is a property of a
model that was stopped after one epoch.** The freed resources should go into
training the *existing* dense model to convergence, which costs on the order of
$5–20 and has never been tried. Every architectural question in the issue —
Mixture-of-Recursions, gist tokens, QAT, a 3M rebuild — is a question about the
shape of a curve that has not been run to its end. Answer the cheap question
first; it also happens to be the one most likely to move the metric.

---

## 1. The finding that reframes everything

quark trained for **one epoch: 134,705,152 tokens against 2,868,352 parameters =
47 tokens per parameter.** For comparison, on the same corpus and the same metric:

| model | tokens/param | vs quark |
|---|---:|---:|
| **quark_3m (1 epoch)** | **47** | 1.0x |
| MicroNet 8.3M — WikiText-103, 34.9 test ppl | 389 | 8.3x |
| SmolLM2-135M | 14,815 | 315x |
| Gemma 3 270M | 22,222 | 473x |
| LFM2.5-230M | 82,609 | 1,759x |

There is a tempting counter-argument, and it is wrong, so it is worth killing
explicitly. It goes: *Chinchilla says ~20 tokens/param is optimal; quark is at 47;
therefore it is already past optimal and more training will not help.* This is a
category error. **Chinchilla answers "given a fixed compute budget, how should I
split it between model size and data?" — it assumes N is free to choose.** Here N
is pinned at 3M by fiat, as a product constraint. Once N is fixed, the Chinchilla
ratio has no bearing on when to stop, and the entire industry demonstrates this:
every model in the table above is 15x–4,000x past "optimal" and none of them are
described as overtrained. Training past Chinchilla is what you *do* when you have
decided the parameter count in advance.

Two independent measurements agree that quark is nowhere near its floor:

- **Perplexity.** MicroNet's own scaling curve (ppl ∝ N^−0.315) puts a 3M model at
  ~54 word ppl; a Kaplan-style fit puts it at ~56. quark measured **108.3 (dense)
  and 115.2 (loop12)** — about 2x its own size's floor.
- **BLiMP.** quark's 58.63 is almost exactly the 58.57 mean of the three 1M-word
  MiniBERTas. The model scores as if it had seen 1% of the data it actually saw —
  which is the signature of a model that saw the data once and did not absorb it.

And the calibration ladder is unkind:

| | BLiMP |
|---|---:|
| chance | 50.0 |
| no-signal floor (tie artifact, see `RESULTS.md` §6) | 54.3 |
| **quark_3m_loop12** | **57.05** |
| **quark_3m_dense** | **58.63** |
| a 5-gram model | 60.5 |
| Transformer-XL, same corpus | 69.6 |
| GPT-2 @ 100M words | 74.8 |
| human (individual agreement) | 88.6 |

**quark currently scores below a 5-gram.** That is not an architecture result. You
do not choose between recursion and density from this position; you first find out
what the model does when it is finished.

Every BLiMP baseline in the literature trains 10–20 epochs. MicroNet used 31.
quark used 1. **This is the single highest-leverage fact in the whole issue, and it
costs $5–20 to act on.**

---

## 2. "Dense wins on all fronts except `wh_vs_that_with_gap_long_distance`, so maybe Mixture-of-Recursions"

**No — and the premise is false in three separate ways, all provable from the
issue's own numbers.** `RESULTS.md` §6 is the full argument; the short form:

1. **Dense does not win on all fronts.** It wins 2 of 5 BLiMP fields. The loop
   model wins syntax, syntax/semantics, and syntax_semantics.
2. **Dense's entire +1.58 is one field.** Semantics contributes **+1.817 = 115% of
   the gap**; the other four fields sum to **−0.237, i.e. they favour the loop
   model.** Semantics is 13% of the suite and delivers 115% of the win.
3. **The `wh_vs_that` edge is not about distance.** BLiMP ships the same
   construction with and without the long dependency. loop12's edge *without* the
   long distance is **+1.30**; *with* it, **+1.70**. The edge is there when the
   distance is not — it is a property of the wh/that construction, not of
   retention. **The stated mechanism is not what produced the number.**

And the number itself does not mean what it looks like. Both models are ~**30
standard deviations *below* chance** on that paradigm (2.30% and 0.60%). They do
not lack the preference; they hold the inverted one, ~99% of the time. These pairs
differ by one function word, so the score reads out which word the corpus made
likelier — a model that fits the corpus better follows that prior harder and scores
*lower*. Dense fits better (108.3 vs 115.2) and scores lower. **"Winning" here is a
symptom of being the weaker language model.**

Finally, the paradigm the issue does not mention points the other way:
`only_npi_licensor_present` is loop12 **0/1000** versus dense **>36.2%**. Linking a
licensor to the item it licenses is the closest thing in BLiMP to "keeping context
across a span", and it is exactly where the loop model is maximally worse.

So: MoR is a reasonable architecture, but **it does not follow from this evidence,
and adopting it here would be building on a reading that the data refutes.** Revisit
it only if it survives §5's discriminator on a converged model.

---

## 3. "Where should the freed resources go?"

The loop model costs **11.6x the compute-equivalent params, 4.0x the wall clock,
2.0x the VRAM** — and loses on both perplexity and BLiMP. Reclaiming that is the
right call. But it should not be spent on a new architecture. Ranked by evidence
per dollar:

1. **Train to convergence (10–30 epochs).** ~$5–20. Never tried. Both independent
   signals above say it is where the missing performance is. Muennighoff et al.
   (2305.16264) shows up to 4 epochs of repeated data is ~as good as fresh data,
   and 2401.00448 shows small-model gains continue to 10³–10⁴ tokens/param.
2. **A non-parametric cache.** MicroNet's cache is worth **6.4 ppl (41.3 → 34.9) at
   zero parameter cost** — it is free under a parameter budget, which is exactly
   quark's constraint. Independently flagged by a second review as "the
   highest-leverage idea in this report", because the long-tail vocabulary is where
   a 3M model bleeds.
3. **A vocabulary sweep, scored in bits-per-byte.** The embedding table is
   8192 x 128 = 1,048,576 = **36.6% of the budget**. Trading vocab for depth is a
   real lever, but only measurable in bits-per-byte — word/token perplexity is not
   comparable across tokenizers, which is the mess §4 is about.
4. **Fix the seven confounds in `RESULTS.md` §5** before any architecture claim —
   most importantly residual scaling, which is simply unimplemented, and which
   Jaggi et al. measure as *the worst* setting for looped models (3.542 vs 3.503).
   The loop model has been competing with a handicap it was never given.

Everything in this list is cheaper than an architecture change and strictly
prerequisite to evaluating one.

---

## 4. Doubting the rest of the ideas

**Gist tokens — no.** They compress *context*, and context is not what is scarce
here. The bottleneck is a 3M-parameter store trained for one epoch.

**QAT — no, not yet, and it may be self-defeating.** The point of QAT is to spend
fewer bits per parameter. quark's problem is that its parameters do not yet contain
what they could — the model is 2x above its own size's perplexity floor. Quantizing
an undertrained model optimizes the wrong axis. It becomes interesting *after* §1,
and specifically under the artifact-size framing in §5, where bits-per-parameter is
the score rather than a footnote.

**Knowledge distillation — no. The regime is inverted.** Every favourable small-scale
KD result lives in an over-parameterized, overfitting-bound regime. quark is
data-saturated and capacity-bound — the opposite. KD has never been validated below
8.3M; on MicroNet it bought 0.7 ppl. The teacher costs 10–120x the student's entire
training run, which is 10–120x the budget that §1 says should go to epochs.
Cross-tokenizer KD is broken (DSKD loses to plain SFT 5 times out of 5), and
sequence-level KD is catastrophic here (10.3 → 97.1 ppl). **Distillation has never
won BabyLM.**

**Data curation — a distraction, and probably harmful.** The metric locks the
distribution: phi-1.5-style curation made Pile perplexity 2.1x *worse*. You are
data-rich, not data-poor, and filtering strictly hurts in that regime. WikiText-103
is already human-vetted Good/Featured articles at 0.39% duplication. Curriculum
learning is verified negative (Campos; 13 BabyLM teams tried and failed).

**Tying — keep it.** Press & Wolf Table 6 lands on quark's exact scale, metric and
data family: 4.65M untied → 114.5 test ppl versus **2.65M tied → 112.4**, with train
perplexity improving too. A strict win with no regularization tradeoff. Note the
existing `d_emb=128` choice is deliberate and correct — with tying the logit matrix
has rank ≤ d_emb (the softmax bottleneck, Yang et al. 1711.03953), so cutting it
further has a cost the parameter count does not show.

**CPU-cache inference — do not build it.** This one is the most interesting to
disprove, because the premise is wrong in a specific way:

- **It already happens.** 5.74 MB re-read every ~200µs never leaves a 32MB L3. On
  x86 you *cannot* pin weights in cache — LRU is already doing this for free.
- **The constraint is not binding.** Measured tiny-model inference draws ~5–7 GB/s
  against a ~50 GB/s DRAM budget and ~182 GB/s L3 — you are at **3% of the ceiling
  you would be raising.** The theoretical prize is only 3.6x.
- **The stated hypothesis is false.** fp32 GEMV has arithmetic intensity 0.5, which
  is below machine balance at *every* cache level including L1. It **never becomes
  compute-bound**, so "fits in cache → compute-bound → fast" does not hold.
- **It does not survive contact with context.** The KV cache overtakes the weights
  at ctx≈1,868 and leaves L3 entirely at ctx≈5–10k.
- The one real citation (2606.25353) targets 3–70B models on a 1152MB LLC — a
  machine 500x bigger. Groq is counter-precedent: they *removed* caches.
- Also worth correcting: the model is **11.48 MB in fp32**, not 5.5 MB (that is
  fp16), which straddles the L3 boundary on smaller machines.

The real speed levers, measured, are dull: **compile flags (34x), cutting vocab
(10.3x), vectorizing `expf` and fixing the sampler, and staying single-threaded —
threading is 3.2x *slower* at this size.**

---

## 5. The niche

The issue offers four options. Three of them fail, but for a reason worth stating
first: **the "arXiv paper" goal and the "useful model" goal are in tension, and the
issue is trying to satisfy both with one artifact.** That is why the niche question
feels unresolvable. They point in different directions and you should pick one.

**Replace GPT-2 / SmolLM2-135M on a task and win — no.** Not at 3M, and not
because of effort. MMLU sits at chance until ~350M: Gemma 3 270M sees 6T tokens and
scores 26.2; LFM2.5-230M sees 19T and scores 25.41 on GPQA-D (chance). **The 2026
sub-150M tier is a competent interface with an empty database.** A 3M model buys
syntax, not a world model (TinyStories-33M: HellaSwag 25.7, MMLU 23.8 — both at or
below chance). Beware the numbers that suggest otherwise: SmolLM2-135M's card says
MMLU 31.5, but a standard harness gives **24.2** — "MMLU (cloze)" is a format
artifact, and essentially every published MMLU above ~28 for a sub-200M model is
one.

**A pure architecture battle — no, not from here.** Your differentiator (recursion)
just lost on its own evidence, and you would be entering a field where you score
below a 5-gram. Come back after §1.

**Factorio (`diffusion-factorio`) — the strongest of the four, with a caveat.** It
is the only option where 3M is *sufficient* rather than *deficient*: a blueprint DSL
is a narrow, low-entropy formal language with a naturally tiny vocabulary (which is
also lever #3), evaluation is objective (does the blueprint work?), and there is no
incumbent to beat. FunctionGemma-270M fine-tuned to 96.7% on smart-home tasks,
matching a 120B teacher — **narrow fine-tuning is where sub-300M models are
genuinely real.** The caveat the issue should not skip: `diffusion-factorio` is a
*diffusion* model, and grafting an autoregressive LM onto it is not free. This
serves the *product* goal, not the paper.

**The "yolo constructor/kit" — no as a research goal.** A kit is a library, not a
result; it cannot produce the arXiv paper that motivated the project. But the Rust
+ burn LM tooling is genuinely thin, so this has real value as a *byproduct* of
whatever else you do, not as the target.

**What I would actually pick, for the paper goal: the artifact-size axis, scored in
bits-per-byte.** OpenAI's "Parameter Golf" — a 16,000,000-byte artifact, ≤10 min on
8xH100, scored by bits-per-byte — drew 2,000+ submissions between March 18 and
April 30, 2026. It is closed and the winner hit 1.0565 bpb, but the repo
(`openai/parameter-golf`, 5,157 stars) is **not archived**, and Runpod describes it
as "the first in its Model Craft Challenge series". That is worth monitoring. More
importantly the *axis* is right even without the contest:

- **Bits-per-byte is tokenizer-agnostic**, which sidesteps the entire word-vs-BPE
  mess that §4 of `RESULTS.md` is about — and quark already reports it (dense =
  1.2730 bpb, different corpus).
- The budget is the *artifact*, which includes the code — so **"written purely in
  Rust" becomes part of the score rather than a footnote.** This is the only framing
  where the Rust angle is load-bearing.
- It is honest: quark never has to beat GPT-2 on GPT-2's terms.

**On BabyLM specifically** — since it is the obvious venue and it does not work.
Legal but not competitive: there has never been a parameter limit, but WikiText-103
is ~103.2M tokens against a 100M cap (over budget); a BLiMP-only score is ~1/10 of
the aggregate because missing tasks score 0; and the **≤10-epoch cap forecloses the
MicroNet recipe (31 epochs)** that §1 says you need. The smallest winner ever is
ELC-BERT-small at 24M — 8x quark. Co4 at 8M is the cautionary tale: titled
"outpaces GPT-2 and GPT-BERT" while its BLiMP was 53.55, at the chance floor, and it
is absent from the Findings table. The 2026 deadline is **July 20, 2026 — four days
out**. If you want the venue, the honest door is the **Workshop Paper track**.

---

## 6. The plan

Three experiments, ~$20 total, none of which change the architecture, all of which
are prerequisite to any claim that would:

1. **Train `quark_3m_dense` to convergence** — 10–30 epochs, same data, same LR
   schedule shape re-annealed to the new horizon. Expected: word ppl toward the
   50–60 floor, BLiMP off the 5-gram floor. If this alone lands, the issue's whole
   framing changes and most of §4 becomes moot.
2. **The discriminator for the loop hypothesis** — 12 unique layers versus 1 layer
   x 12 loops, on identical data, *with residual scaling implemented* (`RESULTS.md`
   §5). If BLiMP jumps to 65–70, the loop is real and MoR is back on the table. If
   it stays ~58, it was capacity all along and recursion is settled. This is the
   experiment that answers the issue's actual question, and it has never been run.
   Note `quark_3m_deep` (2x6) is already in the config and has never been run either.
3. **A vocab sweep in bits-per-byte** — V ∈ {2048, 4096, 8192} at fixed d_model,
   scored in bpb so the results are comparable across tokenizers.

Then, and only then, pick the niche — because right now the choice would be made
from an undertrained model's numbers, and §1 says those numbers are not about the
architecture at all.

---

## 7. What I am not confident about

- The 50–60 word-ppl floor for 3M is extrapolated from MicroNet's curve and a
  Kaplan fit, not measured. It could be optimistic.
- The $5–20 figure assumes rented GPU time at current spot rates and 30 epochs at
  the dense model's measured throughput; it is an order-of-magnitude claim.
- MicroNet's numbers appear in the literature in both val and test form (33.6/32.9
  val versus 34.9/41.3 test). I have used **test** throughout and flagged where the
  cache accounts for the difference; do not mix them.
- The web search surfaced several fabricated, future-dated arXiv IDs during this
  research (2605.09751, 2606.19036, 2605.26935, 2605.04952 are confirmed
  fabrications). I have cited only IDs I could verify; 2604.07466 and 2605.21699
  came up but are unverified and are deliberately **not** cited above.
