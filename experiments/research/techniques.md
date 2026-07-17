# Sources for docs/NEXT.md

Companion to [`competitors.md`](competitors.md), same rules. Every claim in
`docs/NEXT.md` that is not arithmetic from `experiments/next_steps.py` is here,
with the primary source and — this is the part that does the work — **the gap
between the source's fitted range and quark**.

quark is 22M params, 135M tokens, batch 32,768 tokens, 4,111 optimizer steps,
seq_len 512. Almost nothing in the small-LM literature was measured there. The
column that matters in every table below is the last one.

MEASURED = read from the source. DERIVED = computed in `next_steps.py`.
**UNSUPPORTED** = widely repeated, checked here, no primary source found.

---

## 1. Scale gaps at a glance

| source | claim used | its smallest model | gap to quark |
|---|---|---:|---:|
| Muennighoff arXiv:2305.16264 | repeat-data value, Eq 17 | **7M** (incl. a literal 20M) | **contains quark** |
| Wortsman arXiv:2309.14322 | qk-norm/z-loss neutral at tuned LR | **1.9e7**, seq 512 | **contains quark** |
| Levine arXiv:2006.12467 | L_opt ≈ 14.5 at width 345 | 10⁶ | **contains quark** |
| Adam-mini arXiv:2406.16793 | +0.0096 nats | 39M, d_model **384** | 1.8× |
| GPT-BERT arXiv:2410.24159 | dropout 0.1 at quark's arch | 30M, d=384, V=8192 | 1.4× |
| Lester arXiv:2404.03626 | SentencePiece > learned codec | **25m** | **contains quark** |
| Baby Llama arXiv:2308.02019 | +3.73 BLiMP via distillation | 58M | 2.6× |
| MobileLLM | deep-thin | 125M | 5.7× |
| arXiv:2509.02046 | LR tuning 2×, alternatives ≤1.4× | ~100M | 4.5× |
| AdEMAMix | β3 ablation | 110M | 5× |
| Muon (paper) | ~2× speedup | **399M** | **18×** |
| Distillation Scaling Laws arXiv:2502.08606 | supervised > distill when small | 143M | 6.5× |
| DeepSeek arXiv:2401.02954 | B_opt, η_opt (Eq 1) | C ≈ 1e17 | quark is **below** at 1.7e16 |
| Vocab law arXiv:2407.13623 | N_v = 0.20·C^0.42 | **3B** | **136×** |
| modded-nanogpt | logit softcap, p=0.0001 | 124M, batch 524k | 5.6×, batch **16×** |

The three "contains quark" rows are §2, §8 (qk-norm), and §5 of `NEXT.md`, and
they are the only recommendations made without hedging.

---

## 2. Data repetition — the load-bearing source

**Muennighoff et al., arXiv:2305.16264.** Eq 17, R\*_D = 15.387756. Params 7M–9B;
data budgets D_C ∈ {100M, 400M, 1.5B}, bracketing quark's 135M.

- §5 verbatim: "best loss at around **20-60× more parameters and epochs**...
  **one-epoch models significantly under-utilize their training data**."
- **App S/Q — the catch:** every run behind the fit used **dropout 0.1 AND wd
  0.1**. quark has no dropout. The fit is for a *regularized* model repeating
  data. DERIVED numbers in `next_steps.py` inherit that assumption.

**LTG-BERT Table 3** (MEASURED), BLiMP vs epochs: **83.2 / 83.5 / 83.4 / 83.5**
at ~250/500/1000/2000. Flat over 8× compute; no overfitting.

**2025.babylm-main.12 §2**: "most other participants reported training for roughly
20 epochs." **Wilcox et al. 2025** capped at 20 with "only a 2-3 point drop".

**GPT-BERT (arXiv:2410.24159)**: dropout 0.1 at 12 layers / hidden 384 / 6 heads /
vocab 8192 — quark's architecture exactly. Batch ramps 1M → 4M tokens; removing
the ramp costs **−1.1 BLiMP**. Appendix E: "the results on BLiMP greatly depend on
temperature scaling... we report the accuracies achieved with the optimal
temperature for every model" — so its 86.1/81.2 are optimal-temperature numbers,
not directly comparable to quark's.

---

## 3. Tokenizer

**Lester et al., arXiv:2404.03626** — ran the proposal at 25m. MEASURED:

| method | bits/byte | FLOPs/byte |
|---|---:|---:|
| SentencePiece | 1.12 | 11.69M |
| EqualInfoAC | 1.25 | 15.42M |

§4: "**Our SentencePiece baseline outperforms all other methods.**" Gap widens as
scale falls: +0.070 bits/byte at 2b → **+0.130 at 25m**.

Other cited work, and what it actually says:

- **BLT**: needs a separate **100M** entropy patcher; "BPE models perform better
  with small training budgets".
- **MEGABYTE**: compute-controlled table has **no subword baseline**.
- **SpaceByte**: loses on PG-19.
- **H-Net** (760M/1.3B): "start[s] off worse... but scale[s] better" — crossover
  after 30B–200B bytes. WikiText-103 ≈ 0.5B.
- **ICAE / 500xCompressor / Gist / xRAG / AutoCompressor**: context compressors,
  **not tokenizer replacements**.
- **FSQ**: never tested on text.

**Measurement, the fatal objection.** LCM §2.4.1: "cannot produce the probability
explicitly." CALM arXiv:2510.27688 §3.1: "Standard evaluation metrics like
Perplexity... **can no longer be computed**" — with a >99.9%-accurate codec.

**UNSUPPORTED — "ugly BPE fragments hurt quality":** SuperBPE at 200k vocab is
**worse** on bits/byte (0.7465 vs 0.7482). Schmidt et al. arXiv:2402.18376 tested
fewer-tokens⇒better and found it "**not to be the case**".

**Vocab.** arXiv:2407.13623 fits **N_v = V·d, not V**; smallest IsoFLOP row 3B.
Three routes → V_opt ≈ 3.6K–9K at 22M; §5/Fig 7: data-constrained optimum
*shrinks*. Partial (agent hit session limit, unverified): at 33.2M the measured
optimum is pinned at the grid floor (4096) in every FLOPs cut.

**Counter-datapoints, reported because they disagree:**
- arXiv:2311.01955 Table 5 (RoBERTa, **MASKED**, 10M): 8k=72.3, 16k=74.8,
  32k=76.9, **40k=77.8**, 64k=76.7 → 8k→40k = **+5.5 BLiMP**. Transfer to causal
  **unverified**.
- 2025.babylm-main.16 Table 4 (LSTM 39.2M, **CAUSAL**, 8k vocab): BPE 0.640 /
  Unigram 0.630 / SuperBPE 0.661 → "**tokenizer family matters more than
  vocabulary size**".

---

## 4. Architecture

**ReLU² vs SwiGLU.** DERIVED: equal-param d_ff′ = 1728, and matmul FLOPs are
**exactly identical** (2,654,208/token/layer). Real difference: FFN activations
2304 → 1728 per token per layer (**25% less**).

- Only numeric equal-param head-to-head: **arXiv:2402.03804 Table 2, 1B — a TIE**
  (SwiGLU 50.53 vs ReLU² 50.48, ReLU² given 1.59% more FFN params). Note
  `arxiv.org/html/2402.03804` 404s; use ar5iv.
- **UNSUPPORTED:** "95% of SwiGLU's quality, 15% faster" — **fabricated**, no source.
- Primer's ReLU²>SwiGLU claim is **figure-only** (Fig 5); the Appendix A.7 / Fig 26
  isolating table said to exist **does not**. Its Table 4 (~35M, LM1B) gives
  squared-ReLU-on-MDHA = −0.73 PPLX against baseline noise of ±0.46.
- **Shazeer 2002.05202 contains no ReLU² at all.** Its lesson: *all* GLU variants
  beat *all* non-GLU variants — the gain is the gate. (Its GLUE Average is the
  **FIRST** column, not the last.)
- **modded-nanogpt has never compared ReLU² to SwiGLU** (zero grep matches).

**QK-norm / z-loss.** **Wortsman arXiv:2309.14322** tests N=1.9e7, seq 512 —
quark's exact scale. App B verbatim: "the LR vs. loss curves are
**indistinguishable up to some critical learning rate** when using qk-layernorm
(Figure 1), adding z-loss (Figure 3), or changing warm-up." → neutral on loss at a
tuned LR; buys ~25× LR-sensitivity reduction. §3.1.2: **z-loss is redundant given
weight decay** (Fig 3: WD alone ≈ WD+z-loss). qk-norm should be **per-head, no
biases** (Fig E.8). z-loss's origin is the **Mesh-TF codebase, not a paper**.
Regime gap: Wortsman's 1.9e7 saw ~690 tok/param vs quark's 6.5 (**~106×**), and
quark's rank-128 factorized softmax is untested territory.

**Softcap — two sources, opposite conclusions.** Gemma 2's 50.0/30.0 softcap has
**zero ablation**; **Gemma 3 removed it** ("we replace the soft-capping of Gemma 2
with QK-norm"). modded-nanogpt measured **logit** softcap over **80 runs,
p=0.0001**, "improves performance **in the small-scale regime**". Report both.

**Sliding window.** DERIVED: provably a no-op at ≤1024 context. KV cache at full
1024 ctx = **6.29 MB = 0.037% of 16 GB**. Gemma 3 Fig 6: curves indistinguishable
at 1K.

**Depth/width.** DERIVED aspect ratio: quark **32.0**, GPT-2 small 64.0,
MobileLLM-350M final 30.0, Kaplan (48,1600) 33.3.

- **Levine arXiv:2006.12467** — decoder-only AR LMs, 10⁶–6·10⁸ params, vocab 2000,
  R²=0.998, a=5.039±0.030, b=5.55e-2±1.3e-3 → **L_opt ≈ 14.5 at width ≈ 345**;
  N_Transition(12) = 13.0M < 20.6M ⇒ quark is not too deep; slightly deeper is
  indicated. Iso-param swaps: 17×320 (−0.5%), 14×352 (−2.9%).
  - **Honesty note:** the fit's functional form could not be reproduced from
    a/b as reported, so it is **deliberately not in `next_steps.py`** — it is
    cited as a source conclusion, not re-derived. Do not treat L≈14.5 as DERIVED.
  - §5.2.3 identifies a third regime — "a network can be **too deep**" — that
    their own theory explicitly does not predict, and quark lives in it.
  - The raw log₃(d_x) threshold is a red herring.
- **MobileLLM disagrees** (L≈17–20). Its headline 2.7/4.3% is **not** the depth
  effect (isolated: +0.9/+1.1); it reports **zero perplexity**; smallest model 125M.
- **arXiv:2210.00640's "aspect ratio" is not Kaplan's** — do not cite it as such.
- **Kaplan's "factor of 40"** is not reproduced by his own example (21.5×).
- **UNSUPPORTED:** GPT-2's 1/√N residual init has **zero ablation** ("ablat"
  appears 0 times); the official repo uses a flat 0.02. quark's "1/√(2N)" is
  **nanoGPT's interpretation**, not GPT-2's. Depth-μP §10.3 calls 1/√L
  "**brittle**". Lingle (**arXiv:2404.05728** — *not* 2310.17813) shows at
  **L=12**, quark's exact depth, that LR transfers across a 5000× param range.

**Softmax bottleneck.** Yang et al. arXiv:1711.03953. DERIVED: quark's cap is
d_emb = **128**. **ALBERT Table 3, not-shared row: 81.3 / 81.7 / 81.8 / 82.3** at
E=64/128/256/768 — monotone. "E=128 is optimal" holds **only in the all-shared
row**, which `quark_22m` has left.

---

## 5. Optimizers

**arXiv:2509.02046** is the anchor. Fig 1: "**Up to a 2× speedup is achievable by
tuning a single hyperparameter (learning rate)** in the GPT-3 recipe for a 100M
model." Fig 3: "**The highest speedup is capped at 1.4×**" for alternatives. Its
reconciliation, verbatim, cutting **for** AdEMAMix and **against** Muon at quark's
batch:

> Since Mars and AdEMAMix both perform gradient averaging and variance reduction,
> these methods are advantageous in their **noise-dominated small-batch regime**,
> whereas in our larger-batch setting these benefits diminish and matrix-level
> optimizers become more competitive.

**Muon.** DERIVED: **95.15% of quark is Muon-eligible** (Keller's 124M NanoGPT is
~31% embedding) — best coverage of any published Muon result, thanks to d_emb=128.
Against:

- Docstring verbatim: "**We believe it is unlikely to work well for training with
  small batch size.**" His batch: 524,288 tokens. quark's: 32,768 (**16×**).
- DERIVED, by **his own formula** T·m/B: 5·384/32768 = **5.86%** NS overhead vs
  his 0.7% headline (~8×), and optimistic — a 384×384 NS is latency-bound.
- Smallest paper model **399M**. Table 3's own fit **shrinks**: 1.92× @399M →
  1.72× @1.5B.
- Cleanest primary ablation (`records/102924_Optimizers`, SHA 44cce05): Muon
  3.2760 @722,818ms vs Adam 3.3406 @705,525ms → **+0.0646 nats at equal steps,
  +2.45% wallclock**; **SOAP 3.2752 marginally BETTER** at 2.12× wallclock.
- Clean steps-to-target (`records/track_3_optimization`): Muon 3250 vs well-tuned
  **AdamH 4875 = 1.50×**; SpectralDescent (Muon μ=0) needs 8225 — **worse than
  Adam**.
- Muon's share of modded-nanogpt's 34× is **6.6% in log-space**.
- Implement **Eq 4** (not Eq 7) so lr=3e-3 transfers.
- modded-nanogpt is **8×H100, not 8×A100**.

**AdEMAMix.** DERIVED: β3=0.9999 accumulates only **33.8%** of asymptotic mass over
4,111 steps — never engages. **β3=0.999 → 98.4%.** Fig 17 verbatim: "Using
β3=0.999, the gap between AdEMAMix and AdamW **increases as the number of
iterations decreases**." Gap: shortest LM run 32,000 iters @531k tokens/step,
smallest model 110M.

**Sophia — skip.** Sophia-H **deleted from the repo one day after arXiv v1**
(commit 195a786). Reviewer_x5VB read **1.25×** off the authors' own figure. Author:
"6e-4 learning rate fails with some random seeds and we selected the random seed
with which it works." Kaddour arXiv:2307.06440 re-ran **not counting Hessian
steps** — "Surprisingly, we still do not observe any speedup." Semenov
arXiv:2509.01440: "Sophia **diverges in the small-batch setting**."
**UNSUPPORTED:** "sophia" appears **0 times** in all 102 pages of AlgoPerf
(arXiv:2306.07179); its 2024 winner achieved ~28% (**1.28×**).

**Lion — skip at this batch.** §6: "it is likely that Lion performs **no better
than AdamW if the batch size is small (<64)**." quark is at exactly 64 sequences.

**SOAP config trap.** Default `max_precond_dim=10000` → quark's 8192 vocab slips
**under** → **537 MB** of preconditioner.

**Adam-mini arXiv:2406.16793.** Table 8's smallest is **39M with d_model=384** —
quark's width. Table 4: 39M AdamW 40.795 vs Adam-mini 40.407 = **+0.0096 nats**.
It is a *memory* optimizer; ~85MB is irrelevant on 16 GB.

**Cooldown schedules.** WSD / 1-sqrt ≈ **0 gain** vs a tuned cosine
(arXiv:2405.18392 §3.2: "an almost perfect match"). Adopt for checkpoint reuse,
not for loss.

**Value embeddings** (modded-nanogpt): cost **463M params**. Not applicable.
**Untied head** there was bundled 3 ways and the repo **re-tied it in PR 175** ⇒
quark's factorized+tied d_emb=128 has **no primary evidence against it**.

---

## 6. Distillation

**Zero primary evidence for causal-LM pretraining distillation below 30M.** Not
thin — absent. Only two papers qualify (<30M + pretraining-stage + same-size
control) and **both are MLM**:

- **MobileBERT (25.3M)** Table 9: +3.6/+3.3/+2.7/+2.4 — but **pure logit
  distillation alone is +0.3 MNLI-m**; nearly all of it is Feature Map Transfer.
  **Turc et al. arXiv:1908.08962 directly contradicts** that: "We experimented
  with related approaches [intermediate activations], but found only slight gains
  which were dominated by the gains from pre-training and were not complementary."
- **TinyBERT₄ (14.5M)** Table 1: BERT_TINY 70.2 → 77.0 (+6.8) — but includes
  TD+DA. Clean **GD-only ablation: −3.1**, dev-set, 4 tasks, driven mostly by CoLA
  50.8 → 40.8.

Nearest causal datapoints, all above quark: **44M** (Yam & Paek
`2024.conll-babylm.27`, DistilledGPT-44M, **+1.2** BLiMP-Filtered vs same-size
control); **58M** (Baby Llama arXiv:2308.02019 — **58M, not <30M**; **+3.73**
BLiMP vs same-size non-distilled control, computed from Table 1; **BLiMP-only** —
Table 2 has no 58M control); **345M** (BabyLlama-2 arXiv:2409.17312, +1.1–1.3).

**Corrections to the folklore:**
- **Baby Llama did NOT win BabyLM 2023.** **ELC-BERT (24M, zero distillation)** won
  both tracks — the strongest sub-30M BabyLM result in existence is a
  non-distillation result.
- **Trajectory:** 2023 "common and often successful" → 2024 four sentences, no KD
  winner (**GPT-BERT, zero distillation**, won both) → 2025 not discussed, ~zero
  submissions.
- **AntLM (2024.conll-babylm.29) failed to replicate**: "our own replication of the
  BabyLlama model through distillation did not achieve ideal results."
- **Iyer's 28M peer (2024.conll-babylm.17)** is the only sub-30M BabyLM
  distillation student and is **unusable**: every score within ±2pp of chance,
  their own RoBERTa-base baseline scores **49.62 = chance**, no plain-CE control.

**Distillation Scaling Laws arXiv:2502.08606** (starts at 143M): "**smaller models
are more likely to benefit from supervised pretraining**, whereas larger models are
more likely to benefit from distillation"; U-shaped student error; "distillation
can not produce lower model cross-entropies than supervised learning when both
learning processes are given enough data or compute."

**Counterweight:** "a small teacher suffices" appears twice independently (Yam &
Paek at 44M/60M teachers; arXiv:2605.23857 at 0.7B–1.7B). **Law of Capacity Gap
(arXiv:2311.07052) Law 1**: optimal teacher ≈ **2.5× student** ⇒ ~55M for quark,
not 7B.

---

## 7. BLiMP calibration

**Salazar et al. 2020 Table 7**: BERT-base **PLL 84.2** vs GPT-2-345M **true-LL
82.6** — "despite using less than half the data and a third of the capacity."
Scoring method is worth ~+1.6 to +10. The 80–84 BabyLM figures are **not quark's
league** and never were.

Apples-to-apples — causal, WikiText-103, **full unfiltered 67k BLiMP**:
**Transformer-XL = 68.7** (BLiMP TACL 2020 Table 3). That is the honest target.
quark_22m = **61.76**; OPT-125M (causal, 10M words, organizers' pipeline) = 62.6;
5-gram (3.1B words) = 60.5.

Two caveats, **opposite directions**:
1. BabyLM **filters** BLiMP: 13.7% removed, 57,812/67,000 retained; organizers warn
   results "cannot [be] directly compare[d]".
2. The BabyLM 2024 evaluator **counts ties as correct** (2025.babylm-main.16 §4:
   "In **22 of the 67 BLiMP subtasks**, the two sentences in each minimal pair are
   permutations of the same multiset of words... ties are counted as correct, which
   inflates accuracy"). True random floor **0.543, not 0.500**; an order-blind
   Zipf-frequency baseline scores **0.663** (→ 0.498 with those 22 excluded). A
   causal LM never ties ⇒ collects none of it ⇒ **quark's 61.76 is understated**.

**BLiMP §6.3 verbatim:** "**increasing model size (number of parameters) is
unlikely to improve performance**... All models have overall BLiMP accuracy of 0.84
± .01%... **amount of training data has the biggest impact.**" At fixed 10M, causal
models span 62.6 → 70.3 — a **7.7-point spread from architecture/HP, with the
smaller model winning**.

**ELC-BERT hazards:** batch size is in **SEQUENCES not tokens** (32768×128 =
8192×512 = 4,194,304 tokens/step). **BLiMP 75.8 (Table 1, DynaBench) vs 80.5
(Table 2, own pipeline)** — same model, two numbers, **never mix them**. §5.1:
ELC-BERT **loses** to its own LTG-BERT backbone on BLiMP (85.3 vs 85.8), and the
GLUE win "could be caused by random variation" (ASO ε_min = 0.69, **not
significant**). The "Original 8096 / BLiMP 80.00 / suppl 67.00" row in
2025.babylm-main.12 has **unverified provenance**: GLUE 73.7 / MSGS 29.4 match
ELC-BERT Table 1, but BLiMP 80.00/67.00 match **neither** Table 1 (75.8) nor Table
2 (80.5/67.9).

---

## 8. ROCm / Vulkan

`burn-rocm` **0.21.0** is published (`burn-hip` is the dead predecessor, last at
0.16.1, renamed at 0.17.0). Feature `burn/rocm`; a structural exact peer of `cuda`.
Type: `burn_rocm::Rocm<F = f32, I = i32, B = u8>` = `CubeBackend<HipRuntime, F, I,
B>` — note **`B` defaults to `u8`** like Cuda, unlike Wgpu's `u32`. `RocmDevice` =
`cubecl::hip::AmdDevice { index: usize }`. Needs Linux + ROCm 6.4.x–7.1.1.

**Try Vulkan first.** burn's own matmul benchmarks
(burn.dev/blog/sota-multiplatform-matmul): on the RX 7600 the autotune-favored
"Ordered" variant is **~4.6–5.0 TFLOPs on ROCm vs ~11.5–12 on Vulkan**. Vulkan led
on both AMD cards measured.

Hazards, all upstream: **burn#4202** (segfault if `cpu`+`rocm` both enabled),
**cubecl#1365** (**RDNA2 effectively broken**), **cubecl#1147** (RX 7900 XTX + ROCm
7.1 BF16 conflicts), **cubecl#922**, and an `assert_eq!` panic at init on an
unknown `gcnArchName`. `burn-rocm`'s README is **stale** — `CUBECL_ROCM_PATH` and
"ROCm 6.2.2" no longer apply.

**Why CI stays at `cargo check`** (MEASURED on a ROCm-less machine):
`cargo check --no-default-features --features rocm --all-targets` → **exit 0** in
2m57s, because `cubecl-hip-sys`'s build script deliberately falls back to emitting
no link flags when `hipconfig` is absent from PATH. `cargo build` of the binary
gets as far as linking and fails with ~20 undefined `hip*`/`hiprtc*` symbols.
Compiling is the guarantee on offer; linking and running are not. `cuda` is
excluded because it needs a toolkit to build at all — a check job would test the
runner image, not this code.

---

## 9. Corrections logged during this research

Kept because the same mistakes are easy to re-make, and two of them are mine.

- A WebFetch summarizer **fabricated** a "33M → 39K" row in arXiv:2407.13623
  Table 1. The real smallest row is **3B**.
- A summarizer **fabricated a TAKD numeric table** and swapped columns for
  arXiv:2305.12129.
- A summarizer mis-stated that Baby Llama's Table 1 omits the non-distilled
  control. **It does not.**
- The Muon update rule is **Eq 4**, not Eq 7.
- The looped-transformer paper is **arXiv:2502.17416**, not 2410.17976.
- **arXiv:2310.17813 is NOT Lingle's μ-transfer paper** — that is **2404.05728**.
- Shazeer 2002.05202's GLUE Average is the **FIRST** column, not the last.
- `arxiv.org/html/2402.03804` **404s** — use ar5iv.
- modded-nanogpt is **8×H100, not 8×A100**.
- **Mine:** I believed `docs/RESULTS.md` cited a wrong Saunshi arXiv ID. It cited
  **no ID at all**. Added 2502.17416 rather than "fixing" a nonexistent error.
- **Mine:** the optimizer-step count is **4,111** (16,444 / 4), not 4,120. Derived
  in `next_steps.py` rather than transcribed, for exactly this reason.
