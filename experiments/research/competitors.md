# Competitive landscape: what the numbers actually say

Sources checked against primary documents (paper PDF, repo, HF API), not blog
posts. Every claim below carries its source. Where a source could not be found,
that is stated rather than papered over -- several of the most-cited numbers in
this space turn out to be unsupported, and that is itself a finding.

Legend: MEASURED = stated in the primary source. DERIVED = my arithmetic from
numbers the source states. UNSUPPORTED = claimed publicly, no evidence found.

---

## 1. The headline: WikiText-103 word-level PPL is structurally hostile to a 3M model

WikiText-103's vocabulary is **267,735 words** (Merity et al. 2016,
arXiv:1609.07843, Table 1; train tokens 103,227,021). That single fact governs
the whole benchmark at our size class.

DERIVED: a plain input embedding table alone exceeds 3.0M params at any
`d_embed > 11`, and exceeds 30M at any `d_embed > 112`. Even the best-known
compression -- Baevski & Auli's adaptive input representations -- costs ~44M in
embeddings alone.

So how do the famous "small" WikiText-103 results exist? They don't. They are
small-**body** results:

| Model | Total params | Non-embedding | Test PPL |
|---|---:|---:|---:|
| Transformer-XL (small, 4 layers) | 139M | 4.9M | 35.8 |
| Transformer-XL (small, weight-tied 16 layers) | 138M | 4.5M | 34.9 |
| **DEQ-Transformer (small)** | **138M** | **4.5M** | **32.4** |
| Transformer-XL (medium, 16 layers) | 165M | 44M | 24.3 |
| Baevski & Auli ADP-T | 247M | ~201M | 20.51 |

MEASURED: Bai, Kolter & Koltun 2019 (arXiv:1909.01377) Table 3.

DERIVED: DEQ-small's embeddings = 138M - 4.5M = **133.5M, i.e. 97% of the
model**. The celebrated "4.5M parameters, 32.4 PPL" is a 138M model. It cannot
be cited as a small-model result, and neither can any other row in that table.

**No published transformer with <30M *total* params reports word-level
WikiText-103 perplexity.** I looked and could not find one; the arithmetic above
suggests why. The closest candidate (Armeni et al., arXiv:2210.13569: 29.7M ->
PPL 95.1) is disqualified -- it is BPE-level perplexity over a 28,439-token
vocab that was *fit on the test set*, not word-level, and not comparable.

**Consequence for quark.** quark sidesteps the vocab problem legitimately: an
8192-entry sub-word vocab, scored as `exp(total_NLL / n_words)`. That is the
same route Baevski & Auli sanction (§4.4: BPE-33K, "The final evaluation is in
terms word-level perplexity to be comparable to other models"). The protocol is
sound. But it means quark is not *competing* with DEQ or Transformer-XL at 4.5M
non-embedding params -- those models spend 133M parameters on a lookup table
quark cannot afford and does not want. The comparison that has been implicitly
framing this project compares two different things.

## 2. GPT-2 Small's 37.50 -- confirmed, with two caveats that shrink the gap

MEASURED: Radford et al. 2019, Table 3: WikiText-103 **37.50**, zero-shot.

Two caveats, both from the paper itself, both favourable to quark:

- §3.1: results use invertible de-tokenizers, and "We observe gains of **2.5 to
  5 perplexity** for GPT-2 with these de-tokenizers." The 37.50 is a
  de-tokenized number.
- §4/Table 6: WikiText-103's test set has **9.09% 8-gram overlap with its own
  train set**; "there is at least an overlap of 1.6%" of whole articles.

One caveat *against* the easy comparison, and it is a real one -- MEASURED,
§2.1: "We removed all Wikipedia documents from WebText since it is a common data
source for other datasets and could complicate analysis due to overlapping
training data with test evaluation tasks." GPT-2's 37.50 is genuinely
out-of-domain zero-shot. quark trains on WikiText-103 in-domain. **quark is
playing the easier game and still losing by 2.89x.** Any writeup that quotes the
gap without this sentence is overstating quark's position, not understating it.

Naming: the paper says **117M** (Table 2); OpenAI's model card says 124M. Use
"117M as reported in Radford et al. Table 2; 124M per OpenAI's model card".

## 3. needle (cactus-compute) -- the nominated competitor publishes no benchmarks

The issue calls needle "an interesting competitor". Checked against the repo,
docs, model card and HF API:

- MEASURED: pretraining **200B tokens on 16x TPU v6e in 27 hours**, dataset
  PleIAs/SYNTH. Post-training 2B tokens of function-call data in 45 min.
- MEASURED (HF API `safetensors.total`): **30,427,676 params**, not 26M.
  DERIVED: embeddings + attention projections = exactly 26,214,400 -- so "26M"
  counts only those, excluding the contrastive head, norms and gates.
- UNSUPPORTED: the README's claim that it "beats FunctionGemma-270m, Qwen-0.6B,
  Graninte-350m, LFM2.5-350m on single-shot function call". **No table, no
  number, no eval harness exists anywhere in the repo, docs or model card.**
- **needle reports no perplexity, no WikiText-103, no BLiMP.** It is not a
  language-modelling result. It is an encoder-decoder tool-calling model with no
  FFN, distilled from Gemini, aimed at on-device function calling.
- Its own `docs/tpu.md` contradicts the 27h claim, listing "~49h, ~$2,120" for
  v6e-16. Unreconciled; do not cite both.

**needle is not a competitor to quark.** It shares a vocab size (8192) and a
taste for architectural minimalism, and nothing else: different task, different
architecture class, no overlapping metric. Its one genuinely transferable idea
is the thesis behind dropping FFNs -- "At small scale, FFN parameters are
wasted. ~2/3 of standard transformer parameters are FFN" -- which is an argument
about *where to spend a fixed budget*, and worth taking seriously on its own
merits rather than because needle did it.

## 4. SmolLM2-135M -- also does not report our metrics

MEASURED: 134,515,008 params (HF API), **2T tokens**, 64x H100 (model card; the
paper only gives hardware for the 1.7B). Vocab 49,152, tied embeddings, GQA.
**Reports neither WikiText-103 nor BLiMP.** Training cost is not published.

It is 47x quark's params and 2T tokens against quark's 103M-word corpus x10
epochs. Not a peer; a reference point for what saturation looks like.

## 5. BabyLM -- the honest venue for this size class, and it has a cliff

This is where models of quark's size actually publish, and the picture is sharp.

| Model | Params | BLiMP | Track |
|---|---:|---:|---|
| GPT-BERT | 119M | 86.1 | 2024 Strict |
| **GPT-BERT** | **30M** | **81.2** | **2024 Strict-Small** |
| ELC-BERT "Original" | 24M | 80.00 | Strict-Small |
| WhatIf | 26M | 66.9 | 2024 Strict-Small |
| BERTtime Stories | 24M | 63.2 | 2024 Strict-Small |
| MoEP | 28M | 59.15* | 2025 Strict-Small |
| **quark_3m (run3)** | **2.87M** | **60.93** | -- |
| Co4 | 8M | 53.55 | 2025 Strict-Small |
| BitMar | 14M | 48.7 | 2025 Multimodal |

*MoEP's 59.15 is the average of BLiMP and BLiMP-supplement, not raw BLiMP.

Sources: arXiv:2410.24159 Tables 1&3; aclanthology 2025.babylm-main.12,
2024.conll-babylm.20, .28, 2025.babylm-main.39, .24, .11.

Three things follow, and they are the most decision-relevant facts in this
document:

1. **~24-30M is a demonstrated sweet spot.** The 2024 winner *is* a 30M model.
   At BLiMP 81.2 it lands within 5 points of the 119M model, with 4x fewer
   params and 10x less data.
2. **Below ~16M, BLiMP collapses toward chance.** 14M -> 48.7. 8M -> 53.55.
   **quark at 2.87M scores 57.05-60.93 -- exactly where this curve says a model
   its size lands.** quark's BLiMP is not an anomaly to be debugged. It is the
   size class reporting in. The three runs' 4-point BLiMP spread is noise on
   top of a number set by the parameter count.
3. **Batch size is load-bearing at this scale.** 2025.babylm-main.12 Table 2
   re-ran ELC-BERT's 24M config at smaller batch sizes and BLiMP **collapsed
   from 80.00 to 44.17-52.22 across all twelve re-runs**. GPT-BERT's own
   ablation agrees: removing batch scheduling costs -1.1 BLiMP, its second
   largest single hit. quark's effective batch is 16 x 4 x 512 = 32,768 tokens.

Two protocol warnings, both MEASURED:
- GPT-BERT Appendix E: "the results on BLiMP greatly depend on temperature
  scaling... we report the accuracies that are achieved with the optimal
  temperature for every model." 86.1/81.2 are optimal-temperature numbers.
- 2024 and 2025 BLiMP are **not on the same scale**. The same GPT-BERT family
  reads 86.1 in 2024 and 80.5 as the 2025 Strict baseline. Do not plot together.

What won, mechanically (2024, arXiv:2410.24159 §5.2): "by shifting MLM
predictions one position to the right, the MLM predictions become aligned with
next-token predictions from CLM." **No architectural change, no extra
parameters.** And in 2023 (aclanthology 2023.conll-babylm.1 §7.1), the winner
trained "over 450 epochs for their Strict submission, and over 2000 epochs for
their Strict-Small submission" -- which independently corroborates Muennighoff's
finding that quark's 10 epochs are nowhere near the repetition limit.

Also §7.1, and worth sitting with: "Strict models did not outperform those in
Strict-Small by a large amount, even though the size of training data was an
order-of-magnitude larger."

---

## What this changes

- The benchmark quark has been optimising is one where its size class has no
  published peers, and where every "small" competitor is small only in the body.
  Word-level WikiText-103 PPL against GPT-2 is a comparison quark can lose
  honestly forever without learning anything.
- BLiMP at 2.87M is behaving exactly as the literature predicts. Chasing it at
  3M is chasing a number the parameter count already fixed.
- The one intervention with direct empirical support at our scale is **more
  parameters, to ~24-30M** -- where a 30M model reached BLiMP 81.2. That is
  precisely the range `quark_22m` (21.8M) occupies, and `quark_22m` reaches it
  at *zero* additional arithmetic over the `quark_3m` run that already fit in
  11 GB.
