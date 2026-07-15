# quark

A 2.87M-parameter language model in Rust and [burn](https://burn.dev) 0.21,
targeting GPT-2 124M's WikiText-103 perplexity on a 16GB GPU.

**Read [`docs/DESIGN.md`](docs/DESIGN.md) before the code.** It contains the
analysis the target rests on, including the part that says one half of the
original goal is not achievable and why.

## What this claims, and what it does not

The issue asks for 3M parameters matching GPT-2 124M. That is really two targets
with opposite verdicts:

| target | verdict |
|---|---|
| **OpenWebText** perplexity | **Not achievable.** The 3M capacity floor sits ~1.1–1.4 nats above GPT-2's measured loss *at infinite data*. No amount of data or distillation closes a gap that exists at infinite data. |
| **WikiText-103** word-level perplexity | **Plausible**, with a published existence proof: a 4.5M-parameter transformer body already beats GPT-2's zero-shot 37.50. |

The edge is **not** parameter efficiency. GPT-2's WikiText-103 number is
zero-shot and out-of-domain — WebText excluded Wikipedia — and quark trains
in-domain. That asymmetry is the whole advantage, and saying so is what makes the
target credible rather than a marketing claim. See DESIGN.md §1–§2.

Nothing here has been trained. The numbers above are analysis and a citation, not
results. Per the issue, CI runs CPU microtests only; the reference run is yours.

## The model

```
vocab 8192 · d_emb 128 · d_model 384 · 6 heads (2 KV) · d_ff 1152
1 unique layer × 12 loops · RoPE · SwiGLU · RMSNorm · pre-norm · tied embeddings

token_embedding    1,048,576
embed_proj/unembed    98,304
layers             1,721,088
final_norm               384
TOTAL              2,868,352      compute-equivalent 20,643,840
```

The parameter count is asserted against the constructed burn module in
`src/model/lm.rs::analytic_budget_matches_the_real_module` — the analysis and the
code cannot silently disagree.

It is a **family**, not one architecture: `n_unique_layers`, `n_loops`,
`layer_schedule`, `d_emb`, GQA head counts and norm placement are all config, so
variants within the family are a JSON edit rather than a rewrite. `quark train
--dry-run` prints any config's budget without touching the GPU.

## Running it

```sh
cargo build --release --features wgpu     # wgpu is the primary backend
```

Get WikiText-103 (`wiki.train.tokens`, `wiki.valid.tokens`, `wiki.test.tokens`)
from [the original
release](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/),
then:

```sh
# 1. Tokenizer -- on the training split only. Training it on valid or test
#    leaks them into the vocabulary.
quark tokenizer wiki.train.tokens --vocab-size 8192

# 2. Shards. --split-articles makes each ` = Article = ` its own document.
quark prepare wiki.train.tokens --out artifacts/train.bin --split-articles
quark prepare wiki.valid.tokens --out artifacts/valid.bin --split-articles
quark prepare wiki.test.tokens  --out artifacts/test.bin  --split-articles

# 3. Train. Defaults are the reference config; --dry-run prints it and exits.
quark train --backend wgpu

# 4. Evaluate. Each part is opt-in: they cost very different amounts.
quark eval --backend wgpu --ppl artifacts/test.bin --generate
quark eval --backend wgpu --blimp path/to/blimp/data
```

If VRAM runs out, lower `--batch-size` and raise `--grad-accumulation` by the
same factor: the optimizer then sees an identical batch and the run stays
comparable.

## Evaluation

Three numbers, answering different questions (DESIGN.md §3):

- **Corpus perplexity** — reported per **word** and per **byte**, never per
  token. Token perplexity depends on the tokenizer, so quark's and GPT-2's are
  not the same quantity and comparing them is meaningless.
- **BLiMP** — 67 paradigms of minimal pairs. Perplexity can be bought with
  frequency statistics; BLiMP cannot, since both sentences in a pair use nearly
  the same words.
- **Generation** — a frozen prompt set, decoded deterministically. The other two
  are invisible to a human reader.

### The baseline is measured, not cited

GPT-2's published 37.50 is computed after an "invertible de-tokenizer" that was
**never released** and that OpenAI values at 2.5–5 PPL, so it is not reproducible
and not comparable to a number computed without one. BLiMP is worse: there is no
citable GPT-2-small number at all — the BLiMP paper's §6.3 (~84%) contradicts its
own Table 3, unreconciled.

So `experiments/gpt2_baseline.py` runs the checkpoint itself, under quark's
protocol:

```sh
pip install torch transformers

# --split-articles must match what `quark prepare` was given, and --shard checks
# that it did: it compares the denominators against the sidecar and refuses to
# report a perplexity if they disagree.
python experiments/gpt2_baseline.py --text wiki.test.tokens \
    --split-articles --shard artifacts/test.bin
python experiments/gpt2_baseline.py --blimp path/to/blimp/data
```

This is a **protocol match, not a code-path match** — two programs, two
languages, and they could disagree. `experiments/protocol_fixture.json` is what
closes the gap: the protocol frozen as data, asserted by both sides (`cargo test
the_frozen_protocol` and `--self-test`, both enforced in CI). It pins what could
silently diverge and then be misread as a difference between the models: the
document stream, window layout, denominators, formulas, and BLiMP's decision and
aggregation rules.

It is not a complete defence, and DESIGN.md §3.1 works through the case that got
past it: both sides counted bytes with identical functions and still divided by
different numbers, because one counted them per document and the other on the
whole file. A fixture pins the questions someone thought to ask.

## Layout

```
src/config.rs      the model family and its parameter arithmetic
src/model/         attention, SwiGLU, RMSNorm, RoPE, KV cache, the LM
src/data/          BPE, shards, strided windows, batching
src/train/         the burn Learner harness
src/eval/          corpus PPL, BLiMP, generation
experiments/       the scaling analysis, the GPT-2 baseline, the protocol fixture
docs/DESIGN.md     why any of this
```

## Tests

```sh
cargo test --all-targets      # 104, all CPU
```

Microtests only, per the issue. The heaviest trains a 2-layer toy on ~600 tokens
through a real `Learner` and asserts the artifacts exist — that is wiring
verification, not training.
