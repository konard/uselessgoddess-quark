#!/usr/bin/env python3
"""Measure GPT-2's baselines ourselves, under quark's exact protocol.

Produces the two numbers quark is compared against: word-level perplexity
(``--text``) and BLiMP accuracy (``--blimp``).

Why this exists
---------------

Neither baseline can be quoted from the literature.

GPT-2's published WikiText-103 perplexity is 37.50, and we cannot compare quark
to it directly for a reason OpenAI states themselves: the number is computed
after an "invertible de-tokenizer" that undoes WikiText's ``<unk>``, ``@-@`` and
space-before-punctuation artifacts, and that de-tokenizer was **never released**.
OpenAI put it at 2.5-5 perplexity. So 37.50 is not reproducible, and a number
computed without an equivalent de-tokenizer is not comparable to it -- in either
direction, which is the part that makes citing it dishonest rather than merely
imprecise.

For BLiMP there is no GPT-2-small number to quote at all: the BLiMP paper's §6.3
(~84%) contradicts its own Table 3 (GPT-2-*large*, 774M: 80.1), unreconciled, and
BabyLM's 74.88 is BLiMP-filtered. See ``docs/DESIGN.md`` §3.2.

The fix in both cases is to stop citing and start measuring: run the released
GPT-2 checkpoint over the same data, under the same protocol, and report *that*
as the baseline next to the published figure. Only a self-measured baseline is
controlled.

Why Python, and what that costs
-------------------------------

``docs/DESIGN.md`` §3.1 originally promised the baseline would run "through the
same harness code path". It does not, and cannot cheaply: quark is Rust on burn,
and GPT-2's weights are a 124M-parameter HuggingFace checkpoint with a different
architecture and a different tokenizer. Porting it to burn would be a large
amount of unverifiable weight-mapping code whose only output is one number.

So this is a **protocol match, not a code-path match**, and the gap is real:
these are two programs that could disagree. What closes it is
``protocol_fixture.json``, which both implementations assert against --
``cargo test the_frozen_protocol`` on the Rust side, ``--self-test`` here. It
pins the five things that could silently differ:

1. the document stream -- the article split, the per-document denominators, and
   the EOS separator after each document;
2. the window layout and striding -- which tokens get scored, and once each;
3. the denominators -- words and bytes, counted on the source text;
4. the final formulas;
5. BLiMP's decision rule (ties are wrong) and its pair-weighted aggregation.

Everything else *should* differ: that is the model, and the model is the thing
being measured. ``--self-test`` runs automatically before any measurement; if it
fails, these numbers are not comparable to quark's and must not be reported as if
they were.

Item 1 is there because of a bug, and it is worth knowing which: this script used
to tokenize the whole file as one stream while quark split it into articles and
summed the denominators per document. Both sides counted words and bytes with
identical functions, and every fixture case passed -- the fixture pinned how to
count, which is not the same question as what to count. Splitting trims each
document, so the two agreed on words and disagreed on bytes on every real corpus.
Hence ``--split-articles`` below: pass it iff ``quark prepare`` was given it.

Usage
-----

    pip install torch transformers
    python experiments/gpt2_baseline.py --text wiki.test.tokens --split-articles
    python experiments/gpt2_baseline.py --blimp data/blimp/
    python experiments/gpt2_baseline.py --self-test    # no model needed

``--shard`` additionally cross-checks the denominators against the sidecar quark
wrote for the same text, which is the tightest available proof that both are
dividing by the same number -- and the check the fixture cannot perform, since
the fixture does not know what ``quark prepare`` did to your file. See also
``experiments/check_shard_denominators.sh``, which proves the same thing end to
end on a corpus built by the real binary, and runs in CI.
"""

import argparse
import json
import math
import pathlib
import sys

FIXTURE = pathlib.Path(__file__).parent / "protocol_fixture.json"


# ---------------------------------------------------------------------------
# The protocol. Mirrors src/eval/corpus.rs and src/data/{dataset,shard}.rs.
# Every function here has a Rust counterpart pinned by the same fixture.
# ---------------------------------------------------------------------------


def count_words(text: str) -> int:
    """Rust: `str::split_whitespace().count()` (src/data/shard.rs).

    Deliberately the crudest possible definition, because it is the one the
    published WikiText-103 counts use. `text.split()` with no argument splits on
    runs of whitespace and drops empties, which is the same thing.
    """
    return len(text.split())


def count_bytes(text: str) -> int:
    """Rust: `str::len()`, which is already a UTF-8 byte count."""
    return len(text.encode("utf-8"))


def _lines_inclusive(text: str):
    """Rust: `str::split_inclusive('\\n')`.

    Hand-rolled rather than `str.splitlines(keepends=True)`, which also splits on
    `\\v`, `\\f`, `\\x1c` and `\\u2028` -- none of which Rust's `split_inclusive('\\n')`
    treats as a break. A WikiText article containing any of them would be split
    into different documents by the two implementations, and the denominators
    would silently diverge.
    """
    start = 0
    for i, ch in enumerate(text):
        if ch == "\n":
            yield text[start : i + 1]
            start = i + 1
    if start < len(text):
        yield text[start:]


def split_wikitext_articles(text: str):
    """Rust: `split_wikitext_articles` (src/data/mod.rs).

    WikiText marks an article with a ` = Title = ` line and a section with
    ` = = Section = = `, so the article rule is "one leading `= `, and not two".
    Each document is trimmed and empty ones are dropped, which is why the
    denominators must be summed per document rather than counted on the whole
    file -- see `corpus_denominators`.
    """
    articles = []
    start = cursor = 0
    for line in _lines_inclusive(text):
        t = line.strip()
        is_article_heading = t.startswith("= ") and t.endswith(" =") and not t.startswith("= = ")
        if is_article_heading and cursor > start:
            articles.append(text[start:cursor].strip())
            start = cursor
        cursor += len(line)
    if cursor > start:
        articles.append(text[start:cursor].strip())
    return [a for a in articles if a]


def documents(text: str, split_articles: bool):
    """Rust: the `docs` binding in `prepare_shard` (src/data/mod.rs)."""
    return split_wikitext_articles(text) if split_articles else [text]


def corpus_denominators(docs):
    """Rust: `ShardWriter::push_document`'s tallies (src/data/shard.rs).

    Summed over documents, *not* counted on the whole file. Splitting trims each
    document, so the two agree on words but not on bytes: the whitespace between
    articles belongs to no document and is not part of any denominator. Counting
    bytes on the whole file instead would divide quark and GPT-2 by different
    numbers and call the difference a result.
    """
    return sum(count_words(d) for d in docs), sum(count_bytes(d) for d in docs)


def build_stream(doc_tokens, eos_id: int):
    """Rust: `ShardWriter::push_document`'s writes (src/data/shard.rs).

    Each document's tokens, then one EOS separator. The separator is ours, not
    the corpus's: the model must predict it, so it lands in the numerator, but it
    counts toward neither denominator. That costs us a little perplexity for
    tokens the text never contained -- which is the conservative direction, and
    the direction both models are charged in.
    """
    stream = []
    for tokens in doc_tokens:
        stream.extend(tokens)
        stream.append(eos_id)
    return stream


def windows(n_tokens: int, seq_len: int, stride: int):
    """Rust: `TokenDataset` (src/data/dataset.rs).

    Yields `(start, score_from)`. Window `start` reads `tokens[start:start+seq_len+1]`
    and its position `t` predicts `tokens[start+t+1]`. Positions below
    `score_from` were already scored by the previous window, so each token is
    scored exactly once.

    The final partial window is dropped rather than padded: padding would put
    tokens in the loss that the corpus does not contain.
    """
    if stride < 1 or stride > seq_len:
        raise ValueError(f"stride {stride} must be in 1..={seq_len}")
    if n_tokens < seq_len + 1:
        return
    n = (n_tokens - seq_len - 1) // stride + 1
    for i in range(n):
        yield i * stride, (0 if i == 0 else seq_len - stride)


def is_correct(good: float, bad: float) -> bool:
    """Rust: `eval::blimp::is_correct`.

    Ties are wrong. A uniform model ties on every equal-length pair, and scoring
    ties as correct would report it at near 100%.
    """
    return good > bad


def blimp_accuracy(paradigms) -> float:
    """Rust: `BlimpScore::accuracy`. Pair-weighted, not paradigm-weighted."""
    n = sum(p["n_pairs"] for p in paradigms)
    correct = sum(p["n_correct"] for p in paradigms)
    return float("nan") if n == 0 else correct / n


def blimp_by_field(paradigms):
    """Rust: `BlimpScore::by_field`."""
    totals = {}
    for p in paradigms:
        c, n = totals.get(p["field"], (0, 0))
        totals[p["field"]] = (c + p["n_correct"], n + p["n_pairs"])
    return {f: c / n for f, (c, n) in sorted(totals.items()) if n > 0}


def word_ppl(total_nll: float, n_words: int) -> float:
    return math.exp(total_nll / n_words)


def bits_per_byte(total_nll: float, n_bytes: int) -> float:
    return total_nll / (n_bytes * math.log(2))


def token_ppl(total_nll: float, n_scored: int) -> float:
    return math.exp(total_nll / n_scored)


# ---------------------------------------------------------------------------
# The self-test: the whole justification for trusting this script's output.
# ---------------------------------------------------------------------------


def self_test() -> int:
    fixture = json.loads(FIXTURE.read_text())
    failures = []

    for case in fixture["window_layout"]["cases"]:
        got = list(windows(case["n_tokens"], case["seq_len"], case["stride"]))
        want = [(w["start"], w["score_from"]) for w in case["windows"]]
        if got != want:
            failures.append(f"{case['name']}: windows {got} != {want}")
        n_scored = sum(case["seq_len"] - sf for _, sf in got)
        if n_scored != case["n_scored"]:
            failures.append(f"{case['name']}: n_scored {n_scored} != {case['n_scored']}")

    for case in fixture["document_stream"]["cases"]:
        docs = split_wikitext_articles(case["text"])
        if docs != case["documents"]:
            failures.append(f"split {case['name']!r}: {docs} != {case['documents']}")
        got = corpus_denominators(docs)
        want = (case["n_words"], case["n_bytes"])
        if got != want:
            failures.append(f"denominators {case['name']!r}: {got} != {want}")
        # The fixture records what the whole file would have counted precisely so
        # that "summed per document" cannot quietly become "counted on the file"
        # in either implementation. Where the two differ, this asserts the
        # difference is real rather than a stale note.
        whole = (count_words(case["text"]), count_bytes(case["text"]))
        want_whole = (case["whole_file_n_words"], case["whole_file_n_bytes"])
        if whole != want_whole:
            failures.append(f"whole-file counts {case['name']!r}: {whole} != {want_whole}")

    stream = fixture["document_stream"]["stream"]
    got = build_stream(stream["documents"], stream["eos_id"])
    if got != stream["tokens"]:
        failures.append(f"stream layout: {got} != {stream['tokens']}")

    for case in fixture["denominators"]["cases"]:
        text = case["text"]
        if count_words(text) != case["n_words"]:
            failures.append(f"words in {text!r}: {count_words(text)} != {case['n_words']}")
        if count_bytes(text) != case["n_bytes"]:
            failures.append(f"bytes in {text!r}: {count_bytes(text)} != {case['n_bytes']}")

    for case in fixture["blimp"]["decision"]["cases"]:
        got = is_correct(case["good"], case["bad"])
        if got != case["correct"]:
            failures.append(f"blimp decision {case['name']!r}: {got} != {case['correct']}")

    agg = fixture["blimp"]["aggregation"]
    got = blimp_accuracy(agg["paradigms"])
    if abs(got - agg["accuracy"]) > 1e-12:
        failures.append(f"blimp accuracy: {got} != {agg['accuracy']}")
    got_fields = blimp_by_field(agg["paradigms"])
    for field, want in agg["by_field"].items():
        if abs(got_fields.get(field, float("nan")) - want) > 1e-12:
            failures.append(f"blimp field {field}: {got_fields.get(field)} != {want}")

    for case in fixture["formulas"]["cases"]:
        nll = case["total_nll"]
        for name, got in [
            ("word_ppl", word_ppl(nll, case["n_words"])),
            ("bits_per_byte", bits_per_byte(nll, case["n_bytes"])),
            ("token_ppl", token_ppl(nll, case["n_scored_tokens"])),
        ]:
            if abs(got - case[name]) > 1e-9:
                failures.append(f"{name}: {got} != {case[name]}")

    for f in failures:
        print(f"FAIL  {f}", file=sys.stderr)
    if failures:
        print(
            f"\n{len(failures)} protocol mismatches. This script and src/eval/corpus.rs "
            f"are measuring different things; the baseline it produces is NOT comparable "
            f"to quark's perplexity.",
            file=sys.stderr,
        )
        return 1
    print("protocol matches experiments/protocol_fixture.json")
    return 0


# ---------------------------------------------------------------------------
# The measurement.
# ---------------------------------------------------------------------------


def check_denominators(ours, shard_path: pathlib.Path) -> None:
    """Cross-check our counts against the sidecar quark wrote for the same text.

    The numerator of word perplexity is the model's; the denominator is the
    text's, and it must be identical between the two implementations or the
    comparison means nothing. The fixture proves the two counters agree on toy
    strings; this proves they agree on the actual corpus, which additionally
    catches the case the fixture cannot see -- the same counter run over a
    different document set.
    """
    meta = json.loads(shard_path.with_suffix(".json").read_text())
    theirs = (meta["n_words"], meta["n_bytes"])
    if tuple(ours) != theirs:
        raise SystemExit(
            f"denominator mismatch: this script counts {ours[0]} words / {ours[1]} bytes, "
            f"{shard_path.with_suffix('.json')} records {theirs[0]} / {theirs[1]}.\n"
            f"\n"
            f"Most likely the two disagree about --split-articles: pass it here iff `quark "
            f"prepare` was given it, since splitting trims each document and drops the bytes "
            f"between articles. Otherwise the shard was built from different text.\n"
            f"\n"
            f"Either way both models are no longer dividing by the same number, so fix this "
            f"before reporting any perplexity."
        )
    print(f"denominators agree with {shard_path.with_suffix('.json')}: "
          f"{ours[0]} words, {ours[1]} bytes")


def load_blimp(dir_path: pathlib.Path):
    """Rust: `BlimpSuite::load`. Grouped by the UID field, not by filename."""
    by_uid = {}
    files = sorted(p for p in dir_path.iterdir() if p.suffix == ".jsonl")
    if not files:
        raise SystemExit(
            f"no .jsonl files in {dir_path}: download the suite from "
            f"https://github.com/alexwarstadt/blimp (data/ directory)"
        )
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            p = by_uid.setdefault(
                raw["UID"], {"uid": raw["UID"], "field": raw.get("field", ""), "pairs": []}
            )
            p["pairs"].append((raw["sentence_good"], raw["sentence_bad"]))
    return [by_uid[k] for k in sorted(by_uid)]


def measure_blimp(args, model, tok, device) -> int:
    """Score BLiMP with GPT-2, under the rule src/eval/blimp.rs uses.

    Sentences are prefixed with `<|endoftext|>` as a BOS so that the first token
    has a context and gets scored -- BLiMP pairs routinely differ at the very
    first word, and skipping it would be blind to exactly what the paradigm
    tests. Scores are unnormalized sums, per BLiMP's simple_LM_method.
    """
    import torch

    suite = load_blimp(pathlib.Path(args.blimp))
    n_pairs = sum(len(p["pairs"]) for p in suite)
    print(f"scoring BLiMP: {len(suite)} paradigms, {n_pairs} pairs")
    bos = tok.eos_token_id

    def sentence_log_probs(sentences):
        seqs = [tok(s)["input_ids"] for s in sentences]
        max_len = max(len(s) for s in seqs)
        if max_len + 1 > model.config.n_positions:
            raise SystemExit(
                f"a sentence tokenizes to {max_len} tokens, past GPT-2's context of "
                f"{model.config.n_positions}. Truncating would compare two different sentences."
            )
        inp = torch.zeros(len(seqs), max_len, dtype=torch.long)
        tgt = torch.zeros(len(seqs), max_len, dtype=torch.long)
        mask = torch.zeros(len(seqs), max_len)
        for r, seq in enumerate(seqs):
            inp[r, 0] = bos
            for t, tid in enumerate(seq):
                if t + 1 < max_len:
                    inp[r, t + 1] = tid
                tgt[r, t] = tid
                mask[r, t] = 1.0
        with torch.no_grad():
            logits = model(inp.to(device)).logits.float()
            lp = torch.log_softmax(logits, dim=-1)
            picked = lp.gather(2, tgt.to(device).unsqueeze(2)).squeeze(2)
        return (picked * mask.to(device)).sum(dim=1).tolist()

    scored = []
    per_pair = max(args.batch_size // 2, 1)
    for pi, paradigm in enumerate(suite):
        n_correct = 0
        for i in range(0, len(paradigm["pairs"]), per_pair):
            chunk = paradigm["pairs"][i : i + per_pair]
            flat = [s for pair in chunk for s in pair]
            scores = sentence_log_probs(flat)
            n_correct += sum(
                is_correct(scores[j], scores[j + 1]) for j in range(0, len(scores), 2)
            )
        scored.append(
            {
                "uid": paradigm["uid"],
                "field": paradigm["field"],
                "n_pairs": len(paradigm["pairs"]),
                "n_correct": n_correct,
            }
        )
        print(f"\r  {pi + 1}/{len(suite)} paradigms", end="", file=sys.stderr)
    print(file=sys.stderr)

    print(
        f"\n{args.model} on BLiMP\n\n"
        f"BLiMP accuracy {blimp_accuracy(scored) * 100:>7.2f}%   (chance is 50.00%)\n\nby field:"
    )
    for field, acc in blimp_by_field(scored).items():
        print(f"  {field:<28} {acc * 100:>6.2f}%")
    print("\nweakest paradigms:")
    for p in sorted(scored, key=lambda p: p["n_correct"] / p["n_pairs"])[:10]:
        acc = p["n_correct"] / p["n_pairs"] * 100
        print(f"  {p['uid']:<44} {acc:>6.2f}%  ({p['n_correct']}/{p['n_pairs']})")
    print(
        "\nThere is no citable GPT-2-small BLiMP number to check this against "
        "(docs/DESIGN.md §3.2),\nwhich is why it is measured here rather than quoted."
    )
    return 0


def measure_corpus(args, model, tok, device) -> int:
    import torch

    text = pathlib.Path(args.text).read_text(encoding="utf-8")

    # Mirror `quark prepare`: the same document split, so both models are scored
    # over the same document set and divided by the same denominators. Tokenizing
    # the whole file here instead would look equivalent and would not be -- it
    # would hand GPT-2 the inter-article whitespace that quark's documents trim
    # away, in both the numerator and the byte denominator.
    docs = documents(text, args.split_articles)
    n_words, n_bytes = corpus_denominators(docs)
    if args.shard:
        check_denominators((n_words, n_bytes), pathlib.Path(args.shard))

    # GPT-2's own BPE, not quark's. This is the point: each model is measured
    # with the tokenizer it was trained with, and the results are only comparable
    # because the *denominator* is not a token count. The EOS separator is GPT-2's
    # own <|endoftext|> rather than quark's id 0 -- a different id for the same
    # role, which is what "the same protocol" means across two tokenizers.
    ids = build_stream(
        [tok(d, return_tensors=None)["input_ids"] for d in docs], tok.eos_token_id
    )
    n_tokens = len(ids)
    ids_t = torch.tensor(ids, dtype=torch.long)

    seq_len = min(args.seq_len, model.config.n_positions)
    stride = args.stride or max(seq_len // 2, 1)
    layout = list(windows(n_tokens, seq_len, stride))
    if not layout:
        raise SystemExit(f"{n_tokens} tokens is fewer than one {seq_len}-token window")

    total_nll, n_scored = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(layout), args.batch_size):
            chunk = layout[i : i + args.batch_size]
            # Every window is exactly seq_len+1 long by construction, so this
            # batch needs no padding and no mask beyond score_from.
            batch = torch.stack([ids_t[s : s + seq_len + 1] for s, _ in chunk]).to(device)
            logits = model(batch[:, :-1]).logits.float()
            logprobs = torch.log_softmax(logits, dim=-1)
            picked = logprobs.gather(2, batch[:, 1:].unsqueeze(2)).squeeze(2)

            for row, (_, score_from) in enumerate(chunk):
                total_nll += -picked[row, score_from:].sum().item()
                n_scored += seq_len - score_from

            done = min(i + args.batch_size, len(layout))
            print(f"\r  {done}/{len(layout)} windows", end="", file=sys.stderr)
    print(file=sys.stderr)

    coverage = n_scored / (n_tokens - 1)
    print(
        f"\n{args.model} on {args.text}, seq_len={seq_len} stride={stride}, "
        f"{len(docs)} document(s)\n"
        f"\n"
        f"word perplexity      {word_ppl(total_nll, n_words):>12.3f}   <- the comparable number\n"
        f"bits per byte        {bits_per_byte(total_nll, n_bytes):>12.4f}   <- also comparable\n"
        f"token perplexity     {token_ppl(total_nll, n_scored):>12.3f}   (GPT-2 BPE; do not compare)\n"
        f"total NLL (nats)     {total_nll:>12.1f}\n"
        f"scored tokens        {n_scored:>12}   ({coverage * 100:.4f}% of the corpus)\n"
        f"words                {n_words:>12}\n"
        f"bytes                {n_bytes:>12}\n"
        f"\n"
        f"Published zero-shot WikiText-103 for GPT-2 124M is 37.50, computed after an\n"
        f"invertible de-tokenizer that was never released and that OpenAI values at\n"
        f"2.5-5 PPL. The number above is measured without it, so it is expected to be\n"
        f"higher and is NOT the published figure. Compare quark to the number above."
    )
    return 0


def measure(args) -> int:
    """Load the checkpoint once, then run whichever measurements were asked for."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading {args.model} on {device}")
    tok = GPT2TokenizerFast.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device).eval()

    if args.text and measure_corpus(args, model, tok, device) != 0:
        return 1
    if args.blimp and measure_blimp(args, model, tok, device) != 0:
        return 1
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true",
                   help="check this script against protocol_fixture.json and exit; needs no model")
    p.add_argument("--text", help="the raw text split to measure perplexity on, e.g. wikitext-103 test")
    p.add_argument("--blimp", help="BLiMP .jsonl directory; scores the baseline quark's BLiMP is compared to")
    p.add_argument("--split-articles", action="store_true",
                   help="split on ` = Article = ` headings, exactly as `quark prepare "
                        "--split-articles` does; pass it iff the shard was built with it")
    p.add_argument("--shard", help="quark's shard for the same text; cross-checks the denominators")
    p.add_argument("--model", default="gpt2", help="HuggingFace id; `gpt2` is the 124M checkpoint")
    p.add_argument("--seq-len", type=int, default=512,
                   help="match quark's eval seq_len, or the comparison is not controlled")
    p.add_argument("--stride", type=int, default=None, help="defaults to seq_len // 2, as quark's does")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    if args.self_test:
        return self_test()
    if not args.text and not args.blimp:
        p.error("nothing to measure: pass --text, --blimp, or --self-test")
    # The self-test is cheap and the failure it catches is invisible, so it is
    # not opt-in: a mismatched protocol makes the measurement below misleading
    # rather than merely wrong, and misleading is worse.
    if self_test() != 0:
        return 1
    return measure(args)


if __name__ == "__main__":
    sys.exit(main())
