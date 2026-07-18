#!/usr/bin/env python3
"""Collect raw run outputs into experiments/gpu/results.json.

Inputs are the files run_experiments.sh drops in the results dir:

  <name>.eval.txt   -- stdout of `quark eval` (word perplexity / BLiMP lines)
  <name>.secs       -- wall-clock seconds of the training run (a bare number),
                       written only once that run has finished, and summed over
                       every leg of it if it was interrupted and resumed. The
                       driver keeps the running total in <name>.ms, which is its
                       own bookkeeping and not read here.
  backends.json     -- {backend: {"seconds": s, "tokens": t}} from the benchmark

Output is results.json, the file experiments/report.py overlays. This script
parses; it does not run models and it does not invent numbers. A metric absent
from the eval text stays null rather than being guessed.
"""

from __future__ import annotations

import argparse
import json
import os
import re


def _num(pattern: str, text: str):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def parse_eval(path: str) -> dict:
    """Pull the comparable metrics out of a `quark eval` report.

    Formats are fixed by src/eval/corpus.rs and src/eval/blimp.rs:
      "word perplexity      74.965   <- the comparable number"
      "bits per byte         1.2730 ..."
      "BLiMP accuracy  61.76%   (chance is 50.00%)"
    """
    with open(path) as fh:
        text = fh.read()
    return {
        "word_ppl": _num(r"word perplexity\s+([\d.]+)", text),
        "bits_per_byte": _num(r"bits per byte\s+([\d.]+)", text),
        "token_ppl": _num(r"token perplexity\s+([\d.]+)", text),
        "blimp": _num(r"BLiMP accuracy\s+([\d.]+)%", text),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--manifest", required=True, help="configs/manifest.json from gen_configs.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.manifest) as fh:
        manifest = json.load(fh)["experiments"]

    experiments = []
    for exp in manifest:
        name = exp["name"]
        rec = {"name": name, "epochs": exp.get("epochs"), "rationale": exp.get("rationale")}
        eval_path = os.path.join(args.results_dir, f"{name}.eval.txt")
        secs_path = os.path.join(args.results_dir, f"{name}.secs")
        if os.path.exists(eval_path):
            rec.update(parse_eval(eval_path))
        if os.path.exists(secs_path):
            with open(secs_path) as fh:
                try:
                    rec["train_seconds"] = float(fh.read().strip())
                except ValueError:
                    rec["train_seconds"] = None
        experiments.append(rec)

    out = {"experiments": experiments}

    bench_path = os.path.join(args.results_dir, "backends.json")
    if os.path.exists(bench_path):
        with open(bench_path) as fh:
            bench = json.load(fh)
        backend_benchmark = {}
        for backend, v in bench.items():
            secs, toks = v.get("seconds"), v.get("tokens")
            if secs and toks and secs > 0:
                backend_benchmark[backend] = {
                    "seconds": secs,
                    "tokens": toks,
                    "tokens_per_sec": toks / secs,
                }
        if backend_benchmark:
            out["backend_benchmark"] = backend_benchmark

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}: {len(experiments)} experiments, "
          f"{len(out.get('backend_benchmark', {}))} backends benchmarked")


if __name__ == "__main__":
    main()
