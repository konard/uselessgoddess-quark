#!/usr/bin/env python3
"""Generate the TrainConfig JSONs for the issue-#8 experiment sweep.

The experiments are NEXT.md §13's recommended order, encoded as config edits on
top of a baseline. The baseline is whatever `quark train --dry-run` prints for
the built-in reference config (`quark_3m`); we never hand-write the AdamW
constants or the schedule shape, we only override the fields an experiment
changes. That keeps every generated config in lock-step with the binary's
defaults -- if a default moves, so does every experiment built on it.

quark_22m is quark_3m with the loop untied: `n_unique_layers` 1 -> 12 and
`n_loops` 12 -> 1, everything else identical (RESULTS.md §3 table). So the 22M
base is a two-field edit, not a second source of truth.

    quark train --dry-run > /tmp/dry.txt          # binary prints budget + JSON
    python3 gen_configs.py --baseline /tmp/dry.txt \
        --data-dir <shards> --out-dir configs --set quick

Writes configs/*.json and configs/manifest.json. The manifest is what
run_experiments.sh iterates; each entry says which backend to use, whether to
evaluate, and a one-line rationale with its NEXT.md citation.

Nothing here runs a model or invents a metric. It only writes config files.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re


def load_baseline(path: str) -> dict:
    """Extract the pretty-printed TrainConfig JSON from `quark train --dry-run`.

    --dry-run prints a budget table, then the JSON, then a `backend:` line
    (src/bin/quark.rs). We slice out the first balanced `{...}` block and parse
    it, so the surrounding human text is ignored.
    """
    with open(path) as fh:
        text = fh.read()
    start = text.index("{")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError(f"no balanced JSON object found in {path}")


def untie(cfg: dict) -> dict:
    """quark_3m -> quark_22m: untie the shared loop. RESULTS.md §3."""
    cfg = copy.deepcopy(cfg)
    apps = cfg["model"]["n_unique_layers"] * cfg["model"]["n_loops"]
    cfg["model"]["n_unique_layers"] = apps
    cfg["model"]["n_loops"] = 1
    return cfg


def with_shards(cfg: dict, data_dir: str, artifact_dir: str) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["train_shard"] = os.path.join(data_dir, "train.bin")
    cfg["valid_shard"] = os.path.join(data_dir, "valid.bin")
    cfg["artifact_dir"] = artifact_dir
    return cfg


# Experiment sets, smallest-blast-radius first. Each experiment is
# (name, mutate(cfg)->cfg, evaluate?, rationale).
def build_experiments(base22: dict) -> dict:
    def ep(cfg, n):
        cfg = copy.deepcopy(cfg)
        cfg["num_epochs"] = n
        return cfg

    def dropout(cfg, p):
        cfg = copy.deepcopy(cfg)
        cfg["model"]["dropout"] = p
        return cfg

    def wd(cfg, w):
        cfg = copy.deepcopy(cfg)
        cfg["weight_decay"] = w
        return cfg

    def accum(cfg, a):
        cfg = copy.deepcopy(cfg)
        cfg["grad_accumulation"] = a
        return cfg

    def demb(cfg, d):
        cfg = copy.deepcopy(cfg)
        cfg["model"]["d_emb"] = d
        return cfg

    def warm(cfg, r):
        # Run-relative warmup (src/train/mod.rs `warmup_ratio`). Every config that
        # changes the epoch count or the batch split sets this, so the warmup is
        # the same *share* of the run across the sweep instead of drifting with
        # the absolute `warmup_batches` (NEXT.md §3). 0.01 = 1% of the schedule,
        # squarely in the standard 1-2% range and matching what the reference's
        # 200 batches works out to (~1.2%).
        cfg = copy.deepcopy(cfg)
        cfg["warmup_ratio"] = r
        return cfg

    # NEXT.md §2's recommended run: 4 epochs + dropout 0.1, weight decay swept.
    # The 4x epoch count would quarter the reference warmup's share, so pin it.
    def sweep_cfg(w):
        return warm(wd(dropout(ep(base22, 4), 0.1), w), 0.01)

    sets = {}

    # --- benchmark: identical short workload, one per backend (filled by driver)
    # (handled specially in the driver; no config needed beyond the baseline)

    # --- quick: reproduce the measured 22M point, cheapest possible sanity
    sets["quick"] = [
        ("e0_baseline_22m", base22, True,
         "quark_22m, 1 epoch: reproduce the MEASURED word PPL 74.965 (NEXT.md §4). "
         "Validates the whole pipeline before spending on the sweep."),
    ]

    # --- sweep: NEXT.md §13 step 2, the largest expected win
    sets["sweep"] = [
        (f"e2_4ep_do0.1_wd{w}", sweep_cfg(w), True,
         f"4 epochs + dropout 0.1 + weight_decay {w}: the one recommendation fitted "
         f"at quark's scale (Muennighoff 2023; NEXT.md §2). §13 step 2.")
        for w in (0.1, 0.5, 1.0, 2.0)
    ]

    # --- extra: §13 steps 3 and 5, gated on the sweep in the docs but generated
    # here so the maintainer can run them without regenerating. wd 0.5 is a
    # placeholder "best"; re-point after the sweep.
    best = sweep_cfg(0.5)
    sets["extra"] = [
        ("e3_batch64k", accum(best, 8), True,
         "batch 32,768 -> 65,536 tokens (grad_accum 4 -> 8). Two laws say quark is "
         "starved (NEXT.md §3). NOTE: re-tune LR after (B_opt and eta_opt were fit "
         "jointly) -- this config keeps lr fixed and is only the first rung."),
        ("e5_demb256", demb(best, 256), True,
         "d_emb 128 -> 256: the cheapest untested lever, lifts the rank-128 softmax "
         "bottleneck (NEXT.md §4, §13 step 5)."),
    ]

    return sets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, help="output of `quark train --dry-run`")
    ap.add_argument("--data-dir", required=True, help="dir holding train.bin/valid.bin/test.bin")
    ap.add_argument("--out-dir", required=True, help="where to write the config JSONs")
    ap.add_argument("--artifact-root", default="artifacts/exp",
                    help="per-experiment artifact dirs are created under here")
    ap.add_argument("--set", dest="sets", action="append",
                    choices=["quick", "sweep", "extra", "all"], default=None,
                    help="which experiment set(s) to emit; repeatable. Default: quick.")
    args = ap.parse_args()

    baseline = load_baseline(args.baseline)
    base22 = untie(baseline)
    all_sets = build_experiments(base22)

    want = args.sets or ["quick"]
    if "all" in want:
        want = ["quick", "sweep", "extra"]

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = []
    seen = set()
    for setname in want:
        for name, cfg, evaluate, rationale in all_sets[setname]:
            if name in seen:
                continue
            seen.add(name)
            art = os.path.join(args.artifact_root, name)
            full = with_shards(cfg, args.data_dir, art)
            path = os.path.join(args.out_dir, f"{name}.json")
            with open(path, "w") as fh:
                json.dump(full, fh, indent=2)
            manifest.append({
                "name": name,
                "set": setname,
                "config": path,
                "artifact_dir": art,
                "evaluate": evaluate,
                "epochs": full["num_epochs"],
                "rationale": rationale,
            })
            print(f"  {name:22s} epochs={full['num_epochs']} "
                  f"dropout={full['model']['dropout']} wd={full['weight_decay']} "
                  f"d_emb={full['model']['d_emb']} accum={full['grad_accumulation']} "
                  f"warmup_ratio={full.get('warmup_ratio')}")

    mpath = os.path.join(args.out_dir, "manifest.json")
    with open(mpath, "w") as fh:
        json.dump({"experiments": manifest}, fh, indent=2)
    print(f"wrote {len(manifest)} configs and {os.path.relpath(mpath)}")


if __name__ == "__main__":
    main()
