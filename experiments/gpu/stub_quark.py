#!/usr/bin/env python3
"""A stand-in for the `quark` binary, for testing run_experiments.sh.

The driver's resume logic is a set of claims about what the real binary leaves
on disk and what it refuses to start on top of. Those claims are only testable
without a GPU if something else can play the binary's part, so this does --
faithfully on exactly the points the driver depends on:

  * `prepare` writes `<out>.bin` **and** the `<stem>.json` sidecar, sidecar last
    (src/data/shard.rs `ShardWriter::finish`);
  * `train` writes `<artifact_dir>/{train,valid}/epoch-<n>/` logs and
    `<artifact_dir>/checkpoint/model-<n>.mpk` per epoch, and `model.mpk` only
    after the last one (src/train/mod.rs `run`);
  * `train` **refuses** a directory that already holds epoch logs unless
    `--resume-from-epoch` is passed (src/train/mod.rs `refuse_to_merge_runs`) --
    this is the failure a naive re-run walks into, so the stub reproduces it;
  * `eval` prints the metric lines collect.py parses (src/eval/corpus.rs,
    src/eval/blimp.rs).

Failures are injected by environment variable, so a test can stage exactly the
interruption it wants:

  STUB_DIE_AFTER_EPOCH="<name>:<n>"  train <name> dies having finished epoch n
  STUB_FAIL_EVAL="<name>"            eval of <name> exits non-zero
  STUB_NO_CHECKPOINTS="<name>"       train <name> writes logs but no checkpoint
  STUB_CALLS="<path>"                append one line per invocation

`<name>` is the artifact directory's basename, which is the experiment name.
"""

from __future__ import annotations

import json
import os
import sys

# A TrainConfig as `quark train --dry-run` prints it (src/train/mod.rs
# `TrainConfig::default`). gen_configs.py only edits fields on top of this, so
# the shape is what matters here, not the constants.
DRY_RUN_CONFIG = {
    "model": {
        "vocab_size": 8192,
        "d_emb": 128,
        "d_model": 384,
        "n_unique_layers": 1,
        "n_loops": 12,
        "n_heads": 6,
        "n_kv_heads": 2,
        "d_ff": 1024,
        "max_seq_len": 512,
        "dropout": 0.0,
        "rope_theta": 10000.0,
        "norm_eps": 1e-5,
    },
    "train_shard": "artifacts/train.bin",
    "valid_shard": "artifacts/valid.bin",
    "artifact_dir": "artifacts/run",
    "seq_len": 512,
    "batch_size": 16,
    "grad_accumulation": 4,
    "num_epochs": 1,
    "lr": 0.001,
    "min_lr_ratio": 0.1,
    "warmup_batches": 200,
    "warmup_ratio": None,
    "weight_decay": 0.1,
    "beta_1": 0.9,
    "beta_2": 0.95,
    "epsilon": 1e-8,
    "grad_clip_norm": 1.0,
    "z_loss": 0.0,
    "seed": 42,
    "num_workers": 2,
    "resume_from_epoch": None,
}


def opt(argv: list[str], flag: str, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def injected(var: str, name: str) -> str | None:
    """Value of `var` if it is armed for experiment `name`, else None."""
    spec = os.environ.get(var, "")
    if not spec:
        return None
    target, _, payload = spec.partition(":")
    return payload if target == name else None


def cmd_tokenizer(argv: list[str]) -> int:
    out = opt(argv, "--out", "artifacts/tokenizer.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"stub": True, "vocab_size": int(opt(argv, "--vocab-size", "8192"))}, fh)
    return 0


def cmd_prepare(argv: list[str]) -> int:
    out = opt(argv, "--out")
    if out is None:
        print("stub: prepare needs --out", file=sys.stderr)
        return 1
    tok = opt(argv, "--tokenizer", "artifacts/tokenizer.json")
    if not os.path.exists(tok):
        print(f"no tokenizer at {tok}; run `quark tokenizer` first", file=sys.stderr)
        return 1
    src = argv[1]
    n_bytes = os.path.getsize(src) if os.path.exists(src) else 0
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # Tokens are 2 bytes each and the sidecar's count must match the .bin's
    # length, or `Shard::open` rejects the pair (src/data/shard.rs).
    n_tokens = max(1, n_bytes // 4)
    with open(out, "wb") as fh:
        fh.write(b"\0\0" * n_tokens)
    sidecar = os.path.splitext(out)[0] + ".json"
    with open(sidecar, "w") as fh:
        json.dump({"n_tokens": n_tokens, "n_words": n_tokens, "n_bytes": n_bytes,
                   "vocab_size": 8192, "n_documents": 1}, fh)
    return 0


def has_epoch_logs(artifact_dir: str) -> bool:
    """Mirrors `recorded_epochs`: one level of split dirs, then `epoch-<n>`."""
    if not os.path.isdir(artifact_dir):
        return False
    for split in os.listdir(artifact_dir):
        path = os.path.join(artifact_dir, split)
        if os.path.isdir(path) and any(e.startswith("epoch-") for e in os.listdir(path)):
            return True
    return False


def cmd_train(argv: list[str]) -> int:
    if "--dry-run" in argv:
        print("parameter budget: stub")
        print(json.dumps(DRY_RUN_CONFIG, indent=2))
        print("backend: Wgpu")
        return 0

    cfg = {}
    config_path = opt(argv, "--config")
    if config_path:
        with open(config_path) as fh:
            cfg = json.load(fh)
    artifact_dir = opt(argv, "--artifact-dir") or cfg.get("artifact_dir", "artifacts/run")
    num_epochs = int(opt(argv, "--num-epochs") or cfg.get("num_epochs", 1))
    resume = opt(argv, "--resume-from-epoch")
    name = os.path.basename(artifact_dir.rstrip("/"))

    if resume is None and has_epoch_logs(artifact_dir):
        print(f"{artifact_dir} already holds metric logs from an earlier run. Training into "
              f"it would merge the two. Remove the directory, point --artifact-dir somewhere "
              f"else, or pass --resume-from-epoch to continue that run deliberately.",
              file=sys.stderr)
        return 1

    start = int(resume) + 1 if resume is not None else 1
    die_after = injected("STUB_DIE_AFTER_EPOCH", name)
    no_checkpoints = os.environ.get("STUB_NO_CHECKPOINTS", "") == name

    for epoch in range(start, num_epochs + 1):
        for split in ("train", "valid"):
            d = os.path.join(artifact_dir, split, f"epoch-{epoch}")
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "Loss.log"), "a").close()
        if not no_checkpoints:
            ckpt = os.path.join(artifact_dir, "checkpoint")
            os.makedirs(ckpt, exist_ok=True)
            open(os.path.join(ckpt, f"model-{epoch}.mpk"), "wb").close()
        if die_after is not None and epoch == int(die_after):
            print(f"stub: dying after epoch {epoch} of {name}", file=sys.stderr)
            return 1

    with open(os.path.join(artifact_dir, "config.json"), "w") as fh:
        json.dump(cfg or DRY_RUN_CONFIG, fh)
    # `model.mpk` last: the driver treats its presence as "this run finished".
    open(os.path.join(artifact_dir, "model.mpk"), "wb").close()
    return 0


def cmd_eval(argv: list[str]) -> int:
    artifact_dir = opt(argv, "--artifact-dir", "artifacts/run")
    name = os.path.basename(artifact_dir.rstrip("/"))
    if not os.path.exists(os.path.join(artifact_dir, "model.mpk")):
        print(f"no model.mpk in {artifact_dir}", file=sys.stderr)
        return 1
    print(f"evaluating {name}")
    if os.environ.get("STUB_FAIL_EVAL", "") == name:
        # Partial output, then death -- the case that must not leave a file the
        # next run reads as a finished evaluation.
        print("token perplexity      31.400")
        sys.stdout.flush()
        print("stub: eval crashed", file=sys.stderr)
        return 1
    print("token perplexity      31.400")
    print("word perplexity       74.965   <- the comparable number")
    print("bits per byte          1.2730")
    if "--blimp" in argv:
        print("BLiMP accuracy  61.76%   (chance is 50.00%)")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    calls = os.environ.get("STUB_CALLS")
    if calls:
        with open(calls, "a") as fh:
            fh.write(" ".join(argv) + "\n")
    if not argv:
        return 2
    handlers = {"tokenizer": cmd_tokenizer, "prepare": cmd_prepare,
                "train": cmd_train, "eval": cmd_eval}
    handler = handlers.get(argv[0])
    if handler is None:
        print(f"stub: unknown subcommand {argv[0]}", file=sys.stderr)
        return 2
    return handler(argv)


if __name__ == "__main__":
    sys.exit(main())
