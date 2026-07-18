#!/usr/bin/env bash
# Tests for the experiment driver's data discovery and resumability (issue #10).
#
# The driver is the one piece of this repository that only ever runs on a machine
# nobody here has, which is how it shipped with a data locator that could not
# find the data outside GitHub Actions and a loop that restarted every
# interrupted run from zero. Both are shell-level properties, so both are
# testable without a GPU: `stub_quark.py` stands in for the binary and reproduces
# what it leaves on disk and what it refuses to start on top of.
#
# No GPU, no cargo, no model. About a minute, nearly all of it Python startup:
# the driver shells out to gen_configs.py, collect.py and report.py on every one
# of the dozen runs below.
#
#   ./experiments/gpu/test_run_experiments.sh
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_REPO="$(cd "$HERE/../.." && pwd)"
PASS=0
FAIL=0

ok()   { PASS=$((PASS + 1)); printf '\033[1;32mok\033[0m   %s\n' "$*"; }
bad()  { FAIL=$((FAIL + 1)); printf '\033[1;31mFAIL\033[0m %s\n' "$*"; }
check() {  # check <description> <condition-as-command...>
  local desc="$1"; shift
  if "$@" >/dev/null 2>&1; then ok "$desc"; else bad "$desc"; fi
}

# --------------------------------------------------------------------------
# A throwaway copy of the repo laid out exactly the way the self-hosted runner
# lays it out: checkout at `_work/<repo>/<repo>`, corpus two levels above it in
# `_work/`. That is the arrangement the old locator could not see through
# without $GITHUB_WORKSPACE, i.e. the one that broke the local run in issue #10.
# --------------------------------------------------------------------------
new_fixture() {
  local root; root="$(mktemp -d)"
  local repo="$root/_work/quark/quark"
  mkdir -p "$repo/experiments/gpu" "$repo/docs/report" "$root/_work/blimp/data"
  cp "$SRC_REPO/experiments/gpu/run_experiments.sh" \
     "$SRC_REPO/experiments/gpu/gen_configs.py" \
     "$SRC_REPO/experiments/gpu/collect.py" \
     "$SRC_REPO/experiments/gpu/stub_quark.py" "$repo/experiments/gpu/"
  cp "$SRC_REPO/experiments/report.py" "$repo/experiments/"
  chmod +x "$repo/experiments/gpu/run_experiments.sh" "$repo/experiments/gpu/stub_quark.py"
  # Enough bytes that the stub's shards are non-empty; the content is irrelevant
  # because no tokenizer runs.
  for split in train valid test; do
    head -c 4096 /dev/urandom | base64 | head -c 4000 > "$root/_work/wiki.$split.tokens"
  done
  printf '{"sentence_good":"a","sentence_bad":"b","UID":"x"}\n' \
    > "$root/_work/blimp/data/anaphor_agreement.jsonl"
  echo "$root"
}

# Run the driver inside a fixture. Extra env comes from the caller.
drive() {  # drive <root> [env assignments...] ; writes <root>/run.log
  local root="$1"; shift
  local repo="$root/_work/quark/quark"
  env "$@" \
    QUARK_CMD="$repo/experiments/gpu/stub_quark.py" \
    STUB_CALLS="$root/calls.log" \
    "$repo/experiments/gpu/run_experiments.sh" > "$root/run.log" 2>&1
}

train_calls() {  # train_calls <root> -- the `train` invocations, one per line
  grep '^train ' "$1/calls.log" 2>/dev/null || true
}

# --------------------------------------------------------------------------
# 1. Data discovery: the runner layout, without $GITHUB_WORKSPACE
# --------------------------------------------------------------------------
echo "== data discovery =="
root="$(new_fixture)"
(cd / && drive "$root" DRY_RUN=1)
status=$?
check "locates the corpus with GITHUB_WORKSPACE unset and cwd elsewhere" [ "$status" = 0 ]
check "reports the _work corpus dir" \
  grep -q "QUARK_DATA_DIR=$root/_work\$" "$root/run.log"
# The BLiMP loader reads *.jsonl non-recursively, so `blimp/` itself is the
# wrong answer and `blimp/data/` is the right one (src/eval/blimp.rs).
check "descends into blimp/data/ where the .jsonl files actually are" \
  grep -q "QUARK_BLIMP_DIR=$root/_work/blimp/data\$" "$root/run.log"
rm -rf "$root"

root="$(new_fixture)"
mkdir -p "$root/_work/wikitext-103"
mv "$root/_work"/wiki.*.tokens "$root/_work/wikitext-103/"
drive "$root" DRY_RUN=1
check "finds a nested wikitext-103/ layout too" \
  grep -q "QUARK_DATA_DIR=$root/_work/wikitext-103\$" "$root/run.log"
rm -rf "$root"

# The maintainer's own invocation (PR #11): corpus and `blimp/` sitting in the
# checkout itself, pointed at with QUARK_DATA_DIR=./ -- and the old driver handed
# `blimp/` straight to the binary, which reads *.jsonl non-recursively and died
# with "no .jsonl files in .../blimp" after four hours of training.
root="$(new_fixture)"
repo="$root/_work/quark/quark"
mv "$root/_work"/wiki.*.tokens "$repo/"
mv "$root/_work/blimp" "$repo/"
(cd "$repo" && drive "$root" QUARK_DATA_DIR=./ DRY_RUN=1)
check "accepts a relative QUARK_DATA_DIR pointing at the checkout itself" \
  grep -q "QUARK_DATA_DIR=$repo\$" "$root/run.log"
check "and still finds BLiMP one level down, in blimp/data/" \
  grep -q "QUARK_BLIMP_DIR=$repo/blimp/data\$" "$root/run.log"
rm -rf "$root"

root="$(new_fixture)"
rm "$root/_work"/wiki.*.tokens
drive "$root" DRY_RUN=1
check "fails loudly, with the override named, when there is no corpus" \
  grep -q "QUARK_DATA_DIR=/path/to/_work" "$root/run.log"
rm -rf "$root"

# --------------------------------------------------------------------------
# 2. A clean run produces the metrics
# --------------------------------------------------------------------------
echo
echo "== clean run =="
root="$(new_fixture)"
drive "$root" DO_BENCHMARK=0 EXPERIMENT_SET=quick
check "exits 0" [ "$?" = 0 ]
repo="$root/_work/quark/quark"
check "writes results.json" [ -f "$repo/experiments/gpu/results.json" ]
check "collects the parsed word perplexity" \
  grep -q '"word_ppl": 74.965' "$repo/experiments/gpu/results.json"
check "collects BLiMP, so --blimp reached the binary" \
  grep -q '"blimp": 61.76' "$repo/experiments/gpu/results.json"
# A stub run finishes in well under a second. Timing it in whole seconds
# recorded a 0, which collect.py then reads as "no measurement" -- the faster
# the thing being measured, the more likely its number disappears.
check "times the run finely enough that a sub-second run is still a number" \
  python3 -c "import json,sys;d=json.load(open(sys.argv[1]));assert d['experiments'][0]['train_seconds']>0,d" \
  "$repo/experiments/gpu/results.json"

# ... and re-running it does no work twice.
: > "$root/calls.log"
drive "$root" DO_BENCHMARK=0 EXPERIMENT_SET=quick
check "a second run retrains nothing" [ -z "$(train_calls "$root" | grep -v -- --dry-run)" ]
check "a second run re-evaluates nothing" [ -z "$(grep '^eval ' "$root/calls.log" || true)" ]
check "and says so" grep -q "already trained" "$root/run.log"
rm -rf "$root"

# --------------------------------------------------------------------------
# 3. Resume: a run killed mid-training continues from its last epoch
# --------------------------------------------------------------------------
echo
echo "== resume after an interrupted run =="
root="$(new_fixture)"
victim="e2_4ep_do0.1_wd0.1"
drive "$root" DO_BENCHMARK=0 EXPERIMENT_SET=sweep "STUB_DIE_AFTER_EPOCH=$victim:2"
repo="$root/_work/quark/quark"
check "the killed experiment left no model" [ ! -f "$repo/artifacts/exp/$victim/model.mpk" ]
check "it did leave an epoch-2 checkpoint" \
  [ -f "$repo/artifacts/exp/$victim/checkpoint/model-2.mpk" ]
check "one experiment dying does not stop the others" \
  [ -f "$repo/artifacts/exp/e2_4ep_do0.1_wd1.0/model.mpk" ]

killed_ms="$(cat "$repo/experiments/gpu/results/$victim.ms" 2>/dev/null || echo -1)"
check "the interrupted leg's time is remembered, not lost with it" [ "$killed_ms" -gt 0 ]
check "but no duration is reported for a run that has not finished" \
  [ ! -f "$repo/experiments/gpu/results/$victim.secs" ]

: > "$root/calls.log"
drive "$root" DO_BENCHMARK=0 EXPERIMENT_SET=sweep
check "the rerun resumes the killed run from epoch 2" \
  grep -q -- "--resume-from-epoch 2" "$root/calls.log"
check "and says so" grep -q "resuming $victim from epoch 2" "$root/run.log"
check "the rerun finishes it" [ -f "$repo/artifacts/exp/$victim/model.mpk" ]
# The regression this whole section exists for: the old driver `rm -rf`'d the
# artifact dir before every experiment, so a rerun retrained all four from zero.
check "the rerun retrains only the killed one" \
  [ "$(train_calls "$root" | grep -c -- --artifact-dir)" = 1 ]
# The reported cost of a resumed experiment is both legs. Timing only the leg
# that happened to finish would credit the run with a fraction of the GPU hours
# it actually took.
check "and reports the whole run's time, not just the leg after the resume" \
  [ "$(cat "$repo/experiments/gpu/results/$victim.ms")" -gt "$killed_ms" ]
rm -rf "$root"

# A partial run with no checkpoint at all cannot be resumed -- but it also must
# not be left in place, or the binary refuses the directory (`refuse_to_merge_runs`).
root="$(new_fixture)"
drive "$root" DO_BENCHMARK=0 EXPERIMENT_SET=quick \
  "STUB_DIE_AFTER_EPOCH=e0_baseline_22m:1" STUB_NO_CHECKPOINTS=e0_baseline_22m
: > "$root/calls.log"
drive "$root" DO_BENCHMARK=0 EXPERIMENT_SET=quick
status=$?
repo="$root/_work/quark/quark"
check "restarts a checkpointless partial run instead of failing on it" [ "$status" = 0 ]
check "and trains it to completion" [ -f "$repo/artifacts/exp/e0_baseline_22m/model.mpk" ]
rm -rf "$root"

# --------------------------------------------------------------------------
# 4. A failed evaluation is not fatal, and is retried
# --------------------------------------------------------------------------
echo
echo "== a failed eval does not take the run down =="
root="$(new_fixture)"
drive "$root" DO_BENCHMARK=0 EXPERIMENT_SET=quick STUB_FAIL_EVAL=e0_baseline_22m
status=$?
repo="$root/_work/quark/quark"
check "the driver still exits 0" [ "$status" = 0 ]
check "the report is still collected" [ -f "$repo/experiments/gpu/results.json" ]
# The half-written report must not be mistaken for a finished one next time.
check "no complete .eval.txt is left behind" \
  [ ! -f "$repo/experiments/gpu/results/e0_baseline_22m.eval.txt" ]
check "the partial output is kept for diagnosis" \
  [ -f "$repo/experiments/gpu/results/e0_baseline_22m.eval.txt.part" ]

: > "$root/calls.log"
drive "$root" DO_BENCHMARK=0 EXPERIMENT_SET=quick
check "the rerun retries the eval without retraining" \
  [ -n "$(grep '^eval ' "$root/calls.log" || true)" ]
check "and it lands" \
  grep -q '"word_ppl": 74.965' "$repo/experiments/gpu/results.json"
rm -rf "$root"

# --------------------------------------------------------------------------
# 5. The backend benchmark accumulates instead of being rebuilt
# --------------------------------------------------------------------------
echo
echo "== backend benchmark =="
root="$(new_fixture)"
drive "$root" EXPERIMENT_SET=quick "BENCH_BACKENDS=wgpu vulkan" BENCH_MAX_BYTES=1000
repo="$root/_work/quark/quark"
check "backends.json is valid JSON with both backends" \
  python3 -c "import json,sys;d=json.load(open(sys.argv[1]));assert set(d)=={'wgpu','vulkan'},d" \
  "$repo/experiments/gpu/results/backends.json"
check "and reaches results.json as throughput" \
  grep -q '"tokens_per_sec"' "$repo/experiments/gpu/results.json"

: > "$root/calls.log"
drive "$root" EXPERIMENT_SET=quick "BENCH_BACKENDS=wgpu vulkan" BENCH_MAX_BYTES=1000
check "a second run does not re-time the backends" grep -q "already timed" "$root/run.log"
check "and still writes both" \
  python3 -c "import json,sys;d=json.load(open(sys.argv[1]));assert set(d)=={'wgpu','vulkan'},d" \
  "$repo/experiments/gpu/results/backends.json"
rm -rf "$root"

# --------------------------------------------------------------------------
# 6. The time budget stops the loop, and the report is still produced
# --------------------------------------------------------------------------
echo
echo "== time budget =="
root="$(new_fixture)"
drive "$root" DO_BENCHMARK=0 EXPERIMENT_SET=sweep TIME_BUDGET_HOURS=0
status=$?
repo="$root/_work/quark/quark"
check "exits 0" [ "$status" = 0 ]
check "names what it skipped" grep -q "time budget (0h) exhausted" "$root/run.log"
check "trains nothing" [ -z "$(train_calls "$root" | grep -- --artifact-dir || true)" ]
check "and still collects a report" [ -f "$repo/experiments/gpu/results.json" ]
rm -rf "$root"

# --------------------------------------------------------------------------
echo
if [ "$FAIL" = 0 ]; then
  printf '\033[1;32m%d passed\033[0m\n' "$PASS"
else
  printf '\033[1;31m%d passed, %d FAILED\033[0m\n' "$PASS" "$FAIL"
fi
exit $(( FAIL > 0 ))
