#!/usr/bin/env bash
# Issue #8 experiment driver for the `gpu` machine -- as a self-hosted runner
# job, or run by hand from a shell (issue #10 does the latter).
#
# What it does, in order:
#   1. locate the WikiText-103 + BLiMP data (the issue put it in _work/)
#   2. build the release binary and the tokenizer + shards (idempotent)
#   3. benchmark wgpu / vulkan / rocm on an identical short workload
#   4. run NEXT.md §13's experiment set (quick | sweep | extra | all),
#      time-boxed, evaluating each on the WikiText-103 test split + BLiMP
#   5. collect everything into results.json and render docs/report/
#
# It never fabricates a metric: a backend that fails to build is skipped and
# said so; an experiment cut by the time budget is logged, not silently dropped.
#
# EVERY STAGE IS RESUMABLE. Re-running the same command after an interruption
# picks up where the last one stopped: finished shards, timed backends and
# trained models are detected on disk and skipped, and a run that died mid-
# training restarts from its last epoch checkpoint rather than from scratch.
# That property is the point of issue #10 -- a 4-epoch quark_22m run is hours,
# and losing it to a failure in the *next* step is not acceptable. `FORCE=1`
# opts out and redoes everything.
#
# Everything is overridable by env var; defaults fit the issue's "~8 hours,
# don't run too long" budget.
#
#   EXPERIMENT_SET=quick TRAIN_BACKEND=vulkan ./experiments/gpu/run_experiments.sh
#
set -euo pipefail

# --------------------------------------------------------------------------
# Config (env-overridable)
# --------------------------------------------------------------------------
EXPERIMENT_SET="${EXPERIMENT_SET:-quick}"          # quick | sweep | extra | all
TRAIN_BACKEND="${TRAIN_BACKEND:-wgpu}"             # backend used for train + eval
BENCH_BACKENDS="${BENCH_BACKENDS:-wgpu vulkan}"    # space-separated; rocm if it builds
DO_BENCHMARK="${DO_BENCHMARK:-1}"
TIME_BUDGET_HOURS="${TIME_BUDGET_HOURS:-6}"        # soft cap on the experiment loop
VOCAB_SIZE="${VOCAB_SIZE:-8192}"
BENCH_MAX_BYTES="${BENCH_MAX_BYTES:-20000000}"     # ~20 MB of text for the speed test
QUARK_DATA_DIR="${QUARK_DATA_DIR:-}"               # override data auto-location
QUARK_BLIMP_DIR="${QUARK_BLIMP_DIR:-}"             # override BLiMP auto-location
FORCE="${FORCE:-0}"                                # 1 = ignore existing work, redo it
DRY_RUN="${DRY_RUN:-0}"                            # 1 = print the plan, run nothing

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Generated configs carry *relative* artifact_dir paths (`artifacts/exp/<name>`),
# and cargo needs to see Cargo.toml. Both are resolved against the working
# directory, so pin it instead of inheriting whatever the caller happened to be
# in -- otherwise a second run from a different directory writes its artifacts
# somewhere else and `refuse_to_merge_runs` rejects the first run's leftovers.
cd "$REPO"

ART="$REPO/artifacts"
SHARDS="$ART/shards"
RESULTS="$REPO/experiments/gpu/results"
CONFIGS="$REPO/experiments/gpu/configs"
mkdir -p "$ART" "$SHARDS" "$RESULTS" "$CONFIGS"

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
skip() { printf '\033[1;32m[skip]\033[0m %s\n' "$*"; }

# Milliseconds since the epoch, and seconds as a decimal built from them.
#
# Whole seconds are not enough resolution: anything that finishes inside one
# second is timed as `0`, and collect.py drops a zero duration rather than
# dividing by it -- so the fastest measurement is the one that disappears. A
# real backend benchmark takes minutes, but the same code is what the tests
# exercise, and a timer that only works when the work is slow is not a timer.
# `date +%s%N` is GNU-only, so a `date` without it falls back to whole seconds
# rather than emitting a literal "N".
now_ms() {
  local ns
  ns="$(date +%s%N 2>/dev/null)"
  case "$ns" in
    '' | *[!0-9]*) echo $(( $(date +%s) * 1000 )) ;;
    *) echo $(( ns / 1000000 )) ;;
  esac
}
secs_since() {  # secs_since <start-ms> -> seconds, three decimals
  local ms=$(( $(now_ms) - $1 ))
  [ "$ms" -ge 0 ] || ms=0
  printf '%d.%03d' $(( ms / 1000 )) $(( ms % 1000 ))
}

# `cargo run` rebuilds nothing when the binary is current, so routing every call
# through it keeps one code path for "build if needed, then run". QUARK_CMD
# exists so the test harness can substitute a stub for the whole binary.
quark() {  # quark <backend> <args...>
  local backend="$1"; shift
  if [ -n "${QUARK_CMD:-}" ]; then
    "$QUARK_CMD" "$@"
    return
  fi
  cargo run --release --quiet --no-default-features --features "$backend" --bin quark -- "$@"
}

quark_build() {  # quark_build <backend> <logfile>
  if [ -n "${QUARK_CMD:-}" ]; then
    return 0
  fi
  cargo build --release --quiet --no-default-features --features "$1" --bin quark >"$2" 2>&1
}

# --------------------------------------------------------------------------
# 0. Preflight
# --------------------------------------------------------------------------
# Failing here costs a second; failing on the same thing four hours into a sweep
# costs the sweep.
for tool in python3; do
  command -v "$tool" >/dev/null || { warn "$tool is not on PATH"; exit 2; }
done
if [ -z "${QUARK_CMD:-}" ]; then
  command -v cargo >/dev/null || { warn "cargo is not on PATH"; exit 2; }
fi

# --------------------------------------------------------------------------
# 1. Locate the data
# --------------------------------------------------------------------------
# The self-hosted runner checks out into `<runner>/_work/<repo>/<repo>`, and the
# issue put the corpus in `<runner>/_work` -- two levels *above* the repo. The
# first version of this script only reached that through `$GITHUB_WORKSPACE`,
# which Actions sets and a hand-run shell does not, so running it locally (issue
# #10) died at "Could not find wiki.train.tokens" before doing any work.
#
# So: walk the repo's ancestors, and at each level try the directory itself, its
# `_work/`, and the usual extracted-tarball subdirectory names. Bounded by the
# ancestor chain, so it cannot wander off into the filesystem.
has_splits() { [ -f "$1/wiki.train.tokens" ]; }

find_data() {
  local -a candidates=(
    "$QUARK_DATA_DIR"
    "${GITHUB_WORKSPACE:-}/../.."
    "${RUNNER_WORKSPACE:-}/.."
  )
  # Ancestors of the repo, and of the caller's original directory, nearest first.
  local dir="$REPO"
  while :; do
    candidates+=("$dir" "$dir/_work" "$dir/data" "$dir/wikitext-103" "$dir/wikitext-103-raw")
    [ "$dir" = "/" ] && break
    dir="$(dirname "$dir")"
  done
  for d in "${candidates[@]}"; do
    [ -n "$d" ] || continue
    [ -d "$d" ] || continue
    if has_splits "$d"; then (cd "$d" && pwd); return 0; fi
  done
  # Last resort: a shallow search under the ancestors that are plausible roots,
  # which catches layouts like `_work/wikitext-103/wiki.train.tokens`.
  for d in "$REPO" "$REPO/.." "$REPO/../.." "$REPO/../../.."; do
    [ -d "$d" ] || continue
    local hit
    hit="$(find "$d" -maxdepth 3 -name wiki.train.tokens -print -quit 2>/dev/null || true)"
    if [ -n "$hit" ]; then (cd "$(dirname "$hit")" && pwd); return 0; fi
  done
  return 1
}

# BLiMP's loader (`src/eval/blimp.rs`) reads `*.jsonl` directly out of the
# directory it is given, non-recursively -- but the upstream suite ships them in
# `blimp/data/`. Handing it the repo root gets "no .jsonl files in ...", so pick
# the directory that actually holds the files.
find_blimp() {
  local -a candidates=(
    "$QUARK_BLIMP_DIR"
    "$DATA/blimp/data"
    "$DATA/blimp"
    "$DATA/data"
    "$DATA/../blimp/data"
    "$DATA/../blimp"
  )
  for d in "${candidates[@]}"; do
    [ -n "$d" ] || continue
    [ -d "$d" ] || continue
    if compgen -G "$d/*.jsonl" >/dev/null 2>&1; then (cd "$d" && pwd); return 0; fi
  done
  return 1
}

log "Locating WikiText-103 + BLiMP data"
if ! DATA="$(find_data)"; then
  warn "Could not find wiki.train.tokens under $REPO or any of its parent"
  warn "directories. Set QUARK_DATA_DIR to the folder holding"
  warn "wiki.{train,valid,test}.tokens (and blimp/). The issue put them in _work/:"
  warn "  QUARK_DATA_DIR=/path/to/_work $0"
  exit 2
fi
echo "  data: $DATA"
missing=0
for f in wiki.train.tokens wiki.valid.tokens wiki.test.tokens; do
  [ -f "$DATA/$f" ] || { warn "missing $DATA/$f"; missing=1; }
done
[ "$missing" = 0 ] || exit 2

if BLIMP_DIR="$(find_blimp)"; then
  echo "  blimp: $BLIMP_DIR ($(compgen -G "$BLIMP_DIR/*.jsonl" | wc -l) paradigms)"
else
  BLIMP_DIR=""
  warn "no directory with BLiMP *.jsonl files found near $DATA -- BLiMP eval will"
  warn "be skipped. Set QUARK_BLIMP_DIR to the folder holding the .jsonl files."
fi

if [ "$DRY_RUN" = "1" ]; then
  log "DRY_RUN=1: data located, nothing else will run"
  echo "  QUARK_DATA_DIR=$DATA"
  echo "  QUARK_BLIMP_DIR=${BLIMP_DIR:-<none>}"
  exit 0
fi

# --------------------------------------------------------------------------
# 2. Tokenizer + shards (idempotent: skip if already built)
# --------------------------------------------------------------------------
log "Building tokenizer + shards with backend=$TRAIN_BACKEND (first build ~12 min)"
TOKENIZER="$SHARDS/tokenizer.json"
if [ "$FORCE" = "1" ] || [ ! -f "$TOKENIZER" ]; then
  # `quark tokenizer` trains on the TRAIN split only -- training it on valid or
  # test would leak them into the vocabulary (README).
  quark "$TRAIN_BACKEND" tokenizer "$DATA/wiki.train.tokens" \
    --vocab-size "$VOCAB_SIZE" --out "$TOKENIZER"
else
  skip "tokenizer.json already built"
fi

# A shard is finished only when its sidecar is there too: `ShardWriter::finish`
# writes the `.json` last (src/data/shard.rs), so a `.bin` without one is a
# prepare that was interrupted, and reusing it would feed truncated tokens to
# every run downstream.
prepare() {  # prepare <text> <out.bin>
  local sidecar="${2%.bin}.json"
  if [ "$FORCE" != "1" ] && [ -f "$2" ] && [ -f "$sidecar" ]; then
    skip "$(basename "$2") ($(sidecar_tokens "$sidecar") tokens)"
    return
  fi
  rm -f "$2" "$sidecar"
  quark "$TRAIN_BACKEND" prepare "$1" --out "$2" --tokenizer "$TOKENIZER" --split-articles
}

sidecar_tokens() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['n_tokens'])" "$1"; }

prepare "$DATA/wiki.train.tokens" "$SHARDS/train.bin"
prepare "$DATA/wiki.valid.tokens" "$SHARDS/valid.bin"
prepare "$DATA/wiki.test.tokens"  "$SHARDS/test.bin"

# --------------------------------------------------------------------------
# 5. Collect + render -- installed as an exit hook, not a final step
# --------------------------------------------------------------------------
# Whatever stops the run -- a failed experiment, the time budget, Ctrl-C -- the
# runs that *did* finish are still collected and rendered. Previously this lived
# at the bottom of the file, so an interruption threw away the report for work
# that had already been paid for in GPU hours.
collect_and_render() {
  local status=$?
  [ -f "$CONFIGS/manifest.json" ] || return $status
  log "Collecting results and rendering the report"
  # collect.py is stdlib-only, so the raw metrics land no matter what.
  python3 "$REPO/experiments/gpu/collect.py" \
    --results-dir "$RESULTS" --manifest "$CONFIGS/manifest.json" \
    --out "$REPO/experiments/gpu/results.json" || warn "collect.py failed"
  # The figure render needs matplotlib; if it is somehow still missing, keep the
  # collected results.json rather than failing the whole run over the pictures.
  if ! python3 "$REPO/experiments/report.py" --results "$REPO/experiments/gpu/results.json"; then
    warn "report render failed (matplotlib missing?) -- results.json is still complete;"
    warn "run 'python3 experiments/report.py --results experiments/gpu/results.json' after"
    warn "'pip install matplotlib' to regenerate docs/report/."
  fi
  log "Report in docs/report/, raw metrics in experiments/gpu/results.json"
  return $status
}
trap collect_and_render EXIT

# --------------------------------------------------------------------------
# 3. Backend benchmark: identical short workload, wall-clock per backend
# --------------------------------------------------------------------------
if [ "$DO_BENCHMARK" = "1" ]; then
  log "Backend benchmark (wgpu / vulkan / rocm) on ~${BENCH_MAX_BYTES}B of text"
  BENCH_TXT="$SHARDS/bench.tokens"
  [ -s "$BENCH_TXT" ] || head -c "$BENCH_MAX_BYTES" "$DATA/wiki.train.tokens" > "$BENCH_TXT"
  BENCH_BIN="$SHARDS/bench.bin"
  bench_sidecar="${BENCH_BIN%.bin}.json"
  if [ "$FORCE" = "1" ] || [ ! -f "$BENCH_BIN" ] || [ ! -f "$bench_sidecar" ]; then
    rm -f "$BENCH_BIN" "$bench_sidecar"
    quark "$TRAIN_BACKEND" prepare "$BENCH_TXT" --out "$BENCH_BIN" --tokenizer "$TOKENIZER"
  fi
  BENCH_TOKENS="$(sidecar_tokens "${BENCH_BIN%.bin}.json")"
  echo "  bench shard: $BENCH_TOKENS tokens"

  for b in $BENCH_BACKENDS; do
    # One file per backend instead of one appended-to JSON document. The old
    # version rebuilt backends.json from scratch on every run and hand-placed
    # the commas, so an interrupted benchmark left a truncated file *and* threw
    # away the backends already timed. Per-backend files make "already measured"
    # a file test, which is what resumability needs.
    bench_json="$RESULTS/bench-$b.json"
    if [ "$FORCE" != "1" ] && [ -f "$bench_json" ]; then
      skip "backend $b already timed ($(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['seconds'])" "$bench_json")s)"
      continue
    fi
    art="$ART/bench-$b"; rm -rf "$art"
    # Compile FIRST, outside the timed region: otherwise the first uncached
    # backend would be charged ~12 min of rustc against its throughput. This
    # also doubles as the build-availability probe (rocm may not link).
    echo "  building backend=$b ..."
    if ! quark_build "$b" "$RESULTS/bench-$b.build.log"; then
      warn "backend $b failed to BUILD -- skipped (see bench-$b.build.log)"
      continue
    fi
    echo "  timing backend=$b ..."
    start=$(now_ms)
    if quark "$b" train --backend "$b" \
        --train-shard "$BENCH_BIN" --valid-shard "$BENCH_BIN" \
        --artifact-dir "$art" --num-epochs 1 --seq-len 512 --batch-size 8 \
        > "$RESULTS/bench-$b.log" 2>&1; then
      secs="$(secs_since "$start")"
      printf '{"seconds": %s, "tokens": %s}\n' "$secs" "$BENCH_TOKENS" > "$bench_json"
      echo "    $b: ${secs}s"
    else
      warn "backend $b built but failed to RUN -- skipped (see bench-$b.log)"
    fi
  done

  # Merge the per-backend files into the shape collect.py reads.
  python3 - "$RESULTS" <<'PY'
import glob, json, os, sys
results = sys.argv[1]
merged = {}
for path in sorted(glob.glob(os.path.join(results, "bench-*.json"))):
    name = os.path.basename(path)[len("bench-"):-len(".json")]
    with open(path) as fh:
        merged[name] = json.load(fh)
with open(os.path.join(results, "backends.json"), "w") as fh:
    json.dump(merged, fh, indent=2)
PY
fi

# --------------------------------------------------------------------------
# 4. Experiment set (time-boxed, resumable)
# --------------------------------------------------------------------------
log "Generating experiment configs: set=$EXPERIMENT_SET"
quark "$TRAIN_BACKEND" train --dry-run > "$RESULTS/dry.txt" 2>/dev/null || true
python3 "$REPO/experiments/gpu/gen_configs.py" --baseline "$RESULTS/dry.txt" \
  --data-dir "$SHARDS" --out-dir "$CONFIGS" --set "$EXPERIMENT_SET"

# The highest epoch with a checkpoint burn can actually load from
# `<dir>/checkpoint/model-<n>.mpk`. Metric logs are not enough: the checkpointing
# strategy prunes the epochs it did not select, so the newest *log* may name an
# epoch whose weights are gone (src/train/mod.rs `recorded_epochs` / `run`).
last_checkpoint_epoch() {
  local dir="$1/checkpoint" best="" n
  [ -d "$dir" ] || return 1
  for f in "$dir"/model-*.mpk; do
    [ -f "$f" ] || continue
    n="${f##*/model-}"; n="${n%.mpk}"
    case "$n" in ''|*[!0-9]*) continue ;; esac
    if [ -z "$best" ] || [ "$n" -gt "$best" ]; then best="$n"; fi
  done
  [ -n "$best" ] || return 1
  echo "$best"
}

# `quark eval` prints "word perplexity <n>" (src/eval/corpus.rs); collect.py
# parses exactly that. An eval that died halfway leaves a file without the line,
# and that must count as not-done rather than as a finished run with no metric.
eval_is_complete() {
  [ -f "$1" ] && grep -q "word perplexity" "$1"
}

# Read the manifest as TSV: `read -r a b c` on space-separated fields would
# split any path containing a space across two variables.
mapfile -t MANIFEST < <(python3 -c "
import json, sys
for e in json.load(open(sys.argv[1]))['experiments']:
    print('\t'.join([e['name'], e['config'], e['artifact_dir'], str(int(e['evaluate']))]))
" "$CONFIGS/manifest.json")
[ "${#MANIFEST[@]}" -gt 0 ] || { warn "manifest.json lists no experiments"; exit 2; }

DEADLINE=$(( $(date +%s) + TIME_BUDGET_HOURS * 3600 ))
for line in "${MANIFEST[@]}"; do
  IFS=$'\t' read -r name config art evaluate <<<"$line"
  case "$art" in /*) art_abs="$art" ;; *) art_abs="$REPO/$art" ;; esac

  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    warn "time budget (${TIME_BUDGET_HOURS}h) exhausted -- SKIPPING $name and the rest"
    break
  fi
  log "Experiment $name"

  # --- train -------------------------------------------------------------
  # `model.mpk` is written last, after the best epoch has been selected and
  # loaded (src/train/mod.rs `run`), so its presence means this run is finished.
  if [ "$FORCE" != "1" ] && [ -f "$art_abs/model.mpk" ]; then
    skip "$name already trained ($art)"
  else
    resume_arg=()
    if [ "$FORCE" = "1" ]; then
      rm -rf "$art_abs" "$RESULTS/$name.ms"
    elif epoch="$(last_checkpoint_epoch "$art_abs")"; then
      # Interrupted mid-run: continue it. Without --resume-from-epoch, burn
      # would refuse the directory outright (`refuse_to_merge_runs`), and
      # deleting it would throw away every epoch already paid for.
      log "  resuming $name from epoch $epoch"
      resume_arg=(--resume-from-epoch "$epoch")
    elif [ -d "$art_abs" ]; then
      # Logs but no loadable checkpoint: nothing to continue from, and leaving
      # them would make burn read the dead run's epochs as part of this one.
      warn "  $art holds a partial run with no checkpoint -- restarting it"
      rm -rf "$art_abs" "$RESULTS/$name.ms"
    fi
    start=$(now_ms)
    quark "$TRAIN_BACKEND" train --config "$config" --backend "$TRAIN_BACKEND" \
      --artifact-dir "$art_abs" "${resume_arg[@]}" && trained=1 || trained=0
    # A resumed experiment is trained across two or more invocations, and the
    # cost of the run is all of them -- including the leg that was interrupted,
    # whose epochs the resume is built on. `.ms` accumulates every leg; `.secs`
    # is written only once the run is finished, and is what collect.py reads.
    leg_ms=$(( $(now_ms) - start ))
    [ "$leg_ms" -ge 0 ] || leg_ms=0
    total_ms=$(( leg_ms + $(cat "$RESULTS/$name.ms" 2>/dev/null || echo 0) ))
    echo "$total_ms" > "$RESULTS/$name.ms"
    if [ "$trained" != "1" ]; then
      warn "$name training failed -- see logs; continuing with the next experiment"
      continue
    fi
    printf '%d.%03d\n' $(( total_ms / 1000 )) $(( total_ms % 1000 )) > "$RESULTS/$name.secs"
  fi

  # --- evaluate ----------------------------------------------------------
  [ "$evaluate" = "1" ] || continue
  if [ "$FORCE" != "1" ] && eval_is_complete "$RESULTS/$name.eval.txt"; then
    skip "$name already evaluated"
    continue
  fi
  blimp_arg=()
  if [ -n "$BLIMP_DIR" ]; then blimp_arg=(--blimp "$BLIMP_DIR"); fi
  # Not fatal: a failed eval used to abort the whole script through `set -e`,
  # taking every not-yet-run experiment with it. Write to a temporary file so a
  # crashed eval cannot leave a half-written report that the next run mistakes
  # for a complete one.
  if quark "$TRAIN_BACKEND" eval --backend "$TRAIN_BACKEND" \
      --artifact-dir "$art_abs" --tokenizer "$TOKENIZER" \
      --ppl "$SHARDS/test.bin" "${blimp_arg[@]}" \
      | tee "$RESULTS/$name.eval.txt.part"; then
    mv "$RESULTS/$name.eval.txt.part" "$RESULTS/$name.eval.txt"
  else
    warn "$name evaluation failed -- see $RESULTS/$name.eval.txt.part; continuing"
  fi
done

log "Experiment loop finished"
