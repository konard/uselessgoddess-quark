#!/usr/bin/env bash
# Issue #8 experiment driver for the self-hosted `gpu` runner.
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

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ART="$REPO/artifacts"
SHARDS="$ART/shards"
RESULTS="$REPO/experiments/gpu/results"
CONFIGS="$REPO/experiments/gpu/configs"
mkdir -p "$ART" "$SHARDS" "$RESULTS" "$CONFIGS"

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }

quark() {  # quark <backend> <args...>
  local backend="$1"; shift
  cargo run --release --quiet --no-default-features --features "$backend" --bin quark -- "$@"
}

# --------------------------------------------------------------------------
# 1. Locate the data
# --------------------------------------------------------------------------
find_data() {
  local candidates=(
    "$QUARK_DATA_DIR"
    "${GITHUB_WORKSPACE:-}/../.."          # <runner>/_work/<repo>/<repo> -> _work
    "${RUNNER_WORKSPACE:-}/.."
    "$HOME/r/actions-runner/_work"
    "$REPO/_work"
    "$REPO/../_work"
    "./_work"
  )
  for d in "${candidates[@]}"; do
    [ -n "$d" ] || continue
    if [ -f "$d/wiki.train.tokens" ]; then
      (cd "$d" && pwd); return 0
    fi
  done
  return 1
}

log "Locating WikiText-103 + BLiMP data"
if ! DATA="$(find_data)"; then
  warn "Could not find wiki.train.tokens. Set QUARK_DATA_DIR to the folder holding"
  warn "wiki.{train,valid,test}.tokens and blimp/. The issue put them in _work/."
  exit 2
fi
echo "  data: $DATA"
for f in wiki.train.tokens wiki.valid.tokens wiki.test.tokens; do
  [ -f "$DATA/$f" ] || { warn "missing $DATA/$f"; exit 2; }
done
BLIMP_DIR="$DATA/blimp"
[ -d "$BLIMP_DIR" ] || warn "no blimp/ dir at $BLIMP_DIR -- BLiMP eval will be skipped"

# --------------------------------------------------------------------------
# 2. Tokenizer + shards (idempotent: skip if already built)
# --------------------------------------------------------------------------
log "Building tokenizer + shards with backend=$TRAIN_BACKEND (first build ~12 min)"
TOKENIZER="$SHARDS/tokenizer.json"
if [ ! -f "$TOKENIZER" ]; then
  # `quark tokenizer` trains on the TRAIN split only -- training it on valid or
  # test would leak them into the vocabulary (README).
  quark "$TRAIN_BACKEND" tokenizer "$DATA/wiki.train.tokens" \
    --vocab-size "$VOCAB_SIZE" --out "$TOKENIZER"
fi
prepare() {  # prepare <text> <out.bin>
  [ -f "$2" ] && { echo "  reuse $(basename "$2")"; return; }
  quark "$TRAIN_BACKEND" prepare "$1" --out "$2" --tokenizer "$TOKENIZER" --split-articles
}
prepare "$DATA/wiki.train.tokens" "$SHARDS/train.bin"
prepare "$DATA/wiki.valid.tokens" "$SHARDS/valid.bin"
prepare "$DATA/wiki.test.tokens"  "$SHARDS/test.bin"

sidecar_tokens() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['n_tokens'])" "$1"; }

# --------------------------------------------------------------------------
# 3. Backend benchmark: identical short workload, wall-clock per backend
# --------------------------------------------------------------------------
if [ "$DO_BENCHMARK" = "1" ]; then
  log "Backend benchmark (wgpu / vulkan / rocm) on ~${BENCH_MAX_BYTES}B of text"
  BENCH_TXT="$SHARDS/bench.tokens"
  head -c "$BENCH_MAX_BYTES" "$DATA/wiki.train.tokens" > "$BENCH_TXT"
  BENCH_BIN="$SHARDS/bench.bin"
  [ -f "$BENCH_BIN" ] || quark "$TRAIN_BACKEND" prepare "$BENCH_TXT" --out "$BENCH_BIN" --tokenizer "$TOKENIZER"
  BENCH_TOKENS="$(sidecar_tokens "${BENCH_BIN%.bin}.json")"
  echo "  bench shard: $BENCH_TOKENS tokens"

  # A small, fixed config: default reference model, 1 epoch over the bench shard.
  quark "$TRAIN_BACKEND" train --dry-run > "$RESULTS/dry.txt" 2>/dev/null || true
  python3 "$REPO/experiments/gpu/gen_configs.py" --baseline "$RESULTS/dry.txt" \
    --data-dir "$SHARDS" --out-dir "$CONFIGS" --set quick >/dev/null || true

  echo "{" > "$RESULTS/backends.json"
  first=1
  for b in $BENCH_BACKENDS; do
    art="$ART/bench-$b"; rm -rf "$art"
    # Compile FIRST, outside the timed region: otherwise the first uncached
    # backend would be charged ~12 min of rustc against its throughput. This
    # also doubles as the build-availability probe (rocm may not link).
    echo "  building backend=$b ..."
    if ! cargo build --release --quiet --no-default-features --features "$b" --bin quark \
        > "$RESULTS/bench-$b.build.log" 2>&1; then
      warn "backend $b failed to BUILD -- skipped (see bench-$b.build.log)"
      continue
    fi
    echo "  timing backend=$b ..."
    start=$(date +%s)
    if quark "$b" train --backend "$b" \
        --train-shard "$BENCH_BIN" --valid-shard "$BENCH_BIN" \
        --artifact-dir "$art" --num-epochs 1 --seq-len 512 --batch-size 8 \
        > "$RESULTS/bench-$b.log" 2>&1; then
      end=$(date +%s); secs=$((end - start))
      [ "$first" = 1 ] || echo "," >> "$RESULTS/backends.json"
      first=0
      printf '  "%s": {"seconds": %d, "tokens": %s}' "$b" "$secs" "$BENCH_TOKENS" >> "$RESULTS/backends.json"
      echo "    $b: ${secs}s"
    else
      warn "backend $b built but failed to RUN -- skipped (see bench-$b.log)"
    fi
  done
  printf '\n}\n' >> "$RESULTS/backends.json"
fi

# --------------------------------------------------------------------------
# 4. Experiment set (time-boxed)
# --------------------------------------------------------------------------
log "Generating experiment configs: set=$EXPERIMENT_SET"
quark "$TRAIN_BACKEND" train --dry-run > "$RESULTS/dry.txt" 2>/dev/null || true
python3 "$REPO/experiments/gpu/gen_configs.py" --baseline "$RESULTS/dry.txt" \
  --data-dir "$SHARDS" --out-dir "$CONFIGS" --set "$EXPERIMENT_SET"

DEADLINE=$(( $(date +%s) + TIME_BUDGET_HOURS * 3600 ))
python3 -c "import json;[print(e['name'],e['config'],e['artifact_dir'],int(e['evaluate'])) for e in json.load(open('$CONFIGS/manifest.json'))['experiments']]" \
| while read -r name config art evaluate; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    warn "time budget (${TIME_BUDGET_HOURS}h) exhausted -- SKIPPING $name and the rest"
    break
  fi
  log "Experiment $name"
  rm -rf "$REPO/$art"
  start=$(date +%s)
  if ! quark "$TRAIN_BACKEND" train --config "$config" --backend "$TRAIN_BACKEND"; then
    warn "$name training failed -- see logs; continuing"
    continue
  fi
  echo "$(( $(date +%s) - start ))" > "$RESULTS/$name.secs"

  if [ "$evaluate" = "1" ]; then
    blimp_arg=(); [ -d "$BLIMP_DIR" ] && blimp_arg=(--blimp "$BLIMP_DIR")
    quark "$TRAIN_BACKEND" eval --backend "$TRAIN_BACKEND" \
      --artifact-dir "$REPO/$art" --tokenizer "$TOKENIZER" \
      --ppl "$SHARDS/test.bin" "${blimp_arg[@]}" \
      | tee "$RESULTS/$name.eval.txt"
  fi
done

# --------------------------------------------------------------------------
# 5. Collect + render
# --------------------------------------------------------------------------
log "Collecting results and rendering the report"
python3 "$REPO/experiments/gpu/collect.py" \
  --results-dir "$RESULTS" --manifest "$CONFIGS/manifest.json" \
  --out "$REPO/experiments/gpu/results.json"
python3 "$REPO/experiments/report.py" --results "$REPO/experiments/gpu/results.json"

log "Done. Report in docs/report/, raw metrics in experiments/gpu/results.json"
