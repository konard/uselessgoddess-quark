# GPU experiment harness (issue #8)

Everything needed to run the quark_22m sweep and the backend benchmark on a
machine with a GPU, and to turn the raw output into the graphical report. It is
driven by one script and one workflow; the Python around it only generates
configs and parses results — it never runs a model or invents a number.

## What it answers

The issue asks three concrete things of the `gpu` runner:

1. **Which backend is fastest** — `wgpu`, `vulkan`, or `rocm`? →
   the *backend benchmark* times an identical short training run per backend and
   reports tokens/sec.
2. **Does the recommended sweep pay off** — 4 epochs + dropout + a weight-decay
   sweep, the largest expected win in `docs/NEXT.md` §13? →
   the *experiment sets* train quark_22m variants and evaluate each on the
   WikiText-103 test split + BLiMP.
3. **Where is the limiting factor** → the rendered report (`docs/report/`)
   overlays whatever finished onto the frozen MEASURED baselines.

## Running it

On a machine (or self-hosted runner) with a GPU and the data:

```bash
# data lives in _work/ per the issue; override with QUARK_DATA_DIR if elsewhere
EXPERIMENT_SET=quick TRAIN_BACKEND=wgpu ./experiments/gpu/run_experiments.sh
```

or from GitHub: **Actions → GPU experiments → Run workflow**, which runs on
`[self-hosted, gpu]` and uploads the report + `results.json` as an artifact.

### Knobs (env vars, all optional)

| var | default | meaning |
|-----|---------|---------|
| `EXPERIMENT_SET` | `quick` | `quick` \| `sweep` \| `extra` \| `all` |
| `TRAIN_BACKEND` | `wgpu` | backend for training + eval |
| `BENCH_BACKENDS` | `wgpu vulkan` | backends to time against each other |
| `DO_BENCHMARK` | `1` | set `0` to skip the backend benchmark |
| `TIME_BUDGET_HOURS` | `6` | soft cap; remaining experiments are logged as skipped, not dropped silently |
| `QUARK_DATA_DIR` | *(auto)* | folder with `wiki.{train,valid,test}.tokens` + `blimp/` |
| `VOCAB_SIZE` | `8192` | tokenizer vocabulary |
| `BENCH_MAX_BYTES` | `20000000` | text size for the speed test |

### Experiment sets (`docs/NEXT.md` §13 order)

- **quick** — `e0_baseline_22m`: 1 epoch, reproduces the MEASURED word PPL
  74.965. The pipeline sanity check; run this first.
- **sweep** — `e2_4ep_do0.1_wd{0.1,0.5,1.0,2.0}`: 4 epochs + dropout 0.1, weight
  decay swept. §13 step 2, the largest expected win. ~4 runs.
- **extra** — `e3_batch64k` (grad-accum 4→8) and `e5_demb256` (d_emb 128→256):
  §13 steps 3 and 5. `wd 0.5` is a placeholder "best"; re-point after the sweep.

Time cost is real: a 4-epoch quark_22m run is hours. The full `sweep` alone can
exceed the 8h window, which is why the loop is time-boxed and every skipped run
is named in the log.

## Pieces

| file | role |
|------|------|
| `run_experiments.sh` | the driver: locate data → tokenizer + shards → benchmark → time-boxed sweep → eval → collect → render |
| `gen_configs.py` | derives every `TrainConfig` from `quark train --dry-run`, so configs track the binary's defaults; encodes the §13 sweep as field edits on the `quark_22m` base |
| `collect.py` | parses `quark eval` output + timings + `backends.json` into `results.json`; a metric that isn't in the output stays `null` |
| `../report.py` | renders `results.json` + the frozen MEASURED baselines into `docs/report/` (figures + `REPORT.md`) |
| `../../.github/workflows/gpu-experiments.yml` | runs the above on `[self-hosted, gpu]`, uploads the artifact |

## Provenance

`gen_configs.py` reads the AdamW constants and schedule shape from the binary's
own `--dry-run`; it never hand-writes them. `collect.py` only parses; it leaves
absent metrics `null`. The report labels every value MEASURED / DERIVED /
PROJECTED and refuses to promote one to another. If a run didn't happen, its
cell is empty — not filled with a guess.
