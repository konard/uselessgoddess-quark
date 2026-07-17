#!/usr/bin/env python3
"""Graphical report for issue #8: the architecture, the measured evidence, and
the decided path forward.

Provenance rules, identical to run_analysis.py and scaling_budget.py:

  MEASURED  -- transcribed from a training run's console output or from a cited
               primary source. Never invented.
  DERIVED   -- computed here from MEASURED inputs by code you can read.
  PROJECTED -- a scaling-law extrapolation. Plotted in a DIFFERENT colour and
               hatched, and never mixed into a bar next to a MEASURED one.

Every constant below carries its source inline. The figures render only what is
measured plus clearly-marked projections; nothing here is a training result the
project has not actually observed.

    python3 experiments/report.py                 # measured baseline only
    python3 experiments/report.py --results R.json # overlay harness output

The optional --results file is written by experiments/gpu/run_experiments.sh on
the maintainer's GPU. When absent (the state in which this report was first
committed, because the author had no GPU access -- see docs/ANALYSIS.md), the
GPU-dependent panels render as "TO BE MEASURED" placeholders rather than faking
a number.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")  # no display on a CI runner
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Palette. Measured runs are solid; the GPT-2 target is a reference line;
# projections are hatched and greyed so a reader can never mistake one for data.
# ---------------------------------------------------------------------------
C_QUARK = "#2266cc"
C_QUARK22 = "#0b3d91"
C_TARGET = "#cc2222"
C_PROJECT = "#999999"
C_PEER = "#88aacc"
C_ACCENT = "#e08a1e"
C_OK = "#2e8b57"


# ---------------------------------------------------------------------------
# MEASURED: the reference runs. RESULTS.md §1 (run1-3) and NEXT.md §4 (22m).
# valid loss and BLiMP are the model's own console output; word PPL reproduces
# from total NLL alone (RESULTS.md §1 "the eval harness is trustworthy").
# ---------------------------------------------------------------------------
@dataclass
class Run:
    key: str
    label: str
    stored_params: int
    valid_loss: float
    word_ppl: float
    blimp: float
    vram_gb: float
    minutes: int
    epochs: int
    note: str = ""


RUNS = [
    Run("run1", "quark_3m\n1×12 loop, d384", 2_868_352, 3.706, 115.163, 57.05, 11.0, 60, 1),
    Run("run2", "quark_3m_dense\n6×1, d168", 2_868_352, 3.653, 108.275, 58.63, 5.5, 15, 1),
    Run("run3", "quark_3m_dense\n6×1, d168, 10ep", 2_868_352, 3.707, 123.193, 60.93, 5.5, 150, 10,
        "un-annealed ckpt\n(RESULTS.md §2)"),
    Run("quark_22m", "quark_22m\n12×1 dense, d384", 21_800_320, 3.361, 74.965, 61.76, 11.3, 60, 1,
        "untied loop:\n+0 FLOPs, +0.30 GB"),
]

# MEASURED. GPT-2 124M zero-shot WikiText-103 word PPL (Radford et al. 2019),
# recomputed under quark's protocol by experiments/gpt2_baseline.py. It is
# out-of-domain (WebText excluded Wikipedia) and includes 2.5-5 PPL of an
# unreleased de-tokenizer -- README "The baseline is measured, not cited".
GPT2_WORD_PPL = 37.50

# DERIVED (src/config.rs). Both configs run the same compute graph.
COMPUTE_EQUIV_PARAMS = 20_643_840

# MEASURED: the BabyLM BLiMP-vs-size cliff. RESULTS.md §4.3, each row sourced
# there. These are FILTERED BLiMP (BabyLM removes 13.7%); quark's numbers above
# are FULL unfiltered BLiMP -- the axes are not identical and NEXT.md §7 says so.
# Plotted with that caveat printed on the figure.
BABYLM = [
    ("GPT-BERT (Strict winner)", 119.0, 86.1),
    ("GPT-BERT (Strict-Small)", 30.0, 81.2),
    ("ELC-BERT Original", 24.0, 80.00),
    ("WhatIf", 26.0, 66.9),
    ("BERTtime Stories", 24.0, 63.2),
    ("Co4", 8.0, 53.55),
    ("BitMar", 14.0, 48.7),
]

# DERIVED: Chinchilla token budget. quark_22m saw 6.5 tokens/param in one epoch
# (README; NEXT.md §0), against Chinchilla's ~20 tokens/param optimum
# (Hoffmann et al. 2022). 4 epochs -- NEXT.md §2's top recommendation -- is 4x.
TOK_PER_PARAM_1EP = 6.5
CHINCHILLA_OPT = 20.0
EPOCHS_PLAN = 4

# DERIVED: activation memory at seq=1024 (scaling_budget.py §7). The real 16 GB
# constraint; sharing does NOT reduce it. (batch, total GB, fits?)
VRAM_SEQ1024 = [(4, 2.19), (8, 4.38), (16, 8.76), (32, 17.52)]
VRAM_LIMIT = 16.0

# PROJECTED: parameters needed to buy the 1.060 nat gap from run2 to GPT-2, under
# two scaling fits (RESULTS.md §3.1). Untying delivered 7.6x, for free.
GAP_MULTIPLIERS = {"Hoffmann 2022": 4.7, "Besiroglu 2024": 4.1, "untying (actual)": 7.6}

# The recommended order, NEXT.md §13. (label, rough GPU cost, expected win)
ROADMAP = [
    ("1. Fix warmup unit", "cheap", "unblocks the sweep"),
    ("2. 4 epochs + dropout 0.1,\n   wd {0.1,0.5,1,2}", "~4 GPU-h", "largest expected win"),
    ("3. Batch → 64k,\n   re-tune LR", "~4 GPU-h", "two laws say starved"),
    ("4. Flash attention", "~1 GPU-h", "frees ~2.25 GB"),
    ("5. d_emb 128 → 256", "~1 GPU-h", "lifts rank-128 cap"),
    ("6. softcap / 14×352 /\n   vocab 4096 / AdEMAMix", "1 at a time", "second-order"),
]


def out_dir() -> str:
    d = os.path.join(os.path.dirname(__file__), "..", "docs", "report")
    os.makedirs(d, exist_ok=True)
    return d


def save(fig, name: str) -> str:
    path = os.path.join(out_dir(), name)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path)}")
    return name


# ---------------------------------------------------------------------------
# Figure 1: the perplexity journey. Where quark is, and how far GPT-2 is.
# ---------------------------------------------------------------------------
def fig_perplexity() -> str:
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = range(len(RUNS))
    colors = [C_QUARK, C_QUARK, C_QUARK, C_QUARK22]
    bars = ax.bar(xs, [r.word_ppl for r in RUNS], color=colors, zorder=3)
    for r, b in zip(RUNS, bars):
        ax.text(b.get_x() + b.get_width() / 2, r.word_ppl + 1.5, f"{r.word_ppl:.1f}",
                ha="center", va="bottom", fontweight="bold")
        if r.note:
            ax.text(b.get_x() + b.get_width() / 2, 6, r.note, ha="center", va="bottom",
                    fontsize=7.5, color="#333", style="italic")
    ax.axhline(GPT2_WORD_PPL, color=C_TARGET, ls="--", lw=2, zorder=2)
    ax.text(len(RUNS) - 0.5, GPT2_WORD_PPL + 1.5,
            f"GPT-2 124M zero-shot, out-of-domain: {GPT2_WORD_PPL:.2f}",
            color=C_TARGET, ha="right", va="bottom", fontweight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r.label for r in RUNS], fontsize=8.5)
    ax.set_ylabel("WikiText-103 word perplexity  (lower = better)")
    ax.set_title("Fig 1 · Perplexity: untying the loop cut word PPL from 108 to 75\n"
                 "still 2.0× GPT-2, which is 48× larger — MEASURED (RESULTS.md §1, NEXT.md §4)")
    ax.set_ylim(0, 135)
    ax.grid(axis="y", alpha=0.3)
    ax.annotate("", xy=(3, 80), xytext=(1, 108),
                arrowprops=dict(arrowstyle="->", color=C_OK, lw=2, connectionstyle="arc3,rad=-0.2"))
    ax.text(2.05, 97, "+7.6× params\n0 extra FLOPs", color=C_OK, fontweight="bold", fontsize=9)
    return save(fig, "fig1_perplexity.png")


# ---------------------------------------------------------------------------
# Figure 2: the BabyLM BLiMP cliff. The single most decision-relevant chart:
# where is the floor below which this size class stops working?
# ---------------------------------------------------------------------------
def fig_blimp_cliff() -> str:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # sweet-spot band 24-30M
    ax.axvspan(24, 30, color=C_OK, alpha=0.10, zorder=0)
    ax.text(27, 44, "24–30M\ndemonstrated\nsweet spot", ha="center", va="bottom",
            color=C_OK, fontsize=8.5, fontweight="bold")

    px = [p for _, p, _ in BABYLM]
    py = [b for _, _, b in BABYLM]
    ax.scatter(px, py, s=55, color=C_PEER, zorder=3, label="BabyLM peers (filtered BLiMP)")
    for name, p, b in BABYLM:
        ax.annotate(name, (p, b), textcoords="offset points", xytext=(6, -3),
                    fontsize=7, color="#445")

    # quark points (full unfiltered BLiMP)
    qx = [2.87, 2.87, 2.87, 21.8]
    qy = [57.05, 58.63, 60.93, 61.76]
    qn = ["run1", "run2", "run3", "quark_22m"]
    ax.scatter(qx[:3], qy[:3], s=70, color=C_QUARK, zorder=4, marker="D",
               label="quark_3m (full BLiMP)")
    ax.scatter([qx[3]], [qy[3]], s=120, color=C_QUARK22, zorder=5, marker="*",
               label="quark_22m (full BLiMP)")
    ax.annotate("quark_3m\n57–61", (2.87, 60.93), textcoords="offset points",
                xytext=(8, 4), fontsize=7.5, color=C_QUARK, fontweight="bold")
    ax.annotate("quark_22m 61.76", (21.8, 61.76), textcoords="offset points",
                xytext=(-4, 10), fontsize=8, color=C_QUARK22, fontweight="bold")

    ax.axhline(50, color="#bbb", ls=":", lw=1)
    ax.text(3, 50.4, "≈ chance", fontsize=7, color="#888")
    ax.set_xscale("log")
    ax.set_xlabel("stored parameters (millions, log scale)")
    ax.set_ylabel("BLiMP accuracy")
    ax.set_title("Fig 2 · The BabyLM cliff: grammatical competence needs ~24–30M\n"
                 "quark scores exactly where its size predicts — not a bug to fix (RESULTS.md §4.3)")
    ax.set_xlim(2, 160)
    ax.set_ylim(42, 90)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.text(2.1, 43.2,
            "Caveat: peers use BabyLM-FILTERED BLiMP (−13.7%); quark uses FULL BLiMP. "
            "Axes not identical (NEXT.md §7).", fontsize=6.8, color="#a33", style="italic")
    return save(fig, "fig2_blimp_cliff.png")


# ---------------------------------------------------------------------------
# Figure 3: the central finding. Sharing optimizes the resource never scarce.
# ---------------------------------------------------------------------------
def fig_stored_vs_compute() -> str:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    labels = ["quark_3m\n(1×12 tied)", "quark_22m\n(12×1 untied)"]
    stored = [2_868_352 / 1e6, 21_800_320 / 1e6]
    compute = [COMPUTE_EQUIV_PARAMS / 1e6, COMPUTE_EQUIV_PARAMS / 1e6]
    x = range(len(labels))
    w = 0.38
    ax.bar([i - w / 2 for i in x], stored, w, color=C_QUARK, label="stored params (M)", zorder=3)
    ax.bar([i + w / 2 for i in x], compute, w, color=C_ACCENT, label="compute-equivalent params (M)",
           zorder=3)
    for i, (s, c) in enumerate(zip(stored, compute)):
        ax.text(i - w / 2, s + 0.3, f"{s:.2f}M", ha="center", fontsize=8.5, fontweight="bold")
        ax.text(i + w / 2, c + 0.3, f"{c:.1f}M", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("parameters (millions)")
    ax.set_title("Fig 3 · Weight sharing saves storage, not arithmetic\n"
                 "Both run the SAME 20.6M compute graph. Untying stores 7.6× more for +0.30 GB "
                 "(RESULTS.md §3)")
    ax.set_ylim(0, 24)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left")
    ax.annotate("run1 spent 60 min & 11 GB running a 20.6M graph\nto store 2.87M — VRAM was the "
                "binding constraint, not params", xy=(0, 20.6), xytext=(0.15, 15.5),
                fontsize=8, color="#333",
                arrowprops=dict(arrowstyle="->", color="#666"))
    return save(fig, "fig3_stored_vs_compute.png")


# ---------------------------------------------------------------------------
# Figure 4: the token budget. Why the answer is "more tokens", not "more params".
# ---------------------------------------------------------------------------
def fig_token_budget() -> str:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    cats = ["quark_22m\n1 epoch\n(MEASURED)", f"quark_22m\n{EPOCHS_PLAN} epochs\n(NEXT.md §2 plan)",
            "Chinchilla\noptimum"]
    vals = [TOK_PER_PARAM_1EP, TOK_PER_PARAM_1EP * EPOCHS_PLAN, CHINCHILLA_OPT]
    colors = [C_QUARK22, C_OK, C_PROJECT]
    bars = ax.bar(cats, vals, color=colors, zorder=3)
    bars[1].set_hatch("//")  # the plan is a plan, not a measurement
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", fontweight="bold")
    ax.axhline(CHINCHILLA_OPT, color=C_PROJECT, ls="--", lw=1.5)
    ax.set_ylabel("training tokens seen per stored parameter")
    ax.set_title("Fig 4 · quark_22m is undertrained, not undersized\n"
                 "6.5 tok/param = 0.33× Chinchilla. Growing the model makes this WORSE "
                 "(NEXT.md §0–§1)")
    ax.set_ylim(0, 24)
    ax.grid(axis="y", alpha=0.3)
    ax.text(1, TOK_PER_PARAM_1EP * EPOCHS_PLAN + 1.2,
            "4 epochs on repeated data is\nthe one recommendation fitted\nat quark's scale "
            "(Muennighoff 2023)", ha="center", fontsize=7.5, color="#333")
    return save(fig, "fig4_token_budget.png")


# ---------------------------------------------------------------------------
# Figure 5: the VRAM wall. The real 16 GB constraint is activation memory.
# ---------------------------------------------------------------------------
def fig_vram() -> str:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    b = [x for x, _ in VRAM_SEQ1024]
    g = [y for _, y in VRAM_SEQ1024]
    colors = [C_OK if y <= VRAM_LIMIT else C_TARGET for y in g]
    bars = ax.bar([str(x) for x in b], g, color=colors, zorder=3)
    for bar, y in zip(bars, g):
        ax.text(bar.get_x() + bar.get_width() / 2, y + 0.3, f"{y:.1f} GB",
                ha="center", fontweight="bold")
    ax.axhline(VRAM_LIMIT, color=C_TARGET, ls="--", lw=2)
    ax.text(0, VRAM_LIMIT + 0.3, "16 GB card", color=C_TARGET, fontweight="bold")
    ax.set_xlabel("batch size (sequences, seq_len = 1024)")
    ax.set_ylabel("estimated total VRAM (GB)")
    ax.set_title("Fig 5 · The binding constraint is activation memory, not weights\n"
                 "Attention matrix scales seq²; batch 32 overflows 16 GB (scaling_budget.py §7)")
    ax.set_ylim(0, 20)
    ax.grid(axis="y", alpha=0.3)
    ax.text(2.5, 15.0, "Flash attention (§13 step 4)\nfrees ~2.25 GB here", fontsize=8,
            color=C_ACCENT, fontweight="bold", ha="center")
    return save(fig, "fig5_vram.png")


# ---------------------------------------------------------------------------
# Figure 6: the roadmap, NEXT.md §13, as an ordered plan with cost and expected
# payoff. This is the "path forward" the issue asks for.
# ---------------------------------------------------------------------------
def fig_roadmap() -> str:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.axis("off")
    y = len(ROADMAP)
    for i, (step, cost, win) in enumerate(ROADMAP):
        yy = y - i
        hl = i == 1  # step 2 is the headline
        box_c = C_OK if hl else "#eef3fb"
        ax.add_patch(plt.Rectangle((0, yy - 0.4), 6.2, 0.8, facecolor=box_c,
                                   edgecolor="#88a", alpha=0.35 if hl else 1.0, zorder=2))
        ax.text(0.15, yy, step, va="center", fontsize=9,
                fontweight="bold" if hl else "normal", zorder=3)
        ax.text(6.5, yy, cost, va="center", fontsize=8.5, color=C_ACCENT, fontweight="bold")
        ax.text(8.2, yy, win, va="center", fontsize=8, color="#333", style="italic")
    ax.text(0.15, y + 0.9, "step", fontsize=8.5, color="#666", fontweight="bold")
    ax.text(6.5, y + 0.9, "cost", fontsize=8.5, color="#666", fontweight="bold")
    ax.text(8.2, y + 0.9, "why", fontsize=8.5, color="#666", fontweight="bold")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, y + 1.6)
    ax.set_title("Fig 6 · The decided path forward — each step gated on the one before\n"
                 "NEXT.md §13. Everything below step 2 is noise until step 2 runs.",
                 fontsize=11)
    return save(fig, "fig6_roadmap.png")


# ---------------------------------------------------------------------------
# Figure 7: backend throughput. GPU-only; renders as a placeholder until the
# harness fills experiments/gpu/results.json. NEVER invents numbers.
# ---------------------------------------------------------------------------
def fig_backends(results: dict | None) -> str:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bench = (results or {}).get("backend_benchmark") if results else None
    if bench:
        names = list(bench.keys())
        toks = [bench[n]["tokens_per_sec"] for n in names]
        bars = ax.bar(names, toks, color=[C_QUARK, C_ACCENT, C_OK][: len(names)], zorder=3)
        for b, t in zip(bars, toks):
            ax.text(b.get_x() + b.get_width() / 2, t, f"{t:,.0f}", ha="center", va="bottom",
                    fontweight="bold")
        ax.set_ylabel("training throughput (tokens/sec, higher = better)")
        ax.set_title("Fig 7 · Backend throughput — MEASURED on the gpu runner")
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.axis("off")
        ax.text(0.5, 0.62, "Fig 7 · wgpu vs vulkan vs rocm", ha="center", fontsize=13,
                fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.45,
                "TO BE MEASURED by experiments/gpu/run_experiments.sh on the\n"
                "self-hosted 'gpu' runner. No number is shown because none has been\n"
                "observed — the author had no GPU access (see docs/ANALYSIS.md).\n\n"
                "Prior (NEXT.md §11): try vulkan first on AMD; rocm is untested by anyone here.",
                ha="center", va="center", fontsize=9.5, color="#444",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="#f3f3f3", edgecolor="#bbb"))
    return save(fig, "fig7_backends.png")


# ---------------------------------------------------------------------------
# Figure 8: the sweep outcome. GPU-only; renders as a placeholder until the
# harness fills experiments/gpu/results.json with per-run word PPL / BLiMP.
# When present it plots the NEW runs against the MEASURED quark_22m baseline and
# the GPT-2 reference line, so the report literally shows whether the §13 step-2
# sweep paid off. NEVER invents a number: only non-null word_ppl is plotted.
# ---------------------------------------------------------------------------
def _short(name: str) -> str:
    return (name.replace("e2_4ep_do0.1_", "4ep/").replace("e0_baseline_22m", "22m·1ep")
            .replace("e3_batch64k", "4ep/batch64k").replace("e5_demb256", "4ep/d_emb256")
            .replace("_", "\n"))


def fig_sweep(results: dict | None) -> str:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    runs = [e for e in (results or {}).get("experiments", [])
            if e.get("word_ppl") is not None]
    if runs:
        # MEASURED quark_22m baseline first, then whatever the harness ran (dedup
        # if the harness re-measured the baseline as e0).
        base = ("quark_22m\n(prior\nMEASURED)", 74.965, 61.76, True)
        rows = [base] + [(_short(e["name"]), e["word_ppl"], e.get("blimp"), False)
                         for e in runs if e["name"] != "e0_baseline_22m"
                         or e["word_ppl"] != 74.965]
        labels = [r[0] for r in rows]
        ppls = [r[1] for r in rows]
        colors = [C_PROJECT if r[3] else C_OK for r in rows]
        x = range(len(rows))
        bars = ax.bar(x, ppls, color=colors, zorder=3)
        best = min((r[1] for r in rows[1:]), default=None)
        for r, b in zip(rows, bars):
            ax.text(b.get_x() + b.get_width() / 2, r[1] + 0.8, f"{r[1]:.1f}",
                    ha="center", va="bottom", fontweight="bold", fontsize=8.5)
            if r[2] is not None:
                ax.text(b.get_x() + b.get_width() / 2, 3, f"BLiMP\n{r[2]:.1f}", ha="center",
                        va="bottom", fontsize=7, color="#333", style="italic")
        ax.axhline(GPT2_WORD_PPL, color=C_TARGET, ls="--", lw=2, zorder=2)
        ax.text(len(rows) - 0.5, GPT2_WORD_PPL + 0.8, f"GPT-2 {GPT2_WORD_PPL:.2f}",
                color=C_TARGET, ha="right", va="bottom", fontweight="bold")
        ax.axhline(74.965, color=C_PROJECT, ls=":", lw=1.3, zorder=1)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel("WikiText-103 word perplexity  (lower = better)")
        verdict = ("the sweep improved on the baseline"
                   if best is not None and best < 74.965 else
                   "the sweep did not beat the baseline")
        ax.set_title("Fig 8 · Sweep outcome — MEASURED on the gpu runner\n"
                     f"NEXT.md §13 step 2 (4 epochs + dropout + weight-decay sweep): {verdict}")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(max(ppls), 80) * 1.12)
    else:
        ax.axis("off")
        ax.text(0.5, 0.62, "Fig 8 · The §13 step-2 sweep", ha="center", fontsize=13,
                fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.42,
                "TO BE MEASURED by experiments/gpu/run_experiments.sh (set=sweep).\n"
                "Bars will show word PPL for 4 epochs + dropout 0.1 across\n"
                "weight_decay ∈ {0.1, 0.5, 1.0, 2.0}, against the MEASURED\n"
                "quark_22m baseline (74.965) and the GPT-2 line. No number is\n"
                "shown because none has been observed (see docs/ANALYSIS.md).",
                ha="center", va="center", fontsize=9.5, color="#444",
                transform=ax.transAxes,
                bbox=dict(boxstyle="round", facecolor="#f3f3f3", edgecolor="#bbb"))
    return save(fig, "fig8_sweep.png")


def _metrics_table(results: dict | None) -> list[str]:
    """A markdown table of the harness's per-run metrics, when it has run."""
    runs = [e for e in (results or {}).get("experiments", [])
            if e.get("word_ppl") is not None]
    if not runs:
        return []
    lines = ["### Measured sweep results", "",
             "| run | epochs | word PPL | bits/byte | BLiMP % | train (min) |",
             "|-----|-------:|---------:|----------:|--------:|------------:|"]
    for e in runs:
        mins = e.get("train_seconds")
        mins = f"{mins / 60:.0f}" if mins else "—"
        bpb = e.get("bits_per_byte")
        lines.append(
            f"| `{e['name']}` | {e.get('epochs') or '—'} | {e['word_ppl']:.3f} | "
            f"{bpb:.4f} | {(e.get('blimp') or float('nan')):.2f} | {mins} |"
            if bpb is not None else
            f"| `{e['name']}` | {e.get('epochs') or '—'} | {e['word_ppl']:.3f} | — | "
            f"{(e.get('blimp') or float('nan')):.2f} | {mins} |")
    lines.append("")
    return lines


def _backend_table(results: dict | None) -> list[str]:
    bench = (results or {}).get("backend_benchmark") if results else None
    if not bench:
        return []
    fastest = max(bench.items(), key=lambda kv: kv[1]["tokens_per_sec"])[0]
    lines = ["### Backend throughput", "",
             "| backend | tokens/sec | seconds | verdict |",
             "|---------|-----------:|--------:|---------|"]
    for name, v in sorted(bench.items(), key=lambda kv: -kv[1]["tokens_per_sec"]):
        mark = " **fastest**" if name == fastest else ""
        lines.append(f"| `{name}` | {v['tokens_per_sec']:,.0f} | {v['seconds']:.0f} |{mark} |")
    lines.append("")
    return lines


def write_markdown(figs: list[str], results: dict | None) -> None:
    have_gpu = bool(results and results.get("backend_benchmark"))
    md = os.path.join(out_dir(), "REPORT.md")
    lines = [
        "# quark — graphical report (issue #8)",
        "",
        "> Generated by `experiments/report.py`. Every number is **MEASURED** "
        "(a training run or a cited source), **DERIVED** (computed from measured "
        "inputs), or **PROJECTED** (a scaling-law extrapolation, hatched and greyed). "
        "See the script header for the rule.",
        "",
        "**Status of the GPU panels:** "
        + ("filled from a real run on the `gpu` runner."
           if have_gpu else
           "placeholders. The author could not reach the self-hosted `gpu` runner "
           "(fork PR, pull-only access) and this machine has no GPU, so no training "
           "was executed. `experiments/gpu/` is the ready-to-run harness that fills "
           "them — see docs/ANALYSIS.md."),
        "",
    ]
    caps = {
        "fig1_perplexity.png": "Untying the shared loop (quark_3m → quark_22m) cut word "
        "perplexity 108→75 at **zero extra FLOPs**. Still 2.0× GPT-2's zero-shot 37.50 — "
        "but GPT-2 is 48× larger and plays out-of-domain.",
        "fig2_blimp_cliff.png": "**The decision chart.** Grammatical competence (BLiMP) has a "
        "cliff around 24–30M. quark scores exactly where its parameter count predicts; its BLiMP "
        "is the size class reporting in, not an anomaly to debug.",
        "fig3_stored_vs_compute.png": "The finding that reorganised the project: sharing saves "
        "**storage**, not **arithmetic**. VRAM — set by activations, i.e. compute — was always "
        "the binding constraint.",
        "fig4_token_budget.png": "quark_22m is **undertrained, not undersized**: 6.5 tokens/param, "
        "0.33× Chinchilla. The largest expected win is more passes over the data, not more "
        "parameters.",
        "fig5_vram.png": "The real 16 GB wall is the seq² attention matrix. This is why flash "
        "attention (frees ~2.25 GB) funds the batch-size and epoch experiments.",
        "fig6_roadmap.png": "The path forward, in gated order. Step 2 (4 epochs + dropout + weight "
        "decay) is the headline; everything after it is second-order until it runs.",
        "fig7_backends.png": "wgpu / vulkan / rocm throughput — the one wholly GPU-dependent result.",
        "fig8_sweep.png": "**The payoff test.** Word PPL for the NEXT.md §13 step-2 sweep against "
        "the MEASURED quark_22m baseline — the graphical answer to whether the recommended "
        "next runs actually help.",
    }
    for f in figs:
        lines += [f"## {caps.get(f, f)}", "", f"![{f}]({f})", ""]
    lines += _backend_table(results)
    lines += _metrics_table(results)
    lines += [
        "---",
        "",
        "### The one-paragraph conclusion",
        "",
        "The limiting factor is **not** the parameter count and **not** the architecture. "
        "It is the **token budget under a 16 GB VRAM ceiling**: quark_22m has seen a third of "
        "a Chinchilla-optimal number of tokens, and the ceiling is set by activation memory "
        "(which tracks compute), not by stored weights (which are abundant and cheap). The "
        "decided path (Fig 6) spends the next runs on **more tokens per parameter** (4 epochs + "
        "dropout + weight decay) and on **freeing VRAM to afford them** (flash attention, batch "
        "size), before touching anything exotic. The GPT-2 word-PPL race is kept only as a "
        "secondary metric — it is played on easy mode (in-domain vs zero-shot) and is the wrong "
        "headline (README, RESULTS.md §4.1).",
        "",
    ]
    with open(md, "w") as fh:
        fh.write("\n".join(lines))
    print(f"  wrote {os.path.relpath(md)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", help="harness results.json to overlay (optional)")
    args = ap.parse_args()

    results = None
    if args.results and os.path.exists(args.results):
        with open(args.results) as fh:
            results = json.load(fh)
        print(f"overlaying harness results from {args.results}")

    print("rendering report to docs/report/ ...")
    figs = [
        fig_perplexity(),
        fig_blimp_cliff(),
        fig_stored_vs_compute(),
        fig_token_budget(),
        fig_vram(),
        fig_roadmap(),
        fig_backends(results),
        fig_sweep(results),
    ]
    write_markdown(figs, results)
    print("done.")


if __name__ == "__main__":
    main()
