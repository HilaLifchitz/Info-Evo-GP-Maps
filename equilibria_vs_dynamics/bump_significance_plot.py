#!/usr/bin/env python3
r"""Sleek bump + value-range plots for A..D (E excluded), for any varied axis
(a / u=v / r) and either model (diffusion or 2-loci Wright-Fisher chain).

Top row : metric values vs axis, shaded band = A-D spread.
Bottom  : bump / rank chart; marker size = gap to nearest other case (as % of
          the metric range) = significance of that rank.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptation_per_a import CASE_COLORS, CASE_DEFS

SCRIPT_DIR = Path(__file__).resolve().parent
DATA = SCRIPT_DIR / "results" / "big_sweep_20260719" / "sweep_data.csv"
OUT = DATA.parent / "analysis"
CASES = ["A", "B", "C", "D"]
N = 30
MODELS = {"diff": "diffusion (detailed-balance)",
          "chain": "2-loci Wright–Fisher chain"}
BASE_METRICS = [("info", "information", "bits"),
                ("mlf", "mean log-fitness", "nats"),
                ("F", "free fitness F", "nats")]

# axis spec: key -> (column, fixed params, log-x?, x-label, restrict, amax)
AXES = {
    "a": dict(col="a", fixed=dict(u=0.005, v=0.005, r=0.0), logx=False,
              xlabel="selection strength  a", restrict={}, xmax=0.30),
    "u": dict(col="u", fixed=dict(a=0.17, r=0.0), logx=True,
              xlabel="mutation rate  u = v", restrict=dict(mut_kind="sym"),
              xmax=None),
    "r": dict(col="r", fixed=dict(a=0.17, u=0.005, v=0.005), logx=False,
              xlabel="recombination rate  r", restrict={}, xmax=None),
}


def sleek():
    plt.rcParams.update({
        "font.size": 11, "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.7,
        "axes.linewidth": 1.1, "legend.frameon": False,
    })


def make_plot(df, axis_key, model):
    spec = AXES[axis_key]
    xcol = spec["col"]
    s = df.copy()
    for k, v in spec["fixed"].items():
        s = s[np.isclose(s[k], v)]
    for k, v in spec["restrict"].items():
        s = s[s[k] == v]
    if spec["xmax"] is not None:
        s = s[s[xcol] <= spec["xmax"] + 1e-9]
    s = s.sort_values(xcol)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8.4), sharex=True,
                            gridspec_kw=dict(height_ratios=[1.25, 1.0],
                                             hspace=0.12, wspace=0.24))
    for col, (mbase, mlab, unit) in enumerate(BASE_METRICS):
        mcol = f"{mbase}_{model}"
        p = s.pivot_table(index=xcol, columns="case", values=mcol)[CASES]
        x = p.index.values
        vals = p.values
        rng = np.nanmax(vals) - np.nanmin(vals)
        rng = rng if rng > 0 else 1.0

        axv = axs[0][col]
        lo = np.nanmin(vals, axis=1); hi = np.nanmax(vals, axis=1)
        axv.fill_between(x, lo, hi, color="0.80", alpha=0.55, lw=0, zorder=0)
        for c in CASES:
            axv.plot(x, p[c].values, "-", lw=2.2, color=CASE_COLORS[c],
                     marker="o", ms=5, mec="white", mew=0.7, zorder=3)
        sp = hi - lo
        axv.set_title(mlab, fontsize=13, fontweight="bold", pad=8)
        axv.set_ylabel(f"value  [{unit}]")
        axv.text(0.02, 0.97,
                 f"A–D spread: {100*sp.min()/rng:.0f}–{100*sp.max()/rng:.0f}% of range",
                 transform=axv.transAxes, va="top", ha="left", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
        if spec["logx"]:
            axv.set_xscale("log")

        axr = axs[1][col]
        order = np.argsort(-vals, axis=1)
        rank = np.empty_like(vals)
        for k in range(len(x)):
            for rpos, idx in enumerate(order[k]):
                rank[k, idx] = rpos + 1
        sig = np.zeros_like(vals)
        for k in range(len(x)):
            for ci in range(4):
                others = np.delete(vals[k], ci)
                sig[k, ci] = np.min(np.abs(vals[k, ci] - others)) / rng
        smark = 30 + 900 * sig
        for ci, c in enumerate(CASES):
            jit = (ci - 1.5) * 0.045
            axr.plot(x, rank[:, ci] + jit, "-", lw=1.6, color=CASE_COLORS[c],
                     alpha=0.55, zorder=2)
            axr.scatter(x, rank[:, ci] + jit, s=smark[:, ci],
                        color=CASE_COLORS[c], edgecolors="white",
                        linewidths=0.7, zorder=3)
        axr.set_ylim(4.6, 0.4); axr.set_yticks([1, 2, 3, 4])
        axr.set_ylabel("rank (1 = highest)")
        axr.set_xlabel(spec["xlabel"])
        if spec["logx"]:
            axr.set_xscale("log")

    case_handles = [plt.Line2D([0], [0], color=CASE_COLORS[c], lw=2.4, marker="o",
                               mec="white", ms=7, label=f"{c} — {CASE_DEFS[c][0]}")
                    for c in CASES]
    fig.legend(handles=case_handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.995), fontsize=11)
    size_handles = [plt.scatter([], [], s=30 + 900 * v, color="0.5",
                                edgecolors="white", label=f"{int(v*100)}%")
                    for v in (0.02, 0.15, 0.35)]
    fig.legend(handles=size_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=9,
               title="marker size = gap to nearest case (% of range) = significance")

    fixed_txt = ",  ".join(f"{k} = {v:g}" for k, v in spec["fixed"].items())
    fig.suptitle("Case ranking vs "
                 f"{spec['xlabel'].split('  ')[0].strip()} — A,B,C,D (E excluded)\n"
                 f"{MODELS[model]},  N = {N},  {fixed_txt}   "
                 "· big markers = significant rank, small = within noise",
                 y=1.075, fontsize=13)
    fn = OUT / f"fig_bump_ranges_noE_{axis_key}_{model}.png"
    fig.savefig(fn, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fn)


def main():
    sleek()
    df = pd.read_csv(DATA)
    OUT.mkdir(exist_ok=True)
    for model in ("diff", "chain"):
        for axis_key in ("a", "u", "r"):
            make_plot(df, axis_key, model)


if __name__ == "__main__":
    main()
