#!/usr/bin/env python3
r"""Case ranking vs selection strength for the six iso-peak strategies A-F.

Recreation of the A-F "case ranking" figure (top = metric value with the A-F
spread band, bottom = rank/bump chart with marker size = gap-to-nearest-case),
for three metrics: information, mean log-fitness, and the information-fitness
score F.

Display convention: the E<->F letters are swapped relative to the raw data, so
the shown letters are  E = neutral intermediates,  F = reciprocal sign
(colours track the landscape type, so reciprocal stays green, neutral purple).

Data: results/ordering_AG_20260729/sweep_AF_newletters_uv005_r0.csv
      (diffusion / detailed-balance, N=30, u=v=0.005, r=0).
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
DATA = SCRIPT_DIR / "results" / "ordering_AG_20260729" / "sweep_AF_newletters_uv005_r0.csv"
OUT = DATA.parent / "fig_ranking_AF.png"
N = 30
MODEL = "diff"                                   # diffusion columns *_diff

DATA_ORDER = ["A", "B", "C", "D", "E", "F"]      # raw data letters
# display relabel: swap E<->F (E = neutral, F = reciprocal); colours unchanged
DISP = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "F", "F": "E"}
PLOT_ORDER = ["A", "B", "C", "D", "F", "E"]      # iterate so shown letters read A..F

BASE_METRICS = [
    ("info", "information", "bits"),
    ("mlf",  "mean log-fitness", "nats"),
    ("F",    "information–fitness score $F$", "nats"),
]


def sleek():
    plt.rcParams.update({
        "font.size": 11, "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.7,
        "axes.linewidth": 1.1, "legend.frameon": False,
    })


def main():
    sleek()
    df = pd.read_csv(DATA).sort_values("a")
    x = np.sort(df["a"].unique())
    amax = float(x.max())

    fig, axs = plt.subplots(2, 3, figsize=(15, 8.4), sharex=True,
                            gridspec_kw=dict(height_ratios=[1.25, 1.0],
                                             hspace=0.12, wspace=0.24))

    for col, (mbase, mlab, unit) in enumerate(BASE_METRICS):
        mcol = f"{mbase}_{MODEL}"
        p = df.pivot_table(index="a", columns="case", values=mcol)[DATA_ORDER]
        xv = p.index.values
        vals = p.values                                    # (n_a, 6) in DATA_ORDER
        rng = np.nanmax(vals) - np.nanmin(vals)
        rng = rng if rng > 0 else 1.0

        # ---- top: value curves + A-F spread band ----
        axv = axs[0][col]
        lo = np.nanmin(vals, axis=1)
        hi = np.nanmax(vals, axis=1)
        axv.fill_between(xv, lo, hi, color="0.80", alpha=0.55, lw=0, zorder=0)
        for dl in PLOT_ORDER:
            axv.plot(xv, p[dl].values, "-", lw=2.2, color=CASE_COLORS[dl],
                     marker="o", ms=5, mec="white", mew=0.7, zorder=3)
        sp = hi - lo
        axv.set_title(mlab, fontsize=13, fontweight="bold", pad=8)
        axv.set_ylabel(f"value  [{unit}]")
        axv.text(0.02, 0.97,
                 f"A–F spread: {100*sp.min()/rng:.0f}–{100*sp.max()/rng:.0f}% of range",
                 transform=axv.transAxes, va="top", ha="left", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))

        # ---- bottom: rank / bump chart, marker size = gap to nearest ----
        axr = axs[1][col]
        order = np.argsort(-vals, axis=1)
        rank = np.empty_like(vals)
        for k in range(len(xv)):
            for rpos, idx in enumerate(order[k]):
                rank[k, idx] = rpos + 1
        sig = np.zeros_like(vals)
        for k in range(len(xv)):
            for ci in range(len(DATA_ORDER)):
                others = np.delete(vals[k], ci)
                sig[k, ci] = np.min(np.abs(vals[k, ci] - others)) / rng
        smark = 30 + 900 * sig
        di = {dl: DATA_ORDER.index(dl) for dl in DATA_ORDER}
        for pos, dl in enumerate(PLOT_ORDER):
            ci = di[dl]
            jit = (pos - 2.5) * 0.05
            axr.plot(xv, rank[:, ci] + jit, "-", lw=1.6, color=CASE_COLORS[dl],
                     alpha=0.55, zorder=2)
            axr.scatter(xv, rank[:, ci] + jit, s=smark[:, ci],
                        color=CASE_COLORS[dl], edgecolors="white",
                        linewidths=0.7, zorder=3)
        axr.set_ylim(6.6, 0.4)
        axr.set_yticks([1, 2, 3, 4, 5, 6])
        axr.set_ylabel("rank (1 = highest)")
        axr.set_xlabel("selection strength  a")

    # ---- legends ----
    case_handles = [plt.Line2D([0], [0], color=CASE_COLORS[dl], lw=2.4, marker="o",
                               mec="white", ms=7,
                               label=f"{DISP[dl]} — {CASE_DEFS[dl][0]}")
                    for dl in PLOT_ORDER]
    fig.legend(handles=case_handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.995), fontsize=11)
    size_handles = [plt.scatter([], [], s=30 + 900 * v, color="0.5",
                                edgecolors="white", label=f"{int(v*100)}%")
                    for v in (0.02, 0.15, 0.35)]
    fig.legend(handles=size_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=9,
               title="marker size = gap to nearest case (% of range) = significance")

    fig.suptitle(f"$N = {N},\\ u = v = 0.005,\\ r = 0,\\ a \\leq {amax:g}$",
                 y=1.02, fontsize=13)

    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
