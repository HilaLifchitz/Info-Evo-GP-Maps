#!/usr/bin/env python3
r"""Combined figure: the A-F case-ranking panels (a, b, c) next to the
peak-efficiency sketch (d), under a single shared strategy legend.

Left  (a,b,c): from bump_ranking_AF.py -- for each of three metrics
               (information, mean log-fitness, information-fitness score F):
               top = metric value + A-F spread band, bottom = rank/bump chart
               (marker size = gap to nearest case = significance).
Right (d)    : from surprise_analysis_split.py -- peak efficiency eta vs
               selection strength a.

A single big legend (thick coloured lines) sits above everything; the
efficiency panel's own legend is dropped in favour of it.

Data:
  results/ordering_AG_20260729/sweep_AF_newletters_uv005_r0.csv   (ranking)
  results/canon_dynamics_a_grid/curves.npz                        (efficiency)
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptation_per_a import CASE_COLORS, CASE_DEFS, CASE_ORDER

SCRIPT_DIR = Path(__file__).resolve().parent
RANK_DATA = SCRIPT_DIR / "results" / "ordering_AG_20260729" / "sweep_AF_newletters_uv005_r0.csv"
EFF_DATA = SCRIPT_DIR / "results" / "canon_dynamics_a_grid" / "curves.npz"
OUT = SCRIPT_DIR / "results" / "ordering_AG_20260729" / "fig_ranking_eff_combined.png"

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

MINI_TITLE_FS = 15      # "mini title" font (compact but still large)
LETTER_FS = 20          # bold a) b) c) d)
LEGEND_FS = 16          # shared strategy legend


def sleek():
    plt.rcParams.update({
        "font.size": 11, "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.7,
        "axes.linewidth": 1.1, "legend.frameon": False,
    })


def panel_letter(ax, letter):
    ax.text(-0.06, 1.14, f"{letter})", transform=ax.transAxes,
            fontsize=LETTER_FS, fontweight="bold", va="top", ha="left")


def draw_ranking(fig, gs, df):
    """Draw the three metric columns (value + rank) into gs[:, 0:3]."""
    x = np.sort(df["a"].unique())
    letters = "abc"
    for col, (mbase, mlab, unit) in enumerate(BASE_METRICS):
        mcol = f"{mbase}_{MODEL}"
        p = df.pivot_table(index="a", columns="case", values=mcol)[DATA_ORDER]
        xv = p.index.values
        vals = p.values                                    # (n_a, 6) in DATA_ORDER
        rng = np.nanmax(vals) - np.nanmin(vals)
        rng = rng if rng > 0 else 1.0

        # ---- top: value curves + A-F spread band ----
        axv = fig.add_subplot(gs[0, col])
        lo = np.nanmin(vals, axis=1)
        hi = np.nanmax(vals, axis=1)
        axv.fill_between(xv, lo, hi, color="0.80", alpha=0.55, lw=0, zorder=0)
        for dl in PLOT_ORDER:
            axv.plot(xv, p[dl].values, "-", lw=2.2, color=CASE_COLORS[dl],
                     marker="o", ms=5, mec="white", mew=0.7, zorder=3)
        sp = hi - lo
        axv.set_title(mlab, fontsize=MINI_TITLE_FS, fontweight="bold", pad=8)
        axv.set_ylabel(f"value  [{unit}]")
        axv.text(0.02, 0.97,
                 f"A–F spread: {100*sp.min()/rng:.0f}–{100*sp.max()/rng:.0f}% of range",
                 transform=axv.transAxes, va="top", ha="left", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
        axv.tick_params(labelbottom=False)
        panel_letter(axv, letters[col])

        # ---- bottom: rank / bump chart, marker size = gap to nearest ----
        axr = fig.add_subplot(gs[1, col], sharex=axv)
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
        axr.set_xlabel("selection strength  a", fontsize=14)


def draw_efficiency(fig, gs):
    """Draw the peak-efficiency panel (d) into gs[:, 3] (no own legend)."""
    d = np.load(EFF_DATA)
    t = d["times"].astype(float)
    ti = {int(tt): i for i, tt in enumerate(t)}

    # discover the a-grid
    avals = set()
    for k in d.files:
        parts = k.split("_")
        if len(parts) >= 3 and parts[-1] == "info":
            try:
                avals.add(float(parts[0]))
            except ValueError:
                pass
    AVALS = sorted(avals)

    rows = []
    for a in AVALS:
        for s in CASE_ORDER:
            eta = d[f"{a}_{s}_eta"]
            i200 = ti.get(200, int(np.argmin(np.abs(t - 200))))
            pk = int(np.nanargmax(eta[: i200 + 1]))
            rows.append(dict(a=a, case=s, eta_peak=float(eta[pk])))
    dfe = pd.DataFrame(rows)

    ax = fig.add_subplot(gs[:, 3])
    for s in CASE_ORDER:
        g = dfe[dfe.case == s].sort_values("a")
        ax.plot(g.a, g.eta_peak, "-o", ms=5.5, color=CASE_COLORS[s], lw=2)
    ax.set_xlabel("selection strength  a", fontsize=14)
    ax.set_ylabel(r"peak efficiency $\eta$")
    ax.set_title("stronger selection wastes more\n"
                 r"peak $\eta$ DROPS as $a$ rises",
                 fontsize=MINI_TITLE_FS, fontweight="bold", pad=8)
    ax.grid(True, alpha=.3)
    panel_letter(ax, "d")


def main():
    sleek()
    df = pd.read_csv(RANK_DATA).sort_values("a")
    amax = float(np.sort(df["a"].unique()).max())

    fig = plt.figure(figsize=(21, 8.6))
    gs = fig.add_gridspec(2, 4, left=0.045, right=0.99, top=0.80, bottom=0.11,
                          height_ratios=[1.25, 1.0], width_ratios=[1, 1, 1, 1.4],
                          hspace=0.14, wspace=0.30)

    draw_ranking(fig, gs, df)
    draw_efficiency(fig, gs)

    # ---- shared strategy legend: thick coloured horizontal lines on top ----
    case_handles = [plt.Line2D([0], [0], color=CASE_COLORS[dl], lw=5.0,
                               label=f"{DISP[dl]} — {CASE_DEFS[dl][0]}")
                    for dl in PLOT_ORDER]
    fig.legend(handles=case_handles, loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 0.955), fontsize=LEGEND_FS,
               columnspacing=1.6, handlelength=2.2)

    # ---- marker-size legend under the ranking columns ----
    size_handles = [plt.scatter([], [], s=30 + 900 * v, color="0.5",
                                edgecolors="white", label=f"{int(v*100)}%")
                    for v in (0.02, 0.15, 0.35)]
    size_leg = fig.legend(handles=size_handles, loc="lower center", ncol=3,
                          bbox_to_anchor=(0.37, -0.06), fontsize=15,
                          title="marker size = gap to nearest case (% of range) = significance")
    size_leg.get_title().set_fontsize(15)

    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
