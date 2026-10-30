#!/usr/bin/env python3
r"""Significance-aware displays of A..D ordering (E excluded).

Goal: make it visually obvious that
  - mean log-fitness barely separates A..D (near-degenerate ribbon), whereas
  - information and free fitness give wide, meaningful gaps -> a real ordering.

Outputs (results/big_sweep_20260719/analysis/):
  fig_gaps_noE.png        value curves A..D vs a with a shaded min-max band per
                          metric; band width = how separated the cases are.
  fig_relspread_noE.png   normalized spread (max-min)/scale vs a for the three
                          metrics on one axis -> mlf sits near 0, info/F high.
  fig_bump_ties_noE.png   tie-aware rank charts: cases whose gap is below a
                          significance threshold share a rank (drawn merged).
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
CASES = ["A", "B", "C", "D"]                       # E excluded
METRICS = [("info_diff", "information [bits]"),
           ("mlf_diff", "mean log-fitness"),
           ("F_diff", "free fitness F [nats]")]
TIE_FRAC = 0.02          # cases within 2% of the metric's global scale = tied


def slice_df(df, u=0.005, v=0.005, r=0.0):
    s = df[(np.isclose(df.u, u)) & (np.isclose(df.v, v)) & (np.isclose(df.r, r))]
    return s.sort_values("a")


def pivot(s, metric):
    return s.pivot_table(index="a", columns="case", values=metric)[CASES]


def fig_gaps(df):
    s = slice_df(df)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (mcol, mlab) in zip(axs, METRICS):
        p = pivot(s, mcol)
        a = p.index.values
        lo = p.min(axis=1).values
        hi = p.max(axis=1).values
        ax.fill_between(a, lo, hi, color="0.75", alpha=0.5, zorder=0,
                        label="A–D spread")
        for c in CASES:
            ax.plot(a, p[c].values, "-o", ms=4, lw=1.8, color=CASE_COLORS[c],
                    label=c)
        # relative spread annotation at a mid point
        scale = np.nanmax(p.values) - np.nanmin(p.values)
        mid = len(a) // 2
        rel = (hi[mid] - lo[mid]) / scale * 100
        ax.set_title(f"{mlab}\nA–D spread at a={a[mid]:.2f} = "
                     f"{rel:.1f}% of full range", fontsize=10)
        ax.set_xlabel("selection a")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("A–D only (E excluded): grey band = separation between the four "
                 "cases  (u=v=0.005, r=0, diffusion)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig_gaps_noE.png", dpi=150)
    plt.close(fig)


def fig_relspread(df):
    s = slice_df(df)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    styles = {"info_diff": ("#1f77b4", "-o"),
              "mlf_diff": ("#d62728", "-s"),
              "F_diff": ("#2ca02c", "-^")}
    for mcol, mlab in METRICS:
        p = pivot(s, mcol)
        a = p.index.values
        scale = np.nanmax(p.values) - np.nanmin(p.values)
        rel = (p.max(axis=1).values - p.min(axis=1).values) / scale
        col, mk = styles[mcol]
        ax.plot(a, rel, mk, color=col, lw=2, label=mlab)
    ax.axhspan(0, TIE_FRAC, color="0.85", alpha=0.7,
               label=f"'tied' zone (<{TIE_FRAC*100:.0f}% of range)")
    ax.set_xlabel("selection a")
    ax.set_ylabel("normalized A–D spread\n(max−min) / (full metric range)")
    ax.set_title("How separated are A,B,C,D?  (u=v=0.005, r=0, diffusion)\n"
                 "mean log-fitness stays near the tied zone; info & F are well "
                 "separated")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_relspread_noE.png", dpi=150)
    plt.close(fig)


def tie_ranks_agree(vals_d: dict, vals_c: dict):
    """Ranks with ties where 'tie' = the two INDEPENDENT models (diffusion &
    chain) disagree on the sign of the gap -> the order is within method noise.
    Sort by diffusion value; merge adjacent cases if the chain flips their order.
    Returns {case: rank(float)} (tied block gets the block's mean rank)."""
    order = sorted(CASES, key=lambda c: -vals_d[c])
    blocks = [[order[0]]]
    for prev, c in zip(order, order[1:]):
        # do both methods agree that prev >= c ?
        agree = (vals_d[prev] - vals_d[c]) * (vals_c[prev] - vals_c[c]) > 0
        if agree:
            blocks.append([c])
        else:
            blocks[-1].append(c)                       # merge: methods disagree
    ranks, pos = {}, 1
    for blk in blocks:
        shared = pos + (len(blk) - 1) / 2.0
        for c in blk:
            ranks[c] = shared
        pos += len(blk)
    return ranks


def fig_bump_ties(df):
    s = slice_df(df)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (mcol, mlab) in zip(axs, METRICS):
        pd_ = pivot(s, mcol)
        pc_ = pivot(s, mcol.replace("_diff", "_chain"))
        a = pd_.index.values
        series = {c: [] for c in CASES}
        for ai in a:
            r = tie_ranks_agree({c: pd_.loc[ai, c] for c in CASES},
                                {c: pc_.loc[ai, c] for c in CASES})
            for c in CASES:
                series[c].append(r[c])
        for c in CASES:
            jit = (CASES.index(c) - 1.5) * 0.04
            ax.plot(a, np.array(series[c]) + jit, "-o", ms=5, lw=2,
                    color=CASE_COLORS[c], label=c, alpha=0.9)
        ax.set_ylim(4.6, 0.4)
        ax.set_yticks([1, 2, 3, 4])
        ax.set_xlabel("selection a")
        ax.set_title(mlab, fontsize=10)
        ax.grid(True, alpha=0.3)
        if mcol == "info_diff":
            ax.set_ylabel("significant rank (1=highest)\ntied = the two models disagree")
    handles = [plt.Line2D([0], [0], color=CASE_COLORS[c], marker="o", lw=2,
                          label=f"{c}: {CASE_DEFS[c][0]}") for c in CASES]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Significance by cross-model agreement (tie = diffusion & chain "
                 "disagree on order), E excluded  (u=v=0.005, r=0)\n"
                 "merged flat lines = ordering is within method noise "
                 "(mean log-fitness); separated lines = real ordering (info, F)",
                 y=0.97, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT / "fig_bump_ties_noE.png", dpi=150)
    plt.close(fig)


def fig_dominance(df):
    """Pairwise dominance consistency among A..D over ALL params (diffusion).
    Cell (i,j) = P(case i > case j).  ~1 or ~0 = decisive ordering;
    ~0.5 = ambiguous (no meaningful order)."""
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2))
    decis = {}
    for ax, (mcol, mlab) in zip(axs, METRICS):
        # wide table: one row per param point, columns = case values
        w = df.pivot_table(index=["a", "u", "v", "r"], columns="case",
                           values=mcol)[CASES].dropna()
        M = np.full((4, 4), np.nan)
        for i, ci in enumerate(CASES):
            for j, cj in enumerate(CASES):
                if i == j:
                    continue
                M[i, j] = float((w[ci] > w[cj]).mean())
        # decisiveness: mean over pairs of |2P-1|
        pairs = [M[i, j] for i in range(4) for j in range(4) if i != j]
        decis[mlab] = np.mean([abs(2 * p - 1) for p in pairs])
        im = ax.imshow(M, cmap="RdBu", vmin=0, vmax=1)
        ax.set_xticks(range(4)); ax.set_xticklabels(CASES)
        ax.set_yticks(range(4)); ax.set_yticklabels(CASES)
        ax.set_xlabel("loses to  →"); ax.set_ylabel("row beats col")
        for i in range(4):
            for j in range(4):
                if i != j:
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=10,
                            color="white" if abs(M[i, j] - 0.5) > 0.35 else "k")
        ax.set_title(f"{mlab}\ndecisiveness = {decis[mlab]:.2f}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, label="P(row > col)")
    fig.suptitle("Pairwise dominance P(row beats col) over ALL params, A..D "
                 "(diffusion). 0.5 (white) = ambiguous, 0/1 = decisive.\n"
                 "mean log-fitness ≈ 0.5 everywhere (no real order); info & F "
                 "are decisive", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / "fig_dominance_noE.png", dpi=150)
    plt.close(fig)
    return decis


def main():
    df = pd.read_csv(DATA)
    OUT.mkdir(exist_ok=True)
    fig_gaps(df)
    fig_relspread(df)
    fig_bump_ties(df)
    decis = fig_dominance(df)
    print("wrote fig_gaps_noE.png, fig_relspread_noE.png, fig_bump_ties_noE.png, "
          "fig_dominance_noE.png")
    print("decisiveness (0=random order, 1=perfectly decisive):")
    for k, v in decis.items():
        print(f"  {k:20s}: {v:.3f}")


if __name__ == "__main__":
    main()
