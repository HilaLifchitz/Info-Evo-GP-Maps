"""Plots for the canonical-family efficiency experiment (A,B,C,D,F,G).

  fig_canon_within.png        : within each strategy, D_eq vs tau across a
  fig_canon_positioning.png   : case means -- {info,F*} vs {tau, efficiency}
  fig_canon_efficiency_vs_a.png: eta_100 & eta_peak vs a, one line per case
  fig_canon_bars.png          : multi-metric bars at a representative a
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from adaptation_per_a import CASE_DEFS, CASE_COLORS

SCRIPT_DIR = Path(__file__).resolve().parent
SRC = SCRIPT_DIR / "results" / (sys.argv[1] if len(sys.argv) > 1
                                else f"canonical_eff_20260728")
CANON = ["A", "B", "C", "D", "F", "G"]
LABEL = {s: f"{s}: {CASE_DEFS[s][0]}" for s in CANON}


def _rho(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[m], y[m])[0]) if m.sum() >= 4 else float("nan")


def fig_within(df, clock):
    d0 = df[df.regime == clock]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2))
    for ax, s in zip(axes.ravel(), CANON):
        g = d0[d0.case == s].sort_values("a")
        sc = ax.scatter(g.D_eq, g.tau_spec, c=g.a, cmap="plasma", s=70,
                        edgecolor="k", linewidth=.4, zorder=3)
        ax.plot(g.D_eq, g.tau_spec, "-", color=CASE_COLORS[s], lw=1.2, alpha=.6)
        ax.set_yscale("log")
        ax.set_title(f"{LABEL[s]}\n"
                     rf"$\rho(D_{{eq}},\tau)$ across $a$ = {_rho(g.D_eq.values, g.tau_spec.values):+.2f}",
                     fontsize=10)
        ax.set_xlabel(r"$D_{eq}$ info (bits)")
        ax.set_ylabel(r"$\tau_{spec}$ (gen, log)")
        cb = fig.colorbar(sc, ax=ax); cb.set_label("selection strength a", fontsize=8)
    fig.suptitle("Within each strategy: as selection strength a grows, does more "
                 f"information mean faster relaxation?\nclock {clock}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(SRC / "fig_canon_within.png", dpi=130)
    print("wrote", SRC / "fig_canon_within.png")


def fig_positioning(df):
    def ms(col):
        g = df.groupby("case")[col]
        return g.mean(), g.sem()
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 10))

    def panel(a, xcol, ycol, xl, yl):
        mx, sx = ms(xcol); my, sy = ms(ycol)
        for s in CANON:
            if s in mx.index:
                a.errorbar(mx[s], my[s], xerr=sx[s], yerr=sy[s], fmt="o", ms=13,
                           c=CASE_COLORS[s], capsize=4, label=LABEL[s])
        a.set_xlabel(xl); a.set_ylabel(yl)

    panel(ax[0, 0], "D_eq", "tau_spec", r"mean $D_{eq}$ (info)",
          r"mean $\tau_{spec}$ (slow $\to$)")
    ax[0, 0].set_yscale("log"); ax[0, 0].set_title("(i) info vs relaxation time")
    panel(ax[0, 1], "Fstar", "tau_spec", r"mean $F^*$ (free fitness)",
          r"mean $\tau_{spec}$ (slow $\to$)")
    ax[0, 1].set_yscale("log"); ax[0, 1].set_title("(ii) free fitness vs relaxation time")
    panel(ax[1, 0], "D_eq", "eta_100", r"mean $D_{eq}$ (info)",
          r"mean $\eta_{100}$ (efficient $\uparrow$)")
    ax[1, 0].set_title(r"(iii) info vs efficiency $\eta$")
    panel(ax[1, 1], "Fstar", "eta_100", r"mean $F^*$ (free fitness)",
          r"mean $\eta_{100}$ (efficient $\uparrow$)")
    ax[1, 1].set_title(r"(iv) free fitness vs efficiency $\eta$")
    ax[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Strategy positioning: information / free fitness  vs  speed / "
                 "efficiency\n(case means +/- SEM, pooled over 4 clocks)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(SRC / "fig_canon_positioning.png", dpi=130)
    print("wrote", SRC / "fig_canon_positioning.png")


def fig_eff_vs_a(df):
    clocks = list(df.regime.unique())
    fig, axes = plt.subplots(2, len(clocks), figsize=(4.6 * len(clocks), 8.4),
                             squeeze=False)
    for j, clk in enumerate(clocks):
        d0 = df[df.regime == clk]
        for row, (col, lab) in enumerate([("eta_100", r"$\eta_{100}$"),
                                          ("eta_peak", r"peak $\eta$")]):
            ax = axes[row][j]
            for s in CANON:
                g = d0[d0.case == s].sort_values("a")
                ax.plot(g.a, g[col], "-o", ms=4, color=CASE_COLORS[s],
                        label=LABEL[s] if (row == 0 and j == 0) else None)
            ax.set_xlabel("selection strength a"); ax.set_ylabel(lab)
            ax.grid(True, alpha=.3)
            if row == 0:
                ax.set_title(clk.replace("N30_", ""), fontsize=9)
    axes[0][0].legend(fontsize=7, loc="best")
    fig.suptitle("Efficiency vs selection strength, per strategy and clock "
                 r"($\eta$ = info gained / (N·cost paid))", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(SRC / "fig_canon_efficiency_vs_a.png", dpi=130)
    print("wrote", SRC / "fig_canon_efficiency_vs_a.png")


def fig_bars(df, clock, a_rep):
    d0 = df[(df.regime == clock)].copy()
    a_rep = float(d0.a.iloc[(d0.a - a_rep).abs().argmin()])
    d0 = d0[np.isclose(d0.a, a_rep)]
    metrics = [("D_eq", "info (bits)"), ("Fstar", "free fitness F*"),
               ("mlf", "mean log-fit"), ("tau_spec", r"$\tau$ (gen)"),
               ("eta_100", r"$\eta_{100}$"), ("eta_peak", r"peak $\eta$")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.5 * len(metrics), 4.2))
    for ax, (col, lab) in zip(axes, metrics):
        vals = [d0[d0.case == s][col].mean() for s in CANON]
        ax.bar(range(len(CANON)), vals, color=[CASE_COLORS[s] for s in CANON])
        ax.set_xticks(range(len(CANON))); ax.set_xticklabels(CANON, fontsize=9)
        ax.set_title(lab, fontsize=9)
    fig.suptitle(f"Strategy comparison at a={a_rep:g}, clock {clock}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(SRC / "fig_canon_bars.png", dpi=130)
    print("wrote", SRC / "fig_canon_bars.png")


def main():
    df = pd.read_csv(SRC / "canonical_eff.csv")
    clocks = list(df.regime.unique())
    fig_within(df, clocks[0])
    fig_positioning(df)
    fig_eff_vs_a(df)
    fig_bars(df, clocks[0], 0.18)

    print("\n=== efficiency ranking (mean eta_100, pooled clocks) ===")
    print(df.groupby("case")[["eta_100", "eta_peak", "D_eq", "Fstar",
                              "tau_spec", "mlf"]].mean().round(4)
          .reindex(CANON))


if __name__ == "__main__":
    main()
