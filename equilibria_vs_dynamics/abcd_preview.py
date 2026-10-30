"""Corrected Panel A & Panel C using the CANONICAL families A,B,C,D (real D =
neutral intermediates), from the existing AE-family rows.  No grid buckets, no
'11-not-peak', no E.  (Efficiency eta + F,G arrive from the queued canonical run.)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from adaptation_per_a import CASE_DEFS, CASE_COLORS

SRC = Path(__file__).resolve().parent / "results" / "info_speed_20260727"
CASES = ["A", "B", "C", "D"]
LABEL = {s: f"{s}: {CASE_DEFS[s][0]}" for s in CASES}
CLK_MARK = {}          # filled at runtime


def _rho(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[m], y[m])[0]) if m.sum() >= 4 else float("nan")


def main():
    df = pd.read_csv(SRC / "info_speed.csv")
    ae = df[(df.family == "AE") & (df.case.isin(CASES))].copy()
    clocks = list(ae.regime.unique())
    marks = ["o", "s", "^", "D"]
    for c, mk in zip(clocks, marks):
        CLK_MARK[c] = mk

    # ---- Panel A (within strategy): D_eq vs tau across a & clocks ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    for ax, s in zip(axes.ravel(), CASES):
        g = ae[ae.case == s]
        for clk in clocks:
            gg = g[g.regime == clk]
            sc = ax.scatter(gg.D_eq, gg.tau_spec, c=gg.a, cmap="plasma",
                            marker=CLK_MARK[clk], s=90, edgecolor="k",
                            linewidth=.4, vmin=0.05, vmax=0.2)
        ax.set_yscale("log")
        ax.set_title(f"{LABEL[s]}\n"
                     rf"$\rho(D_{{eq}},\tau)$ (pooled) = {_rho(g.D_eq.values, g.tau_spec.values):+.2f}",
                     fontsize=10)
        ax.set_xlabel(r"$D_{eq}$ information (bits)")
        ax.set_ylabel(r"$\tau_{spec}$ relaxation time (gen, log)")
    cb = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=.6)
    cb.set_label("selection strength a")
    hm = [plt.Line2D([], [], marker=CLK_MARK[c], ls="", mec="k", mfc="0.7",
                     label=c.replace("N30_", "")) for c in clocks]
    axes[0, 0].legend(handles=hm, fontsize=7, title="clock", loc="upper left")
    fig.suptitle("Panel A (corrected): within each canonical strategy A,B,C,D "
                 "(D = neutral intermediates)\nmarker = clock, colour = a",
                 fontsize=12)
    fig.savefig(SRC / "fig_ABCD_within.png", dpi=130, bbox_inches="tight")
    print("wrote", SRC / "fig_ABCD_within.png")

    # ---- Panel C (positioning): case means, info/F* vs tau ----
    def ms(col):
        gg = ae.groupby("case")[col]
        return gg.mean(), gg.sem()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    for a_, xcol, xl in [(ax[0], "D_eq", r"mean $D_{eq}$ (info, bits)"),
                         (ax[1], "Fstar", r"mean $F^*$ (free fitness)")]:
        mx, sx = ms(xcol); my, sy = ms("tau_spec")
        for s in CASES:
            a_.errorbar(mx[s], my[s], xerr=sx[s], yerr=sy[s], fmt="o", ms=14,
                        c=CASE_COLORS[s], capsize=4, label=LABEL[s])
        a_.set_yscale("log"); a_.set_xlabel(xl)
        a_.set_ylabel(r"mean $\tau_{spec}$ (slow $\to$, log)")
    ax[0].set_title("(i) information vs relaxation time")
    ax[1].set_title("(ii) free fitness vs relaxation time")
    ax[0].legend(fontsize=9)
    fig.suptitle("Panel C (corrected): canonical strategies A,B,C,D  "
                 "(efficiency panels arrive from queued canonical run)\n"
                 "points = case means +/- SEM, pooled over 4 clocks", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(SRC / "fig_ABCD_positioning.png", dpi=130)
    print("wrote", SRC / "fig_ABCD_positioning.png")

    print("\ncase means (pooled clocks):")
    print(ae.groupby("case")[["D_eq", "tau_spec", "t95_info", "Fstar", "mlf"]]
          .mean().round(3).reindex(CASES))


if __name__ == "__main__":
    main()
