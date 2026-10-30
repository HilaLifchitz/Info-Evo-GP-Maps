"""The two iso-peak efficiency sketches, each as its own standalone figure.

Same content as ``surprise_analysis_2panel.py`` but written to TWO separate
files so each sketch can be placed independently (e.g. the peak-efficiency
sketch next to the ranking figure, under a joined caption):

  fig_peak_efficiency.png : peak efficiency eta vs selection strength a
  fig_info_vs_cost.png    : information D(t) vs cumulative selective cost

Data: results/canon_dynamics_a_grid/curves.npz  (A-F lettering, denser a-grid).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from adaptation_per_a import CASE_DEFS, CASE_COLORS, CASE_ORDER

SRC = Path(__file__).resolve().parent / "results" / "canon_dynamics_a_grid"
CASES = list(CASE_ORDER)  # A..F (data keys)
# display relabel: swap E<->F letters (E = neutral, F = reciprocal); colours unchanged
DISP = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "F", "F": "E"}
PLOT_ORDER = ["A", "B", "C", "D", "F", "E"]     # iterate so legend reads A..F
LABEL = {s: f"{DISP[s]}: {CASE_DEFS[s][0]}" for s in CASES}
N = 30
REF_A_PREF = 0.18   # reference a for the info-vs-cost sketch


def first_cross(t, y, lvl):
    """Linear-interpolated first time y crosses level lvl."""
    a = y >= lvl
    if not a.any():
        return np.nan
    k = int(np.argmax(a))
    if k == 0:
        return float(t[0])
    return float(t[k - 1] + (lvl - y[k - 1]) * (t[k] - t[k - 1]) / (y[k] - y[k - 1]))


def discover_avals(d) -> list[float]:
    if "avals" in d.files:
        return [float(x) for x in d["avals"]]
    avals = set()
    for k in d.files:
        parts = k.split("_")
        if len(parts) >= 3 and parts[-1] == "info":
            try:
                avals.add(float(parts[0]))
            except ValueError:
                pass
    return sorted(avals)


def strategy_legend(ax):
    handles = [plt.Line2D([0], [0], color=CASE_COLORS[s], lw=2.4, marker="o",
                          ms=6, label=LABEL[s]) for s in PLOT_ORDER]
    ax.legend(handles=handles, fontsize=8, ncol=2, loc="best")


def fig_peak_efficiency(d, t, ti, AVALS):
    """Sketch 1: peak efficiency eta vs selection strength a (full a-grid)."""
    rows = []
    for a in AVALS:
        for s in CASES:
            eta = d[f"{a}_{s}_eta"]
            i200 = ti.get(200, int(np.argmin(np.abs(t - 200))))
            pk = int(np.nanargmax(eta[: i200 + 1]))
            rows.append(dict(a=a, case=s, eta_peak=float(eta[pk])))
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for s in CASES:
        g = df[df.case == s].sort_values("a")
        ax.plot(g.a, g.eta_peak, "-o", ms=5.5, color=CASE_COLORS[s], lw=2)
    ax.set_xlabel("selection strength a")
    ax.set_ylabel(r"peak efficiency $\eta$")
    ax.set_title("stronger selection wastes more\n"
                 r"peak $\eta$ DROPS for every strategy as $a$ rises"
                 "\n(more pressure ≠ more efficient)", fontsize=11)
    strategy_legend(ax)
    ax.grid(True, alpha=.3)

    out = SRC / "fig_peak_efficiency.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_info_vs_cost(d, t, ref_a):
    """Sketch 2: information D(t) vs cumulative selective cost (at ref_a)."""
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for s in CASES:
        info = d[f"{ref_a}_{s}_info"]
        cost = d[f"{ref_a}_{s}_cost"]
        cum = np.zeros_like(info)
        running = 0.0
        for i, tt in enumerate(t):
            if tt >= 1 and abs(tt - round(tt)) < 1e-9:
                running += float(cost[i])
            cum[i] = running
        Deq = float(info[-1])
        t95 = first_cross(t, info, 0.95 * Deq)
        m = t <= (t95 if np.isfinite(t95) else t[-1])
        ax.plot(cum[m], info[m], "-", lw=2.0, color=CASE_COLORS[s])
        ax.plot(cum[m][-1], info[m][-1], "o", ms=7, color=CASE_COLORS[s])
    ax.set_xlabel(r"cumulative selective cost  $\sum_{t'=1}^{t} C(t')$  [bits]")
    ax.set_ylabel(r"information $D(t)$ [bits]")
    ax.set_title(f"same bits, very different price  (a={ref_a:g})\n"
                 "slope = bits per unit cost  (steeper = thriftier)\n"
                 "A/B buy info cheap; E buys it expensive", fontsize=11)
    strategy_legend(ax)
    ax.grid(True, alpha=.3)

    out = SRC / "fig_info_vs_cost.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    d = np.load(SRC / "curves.npz")
    t = d["times"].astype(float)
    ti = {int(tt): i for i, tt in enumerate(t)}
    AVALS = discover_avals(d)
    ref_a = min(AVALS, key=lambda x: abs(x - REF_A_PREF))
    print(f"data {SRC / 'curves.npz'}")
    print(f"a-grid ({len(AVALS)}): {AVALS}")
    print(f"info-vs-cost reference a={ref_a:g}")

    fig_peak_efficiency(d, t, ti, AVALS)
    fig_info_vs_cost(d, t, ref_a)


if __name__ == "__main__":
    main()
