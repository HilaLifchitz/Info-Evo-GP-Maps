"""Two-panel efficiency view from the iso-peak canonical dynamics.

Same as ``surprise_analysis.py`` but WITHOUT the middle "early vs late eta"
bar chart.  Keeps:

  left  : peak efficiency eta vs selection strength a  (full a-grid)
  right : information D(t) vs cumulative selective cost  (at one reference a)

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
CASES = list(CASE_ORDER)  # A..F
LABEL = {s: f"{s}: {CASE_DEFS[s][0]}" for s in CASES}
N = 30
REF_A_PREF = 0.18   # reference a for the right panel; nearest grid value is used


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


def main():
    d = np.load(SRC / "curves.npz")
    t = d["times"].astype(float)
    ti = {int(tt): i for i, tt in enumerate(t)}
    AVALS = discover_avals(d)
    ref_a = min(AVALS, key=lambda x: abs(x - REF_A_PREF))
    print(f"data {SRC / 'curves.npz'}")
    print(f"a-grid ({len(AVALS)}): {AVALS}")
    print(f"right panel reference a={ref_a:g}")

    # ---- peak efficiency per (a, case) for the left panel ----------------
    rows = []
    for a in AVALS:
        for s in CASES:
            eta = d[f"{a}_{s}_eta"]
            i200 = ti.get(200, int(np.argmin(np.abs(t - 200))))
            pk = int(np.nanargmax(eta[: i200 + 1]))
            rows.append(dict(a=a, case=s, eta_peak=float(eta[pk])))
    df = pd.DataFrame(rows)

    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.8))
    fig.subplots_adjust(wspace=0.28)

    # ========== (left) peak eta vs a (full grid) ==========
    ax = axs[0]
    for s in CASES:
        g = df[df.case == s].sort_values("a")
        ax.plot(g.a, g.eta_peak, "-o", ms=5.5, color=CASE_COLORS[s], lw=2, label=s)
    ax.set_xlabel("selection strength a")
    ax.set_ylabel(r"peak efficiency $\eta$")
    ax.set_title("stronger selection wastes more\n"
                 r"peak $\eta$ DROPS for every strategy as $a$ rises"
                 "\n(more pressure ≠ more efficient)", fontsize=10)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=.3)

    # ========== (right) D vs cumulative cost at ref_a ==========
    ax = axs[1]
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
        ax.plot(cum[m], info[m], "-", lw=2.0, color=CASE_COLORS[s], label=s)
        ax.plot(cum[m][-1], info[m][-1], "o", ms=7, color=CASE_COLORS[s])
    ax.set_xlabel(r"cumulative selective cost  $\sum_{t'=1}^{t} C(t')$  [bits]")
    ax.set_ylabel(r"information $D(t)$ [bits]")
    ax.set_title(f"same bits, very different price  (a={ref_a:g})\n"
                 "slope = bits per unit cost  (steeper = thriftier)\n"
                 "A/B buy info cheap; E buys it expensive", fontsize=10)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=.3)

    # shared strategy legend on top
    handles = [plt.Line2D([0], [0], color=CASE_COLORS[s], lw=2.4, marker="o",
                          ms=6, label=LABEL[s]) for s in CASES]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.14), fontsize=9, frameon=False)

    a_blurb = ", ".join(f"{a:g}" for a in AVALS)
    fig.suptitle(
        f"Iso-peak dynamics  ·  N=30, u=v=0.005, r=0  ·  A–F\n"
        f"left: full a-grid ({len(AVALS)} values: {a_blurb})  ·  "
        f"right: a={ref_a:g}",
        fontsize=12, fontweight="bold", y=1.22)

    out = SRC / "fig_surprises_2panel.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
