"""Panel C + efficiency for iso-peak strategies A,B,C,D,F,G.

Uses the dynamical curves already saved by canon_dynamics.py
(N=30, u=v=0.005, r=0; a in {0.05,0.1,0.2}).

Each point = case mean +/- SEM over the 3 a-values.
Panels:
  (i)   D_eq vs t95
  (ii)  F*   vs t95
  (iii) D_eq vs eta_100
  (iv)  F*   vs eta_100
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adaptation_per_a import CASE_DEFS, CASE_COLORS

SRC = Path(__file__).resolve().parent / "results" / "canon_dynamics_20260728"
CASES = ["A", "B", "C", "D", "F", "G"]
LABEL = {s: f"{s}: {CASE_DEFS[s][0]}" for s in CASES}
AVALS = [0.05, 0.1, 0.2]


def first_cross(t, y, lvl):
    a = y >= lvl
    if not a.any():
        return np.nan
    k = int(np.argmax(a))
    if k == 0:
        return float(t[0])
    t0, t1, y0, y1 = t[k - 1], t[k], y[k - 1], y[k]
    return float(t0 + (lvl - y0) * (t1 - t0) / (y1 - y0)) if y1 != y0 else float(t1)


def main():
    d = np.load(SRC / "curves.npz")
    t = d["times"].astype(float)
    # integer times 0..200 for horizon eta
    ti = {int(tt): i for i, tt in enumerate(t)}

    rows = []
    for a in AVALS:
        for s in CASES:
            info = d[f"{a}_{s}_info"]
            cost = d[f"{a}_{s}_cost"]
            eta = d[f"{a}_{s}_eta"]
            F = d[f"{a}_{s}_F"]
            D_eq = float(info[-1])
            Fstar = float(F[-1])          # F5 -> F* at long t
            t95 = first_cross(t, info, 0.95 * D_eq)
            # cumulative eta_100 = D(100) / (N * sum_{t=1..100} C(t))
            i100 = ti.get(100)
            if i100 is None:
                i100 = int(np.argmin(np.abs(t - 100)))
            D100 = float(info[i100])
            # cost at integer gens 1..100
            idxs = [ti[k] for k in range(1, 101) if k in ti]
            cumC = float(cost[idxs].sum()) if idxs else np.nan
            eta100 = D100 / (30.0 * cumC) if cumC > 1e-12 else np.nan
            peak = float(np.nanmax(eta[:i100 + 1])) if np.isfinite(eta[:i100 + 1]).any() else np.nan
            rows.append(dict(a=a, case=s, D_eq=D_eq, Fstar=Fstar, t95=t95,
                             eta_100=eta100, eta_peak=peak))

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(SRC / "panel_c_eff_scalars.csv", index=False)

    def ms(col):
        g = df.groupby("case")[col]
        return g.mean(), g.sem()

    fig, ax = plt.subplots(2, 2, figsize=(13.2, 10.2))

    def panel(a, xcol, ycol, xl, yl, ylog=False):
        mx, sx = ms(xcol); my, sy = ms(ycol)
        for s in CASES:
            a.errorbar(mx[s], my[s], xerr=sx[s], yerr=sy[s], fmt="o", ms=14,
                       c=CASE_COLORS[s], capsize=4, label=LABEL[s], zorder=3)
            a.annotate(s, (mx[s], my[s]), fontsize=9, ha="center", va="bottom",
                       xytext=(0, 7), textcoords="offset points",
                       fontweight="bold", color=CASE_COLORS[s])
        a.set_xlabel(xl); a.set_ylabel(yl)
        if ylog:
            a.set_yscale("log")
        a.grid(True, which="both", alpha=.25)

    panel(ax[0, 0], "D_eq", "t95",
          r"mean $D_{eq}$ (info, bits)",
          r"mean $t_{95}$ arrival (gen)", ylog=True)
    ax[0, 0].set_title(r"(i) information vs arrival time $t_{95}$")

    panel(ax[0, 1], "Fstar", "t95",
          r"mean $F^*$ (free fitness, nats)",
          r"mean $t_{95}$ arrival (gen)", ylog=True)
    ax[0, 1].set_title(r"(ii) free fitness vs arrival time $t_{95}$")

    panel(ax[1, 0], "D_eq", "eta_100",
          r"mean $D_{eq}$ (info, bits)",
          r"mean $\eta_{100}$ (efficient $\uparrow$)")
    ax[1, 0].set_title(r"(iii) information vs efficiency $\eta_{100}$")

    panel(ax[1, 1], "Fstar", "eta_100",
          r"mean $F^*$ (free fitness, nats)",
          r"mean $\eta_{100}$ (efficient $\uparrow$)")
    ax[1, 1].set_title(r"(iv) free fitness vs efficiency $\eta_{100}$")

    ax[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(
        "Panel C + efficiency  (iso-peak A,B,C,D,F,G)\n"
        r"$N=30$, $u=v=0.005$, $r=0$ · points = mean $\pm$ SEM over $a\in\{0.05,0.1,0.2\}$"
        "\n"
        r"$\eta_{100}=D(100)/(N\cdot\sum_{t=1}^{100} C(t))$  "
        r"$C(x)=\sum_g x_g r_g\log_2 r_g$",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = SRC / "fig_panelC_with_efficiency.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)
    print("\ncase means:")
    print(df.groupby("case")[["D_eq", "Fstar", "t95", "eta_100", "eta_peak"]]
          .mean().round(4).reindex(CASES))


if __name__ == "__main__":
    main()
