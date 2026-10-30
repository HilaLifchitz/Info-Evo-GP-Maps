"""Panel-C positioning WITH efficiency, for the 6 iso-peak strategies.

Reads the saved dynamical curves (weak-mut r=0 clock, 3 a-values) and plots case
means (+/- SEM over a) in four planes:
   (i)  info  D_eq        vs  arrival time t95      (speed)
   (ii) free fitness F*   vs  arrival time t95      (speed)
   (iii) info D_eq        vs  peak efficiency eta   (efficiency)
   (iv) free fitness F*   vs  peak efficiency eta   (efficiency)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adaptation_per_a import CASE_DEFS, CASE_COLORS

OUT = Path(__file__).resolve().parent / "results" / "canon_dynamics_20260728"
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
    z = np.load(OUT / "curves.npz")
    t = z["times"].astype(float)
    # collect per-(case) arrays across a-values
    stat = {s: dict(D_eq=[], Fstar=[], t95=[], peak_eta=[], eta_cum=[]) for s in CASES}
    for a in AVALS:
        for s in CASES:
            info = z[f"{a}_{s}_info"]; cost = z[f"{a}_{s}_cost"]
            eta = z[f"{a}_{s}_eta"]; F = z[f"{a}_{s}_F"]
            D_eq = float(info[-1])
            t95 = first_cross(t, info, 0.95 * D_eq)
            # cumulative efficiency to arrival: total bits / (N * integral of cost dt)
            mask = t <= (t95 if np.isfinite(t95) else t[-1])
            cost_int = np.trapz(cost[mask], t[mask])
            eta_cum = D_eq / (30.0 * cost_int) if cost_int > 1e-12 else np.nan
            stat[s]["D_eq"].append(D_eq)
            stat[s]["Fstar"].append(float(F[-1]))
            stat[s]["t95"].append(t95)
            stat[s]["peak_eta"].append(float(np.nanmax(eta)))
            stat[s]["eta_cum"].append(eta_cum)

    def m(s, k):
        return float(np.nanmean(stat[s][k]))

    def e(s, k):
        v = np.asarray(stat[s][k], float)
        return float(np.nanstd(v) / max(1, np.isfinite(v).sum() ** 0.5))

    fig, ax = plt.subplots(2, 2, figsize=(13, 10.5))

    def panel(a_, xk, yk, xl, yl, ylog=False):
        for s in CASES:
            a_.errorbar(m(s, xk), m(s, yk), xerr=e(s, xk), yerr=e(s, yk),
                        fmt="o", ms=14, c=CASE_COLORS[s], capsize=4, label=LABEL[s])
        if ylog:
            a_.set_yscale("log")
        a_.set_xlabel(xl); a_.set_ylabel(yl); a_.grid(True, alpha=.25)

    panel(ax[0, 0], "D_eq", "t95", r"info $D_{eq}$ (bits)",
          r"arrival $t_{95}$ (gen, slow $\to$)", ylog=True)
    ax[0, 0].set_title("(i) info vs speed")
    panel(ax[0, 1], "Fstar", "t95", r"free fitness $F^*$ (nats)",
          r"arrival $t_{95}$ (gen, slow $\to$)", ylog=True)
    ax[0, 1].set_title("(ii) free fitness vs speed")
    panel(ax[1, 0], "D_eq", "peak_eta", r"info $D_{eq}$ (bits)",
          r"peak efficiency $\eta$ (efficient $\uparrow$)")
    ax[1, 0].set_title(r"(iii) info vs efficiency")
    panel(ax[1, 1], "Fstar", "peak_eta", r"free fitness $F^*$ (nats)",
          r"peak efficiency $\eta$ (efficient $\uparrow$)")
    ax[1, 1].set_title(r"(iv) free fitness vs efficiency")
    ax[0, 0].legend(fontsize=9, loc="best")
    fig.suptitle("Panel C + efficiency: 6 iso-peak strategies  "
                 "(case means +/- SEM over a=0.05,0.1,0.2;  N=30, u=v=0.005, r=0)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig_canon_positioning_eff.png", dpi=130)
    print("wrote", OUT / "fig_canon_positioning_eff.png")

    print("\ncase means:")
    print(f"{'case':<6}{'D_eq':>7}{'F*':>8}{'t95':>8}{'peak_eta':>10}{'eta_cum':>9}")
    for s in CASES:
        print(f"{s:<6}{m(s,'D_eq'):>7.3f}{m(s,'Fstar'):>8.3f}{m(s,'t95'):>8.1f}"
              f"{m(s,'peak_eta'):>10.3f}{m(s,'eta_cum'):>9.3f}")


if __name__ == "__main__":
    main()
