"""Figure 1 with an extra bottom row: D_eq vs spectral mixing time tau_spec.

Row 1: D(t) trajectories
Row 2: D_eq vs t95   (information arrival)
Row 3: D_eq vs tau   (spectral mixing time), same scatter style as row 2

Uses saved dynamical curves from canon_dynamics_relabel; only recomputes
the cheap spectral gap for each (a, case).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from node_jump_analysis import build_chain, Landscape, selection_P
from adaptation_per_a import CASE_DEFS, CASE_COLORS, CASE_ORDER
from info_speed_experiment import spectral_gap

SRC = Path(__file__).resolve().parent / "results" / "canon_dynamics_relabel"
CASES = list(CASE_ORDER)
LABEL = {s: f"{s}: {CASE_DEFS[s][0]}" for s in CASES}
AVALS = [0.05, 0.1, 0.2]
N, U, V, R, TMAX = 30, 5e-3, 5e-3, 0.0, 1500


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
    z = np.load(SRC / "curves.npz")
    t = z["times"].astype(float)

    # spectral mixing times
    print("building chain...", flush=True)
    chain = build_chain(N, U, V, R, 1)
    tau = {}
    for a in AVALS:
        for s in CASES:
            s1, s2, eps = CASE_DEFS[s][1](a)
            L = Landscape(f"{s}", s, s1=s1, s2=s2, eps12=eps, blurb="")
            P = selection_P(chain, L, 1)
            lam2 = spectral_gap(P)
            tau[(a, s)] = (-1.0 / np.log(lam2)) if (0 < lam2 < 1) else np.nan
            print(f"  a={a} {s}: tau={tau[(a,s)]:.2f}", flush=True)

    # arrival times from saved D(t)
    t95, Deq = {}, {}
    for a in AVALS:
        for s in CASES:
            info = z[f"{a}_{s}_info"]
            Deq[(a, s)] = float(info[-1])
            t95[(a, s)] = first_cross(t, info, 0.95 * Deq[(a, s)])

    fig, ax = plt.subplots(3, 3, figsize=(16, 12.5))
    for j, a in enumerate(AVALS):
        # row 0: D(t)
        for s in CASES:
            ax[0, j].plot(t, z[f"{a}_{s}_info"], "-", lw=1.8, color=CASE_COLORS[s],
                          label=f"{LABEL[s]} (D={Deq[(a,s)]:.2f})")
        ax[0, j].set_xscale("log"); ax[0, j].set_xlim(1, TMAX)
        ax[0, j].set_xlabel("generation t"); ax[0, j].set_ylabel("D(t) [bits]")
        ax[0, j].set_title(f"a={a}: info gain D(t)")
        ax[0, j].legend(fontsize=6.5)

        # row 1: D_eq vs t95
        xs = np.array([Deq[(a, s)] for s in CASES])
        ys = np.array([t95[(a, s)] for s in CASES])
        rho = spearmanr(xs, ys)[0]
        for s, x, y in zip(CASES, xs, ys):
            ax[1, j].scatter(x, y, s=110, color=CASE_COLORS[s], edgecolor="k", zorder=3)
            ax[1, j].annotate(s, (x, y), fontsize=8, ha="center", va="center")
        ax[1, j].set_xlabel(r"$D_{eq}$ info at equilibrium [bits]")
        ax[1, j].set_ylabel(r"$t_{95}$ arrival time [gen]")
        ax[1, j].set_title(rf"a={a}: Spearman$(D,t_{{95}})$={rho:+.2f}")
        ax[1, j].grid(True, alpha=.3)

        # row 2: D_eq vs tau_spec  (same style)
        ys2 = np.array([tau[(a, s)] for s in CASES])
        rho2 = spearmanr(xs, ys2)[0]
        for s, x, y in zip(CASES, xs, ys2):
            ax[2, j].scatter(x, y, s=110, color=CASE_COLORS[s], edgecolor="k", zorder=3)
            ax[2, j].annotate(s, (x, y), fontsize=8, ha="center", va="center")
        ax[2, j].set_xlabel(r"$D_{eq}$ info at equilibrium [bits]")
        ax[2, j].set_ylabel(r"$\tau_{\mathrm{spec}}=-1/\ln|\lambda_2|$ [gen]")
        ax[2, j].set_title(rf"a={a}: Spearman$(D,\tau)$={rho2:+.2f}")
        ax[2, j].grid(True, alpha=.3)
        ax[2, j].set_yscale("log")

    fig.suptitle(
        "Part 1 — dynamical info + arrival + spectral mixing "
        "(iso-peak A–F, lf[11]=2a; N=30, u=v=0.005, r=0)\n"
        "A=asymm. additive  B=additive  C=synergy  D=sign  "
        "E=reciprocal  F=neutral intermediates",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = SRC / "fig_canon_dyn_info_with_tau.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)

    print("\nsummary a=0.1:")
    for s in CASES:
        print(f"  {s}: D={Deq[(0.1,s)]:.3f}  t95={t95[(0.1,s)]:6.1f}  "
              f"tau={tau[(0.1,s)]:6.1f}")


if __name__ == "__main__":
    main()
