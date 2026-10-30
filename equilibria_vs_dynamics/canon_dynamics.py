"""Dynamical plots for the 6 iso-peak canonical strategies A–F (new lettering).

  A asymmetric additive · B additive · C synergy (iso-peak)
  D sign · E reciprocal · F neutral intermediates
  All share lf[11]=2a (iso-peak).

Mirrors the "Part 1" figure (info-gain D(t) trajectories + D_eq-vs-arrival scatter)
and adds the dynamical COST C(t), efficiency eta(t) and free fitness F(t).

Clock: weak-mutation reversible  u=v=0.005, r=0, N=30  (same as the Part-1 figure).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from node_jump_analysis import build_chain, Landscape
from adaptation_per_a import CASE_DEFS, CASE_COLORS, CASE_ORDER, curves_for

OUT = Path(__file__).resolve().parent / "results" / "canon_dynamics_relabel"
CASES = list(CASE_ORDER)  # A..F (table lettering; all iso-peak lf[11]=2a)
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
    OUT.mkdir(parents=True, exist_ok=True)
    times = np.unique(np.concatenate([
        np.arange(0, 201), np.round(np.geomspace(1, TMAX, 160)).astype(int)]))
    chain = build_chain(N, U, V, R, 1)
    print(f"chain S={len(chain.phi)}", flush=True)

    res = {}     # (a, case) -> curves dict (+ D_eq, t95)
    for a in AVALS:
        for s in CASES:
            s1, s2, eps = CASE_DEFS[s][1](a)
            # hard check: every landscape must be iso-peak at 11 = 2a
            lf11 = s1 + s2 + eps
            assert abs(lf11 - 2.0 * a) < 1e-12, (s, a, lf11, 2 * a)
            L = Landscape(f"{s}: {CASE_DEFS[s][0]}", s, s1=s1, s2=s2,
                          eps12=eps, blurb="")
            c = curves_for(chain, L, times, 1)
            D_eq = float(c["info"][-1])
            c["D_eq"] = D_eq
            c["t95"] = first_cross(times.astype(float), c["info"], 0.95 * D_eq)
            res[(a, s)] = c
        print(f"  a={a} done", flush=True)
    np.savez_compressed(OUT / "curves.npz", times=times,
                        **{f"{a}_{s}_{k}": np.asarray(res[(a, s)][k])
                           for a in AVALS for s in CASES
                           for k in ("info", "cost", "eta", "F", "mlf")})

    tf = times.astype(float)

    # ---- Figure 1: info gain D(t) + D_eq-vs-t95 scatter (mirror of Part 1) ----
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for j, a in enumerate(AVALS):
        for s in CASES:
            c = res[(a, s)]
            ax[0, j].plot(tf, c["info"], "-", lw=1.8, color=CASE_COLORS[s],
                          label=f"{LABEL[s]} (D={c['D_eq']:.2f})")
        ax[0, j].set_xscale("log"); ax[0, j].set_xlim(1, TMAX)
        ax[0, j].set_xlabel("generation t"); ax[0, j].set_ylabel("D(t) [bits]")
        ax[0, j].set_title(f"a={a}: info gain D(t)")
        ax[0, j].legend(fontsize=7)
        xs = np.array([res[(a, s)]["D_eq"] for s in CASES])
        ys = np.array([res[(a, s)]["t95"] for s in CASES])
        rho = spearmanr(xs, ys)[0]
        for s, x, y in zip(CASES, xs, ys):
            ax[1, j].scatter(x, y, s=110, color=CASE_COLORS[s], edgecolor="k",
                             zorder=3)
            ax[1, j].annotate(s, (x, y), fontsize=8, ha="center", va="center")
        ax[1, j].set_xlabel(r"$D_{eq}$ info at equilibrium [bits]")
        ax[1, j].set_ylabel(r"$t_{95}$ arrival time [gen]")
        ax[1, j].set_title(rf"a={a}: Spearman(D,$t_{{95}}$)={rho:+.2f}")
        ax[1, j].grid(True, alpha=.3)
    fig.suptitle(
        "Part 1 — dynamical info gain (iso-peak A–F, all lf[11]=2a; "
        "N=30, u=v=0.005, r=0)\n"
        "A=asymm. additive  B=additive  C=synergy  D=sign  "
        "E=reciprocal  F=neutral intermediates",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "fig_canon_dyn_info.png", dpi=130)
    print("wrote", OUT / "fig_canon_dyn_info.png")

    # ---- Figure 2: dynamical cost C(t), efficiency eta(t), free fitness F(t) ----
    fig, ax = plt.subplots(3, 3, figsize=(16, 12))
    rows = [("cost", "cost C(t) [bits/gen]"),
            ("eta", r"efficiency $\eta(t)$"),
            ("F", "free fitness F(t) [nats]")]
    for j, a in enumerate(AVALS):
        for r_, (key, lab) in enumerate(rows):
            for s in CASES:
                ax[r_, j].plot(tf, res[(a, s)][key], "-", lw=1.7,
                               color=CASE_COLORS[s], label=LABEL[s])
                if key == "F":
                    ax[r_, j].axhline(res[(a, s)]["Fstar"], ls=":", lw=.8,
                                      color=CASE_COLORS[s], alpha=.5)
            ax[r_, j].set_xscale("log"); ax[r_, j].set_xlim(1, TMAX)
            ax[r_, j].set_xlabel("generation t"); ax[r_, j].set_ylabel(lab)
            ax[r_, j].grid(True, alpha=.3)
            if r_ == 0:
                ax[r_, j].set_title(f"a={a}")
    ax[1, 0].axhline(1.0, color="0.5", ls=":", lw=.9)
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(
        "Dynamical cost, efficiency η(t), free fitness "
        "(iso-peak A–F; N=30, u=v=0.005, r=0)\n"
        "A=asymm. additive  B=additive  C=synergy  D=sign  "
        "E=reciprocal  F=neutral intermediates",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "fig_canon_dyn_eff.png", dpi=130)
    print("wrote", OUT / "fig_canon_dyn_eff.png")

    print("\nequilibrium summary (a=0.1):")
    for s in CASES:
        c = res[(0.1, s)]
        pk = np.nanmax(c["eta"])
        print(f"  {s} {CASE_DEFS[s][0]:20s} D_eq={c['D_eq']:.3f} t95={c['t95']:6.1f}"
              f" F*={c['Fstar']:.3f} peak_eta={pk:.3f}")


if __name__ == "__main__":
    main()
