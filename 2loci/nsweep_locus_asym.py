#!/usr/bin/env python3
r"""Diffusion-limit convergence: which kind of mutation asymmetry breaks it?

Three per-locus mutation regimes, all with the SAME selection scaling
(N*s_eff(11)=0.76, Sigma direction (0.08,-0.04,0.15)) and r=0:

  (A) symmetric / reversible      u1=v1=u2=v2   (2N* = 6.4 all four)
  (B) direction-asym (non-rev.)   u1=u2=6.4/2N, v1=v2=1.6/2N   (forward!=backward)
  (C) locus-asym but reversible   u1=v1=6.4/2N, u2=v2=1.6/2N   (each locus u_i=v_i,
                                                                loci differ)

For each we sweep N and record interior TV / KL(psi* || cond-alpha diffusion),
using the 2N cond-alpha convention that reproduces the original figure
(alpha_i = 2N influx_i/(1-x_i), p(x) ~ W(x)^N prod x_i^{alpha_i-1}).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simplex as sp
from selection_diffusion_vs_chain import _normalize_log, _tv_kl

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = (SCRIPT_DIR.parent / "equilibria_vs_dynamics" / "results" /
       "sel_diff_Nsweep_locusasym")

DIR_S1, DIR_S2, DIR_EPS = 0.08, -0.04, 0.15
SEFF11_DIR = DIR_S1 + DIR_S2 + DIR_EPS          # 0.19
NSEFF11 = 0.76
N_VALUES = [20, 30, 40, 50]


def psi_star_and_diff(N, u_vec, v_vec):
    c = (NSEFF11 / N) / SEFF11_DIR
    s1, s2, eps12 = c * DIR_S1, c * DIR_S2, c * DIR_EPS
    Sigma = sp.build_sigma(s1, s2, eps12)
    omega = sp.fitness_exp(Sigma)
    M_mut = sp.mutation_matrix_asymmetric(np.asarray(u_vec), np.asarray(v_vec))
    fracs = sp.make_simplex_grid(N)
    T_row, _, _ = sp.build_transition_matrix_row_stochastic(
        N, omega, M_mut, 0.0, n_jobs=-1, label="", verbose=False)
    P_col = sp.column_stochastic(T_row)
    psi = np.clip(sp.stationary_fast(P_col), 0.0, None)
    psi /= psi.sum()

    X = fracs
    interior = np.all(X > 0.0, axis=1)
    boundary = float(psi[~interior].sum())
    W = X @ omega
    logW = np.log(np.clip(W, 1e-300, None))
    influx = X @ M_mut - X * np.diag(M_mut)[None, :]
    alpha = 2.0 * float(N) * influx / np.clip(1.0 - X, 1e-300, None)
    lp = np.full(X.shape[0], -np.inf)
    lp[interior] = N * logW[interior] + np.sum((alpha[interior] - 1.0) * np.log(X[interior]), axis=1)
    psi_int = np.zeros_like(psi); psi_int[interior] = psi[interior]
    psi_int = psi_int / psi_int.sum() if psi_int.sum() > 0 else psi_int
    pdiff = _normalize_log(lp, interior)
    tv, kl = _tv_kl(psi_int, pdiff)
    amin = float(np.min(alpha[interior]))
    amed = float(np.median(alpha[interior]))
    return dict(N=N, invN=1.0 / N, tv=tv, kl=kl, boundary=boundary,
                alpha_min=amin, alpha_med=amed)


def sweep(label, u_of_N, v_of_N):
    print(f"=== {label} ===")
    rows = []
    for N in N_VALUES:
        r = psi_star_and_diff(N, u_of_N(N), v_of_N(N))
        rows.append(r)
        print(f"  N={N:3d} u={u_of_N(N)} v={v_of_N(N)} | "
              f"TV={r['tv']:.4f} KL={r['kl']:.4f} boundary={r['boundary']:.4f} "
              f"alpha_min={r['alpha_min']:.2f} alpha_med={r['alpha_med']:.2f}", flush=True)
    return rows


def fit(rows, key):
    x = np.array([r["invN"] for r in rows]); y = np.array([r[key] for r in rows])
    m, b = np.polyfit(x, y, 1); return m, b


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    configs = [
        ("A symmetric/reversible  2N=(6.4,6.4,6.4,6.4)",
         lambda N: [6.4/(2*N), 6.4/(2*N)], lambda N: [6.4/(2*N), 6.4/(2*N)], "#3b528b"),
        ("B direction-asym, WEAK dir  2N=(6.4,6.4,1.6,1.6)",
         lambda N: [6.4/(2*N), 6.4/(2*N)], lambda N: [1.6/(2*N), 1.6/(2*N)], "#c44e52"),
        ("C locus-asym (rev), WEAK dir  2N=(6.4,1.6,6.4,1.6)",
         lambda N: [6.4/(2*N), 1.6/(2*N)], lambda N: [6.4/(2*N), 1.6/(2*N)], "#55a868"),
        ("D direction-asym, no weak dir  2N=(6.4,6.4,4.8,4.8)",
         lambda N: [6.4/(2*N), 6.4/(2*N)], lambda N: [4.8/(2*N), 4.8/(2*N)], "#8172b3"),
        ("E locus-asym (rev), no weak dir  2N=(6.4,4.8,6.4,4.8)",
         lambda N: [6.4/(2*N), 4.8/(2*N)], lambda N: [6.4/(2*N), 4.8/(2*N)], "#ccb974"),
        ("F fully-asym, all-distinct all-large  2N=(12,8,9,6)",
         lambda N: [12.0/(2*N), 8.0/(2*N)], lambda N: [9.0/(2*N), 6.0/(2*N)], "#d1495b"),
    ]
    data = [(lab, sweep(lab, uf, vf), col) for lab, uf, vf, col in configs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for lab, rows, col in data:
        x = np.array([r["invN"] for r in rows])
        for ax, key in ((axes[0], "tv"), (axes[1], "kl")):
            y = np.array([r[key] for r in rows])
            m, b = fit(rows, key)
            ax.plot(x, y, "o-", color=col, lw=1.8, ms=6,
                    label=f"{lab}\n  slope={m:.2f}, intercept={b:.3f}")
            xx = np.linspace(0, x.max(), 50)
            ax.plot(xx, m*xx + b, "--", color=col, lw=1, alpha=0.6)
            ax.scatter([0], [b], marker="<", color=col, zorder=5)
    axes[0].set_title("Total-variation vs 1/N"); axes[0].set_ylabel("interior TV")
    axes[1].set_title("KL vs 1/N"); axes[1].set_ylabel("interior KL [bits]")
    for ax in axes:
        ax.set_xlabel("1 / N"); ax.set_xlim(left=-0.002)
        ax.legend(fontsize=7.5); ax.grid(True, alpha=0.3)
    fig.suptitle("Which asymmetry breaks diffusion convergence? "
                 f"(Ns_eff(11)={NSEFF11}), α=cond(2N)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = OUT / "tv_kl_vs_invN_locusasym.png"
    fig.savefig(p, dpi=160, bbox_inches="tight"); print("wrote", p)

    with (OUT / "locusasym.csv").open("w") as f:
        f.write("config,N,invN,tv,kl_bits,boundary\n")
        for lab, rows, _ in data:
            for r in rows:
                f.write(f"{lab},{r['N']},{r['invN']:.5f},{r['tv']:.6g},"
                        f"{r['kl']:.6g},{r['boundary']:.6g}\n")
    print("wrote", OUT / "locusasym.csv")


if __name__ == "__main__":
    main()
