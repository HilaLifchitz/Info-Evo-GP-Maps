#!/usr/bin/env python3
r"""Diffusion-limit convergence (interior TV / KL of chain psi* vs cond-alpha
diffusion) vs 1/N, comparing SYMMETRIC (u=v) and ASYMMETRIC (u!=v) mutation.

Reconstruction of results/sel_diff_Nsweep/tv_kl_vs_invN.png (symmetric case),
plus an asymmetric-mutation run to test whether the 1/N scaling / intercept
changes when the per-locus mutation is non-symmetric (more strongly
non-reversible at the genotype level).

Scaling held fixed as N grows (the diffusion limit):
    N * s_eff(11) = NSEFF11      (selection)
    2 N u         = TWO_N_U      (forward mutation)
    2 N v         = TWO_N_V      (backward mutation; == TWO_N_U for symmetric)

Sigma direction (s1,s2,eps12) is the standard (0.08,-0.04,0.15); s_eff(11)=0.19.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simplex as sp
from selection_diffusion_vs_chain import (
    Setting, discrete_psi_star, _normalize_log, _tv_kl,
)


def diffusion_logdensity_2N(st, fracs, omega, M_mut):
    r"""cond-alpha diffusion log-density that reproduces the original figure:
        alpha_i = 2N * influx_i / (1 - x_i),   p(x) ∝ W(x)^N * prod x_i^{alpha_i-1}.
    (selection_diffusion_vs_chain.diffusion_logdensity uses factor N; the June-10
    tv_kl_vs_invN figure used factor 2N — verified to match Nsweep.csv exactly.)"""
    X = np.asarray(fracs, dtype=float)
    interior = np.all(X > 0.0, axis=1)
    W = X @ omega
    logW = np.log(np.clip(W, 1e-300, None))
    XM = X @ M_mut
    influx = XM - X * np.diag(M_mut)[None, :]
    denom = np.clip(1.0 - X, 1e-300, None)
    alpha = 2.0 * float(st.N) * influx / denom
    lp = np.full(X.shape[0], -np.inf, dtype=float)
    Xi = X[interior]
    lp[interior] = st.N * logW[interior] + np.sum((alpha[interior] - 1.0) * np.log(Xi), axis=1)
    return lp, interior

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = (SCRIPT_DIR.parent / "equilibria_vs_dynamics" / "results" /
       "sel_diff_Nsweep_symasym")

DIR_S1, DIR_S2, DIR_EPS = 0.08, -0.04, 0.15
SEFF11_DIR = DIR_S1 + DIR_S2 + DIR_EPS          # 0.19
NSEFF11 = 0.76
TWO_N_U = 6.4
N_VALUES = [20, 30, 40, 50]


def setting_for(N: int, two_n_u: float, two_n_v: float) -> tuple[Setting, dict]:
    c = (NSEFF11 / N) / SEFF11_DIR              # scale Sigma so N*s_eff(11)=0.76
    s1, s2, eps12 = c * DIR_S1, c * DIR_S2, c * DIR_EPS
    u = two_n_u / (2.0 * N)
    v = two_n_v / (2.0 * N)
    st = Setting(label=f"N={N}", N=N, s1=s1, s2=s2, eps12=eps12, u=u, v=v, r=0.0)
    info = dict(s1=s1, s2=s2, eps12=eps12, u=u, v=v,
                Nseff11=N * (s1 + s2 + eps12), two_n_u=2 * N * u, two_n_v=2 * N * v)
    return st, info


def sweep(two_n_u: float, two_n_v: float) -> list[dict]:
    rows = []
    for N in N_VALUES:
        st, info = setting_for(N, two_n_u, two_n_v)
        psi, fracs, counts, omega, M_mut = discrete_psi_star(st)
        interior = np.all(fracs > 0.0, axis=1)
        boundary = float(psi[~interior].sum())
        log_p, _ = diffusion_logdensity_2N(st, fracs, omega, M_mut)
        psi_int = np.zeros_like(psi); psi_int[interior] = psi[interior]
        s = psi_int.sum(); psi_int = psi_int / s if s > 0 else psi_int
        pdiff_int = _normalize_log(log_p, interior)
        tv, kl = _tv_kl(psi_int, pdiff_int)
        rows.append(dict(N=N, invN=1.0 / N, tv=tv, kl=kl, boundary=boundary, **info))
        print(f"  N={N:3d} u={info['u']:.4f} v={info['v']:.4f} "
              f"(2Nu={info['two_n_u']:.2f},2Nv={info['two_n_v']:.2f}) "
              f"Nseff11={info['Nseff11']:.3f} | TV={tv:.4f} KL={kl:.4f} bits "
              f"boundary={boundary:.4f}", flush=True)
    return rows


def fit(rows, key):
    x = np.array([r["invN"] for r in rows])
    y = np.array([r[key] for r in rows])
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== SYMMETRIC u=v (2Nu=2Nv=6.4) ===")
    sym = sweep(TWO_N_U, TWO_N_U)
    print("=== ASYMMETRIC u!=v (2Nu=6.4, 2Nv=1.6, ratio 4:1) ===")
    asym = sweep(TWO_N_U, 1.6)
    print("=== ASYMMETRIC extreme (2Nu=6.4, 2Nv=0.8, ratio 8:1) ===")
    asym2 = sweep(TWO_N_U, 0.8)

    datasets = [("symmetric  2Nu=2Nv=6.4", sym, "#3b528b"),
                ("asymmetric 2Nu=6.4,2Nv=1.6", asym, "#c44e52"),
                ("asymmetric 2Nu=6.4,2Nv=0.8", asym2, "#55a868")]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for label, rows, color in datasets:
        x = np.array([r["invN"] for r in rows])
        for ax, key, ylab in ((axes[0], "tv", "interior TV"),
                              (axes[1], "kl", "interior KL [bits]")):
            y = np.array([r[key] for r in rows])
            slope, icpt = fit(rows, key)
            ax.plot(x, y, "o-", color=color, lw=1.8, ms=6,
                    label=f"{label}\n  slope={slope:.2f}, intercept={icpt:.3f}")
            xx = np.linspace(0, x.max(), 50)
            ax.plot(xx, slope * xx + icpt, "--", color=color, lw=1, alpha=0.6)
            ax.scatter([0], [icpt], marker="<", color=color, zorder=5)
    for ax, ylab in ((axes[0], "interior TV"), (axes[1], "interior KL [bits]")):
        ax.set_xlabel("1 / N"); ax.set_ylabel(ylab)
        ax.set_xlim(left=-0.002); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    axes[0].set_title("Total-variation vs 1/N")
    axes[1].set_title("KL vs 1/N")
    fig.suptitle("Diffusion-limit convergence: symmetric vs asymmetric mutation "
                 f"(Ns_eff(11)={NSEFF11}, 2Nu={TWO_N_U}), α=cond", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = OUT / "tv_kl_vs_invN_symasym.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    print("wrote", p)

    with (OUT / "symasym.csv").open("w") as f:
        f.write("kind,N,invN,tv,kl_bits,boundary,two_n_u,two_n_v,Nseff11\n")
        for label, rows, _ in datasets:
            for r in rows:
                f.write(f"{label},{r['N']},{r['invN']:.5f},{r['tv']:.6g},{r['kl']:.6g},"
                        f"{r['boundary']:.6g},{r['two_n_u']:.4g},{r['two_n_v']:.4g},"
                        f"{r['Nseff11']:.4g}\n")
    print("wrote", OUT / "symasym.csv")


if __name__ == "__main__":
    main()
