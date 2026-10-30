#!/usr/bin/env python3
r"""
Three-panel adaptation curves (information gain, mean log-fitness, free fitness)
for A (sign epistasis) vs B (additive), one row per selection strength `a`.

The neutral chain depends only on (N, u, r), so it is built ONCE and reused for
every `a`; only the selection transition matrix and the time-integration from phi
are recomputed per (a, landscape).  Evolution starts from phi (single column).

Free fitness (per panel 3):
    F(psi) = F* - (1/N) D_KL(psi || psi*),
    F*     = <log W>_{psi*} - (1/N) D_KL(psi* || phi)         (logB(alpha) excluded)

Run:
  conda activate miso
  cd equilibria_vs_dynamics
  python adaptation_per_a.py --a 0.02 0.05 0.1 0.15 0.2 0.3 0.4 0.5 --N 30 --u 2e-4 --tmax 20000
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from node_jump_analysis import (  # noqa: E402
    Landscape, build_chain, evolve_many, sample_times, selection_P,
)
from compare_two_cases import (  # noqa: E402
    free_fitness5_asymptote, free_fitness5_series, kl_series, mean_logfit_series,
)
from simplex import (  # noqa: E402
    build_sigma, fitness_exp, mutation_matrix_asymmetric, stationary_fast,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# Four landscapes, all sharing 00=0 and 11=2a, differing only at the intermediates.
#   slug -> (short name, (s1, s2, eps12) as functions of a)
#   E (multiple peaks): Sigma = [[2a, -a],[-a, 2a]] -> (s1,s2,eps12) = (2a, 2a, -2a),
#     log-fitness [00,01,10,11] = [0, 2a, 2a, 2a]  (three tied mutant optima).
# Canonical lettering (Jul 2026 remapping).  All iso-peak at 11 = 2a except where noted.
#   A  asymmetric additive   Sigma=[[a/2,0],[0,3a/2]]     lf=[0,3a/2,a/2,2a]  2 routes
#   B  additive              Sigma=[[a,0],[0,a]]           lf=[0,a,a,2a]       2 routes
#   C  synergy (iso-peak)    Sigma=[[a/2,a/2],[a/2,a/2]]   lf=[0,a/2,a/2,2a]   2 + epi
#   D  sign epistasis        Sigma=[[a,a],[a,-a]]          lf=[0,-a,a,2a]      1 route
#   E  reciprocal sign       Sigma=[[-a,2a],[2a,-a]]       lf=[0,-a,-a,2a]     0 valley
#   F  neutral intermediates Sigma=[[0,a],[a,0]]           lf=[0,0,0,2a]       0 flat
# (Old lettering A=sign,B=add,C=recip,D=neutral,F=synergy,G=asymm is OBSOLETE.
#  Old "multiple peaks" slug is no longer in CASE_ORDER.)
CASE_DEFS = {
    "A": ("asymmetric additive",   lambda a: (a / 2.0, 3.0 * a / 2.0, 0.0)),
    "B": ("additive",              lambda a: (a, a, 0.0)),
    "C": ("synergy (iso-peak)",    lambda a: (a / 2.0, a / 2.0, a)),
    "D": ("sign epistasis",        lambda a: (a, -a, 2.0 * a)),
    "E": ("reciprocal sign",       lambda a: (-a, -a, 4.0 * a)),
    "F": ("neutral intermediates", lambda a: (0.0, 0.0, 2.0 * a)),
}
# colours keyed to landscape type (stable under the remapping)
CASE_COLORS = {
    "A": "#da8bc3",  # asymmetric additive (was old G)
    "B": "#4c72b0",  # additive
    "C": "#937860",  # synergy (was old F)
    "D": "#c44e52",  # sign (was old A)
    "E": "#55a868",  # reciprocal (was old C)
    "F": "#8172b3",  # neutral (was old D)
}
CASE_ORDER = ["A", "B", "C", "D", "E", "F"]
# backward-compatible aliases
CA, CB = CASE_COLORS["D"], CASE_COLORS["B"]  # sign, additive


def all_landscapes(a: float) -> list[Landscape]:
    out = []
    for slug in CASE_ORDER:
        name, fn = CASE_DEFS[slug]
        s1, s2, eps = fn(a)
        out.append(Landscape(f"{slug}: {name}", slug, s1=s1, s2=s2, eps12=eps,
                             blurb=""))
    return out


def cost_per_state(fracs: np.ndarray, omega: np.ndarray,
                   eps: float = 1e-300) -> np.ndarray:
    r"""Per-generation selective cost  C(x) = sum_g x_g r_g log2(r_g)  [bits],
    with r_g = omega(g) / (x . omega).  This is D_KL(x' || x) in bits, where
    x'_g = x_g r_g is the post-selection frequency: the info injected by one
    selection step at state x."""
    x = np.maximum(fracs, eps)
    wbar = np.maximum(x @ omega, eps)
    r = np.maximum(omega / wbar[:, None], eps)
    return np.sum(x * r * np.log2(r), axis=1)


def curves_for(chain, L: Landscape, times: np.ndarray, n_jobs: int) -> dict:
    P = selection_P(chain, L, n_jobs)
    psi_star = stationary_fast(P)
    Psi0 = chain.phi.reshape(-1, 1)
    snaps = evolve_many(P, Psi0, times)                       # (T, S, 1)
    omega = fitness_exp(build_sigma(L.s1, L.s2, L.eps12))
    M_mut = mutation_matrix_asymmetric(np.array([chain.u, chain.u]),
                                       np.array([chain.v, chain.v]))
    Fstar, _ = free_fitness5_asymptote(psi_star, chain.phi, chain.fracs,
                                       omega, M_mut, chain.N)
    info = kl_series(snaps, chain.phi, 0)                     # bits
    c_state = cost_per_state(chain.fracs, omega)              # bits, per state
    cost = snaps[:, :, 0] @ c_state                           # C(t)=<C(x)>_psi
    tfl = times.astype(float)
    dDdt = np.gradient(info, tfl)                             # bits / generation
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(cost > 1e-15, dDdt / (float(chain.N) * cost), np.nan)
    return dict(
        info=info,
        mlf=mean_logfit_series(snaps, chain.fracs, L.logfit, 0),
        F=free_fitness5_series(snaps, psi_star, Fstar, chain.N, 0),
        Fstar=Fstar,
        cost=cost,
        eta=eta,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float, nargs="+",
                    default=[0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--u", type=float, default=2e-4)
    ap.add_argument("--v", type=float, default=2e-4)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--tmax", type=int, default=20000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", type=Path,
                    default=SCRIPT_DIR / "results" /
                    f"adaptation_per_a_{datetime.now():%Y%m%d}")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print(f"Building neutral chain N={args.N}, u=v={args.u}, r={args.r} ...", flush=True)
    chain = build_chain(args.N, args.u, args.v, args.r, args.n_jobs)
    print(f"  S={len(chain.phi)}  phi corner mass={chain.phi[chain.cls==1].sum():.4f}"
          f"  ({time.perf_counter()-t0:.1f}s)", flush=True)

    times = sample_times(args.tmax)
    res = {}   # a -> {slug: curves}
    for a in args.a:
        row = {}
        for L in all_landscapes(a):
            row[L.slug] = curves_for(chain, L, times, args.n_jobs)
        res[a] = row
        fs = "  ".join(f"F*_{s}={row[s]['Fstar']:.5f}" for s in CASE_ORDER)
        print(f"  a={a:.3f}: {fs}  ({time.perf_counter()-t0:.1f}s)", flush=True)

    labels = {s: f"{s}: {CASE_DEFS[s][0]}" for s in CASE_ORDER}
    col_titles = ["Information gain vs time", "Mean log-fitness", "Free fitness",
                  "Cost per generation", "Efficiency $\\eta$"]
    cases_blurb = "A=sign, B=additive, C=reciprocal, D=neutral, E=multiple peaks"

    # combined figure: one row per a, five columns, five case-lines each
    na = len(args.a)
    fig, axes = plt.subplots(na, 5, figsize=(26.0, 3.1 * na), squeeze=False)
    for i, a in enumerate(args.a):
        row = res[a]
        ax_i, ax_m, ax_f, ax_c, ax_e = axes[i]
        for s in CASE_ORDER:
            c = CASE_COLORS[s]
            ax_i.plot(times, row[s]["info"], "-", lw=1.8, color=c, label=labels[s])
            ax_m.plot(times, row[s]["mlf"], "-", lw=1.8, color=c)
            ax_f.plot(times, row[s]["F"], "-", lw=1.8, color=c)
            ax_f.axhline(row[s]["Fstar"], ls=":", lw=0.9, color=c, alpha=0.6)
            ax_c.plot(times, row[s]["cost"], "-", lw=1.8, color=c)
            ax_e.plot(times, row[s]["eta"], "-", lw=1.8, color=c)
        ax_e.axhline(1.0, color="0.5", ls=":", lw=0.9, alpha=0.7)
        for ax in (ax_i, ax_m, ax_f, ax_c, ax_e):
            ax.set_xscale("log"); ax.grid(True, alpha=0.3)
            if i == na - 1:
                ax.set_xlabel("generation $t$")
        ax_i.set_ylabel(f"$a={a:g}$\n$D_{{KL}}(\\psi\\|\\varphi)$ [bits]")
        ax_m.set_ylabel(r"$\langle\sum_g x_g\log w_g\rangle$")
        ax_f.set_ylabel(r"$F^*-\frac{1}{N}D_{KL}(\psi\|\psi^*)$ [nats]")
        ax_c.set_ylabel(r"$C(t)=\langle C(x)\rangle_\psi$ [bits]")
        ax_e.set_ylabel(r"$\eta(t)=\frac{dD/dt}{N\,C(t)}$")
        if i == 0:
            ax_i.legend(fontsize=7, ncol=2)
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=12)
    fig.suptitle(f"Adaptation curves per selection strength $a$  (N={args.N}, "
                 f"u=v={args.u:g})   {cases_blurb}",
                 fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    combined = out / f"adaptation_per_a_N{args.N}.png"
    fig.savefig(combined, dpi=150)
    plt.close(fig)

    # standalone 5-panel per a
    for a in args.a:
        row = res[a]
        fig, (ax_i, ax_m, ax_f, ax_c, ax_e) = plt.subplots(1, 5, figsize=(26.0, 4.6))
        for s in CASE_ORDER:
            c = CASE_COLORS[s]
            ax_i.plot(times, row[s]["info"], "-", lw=1.8, color=c, label=labels[s])
            ax_m.plot(times, row[s]["mlf"], "-", lw=1.8, color=c, label=labels[s])
            ax_f.plot(times, row[s]["F"], "-", lw=1.8, color=c, label=labels[s])
            ax_f.axhline(row[s]["Fstar"], ls=":", lw=0.9, color=c, alpha=0.6)
            ax_c.plot(times, row[s]["cost"], "-", lw=1.8, color=c, label=labels[s])
            ax_e.plot(times, row[s]["eta"], "-", lw=1.8, color=c, label=labels[s])
        ax_e.axhline(1.0, color="0.5", ls=":", lw=0.9, alpha=0.7, label=r"bound $\eta=1$")
        for ax in (ax_i, ax_m, ax_f, ax_c, ax_e):
            ax.set_xscale("log"); ax.grid(True, alpha=0.3)
            ax.set_xlabel("generation $t$"); ax.legend(fontsize=8)
        ax_i.set_ylabel(r"$D_{KL}(\psi(t)\|\varphi)$ [bits]")
        ax_i.set_title("Information gain vs time")
        ax_m.set_ylabel(r"mean log-fitness $\langle\sum_g x_g\log w_g\rangle$")
        ax_m.set_title("Mean log-fitness")
        ax_f.set_ylabel(r"$F^*-\frac{1}{N}D_{KL}(\psi\|\psi^*)$ [nats]")
        ax_f.set_title("Free fitness")
        ax_c.set_ylabel(r"$C(t)=\langle C(x)\rangle_\psi$ [bits]")
        ax_c.set_title("Cost per generation")
        ax_e.set_ylabel(r"$\eta(t)=\Delta D/(N\,C(t))$")
        ax_e.set_title(r"Efficiency $\eta$")
        fig.suptitle(f"a={a:g}  (N={args.N}, u=v={args.u:g})", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out / f"adaptation_a{a:g}_N{args.N}.png", dpi=150)
        plt.close(fig)

    print(f"\nDone in {time.perf_counter()-t0:.1f}s. Combined: {combined}")


if __name__ == "__main__":
    main()
