#!/usr/bin/env python3
r"""
Robustness / scaling sweep of Case A (sign epistasis) vs Case B (additive).

For a grid of s1 magnitudes `a` (= alpha, the epistasis/selection strength),
mutation rates `u=v`, and population sizes `N`, recompute the equilibrium
scalar summaries.  Everything here is linear algebra on the transition matrix
(stationary distribution only -- no long time-integration), so it is cheap.

Per (N, u, a) cell, for each landscape we compute the EQUILIBRIUM FREE FITNESS

    F* = <log W>_{psi*}  -  (1/N) D_KL(psi* || phi)          (log B(alpha) excluded)

and the main object of interest, the free-fitness GAP between the two paths

    dF = F*_B - F*_A .

Also reported: KL_eq, 11-basin mass, MFPT(00->11), cycle affinity.

Question: does the free-fitness gap between additive (B) and sign-epistasis (A)
widen as the selection strength `a` grows?

Run:
  conda activate miso
  cd equilibria_vs_dynamics
  python compare_sweep.py --a 0.05 0.1 0.2 0.3 0.4 --u 1e-4 2e-4 --N 16 24
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from node_jump_analysis import (  # noqa: E402
    Landscape, build_chain, lumped_Q, selection_P,
)
from compare_two_cases import (  # noqa: E402
    cycle_affinity, free_fitness5_asymptote, mfpt_to_basin,
)
from simplex import (  # noqa: E402
    build_sigma, fitness_exp, kl_divergence_bits,
    mutation_matrix_asymmetric, stationary_fast,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def one_cell(chain, a: float, case: str, n_jobs: int) -> dict:
    if case == "A":
        L = Landscape("A", "A", s1=a, s2=-a, eps12=2 * a, blurb="")
    else:
        L = Landscape("B", "B", s1=a, s2=a, eps12=0.0, blurb="")
    P = selection_P(chain, L, n_jobs)
    psi_star = stationary_fast(P)
    basin11 = float(psi_star[chain.basin == 3].sum())
    kl_eq = kl_divergence_bits(psi_star, chain.phi)
    Q = lumped_Q(P, chain)
    aff = cycle_affinity(Q)
    pure00 = int(np.where(chain.counts[:, 0] == chain.N)[0][0])
    mfpt = float(mfpt_to_basin(P, chain.basin, 3)[pure00])

    omega = fitness_exp(build_sigma(L.s1, L.s2, L.eps12))
    M_mut = mutation_matrix_asymmetric(np.array([chain.u, chain.u]),
                                       np.array([chain.v, chain.v]))
    Fstar, parts = free_fitness5_asymptote(psi_star, chain.phi, chain.fracs,
                                           omega, M_mut, chain.N)
    return dict(kl_eq=kl_eq, basin11=basin11, affinity=aff, mfpt=mfpt,
                Fstar=Fstar, mean_logW=parts["mean_logW"],
                info=parts["dkl"] / chain.N)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float, nargs="+",
                    default=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
    ap.add_argument("--u", type=float, nargs="+", default=[1e-4, 2e-4])
    ap.add_argument("--N", type=int, nargs="+", default=[16, 24])
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", type=Path,
                    default=SCRIPT_DIR / "results" /
                    f"compare_sweep_{datetime.now():%Y%m%d}")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    rows = []
    data = {}   # (N, u) -> dict of arrays
    keys = ("a", "FA", "FB", "gap", "logWA", "logWB", "infoA", "infoB",
            "klA", "klB", "b11A", "b11B", "mfptA", "mfptB", "affA", "affB", "ratio")
    for N, u in product(args.N, args.u):
        print(f"\n# N={N}  u=v={u:g}  building neutral chain ...", flush=True)
        chain = build_chain(N, u, u, args.r, args.n_jobs)
        cm = chain.phi[chain.cls == 1].sum()
        print(f"   S={len(chain.phi)}  phi corner mass = {cm:.4f}", flush=True)
        d = {k: [] for k in keys}
        for a in args.a:
            rA = one_cell(chain, a, "A", args.n_jobs)
            rB = one_cell(chain, a, "B", args.n_jobs)
            gap = rB["Fstar"] - rA["Fstar"]
            ratio = rA["mfpt"] / rB["mfpt"] if rB["mfpt"] > 0 else np.nan
            print(f"   a={a:.3f}: F* A={rA['Fstar']:.5f} B={rB['Fstar']:.5f} "
                  f"gap(B-A)={gap:+.5f} | info A={rA['info']:.5f} B={rB['info']:.5f} "
                  f"| KL_eq A={rA['kl_eq']:.3f} B={rB['kl_eq']:.3f}", flush=True)
            d["a"].append(a)
            d["FA"].append(rA["Fstar"]); d["FB"].append(rB["Fstar"])
            d["gap"].append(gap)
            d["logWA"].append(rA["mean_logW"]); d["logWB"].append(rB["mean_logW"])
            d["infoA"].append(rA["info"]); d["infoB"].append(rB["info"])
            d["klA"].append(rA["kl_eq"]); d["klB"].append(rB["kl_eq"])
            d["b11A"].append(rA["basin11"]); d["b11B"].append(rB["basin11"])
            d["mfptA"].append(rA["mfpt"]); d["mfptB"].append(rB["mfpt"])
            d["affA"].append(rA["affinity"]); d["affB"].append(rB["affinity"])
            d["ratio"].append(ratio)
            rows.append(dict(N=N, u=u, a=a, phi_corner=cm, gap=gap, mfpt_ratio=ratio,
                             **{f"A_{k}": v for k, v in rA.items()},
                             **{f"B_{k}": v for k, v in rB.items()}))
        data[(N, u)] = {k: np.array(v) for k, v in d.items()}

    # CSV
    with (out / "sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    combos = list(data.keys())
    cmap = plt.cm.viridis(np.linspace(0.12, 0.88, max(len(combos), 1)))

    # ---- primary figure: free fitness ----
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    for c, (N, u) in zip(cmap, combos):
        lab = f"N={N}, u={u:g}"
        ax.plot(data[(N, u)]["a"], data[(N, u)]["FA"], "o-", color=c, lw=2,
                label=f"A  {lab}")
        ax.plot(data[(N, u)]["a"], data[(N, u)]["FB"], "s--", color=c, lw=1.4,
                alpha=0.75)
    ax.set_xlabel(r"selection strength $a=s_1$ ($\alpha$)")
    ax.set_ylabel(r"$F^*=\langle\log W\rangle_{\psi^*}-\frac{1}{N}D_{KL}(\psi^*\|\varphi)$")
    ax.set_title("Equilibrium free fitness (A solid, B dashed)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

    ax = axes[1]
    for c, (N, u) in zip(cmap, combos):
        ax.plot(data[(N, u)]["a"], data[(N, u)]["gap"], "o-", color=c, lw=2,
                label=f"N={N}, u={u:g}")
    ax.axhline(0.0, color="0.5", ls="--", lw=1)
    ax.set_xlabel(r"selection strength $a=s_1$ ($\alpha$)")
    ax.set_ylabel(r"free-fitness gap  $F^*_B - F^*_A$ [nats]")
    ax.set_title("Gap (additive $-$ sign epistasis) vs $\\alpha$")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[2]
    for c, (N, u) in zip(cmap, combos):
        dW = data[(N, u)]["logWB"] - data[(N, u)]["logWA"]
        dInfo = data[(N, u)]["infoA"] - data[(N, u)]["infoB"]  # info cost saved by B
        ax.plot(data[(N, u)]["a"], dW, "o-", color=c, lw=2,
                label=f"$\\Delta\\langle\\log W\\rangle$  N={N},u={u:g}")
        ax.plot(data[(N, u)]["a"], dInfo, "^:", color=c, lw=1.6, alpha=0.8)
    ax.axhline(0.0, color="0.5", ls="--", lw=1)
    ax.set_xlabel(r"selection strength $a=s_1$ ($\alpha$)")
    ax.set_ylabel("gap contributions [nats]")
    ax.set_title(r"Decomposition: $\Delta\langle\log W\rangle$ (o) vs $\Delta$info/N ($\triangle$)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

    fig.suptitle(f"Free-fitness gap A(sign) vs B(additive), r={args.r}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "sweep_freefitness.png", dpi=150)
    plt.close(fig)

    # ---- secondary figure: dynamics/equilibrium summaries ----
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
    ax = axes[0]
    for c, (N, u) in zip(cmap, combos):
        ax.plot(data[(N, u)]["a"], data[(N, u)]["ratio"], "o-", color=c, lw=2,
                label=f"N={N}, u={u:g}")
    ax.axhline(1.0, color="0.5", ls="--", lw=1)
    ax.set_xlabel(r"$a=s_1$"); ax.set_ylabel("MFPT ratio  A/B")
    ax.set_title("Sign epistasis is slower (A/B > 1)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    for c, (N, u) in zip(cmap, combos):
        ax.plot(data[(N, u)]["a"], data[(N, u)]["affA"], "o-", color=c, lw=2,
                label=f"A  N={N},u={u:g}")
        ax.plot(data[(N, u)]["a"], data[(N, u)]["affB"], "s--", color=c, lw=1.4,
                alpha=0.7)
    ax.axhline(0.0, color="0.5", ls="--", lw=1)
    ax.set_xlabel(r"$a=s_1$"); ax.set_ylabel("cycle affinity (nats)")
    ax.set_title("Circulation: A (solid) grows, B (dashed) ~ 0")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

    ax = axes[2]
    for c, (N, u) in zip(cmap, combos):
        ax.plot(data[(N, u)]["a"], data[(N, u)]["klA"], "o-", color=c, lw=2,
                label=f"A  N={N},u={u:g}")
        ax.plot(data[(N, u)]["a"], data[(N, u)]["klB"], "s--", color=c, lw=1.4,
                alpha=0.7)
    ax.set_xlabel(r"$a=s_1$"); ax.set_ylabel(r"$KL_{eq}$ (bits)")
    ax.set_title("Equilibrium score (A solid, B dashed)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

    fig.suptitle(f"Robustness of A(sign) vs B(additive), r={args.r}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "sweep_summary.png", dpi=150)
    plt.close(fig)

    print(f"\nDone in {time.perf_counter()-t0:.1f}s. Outputs in {out}")


if __name__ == "__main__":
    main()
