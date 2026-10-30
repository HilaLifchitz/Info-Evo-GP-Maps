#!/usr/bin/env python3
r"""
Sweep selection strength `a` and, for each case (A..E), report at the analytic
DIFFUSION / detailed-balance equilibrium (Boltzmann reference psi*_boltz):

  - mean log-fitness         mlf   = E_{psi*}[ sum_g x_g log w_g ]   (population)
  - log mean-fitness         lwb   = E_{psi*}[ log sum_g x_g w_g ]
  - information              Info  = D_KL(psi*_boltz || phi*_boltz)  [bits]
  - free fitness             F*    = lwb - (1/N) Info                [nats]
  - concentration of mass:
        corner_mass = mass on monomorphic (corner) lattice states
        PR          = 1 / sum_x psi(x)^2   (effective # occupied lattice points)
        xbar        = mean genotype composition E_{psi*}[x]  (length 4)
        top_geno    = max_g xbar_g  and which genotype

The 2-loci Markov-chain equilibrium corner mass is also printed for comparison.

Run:
  conda activate miso
  cd equilibria_vs_dynamics
  python diffusion_sweep_a.py --N 30 --u 0.005 --v 0.005 --r 0.0 \
      --a 0.02 0.05 0.1 0.2 0.35 0.5
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from math import log
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from node_jump_analysis import build_chain, selection_P  # noqa: E402
from adaptation_per_a import (  # noqa: E402
    CASE_COLORS, CASE_DEFS, CASE_ORDER, all_landscapes,
)
from compare_two_cases import boltz_reference, free_fitness5_asymptote  # noqa: E402
from simplex import (  # noqa: E402
    build_sigma, fitness_exp, log_mean_fitness_genotype_simplex,
    mutation_matrix_asymmetric, stationary_fast,
)

SCRIPT_DIR = Path(__file__).resolve().parent
LN2 = log(2.0)
GENO = ["00", "01", "10", "11"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--u", type=float, default=0.005)
    ap.add_argument("--v", type=float, default=0.005)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--a", type=float, nargs="+",
                    default=[0.02, 0.05, 0.1, 0.2, 0.35, 0.5])
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", type=Path,
                    default=SCRIPT_DIR / "results" /
                    f"diffusion_sweep_a_{datetime.now():%Y%m%d}")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print(f"# N={args.N}, u={args.u:g}, v={args.v:g}, r={args.r:g}", flush=True)
    chain = build_chain(args.N, args.u, args.v, args.r, args.n_jobs)
    fracs = chain.fracs
    corner = (chain.cls == 1)
    M_mut = mutation_matrix_asymmetric(np.array([args.u, args.u]),
                                       np.array([args.v, args.v]))
    omega_neutral = fitness_exp(np.zeros((2, 2)))
    phi_boltz, _ = boltz_reference(fracs, omega_neutral, M_mut, args.N)
    print(f"  S={len(chain.phi)}  chain-phi corner={chain.phi[corner].sum():.4f}"
          f"  boltz-phi corner={phi_boltz[corner].sum():.4f}"
          f"  ({time.perf_counter()-t0:.1f}s)", flush=True)

    # res[a][slug] = dict(...)
    res: dict[float, dict[str, dict]] = {}
    for a in args.a:
        res[a] = {}
        for L in all_landscapes(a):
            omega = fitness_exp(build_sigma(L.s1, L.s2, L.eps12))
            logmf = log_mean_fitness_genotype_simplex(fracs, omega)   # (S,)
            per_state_mlf = fracs @ L.logfit                          # (S,)

            psi_b, _ = boltz_reference(fracs, omega, M_mut, args.N)
            Fb, pb = free_fitness5_asymptote(psi_b, phi_boltz, fracs, omega,
                                             M_mut, args.N)
            xbar_b = psi_b @ fracs
            psi_c = stationary_fast(selection_P(chain, L, args.n_jobs))
            xbar_c = psi_c @ fracs

            res[a][L.slug] = dict(
                mlf=float(psi_b @ per_state_mlf),
                lwb=float(psi_b @ logmf),
                info=pb["dkl"] / LN2,
                F=Fb,
                corner_b=float(psi_b[corner].sum()),
                PR_b=float(1.0 / np.sum(psi_b ** 2)),
                xbar_b=xbar_b,
                top_b=float(xbar_b.max()),
                topg_b=GENO[int(xbar_b.argmax())],
                corner_c=float(psi_c[corner].sum()),
                info_c=free_fitness5_asymptote(psi_c, chain.phi, fracs, omega,
                                               M_mut, args.N)[1]["dkl"] / LN2,
            )
        # ---- per-a table ----
        print(f"\n=== a={a:g}  (DIFFUSION / Boltzmann equilibrium) ===")
        print(f"  {'case':22s} {'mlf':>8s} {'logwbar':>8s} {'Info':>7s} "
              f"{'F*':>8s} {'corner':>7s} {'PR':>7s} {'top geno (freq)':>16s}")
        for s in CASE_ORDER:
            d = res[a][s]
            print(f"  {s+' '+CASE_DEFS[s][0]:22s} {d['mlf']:8.4f} {d['lwb']:8.4f} "
                  f"{d['info']:7.3f} {d['F']:8.4f} {d['corner_b']:7.3f} "
                  f"{d['PR_b']:7.1f} {d['topg_b']:>6s} ({d['top_b']:.3f})")

    # ---- figure: metrics vs a, one line per case (diffusion solid) ----
    labels = {s: f"{s}: {CASE_DEFS[s][0]}" for s in CASE_ORDER}
    avals = np.array(args.a, dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.2))
    specs = [
        ("mlf",     r"mean log-fitness $\langle\sum_g x_g\log w_g\rangle$"),
        ("info",    r"information $D_{KL}(\psi^*\|\varphi^*)$ [bits]"),
        ("F",       r"free fitness $F^*=\langle\log\bar w\rangle-\frac{1}{N}$Info [nats]"),
        ("corner_b", r"corner (monomorphic) mass"),
        ("PR_b",    r"participation ratio $1/\sum_x\psi^2$ (eff. # states)"),
        ("top_b",   r"dominant genotype mean freq $\max_g\bar x_g$"),
    ]
    for ax, (key, ylab) in zip(axes.flat, specs):
        for s in CASE_ORDER:
            ys = [res[a][s][key] for a in args.a]
            ax.plot(avals, ys, "-o", ms=4, lw=1.8, color=CASE_COLORS[s],
                    label=labels[s])
        ax.set_xlabel("selection strength $a$"); ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"Diffusion (detailed-balance) equilibrium vs selection strength "
                 f"a   (N={args.N}, u={args.u:g}, v={args.v:g}, r={args.r:g})",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png = out / f"diffusion_sweep_a_N{args.N}_u{args.u:g}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"\nDone in {time.perf_counter()-t0:.1f}s. Figure: {png}")


if __name__ == "__main__":
    main()
