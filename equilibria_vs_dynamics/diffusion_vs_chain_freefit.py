#!/usr/bin/env python3
r"""
Equilibrium free fitness and information for the 5 cases (A,B,C,D,E), computed
TWO ways and compared for consistency of the A..E ordering:

  (1) diffusion / detailed-balance (analytic Boltzmann reference)
        psi*_boltz  propto  exp( N log wbar(x) + sum_g (alpha_g - 1) log x_g )
        phi*_boltz  = same with omega == 1   (neutral Dirichlet on the simplex)
        Info_diff = D_KL(psi*_boltz || phi*_boltz)
        F*_diff   = <log wbar>_{psi*_boltz} - (1/N) Info_diff        [logB dropped]

  (2) 2-loci Wright-Fisher Markov chain (the "make-do" free fitness -- the chain is
      NOT reversible, so this F* is not a true thermodynamic potential)
        psi*_chain = stationary of selection_P ;  phi_chain = neutral chain stationary
        Info_chain = D_KL(psi*_chain || phi_chain)
        F*_chain   = <log wbar>_{psi*_chain} - (1/N) Info_chain

Both use the SAME formula free_fitness5_asymptote(psi*, phi, ...); only psi*/phi differ.

Run:
  conda activate miso
  cd equilibria_vs_dynamics
  python diffusion_vs_chain_freefit.py --N 30 --u 0.005 --v 0.005 --r 0.0
"""

from __future__ import annotations

import argparse
import time
from math import log

import numpy as np

from node_jump_analysis import build_chain, selection_P  # noqa: E402
from adaptation_per_a import CASE_DEFS, CASE_ORDER, all_landscapes  # noqa: E402
from compare_two_cases import boltz_reference, free_fitness5_asymptote  # noqa: E402
from simplex import (  # noqa: E402
    build_sigma, fitness_exp, mutation_matrix_asymmetric, stationary_fast,
)

LN2 = log(2.0)


def rank_str(order_by_value: list[str]) -> str:
    return " > ".join(order_by_value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--u", type=float, default=0.005)
    ap.add_argument("--v", type=float, default=0.005)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--a", type=float, default=0.05, help="selection magnitude")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()
    t0 = time.perf_counter()

    print(f"# N={args.N}, u={args.u:g}, v={args.v:g}, r={args.r:g}, a={args.a:g}", flush=True)
    chain = build_chain(args.N, args.u, args.v, args.r, args.n_jobs)
    fracs = chain.fracs
    M_mut = mutation_matrix_asymmetric(np.array([args.u, args.u]),
                                       np.array([args.v, args.v]))
    cm = float(chain.phi[chain.cls == 1].sum())
    print(f"  S={len(chain.phi)}  chain-phi corner mass={cm:.4f}"
          f"  ({time.perf_counter()-t0:.1f}s)", flush=True)

    # neutral Boltzmann reference phi* (omega == 1)
    omega_neutral = fitness_exp(np.zeros((2, 2)))
    phi_boltz, _ = boltz_reference(fracs, omega_neutral, M_mut, args.N)

    rows = {}
    for L in all_landscapes(args.a):
        omega = fitness_exp(build_sigma(L.s1, L.s2, L.eps12))
        # (1) diffusion / detailed balance
        psi_boltz, logZ = boltz_reference(fracs, omega, M_mut, args.N)
        F_diff, p_diff = free_fitness5_asymptote(psi_boltz, phi_boltz, fracs,
                                                 omega, M_mut, args.N)
        # (2) 2-loci Markov chain
        P = selection_P(chain, L, args.n_jobs)
        psi_chain = stationary_fast(P)
        F_ch, p_ch = free_fitness5_asymptote(psi_chain, chain.phi, fracs,
                                             omega, M_mut, args.N)
        rows[L.slug] = dict(
            F_diff=F_diff, info_diff=p_diff["dkl"] / LN2, mlw_diff=p_diff["mean_logW"],
            F_ch=F_ch,    info_ch=p_ch["dkl"] / LN2,      mlw_ch=p_ch["mean_logW"],
        )
        print(f"  {L.slug} ({CASE_DEFS[L.slug][0]:>20s}) done  "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)

    # ---- report -----------------------------------------------------------
    def tab(kind_F, kind_I, kind_M, title):
        print(f"\n=== {title} ===")
        print(f"  {'case':22s} {'<log wbar>':>11s} {'Info[bits]':>11s} {'F*[nats]':>11s}")
        for s in CASE_ORDER:
            r = rows[s]
            print(f"  {s+' '+CASE_DEFS[s][0]:22s} {r[kind_M]:11.5f} "
                  f"{r[kind_I]:11.5f} {r[kind_F]:11.5f}")
        by_F = sorted(CASE_ORDER, key=lambda s: rows[s][kind_F], reverse=True)
        by_I = sorted(CASE_ORDER, key=lambda s: rows[s][kind_I], reverse=True)
        print(f"  free-fitness ranking: {rank_str(by_F)}")
        print(f"  information  ranking: {rank_str(by_I)}")
        return by_F, by_I

    fD, iD = tab("F_diff", "info_diff", "mlw_diff", "DIFFUSION (Boltzmann, detailed balance)")
    fC, iC = tab("F_ch",  "info_ch",  "mlw_ch",  "2-loci Markov chain (make-do)")

    print("\n=== ordering match ? ===")
    print(f"  free fitness : diffusion {rank_str(fD)}   |   chain {rank_str(fC)}   "
          f"->  {'MATCH' if fD==fC else 'DIFFER'}")
    print(f"  information  : diffusion {rank_str(iD)}   |   chain {rank_str(iC)}   "
          f"->  {'MATCH' if iD==iC else 'DIFFER'}")
    print(f"\nDone in {time.perf_counter()-t0:.1f}s.")


if __name__ == "__main__":
    main()
