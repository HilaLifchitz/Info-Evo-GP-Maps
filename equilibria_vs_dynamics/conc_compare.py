#!/usr/bin/env python3
r"""
Concentration of the equilibrium mass, CHAIN vs DIFFUSION, with continuum-valid
metrics (since a continuous diffusion is never exactly monomorphic).

For each case (A..E) and each a, at both equilibria:
  psi*_chain  = stationary of selection_P            (discrete Markov chain)
  psi*_boltz  = analytic Boltzmann / detailed balance (diffusion reference)
report:
  vtx    = mass on exact monomorphic vertices (lattice-only notion)
  near90 = P(max_g x_g >= 0.90)   (continuum-valid 'almost fixed')
  near99 = P(max_g x_g >= 0.99)
  domf   = dominant genotype mean freq  max_g E[x_g]   (continuum-valid moment)

Run:
  conda activate miso
  cd equilibria_vs_dynamics
  python conc_compare.py --N 30 --u 0.005 --v 0.005 --r 0.0 --a 0.05 0.2 0.5
"""

from __future__ import annotations

import argparse
from math import log

import numpy as np

from node_jump_analysis import build_chain, selection_P
from adaptation_per_a import CASE_DEFS, CASE_ORDER, all_landscapes
from compare_two_cases import boltz_reference
from simplex import build_sigma, fitness_exp, mutation_matrix_asymmetric, stationary_fast

LN2 = log(2.0)
GENO = ["00", "01", "10", "11"]


def metrics(psi, fracs, corner):
    xmax = fracs.max(axis=1)                      # max_g x_g at each lattice point
    xbar = psi @ fracs
    return dict(
        vtx=float(psi[corner].sum()),
        near90=float(psi[xmax >= 0.90].sum()),
        near99=float(psi[xmax >= 0.99].sum()),
        domf=float(xbar.max()),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--u", type=float, default=0.005)
    ap.add_argument("--v", type=float, default=0.005)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--a", type=float, nargs="+", default=[0.05, 0.2, 0.5])
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    print(f"# N={args.N}, u={args.u:g}, v={args.v:g}, r={args.r:g}")
    chain = build_chain(args.N, args.u, args.v, args.r, args.n_jobs)
    fracs = chain.fracs
    corner = (chain.cls == 1)
    M_mut = mutation_matrix_asymmetric(np.array([args.u, args.u]),
                                       np.array([args.v, args.v]))
    print(f"  S={len(chain.phi)}   # monomorphic vertices={int(corner.sum())}")

    for a in args.a:
        print(f"\n=== a={a:g} ===")
        print(f"  {'case':22s} | {'CHAIN: vtx  near90 near99  domf':34s} | "
              f"{'DIFFUSION: vtx  near90 near99  domf':34s}")
        for s in CASE_ORDER:
            L = next(x for x in all_landscapes(a) if x.slug == s)
            omega = fitness_exp(build_sigma(L.s1, L.s2, L.eps12))
            pc = stationary_fast(selection_P(chain, L, args.n_jobs))
            pb, _ = boltz_reference(fracs, omega, M_mut, args.N)
            mc = metrics(pc, fracs, corner)
            mb = metrics(pb, fracs, corner)
            print(f"  {s+' '+CASE_DEFS[s][0]:22s} | "
                  f"{mc['vtx']:5.3f} {mc['near90']:6.3f} {mc['near99']:6.3f} {mc['domf']:6.3f}"
                  f"        | "
                  f"{mb['vtx']:5.3f} {mb['near90']:6.3f} {mb['near99']:6.3f} {mb['domf']:6.3f}")


if __name__ == "__main__":
    main()
