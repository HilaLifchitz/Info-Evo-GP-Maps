#!/usr/bin/env python3
r"""
BIG equilibrium sweep over (a, u, v, r) for the 5 cases (A,B,C,D,E), N fixed.

For every (u,v,r,a,case) it records equilibrium quantities computed TWO ways:
  - diffusion / detailed-balance (analytic Boltzmann reference psi*_boltz, phi*_boltz)
  - 2-loci Wright-Fisher Markov chain (psi*_chain, phi_chain), which is NOT reversible

Quantities (per row):
  mean log-fitness   mlf  = E_{psi*}[ sum_g x_g log w_g ]        (population)
  log mean-fitness   lwb  = E_{psi*}[ log sum_g x_g w_g ]
  information        info = D_KL(psi* || phi)                    [bits]
  free fitness (several conventions, all in nats):
      F  = lwb - info_nats/N      (log-mean-fitness form)   -> F_diff, F_chain
      G  = mlf - info_nats/N      (pop mean-log-fit form)   -> G_diff, G_chain
      Fboltz = logZ/N             (Boltzmann partition form, diffusion only)
  concentration:
      vtx    mass on exact monomorphic vertices (lattice-only)
      near90 P(max_g x_g >= 0.90)      near99 P(>=0.99)        (continuum-valid)
      domf   max_g E[x_g]              (continuum-valid moment)
      x00,x01,x10,x11  mean composition E[x]

All rows are written incrementally to <out>/sweep_data.csv (rewritten after each
(u,v,r) block) so it can always be re-plotted, plus a final .npz snapshot.

Run (full ~few-hour grid):
  conda activate miso
  cd equilibria_vs_dynamics
  python big_sweep.py --N 30
Quick test:
  python big_sweep.py --N 30 --quick
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from math import log
from pathlib import Path

import numpy as np
import pandas as pd

from node_jump_analysis import build_chain, selection_P
from adaptation_per_a import CASE_DEFS, CASE_ORDER, all_landscapes
from compare_two_cases import boltz_reference
from simplex import (
    build_sigma, fitness_exp, log_mean_fitness_genotype_simplex,
    mutation_matrix_asymmetric, stationary_fast,
)
from f2b_reference import kl_divergence_nats

SCRIPT_DIR = Path(__file__).resolve().parent
LN2 = log(2.0)
GENO = ["00", "01", "10", "11"]


def conc(psi, fracs, corner, xmax):
    xbar = psi @ fracs
    return dict(
        vtx=float(psi[corner].sum()),
        near90=float(psi[xmax >= 0.90].sum()),
        near99=float(psi[xmax >= 0.99].sum()),
        domf=float(xbar.max()),
        x00=float(xbar[0]), x01=float(xbar[1]),
        x10=float(xbar[2]), x11=float(xbar[3]),
    )


def build_grid(quick: bool):
    """Return (a_vals, list of (u,v,r,kind))."""
    if quick:
        a_vals = [0.05, 0.2]
        sym = [5e-3]
        asym = [(2e-2, 2e-3)]
        r_vals = [0.0, 0.5]
    else:
        a_vals = [0.02, 0.03, 0.04, 0.05, 0.07, 0.09, 0.11, 0.14, 0.17,
                  0.20, 0.24, 0.28, 0.32, 0.37, 0.42, 0.46, 0.50]
        sym = [1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3, 3e-3, 5e-3,
               1e-2, 2e-2, 3e-2, 5e-2]
        asym = [(2e-2, 2e-3), (2e-3, 2e-2), (1e-2, 1e-3), (1e-3, 1e-2),
                (4e-2, 1e-3), (1e-3, 4e-2), (5e-2, 5e-3), (5e-3, 5e-2),
                (2e-2, 5e-4), (5e-4, 2e-2), (1e-2, 1e-4)]
        r_vals = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]
    combos = []
    for r in r_vals:
        for u in sym:
            combos.append((u, u, r, "sym"))
        for (u, v) in asym:
            combos.append((u, v, r, "asym"))
    return a_vals, combos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", type=Path,
                    default=SCRIPT_DIR / "results" /
                    f"big_sweep_{datetime.now():%Y%m%d}")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "sweep_data.csv"
    t0 = time.perf_counter()

    a_vals, combos = build_grid(args.quick)
    n_total = len(combos) * len(a_vals) * len(CASE_ORDER)
    print(f"# N={args.N}  |combos(u,v,r)|={len(combos)}  |a|={len(a_vals)}  "
          f"cases={len(CASE_ORDER)}  -> {n_total} rows", flush=True)

    rows = []
    done = 0
    for ci, (u, v, r, kind) in enumerate(combos):
        tc = time.perf_counter()
        chain = build_chain(args.N, u, v, r, args.n_jobs)
        fracs = chain.fracs
        corner = (chain.cls == 1)
        xmax = fracs.max(axis=1)
        M_mut = mutation_matrix_asymmetric(np.array([u, u]), np.array([v, v]))
        omega_neutral = fitness_exp(np.zeros((2, 2)))
        phi_boltz, _ = boltz_reference(fracs, omega_neutral, M_mut, args.N)
        phi_chain = chain.phi
        corner_phi_chain = float(phi_chain[corner].sum())
        corner_phi_boltz = float(phi_boltz[corner].sum())

        for a in a_vals:
            for L in all_landscapes(a):
                if L.slug not in CASE_ORDER:
                    continue
                omega = fitness_exp(build_sigma(L.s1, L.s2, L.eps12))
                logmf = log_mean_fitness_genotype_simplex(fracs, omega)
                psm = fracs @ L.logfit

                psi_b, logZ = boltz_reference(fracs, omega, M_mut, args.N)
                dkl_b = kl_divergence_nats(psi_b, phi_boltz)
                lwb_b = float(psi_b @ logmf); mlf_b = float(psi_b @ psm)

                psi_c = stationary_fast(selection_P(chain, L, args.n_jobs))
                dkl_c = kl_divergence_nats(psi_c, phi_chain)
                lwb_c = float(psi_c @ logmf); mlf_c = float(psi_c @ psm)

                cb = conc(psi_b, fracs, corner, xmax)
                cc = conc(psi_c, fracs, corner, xmax)

                rec = dict(
                    N=args.N, u=u, v=v, r=r, mut_kind=kind, a=a, case=L.slug,
                    case_name=CASE_DEFS[L.slug][0],
                    s1=L.s1, s2=L.s2, eps12=L.eps12,
                    logfit00=L.logfit[0], logfit01=L.logfit[1],
                    logfit10=L.logfit[2], logfit11=L.logfit[3],
                    corner_phi_chain=corner_phi_chain,
                    corner_phi_boltz=corner_phi_boltz,
                    # diffusion
                    mlf_diff=mlf_b, lwb_diff=lwb_b, info_diff=dkl_b / LN2,
                    F_diff=lwb_b - dkl_b / args.N, G_diff=mlf_b - dkl_b / args.N,
                    Fboltz=logZ / args.N, logZ=logZ,
                    **{f"{k}_diff": val for k, val in cb.items()},
                    # chain
                    mlf_chain=mlf_c, lwb_chain=lwb_c, info_chain=dkl_c / LN2,
                    F_chain=lwb_c - dkl_c / args.N, G_chain=mlf_c - dkl_c / args.N,
                    **{f"{k}_chain": val for k, val in cc.items()},
                )
                rows.append(rec)
                done += 1

        # incremental save after each (u,v,r) block
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        el = time.perf_counter() - t0
        rate = done / el
        eta = (n_total - done) / rate if rate > 0 else float("nan")
        print(f"  [{ci+1:2d}/{len(combos)}] u={u:g} v={v:g} r={r:g} "
              f"({kind}) block {time.perf_counter()-tc:.1f}s | "
              f"{done}/{n_total} rows | elapsed {el/60:.1f}m | ETA {eta/60:.1f}m",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    np.savez(out / "sweep_data.npz",
             **{c: df[c].to_numpy() for c in df.columns})
    print(f"\nDone {done} rows in {(time.perf_counter()-t0)/60:.1f} min.")
    print(f"Data: {csv_path}")


if __name__ == "__main__":
    main()
