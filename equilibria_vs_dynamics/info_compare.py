#!/usr/bin/env python3
r"""
Equilibrium information  I = D_KL(psi* || phi)  for A (sign epistasis) vs
B (additive), as a function of the mutation rate u=v (fixed a) and of the
selection strength a (fixed u).  N fixed.  Stationary distributions only.

Question: do A and B carry the same information?  (They do not: A > B always.)

Run:
  conda activate miso
  cd equilibria_vs_dynamics
  python info_compare.py --N 30 --a-fixed 0.05 --u-fixed 2e-4
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from node_jump_analysis import Landscape, build_chain, selection_P  # noqa: E402
from simplex import kl_divergence_bits, stationary_fast  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
CA, CB = "#c44e52", "#4c72b0"


def info_AB(chain, a: float, n_jobs: int) -> tuple[float, float]:
    LA = Landscape("A", "A", s1=a, s2=-a, eps12=2 * a, blurb="")
    LB = Landscape("B", "B", s1=a, s2=a, eps12=0.0, blurb="")
    iA = kl_divergence_bits(stationary_fast(selection_P(chain, LA, n_jobs)), chain.phi)
    iB = kl_divergence_bits(stationary_fast(selection_P(chain, LB, n_jobs)), chain.phi)
    return iA, iB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--a-fixed", type=float, default=0.05)
    ap.add_argument("--u-fixed", type=float, default=2e-4)
    ap.add_argument("--u", type=float, nargs="+",
                    default=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3])
    ap.add_argument("--a", type=float, nargs="+",
                    default=[0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", type=Path,
                    default=SCRIPT_DIR / "results" /
                    f"info_compare_{datetime.now():%Y%m%d}")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    N = args.N

    # ---- vs u (fixed a) ----
    uA, uB = [], []
    print(f"# vs u   (a={args.a_fixed}, N={N})")
    for u in args.u:
        ch = build_chain(N, u, u, args.r, args.n_jobs)
        iA, iB = info_AB(ch, args.a_fixed, args.n_jobs)
        uA.append(iA); uB.append(iB)
        print(f"  u={u:.0e}: A={iA:.4f} B={iB:.4f} gap={iA-iB:+.4f} bits")
    uA, uB = np.array(uA), np.array(uB)

    # ---- vs a (fixed u) ----
    aA, aB = [], []
    print(f"# vs a   (u={args.u_fixed:g}, N={N})")
    ch = build_chain(N, args.u_fixed, args.u_fixed, args.r, args.n_jobs)
    for a in args.a:
        iA, iB = info_AB(ch, a, args.n_jobs)
        aA.append(iA); aB.append(iB)
        print(f"  a={a:.3f}: A={iA:.4f} B={iB:.4f} gap={iA-iB:+.4f} bits")
    aA, aB = np.array(aA), np.array(aB)

    # ---- figure: 2 rows (info; gap) x 2 cols (vs u; vs a) ----
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))
    (ax_u, ax_a), (ax_ug, ax_ag) = axes

    ax_u.plot(args.u, uA, "o-", color=CA, lw=2, label="A: sign epistasis")
    ax_u.plot(args.u, uB, "s-", color=CB, lw=2, label="B: additive")
    ax_u.set_xscale("log")
    ax_u.set_xlabel("mutation rate $u=v$")
    ax_u.set_ylabel(r"information $D_{KL}(\psi^*\|\varphi)$ [bits]")
    ax_u.set_title(f"Information vs $u$   (a={args.a_fixed:g}, N={N})")
    ax_u.grid(True, alpha=0.3); ax_u.legend(fontsize=9)

    ax_ug.plot(args.u, uA - uB, "^-", color="#333333", lw=2)
    ax_ug.axhline(0.0, color="0.6", ls="--", lw=1)
    ax_ug.set_xscale("log")
    ax_ug.set_xlabel("mutation rate $u=v$")
    ax_ug.set_ylabel(r"info gap  $I_A - I_B$ [bits]")
    ax_ug.set_title("A carries MORE information; gap grows with $u$")
    ax_ug.grid(True, alpha=0.3)

    ax_a.plot(args.a, aA, "o-", color=CA, lw=2, label="A: sign epistasis")
    ax_a.plot(args.a, aB, "s-", color=CB, lw=2, label="B: additive")
    ax_a.set_xlabel(r"selection strength $a=s_1$")
    ax_a.set_ylabel(r"information $D_{KL}(\psi^*\|\varphi)$ [bits]")
    ax_a.set_title(f"Information vs $a$   (u={args.u_fixed:g}, N={N})")
    ax_a.grid(True, alpha=0.3); ax_a.legend(fontsize=9)

    ax_ag.plot(args.a, aA - aB, "^-", color="#333333", lw=2)
    ax_ag.axhline(0.0, color="0.6", ls="--", lw=1)
    ax_ag.set_xlabel(r"selection strength $a=s_1$")
    ax_ag.set_ylabel(r"info gap  $I_A - I_B$ [bits]")
    ax_ag.set_title("Gap peaks at weak selection, vanishes as $a$ grows")
    ax_ag.grid(True, alpha=0.3)

    fig.suptitle("Equilibrium information: sign epistasis (A) vs additive (B) "
                 "-- they are NOT equal", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / f"info_compare_N{N}.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved {out / f'info_compare_N{N}.png'}")


if __name__ == "__main__":
    main()
