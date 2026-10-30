#!/usr/bin/env python3
r"""
Equilibrium summary across the FOUR landscapes (A sign, B additive, C reciprocal
sign, D neutral intermediates), all sharing 00=0 and 11=2a.

Two scalar equilibrium quantities per (case):
  * information   I  = D_KL(psi* || phi)                        [bits]
  * free fitness  F* = <log W>_{psi*} - (1/N) D_KL(psi* || phi) [nats]

Plotted vs selection strength a (fixed u) and vs mutation rate u=v (fixed a),
one line per case.  Stationary distributions only (fast).

Run:
  conda activate miso
  cd equilibria_vs_dynamics
  python summary_4cases.py --N 30 --a-fixed 0.05 --u-fixed 2e-4
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from node_jump_analysis import build_chain, selection_P  # noqa: E402
from adaptation_per_a import CASE_COLORS, CASE_DEFS, CASE_ORDER, all_landscapes  # noqa: E402
from compare_two_cases import free_fitness5_asymptote  # noqa: E402
from simplex import (  # noqa: E402
    build_sigma, fitness_exp, kl_divergence_bits,
    mutation_matrix_asymmetric, stationary_fast,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def metrics(chain, L) -> tuple[float, float]:
    """Return (information [bits], free fitness F* [nats]) for landscape L."""
    psi_star = stationary_fast(selection_P(chain, L, -1))
    info = kl_divergence_bits(psi_star, chain.phi)
    omega = fitness_exp(build_sigma(L.s1, L.s2, L.eps12))
    M_mut = mutation_matrix_asymmetric(np.array([chain.u, chain.u]),
                                       np.array([chain.v, chain.v]))
    Fstar, _ = free_fitness5_asymptote(psi_star, chain.phi, chain.fracs,
                                       omega, M_mut, chain.N)
    return info, Fstar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--a-fixed", type=float, default=0.05)
    ap.add_argument("--u-fixed", type=float, default=2e-4)
    ap.add_argument("--a", type=float, nargs="+",
                    default=[0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--u", type=float, nargs="+",
                    default=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3])
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--out-dir", type=Path,
                    default=SCRIPT_DIR / "results" /
                    f"summary_4cases_{datetime.now():%Y%m%d}")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    N = args.N

    # vs a (fixed u): one chain reused
    infoA = {s: [] for s in CASE_ORDER}
    freeA = {s: [] for s in CASE_ORDER}
    print(f"# vs a  (u={args.u_fixed:g}, N={N})")
    ch = build_chain(N, args.u_fixed, args.u_fixed, args.r, -1)
    for a in args.a:
        for L in all_landscapes(a):
            i, f = metrics(ch, L)
            infoA[L.slug].append(i); freeA[L.slug].append(f)
        print(f"  a={a:.3f}: " + "  ".join(
            f"{s} I={infoA[s][-1]:.3f} F*={freeA[s][-1]:.4f}" for s in CASE_ORDER))

    # vs u (fixed a): rebuild chain per u
    infoU = {s: [] for s in CASE_ORDER}
    freeU = {s: [] for s in CASE_ORDER}
    print(f"# vs u  (a={args.a_fixed:g}, N={N})")
    for u in args.u:
        ch = build_chain(N, u, u, args.r, -1)
        for L in all_landscapes(args.a_fixed):
            i, f = metrics(ch, L)
            infoU[L.slug].append(i); freeU[L.slug].append(f)
        print(f"  u={u:.0e}: " + "  ".join(
            f"{s} I={infoU[s][-1]:.3f} F*={freeU[s][-1]:.4f}" for s in CASE_ORDER))

    labels = {s: f"{s}: {CASE_DEFS[s][0]}" for s in CASE_ORDER}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    (ax_ia, ax_iu), (ax_fa, ax_fu) = axes

    # free fitness shown RELATIVE to case B (additive), the highest, so the tiny
    # cross-case differences are visible (absolute F* is dominated by the 2a growth)
    freeA_rel = {s: np.array(freeA[s]) - np.array(freeA["B"]) for s in CASE_ORDER}
    freeU_rel = {s: np.array(freeU[s]) - np.array(freeU["B"]) for s in CASE_ORDER}
    for s in CASE_ORDER:
        c = CASE_COLORS[s]
        ax_ia.plot(args.a, infoA[s], "o-", color=c, lw=2, label=labels[s])
        ax_fa.plot(args.a, freeA_rel[s], "o-", color=c, lw=2, label=labels[s])
        ax_iu.plot(args.u, infoU[s], "o-", color=c, lw=2, label=labels[s])
        ax_fu.plot(args.u, freeU_rel[s], "o-", color=c, lw=2, label=labels[s])
    ax_fa.axhline(0.0, color="0.6", ls="--", lw=1)
    ax_fu.axhline(0.0, color="0.6", ls="--", lw=1)

    ax_ia.set_xlabel(r"selection strength $a=s_1$")
    ax_ia.set_ylabel(r"information $D_{KL}(\psi^*\|\varphi)$ [bits]")
    ax_ia.set_title(f"Information vs $a$   (u={args.u_fixed:g}, N={N})")
    ax_fa.set_xlabel(r"selection strength $a=s_1$")
    ax_fa.set_ylabel(r"free fitness relative to B  $F^*-F^*_B$ [nats]")
    ax_fa.set_title(f"Free-fitness gap from additive vs $a$   (u={args.u_fixed:g}, N={N})")
    ax_iu.set_xscale("log"); ax_iu.set_xlabel(r"mutation rate $u=v$")
    ax_iu.set_ylabel(r"information $D_{KL}(\psi^*\|\varphi)$ [bits]")
    ax_iu.set_title(f"Information vs $u$   (a={args.a_fixed:g}, N={N})")
    ax_fu.set_xscale("log"); ax_fu.set_xlabel(r"mutation rate $u=v$")
    ax_fu.set_ylabel(r"free fitness relative to B  $F^*-F^*_B$ [nats]")
    ax_fu.set_title(f"Free-fitness gap from additive vs $u$   (a={args.a_fixed:g}, N={N})")
    for ax in (ax_ia, ax_iu, ax_fa, ax_fu):
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("Four-case equilibrium summary  "
                 "(A=sign, B=additive, C=reciprocal sign, D=neutral intermediates)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / f"summary_4cases_N{N}.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved {out / f'summary_4cases_N{N}.png'}")


if __name__ == "__main__":
    main()
