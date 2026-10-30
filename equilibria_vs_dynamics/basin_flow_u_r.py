#!/usr/bin/env python3
r"""
"Flow between nodes" (basin-from-phi) as a function of mutation rate u=v, and
then as a function of recombination r.

Reproduces the basin_from_phi experiment (start psi(0)=phi, watch corner-basin
mass over time) but:
  PART 1: swept over several u=v values, so you can see how MONOMORPHISM (mass
          pinned to the four corners of the simplex) depends on the mutation
          rate.  Lower u  ->  sharper corners (the SSWM regime); higher u ->
          the neutral cloud smears onto edges/interior and the "nodes" dissolve.
  PART 2: swept over recombination r at fixed u, on two landscapes, to see
          whether linkage relief changes the node-to-node flow.

Sigma (and the fitness table) is printed to screen for every landscape used.

Run:
  conda activate miso
  cd equilibria_vs_dynamics
  python basin_flow_u_r.py
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
    GENO, Landscape, basin_mass, build_chain, evolve_many, sample_times,
    selection_P,
)
from simplex import (  # noqa: E402
    build_sigma, fitness_exp, print_fitness_table, print_sigma,
)

SCRIPT_DIR = Path(__file__).resolve().parent
COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]


def print_landscape(L: Landscape, N: int) -> None:
    Sigma = build_sigma(L.s1, L.s2, L.eps12)
    omega = fitness_exp(Sigma)
    print("=" * 66)
    print(f"LANDSCAPE: {L.name}   (s1={L.s1:+g}, s2={L.s2:+g}, eps12={L.eps12:+g})")
    print("=" * 66)
    print_sigma(Sigma)
    print_fitness_table(Sigma, omega)
    print(f"  log-fitness g^T Sigma g : "
          f"{dict(zip(GENO, np.round(L.logfit, 4)))}\n", flush=True)


def tmax_for_u(u: float, N: int) -> int:
    """A few substitution timescales ~ 1/(N u pi_fix); scaled so each panel
    shows the full transient regardless of u."""
    return int(np.clip(10.0 / (N * u * 0.1), 400, 60000))


def corner_support_masses(phi: np.ndarray, cls: np.ndarray) -> dict[int, float]:
    return {c: float(phi[cls == c].sum()) for c in (1, 2, 3, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=24)
    ap.add_argument("--r", type=float, default=0.0, help="r for the u-sweep")
    ap.add_argument("--u-list", type=float, nargs="+",
                    default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--u-fine", type=float, nargs="+",
                    default=[5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2])
    ap.add_argument("--r-list", type=float, nargs="+", default=[0.0, 0.1, 0.5])
    ap.add_argument("--u-for-r", type=float, default=2e-4)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", type=Path,
                    default=SCRIPT_DIR / "results" /
                    f"basin_flow_u_r_{datetime.now():%Y%m%d}")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # landscape used for the u-sweep: additive clear winner
    L_add = Landscape("Additive clear winner", "additive",
                      s1=0.1, s2=0.1, eps12=0.0,
                      blurb="both single steps uphill, 11 global optimum")
    L_sign = Landscape("Sign epistasis (routed)", "sign",
                       s1=0.15, s2=-0.05, eps12=0.10,
                       blurb="00->01 downhill; route 00->10->11")

    print("\n########## FITNESS COEFFICIENTS (printed for the record) ##########\n")
    print_landscape(L_add, args.N)

    # ------------------------------------------------------------------
    # monomorphism vs u  (neutral-only, cheap)
    # ------------------------------------------------------------------
    print("\n---- PART 0: monomorphism of neutral phi vs u=v ----", flush=True)
    mono = {c: [] for c in (1, 2, 3, 4)}
    theta = []
    for u in args.u_fine:
        chain = build_chain(args.N, u, u, args.r, args.n_jobs)
        m = corner_support_masses(chain.phi, chain.cls)
        for c in (1, 2, 3, 4):
            mono[c].append(m[c])
        theta.append(2 * args.N * u)
        print(f"  u={u:.1e}  theta=2Nu={2*args.N*u:.3f}  "
              f"corner(mono)={m[1]:.4f}  edge={m[2]:.4f}  "
              f"face={m[3]:.2e}  interior={m[4]:.2e}", flush=True)

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    labels = {1: "monomorphic (corners)", 2: "2-genotype (edges)",
              3: "3-genotype (faces)", 4: "interior"}
    cols = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}
    for c in (1, 2, 3, 4):
        ax.plot(args.u_fine, mono[c], "o-", color=cols[c], lw=2, label=labels[c])
    ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("mutation rate $u=v$  (log)")
    ax.set_ylabel(r"mass of neutral $\varphi$ by simplex support size")
    ax.set_title(rf"Monomorphism vs mutation rate ($N={args.N}$, $r={args.r}$)"
                 "\n" r"corners dominate when $u\to 0$ ($\theta=2Nu\ll1$)")
    ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=9)
    # annotate theta on top axis
    axt = ax.twiny(); axt.set_xscale("log")
    axt.set_xlim(ax.get_xlim())
    axt.set_xticks(args.u_fine)
    axt.set_xticklabels([f"{2*args.N*u:.2g}" for u in args.u_fine], fontsize=7)
    axt.set_xlabel(r"$\theta = 2Nu$", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "monomorphism_vs_u.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # PART 1: basin flow from phi, swept over u  (additive landscape)
    # ------------------------------------------------------------------
    print("\n---- PART 1: basin-from-phi flow vs u (additive landscape) ----",
          flush=True)
    nU = len(args.u_list)
    ncol = 3
    nrow = int(np.ceil(nU / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.9 * nrow),
                             squeeze=False)
    for k, u in enumerate(args.u_list):
        chain = build_chain(args.N, u, u, args.r, args.n_jobs)
        P = selection_P(chain, L_add, args.n_jobs)
        cmass = float(chain.phi[chain.cls == 1].sum())
        tmax = tmax_for_u(u, args.N)
        times = sample_times(tmax)
        Psi0 = chain.phi.reshape(-1, 1)
        snaps = evolve_many(P, Psi0, times)
        bm = basin_mass(snaps, chain.basin, 0)
        ax = axes[k // ncol][k % ncol]
        for g in range(4):
            ax.plot(times, bm[:, g], "-", lw=2, color=COLORS[g], label=GENO[g])
        ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
        ax.set_title(rf"$u=v={u:.0e}$,  $\theta=2Nu={2*args.N*u:.2g}$"
                     "\n" rf"corner mass of $\varphi$ = {cmass:.3f}", fontsize=9)
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8, title="basin")
        print(f"  u={u:.1e} corner_mass={cmass:.4f} tmax={tmax}", flush=True)
    for k in range(nU, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("generation $t$")
    for row in axes:
        row[0].set_ylabel("corner-basin mass")
    fig.suptitle("Flow between nodes from $\\psi(0)=\\varphi$, swept over "
                 f"mutation rate  (additive: {L_add.blurb})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "basin_flow_u_sweep.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # PART 2: basin flow from phi, swept over r  (two landscapes)
    # ------------------------------------------------------------------
    for L in (L_add, L_sign):
        print(f"\n---- PART 2: basin-from-phi flow vs r  ({L.name}) ----", flush=True)
        print_landscape(L, args.N)
        nR = len(args.r_list)
        fig, axes = plt.subplots(1, nR, figsize=(5.0 * nR, 4.2), squeeze=False)
        for j, rr in enumerate(args.r_list):
            chain = build_chain(args.N, args.u_for_r, args.u_for_r, rr, args.n_jobs)
            P = selection_P(chain, L, args.n_jobs)
            cmass = float(chain.phi[chain.cls == 1].sum())
            mean_ld = float(chain.phi @ (chain.fracs[:, 0] * chain.fracs[:, 3]
                                         - chain.fracs[:, 1] * chain.fracs[:, 2]))
            tmax = tmax_for_u(args.u_for_r, args.N)
            times = sample_times(tmax)
            snaps = evolve_many(P, chain.phi.reshape(-1, 1), times)
            bm = basin_mass(snaps, chain.basin, 0)
            ax = axes[0][j]
            for g in range(4):
                ax.plot(times, bm[:, g], "-", lw=2, color=COLORS[g],
                        label=f"{GENO[g]} (logw={L.logfit[g]:+.2f})")
            ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel("generation $t$")
            ax.set_title(rf"$r={rr:g}$   corner$(\varphi)$={cmass:.3f}"
                         "\n" rf"mean LD$(\varphi)$={mean_ld:.1e}", fontsize=9)
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
            print(f"  r={rr:g} corner_mass={cmass:.4f} mean_LD(phi)={mean_ld:.3e}",
                  flush=True)
        axes[0][0].set_ylabel("corner-basin mass")
        fig.suptitle(f"Node flow vs recombination r  ({L.name}, "
                     f"u=v={args.u_for_r:g}, N={args.N})", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out / f"basin_flow_r_sweep_{L.slug}.png", dpi=150)
        plt.close(fig)

    print(f"\nDone in {time.perf_counter()-t0:.1f}s. Outputs in {out}")


if __name__ == "__main__":
    main()
