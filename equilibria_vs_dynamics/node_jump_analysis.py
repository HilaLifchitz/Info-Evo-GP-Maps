#!/usr/bin/env python3
r"""
Node-jump / SSWM view of the two-locus simplex chain.
=====================================================

Question (from the user).
-------------------------
We run the two-locus Wright-Fisher chain as a *distribution* psi(t) over the
4-genotype simplex.  With weak mutation (u << s, N u << 1) the neutral start
phi is concentrated on the FOUR CORNERS of the simplex (monomorphic
populations 00, 01, 10, 11).  Under selection with a clear winner the
equilibrium psi* piles onto the best corner.

Claim to test:  the whole process can be read as a *superposition of
single-population paths* -- each path a nearly-monomorphic population that
hops from corner to corner (the SSWM / strong-selection-weak-mutation
adaptive walk).  The distribution-level master equation is linear, so
"follow one grain of sand" (one population) is the Lagrangian reading of the
same continuity/Fokker-Planck equation.

What this script computes
-------------------------
For each fitness landscape (additive winner / sign epistasis / reciprocal
sign epistasis a.k.a. fitness valley):

  (1) CONCENTRATION ON NODES.
      Decompose the mass of psi(t) by SUPPORT SIZE:
        class 1 = monomorphic  (exactly one genotype present)  <- the CORNERS
        class 2 = 2 genotypes  (EDGES of the simplex)
        class 3 = 3 genotypes  (FACES)
        class 4 = 4 genotypes  (INTERIOR)
      Verify class-1 mass stays ~1 throughout (start = phi).

  (2) BASIN (corner) mass vs time  -> the flow of probability between nodes,
      both starting from phi and starting from each pure corner (the four
      single-population experiments whose weighted sum reproduces the phi run
      by linearity).

  (3) EFFECTIVE NODE-JUMP MATRIX Q  (4x4) by lumping the full chain into its
      four corner basins (within-basin measure = phi restricted to the basin).
      Q[g,h] = per-generation probability of hopping g -> h.
      Compared with the analytic SSWM substitution rate
        k(g->h) = N * u * pi_fix(s_gh),   s_gh = log w(h) - log w(g),
      for the four single-mutation edges.

  (4) DETAILED BALANCE / reversibility test:  max |pi_i P_ij - pi_j P_ji|
      for the neutral chain (pi=phi) and the selected chain (pi=psi*),
      at r=0 and at r>0, to check the "diffusion with detailed balance /
      continuity equation" reading (and how recombination breaks it).

Run:
  conda activate miso
  cd equilibria_vs_dynamics
  python node_jump_analysis.py                 # all three landscapes
  python node_jump_analysis.py --N 24 --u 2e-3 --tmax 8000
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

# --- make the 2loci codebase importable ------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TWOLOCI_DIR = REPO_ROOT / "2loci"
if str(TWOLOCI_DIR) not in sys.path:
    sys.path.insert(0, str(TWOLOCI_DIR))

from simplex import (  # noqa: E402
    build_sigma,
    build_transition_matrix_row_stochastic,
    column_stochastic,
    fitness_exp,
    make_simplex_counts,
    make_simplex_grid,
    mutation_matrix_asymmetric,
    stationary_fast,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GENO = ("00", "01", "10", "11")
# unit-square coordinates for the node-jump graph (locus1 = y, locus2 = x)
NODE_XY = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.0, 1.0), 3: (1.0, 1.0)}
# single-mutation neighbours (differ at exactly one locus)
EDGES = [(0, 1), (0, 2), (1, 3), (2, 3)]


# =============================================================================
# Landscapes
# =============================================================================
@dataclass
class Landscape:
    name: str
    slug: str
    s1: float
    s2: float
    eps12: float
    blurb: str

    @property
    def logfit(self) -> np.ndarray:
        """log omega(g) = g^T Sigma g for g = 00,01,10,11."""
        return np.array(
            [0.0, self.s2, self.s1, self.s1 + self.s2 + self.eps12], dtype=float
        )


def default_landscapes() -> list[Landscape]:
    return [
        Landscape(
            name="Additive clear winner",
            slug="additive",
            s1=0.10, s2=0.10, eps12=0.0,
            blurb="monotone: both single steps uphill, 11 global optimum",
        ),
        Landscape(
            name="Sign epistasis (routed)",
            slug="sign",
            s1=0.15, s2=-0.05, eps12=0.10,
            blurb="00->01 downhill, 00->10->11 uphill: adaptation channelled via 10",
        ),
        Landscape(
            name="Reciprocal sign epistasis (valley)",
            slug="valley",
            s1=-0.05, s2=-0.05, eps12=0.25,
            blurb="00 & 11 peaks, 01/10 valley: 00->11 needs valley crossing",
        ),
    ]


# =============================================================================
# Simplex geometry helpers
# =============================================================================
def support_class(counts: np.ndarray) -> np.ndarray:
    """Number of present genotypes per state (1=corner .. 4=interior)."""
    return np.count_nonzero(counts, axis=1)


def basin_of(counts: np.ndarray) -> np.ndarray:
    """Plurality genotype per state (argmax count) = which corner it belongs to."""
    return np.argmax(counts, axis=1)


# =============================================================================
# Chain construction
# =============================================================================
@dataclass
class Chain:
    N: int
    u: float
    v: float
    r: float
    counts: np.ndarray
    fracs: np.ndarray
    cls: np.ndarray          # support size per state
    basin: np.ndarray        # corner per state
    phi: np.ndarray          # neutral stationary
    P_neutral: np.ndarray


def build_chain(N: int, u: float, v: float, r: float, n_jobs: int) -> Chain:
    counts = make_simplex_counts(N)
    fracs = make_simplex_grid(N)
    M = mutation_matrix_asymmetric(np.array([u, u]), np.array([v, v]))
    omega_neutral = fitness_exp(np.zeros((2, 2)))
    T_row, _, _ = build_transition_matrix_row_stochastic(
        N, omega_neutral, M, r, n_jobs=n_jobs, label="neutral", verbose=False
    )
    P_neutral = column_stochastic(T_row)
    phi = stationary_fast(P_neutral)
    return Chain(
        N=N, u=u, v=v, r=r,
        counts=counts, fracs=fracs,
        cls=support_class(counts), basin=basin_of(counts),
        phi=phi, P_neutral=P_neutral,
    )


def selection_P(chain: Chain, land: Landscape, n_jobs: int) -> np.ndarray:
    Sigma = build_sigma(land.s1, land.s2, land.eps12)
    omega = fitness_exp(Sigma)
    M = mutation_matrix_asymmetric(np.array([chain.u, chain.u]),
                                   np.array([chain.v, chain.v]))
    T_row, _, _ = build_transition_matrix_row_stochastic(
        chain.N, omega, M, chain.r, n_jobs=n_jobs, label=land.slug, verbose=False
    )
    return column_stochastic(T_row)


# =============================================================================
# Dynamics: evolve several initial conditions at once (columns)
# =============================================================================
def sample_times(tmax: int, n: int = 60) -> np.ndarray:
    ts = np.unique(np.round(np.geomspace(1, tmax, n)).astype(int))
    return np.concatenate([[0], ts])


def evolve_many(P: np.ndarray, Psi0: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return snapshots (len(times), S, ncols): Psi(t) for each sampled t."""
    times = np.asarray(times, dtype=int)
    tmax = int(times.max())
    snaps = np.empty((len(times), Psi0.shape[0], Psi0.shape[1]), dtype=float)
    tset = {int(t): k for k, t in enumerate(times)}
    Psi = Psi0.copy()
    if 0 in tset:
        snaps[tset[0]] = Psi
    for t in range(1, tmax + 1):
        Psi = P @ Psi
        s = Psi.sum(axis=0, keepdims=True)
        s[s == 0] = 1.0
        Psi = Psi / s
        if t in tset:
            snaps[tset[t]] = Psi
    return snaps


def class_mass(snaps: np.ndarray, cls: np.ndarray, col: int) -> dict[int, np.ndarray]:
    """Mass in each support-size class over time for one column."""
    out = {}
    for c in (1, 2, 3, 4):
        mask = cls == c
        out[c] = snaps[:, mask, col].sum(axis=1)
    return out


def basin_mass(snaps: np.ndarray, basin: np.ndarray, col: int) -> np.ndarray:
    """(len(times), 4) mass in each corner basin over time for one column."""
    out = np.empty((snaps.shape[0], 4), dtype=float)
    for g in range(4):
        out[:, g] = snaps[:, basin == g, col].sum(axis=1)
    return out


# =============================================================================
# Effective node-jump matrix (lumping) + SSWM analytic
# =============================================================================
def lumped_Q(P: np.ndarray, chain: Chain) -> np.ndarray:
    """Per-generation 4x4 corner-to-corner transition, within-basin measure = phi."""
    Q = np.zeros((4, 4), dtype=float)
    for g in range(4):
        idx = np.where(chain.basin == g)[0]
        w = chain.phi[idx]
        wsum = w.sum()
        if wsum <= 0:
            continue
        v0 = np.zeros(P.shape[0])
        v0[idx] = w / wsum
        v1 = P @ v0
        for h in range(4):
            Q[g, h] = v1[chain.basin == h].sum()
    return Q


def pi_fix_haploid(s: float, N: int) -> float:
    """Kimura haploid fixation probability of a single copy, sel. coeff s."""
    if abs(s) < 1e-12:
        return 1.0 / N
    num = 1.0 - np.exp(-s)
    den = 1.0 - np.exp(-N * s)
    return float(num / den)


def sswm_rates(land: Landscape, N: int, u: float) -> np.ndarray:
    """Analytic SSWM substitution rate matrix on the four single-mutation edges."""
    lf = land.logfit
    K = np.zeros((4, 4), dtype=float)
    for g, h in EDGES:
        s_gh = lf[h] - lf[g]
        K[g, h] = N * u * pi_fix_haploid(s_gh, N)
        K[h, g] = N * u * pi_fix_haploid(-s_gh, N)
    return K


# =============================================================================
# Detailed balance
# =============================================================================
def detailed_balance_residual(P: np.ndarray, pi: np.ndarray) -> dict[str, float]:
    """Reversibility test for pi.  P column-stochastic (P[j,i] = prob i->j).

    Flux F[j,i] = pi_i P(i->j).  Detailed balance <=> F symmetric.
    Report absolute max asymmetry, and it relative to the largest off-diagonal
    flux (so 'how reversible' is scale-free).
    """
    F = P * pi[np.newaxis, :]
    asym = np.abs(F - F.T)
    Foff = F.copy()
    np.fill_diagonal(Foff, 0.0)
    off_max = float(Foff.max()) if Foff.max() > 0 else 1.0
    return {
        "abs_max": float(asym.max()),
        "rel_max": float(asym.max() / off_max),
        "l1": float(asym.sum()),
    }


# =============================================================================
# Plots
# =============================================================================
def plot_concentration(times, per_land, out: Path) -> None:
    n = len(per_land)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), squeeze=False)
    for ax, (land, cm) in zip(axes[0], per_land):
        for c, lab, color in [
            (1, "monomorphic (corners)", "#1f77b4"),
            (2, "2-genotype (edges)", "#ff7f0e"),
            (3, "3-genotype (faces)", "#2ca02c"),
            (4, "interior", "#d62728"),
        ]:
            ax.plot(times, cm[c], "-", lw=1.8, color=color, label=lab)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("generation $t$")
        ax.set_title(f"{land.name}", fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("mass by simplex support size")
    axes[0, 0].legend(fontsize=8, loc="center left")
    fig.suptitle("Mass stays on the CORNERS: support-size decomposition of "
                 r"$\psi(t)$, start $\psi(0)=\varphi$", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_basin_from_phi(times, per_land, out: Path) -> None:
    n = len(per_land)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), squeeze=False)
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    for ax, (land, bm) in zip(axes[0], per_land):
        for g in range(4):
            ax.plot(times, bm[:, g], "-", lw=1.9, color=colors[g],
                    label=f"{GENO[g]} (logw={land.logfit[g]:+.2f})")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("generation $t$")
        ax.set_title(land.name, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("corner-basin mass")
    fig.suptitle(r"Flow of probability between nodes, start $\psi(0)=\varphi$",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_basin_from_corners(times, land, bm_by_start, out: Path) -> None:
    """bm_by_start: dict start_corner -> (T,4) basin mass."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    for a, ax in zip(range(4), axes.flat):
        bm = bm_by_start[a]
        for g in range(4):
            ax.plot(times, bm[:, g], "-", lw=1.9, color=colors[g], label=GENO[g])
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"start = pure {GENO[a]} (monomorphic population)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, title="basin")
    for ax in axes[-1, :]:
        ax.set_xlabel("generation $t$")
    for ax in axes[:, 0]:
        ax.set_ylabel("corner-basin mass")
    fig.suptitle(f"Single-population adaptive walk: {land.name}\n{land.blurb}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_jump_graph(land, Q, K, N, out: Path) -> None:
    """Node-jump graph: nodes=corners, arrows weighted by empirical Q (per gen)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    lf = land.logfit
    for ax, M, title in [
        (axes[0], Q, "empirical lumped $Q$ (per generation)"),
        (axes[1], K, "analytic SSWM $Nu\\,\\pi_{fix}$ (single-mut edges)"),
    ]:
        # node colouring by log-fitness
        vmax = max(abs(lf).max(), 1e-9)
        for g in range(4):
            x, y = NODE_XY[g]
            col = plt.cm.RdBu((lf[g] / (2 * vmax)) + 0.5)
            ax.scatter([x], [y], s=2600, c=[col], edgecolors="k",
                       zorder=3, linewidths=1.5)
            ax.text(x, y, f"{GENO[g]}\nlogw={lf[g]:+.2f}", ha="center",
                    va="center", fontsize=10, zorder=4)
        # arrows for off-diagonal transitions
        off = M.copy()
        np.fill_diagonal(off, 0.0)
        mx = off.max() if off.max() > 0 else 1.0
        for g in range(4):
            for h in range(4):
                if g == h or M[g, h] <= 0:
                    continue
                x0, y0 = NODE_XY[g]
                x1, y1 = NODE_XY[h]
                dx, dy = x1 - x0, y1 - y0
                # shrink to node edge; offset perpendicular so g->h and h->g separate
                L = np.hypot(dx, dy)
                ux, uy = dx / L, dy / L
                px, py = -uy, ux
                off_amt = 0.045
                sh = 0.16
                xa, ya = x0 + ux * sh + px * off_amt, y0 + uy * sh + py * off_amt
                xb, yb = x1 - ux * sh + px * off_amt, y1 - uy * sh + py * off_amt
                w = M[g, h] / mx
                ax.annotate(
                    "", xy=(xb, yb), xytext=(xa, ya),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6 + 5.5 * w,
                                    color=plt.cm.viridis(0.15 + 0.8 * w),
                                    shrinkA=0, shrinkB=0),
                    zorder=2,
                )
                xm, ym = (xa + xb) / 2 + px * 0.05, (ya + yb) / 2 + py * 0.05
                ax.text(xm, ym, f"{M[g, h]:.1e}", fontsize=7.5,
                        ha="center", va="center", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                  ec="none", alpha=0.75))
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(-0.4, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=10)
    fig.suptitle(f"Node-jump probabilities: {land.name}  (N={N})\n{land.blurb}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=24)
    ap.add_argument("--u", type=float, default=2e-3)
    ap.add_argument("--v", type=float, default=2e-3)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--tmax", type=int, default=8000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", type=Path,
                    default=SCRIPT_DIR / "results" /
                    f"node_jumps_{datetime.now():%Y%m%d}")
    ap.add_argument("--db-r", type=float, default=0.5,
                    help="second r value for detailed-balance comparison")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print(f"Building neutral chain: N={args.N}, u=v={args.u}, r={args.r} ...",
          flush=True)
    chain = build_chain(args.N, args.u, args.v, args.r, args.n_jobs)
    S = len(chain.phi)
    print(f"  S = {S} states.  built in {time.perf_counter()-t0:.1f}s", flush=True)

    # phi concentration report
    corner_mass = chain.phi[chain.cls == 1].sum()
    print(f"\nphi concentration:  monomorphic(corner) mass = {corner_mass:.6f}")
    for c in (1, 2, 3, 4):
        print(f"    support={c}: mass = {chain.phi[chain.cls == c].sum():.6e}")
    # pure-corner masses of phi
    pure_idx = {g: np.where((chain.counts[:, g] == chain.N))[0][0] for g in range(4)}
    print("  phi at the four pure corners:",
          {GENO[g]: round(float(chain.phi[pure_idx[g]]), 5) for g in range(4)})

    times = sample_times(args.tmax)
    landscapes = default_landscapes()

    conc_data, basin_phi_data = [], []
    summary_lines: list[str] = []

    for land in landscapes:
        print(f"\n=== {land.name}  (s1={land.s1}, s2={land.s2}, "
              f"eps12={land.eps12}) ===", flush=True)
        print(f"    log-fitness {dict(zip(GENO, np.round(land.logfit,3)))}")
        P = selection_P(chain, land, args.n_jobs)

        # selection equilibrium
        psi_star = stationary_fast(P)
        star_basin = np.array([psi_star[chain.basin == g].sum() for g in range(4)])
        print(f"    psi* corner-basin mass: "
              f"{dict(zip(GENO, np.round(star_basin,4)))}")

        # initial conditions: phi + 4 pure corners
        Psi0 = np.zeros((S, 5))
        Psi0[:, 0] = chain.phi
        for g in range(4):
            Psi0[pure_idx[g], 1 + g] = 1.0
        snaps = evolve_many(P, Psi0, times)

        conc_data.append((land, class_mass(snaps, chain.cls, 0)))
        basin_phi_data.append((land, basin_mass(snaps, chain.basin, 0)))
        bm_by_start = {g: basin_mass(snaps, chain.basin, 1 + g) for g in range(4)}
        plot_basin_from_corners(times, land, bm_by_start,
                                out / f"walk_{land.slug}.png")

        # node-jump matrices
        Q = lumped_Q(P, chain)
        K = sswm_rates(land, chain.N, chain.u)
        plot_jump_graph(land, Q, K, chain.N, out / f"jumps_{land.slug}.png")

        # detailed balance
        db_neu = detailed_balance_residual(chain.P_neutral, chain.phi)
        db_sel = detailed_balance_residual(P, psi_star)
        db_neu_r = db_neu["rel_max"]
        db_sel_r = db_sel["rel_max"]

        print("    lumped Q (per gen), rows=from, cols=to [00,01,10,11]:")
        for g in range(4):
            print("      " + GENO[g] + " " +
                  "  ".join(f"{Q[g,h]:.3e}" for h in range(4)))
        print("    SSWM analytic Nu*pi_fix (single-mut edges):")
        for g in range(4):
            print("      " + GENO[g] + " " +
                  "  ".join(f"{K[g,h]:.3e}" for h in range(4)))
        print(f"    detailed-balance (relative asym): neutral={db_neu_r:.2e}, "
              f"selected={db_sel_r:.2e}  (r={chain.r})  "
              f"[abs: neu={db_neu['abs_max']:.1e}, sel={db_sel['abs_max']:.1e}]")

        summary_lines.append(
            f"{land.name}: psi* basins {np.round(star_basin,3).tolist()}, "
            f"DB_rel(sel,r={chain.r})={db_sel_r:.1e}")

    plot_concentration(times, conc_data, out / "concentration.png")
    plot_basin_from_phi(times, basin_phi_data, out / "basin_from_phi.png")

    # detailed balance vs recombination, on the additive landscape
    if args.db_r is not None and args.db_r != args.r:
        print(f"\n=== Detailed balance vs recombination (r={args.r} vs "
              f"r={args.db_r}) ===", flush=True)
        chain_r = build_chain(args.N, args.u, args.v, args.db_r, args.n_jobs)
        land0 = landscapes[0]
        db0_neu = detailed_balance_residual(chain.P_neutral, chain.phi)["rel_max"]
        db1_neu = detailed_balance_residual(chain_r.P_neutral, chain_r.phi)["rel_max"]
        P0 = selection_P(chain, land0, args.n_jobs)
        P1 = selection_P(chain_r, land0, args.n_jobs)
        db0_sel = detailed_balance_residual(P0, stationary_fast(P0))["rel_max"]
        db1_sel = detailed_balance_residual(P1, stationary_fast(P1))["rel_max"]
        print(f"  (relative asymmetry)")
        print(f"  neutral : r={args.r} -> {db0_neu:.2e}   "
              f"r={args.db_r} -> {db1_neu:.2e}")
        print(f"  selected: r={args.r} -> {db0_sel:.2e}   "
              f"r={args.db_r} -> {db1_sel:.2e}")
        summary_lines.append(
            f"DB_rel neutral r={args.r}:{db0_neu:.1e} vs r={args.db_r}:{db1_neu:.1e}; "
            f"DB_rel selected r={args.r}:{db0_sel:.1e} vs r={args.db_r}:{db1_sel:.1e}")

    (out / "summary.txt").write_text(
        f"N={args.N} u=v={args.u} r={args.r} tmax={args.tmax}\n"
        f"phi corner mass = {corner_mass:.6f}\n" + "\n".join(summary_lines) + "\n")
    print(f"\nDone in {time.perf_counter()-t0:.1f}s. Outputs in {out}")


if __name__ == "__main__":
    main()
