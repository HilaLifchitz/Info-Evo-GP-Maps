#!/usr/bin/env python3
r"""
Explore the NEUTRAL equilibrium phi of the 2-locus Wright-Fisher chain.

phi is the stationary distribution of the Sigma=0 transition matrix:
  state = integer counts (n00, n01, n10, n11) summing to N,
  one generation = (mutation [+ recombination]) then Multinomial(N, p) drift.
With Sigma=0 there is no selection, so phi is a pure mutation-drift(-recombination)
equilibrium.

This script sweeps a GRID of scenarios (N, mutation vectors u/v, recombination r)
and for each one:
  * solves phi (ARPACK, power-iteration fallback),
  * prints an analysis (corner / edge / interior mass, per-locus marginals),
  * draws a 2D heatmap of phi over the two MARGINAL allele frequencies
        p_A = (n10 + n11)/N   [locus 1 derived allele],
        p_B = (n01 + n11)/N   [locus 2 derived allele].

The marginal heatmap is the natural 2D window on the 3-simplex: the 4 corners
are the monomorphic genotypes; weak mutation piles mass on corners (U-shape),
strong mutation fills the centre, asymmetric u/v shifts it, and r>0 makes the
two loci independent (the joint factorizes into a product of 1D marginals).

Run:
  python3 explore_neutral_phi.py
  python3 explore_neutral_phi.py --out results/phi_explore
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

import simplex as sp

SCRIPT_DIR = Path(__file__).resolve().parent


# =============================================================================
# SCENARIO GRID — edit here. Each entry is one phi to compute + plot.
#   N : gene copies (haploid)        u,v : per-locus mutation (float or [l1,l2])
#   r : recombination rate           label : short title
# =============================================================================
@dataclass
class Scenario:
    label: str
    N: int
    u: object        # float or length-2 list  (0->1 mutation per locus)
    v: object        # float or length-2 list  (1->0 mutation per locus)
    r: float = 0.0


SCENARIOS: list[Scenario] = [
    Scenario("weak symmetric (U-shape)",        N=20, u=1e-4, v=1e-4, r=0.0),
    Scenario("moderate symmetric",              N=30, u=1e-2, v=1e-2, r=0.0),
    Scenario("strong symmetric (central hump)", N=30, u=0.2,  v=0.2,  r=0.0),
    Scenario("asymmetric u != v (biased)",      N=30, u=0.05, v=0.005, r=0.0),
    Scenario("asymmetric across loci",          N=30, u=[0.1, 1e-3], v=[1e-3, 0.1], r=0.0),
    Scenario("moderate, recombination r=0.5",   N=30, u=1e-2, v=1e-2, r=0.5),
]


def neutral_phi(sc: Scenario) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the Sigma=0 chain for one scenario and return (phi, fracs, counts)."""
    u = _vec(sc.u)
    v = _vec(sc.v)
    counts = sp.make_simplex_counts(sc.N)
    fracs = sp.make_simplex_grid(sc.N)
    M_mut = sp.mutation_matrix_asymmetric(u, v)
    omega_neutral = sp.fitness_exp(np.zeros((2, 2)))   # Sigma = 0 -> all ones
    T_row, _, _ = sp.build_transition_matrix_row_stochastic(
        sc.N, omega_neutral, M_mut, sc.r, n_jobs=-1, label=sc.label, verbose=False,
    )
    P_col = sp.column_stochastic(T_row)
    phi = sp.stationary_fast(P_col)
    phi = np.clip(phi, 0.0, None)
    phi /= phi.sum()
    return phi, fracs, counts


def _vec(x) -> np.ndarray:
    a = np.atleast_1d(np.asarray(x, dtype=float))
    if a.size == 1:
        a = np.full(2, float(a[0]))
    if a.size != 2:
        raise ValueError("u/v must be scalar or length-2")
    return a


def diffusion_marginal(N: int, a: float, b: float) -> np.ndarray:
    """
    Discretize the neutral DIFFUSION density rho(x) ∝ x^{2Nu-1}(1-x)^{2Nv-1}
    = Beta(a, b) with a=2Nu, b=2Nv, onto the grid k/N (k=0..N).

    Returns a pmf over k by integrating the Beta density over each cell
    [ (k-0.5)/N , (k+0.5)/N ] (clipped to [0,1]). This is the direct discrete
    image of the formula, comparable to the chain's per-locus marginal.
    """
    from scipy.stats import beta as beta_dist

    edges = (np.arange(N + 2, dtype=float) - 0.5) / N
    edges = np.clip(edges, 0.0, 1.0)
    cdf = beta_dist.cdf(edges, a, b)
    pmf = np.diff(cdf)
    s = pmf.sum()
    if s > 0:
        pmf /= s
    return pmf


def _tv_kl(p: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    """Total-variation distance and KL(p||q) in bits between two pmfs."""
    p = np.asarray(p, float); q = np.asarray(q, float)
    tv = 0.5 * float(np.sum(np.abs(p - q)))
    pe = p + 1e-15; qe = q + 1e-15
    kl = float(np.sum(pe * (np.log(pe) - np.log(qe))) / np.log(2.0))
    return tv, kl


def analyze(phi: np.ndarray, fracs: np.ndarray, counts: np.ndarray, sc: Scenario) -> dict:
    """Corner/edge/interior mass, per-locus marginals, average LD, diffusion fit."""
    N = sc.N
    S = counts.shape[0]
    n_present = (counts > 0).sum(axis=1)
    vertex_mass = float(phi[n_present == 1].sum())
    edge_mass = float(phi[n_present == 2].sum())
    face_mass = float(phi[n_present == 3].sum())
    interior_mass = float(phi[n_present == 4].sum())

    # per-locus derived-allele count distributions (the chain's marginals)
    nA = counts[:, 2] + counts[:, 3]    # locus 1 derived allele
    nB = counts[:, 1] + counts[:, 3]    # locus 2 derived allele
    margA = np.bincount(nA, weights=phi, minlength=N + 1)
    margB = np.bincount(nB, weights=phi, minlength=N + 1)

    # average linkage disequilibrium under phi:  D = x00 x11 - x01 x10
    D = sp.linkage_disequilibrium_per_state(fracs)
    mean_D = float(np.sum(phi * D))
    mean_absD = float(np.sum(phi * np.abs(D)))
    rms_D = float(np.sqrt(np.sum(phi * D * D)))

    # diffusion prediction (independent Beta per locus) on the same grid
    u = _vec(sc.u); v = _vec(sc.v)
    diffA = diffusion_marginal(N, 2.0 * N * u[0], 2.0 * N * v[0])
    diffB = diffusion_marginal(N, 2.0 * N * u[1], 2.0 * N * v[1])
    tvA, klA = _tv_kl(margA, diffA)
    tvB, klB = _tv_kl(margB, diffB)

    return {
        "S": S, "uniform": 1.0 / S,
        "max": float(phi.max()), "max_over_uniform": float(phi.max() * S),
        "vertex_mass": vertex_mass, "edge_mass": edge_mass,
        "face_mass": face_mass, "interior_mass": interior_mass,
        "margA": margA, "margB": margB, "nA": nA, "nB": nB,
        "mean_D": mean_D, "mean_absD": mean_absD, "rms_D": rms_D,
        "diffA": diffA, "diffB": diffB,
        "tvA": tvA, "klA": klA, "tvB": tvB, "klB": klB,
    }


def marginal_heatmap(phi: np.ndarray, info: dict, N: int) -> np.ndarray:
    """(N+1)x(N+1) joint distribution of (n_A, n_B) marginal allele counts."""
    H = np.zeros((N + 1, N + 1), dtype=float)
    np.add.at(H, (info["nB"], info["nA"]), phi)   # rows = locus2 (y), cols = locus1 (x)
    return H


def fmt_uv(sc: Scenario) -> str:
    u = _vec(sc.u)
    v = _vec(sc.v)
    if np.allclose(u, u[0]) and np.allclose(v, v[0]):
        return f"u={u[0]:g}, v={v[0]:g}"
    return f"u={u.tolist()}, v={v.tolist()}"


def run(out_dir: Path, show: bool = False) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    try:
        import seaborn as sns
        sns.set_theme(style="white", context="notebook", font_scale=0.9)
    except ImportError:
        pass
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "figure.facecolor": "white"})

    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    n = len(SCENARIOS)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig_h, axes_h = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.4 * nrow), squeeze=False)
    fig_m, axes_m = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.6 * nrow), squeeze=False)

    for k, sc in enumerate(SCENARIOS):
        print(f"[{k+1}/{n}] {sc.label}:  N={sc.N}, {fmt_uv(sc)}, r={sc.r} ...", flush=True)
        phi, fracs, counts = neutral_phi(sc)
        info = analyze(phi, fracs, counts, sc)
        H = marginal_heatmap(phi, info, sc.N)

        block = [
            f"=== {sc.label} ===",
            f"  N={sc.N}, {fmt_uv(sc)}, r={sc.r}",
            f"  states S={info['S']},  uniform=1/S={info['uniform']:.3e}",
            f"  phi max={info['max']:.4e}  ({info['max_over_uniform']:.1f}x uniform)",
            f"  mass on  4 vertices (monomorphic) : {info['vertex_mass']:.4f}",
            f"  mass on    edges (2 genotypes)    : {info['edge_mass']:.4f}",
            f"  mass on    faces (3 genotypes)    : {info['face_mass']:.4f}",
            f"  mass in interior (4 genotypes)    : {info['interior_mass']:.4f}",
            f"  2N*u (locus1) = {2*sc.N*_vec(sc.u)[0]:.3g}   (>>1 => central, <<1 => U-shape)",
            f"  average LD:  E[D]={info['mean_D']:+.3e}  E|D|={info['mean_absD']:.3e}  rms(D)={info['rms_D']:.3e}",
            f"    (diffusion predicts D=0: loci independent; r={sc.r})",
            f"  diffusion fit (chain marginal vs Beta(2Nu,2Nv)):",
            f"    locus1: TV={info['tvA']:.3e}  KL={info['klA']:.3e} bits",
            f"    locus2: TV={info['tvB']:.3e}  KL={info['klB']:.3e} bits",
            "",
        ]
        log_lines += block
        print("\n".join(block), flush=True)

        # --- heatmap of joint marginal allele frequencies (annotate average LD) ---
        ax = axes_h[k // ncol][k % ncol]
        im = ax.imshow(H, origin="lower", extent=[0, 1, 0, 1], aspect="auto",
                       cmap="magma", norm=mpl.colors.LogNorm(
                           vmin=max(H[H > 0].min(), 1e-12) if np.any(H > 0) else 1e-12,
                           vmax=H.max() if H.max() > 0 else 1.0))
        ax.set_title(f"{sc.label}\nN={sc.N}, {fmt_uv(sc)}, r={sc.r}", fontsize=9)
        ax.set_xlabel(r"$p_A$ (locus 1 derived freq)")
        ax.set_ylabel(r"$p_B$ (locus 2 derived freq)")
        ax.text(0.03, 0.97,
                rf"$\langle D\rangle={info['mean_D']:+.2e}$" + "\n" +
                rf"$\langle|D|\rangle={info['mean_absD']:.2e}$",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                color="white",
                bbox=dict(boxstyle="round", fc="black", alpha=0.45, ec="none"))
        fig_h.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="phi mass (log)")

        # --- per-locus 1D marginals: chain (markers) vs diffusion Beta (lines) ---
        axm = axes_m[k // ncol][k % ncol]
        xs = np.arange(sc.N + 1) / sc.N
        axm.plot(xs, info["margA"], "o", ms=3.5, color="C0", label="locus 1 (chain)")
        axm.plot(xs, info["diffA"], "-", lw=1.6, color="C0", alpha=0.8, label="locus 1 (diffusion)")
        axm.plot(xs, info["margB"], "s", ms=3.0, color="C1", label="locus 2 (chain)")
        axm.plot(xs, info["diffB"], "--", lw=1.4, color="C1", alpha=0.8, label="locus 2 (diffusion)")
        axm.set_title(f"{sc.label}  (TV$_1$={info['tvA']:.2e}, TV$_2$={info['tvB']:.2e})\n"
                      f"N={sc.N}, {fmt_uv(sc)}, r={sc.r}", fontsize=8.5)
        axm.set_xlabel("derived-allele frequency")
        axm.set_ylabel("marginal probability")
        axm.legend(fontsize=7)
        axm.grid(True, alpha=0.3)

    # blank any unused panels
    for k in range(n, nrow * ncol):
        axes_h[k // ncol][k % ncol].axis("off")
        axes_m[k // ncol][k % ncol].axis("off")

    fig_h.suptitle("Neutral equilibrium phi — joint marginal allele frequencies (with average LD)", y=1.0, fontsize=13)
    fig_m.suptitle(r"Per-locus marginal: chain $\phi$ (markers) vs diffusion Beta$(2Nu,2Nv)$ (lines)", y=1.0, fontsize=13)
    fig_h.tight_layout()
    fig_m.tight_layout()

    p_heat = out_dir / "phi_marginal_heatmaps.png"
    p_marg = out_dir / "phi_locus_marginals.png"
    fig_h.savefig(p_heat, dpi=170, bbox_inches="tight")
    fig_m.savefig(p_marg, dpi=170, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig_h)
    plt.close(fig_m)
    print(f"Wrote {p_heat}")
    print(f"Wrote {p_marg}")

    (out_dir / "phi_explore.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'phi_explore.log'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explore neutral equilibrium phi across N, u/v, r.")
    p.add_argument("--out", type=Path, default=None, help="Output dir (default: results/<ts>_phi_explore)")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or (SCRIPT_DIR / "results" /
                       f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_phi_explore")
    run(out, show=args.show)
    print(f"\nAll outputs in: {out}")


if __name__ == "__main__":
    main()
