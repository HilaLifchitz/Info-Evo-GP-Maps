#!/usr/bin/env python3
"""
Stationary epistasis sweep using the 2-locus Wright-Fisher Markov chain.

For fixed s1 and s2, sweep eps12 in Sigma, compute

    D_KL(psi* || phi*)

where phi* is the neutral stationary distribution and psi* is the selected
stationary distribution for the same N, mutation, and recombination settings.

This script intentionally does not compute free fitness or trajectories.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib

    # Force a non-interactive backend so PNGs always write in headless runs.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    plt = None
    HAS_MPL = False

try:
    import seaborn as sns

    HAS_SNS = True
except ImportError:
    sns = None
    HAS_SNS = False

from simplex import (
    SimplexParams,
    build_sigma,
    build_transition_matrix_row_stochastic,
    column_stochastic,
    compute_phi,
    fitness_exp,
    kl_divergence_bits,
    mutation_matrix_asymmetric,
    params_from_config,
    try_selection_equilibrium,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# =============================================================================
# DEFAULT SWEEP  SETTINGS
# =============================================================================
# These defaults keep max(N) * max(|s1|, |s2|, |eps12|) = 0.08 for N=(30, 40).
N_VALUES: tuple[int, ...] = (30, 40)
R: float =  0.001
U: np.ndarray = np.array([1.0e-3, 1.0e-3], dtype=float)
V: np.ndarray = np.array([1.0e-3, 1.0e-3], dtype=float)
S1: float = -1.0e-3
S2: float = 5.0e-4

EPS12_MIN: float = -2.0e-3
EPS12_MAX: float = 2.0e-3
EPS12_STEP: float = 2.5e-4

NEUTRAL_CACHE_DIR: str = "cache"
N_JOBS: int = -1
OUTPUT_DIR: Path = SCRIPT_DIR / "plots" / "eps12_equilibrium_kl"
SHOW_PLOTS: bool = False


@dataclass
class EquilibriumSweepResult:
    N: int
    eps12: float
    kl_bits: float
    psi_residual_l1: float
    converged: bool
    phi_source: str
    diffusion_scale: float


def _float_pair(value: str, name: str) -> np.ndarray:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) == 1:
        x = float(parts[0])
        return np.array([x, x], dtype=float)
    if len(parts) == 2:
        return np.array([float(parts[0]), float(parts[1])], dtype=float)
    raise argparse.ArgumentTypeError(f"{name} must be a scalar or comma-separated pair")


def _int_list(value: str) -> tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("N list cannot be empty")
    return tuple(int(p) for p in parts)


def _eps12_grid(eps_min: float, eps_max: float, eps_step: float) -> np.ndarray:
    if eps_step <= 0.0:
        raise ValueError("eps12 step must be positive")
    if eps_max < eps_min:
        raise ValueError("eps12 max must be >= eps12 min")
    n = int(round((eps_max - eps_min) / eps_step)) + 1
    grid = eps_min + eps_step * np.arange(n)
    grid = np.clip(grid, eps_min, eps_max)
    return np.unique(np.round(grid, 12))


def _base_cfg(
    *,
    N: int,
    r: float,
    u: np.ndarray,
    v: np.ndarray,
    s1: float,
    s2: float,
    eps12: float,
    n_jobs: int,
    neutral_cache_dir: str,
) -> dict:
    return {
        "_config_dir": str(SCRIPT_DIR),
        "N": int(N),
        "r": float(r),
        "u": u.tolist(),
        "v": v.tolist(),
        "sigma": build_sigma(s1, s2, eps12).tolist(),
        "times": [1],
        "n_jobs": int(n_jobs),
        "neutral_cache_dir": neutral_cache_dir,
        "neutral": False,
        "top_k": 5,
        "plot_dynamics": False,
        "compute_f2": False,
        "ld_top_k_times": "first_last",
        "debug_kl": False,
    }


def _params_for_eps12(
    *,
    N: int,
    r: float,
    u: np.ndarray,
    v: np.ndarray,
    s1: float,
    s2: float,
    eps12: float,
    n_jobs: int,
    neutral_cache_dir: str,
) -> SimplexParams:
    cfg = _base_cfg(
        N=N,
        r=r,
        u=u,
        v=v,
        s1=s1,
        s2=s2,
        eps12=eps12,
        n_jobs=n_jobs,
        neutral_cache_dir=neutral_cache_dir,
    )
    return params_from_config(cfg, config_dir=SCRIPT_DIR)


def _diffusion_scale(N: int, s1: float, s2: float, eps12: float) -> float:
    return float(N * max(abs(s1), abs(s2), abs(eps12)))


def run_one_N(
    *,
    N: int,
    eps_grid: np.ndarray,
    r: float,
    u: np.ndarray,
    v: np.ndarray,
    s1: float,
    s2: float,
    n_jobs: int,
    neutral_cache_dir: str,
    verbose: bool,
) -> list[EquilibriumSweepResult]:
    base_params = _params_for_eps12(
        N=N,
        r=r,
        u=u,
        v=v,
        s1=s1,
        s2=s2,
        eps12=0.0,
        n_jobs=n_jobs,
        neutral_cache_dir=neutral_cache_dir,
    )
    M_mut = mutation_matrix_asymmetric(base_params.u, base_params.v)
    phi, _, _, _, phi_source = compute_phi(base_params, M_mut)

    results: list[EquilibriumSweepResult] = []
    for i, eps12 in enumerate(eps_grid, start=1):
        scale = _diffusion_scale(N, s1, s2, float(eps12))
        if verbose:
            print(
                f"[N={N}] [{i}/{len(eps_grid)}] eps12={eps12:+.6g} "
                f"(N*max effect={scale:.3g})",
                flush=True,
            )

        params = _params_for_eps12(
            N=N,
            r=r,
            u=u,
            v=v,
            s1=s1,
            s2=s2,
            eps12=float(eps12),
            n_jobs=n_jobs,
            neutral_cache_dir=neutral_cache_dir,
        )
        omega = fitness_exp(params.Sigma)
        T_row, _, _ = build_transition_matrix_row_stochastic(
            params.N,
            omega,
            M_mut,
            params.r,
            n_jobs=params.n_jobs,
            label=f"N={N}, eps12={eps12:+.6g}",
            verbose=verbose,
        )
        psi_star, residual, converged = try_selection_equilibrium(column_stochastic(T_row))
        kl_bits = float("nan")
        if psi_star is not None:
            kl_bits = kl_divergence_bits(psi_star, phi)

        results.append(
            EquilibriumSweepResult(
                N=int(N),
                eps12=float(eps12),
                kl_bits=kl_bits,
                psi_residual_l1=float(residual),
                converged=bool(converged),
                phi_source=phi_source,
                diffusion_scale=scale,
            )
        )

    return results


def run_sweep(args: argparse.Namespace) -> list[EquilibriumSweepResult]:
    eps_grid = _eps12_grid(args.eps_min, args.eps_max, args.eps_step)
    max_scale = max(
        _diffusion_scale(N, args.s1, args.s2, float(eps12))
        for N in args.N
        for eps12 in eps_grid
    )

    if not args.quiet:
        print("=" * 72)
        print("eps12_equilibrium_kl_sweep.py")
        print(f"  N values: {args.N}")
        print(f"  r={args.r}, u={args.u}, v={args.v}")
        print(f"  s1={args.s1}, s2={args.s2}")
        print(f"  eps12 grid: {len(eps_grid)} values from {eps_grid[0]:g} to {eps_grid[-1]:g}")
        print(f"  max N*max(|s1|, |s2|, |eps12|) = {max_scale:.3g}")
        if max_scale >= 0.1:
            print("  WARNING: diffusion scale is not very small; consider smaller effects.")
        print("=" * 72)

    t0 = time.perf_counter()
    results: list[EquilibriumSweepResult] = []
    for N in args.N:
        results.extend(
            run_one_N(
                N=N,
                eps_grid=eps_grid,
                r=args.r,
                u=args.u,
                v=args.v,
                s1=args.s1,
                s2=args.s2,
                n_jobs=args.n_jobs,
                neutral_cache_dir=args.neutral_cache_dir,
                verbose=not args.quiet,
            )
        )

    if not args.quiet:
        elapsed = time.perf_counter() - t0
        print(f"Done in {elapsed:.1f}s", flush=True)

    return results


def _setup_plot_style() -> None:
    if not HAS_MPL or plt is None:
        raise RuntimeError("matplotlib is required for plotting; rerun with --no-plot to save CSV only")
    if HAS_SNS and sns is not None:
        sns.set_theme(
            style="whitegrid",
            context="notebook",
            font_scale=1.05,
            rc={
                "axes.spines.top": False,
                "axes.spines.right": False,
                "grid.alpha": 0.35,
            },
        )
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
    })


def _slug(args: argparse.Namespace) -> str:
    def fmt(x: float) -> str:
        return f"{x:.4g}".replace("+", "")

    n_slug = "-".join(str(N) for N in args.N)
    return (
        f"N_{n_slug}_r_{fmt(args.r)}_s1_{fmt(args.s1)}_s2_{fmt(args.s2)}_"
        f"eps_{fmt(args.eps_min)}_{fmt(args.eps_max)}"
    )


def save_csv(results: list[EquilibriumSweepResult], output_dir: Path, args: argparse.Namespace) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"eps12_equilibrium_kl_{_slug(args)}.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "N",
            "eps12",
            "KL_psi_star_vs_phi_star_bits",
            "psi_residual_l1",
            "converged",
            "phi_source",
            "N_max_effect",
        ])
        for res in results:
            writer.writerow([
                res.N,
                f"{res.eps12:.12g}",
                f"{res.kl_bits:.12g}",
                f"{res.psi_residual_l1:.12g}",
                int(res.converged),
                res.phi_source,
                f"{res.diffusion_scale:.12g}",
            ])
    return path


def plot_results(results: list[EquilibriumSweepResult], output_dir: Path, args: argparse.Namespace) -> Path:
    _setup_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for N in args.N:
        subset = [res for res in results if res.N == N]
        subset.sort(key=lambda res: res.eps12)
        eps = np.array([res.eps12 for res in subset], dtype=float)
        kl = np.array([res.kl_bits for res in subset], dtype=float)
        ax.plot(eps, kl, "o-", lw=2.0, ms=5, label=rf"$N={N}$")

        failed = [res for res in subset if not res.converged]
        if failed:
            ax.scatter(
                [res.eps12 for res in failed],
                [res.kl_bits for res in failed],
                marker="x",
                s=55,
                color="black",
                zorder=4,
                label=rf"$N={N}$ not converged",
            )

    ax.axhline(0.0, color="0.35", lw=0.9, zorder=1)

    # Weak-selection / symmetric-mutation prediction for the dip location:
    #   eps12* = -Cov(B, x11)/Var(x11)  ->  -2/3 (s1 + s2)  for symmetric u = v.
    eps12_star = -2.0 / 3.0 * (args.s1 + args.s2)
    eps_lo = min(res.eps12 for res in results)
    eps_hi = max(res.eps12 for res in results)
    in_window = eps_lo <= eps12_star <= eps_hi
    label = rf"predicted dip $\varepsilon_{{12}}^*=-\frac{{2}}{{3}}(s_1+s_2)={eps12_star:.2g}$"
    if not in_window:
        label += " (outside window)"
    ax.axvline(
        eps12_star,
        color="0.45",
        lw=1.3,
        ls="--",
        zorder=2,
        label=label,
    )

    ax.set_xlabel(r"epistasis $\epsilon_{12}$")
    ax.set_ylabel(r"$D_{\mathrm{KL}}(\psi^*\,\|\,\varphi^*)$ [bits]")
    ax.set_title(
        "\n".join([
            r"Stationary selected-vs-neutral KL under epistasis sweep",
            rf"$s_1={args.s1:.3g},\; s_2={args.s2:.3g},\; r={args.r:.3g}$",
            rf"$u=({args.u[0]:.2g},{args.u[1]:.2g}),\; v=({args.v[0]:.2g},{args.v[1]:.2g})$",
        ]),
        pad=14,
    )
    ax.grid(True, alpha=0.35, linestyle="-", linewidth=0.6)
    ax.legend(framealpha=0.92)

    fig.tight_layout()
    path = output_dir / f"eps12_equilibrium_kl_{_slug(args)}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep eps12 and plot stationary KL(psi* || phi*) for N=30,40 by default."
    )
    p.add_argument("--N", type=_int_list, default=N_VALUES, help="Comma-separated N values")
    p.add_argument("--r", type=float, default=R, help="Recombination rate")
    p.add_argument("--u", type=lambda x: _float_pair(x, "u"), default=U, help="Scalar or pair, e.g. 1e-3 or 1e-3,1e-3")
    p.add_argument("--v", type=lambda x: _float_pair(x, "v"), default=V, help="Scalar or pair, e.g. 1e-3 or 1e-3,1e-3")
    p.add_argument("--s1", type=float, default=S1, help="Selection coefficient on locus 1")
    p.add_argument("--s2", type=float, default=S2, help="Selection coefficient on locus 2")
    p.add_argument("--eps-min", type=float, default=EPS12_MIN, help="Minimum eps12")
    p.add_argument("--eps-max", type=float, default=EPS12_MAX, help="Maximum eps12")
    p.add_argument("--eps-step", type=float, default=EPS12_STEP, help="eps12 step")
    p.add_argument("--n-jobs", type=int, default=N_JOBS, help="-1 means all available cores")
    p.add_argument("--neutral-cache-dir", default=NEUTRAL_CACHE_DIR, help="Neutral cache directory")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for CSV and PNG output")
    p.add_argument("--no-plot", action="store_true", help="Skip PNG plotting and save CSV only")
    p.add_argument("--show", action="store_true", default=SHOW_PLOTS, help="Display figure interactively")
    p.add_argument("-q", "--quiet", action="store_true", help="Less console output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results = run_sweep(args)
    output_dir = args.output_dir.resolve()
    csv_path = save_csv(results, output_dir, args)
    plot_path = None
    if not args.no_plot:
        if HAS_MPL:
            plot_path = plot_results(results, output_dir, args)
        elif not args.quiet:
            print("matplotlib is not installed; saved CSV only. Install matplotlib or rerun in the project env for PNG output.")
    if not args.quiet:
        print("\nSaved outputs:")
        print(f"  CSV:  {csv_path}")
        if plot_path is not None:
            print(f"  Plot: {plot_path}")


if __name__ == "__main__":
    main()
