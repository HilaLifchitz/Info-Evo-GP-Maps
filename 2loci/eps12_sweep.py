#!/usr/bin/env python3
"""
Epistatic sweep: overlay KL and free-fitness trajectories for many ε₁₂ values.

Fixes N, r, u, v, s₁, s₂ and sweeps ε₁₂ (= 2 Σ₁₂). Saves four figures with one
colored curve per ε₁₂ (colorbar legend).

  python eps12_sweep.py
  python eps12_sweep.py --output-dir plots/my_sweep

Edit the block marked SWEEP SETTINGS below to generate other sweep families.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

try:
    import seaborn as sns

    HAS_SNS = True
except ImportError:
    HAS_SNS = False

from simplex import (
    SimplexParams,
    build_sigma,
    build_transition_matrix_row_stochastic,
    column_stochastic,
    compute_phi,
    dynamics_series,
    fitness_exp,
    linkage_disequilibrium_per_state,
    mutation_matrix_asymmetric,
    params_from_config,
    w_potential_per_state,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# =============================================================================
# SWEEP SETTINGS — edit these to define a new sweep family
# =============================================================================
N: int = 30
R: float = 0.0
U: np.ndarray = np.array([1.0e-3, 1.0e-3], dtype=float)
V: np.ndarray = np.array([1.0e-3, 1.0e-3], dtype=float)
S1: float = -0.1
S2: float = -0.005

EPS12_MIN: float = -1.0
EPS12_MAX: float = 1.0
EPS12_STEP: float = 0.2

# Sample times (generations) — edit here for each sweep run.
TIMES: list[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    20, 30, 40, 50, 60, 70, 80, 90, 100,
    150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950,
    1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700,
    1750, 1800, 1850, 1900, 1950, 2000,
    # 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000,
    # 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000,
    # 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000,
    # 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000,
    # 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900, 7000,
    # 7100, 7200, 7300, 7400, 7500, 7600, 7700, 7800, 7900, 8000,
    # 8100, 8200, 8300, 8400, 8500, 8600, 8700, 8800, 8900, 9000,
    # 9100, 9200, 9300, 9400, 9500, 9600, 9700, 9800, 9900, 10000,
]

NEUTRAL_CACHE_DIR: str = "cache"
N_JOBS: int = -1
OUTPUT_DIR: Path = SCRIPT_DIR / "plots"
SHOW_PLOTS: bool = False
# =============================================================================


@dataclass
class SweepResult:
    eps12: float
    times: np.ndarray
    kl_bits: np.ndarray
    f1: np.ndarray
    rate_times: np.ndarray
    rate_kl_bits: np.ndarray
    rate_f1: np.ndarray


def _eps12_grid() -> np.ndarray:
    n = int(round((EPS12_MAX - EPS12_MIN) / EPS12_STEP)) + 1
    grid = EPS12_MIN + EPS12_STEP * np.arange(n)
    grid = np.clip(grid, EPS12_MIN, EPS12_MAX)
    return np.unique(np.round(grid, 10))


def _base_cfg(times: list[int]) -> dict:
    return {
        "_config_dir": str(SCRIPT_DIR),
        "N": N,
        "r": R,
        "u": U.tolist(),
        "v": V.tolist(),
        "sigma": [[S1, 0.0], [0.0, S2]],
        "times": times,
        "n_jobs": N_JOBS,
        "neutral_cache_dir": NEUTRAL_CACHE_DIR,
        "neutral": False,
        "top_k": 5,
        "plot_dynamics": False,
        "compute_f2": False,
        "ld_top_k_times": "first_last",
        "debug_kl": False,
    }


def _params_for_eps12(times: list[int], eps12: float) -> SimplexParams:
    cfg = _base_cfg(times)
    cfg["sigma"] = build_sigma(S1, S2, float(eps12)).tolist()
    return params_from_config(cfg, config_dir=SCRIPT_DIR)


def _fixed_param_title_lines() -> list[str]:
    u_str = ",\\,".join(f"{x:.2g}" for x in U)
    v_str = ",\\,".join(f"{x:.2g}" for x in V)
    return [
        r"Epistatic sweep  ($\psi(0)=\varphi$)",
        rf"$N={N},\quad r={R}$",
        rf"$u=({u_str}),\quad v=({v_str})$",
        rf"$s_1={S1:.3g},\quad s_2={S2:.3g}$",
        rf"$\epsilon_{{12}} \in [{EPS12_MIN:g},\, {EPS12_MAX:g}]$ step {EPS12_STEP:g}",
    ]


def _output_slug() -> str:
    def _fmt(x: float) -> str:
        return f"{x:.4g}".replace("+", "")

    return f"r_{_fmt(R)}_s1_{_fmt(S1)}_s2_{_fmt(S2)}"


def _setup_plot_style() -> None:
    if HAS_SNS:
        sns.set_theme(
            style="whitegrid",
            context="notebook",
            font_scale=1.05,
            rc={
                "axes.spines.top": False,
                "axes.spines.right": False,
                "grid.alpha": 0.35,
                "grid.linestyle": "-",
            },
        )
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
    })


def run_one_eps12(
    eps12: float,
    times: list[int],
    phi: np.ndarray,
    fracs: np.ndarray,
    counts: np.ndarray,
    M_mut: np.ndarray,
    d_per_state: np.ndarray,
    n_jobs: int,
) -> SweepResult:
    params = _params_for_eps12(times, eps12)
    omega = fitness_exp(params.Sigma)
    T_row, _, _ = build_transition_matrix_row_stochastic(
        params.N,
        omega,
        M_mut,
        params.r,
        n_jobs=n_jobs,
        label=f"eps12={eps12:.2g}",
        verbose=False,
    )
    P_col = column_stochastic(T_row)
    w_per_state = w_potential_per_state(fracs, omega, M_mut, params.N)
    result = dynamics_series(
        P_col,
        phi,
        times,
        w_per_state,
        d_per_state,
        counts,
        fracs,
        params.N,
        params.top_k,
        params.ld_top_k_times,
        psi_star=None,
    )
    return SweepResult(
        eps12=float(eps12),
        times=np.asarray(result.times, dtype=float),
        kl_bits=np.asarray(result.kl_phi_bits, dtype=float),
        f1=np.asarray(result.f1, dtype=float),
        rate_times=np.asarray(result.rate_times, dtype=float),
        rate_kl_bits=np.asarray(result.rate_kl_bits, dtype=float),
        rate_f1=np.asarray(result.rate_f1, dtype=float),
    )


def run_sweep(verbose: bool = True) -> list[SweepResult]:
    times = [int(t) for t in TIMES]
    eps_grid = _eps12_grid()
    base_params = _params_for_eps12(times, 0.0)
    M_mut = mutation_matrix_asymmetric(base_params.u, base_params.v)

    if verbose:
        print("=" * 72)
        print("eps12_sweep.py")
        print(f"  N={N}, r={R}, s1={S1}, s2={S2}")
        print(f"  u={U}, v={V}")
        print(f"  eps12 grid: {len(eps_grid)} values from {eps_grid[0]:g} to {eps_grid[-1]:g}")
        print(f"  times: {len(times)} points, t in [{min(times)}, {max(times)}]")
        print("=" * 72)

    phi, _, fracs, counts, source = compute_phi(base_params, M_mut)
    if verbose:
        print(f"  phi source: {source}", flush=True)

    d_per_state = linkage_disequilibrium_per_state(fracs)
    results: list[SweepResult] = []
    t0 = time.perf_counter()

    for i, eps12 in enumerate(eps_grid, start=1):
        if verbose:
            print(f"[{i}/{len(eps_grid)}] eps12 = {eps12:+.2f} ...", flush=True)
        results.append(
            run_one_eps12(
                eps12,
                times,
                phi,
                fracs,
                counts,
                M_mut,
                d_per_state,
                base_params.n_jobs,
            )
        )

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"Done in {elapsed:.1f}s ({elapsed / len(eps_grid):.1f}s per eps12)", flush=True)

    return results


def _plot_family(
    results: list[SweepResult],
    *,
    x_key: str,
    y_key: str,
    ylabel: str,
    filename: str,
    output_dir: Path,
    show: bool,
) -> Path:
    eps_values = np.array([r.eps12 for r in results])
    cmap = plt.get_cmap("coolwarm")
    norm = mpl.colors.Normalize(vmin=EPS12_MIN, vmax=EPS12_MAX)

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for res in results:
        x = getattr(res, x_key)
        y = getattr(res, y_key)
        ax.plot(
            x,
            y,
            color=cmap(norm(res.eps12)),
            lw=1.6,
            alpha=0.92,
            solid_capstyle="round",
        )

    ax.set_xlabel(r"generation $t$")
    ax.set_ylabel(ylabel)
    ax.set_title("\n".join(_fixed_param_title_lines()), pad=14)
    ax.grid(True, alpha=0.35, linestyle="-", linewidth=0.6)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(r"epistasis $\epsilon_{12}$")

    fig.tight_layout()
    out = output_dir / filename
    fig.savefig(out, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out


def plot_sweep(results: list[SweepResult], output_dir: Path, show: bool = False) -> dict[str, Path]:
    _setup_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = _output_slug()

    paths = {
        "kl_values": _plot_family(
            results,
            x_key="times",
            y_key="kl_bits",
            ylabel=r"$D_{\mathrm{KL}}(\psi\,\|\,\varphi)$ [bits]",
            filename=f"eps12_sweep_kl_vs_t_{slug}.png",
            output_dir=output_dir,
            show=show,
        ),
        "kl_rates": _plot_family(
            results,
            x_key="rate_times",
            y_key="rate_kl_bits",
            ylabel=r"$\Delta D_{\mathrm{KL}} / \Delta t$ [bits / gen]",
            filename=f"eps12_sweep_kl_rates_vs_t_{slug}.png",
            output_dir=output_dir,
            show=show,
        ),
        "f1_values": _plot_family(
            results,
            x_key="times",
            y_key="f1",
            ylabel=r"FREE FITNESS $\mathcal{F}_N$",
            filename=f"eps12_sweep_f1_vs_t_{slug}.png",
            output_dir=output_dir,
            show=show,
        ),
        "f1_rates": _plot_family(
            results,
            x_key="rate_times",
            y_key="rate_f1",
            ylabel=r"$\Delta \mathcal{F}_N / \Delta t$",
            filename=f"eps12_sweep_f1_rates_vs_t_{slug}.png",
            output_dir=output_dir,
            show=show,
        ),
    }
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep eps12 and plot KL / F_N families")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for PNG output (default: plots/ next to this script)",
    )
    p.add_argument("--show", action="store_true", help="Display figures interactively")
    p.add_argument("-q", "--quiet", action="store_true", help="Less console output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results = run_sweep(verbose=not args.quiet)
    paths = plot_sweep(results, args.output_dir.resolve(), show=args.show or SHOW_PLOTS)
    if not args.quiet:
        print("\nSaved plots:")
        for name, path in paths.items():
            print(f"  [{name}] {path}")


if __name__ == "__main__":
    main()
