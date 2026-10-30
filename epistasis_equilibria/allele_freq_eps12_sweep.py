r"""
n=2 haploid: sweep epistasis coefficient ε₁₂ at fixed s₁, s₂.

Writes CSV (ε₁₂, free fitness, KL in bits, mean_W_eq), where **mean_W_eq** is
$\mathbb{E}_{p_{\mathrm{eq}}}[W(x)] = \sum_k p_k W(x_k)$ with the same $p_k$ as
for KL / $F_N$ on the allele-frequency grid.

Output layout
-------------
If --out-prefix is omitted, creates a timestamped subdirectory under --out-dir
whose name includes the date and s₁, s₂, e.g.
  results/eps12_sweep_2026-05-07_143022_s1_0.08_s2_0.10/
    sweep.csv
    run.log
    sweep.png   (unless --no-plot)

If --out-prefix is set, behavior matches the older single-stem convention:
  <prefix>.csv, <prefix>.log, <prefix>.png

--------------------------------------------------------------------
Fitness matrix M
--------------------------------------------------------------------
M₁₁=s₁, M₂₂=s₂, M₁₂=M₂₁=ε₁₂/2 (from CLI). See SWEEP_DEFAULTS for N, u, v, grid.

Run:
  python allele_freq_eps12_sweep.py --s1 0.08 --s2 0.10 --eps-min -1 --eps-max 1 --eps-step 0.1
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from equilibria_allele_freq import _as_float_vec, matrix_M_from_sigma
from haploid_allele_grid_metrics import free_fitness_and_kl_bits_haploid

# =============================================================================
# Sweep defaults — edit here (population, mutation, grid).
# u / v: one float per locus pair, or length-2 list/tuple.
# =============================================================================
SWEEP_DEFAULTS: dict[str, object] = {
    "N_pop": 500.0,
    "u": 1e-5,
    "v": 1e-5,
    "x_eps": 1e-8,
    "resolution": 401,
}


def _epsilon_grid(eps_min: float, eps_max: float, eps_step: float) -> np.ndarray:
    if eps_step <= 0:
        raise ValueError("eps_step must be positive")
    n = int(round((eps_max - eps_min) / eps_step)) + 1
    grid = eps_min + eps_step * np.arange(n, dtype=float)
    if n > 0 and abs(grid[-1] - eps_max) > 1e-9 * max(1.0, abs(eps_max)):
        grid[-1] = eps_max
    return grid


def _ascii_table(
    eps_grid: np.ndarray, free_rows: list[float], kl_rows: list[float], mean_w_rows: list[float]
) -> str:
    rows = [
        "eps12          free_fitness   information_kl_bits  mean_W_eq",
        "-" * 72,
    ]
    for e, ff, kl, mw in zip(eps_grid, free_rows, kl_rows, mean_w_rows, strict=True):
        rows.append(f"{e:14.8g} {ff:14.8g} {kl:18.8g} {mw:14.8g}")
    return "\n".join(rows) + "\n"


def _negative_eps_free_fitness_summary(eps_grid: np.ndarray, free_rows: list[float]) -> str:
    """Stats for ε₁₂ < 0 only (to see if F_N is nearly flat there)."""
    mask = eps_grid < 0.0
    if not np.any(mask):
        return "No grid points with eps12 < 0; skipping negative-epsilon summary.\n"
    ff = np.asarray(free_rows, dtype=float)[mask]
    lines = [
        "Summary for eps12 < 0 (free_fitness):",
        f"  count        : {ff.size}",
        f"  min          : {float(np.min(ff)):.12g}",
        f"  max          : {float(np.max(ff)):.12g}",
        f"  mean         : {float(np.mean(ff)):.12g}",
        f"  std          : {float(np.std(ff)):.12g}",
        f"  range (max-min): {float(np.max(ff) - np.min(ff)):.12g}",
    ]
    zmask = np.isclose(eps_grid, 0.0, rtol=0.0, atol=1e-12 + 1e-9 * max(1.0, float(np.max(np.abs(eps_grid)))))
    if np.any(zmask):
        fz = float(np.asarray(free_rows, dtype=float)[zmask][0])
        lines.append(f"  free_fitness at eps12==0 : {fz:.12g}")
        lines.append(f"  max |F_N(eps<0) - F_N(0)| : {float(np.max(np.abs(ff - fz))):.12g}")
    else:
        lines.append("  (no grid point exactly at eps12==0; add 0 to the sweep to compare)")
    lines.append("")
    return "\n".join(lines)


def _resolve_outputs(
    out_prefix: str,
    out_dir: str,
    *,
    s1: float,
    s2: float,
    eps_min: float,
    eps_max: float,
    eps_step: float,
) -> tuple[Path, Path, Path]:
    """Returns (csv_path, log_path, png_path)."""
    if out_prefix:
        base = Path(out_prefix)
        base.parent.mkdir(parents=True, exist_ok=True)
        stem = base if base.suffix == "" else base.with_suffix("")
        csv_p = stem.with_suffix(".csv") if stem.suffix != ".csv" else stem
        if base.suffix == ".csv":
            csv_p = base
            stem = base.with_suffix("")
        log_p = stem.with_suffix(".log")
        png_p = stem.with_suffix(".png")
        return csv_p, log_p, png_p

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    sub = (
        f"eps12_sweep_{ts}_s1_{s1:g}_s2_{s2:g}_eps_{eps_min:g}_to_{eps_max:g}_step_{eps_step:g}"
    )
    run_dir = Path(out_dir) / sub
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "sweep.csv", run_dir / "run.log", run_dir / "sweep.png"


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep ε₁₂ for n=2 haploid allele-frequency equilibrium.")
    p.add_argument("--s1", type=float, required=True, help="Diagonal M₁₁ (= fitness coefficient s₁).")
    p.add_argument("--s2", type=float, required=True, help="Diagonal M₂₂ (= fitness coefficient s₂).")
    p.add_argument("--eps-min", type=float, default=-1.0, help="Minimum ε₁₂ (interaction parameter; M₁₂=ε₁₂/2).")
    p.add_argument("--eps-max", type=float, default=1.0, help="Maximum ε₁₂.")
    p.add_argument("--eps-step", type=float, default=0.1, help="Step for ε₁₂ grid.")
    p.add_argument(
        "--N",
        type=float,
        default=None,
        help="Override population size (default: SWEEP_DEFAULTS['N_pop'] in this file).",
    )
    p.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Grid points per axis (default: SWEEP_DEFAULTS['resolution']). Lower for speed.",
    )
    p.add_argument(
        "--x-eps",
        type=float,
        default=None,
        help="Clip allele freqs to (x_eps, 1−x_eps); default: SWEEP_DEFAULTS['x_eps'].",
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        default="",
        help="Optional path stem: writes <stem>.csv/.log/.png. If empty, uses timestamped dir under --out-dir.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="When --out-prefix is empty, create results here (default: results/).",
    )
    p.add_argument("--no-plot", action="store_true", help="Only write CSV and log.")
    args = p.parse_args()

    N_pop = float(SWEEP_DEFAULTS["N_pop"])
    u_list: object = SWEEP_DEFAULTS["u"]
    v_list: object = SWEEP_DEFAULTS["v"]
    x_eps = float(SWEEP_DEFAULTS["x_eps"])
    resolution = int(SWEEP_DEFAULTS["resolution"])

    if args.N is not None:
        N_pop = float(args.N)
    if args.resolution is not None:
        resolution = int(args.resolution)
    if args.x_eps is not None:
        x_eps = float(args.x_eps)

    u_vec = _as_float_vec(u_list, 2, "u")
    v_vec = _as_float_vec(v_list, 2, "v")

    eps_grid = _epsilon_grid(args.eps_min, args.eps_max, args.eps_step)
    free_rows: list[float] = []
    kl_rows: list[float] = []
    mean_w_rows: list[float] = []

    for eps in eps_grid:
        m_off = 0.5 * float(eps)
        sigma_raw = np.array([[args.s1, m_off], [m_off, args.s2]], dtype=float)
        M = matrix_M_from_sigma(sigma_raw)
        ff, kl_b, mw = free_fitness_and_kl_bits_haploid(
            M,
            N_pop=N_pop,
            u_vec=u_vec,
            v_vec=v_vec,
            x_eps=x_eps,
            resolution=resolution,
        )
        free_rows.append(ff)
        kl_rows.append(kl_b)
        mean_w_rows.append(mw)

    csv_path, log_path, png_path = _resolve_outputs(
        args.out_prefix,
        args.out_dir,
        s1=args.s1,
        s2=args.s2,
        eps_min=args.eps_min,
        eps_max=args.eps_max,
        eps_step=args.eps_step,
    )

    header = "eps12,free_fitness,information_kl_bits,mean_W_eq\n"
    lines = [header]
    for e, ff, kl_b, mw in zip(eps_grid, free_rows, kl_rows, mean_w_rows, strict=True):
        lines.append(f"{e:.12g},{ff:.12g},{kl_b:.12g},{mw:.12g}\n")
    csv_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path}")

    log_header = [
        f"allele_freq_eps12_sweep ({datetime.now().isoformat(timespec='seconds')})",
        f"cwd: {os.getcwd()}",
        f"M_11=s1={args.s1:g}, M_22=s2={args.s2:g}, M_12=M_21=eps12/2",
        f"eps12 grid: min={args.eps_min:g} max={args.eps_max:g} step={args.eps_step:g} (n={len(eps_grid)})",
        f"N={N_pop:g}, resolution={resolution}, x_eps={x_eps:g}",
        f"u={u_list!r}, v={v_list!r}",
        "mean_W_eq = E_{p_eq}[W(x)] on the discrete equilibrium (same p_k as KL / F_N).",
        "",
        _negative_eps_free_fitness_summary(eps_grid, free_rows),
        "Full table:",
        _ascii_table(eps_grid, free_rows, kl_rows, mean_w_rows),
    ]
    log_path.write_text("\n".join(log_header), encoding="utf-8")
    print(f"Wrote {log_path}")

    if not args.no_plot:
        fig, ax_ff = plt.subplots(figsize=(9, 5))
        ax_kl = ax_ff.twinx()
        (ln0,) = ax_ff.plot(eps_grid, free_rows, "o-", color="C0", lw=1.8, ms=5, label="free fitness")
        (ln2,) = ax_ff.plot(
            eps_grid,
            mean_w_rows,
            "^-",
            color="C2",
            lw=1.6,
            ms=4,
            label=r"$\mathbb{E}_{p_{\mathrm{eq}}}[W]$",
        )
        (ln1,) = ax_kl.plot(eps_grid, kl_rows, "s-", color="C1", lw=1.8, ms=4, alpha=0.9, label="KL (bits)")
        ax_ff.set_xlabel(r"$\varepsilon_{12}$  (matrix: $M_{12}=\varepsilon_{12}/2$)")
        ax_ff.set_ylabel(r"free fitness $F_N$ and $\mathbb{E}_{p_{\mathrm{eq}}}[W]$", color="C0")
        ax_kl.set_ylabel("information: KL$(\\rho_{eq} \\,\\|\\, \\rho_{neutral})$ [bits]", color="C1")
        ax_ff.tick_params(axis="y", labelcolor="C0")
        ax_kl.tick_params(axis="y", labelcolor="C1")
        n_cells = resolution * resolution
        ax_ff.set_title(
            rf"$n=2$ haploid sweep: $M_{{11}}={args.s1:g},\; M_{{22}}={args.s2:g},\; M_{{12}}=\varepsilon_{{12}}/2$, "
            rf"$N={N_pop:g}$, allele grid ${resolution}\times{resolution}$ ({n_cells:g} cells)"
        )
        ax_ff.grid(True, alpha=0.35)
        ax_ff.legend(handles=[ln0, ln2, ln1], loc="best", framealpha=0.95)
        fig.tight_layout()
        fig.savefig(png_path, dpi=180)
        print(f"Wrote {png_path}")
        plt.show()


# Example:
#   python3 allele_freq_eps12_sweep.py --s1 0.08 --s2 0.10 --eps-min -1 --eps-max 1 --eps-step 0.1

if __name__ == "__main__":
    main()
