r"""
n=3 haploid: grid sweep of pairwise epistasis ε₁₂, ε₁₃, ε₂₃ at fixed diagonals s₁,s₂,s₃.

Convention (same as elsewhere): M_ii = s_i, M_ij = M_ji = ε_ij / 2 for i != j.

Uses a uniform tensor grid in allele-frequency space (resolution^n cells). For n=3,
resolution^3 can be large — default resolution in SWEEP_DEFAULTS_N3 is moderate; raise
with care.

Outputs (default: timestamped directory under --out-dir):
  sweep.csv   — eps12,eps13,eps23,free_fitness,information_kl_bits,mean_W_eq
  run.log     — parameters, row count, ASCII sample, summary stats

**mean_W_eq** = $\mathbb{E}_{p_\mathrm{eq}}[W(x)]$ with the same discrete $p_k$ as KL / $F_N$.

This reuses haploid_allele_grid_metrics.free_fitness_and_kl_bits_haploid — the same
ℓ(x) / Z / KL definitions as the n=2 sweep, so results are comparable when n and M match.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

from equilibria_allele_freq import _as_float_vec, matrix_M_from_sigma
from haploid_allele_grid_metrics import free_fitness_and_kl_bits_haploid

# =============================================================================
# Defaults for n=3 (per-locus u,v length 3 via scalar or 3-tuple). Grid: resolution^3.
# =============================================================================
SWEEP_DEFAULTS_N3: dict[str, object] = {
    "N_pop": 50.0,
    "u": 1e-5,
    "v": 1e-5,
    "x_eps": 1e-8,
    "resolution": 71,
}


def _lin_grid(vmin: float, vmax: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("step must be positive")
    n = int(round((vmax - vmin) / step)) + 1
    g = vmin + step * np.arange(n, dtype=float)
    if n > 0 and abs(g[-1] - vmax) > 1e-9 * max(1.0, abs(vmax)):
        g[-1] = vmax
    return g


def _summary_stats(name: str, values: np.ndarray) -> str:
    if values.size == 0:
        return f"{name}: (no rows)\n"
    return (
        f"{name}:\n"
        f"  count: {values.size}, min: {float(np.min(values)):.12g}, max: {float(np.max(values)):.12g}, "
        f"mean: {float(np.mean(values)):.12g}, std: {float(np.std(values)):.12g}, "
        f"range: {float(np.ptp(values)):.12g}\n"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep (ε₁₂,ε₁₃,ε₂₃) for n=3 haploid equilibrium.")
    p.add_argument("--s1", type=float, required=True)
    p.add_argument("--s2", type=float, required=True)
    p.add_argument("--s3", type=float, required=True)
    p.add_argument("--eps12-min", type=float, default=-1.0)
    p.add_argument("--eps12-max", type=float, default=1.0)
    p.add_argument("--eps12-step", type=float, default=0.25)
    p.add_argument("--eps13-min", type=float, default=-1.0)
    p.add_argument("--eps13-max", type=float, default=1.0)
    p.add_argument("--eps13-step", type=float, default=0.25)
    p.add_argument("--eps23-min", type=float, default=-1.0)
    p.add_argument("--eps23-max", type=float, default=1.0)
    p.add_argument("--eps23-step", type=float, default=0.25)
    p.add_argument("--N", type=float, default=None)
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--x-eps", type=float, default=None)
    p.add_argument("--out-prefix", type=str, default="")
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--max-points", type=int, default=200_000, help="Abort if eps-grid product exceeds this.")
    args = p.parse_args()

    N_pop = float(SWEEP_DEFAULTS_N3["N_pop"])
    u_list: object = SWEEP_DEFAULTS_N3["u"]
    v_list: object = SWEEP_DEFAULTS_N3["v"]
    x_eps = float(SWEEP_DEFAULTS_N3["x_eps"])
    resolution = int(SWEEP_DEFAULTS_N3["resolution"])
    if args.N is not None:
        N_pop = float(args.N)
    if args.resolution is not None:
        resolution = int(args.resolution)
    if args.x_eps is not None:
        x_eps = float(args.x_eps)

    g12 = _lin_grid(args.eps12_min, args.eps12_max, args.eps12_step)
    g13 = _lin_grid(args.eps13_min, args.eps13_max, args.eps13_step)
    g23 = _lin_grid(args.eps23_min, args.eps23_max, args.eps23_step)
    ntot = int(g12.size * g13.size * g23.size)
    if ntot > args.max_points:
        raise SystemExit(f"Total sweep points {ntot} > --max-points {args.max_points}; widen steps or raise limit.")

    grid_vol = resolution**3
    if grid_vol > 5_000_000:
        raise SystemExit(f"resolution^3 = {grid_vol} is very large; reduce --resolution.")

    u_vec = _as_float_vec(u_list, 3, "u")
    v_vec = _as_float_vec(v_list, 3, "v")

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if args.out_prefix:
        base = Path(args.out_prefix)
        base.parent.mkdir(parents=True, exist_ok=True)
        stem = base.with_suffix("") if base.suffix == ".csv" else base
        if base.suffix == ".csv":
            stem = base.with_suffix("")
        csv_path = stem.with_suffix(".csv")
        log_path = stem.with_suffix(".log")
    else:
        sub = f"eps3_sweep_{ts}_s1_{args.s1:g}_s2_{args.s2:g}_s3_{args.s3:g}"
        run_dir = Path(args.out_dir) / sub
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_dir / "sweep.csv"
        log_path = run_dir / "run.log"

    rows_out: list[tuple[float, float, float, float, float, float]] = []

    for e12, e13, e23 in product(g12, g13, g23):
        M = np.array(
            [
                [args.s1, 0.5 * e12, 0.5 * e13],
                [0.5 * e12, args.s2, 0.5 * e23],
                [0.5 * e13, 0.5 * e23, args.s3],
            ],
            dtype=float,
        )
        M = matrix_M_from_sigma(M)
        ff, kl, mw = free_fitness_and_kl_bits_haploid(
            M,
            N_pop=N_pop,
            u_vec=u_vec,
            v_vec=v_vec,
            x_eps=x_eps,
            resolution=resolution,
        )
        rows_out.append((float(e12), float(e13), float(e23), ff, kl, mw))

    header = "eps12,eps13,eps23,free_fitness,information_kl_bits,mean_W_eq\n"
    lines = [header]
    for e12, e13, e23, ff, kl, mw in rows_out:
        lines.append(f"{e12:.12g},{e13:.12g},{e23:.12g},{ff:.12g},{kl:.12g},{mw:.12g}\n")
    csv_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {csv_path} ({len(rows_out)} rows)")

    ff_all = np.array([r[3] for r in rows_out], dtype=float)
    kl_all = np.array([r[4] for r in rows_out], dtype=float)
    mw_all = np.array([r[5] for r in rows_out], dtype=float)
    neg_mask = np.array([(r[0] < 0 and r[1] < 0 and r[2] < 0) for r in rows_out], dtype=bool)
    ff_neg = ff_all[neg_mask]

    # small ASCII sample (head)
    sample_lines = ["eps12        eps13        eps23        free_fitness   KL_bits      mean_W_eq", "-" * 72]
    for row in rows_out[: min(15, len(rows_out))]:
        sample_lines.append(
            f"{row[0]:12.6g} {row[1]:12.6g} {row[2]:12.6g} {row[3]:14.8g} {row[4]:12.8g} {row[5]:12.8g}"
        )
    if len(rows_out) > 15:
        sample_lines.append(f"... ({len(rows_out) - 15} more rows in CSV)")

    log_body = [
        f"allele_freq_eps3_sweep ({datetime.now().isoformat(timespec='seconds')})",
        f"cwd: {os.getcwd()}",
        f"diagonal M: s1={args.s1:g}, s2={args.s2:g}, s3={args.s3:g}; off-diag M_ij=eps_ij/2",
        f"eps12: [{args.eps12_min:g}, {args.eps12_max:g}] step {args.eps12_step:g} ({g12.size} pts)",
        f"eps13: [{args.eps13_min:g}, {args.eps13_max:g}] step {args.eps13_step:g} ({g13.size} pts)",
        f"eps23: [{args.eps23_min:g}, {args.eps23_max:g}] step {args.eps23_step:g} ({g23.size} pts)",
        f"total epsilon tuples: {len(rows_out)}; allele grid resolution^n = {resolution}^3 = {grid_vol}",
        f"N={N_pop:g}, x_eps={x_eps:g}",
        f"u={u_list!r}, v={v_list!r}",
        "mean_W_eq = E_{p_eq}[W(x)] on the discrete equilibrium (same p_k as KL / F_N).",
        "",
        "Free fitness, mean_W_eq & KL over all sweep rows:",
        _summary_stats("  free_fitness (all)", ff_all),
        _summary_stats("  mean_W_eq (all)", mw_all),
        _summary_stats("  information_kl_bits (all)", kl_all),
        "Subset with eps12<0 AND eps13<0 AND eps23<0:",
        _summary_stats("  free_fitness (all neg eps)", ff_neg),
        _summary_stats("  mean_W_eq (all neg eps)", mw_all[neg_mask]),
        _summary_stats("  information_kl_bits (all neg eps)", kl_all[neg_mask]),
        "",
        "Sample (first rows):",
        "\n".join(sample_lines),
        "",
    ]
    log_path.write_text("\n".join(log_body), encoding="utf-8")
    print(f"Wrote {log_path}")


if __name__ == "__main__":
    main()

Read