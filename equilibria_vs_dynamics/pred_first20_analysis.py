#!/usr/bin/env python3
"""Analyze KL_eq vs first-20-gen adaptation metrics from pred_grid CSV.

Reads stage_a.csv (optionally extends with kl_at_20 if missing), computes
Spearman correlations, and writes comparison plots.

Run:
  python pred_first20_analysis.py
  python pred_first20_analysis.py --csv results/pred_grid/stage_a.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from eps12_predictiveness_grid import (
    BACKGROUNDS,
    EARLY_WINDOW,
    N_MARKOV,
    R_BASE,
    U_BASE,
    V_BASE,
    _eps12_grid,
    _make_params,
    _spearman,
    _summary_table,
    evaluate_background,
    plot_spearman_bars,
)
from simplex import (
    compute_phi,
    linkage_disequilibrium_per_state,
    mutation_matrix_asymmetric,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / "results" / "pred_grid" / "stage_a.csv"
OUT_DIR = SCRIPT_DIR / "results" / "pred_first20"


def _bg_map() -> dict[str, tuple[float, float]]:
    return {b.label: (b.s1, b.s2) for b in BACKGROUNDS}


def _load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _needs_kl20(rows: list[dict]) -> bool:
    return "kl_at_20" not in rows[0] or any(not r.get("kl_at_20") for r in rows)


def _extend_csv(rows: list[dict], out_path: Path) -> list[dict]:
    """Re-run dynamics to fill kl_at_20 for each background (from cached settings)."""
    eps_grid = _eps12_grid()
    times = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 170, 300, 500, 800]
    M_mut = mutation_matrix_asymmetric(
        np.array([U_BASE, U_BASE]), np.array([V_BASE, V_BASE])
    )
    params0 = _make_params(N_MARKOV, U_BASE, V_BASE, R_BASE, 0.0, 0.0, 0.0)
    phi, _, fracs, counts, _ = compute_phi(params0, M_mut)
    d_per_state = linkage_disequilibrium_per_state(fracs)

    by_bg: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bg[r["background"]].append(r)

    extended: list[dict] = []
    for bg in BACKGROUNDS:
        print(f"  extending {bg.label} ...", flush=True)
        pts = evaluate_background(
            bg, N_MARKOV, U_BASE, V_BASE, R_BASE,
            eps_grid, phi, fracs, counts, d_per_state, M_mut, times,
        )
        lookup = {float(p.eps12): p for p in pts}
        for r in by_bg[bg.label]:
            p = lookup[float(r["eps12"])]
            r = dict(r)
            r["kl_at_20"] = p.kl_at_20
            extended.append(r)

    fieldnames = list(extended[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(extended)
    print(f"  wrote {out_path}", flush=True)
    return extended


def _to_results(rows: list[dict]) -> dict:
    """Convert flat CSV rows to format expected by _summary_table."""
    from eps12_predictiveness_grid import EpsPoint

    out: dict[str, list[EpsPoint]] = defaultdict(list)
    for r in rows:
        out[r["background"]].append(EpsPoint(
            eps12=float(r["eps12"]),
            kl_eq=float(r["kl_eq"]),
            ff_eq=float(r["ff_eq"]),
            psi_ok=bool(int(r["psi_ok"])),
            init_slope_kl=float(r["init_slope_kl"]),
            early_avg_kl=float(r["early_avg_kl"]),
            t_half_kl=float(r["t_half_kl"]) if r["t_half_kl"] else float("nan"),
            init_slope_f1=float(r["init_slope_f1"]),
            kl_final=float(r["kl_final"]),
            kl_at_20=float(r["kl_at_20"]),
        ))
    return dict(out)


def plot_kl20_scatter(results: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    nrow = int(np.ceil(len(labels) / 3))
    ncol = min(3, len(labels))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.4 * nrow), squeeze=False)
    for k, label in enumerate(labels):
        ax = axes[k // ncol][k % ncol]
        pts = results[label]
        kl_eq = np.array([p.kl_eq for p in pts])
        kl20 = np.array([p.kl_at_20 for p in pts])
        slope020 = kl20 / EARLY_WINDOW
        eps = np.array([p.eps12 for p in pts])
        ax.scatter(kl_eq, kl20, c=eps, cmap="coolwarm", s=40, edgecolor="k", lw=0.4)
        lo = 0.0
        hi = max(kl_eq.max(), kl20.max()) * 1.05
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.9, alpha=0.5)
        rho = _spearman(kl_eq, kl20)
        ax.set_title(f"{label}  $\\rho$={rho:+.2f}", fontsize=9)
        ax.set_xlabel(r"$KL_{eq}$ [bits]")
        ax.set_ylabel(rf"$KL({EARLY_WINDOW})$ [bits]")
    for k in range(len(labels), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(rf"$KL_{{eq}}$ vs accumulated $KL({EARLY_WINDOW})$ (y=x = perfect calibration)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--extend", action="store_true", help="Re-run dynamics to add kl_at_20")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_csv(args.csv)
    if _needs_kl20(rows) or args.extend:
        print("Extending CSV with kl_at_20 (re-running dynamics) ...", flush=True)
        rows = _extend_csv(rows, args.out_dir / "stage_a_extended.csv")
    else:
        print(f"Using existing kl_at_20 from {args.csv}", flush=True)

    results = _to_results(rows)
    summary = _summary_table(results)

    with (args.out_dir / "first20_spearman.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    plot_kl20_scatter(results, args.out_dir / "kl_eq_vs_kl20_scatter.png")
    plot_spearman_bars(summary, args.out_dir / "first20_spearman.png")

    print("\n  Spearman rho(KL_eq, early metrics):")
    for row in summary:
        print(f"    {row['background']:10s}  init={row['rho_KLeq_slope']:+.3f}  "
              f"early_avg={row['rho_KLeq_early']:+.3f}  "
              f"KL(20)={row['rho_KLeq_kl20']:+.3f}  "
              f"KL(20)/20={row['rho_KLeq_slope020']:+.3f}")


if __name__ == "__main__":
    main()
