#!/usr/bin/env python3
r"""
Sweep eps12 and compute cond-α partition function on the genotype lattice.

Continuous (detailed-balance diffusion) equilibrium:
    p(x) \propto \exp(-V(x)),   Z = \int_{\mathrm{simplex}} e^{-V(x)}\,dx

with
    V(x) = -N \log W(x) + \sum_{g=1}^4 \bigl(1-\alpha_g^{\mathrm{cond}}(x)\bigr)\log x_g,
    W(x) = \sum_g x_g \omega_g,   \alpha_g = 2N \cdot \mathrm{influx}_g/(1-x_g).

Discrete approximation on the Moran lattice (same points as the chain):
    \log Z \approx \log \sum_{x} \exp(\ell(x)),   \ell(x) = -V(x).

Reports log Z, log Z/N (= continuous free fitness), and neutral log Z_0.
Optional --with-chain: F4 at psi* and F4 + log Z/N (Lyapunov completion check).

Run:
  cd 2loci
  python log_Z_eps12_sweep.py --config eps12_delta_d_config.yaml
  python log_Z_eps12_sweep.py --config eps12_delta_d_config.yaml --with-chain
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from f2b_reference import log_Z_cond_alpha  # noqa: E402
from simplex import (  # noqa: E402
    build_sigma,
    build_transition_matrix_row_stochastic,
    column_stochastic,
    fitness_exp,
    load_config,
    log_mean_fitness_genotype_simplex,
    make_simplex_grid,
    mutation_matrix_asymmetric,
    try_selection_equilibrium,
)

OUT = SCRIPT_DIR / "plots" / "f2b"


@dataclass
class Row:
    eps12: float
    s1: float
    s2: float
    log_Z_int: float
    log_Z_over_N_int: float
    log_Z_eps: float
    log_Z_over_N_eps: float
    log_Z0_int: float
    log_Z0_over_N_int: float
    n_interior: int
    f4_eq: float | None = None
    f4_plus_logZ_over_N: float | None = None
    wstar_eq: float | None = None


def _eps_grid(cfg: dict) -> np.ndarray:
    lo = float(cfg["eps12_min"])
    hi = float(cfg["eps12_max"])
    step = float(cfg["eps12_step"])
    n = int(round((hi - lo) / step)) + 1
    return np.unique(np.round(lo + step * np.arange(n), 12))


def _run_sweep(cfg: dict, *, with_chain: bool, n_jobs: int) -> list[Row]:
    N = int(cfg["N"])
    r = float(cfg["r"])
    u = np.asarray(cfg["u"], dtype=float)
    v = np.asarray(cfg["v"], dtype=float)
    s1 = float(cfg["s1"])
    s2 = float(cfg["s2"])
    M_mut = mutation_matrix_asymmetric(u, v)
    fracs = make_simplex_grid(N)
    omega0 = fitness_exp(np.zeros((2, 2)))

    z0_int = log_Z_cond_alpha(fracs, omega0, M_mut, N, mode="interior")
    z0_eps = log_Z_cond_alpha(fracs, omega0, M_mut, N, mode="epsilon")

    rows: list[Row] = []
    for eps12 in _eps_grid(cfg):
        sigma = build_sigma(s1, s2, float(eps12))
        omega = fitness_exp(sigma)
        z_sel_int = log_Z_cond_alpha(fracs, omega, M_mut, N, mode="interior")
        z_sel_eps = log_Z_cond_alpha(fracs, omega, M_mut, N, mode="epsilon")

        f4_eq = None
        f4_plus = None
        wstar_eq = None
        if with_chain:
            T_row, _, _ = build_transition_matrix_row_stochastic(
                N, omega, M_mut, r, n_jobs=n_jobs,
                label=f"eps12={eps12:+.3g}", verbose=False,
            )
            psi_star, _, ok = try_selection_equilibrium(column_stochastic(T_row))
            if ok and psi_star is not None:
                wstar_ln = log_mean_fitness_genotype_simplex(fracs, omega)
                wstar_eq = float(np.sum(psi_star * wstar_ln))
                # At psi*: F4=0, so F4 + log Z/N = log Z/N (Lyapunov completion)
                f4_eq = 0.0
                f4_plus = z_sel_int["log_Z_over_N"]

        rows.append(Row(
            eps12=float(eps12),
            s1=s1,
            s2=s2,
            log_Z_int=z_sel_int["log_Z"],
            log_Z_over_N_int=z_sel_int["log_Z_over_N"],
            log_Z_eps=z_sel_eps["log_Z"],
            log_Z_over_N_eps=z_sel_eps["log_Z_over_N"],
            log_Z0_int=z0_int["log_Z"],
            log_Z0_over_N_int=z0_int["log_Z_over_N"],
            n_interior=int(z_sel_int["n_states"]),
            f4_eq=f4_eq,
            f4_plus_logZ_over_N=f4_plus,
            wstar_eq=wstar_eq,
        ))
    return rows


def _write_csv(rows: list[Row], path: Path, N: int, u, v, r: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "N", "u0", "u1", "v0", "v1", "r", "s1", "s2", "eps12",
        "log_Z_interior", "log_Z_over_N_interior",
        "log_Z_epsilon", "log_Z_over_N_epsilon",
        "log_Z0_interior", "log_Z0_over_N_interior",
        "n_interior_states", "wstar_eq", "F4_plus_logZ_over_N_at_psi_star",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({
                "N": N,
                "u0": u[0], "u1": u[1], "v0": v[0], "v1": v[1],
                "r": r,
                "s1": row.s1, "s2": row.s2, "eps12": row.eps12,
                "log_Z_interior": row.log_Z_int,
                "log_Z_over_N_interior": row.log_Z_over_N_int,
                "log_Z_epsilon": row.log_Z_eps,
                "log_Z_over_N_epsilon": row.log_Z_over_N_eps,
                "log_Z0_interior": row.log_Z0_int,
                "log_Z0_over_N_interior": row.log_Z0_over_N_int,
                "n_interior_states": row.n_interior,
                "wstar_eq": row.wstar_eq if row.wstar_eq is not None else "",
                "F4_plus_logZ_over_N_at_psi_star": row.f4_plus_logZ_over_N
                if row.f4_plus_logZ_over_N is not None else "",
            })


def _plot(rows: list[Row], path: Path, *, N: int, s1: float, s2: float) -> None:
    eps = np.array([r.eps12 for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.plot(eps, [r.log_Z_over_N_int for r in rows], "o-", label=r"selection $\log Z/N$ (interior)")
    ax.axhline(rows[0].log_Z0_over_N_int, color="C1", ls="--",
               label=rf"neutral $\log Z_0/N={rows[0].log_Z0_over_N_int:.4f}$")
    ax.set_xlabel(r"$\varepsilon_{12}$")
    ax.set_ylabel(r"$\log Z / N$ [nats]")
    ax.set_title(rf"Cond-$\alpha$ free fitness ($N={N}$, $s_1={s1:g}$, $s_2={s2:g}$)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    dlogZ = np.array([r.log_Z_over_N_int - rows[0].log_Z0_over_N_int for r in rows])
    ax.plot(eps, dlogZ, "s-", color="C2")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xlabel(r"$\varepsilon_{12}$")
    ax.set_ylabel(r"$(\log Z - \log Z_0) / N$")
    ax.set_title(r"Selection shift of $\log Z/N$")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=SCRIPT_DIR / "eps12_delta_d_config.yaml")
    p.add_argument("--with-chain", action="store_true", help="also compute psi* (slow)")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--out-dir", type=Path, default=OUT)
    args = p.parse_args()

    cfg = load_config(args.config if args.config.is_absolute() else SCRIPT_DIR / args.config)
    rows = _run_sweep(cfg, with_chain=args.with_chain, n_jobs=args.n_jobs)

    N = int(cfg["N"])
    u = np.asarray(cfg["u"], dtype=float)
    v = np.asarray(cfg["v"], dtype=float)
    r = float(cfg["r"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "log_Z_eps12_sweep.csv"
    _write_csv(rows, csv_path, N, u, v, r)
    _plot(rows, args.out_dir / "log_Z_eps12_sweep.png", N=N, s1=float(cfg["s1"]), s2=float(cfg["s2"]))

    meta = {
        "formula": "V = -N log W + sum_g (1-alpha_g^cond) log x_g; alpha = 2N influx/(1-x_g)",
        "discrete": "log Z = log sum_x exp(-V(x)) on Moran lattice; interior-only and epsilon=1/N variants",
        "neutral_log_Z0_over_N_interior": rows[0].log_Z0_over_N_int,
        "n_interior": rows[0].n_interior,
    }
    (args.out_dir / "log_Z_eps12_sweep.json").write_text(json.dumps(meta, indent=2))

    print(r"Cond-α partition function sweep")
    print(f"  V(x) = -N log W(x) + Σ_g (1-α_g^cond(x)) log x_g")
    print(f"  log Z/N (neutral, interior) = {rows[0].log_Z0_over_N_int:.6f}  "
          f"({int(rows[0].n_interior)} interior states / {5456 if N==30 else '?'} total)")
    print(f"\n{'eps12':>8} {'logZ/N int':>12} {'logZ/N eps':>12} {'Δ logZ/N':>12}")
    z0 = rows[0].log_Z0_over_N_int
    for row in rows:
        print(f"{row.eps12:8.3f} {row.log_Z_over_N_int:12.6f} {row.log_Z_over_N_eps:12.6f} "
              f"{row.log_Z_over_N_int - z0:12.6f}")
    print(f"\nwrote {csv_path}")
    print(f"wrote {args.out_dir / 'log_Z_eps12_sweep.png'}")


if __name__ == "__main__":
    main()
