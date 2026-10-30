#!/usr/bin/env python3
"""Compare discrete φ* to cond-α refs with boundary-mass tricks."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from compare_sigma_dynamics import _make_cfg, cond_alpha_reference_pmf  # noqa: E402
from f2b_reference import (  # noqa: E402
    build_phi_boundary_smeared,
    build_phi_f2b_a,
    build_phi_f2b_b,
    build_phi_f2b_b_epsilon,
    build_phi_shell_blend,
    compare_phi_refs,
)
from simplex import (  # noqa: E402
    build_transition_matrix_row_stochastic,
    column_stochastic,
    compute_phi,
    fitness_exp,
    mutation_matrix_asymmetric,
    params_from_config,
)

OUT = SCRIPT_DIR / "plots" / "f2b"


def main() -> None:
    N = 30
    u = v = 0.01
    r = 0.0
    M_mut = mutation_matrix_asymmetric(np.array([u, u]), np.array([v, v]))
    params0 = params_from_config(_make_cfg(np.zeros((2, 2)), N, u, v, r, [0]), SCRIPT_DIR)
    phi_chain, _, fracs, _, src = compute_phi(params0, M_mut)

    omega0 = fitness_exp(np.zeros((2, 2)))
    T_row, _, _ = build_transition_matrix_row_stochastic(
        N, omega0, M_mut, r, n_jobs=-1, label="neutral", verbose=False,
    )
    P_col = column_stochastic(T_row)

    q_a = build_phi_f2b_a(fracs, M_mut, N, phi_chain)
    q_b = build_phi_f2b_b(fracs, M_mut, N)
    q_f3 = cond_alpha_reference_pmf(fracs, M_mut, N)

    refs = {
        "F2b-B interior only": q_b,
        "F2b-A interior only": q_a,
        "F3 (N) interior": q_f3,
        "F2b-B ε=1/N full lattice": build_phi_f2b_b_epsilon(fracs, M_mut, N),
        "F2b-B + 1-step smear (mass-matched)": build_phi_boundary_smeared(
            fracs, phi_chain, q_b, P_col,
        ),
        "F2b-B + 3-step shell blend": build_phi_shell_blend(
            fracs, phi_chain, q_b, P_col, n_sweep=3,
        ),
        "F2b-A + 1-step smear": build_phi_boundary_smeared(
            fracs, phi_chain, q_a, P_col,
        ),
    }

    rows = compare_phi_refs(phi_chain, refs, fracs)
    print(f"Discrete φ* ({src})  N={N}  u=v={u}\n")
    print(f"{'reference':<42} {'TV_full':>8} {'KL_full':>8} {'bnd_c':>6} {'bnd_r':>6} "
          f"{'TV_int':>8} {'KL_int':>8} {'corr':>6}")
    print("-" * 100)
    for row in rows:
        print(
            f"{row['name']:<42} {row['tv_full']:8.4f} {row['kl_full_bits']:8.4f} "
            f"{row['bnd_mass_chain']:6.3f} {row['bnd_mass_ref']:6.3f} "
            f"{row['tv_int']:8.4f} {row['kl_int_bits']:8.4f} {row['corr_int']:6.3f}",
        )

    best_full = min(rows, key=lambda r: r["tv_full"])
    best_int = min(rows, key=lambda r: r["tv_int"])
    print(f"\nBest full-lattice TV: {best_full['name']}  (TV={best_full['tv_full']:.4f})")
    print(f"Best interior TV:     {best_int['name']}  (TV={best_int['tv_int']:.4f})")

    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    picks = [
        "F2b-B interior only",
        "F2b-B ε=1/N full lattice",
        "F2b-B + 1-step smear (mass-matched)",
        "F2b-B + 3-step shell blend",
        "F3 (N) interior",
        "F2b-A + 1-step smear",
    ]
    interior = np.all(fracs > 0, axis=1)
    pc = phi_chain[interior]
    for ax, name in zip(axes.flat, picks):
        pr = refs[name][interior]
        lo = min(pc[pc > 0].min(), pr[pr > 0].min())
        hi = max(pc.max(), pr.max())
        ax.scatter(pr, pc, s=6, alpha=0.35, c="C0")
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        row = next(r for r in rows if r["name"] == name)
        ax.set_title(
            f"{name[:28]}…\nTV_full={row['tv_full']:.3f}  bnd={row['bnd_mass_ref']:.2f}",
            fontsize=7,
        )
        ax.set_xlabel("ref mass")
        ax.set_ylabel("chain φ*")
        ax.grid(True, alpha=0.3, which="both")
    fig.suptitle(
        rf"Interior scatter: chain $\varphi^*$ vs analytic refs ($N={N}$)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = OUT / "phi_boundary_tricks_compare.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
