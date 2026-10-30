#!/usr/bin/env python3
r"""
Coupled-mutation variant of the two-locus simplex Moran chain.

Same Wright–Fisher pipeline as ``simplex.py`` (selection → recombination → mutation → drift),
but the 4×4 mutation matrix **M** uses coupled flips in addition to per-locus u, v:

Per-locus singles (unchanged from the independent model):
  u_i : allele 0 → 1 at locus i
  v_i : allele 1 → 0 at locus i

Coupled channels (typically ux, vx ≫ u, v):
  ux  : 00 ↔ 11  (both loci flip together; replaces the old u₁u₂ / v₁v₂ product)
  vx  : 01 ↔ 10  (transposition / reciprocal swap)

Rows of M need **not** sum to 1; ``deterministic_p`` renormalizes after ``p_rec @ M``.

Config keys (in addition to simplex keys):
  ux, vx  — scalar or length-2 list (defaults: 10× mean(u), 10× mean(v))
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from simplex import (  # noqa: F401 — re-export core API
    G,
    GENOTYPE_LABELS,
    SimplexParams,
    build_sigma,
    build_transition_matrix_row_stochastic,
    column_stochastic,
    compute_phi,
    deterministic_p,
    fitness_exp,
    kl_divergence_bits,
    load_config,
    log_mean_fitness_genotype_simplex,
    make_simplex_counts,
    make_simplex_grid,
    params_from_config as _params_from_config_base,
    print_mutation_matrix_numeric,
    stationary_fast,
    try_selection_equilibrium,
    w_potential_per_state,
    _as_locus_vec,
)


@dataclass
class NotSymParams(SimplexParams):
    ux: np.ndarray
    vx: np.ndarray


def _as_rate_vec(value: Any, name: str, default: float | None = None) -> np.ndarray:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return np.full(2, float(default), dtype=float)
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.size == 1:
        return np.full(2, float(arr[0]), dtype=float)
    if arr.size != 2:
        raise ValueError(f"{name} must be a scalar or length-2 list")
    return arr.astype(float)


def mutation_matrix_coupled(
    u: np.ndarray,
    v: np.ndarray,
    ux: np.ndarray,
    vx: np.ndarray,
) -> np.ndarray:
    """
    Build 4×4 mutation matrix with singles + coupled ux (00↔11) and vx (01↔10).

    Genotype order: 00, 01, 10, 11.
    Row sums may differ from 1 (non-stochastic OK).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    ux = np.asarray(ux, dtype=float)
    vx = np.asarray(vx, dtype=float)
    if u.shape != (2,) or v.shape != (2,) or ux.shape != (2,) or vx.shape != (2,):
        raise ValueError("u, v, ux, vx must have length 2")

    u1, u2 = u
    v1, v2 = v
    ux1, ux2 = ux
    vx1, vx2 = vx

    # Use locus-1 rates for 00↔11 and locus-2 for symmetry when vectors differ.
    ux_eff = float(np.sqrt(max(ux1 * ux2, 0.0)))
    vx_eff = float(np.sqrt(max(vx1 * vx2, 0.0)))

    M = np.zeros((4, 4), dtype=float)

    # 00
    M[0, 0] = (1.0 - u1) * (1.0 - u2)
    M[0, 1] = (1.0 - u1) * u2
    M[0, 2] = u1 * (1.0 - u2)
    M[0, 3] = ux_eff

    # 01
    M[1, 0] = (1.0 - u1) * v2
    M[1, 1] = (1.0 - u1) * (1.0 - v2)
    M[1, 2] = vx_eff
    M[1, 3] = u1 * (1.0 - v2)

    # 10
    M[2, 0] = v1 * (1.0 - u2)
    M[2, 1] = vx_eff
    M[2, 2] = (1.0 - v1) * (1.0 - u2)
    M[2, 3] = (1.0 - v1) * u2

    # 11
    M[3, 0] = ux_eff
    M[3, 1] = (1.0 - v1) * v2
    M[3, 2] = v1 * (1.0 - v2)
    M[3, 3] = (1.0 - v1) * (1.0 - v2)

    return M


def default_coupled_rates(u: np.ndarray, v: np.ndarray, *, factor: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Default ux, vx = factor × u, factor × v (per locus)."""
    return factor * u.astype(float), factor * v.astype(float)


def build_mutation_matrix(
    u: np.ndarray,
    v: np.ndarray,
    ux: np.ndarray | None = None,
    vx: np.ndarray | None = None,
    *,
    coupled_factor: float = 10.0,
) -> np.ndarray:
    if ux is None or vx is None:
        ux_d, vx_d = default_coupled_rates(u, v, factor=coupled_factor)
        ux = ux_d if ux is None else _as_locus_vec(ux, "ux")
        vx = vx_d if vx is None else _as_locus_vec(vx, "vx")
    return mutation_matrix_coupled(
        np.asarray(u, dtype=float),
        np.asarray(v, dtype=float),
        _as_locus_vec(ux, "ux"),
        _as_locus_vec(vx, "vx"),
    )


def params_from_config(cfg: dict[str, Any], config_dir: Path | None = None) -> NotSymParams:
    base = _params_from_config_base(cfg, config_dir)
    u, v = base.u, base.v
    factor = float(cfg.get("coupled_factor", 10.0))
    ux = _as_rate_vec(cfg.get("ux"), "ux", default=factor * float(np.mean(u)))
    vx = _as_rate_vec(cfg.get("vx"), "vx", default=factor * float(np.mean(v)))
    return NotSymParams(
        N=base.N,
        r=base.r,
        u=base.u,
        v=base.v,
        Sigma=base.Sigma,
        times=base.times,
        n_jobs=base.n_jobs,
        neutral=base.neutral,
        top_k=base.top_k,
        neutral_cache_file=None,  # coupled M invalidates symmetric cache
        plot_dir=base.plot_dir,
        show_plots=base.show_plots,
        plot_dynamics=base.plot_dynamics,
        compute_f2=base.compute_f2,
        ld_top_k_times=base.ld_top_k_times,
        debug_kl=base.debug_kl,
        ux=ux,
        vx=vx,
    )


def print_mutation_matrix_coupled(M: np.ndarray, *, u, v, ux, vx) -> None:
    print("Coupled mutation matrix M (rows=source, cols=target; 00,01,10,11)")
    print(f"  u={np.asarray(u)}, v={np.asarray(v)}, ux={np.asarray(ux)}, vx={np.asarray(vx)}")
    print_mutation_matrix_numeric(M)
