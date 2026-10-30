#!/usr/bin/env python3
"""
Precompute neutral (Sigma=0) stationary distributions for several N values.

Writes one .npz file (default: neutral_cache.npz) used by simplex.py at runtime.

  python build_neutral_cache.py --config neutral_cache_config.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Import core routines from simplex (same directory)
from neutral_cache import list_cached_N, params_signature, resolve_cache_path, save_neutral_phi
from simplex import (
    DEFAULT_CONFIG,
    _as_locus_vec,
    _resolve_n_jobs,
    build_transition_matrix_row_stochastic,
    column_stochastic,
    count_simplex_states,
    fitness_exp,
    load_config,
    mutation_matrix_asymmetric,
    stationary_fast,
)

DEFAULT_BUILD_CONFIG = Path(__file__).resolve().parent / "neutral_cache_config.yaml"


def build_for_N(
    N: int,
    u: np.ndarray,
    v: np.ndarray,
    r: float,
    n_jobs: int,
) -> np.ndarray:
    M_mut = mutation_matrix_asymmetric(u, v)
    omega_neutral = fitness_exp(np.zeros((2, 2)))
    S = count_simplex_states(N)
    print(f"\n--- N = {N}  (S = {S} states) ---", flush=True)
    t0 = time.perf_counter()
    T_row, _, _ = build_transition_matrix_row_stochastic(
        N, omega_neutral, M_mut, r, n_jobs=n_jobs, label=f"neutral N={N}", verbose=True,
    )
    P_col = column_stochastic(T_row)
    pi = stationary_fast(P_col)
    elapsed = time.perf_counter() - t0
    print(f"  stationary phi: sum={pi.sum():.12f}, max={pi.max():.6e}, time={elapsed:.1f}s", flush=True)
    return pi


def main() -> None:
    p = argparse.ArgumentParser(description="Build neutral-equilibrium cache for several N")
    p.add_argument(
        "--config",
        default=str(DEFAULT_BUILD_CONFIG),
        help="YAML config (default: neutral_cache_config.yaml)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if N already present in cache",
    )
    args = p.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    config_dir = config_path.parent

    u = _as_locus_vec(cfg["u"], "u")
    v = _as_locus_vec(cfg["v"], "v")
    r = float(cfg["r"])
    cache_path = resolve_cache_path(cfg, config_dir, u, v, r)
    if cache_path is None:
        raise SystemExit("Caching disabled. Set neutral_cache_dir: cache in config.")

    N_values = [int(n) for n in cfg.get("N_values", [])]
    if not N_values:
        raise SystemExit("Config must list N_values: [5, 10, ...]")

    n_jobs = _resolve_n_jobs(int(cfg.get("n_jobs", -1)))

    print("=" * 72)
    print("build_neutral_cache.py")
    print(f"config:   {config_path}")
    print(f"cache:    {cache_path}")
    print(f"params:   u={u}, v={v}, r={r}")
    print(f"N_values: {N_values}")
    print(f"n_jobs:   {n_jobs}")
    print("=" * 72)

    already = set(list_cached_N(cache_path)) if cache_path.exists() else set()
    if already:
        print(f"Already in cache: {sorted(already)}")

    t_total = time.perf_counter()
    for N in N_values:
        if N in already and not args.force:
            print(f"\n--- N = {N}: skip (already cached; use --force to rebuild) ---")
            continue
        phi = build_for_N(N, u, v, r, n_jobs)
        save_neutral_phi(cache_path, N, phi, u, v, r)
        print(f"  saved to {cache_path}", flush=True)

    print("\n" + "=" * 72)
    print(f"Done. Cached N values: {list_cached_N(cache_path)}")
    print(f"Total time: {time.perf_counter() - t_total:.1f}s")
    print(f"Meta: {params_signature(u, v, r)}")


if __name__ == "__main__":
    main()
