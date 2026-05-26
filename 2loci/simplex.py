#!/usr/bin/env python3
"""
Two-locus Wright–Fisher Markov chain on the 4-genotype simplex.

State space
-----------
Population haplotype frequencies live on the 3-simplex. With four genotypes
(00, 01, 10, 11) we write x = (x_00, x_01, x_10, x_11) with sum_g x_g = 1.

For population size N we discretize to lattice points whose count vectors
(c_00, c_01, c_10, c_11) are non-negative integers summing to N. There are
  S = C(N+3, 3) = (N+3 choose 3)
such states.

One Wright–Fisher generation at frequency vector x
-------------------------------------------------
  1. Selection:   p_sel(g) = x_g * omega(g) / sum_g' x_g' * omega(g')
  2. Recombination (LD D-shift, same as Markov_2_loci.recombine_D_shift):
       D = x_00 * x_11 - x_01 * x_10
       x_00 <- x_00 - r*D;  x_01 <- x_01 + r*D
       x_10 <- x_10 + r*D;  x_11 <- x_11 - r*D   (then renormalize)
  3. Mutation:    p_mut = p_rec @ M   (row vector times row-stochastic M)
  4. Drift:       offspring counts y ~ Multinomial(N, p_mut)

Fitness
-------
  omega(g) = exp(g^T Sigma g)
with 2×2 symmetric Sigma (Sigma_ii = s_i, Sigma_ij = eps_12/2 for i != j).

Config: YAML file (see simplex_config.yaml). Run:
  python simplex.py --config simplex_config.yaml

Neutral equilibrium phi (recommended for N >= 20):
  python build_neutral_cache.py --config neutral_cache_config.yaml
  # writes cache/neutral_u0_...npz; simplex.py loads phi for matching u, v, r, N
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import gammaln

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from neutral_cache import load_neutral_phi, resolve_cache_path, save_neutral_phi

# ---------------------------------------------------------------------------
# Genotype encoding
# ---------------------------------------------------------------------------
# Flat index order used internally for vectors and the 4×4 mutation matrix:
#   index 0 → 00,  index 1 → 01,  index 2 → 10,  index 3 → 11
#
# Fitness display uses a 2×2 grid in allele space (locus 1 = rows, locus 2 = cols):
#
#              locus 2
#              allele 0    allele 1
#   locus 1 0    ω(00)       ω(01)
#           1    ω(10)       ω(11)
# ---------------------------------------------------------------------------

GENOTYPE_LABELS = ("00", "01", "10", "11")
G = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
DEFAULT_CONFIG = Path(__file__).resolve().parent / "simplex_config.yaml"

# Index map: flat genotype index → (row, col) in the 2×2 fitness grid
_FITNESS_GRID_INDEX = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}


@dataclass
class SimplexParams:
    """Runtime parameters loaded from YAML."""

    N: int
    r: float
    u: np.ndarray
    v: np.ndarray
    Sigma: np.ndarray
    times: list[int]
    n_jobs: int
    neutral: bool
    top_k: int
    neutral_cache_file: Path | None
    plot_dir: Path | None
    show_plots: bool
    plot_dynamics: bool
    compute_f2: bool
    ld_top_k_times: str
    debug_kl: bool


# =============================================================================
# Config loading
# =============================================================================


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML or JSON config file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        if not HAS_YAML:
            raise ImportError("YAML config requires PyYAML: pip install pyyaml")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def _as_locus_vec(value: Any, name: str) -> np.ndarray:
    """Scalar → repeated vector; length-2 list → per-locus rates."""
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.size == 1:
        return np.full(2, float(arr[0]), dtype=float)
    if arr.size != 2:
        raise ValueError(f"{name} must be a scalar or length-2 list")
    return arr.astype(float)


def _resolve_n_jobs(n_jobs: int) -> int:
    """n_jobs = -1 means all available CPU cores."""
    if n_jobs < 0:
        return max(1, os.cpu_count() or 1)
    return max(1, n_jobs)


def sigma_from_config(cfg: dict[str, Any]) -> np.ndarray:
    """Parse Sigma from config: direct matrix, file, or (s, eps12)."""
    if cfg.get("sigma_file"):
        path = Path(cfg["sigma_file"])
        if not path.is_absolute():
            path = Path(cfg.get("_config_dir", ".")) / path
        Sigma = np.load(path).astype(float)
    elif cfg.get("sigma") is not None:
        Sigma = np.array(cfg["sigma"], dtype=float)
    elif "s" in cfg and "eps12" in cfg:
        s = _as_locus_vec(cfg["s"], "s")
        eps12 = float(cfg["eps12"])
        Sigma = build_sigma(float(s[0]), float(s[1]), eps12)
    else:
        Sigma = np.zeros((2, 2), dtype=float)

    if Sigma.shape != (2, 2):
        raise ValueError(f"Sigma must be 2x2, got shape {Sigma.shape}")
    if not np.allclose(Sigma, Sigma.T):
        asym = float(np.max(np.abs(Sigma - Sigma.T)))
        print(
            f"WARNING: Sigma not symmetric (max |Sigma-Sigma.T| = {asym:.3g}); "
            "using (Sigma + Sigma.T) / 2",
            flush=True,
        )
        Sigma = (Sigma + Sigma.T) / 2.0
    return Sigma


def params_from_config(cfg: dict[str, Any], config_dir: Path | None = None) -> SimplexParams:
    if "N" not in cfg:
        raise ValueError("Config must include N")
    if "r" not in cfg:
        raise ValueError("Config must include r")
    if "u" not in cfg or "v" not in cfg:
        raise ValueError("Config must include u and v")

    if config_dir is None:
        config_dir = Path(cfg.get("_config_dir", "."))

    u = _as_locus_vec(cfg["u"], "u")
    v = _as_locus_vec(cfg["v"], "v")
    r = float(cfg["r"])

    times = cfg.get("times")
    if times is None:
        times = [1, 5, 10, 50]
    times = [int(t) for t in times]

    plot_dir_raw = cfg.get("plot_dir")
    if plot_dir_raw is None:
        plot_dir = config_dir / "plots"
    else:
        plot_dir = Path(plot_dir_raw)
        if not plot_dir.is_absolute():
            plot_dir = config_dir / plot_dir

    return SimplexParams(
        N=int(cfg["N"]),
        r=r,
        u=u,
        v=v,
        Sigma=sigma_from_config(cfg),
        times=times,
        n_jobs=_resolve_n_jobs(int(cfg.get("n_jobs", -1))),
        neutral=bool(cfg.get("neutral", False)),
        top_k=int(cfg.get("top_k", 20)),
        neutral_cache_file=resolve_cache_path(cfg, config_dir, u, v, r),
        plot_dir=plot_dir,
        show_plots=bool(cfg.get("show_plots", False)),
        plot_dynamics=bool(cfg.get("plot_dynamics", True)),
        compute_f2=bool(cfg.get("compute_f2", False)),
        ld_top_k_times=str(cfg.get("ld_top_k_times", "first_last")),
        debug_kl=bool(cfg.get("debug_kl", False)),
    )


# =============================================================================
# Simplex lattice
# =============================================================================


def count_simplex_states(N: int, k: int = 4) -> int:
    """Number of lattice points on the (k-1)-simplex with denominator N."""
    return comb(N + k - 1, k - 1)


def _compositions(n: int, k: int):
    """Yield all k-tuples of non-negative integers summing to n."""
    if k == 1:
        yield (n,)
        return
    for i in range(n + 1):
        for rest in _compositions(n - i, k - 1):
            yield (i,) + rest


def make_simplex_grid(N: int) -> np.ndarray:
    """Return (S, 4) array of frequency vectors on the simplex grid."""
    points = [tuple(c / float(N) for c in comp) for comp in _compositions(N, 4)]
    return np.array(points, dtype=float)


def make_simplex_counts(N: int) -> np.ndarray:
    """Return (S, 4) array of integer count vectors summing to N."""
    return np.array(list(_compositions(N, 4)), dtype=int)


# =============================================================================
# Epistasis and fitness
# =============================================================================


def build_sigma(s1: float, s2: float, eps12: float) -> np.ndarray:
    """Build 2×2 Sigma with diagonal s_i and off-diagonal eps_12/2."""
    return np.array([[s1, eps12 / 2.0], [eps12 / 2.0, s2]], dtype=float)


def fitness_exp(Sigma: np.ndarray) -> np.ndarray:
    """
    omega(g) = exp(g^T Sigma g) for all four genotypes.

    Returns length-4 vector in flat order [00, 01, 10, 11].
    """
    quad = np.einsum("gi,ij,gj->g", G, Sigma, G)
    return np.exp(quad)


def omega_to_fitness_grid(omega: np.ndarray) -> np.ndarray:
    """
    Map flat omega vector to 2×2 allele grid.

    Rows = locus-1 allele (0 then 1); columns = locus-2 allele (0 then 1).
    """
    grid = np.empty((2, 2), dtype=float)
    for idx, (row, col) in _FITNESS_GRID_INDEX.items():
        grid[row, col] = omega[idx]
    return grid


# =============================================================================
# Mutation matrix M
# =============================================================================
#
# Independent per-locus mutation. For locus i:
#   P(0→0) = 1 - u_i,   P(0→1) = u_i
#   P(1→0) = v_i,       P(1→1) = 1 - v_i
#
# Joint 4×4 matrix M (rows = source genotype, cols = target genotype).
# Genotype order: 00, 01, 10, 11.  Entry = product of the two locus factors.
#
#              target
#              00              01              10              11
# from 00   (1-u_1)(1-u_2)      (1-u_1)u_2      u_1(1-u_2)         u_1*u_2
#      01       (1-u_1)v_2  (1-u_1)(1-v_2)         u_1*v_2      u_1(1-v_2)
#      10       v_1(1-u_2)         v_1*u_2  (1-v_1)(1-u_2)      (1-v_1)u_2
#      11          v_1*v_2      v_1(1-v_2)      (1-v_1)v_2  (1-v_1)(1-v_2)
#
# After selection/recombination, post-mutation gamete frequencies are
#   p_mut = p_rec @ M   (1×4 row vector).
# =============================================================================


def mutation_matrix_asymmetric(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Build the 4×4 row-stochastic mutation matrix for asymmetric per-locus rates.

    See the comment block above for the closed-form layout in u_i, v_i.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != (2,) or v.shape != (2,):
        raise ValueError("u and v must have length 2")
    M = np.zeros((4, 4), dtype=float)
    for g_from in range(4):
        for g_to in range(4):
            prob = 1.0
            for locus in range(2):
                a = int(G[g_from, locus])
                b = int(G[g_to, locus])
                if a == 0 and b == 0:
                    prob *= 1.0 - u[locus]
                elif a == 0 and b == 1:
                    prob *= u[locus]
                elif a == 1 and b == 0:
                    prob *= v[locus]
                else:
                    prob *= 1.0 - v[locus]
            M[g_from, g_to] = prob
    return M


def _mutation_entry_formula(g_from: int, g_to: int) -> str:
    """Symbolic formula for M[g_from, g_to] in u_i, v_i."""
    factors: list[str] = []
    for locus in range(2):
        i = locus + 1
        a = int(G[g_from, locus])
        b = int(G[g_to, locus])
        if a == 0 and b == 0:
            factors.append(f"(1-u_{i})")
        elif a == 0 and b == 1:
            factors.append(f"u_{i}")
        elif a == 1 and b == 0:
            factors.append(f"v_{i}")
        else:
            factors.append(f"(1-v_{i})")
    if len(factors) == 1:
        return factors[0]
    out = factors[0]
    for f in factors[1:]:
        if out.endswith(")") or f.startswith("("):
            out += f
        else:
            out += "*" + f
    return out


# =============================================================================
# One-generation deterministic map (before multinomial drift)
# =============================================================================


def selection_step(x: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Post-selection gamete frequencies: p_sel(g) ∝ x_g * omega(g)."""
    x = np.asarray(x, dtype=float)
    xw = x * omega
    sum_of_fitnesses = float(xw.sum()) # sum of fitnesses, what we normalize with
    if sum_of_fitnesses <= 0.0:
        return np.full(4, 0.25, dtype=float) # this means some mistake happens--> we dub every genotype as equally likely =0.25-- should never reach here
    return xw / sum_of_fitnesses # a vector of relative fitness


def recombine_D_shift(x: np.ndarray, r: float) -> np.ndarray:
    """
    Recombination via linkage disequilibrium D-shift (Markov_2_loci).

    D = x_00*x_11 - x_01*x_10 measures LD. Recombination at rate r pushes
    haplotype frequencies toward linkage equilibrium while preserving both
    marginal allele frequencies.
    """
    x = np.asarray(x, dtype=float)
    x00, x01, x10, x11 = x
    D = x00 * x11 - x01 * x10
    xr = np.array([x00 - r * D, x01 + r * D, x10 + r * D, x11 - r * D], dtype=float)
    xr = np.maximum(xr, 0.0)
    sum_of_frequencies = float(xr.sum())
    if sum_of_frequencies > 0.0:
        return xr / sum_of_frequencies
    return np.full(4, 0.25, dtype=float)


def deterministic_p(
    x: np.ndarray,
    omega: np.ndarray,
    M_mut: np.ndarray,
    r: float,
) -> np.ndarray:
    """Full deterministic step x → selection → recombination → mutation."""
    x = np.asarray(x, dtype=float)
    sum_of_frequencies = float(x.sum())
    if sum_of_frequencies <= 0.0:
        x = np.full(4, 0.25, dtype=float)
    else:
        x = x / sum_of_frequencies
    xs = selection_step(x, omega)
    xr = recombine_D_shift(xs, r)
    xm = xr @ M_mut
    xm = np.maximum(xm, 0.0) # making sure only positive values
    sum_of_frequencies_after_mutation = float(xm.sum())
    if sum_of_frequencies_after_mutation > 0.0:
        return xm / sum_of_frequencies_after_mutation
    return np.full(4, 0.25, dtype=float)


# =============================================================================
# Markov transition matrix
# =============================================================================
#
# Row i of T corresponds to source state i (count vector summing to N).
# Entry T[i,j] = Multinomial(N, p_mut(x_i))[y_j].
#
# Parallel strategy (n_jobs > 1):
#   - Split row indices into chunks (~S / (4 * n_jobs) rows each)
#   - concurrent.futures.ProcessPoolExecutor runs chunks in separate processes
#   - Each worker vectorizes the multinomial log-PMF over all target states
#   - Optional: install joblib and set use_joblib=True for alternative backend
# Memory note: full matrix has S² entries (~1 GB float64 at N=40).
# =============================================================================


def transition_row_logpmf(
    x_counts: np.ndarray,
    N: int,
    omega: np.ndarray,
    M_mut: np.ndarray,
    r: float,
    states_counts: np.ndarray,
    logNfact: float,
    log_counts_fact_sum: np.ndarray,
) -> np.ndarray:
    """Log Multinomial(N, p_mut) probability for every target state (vectorized)."""
    x = x_counts.astype(float)
    p = deterministic_p(x / float(N), omega, M_mut, r)
    p = np.maximum(p, 1e-300)
    logp = np.log(p)
    y_dot_logp = states_counts @ logp
    return logNfact - log_counts_fact_sum + y_dot_logp


def _build_row(
    i: int,
    counts: np.ndarray,
    N: int,
    omega: np.ndarray,
    M_mut: np.ndarray,
    r: float,
    states_counts: np.ndarray,
    logNfact: float,
    log_counts_fact_sum: np.ndarray,
) -> np.ndarray:
    """One row of the row-stochastic transition matrix."""
    log_row = transition_row_logpmf(
        counts[i], N, omega, M_mut, r, states_counts, logNfact, log_counts_fact_sum
    )
    row = np.exp(log_row)
    row = np.maximum(row, 0.0)
    s = row.sum()
    if s > 0.0:
        row /= s
    return row


def _build_row_chunk(
    row_indices: np.ndarray,
    counts: np.ndarray,
    N: int,
    omega: np.ndarray,
    M_mut: np.ndarray,
    r: float,
    logNfact: float,
    log_counts_fact_sum: np.ndarray,
) -> list[tuple[int, np.ndarray]]:
    """Build several rows in one worker task (reduces scheduling overhead)."""
    out: list[tuple[int, np.ndarray]] = []
    for i in row_indices:
        row = _build_row(
            int(i), counts, N, omega, M_mut, r,
            counts, logNfact, log_counts_fact_sum,
        )
        out.append((int(i), row))
    return out


def build_transition_matrix_row_stochastic(
    N: int,
    omega: np.ndarray,
    M_mut: np.ndarray,
    r: float,
    n_jobs: int = 1,
    label: str = "transition",
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build row-stochastic T[i,j] = Pr(state i → state j).

    Parallel strategy (n_jobs > 1):
      - ProcessPoolExecutor dispatches row chunks to separate processes
      - Each process runs vectorized NumPy (avoids the GIL)
    """
    fracs = make_simplex_grid(N)
    counts = make_simplex_counts(N)
    S = counts.shape[0]
    logNfact = float(gammaln(N + 1.0))
    log_counts_fact_sum = np.sum(gammaln(counts.astype(float) + 1.0), axis=1)

    T = np.empty((S, S), dtype=float)
    t0 = time.perf_counter()

    if verbose:
        print(
            f"Building {label} matrix: S = C(N+3,3) = {S} states, "
            f"n_jobs = {n_jobs} ...",
            flush=True,
        )

    if n_jobs <= 1:
        step = max(1, S // 20)
        for i in range(S):
            T[i, :] = _build_row(
                i, counts, N, omega, M_mut, r,
                counts, logNfact, log_counts_fact_sum,
            )
            if verbose and (i % step == 0 or i == S - 1):
                print(f"  {label}: row {i + 1}/{S}", flush=True)
    else:
        chunk_size = max(1, S // (4 * n_jobs))
        chunks = [
            np.arange(start, min(start + chunk_size, S), dtype=int)
            for start in range(0, S, chunk_size)
        ]
        if verbose:
            print(
                f"  {label}: {len(chunks)} chunks × ~{chunk_size} rows "
                f"(ProcessPoolExecutor, {n_jobs} workers)",
                flush=True,
            )

        # Try joblib first if installed; otherwise stdlib multiprocessing.
        try:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
                delayed(_build_row_chunk)(
                    chunk, counts, N, omega, M_mut, r, logNfact, log_counts_fact_sum,
                )
                for chunk in chunks
            )
            done = 0
            for chunk_result in results:
                for i, row in chunk_result:
                    T[i, :] = row
                done += len(chunk_result)
                if verbose:
                    print(f"  {label}: assembled {done}/{S} rows", flush=True)
        except ImportError:
            done = 0
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                futures = {
                    pool.submit(
                        _build_row_chunk,
                        chunk, counts, N, omega, M_mut, r, logNfact, log_counts_fact_sum,
                    ): chunk
                    for chunk in chunks
                }
                for fut in as_completed(futures):
                    chunk_result = fut.result()
                    for i, row in chunk_result:
                        T[i, :] = row
                    done += len(chunk_result)
                    if verbose:
                        print(f"  {label}: assembled {done}/{S} rows", flush=True)

    if verbose:
        elapsed = time.perf_counter() - t0
        print(f"  {label}: finished in {elapsed:.1f}s", flush=True)

    return T, fracs, counts


def column_stochastic(T_row: np.ndarray) -> np.ndarray:
    """P_col = T_row.T so evolution is v_next = P_col @ v."""
    return T_row.T


def stationary_from_matrix(
    P_col: np.ndarray,
    tol: float = 1e-12,
    max_iter: int = 5000,
) -> np.ndarray:
    """
    Stationary distribution pi with P_col @ pi = pi.

    Uses power iteration (robust for large stochastic matrices). Starts
    from uniform; converges when the chain is irreducible (u, v > 0).
    """
    S = P_col.shape[0]
    pi = np.full(S, 1.0 / S, dtype=float)
    for _ in range(max_iter):
        pi_next = P_col @ pi
        s = float(pi_next.sum())
        if s <= 0.0:
            raise RuntimeError("Power iteration collapsed to zero vector")
        pi_next /= s
        diff = float(np.sum(np.abs(pi_next - pi)))
        if diff < tol:
            return pi_next
        pi = pi_next
    return pi


def kl_divergence_bits(p: np.ndarray, q: np.ndarray, eps: float = 1e-16) -> float:
    """
    D_KL(p || q) with log base 2 (bits):

        D_KL = sum_i p_i log2(p_i / q_i)
    """
    p_ = np.asarray(p, dtype=float) + eps
    q_ = np.asarray(q, dtype=float) + eps
    p_ /= p_.sum()
    q_ /= q_.sum()
    log2 = math.log(2.0)
    return float(np.sum(p_ * (np.log(p_) - np.log(q_))) / log2)


def print_kl_debug_sanity(
    *,
    omega: np.ndarray,
    M_mut: np.ndarray,
    r: float,
    P_col: np.ndarray,
    phi: np.ndarray,
    fracs: np.ndarray,
    result: DynamicsResult | None = None,
    P_col_neutral: np.ndarray | None = None,
    sample_rows: int = 8,
) -> None:
    """
    Print checks that selection enters dynamics and KL responds to Sigma.

    KL is D_KL(psi(t) || phi) with fixed neutral phi; psi evolves via P_col(omega).
    """
    omega = np.asarray(omega, dtype=float)
    omega_n = np.ones(4, dtype=float)
    x_uniform = np.full(4, 0.25, dtype=float)

    print("KL debug  (D_KL(psi(t) || phi),  phi = neutral equilibrium,  psi(0) = phi)")
    print(f"  omega = {omega}  (genotypes 00,01,10,11)")
    print(f"  fitness ratio max/min = {float(omega.max() / max(omega.min(), 1e-300)):.6g}")
    p_sel = deterministic_p(x_uniform, omega, M_mut, r)
    p_neu = deterministic_p(x_uniform, omega_n, M_mut, r)
    print(f"  one-gen gamete freq change at x=uniform: L1={float(np.sum(np.abs(p_sel - p_neu))):.6e}")
    if np.allclose(omega, 1.0):
        print("  WARNING: omega = 1 for all genotypes — selection off; KL should stay ~ 0")

    psi1 = P_col @ phi
    s1 = float(psi1.sum())
    if s1 > 0.0:
        psi1 /= s1
    kl1 = kl_divergence_bits(psi1, phi)
    print(f"  KL after 1 generation: {kl1:.8e} bits")

    if P_col_neutral is not None:
        rng = np.random.default_rng(0)
        n = min(sample_rows, P_col.shape[0])
        idx = rng.choice(P_col.shape[0], size=n, replace=False)
        max_row_diff = max(float(np.max(np.abs(P_col[i] - P_col_neutral[i]))) for i in idx)
        print(f"  sampled max row L1 |P_sel - P_neu| over {n} states: {max_row_diff:.6e}")
        if max_row_diff < 1e-12:
            print("  WARNING: P_col identical to neutral — KL cannot grow with this Sigma")

    if result is not None and len(result.times) >= 2:
        t_last = int(result.times[-1])
        kl_last = float(result.kl_phi_bits[-1])
        print(f"  KL at t={t_last}: {kl_last:.8e} bits")
        if not np.allclose(omega, 1.0) and kl_last < 1e-10 and t_last >= 50:
            print(
                "  WARNING: KL still ~ 0 at late t despite selection — "
                "check Sigma magnitude or neutral flag"
            )
    print()


def allele_freqs_from_genotype_fracs(fracs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Marginal allele-1 frequencies (p1, p2) from genotype fractions (00, 01, 10, 11)."""
    p1 = fracs[:, 1] + fracs[:, 3]
    p2 = fracs[:, 1] + fracs[:, 2]
    return p1, p2


def mutation_log_terms_genotype_simplex(
    fracs: np.ndarray,
    M_mut: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Mutation potential at genotype composition x = (x_00, x_01, x_10, x_11):

        Σ_i (x @ M)_i log x_i

    (x @ M)_i is the post-mutation mass at genotype i from current frequencies x.
    Genotype order: 00, 01, 10, 11 (same as M rows/columns).
    """
    x = np.clip(fracs, eps, 1.0 - eps)
    x_after_mutation = x @ M_mut
    return np.sum(x_after_mutation * np.log(x), axis=1)


def log_mean_fitness_genotype_simplex(
    fracs: np.ndarray,
    omega: np.ndarray,
    eps: float = 1e-300,
) -> np.ndarray:
    """log E_x[e^{g^T Sigma g}] = log sum_g x_g omega(g)."""
    x = np.maximum(fracs, eps)
    return np.log(x @ omega)


def w_potential_per_state(
    fracs: np.ndarray,
    omega: np.ndarray,
    M_mut: np.ndarray,
    N: int,
) -> np.ndarray:
    """
    W(x) = log E_x[e^{g^T Sigma g}] - (1/2N) Σ_i (x @ M)_i log x_i.

    Shape (S,) — one value per simplex lattice point.
    """
    log_w = log_mean_fitness_genotype_simplex(fracs, omega)
    mut = mutation_log_terms_genotype_simplex(fracs, M_mut)
    return log_w - mut / (2.0 * float(N))


def linkage_disequilibrium_per_state(fracs: np.ndarray) -> np.ndarray:
    """D(x) = x_00 x_11 - x_01 x_10 for each lattice point."""
    return fracs[:, 0] * fracs[:, 3] - fracs[:, 1] * fracs[:, 2]


def shannon_entropy_nats(p: np.ndarray, eps: float = 1e-300) -> float:
    """H(p) = -sum_i p_i log p_i in nats."""
    p_ = np.asarray(p, dtype=float)
    p_ = np.maximum(p_, eps)
    return float(-np.sum(p_ * np.log(p_)))


def free_fitness_1(
    rho: np.ndarray,
    w_per_state: np.ndarray,
    N: int,
) -> float:
    """
    F_1(rho) = <W>_rho + H(rho) / (2N).

    W includes selection (log mean fitness) and mutation term Σ_i (x@M)_i log x_i.
    """
    rho = np.asarray(rho, dtype=float)
    return float(np.sum(rho * w_per_state) + shannon_entropy_nats(rho) / (2.0 * float(N)))


def mean_linkage_disequilibrium(rho: np.ndarray, d_per_state: np.ndarray) -> float:
    """E_rho[D(x)] with D = x_00 x_11 - x_01 x_10."""
    return float(np.sum(np.asarray(rho, dtype=float) * d_per_state))


def consecutive_rates(
    times: list[int],
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_end, rate) for each consecutive interval in times."""
    t_arr = np.asarray(times, dtype=float)
    v_arr = np.asarray(values, dtype=float)
    dt = np.diff(t_arr)
    delta = np.diff(v_arr)
    rate = np.divide(delta, dt, out=np.full_like(delta, np.nan), where=dt > 0)
    return t_arr[1:], rate


def try_selection_equilibrium(
    P_col: np.ndarray,
    *,
    tol: float = 1e-12,
    max_iter: int = 100000,
    residual_tol: float = 1e-6,
) -> tuple[np.ndarray | None, float, bool]:
    """
    Attempt psi* with P_col @ psi* = psi* (selection+r recombination stationary).

    Returns (psi_star, ||P psi* - psi*||_1, converged).
    converged is True when the residual is below residual_tol after power iteration.
    """
    try:
        psi_star = stationary_from_matrix(P_col, tol=tol, max_iter=max_iter)
    except RuntimeError:
        return None, float("nan"), False
    residual = float(np.linalg.norm(P_col @ psi_star - psi_star, ord=1))
    converged = bool(np.isfinite(residual) and residual < residual_tol)
    return psi_star, residual, converged


@dataclass
class DynamicsResult:
    times: list[int]
    kl_phi_bits: np.ndarray
    f1: np.ndarray
    f2_bits: np.ndarray | None
    mean_ld: np.ndarray
    rate_times: np.ndarray
    rate_kl_bits: np.ndarray
    rate_f1: np.ndarray
    rate_f2_bits: np.ndarray | None
    top_k_ld_times: list[int]
    top_k_ld_snapshots: dict[int, list[tuple[int, np.ndarray, np.ndarray, float, float]]]


def _record_top_k_ld(
    psi: np.ndarray,
    counts: np.ndarray,
    fracs: np.ndarray,
    d_per_state: np.ndarray,
    top_k: int,
) -> list[tuple[int, np.ndarray, np.ndarray, float, float]]:
    """Return top_k lattice states: (rank, counts, freqs, mass, D)."""
    order = np.argsort(-psi)
    out: list[tuple[int, np.ndarray, np.ndarray, float, float]] = []
    for rank, idx in enumerate(order[:top_k], start=1):
        out.append((
            rank,
            counts[idx].copy(),
            fracs[idx].copy(),
            float(psi[idx]),
            float(d_per_state[idx]),
        ))
    return out


def _ld_top_k_times_to_report(
    times: list[int],
    mode: str,
) -> set[int]:
    """Which sampled times get a full top-k LD table."""
    if mode == "all":
        return set(times)
    if mode == "first_last":
        return {times[0], times[-1]}
    raise ValueError(f"ld_top_k_times must be 'first_last' or 'all', got {mode!r}")


def dynamics_series(
    P_col: np.ndarray,
    phi: np.ndarray,
    times: list[int],
    w_per_state: np.ndarray,
    d_per_state: np.ndarray,
    counts: np.ndarray,
    fracs: np.ndarray,
    N: int,
    top_k: int,
    ld_top_k_times: str,
    psi_star: np.ndarray | None = None,
) -> DynamicsResult:
    """
    Evolve psi from phi; record KL(psi||phi), F_1, optional F_2=KL(psi||psi*), mean LD.
    """
    times_unique = sorted(set(int(t) for t in times))
    if 0 not in times_unique:
        times_unique = [0] + times_unique

    kl_vals: list[float] = []
    f1_vals: list[float] = []
    f2_vals: list[float] | None = [] if psi_star is not None else None
    ld_vals: list[float] = []
    ld_report_times = _ld_top_k_times_to_report(times_unique, ld_top_k_times)
    top_k_ld_snapshots: dict[int, list[tuple[int, np.ndarray, np.ndarray, float, float]]] = {}

    psi = np.asarray(phi, dtype=float).copy()
    prev_t = 0
    kl_vals.append(0.0 if times_unique[0] == 0 else kl_divergence_bits(psi, phi))
    f1_vals.append(free_fitness_1(psi, w_per_state, N))
    if f2_vals is not None:
        f2_vals.append(kl_divergence_bits(psi, psi_star))
    ld_vals.append(mean_linkage_disequilibrium(psi, d_per_state))
    if times_unique[0] in ld_report_times:
        top_k_ld_snapshots[times_unique[0]] = _record_top_k_ld(
            psi, counts, fracs, d_per_state, top_k,
        )

    for t in times_unique[1:]:
        if t < prev_t:
            raise ValueError("times must be non-negative; re-sort internally failed")
        for _ in range(t - prev_t):
            psi = P_col @ psi
            s = float(psi.sum())
            if s > 0.0:
                psi /= s
        kl_vals.append(kl_divergence_bits(psi, phi))
        f1_vals.append(free_fitness_1(psi, w_per_state, N))
        if f2_vals is not None:
            f2_vals.append(kl_divergence_bits(psi, psi_star))
        ld_vals.append(mean_linkage_disequilibrium(psi, d_per_state))
        if t in ld_report_times:
            top_k_ld_snapshots[t] = _record_top_k_ld(
                psi, counts, fracs, d_per_state, top_k,
            )
        prev_t = t

    kl_arr = np.array(kl_vals, dtype=float)
    f1_arr = np.array(f1_vals, dtype=float)
    f2_arr = np.array(f2_vals, dtype=float) if f2_vals is not None else None
    ld_arr = np.array(ld_vals, dtype=float)

    rate_times, rate_kl = consecutive_rates(times_unique, kl_arr)
    _, rate_f1 = consecutive_rates(times_unique, f1_arr)
    rate_f2 = consecutive_rates(times_unique, f2_arr)[1] if f2_arr is not None else None

    return DynamicsResult(
        times=times_unique,
        kl_phi_bits=kl_arr,
        f1=f1_arr,
        f2_bits=f2_arr,
        mean_ld=ld_arr,
        rate_times=rate_times,
        rate_kl_bits=rate_kl,
        rate_f1=rate_f1,
        rate_f2_bits=rate_f2,
        top_k_ld_times=sorted(top_k_ld_snapshots.keys()),
        top_k_ld_snapshots=top_k_ld_snapshots,
    )


def _setup_plot_style() -> None:
    """Serif + Computer Modern mathtext for LaTeX-like figures."""
    if not HAS_MPL:
        return
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })


def _plot_param_title_lines(params: SimplexParams) -> list[str]:
    """Multi-line parameter block for figure titles (each line one math string)."""
    sig = params.Sigma
    u_str = ",\\,".join(f"{x:.2g}" for x in params.u)
    v_str = ",\\,".join(f"{x:.2g}" for x in params.v)
    s1 = float(sig[0, 0])
    s2 = float(sig[1, 1])
    eps12 = 2.0 * float(sig[0, 1])
    return [
        rf"$N={params.N},\quad r={params.r}$",
        rf"$u=({u_str}),\quad v=({v_str})$",
        rf"$s_1={s1:.3g},\quad s_2={s2:.3g},\quad \epsilon_{{12}}={eps12:.3g}$",
    ]


def _selection_param_slug(params: SimplexParams) -> str:
    """Filename-safe slug from r, s1, s2, eps12 (= 2 Sigma_12)."""
    sig = params.Sigma
    s1 = float(sig[0, 0])
    s2 = float(sig[1, 1])
    eps12 = 2.0 * float(sig[0, 1])

    def _fmt(x: float) -> str:
        return f"{x:.4g}".replace("+", "")

    return f"r_{_fmt(params.r)}_s1_{_fmt(s1)}_s2_{_fmt(s2)}_eps12_{_fmt(eps12)}"


def _style_twin_axes(ax_left, ax_right, *, left_color: str, right_color: str) -> None:
    ax_left.tick_params(axis="y", labelcolor=left_color)
    ax_right.tick_params(axis="y", labelcolor=right_color)
    ax_left.spines["left"].set_color(left_color)
    ax_right.spines["right"].set_color(right_color)
    ax_left.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)


def plot_information_dynamics(
    result: DynamicsResult,
    params: SimplexParams,
    output_dir: Path,
    *,
    f2_enabled: bool,
    psi_star_ok: bool,
    show: bool = False,
) -> dict[str, Path]:
    """
    Save three figures:

    1. Values vs t:  D_KL(psi||phi) [left, blue] and F_N [right, orange]
    2. Rates vs t:   d(KL)/dt [left, blue] and dF_N/dt [right, orange]
       (discrete derivative between consecutive sampled times)
    3. Mean LD vs t
    """
    if not HAS_MPL:
        raise ImportError("Plotting requires matplotlib")

    _setup_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    param_lines = _plot_param_title_lines(params)
    param_slug = _selection_param_slug(params)
    paths: dict[str, Path] = {}
    show_f2 = f2_enabled and psi_star_ok and result.f2_bits is not None

    times = np.asarray(result.times, dtype=float)
    kl = result.kl_phi_bits
    f1 = result.f1
    left_color = "#1f77b4"
    right_color = "#ff7f0e"

    # --- Plot 1: accumulated quantities vs t ---
    fig, ax_kl = plt.subplots(figsize=(9.5, 5.5))
    ax_f1 = ax_kl.twinx()
    line_kl, = ax_kl.plot(
        times, kl, "o-", color=left_color, lw=2.0, ms=5,
        label=r"$D_{\mathrm{KL}}(\psi(t)\,\|\,\varphi)$ [bits]",
        zorder=3,
    )
    line_f1, = ax_f1.plot(
        times, f1, "s-", color=right_color, lw=2.0, ms=5,
        label=r"$\mathcal{F}_N(\psi(t))$",
        zorder=3,
    )
    ax_kl.set_xlabel(r"generation $t$")
    ax_kl.set_ylabel(r"information $D_{\mathrm{KL}}(\psi\,\|\,\varphi)$ [bits]", color=left_color)
    ax_f1.set_ylabel(r"FREE FITNESS $\mathcal{F}_N$", color=right_color)
    title1 = (
        r"Information and free fitness vs. time" + "\n"
        r"(neutral start: $\psi(0)=\varphi$)" + "\n"
        + "\n".join(param_lines)
    )
    ax_kl.set_title(title1, pad=14)
    _style_twin_axes(ax_kl, ax_f1, left_color=left_color, right_color=right_color)
    lines = [line_kl, line_f1]
    labels = [ln.get_label() for ln in lines]
    if show_f2 and result.f2_bits is not None:
        line_f2, = ax_kl.plot(
            times, result.f2_bits, "^--", color="#2ca02c", lw=1.6, ms=4,
            label=r"$F_2=D_{\mathrm{KL}}(\psi\,\|\,\psi^*)$ [bits]",
            zorder=2,
        )
        lines.append(line_f2)
        labels.append(line_f2.get_label())
    ax_kl.legend(lines, labels, loc="upper left", framealpha=0.92)
    fig.tight_layout()
    paths["values"] = output_dir / f"information_and_f1_vs_t_{param_slug}.png"
    fig.savefig(paths["values"], dpi=160)
    if show:
        plt.show()
    plt.close(fig)

    # --- Plot 2: discrete rates (derivatives) vs t ---
    fig, ax_dkl = plt.subplots(figsize=(9.5, 5.5))
    ax_df1 = ax_dkl.twinx()
    rt = result.rate_times
    line_dkl, = ax_dkl.plot(
        rt, result.rate_kl_bits, "o-", color=left_color, lw=2.0, ms=5,
        zorder=3,
    )
    ax_df1.plot(
        rt, result.rate_f1, "s-", color=right_color, lw=2.0, ms=5,
        zorder=3,
    )
    ax_dkl.axhline(0.0, color="0.45", lw=0.8, zorder=1)
    ax_dkl.set_xlabel(r"generation $t$ (end of interval)")
    ax_dkl.set_ylabel(
        r"$\Delta D_{\mathrm{KL}}/\Delta t$ [bits per generation]", color=left_color,
    )
    ax_df1.set_ylabel(r"FREE FITNESS $\Delta\mathcal{F}_N/\Delta t$", color=right_color)
    title2 = (
        r"Rates of change (discrete derivative between sampled times)" + "\n"
        + "\n".join(param_lines)
    )
    ax_dkl.set_title(title2, pad=14)
    _style_twin_axes(ax_dkl, ax_df1, left_color=left_color, right_color=right_color)
    if show_f2 and result.rate_f2_bits is not None:
        ax_df1.plot(
            rt, result.rate_f2_bits, "^--", color="#2ca02c", lw=1.6, ms=4,
            zorder=2,
        )
    fig.tight_layout()
    paths["rates"] = output_dir / f"information_and_f1_rates_vs_t_{param_slug}.png"
    fig.savefig(paths["rates"], dpi=160)
    if show:
        plt.show()
    plt.close(fig)

    # --- Plot 3: mean LD ---
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.plot(times, result.mean_ld, "o-", color="#9467bd", lw=2.0, ms=5)
    ax.set_xlabel(r"generation $t$")
    ax.set_ylabel(r"$\mathbb{E}_\rho[D],\ D=x_{00}x_{11}-x_{01}x_{10}$")
    ax.set_title(
        r"Mean linkage disequilibrium vs. time" + "\n" + "\n".join(param_lines),
        pad=14,
    )
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
    fig.tight_layout()
    paths["mean_ld"] = output_dir / f"mean_ld_vs_t_{param_slug}.png"
    fig.savefig(paths["mean_ld"], dpi=160)
    if show:
        plt.show()
    plt.close(fig)

    return paths


def print_top_k_ld_table(
    snapshots: list[tuple[int, np.ndarray, np.ndarray, float, float]],
    t: int,
    top_k: int,
    N: int,
) -> None:
    """Print LD at the top-k highest-mass simplex lattice points at time t."""
    print(f"  t = {t}  top {top_k} simplex states by mass  (count vectors summing to N={N})")
    print("    rank  counts (00,01,10,11)     freqs (00,01,10,11)       mass        D=LD")
    for rank, c, f, mass, d_val in snapshots:
        print(
            f"  {rank:2d}. ({c[0]:3d},{c[1]:3d},{c[2]:3d},{c[3]:3d})   "
            f"({f[0]:.4f},{f[1]:.4f},{f[2]:.4f},{f[3]:.4f})   {mass:.8f}   {d_val:+.6e}"
        )
    print()


def print_dynamics_summary(
    result: DynamicsResult,
    params: SimplexParams,
    *,
    f2_enabled: bool,
    psi_star_ok: bool,
    psi_star_residual: float | None,
    plot_paths: dict[str, Path] | None = None,
) -> None:
    """Print KL, F_1, mean LD, optional F_2, and top-k LD snapshots."""
    print("Dynamics from psi(0) = phi  (neutral start)")
    print("  F_1(rho) = <W>_rho + H(rho)/(2N)  [always computed when times is set]")
    print("    W(x) = log E_x[e^{g^T Sigma g}] - (1/2N) sum_i (x@M)_i log x_i")
    print("  mean LD = E_rho[x_00 x_11 - x_01 x_10]  [always computed when times is set]")
    print()

    if f2_enabled:
        if psi_star_ok:
            print(f"  F_2: psi* FOUND  (||P psi* - psi*||_1 = {psi_star_residual:.3e})")
            print("       F_2(t) = D_KL(psi(t) || psi*) [bits]")
        else:
            print("  F_2: psi* NOT CONFIRMED — F_2 skipped")
            if psi_star_residual is not None and np.isfinite(psi_star_residual):
                print(f"       residual ||P psi* - psi*||_1 = {psi_star_residual:.3e}")
    else:
        print("  F_2: disabled  (set compute_f2: true in config or use --f2)")
    print()

    header = "  t        D_KL(psi||phi)   F_1          mean LD"
    if f2_enabled and psi_star_ok and result.f2_bits is not None:
        header += "      F_2=KL(psi||psi*)"
    print(header)
    print("           [bits]           [nats-scale]")
    for i, t in enumerate(result.times):
        line = (
            f"  {int(t):6d}   {result.kl_phi_bits[i]:14.8f}   "
            f"{result.f1[i]:12.8f}   {result.mean_ld[i]:12.8e}"
        )
        if f2_enabled and psi_star_ok and result.f2_bits is not None:
            line += f"   {result.f2_bits[i]:14.8f}"
        print(line)

    if len(result.times) >= 2:
        print()
        print("  Interval rates (per generation):")
        hdr = "  t_end    dKL/dt       dF1/dt"
        if f2_enabled and psi_star_ok and result.rate_f2_bits is not None:
            hdr += "       dF2/dt"
        print(hdr)
        for i, t_end in enumerate(result.rate_times):
            row = (
                f"  {int(t_end):6d}   {result.rate_kl_bits[i]:+.8e}   "
                f"{result.rate_f1[i]:+.8e}"
            )
            if f2_enabled and psi_star_ok and result.rate_f2_bits is not None:
                row += f"   {result.rate_f2_bits[i]:+.8e}"
            print(row)

    if result.top_k_ld_snapshots:
        print()
        print(
            f"  Top-{params.top_k} LD at highest-mass lattice points "
            f"(ld_top_k_times={params.ld_top_k_times!r}):"
        )
        print("  Each row is a discrete simplex coordinate (count vector / N = freqs).")
        print()
        for t in result.top_k_ld_times:
            print_top_k_ld_table(result.top_k_ld_snapshots[t], t, params.top_k, params.N)

    if plot_paths:
        print()
        for key, path in plot_paths.items():
            print(f"  plot [{key}]: {path}")
    print()


def evolve(P_col: np.ndarray, psi0: np.ndarray, T: int) -> np.ndarray:
    """Evolve psi forward T generations from psi(0): psi(t) = P_col^t @ psi(0)."""
    psi = np.asarray(psi0, dtype=float).copy()
    for _ in range(int(T)):
        psi = P_col @ psi
    s = float(psi.sum())
    if s > 0.0:
        psi /= s
    return psi


# =============================================================================
# Screen output helpers
# =============================================================================


def print_mutation_rates(u: np.ndarray, v: np.ndarray) -> None:
    """Print u and v vectors with mutation-direction labels."""
    print(f"u = {u}     mutation  0 --> 1  (per locus)")
    print(f"v = {v}     mutation  1 --> 0  (per locus)")
    print()


def print_sigma(Sigma: np.ndarray) -> None:
    print("Sigma (epistasis matrix):")
    for row in Sigma:
        print("  " + "  ".join(f"{x:10.6g}" for x in row))
    print()


def print_fitness_table(Sigma: np.ndarray, omega: np.ndarray) -> None:
    """Print omega(g) as a 2×2 allele grid plus flat list."""
    grid = omega_to_fitness_grid(omega)
    print("Fitness  omega(g) = exp(g^T Sigma g)")
    print()
    print("  2×2 layout (rows = locus-1 allele, cols = locus-2 allele):")
    print()
    print("              locus 2")
    print("              allele 0       allele 1")
    print(f"  locus 1  0    {grid[0, 0]:12.6g}    {grid[0, 1]:12.6g}     genotypes 00, 01")
    print(f"           1    {grid[1, 0]:12.6g}    {grid[1, 1]:12.6g}     genotypes 10, 11")
    print()
    print("  Flat list:")
    print("    genotype     fitness")
    for lab, w in zip(GENOTYPE_LABELS, omega):
        print(f"    {lab:>4}       {w:12.6g}")
    print()


def print_mutation_matrix_formulas() -> None:
    """Print symbolic mutation matrix (same layout as comment block in code)."""
    formulas = [
        [_mutation_entry_formula(i, j) for j in range(4)]
        for i in range(4)
    ]
    col_widths = [
        max(len(GENOTYPE_LABELS[j]), max(len(formulas[i][j]) for i in range(4)))
        for j in range(4)
    ]

    print("Mutation matrix M — symbolic (rows = source, cols = target genotype)")
    print("Genotype order: " + ", ".join(GENOTYPE_LABELS))
    print("Per locus i:  P(0→0)=1-u_i,  P(0→1)=u_i,  P(1→0)=v_i,  P(1→1)=1-v_i")
    print()
    header = "from\\to  " + "  ".join(
        f"{lab:>{col_widths[j]}}" for j, lab in enumerate(GENOTYPE_LABELS)
    )
    print(header)
    for i, lab in enumerate(GENOTYPE_LABELS):
        row = "  ".join(f"{formulas[i][j]:>{col_widths[j]}}" for j in range(4))
        print(f"  {lab}    {row}")
    print()


def print_mutation_matrix_numeric(M: np.ndarray) -> None:
    """Print numeric mutation matrix at the configured u, v values."""
    print("Mutation matrix M — numeric (at configured u, v)")
    print("Genotype order: " + ", ".join(GENOTYPE_LABELS))
    header = "from\\to  " + "  ".join(f"{lab:>10}" for lab in GENOTYPE_LABELS)
    print(header)
    for i, lab in enumerate(GENOTYPE_LABELS):
        row = "  ".join(f"{M[i, j]:10.6f}" for j in range(4))
        print(f"  {lab}    {row}")
    print(f"Row sums: {M.sum(axis=1)}")
    print()


def print_phi_distribution(
    phi: np.ndarray,
    fracs: np.ndarray,
    counts: np.ndarray,
    title: str,
    top_k: int = 20,
    d_per_state: np.ndarray | None = None,
) -> None:
    print(title)
    print(f"  number of states S = {len(phi)}")
    print(f"  sum(phi) = {phi.sum():.12f}")
    print(f"  min(phi) = {phi.min():.6e},  max(phi) = {phi.max():.6e}")
    print(f"  entropy (nats) = {-float(np.sum(phi * np.log(phi + 1e-300))):.6f}")
    if d_per_state is not None:
        print(f"  mean LD = {float(np.sum(phi * d_per_state)):.6e}")
    order = np.argsort(-phi)
    print(f"\n  Top {top_k} states by probability:")
    if d_per_state is not None:
        print("    counts (00,01,10,11)     freqs (00,01,10,11)       phi          D=LD")
    else:
        print("    counts (00,01,10,11)     freqs (00,01,10,11)       phi")
    for rank, idx in enumerate(order[:top_k], start=1):
        c = counts[idx]
        f = fracs[idx]
        line = (
            f"  {rank:2d}. ({c[0]:3d},{c[1]:3d},{c[2]:3d},{c[3]:3d})   "
            f"({f[0]:.4f},{f[1]:.4f},{f[2]:.4f},{f[3]:.4f})   {phi[idx]:.8f}"
        )
        if d_per_state is not None:
            line += f"   {d_per_state[idx]:+.6e}"
        print(line)
    print()


def compute_phi(
    params: SimplexParams,
    M_mut: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, str]:
    """
    Load or compute neutral equilibrium phi (Sigma = 0 stationary distribution).

    Returns (phi, P_col_neutral, fracs, counts, source).
    P_col_neutral is None when phi was loaded from cache.
    """
    fracs = make_simplex_grid(params.N)
    counts = make_simplex_counts(params.N)

    cache_path = params.neutral_cache_file
    if cache_path is not None:
        phi_cached = load_neutral_phi(cache_path, params.N, params.u, params.v, params.r)
        if phi_cached is not None:
            if phi_cached.shape[0] != counts.shape[0]:
                raise ValueError(
                    f"Cache length {phi_cached.shape[0]} != S={counts.shape[0]} for N={params.N}"
                )
            print(f"  phi loaded from cache: {cache_path}  (N={params.N})", flush=True)
            return phi_cached, None, fracs, counts, "cache"

    print("  No matching cache — computing phi (Sigma=0 transition matrix) ...", flush=True)
    omega_neutral = fitness_exp(np.zeros((2, 2)))
    T_row_neutral, _, _ = build_transition_matrix_row_stochastic(
        params.N,
        omega_neutral,
        M_mut,
        params.r,
        n_jobs=params.n_jobs,
        label="neutral",
        verbose=True,
    )
    P_col_neutral = column_stochastic(T_row_neutral)
    phi = stationary_from_matrix(P_col_neutral)

    if cache_path is not None:
        save_neutral_phi(cache_path, params.N, phi, params.u, params.v, params.r)
        print(f"  phi saved to cache: {cache_path}  (N={params.N})", flush=True)

    return phi, P_col_neutral, fracs, counts, "computed"


# =============================================================================
# Main run
# =============================================================================


def run(params: SimplexParams, config_path: Path | None = None) -> None:
    Sigma_dyn = np.zeros((2, 2)) if params.neutral else params.Sigma.copy()
    u, v = params.u, params.v
    M_mut = mutation_matrix_asymmetric(u, v)
    omega = fitness_exp(Sigma_dyn)

    print("=" * 72)
    print("2loci/simplex.py")
    if config_path is not None:
        print(f"config: {config_path}")
    print("=" * 72)
    print(f"N = {params.N},  r = {params.r}")
    print()
    print_mutation_rates(u, v)
    print_sigma(Sigma_dyn)
    print_fitness_table(Sigma_dyn, omega)

    # Neutral equilibrium phi (Sigma = 0); always printed (from cache or computed)
    print("-" * 72)
    print("phi — neutral equilibrium (Sigma = 0, stationary distribution)")
    print("-" * 72)
    phi, P_col_neutral, fracs, counts, phi_source = compute_phi(params, M_mut)
    d_per_state = linkage_disequilibrium_per_state(fracs)
    if P_col_neutral is not None:
        residual = float(np.linalg.norm(P_col_neutral @ phi - phi, ord=1))
        print(f"  ||P phi - phi||_1 = {residual:.3e}")
    print_phi_distribution(
        phi, fracs, counts,
        f"phi  ({phi_source}):",
        top_k=params.top_k,
        d_per_state=d_per_state,
    )

    print("-" * 72)
    print_mutation_matrix_formulas()
    print_mutation_matrix_numeric(M_mut)

    if params.neutral and P_col_neutral is not None:
        P_col = P_col_neutral
        print("Dynamics use Sigma = 0 (neutral flag set); reusing neutral matrix.")
    elif params.neutral:
        print("Dynamics use Sigma = 0; phi from cache — building neutral matrix for psi(t) ...")
        omega_neutral = fitness_exp(np.zeros((2, 2)))
        T_row, _, _ = build_transition_matrix_row_stochastic(
            params.N, omega_neutral, M_mut, params.r,
            n_jobs=params.n_jobs, label="neutral-dynamics", verbose=True,
        )
        P_col = column_stochastic(T_row)
    else:
        print("-" * 72)
        T_row, _, _ = build_transition_matrix_row_stochastic(
            params.N, omega, M_mut, params.r,
            n_jobs=params.n_jobs,
            label="selection",
            verbose=True,
        )
        P_col = column_stochastic(T_row)
        row_sums = T_row.sum(axis=1)
        col_sums = P_col.sum(axis=0)
        print(f"  max |row_sum - 1| = {np.max(np.abs(row_sums - 1.0)):.3e}")
        print(f"  max |col_sum - 1| = {np.max(np.abs(col_sums - 1.0)):.3e}")
        print()

    if params.times:
        print("-" * 72)
        print("Dynamics  (KL, F_1, mean LD always; F_2 optional)")
        print("-" * 72)
        w_per_state = w_potential_per_state(fracs, omega, M_mut, params.N)

        psi_star: np.ndarray | None = None
        psi_star_ok = False
        psi_star_residual: float | None = None
        if params.compute_f2 and not params.neutral:
            psi_star, psi_star_residual, psi_star_ok = try_selection_equilibrium(P_col)
            if psi_star_ok:
                print(
                    f"  psi* (selection equilibrium): converged, "
                    f"||P psi* - psi*||_1 = {psi_star_residual:.3e}",
                    flush=True,
                )
            else:
                print(
                    "  psi* (selection equilibrium): NOT confirmed — F_2 will not be reported",
                    flush=True,
                )
                if psi_star_residual is not None and np.isfinite(psi_star_residual):
                    print(f"    residual ||P psi* - psi*||_1 = {psi_star_residual:.3e}", flush=True)
        elif params.compute_f2 and params.neutral:
            print("  compute_f2 set but neutral: true — F_2 skipped", flush=True)

        result = dynamics_series(
            P_col,
            phi,
            params.times,
            w_per_state,
            d_per_state,
            counts,
            fracs,
            params.N,
            params.top_k,
            params.ld_top_k_times,
            psi_star if (params.compute_f2 and psi_star_ok) else None,
        )

        P_neu_for_debug = P_col_neutral
        if params.debug_kl and P_neu_for_debug is None and not params.neutral:
            omega_n = fitness_exp(np.zeros((2, 2)))
            T_n, _, _ = build_transition_matrix_row_stochastic(
                params.N, omega_n, M_mut, params.r,
                n_jobs=params.n_jobs, label="neutral-debug", verbose=False,
            )
            P_neu_for_debug = column_stochastic(T_n)

        if params.debug_kl:
            print_kl_debug_sanity(
                omega=omega,
                M_mut=M_mut,
                r=params.r,
                P_col=P_col,
                phi=phi,
                fracs=fracs,
                result=result,
                P_col_neutral=P_neu_for_debug,
            )
        plot_paths = None
        if params.plot_dynamics and HAS_MPL:
            plot_paths = plot_information_dynamics(
                result,
                params,
                params.plot_dir,
                f2_enabled=params.compute_f2,
                psi_star_ok=psi_star_ok,
                show=params.show_plots,
            )
        elif params.plot_dynamics:
            print("  (matplotlib not installed — skipping plots)", flush=True)
        print_dynamics_summary(
            result,
            params,
            f2_enabled=params.compute_f2,
            psi_star_ok=psi_star_ok,
            psi_star_residual=psi_star_residual,
            plot_paths=plot_paths,
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-locus WF Markov chain on genotype simplex")
    p.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="YAML/JSON config (default: simplex_config.yaml next to this script)",
    )
    p.add_argument(
        "--f2",
        "--compute-f2",
        dest="compute_f2",
        action="store_true",
        default=None,
        help="Compute F_2 = KL(psi(t)||psi*) and include in plots (overrides config)",
    )
    p.add_argument(
        "--no-f2",
        action="store_true",
        help="Skip F_2 even if enabled in config",
    )
    p.add_argument(
        "--debug-kl",
        action="store_true",
        default=None,
        help="Print KL sanity checks (omega, P vs neutral, KL after 1 gen)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)
    cfg["_config_dir"] = str(config_path.parent)
    if args.no_f2:
        cfg["compute_f2"] = False
    elif args.compute_f2:
        cfg["compute_f2"] = True
    if args.debug_kl:
        cfg["debug_kl"] = True
    params = params_from_config(cfg, config_dir=config_path.parent)
    run(params, config_path=config_path)


if __name__ == "__main__":
    main()
