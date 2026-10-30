"""
Shared helpers: haploid stationary on a uniform tensor grid in [x_eps, 1-x_eps]^n.

Free fitness F_N = (1/N) log Z, KL(eq || neutral) in bits, and mean fitness
E_{p_eq}[W(x)] = sum_k p_k W(x_k) — same p_k as for KL (discrete equilibrium on the grid).
"""

from __future__ import annotations

import numpy as np

from equilibria_allele_freq import (
    all_genotypes_haploid,
    ell_batch,
    log_fitness_haploid,
    log_mean_fitness_batch,
    matrix_M_from_sigma,
)


def tensor_grid_centers_haploid(n: int, resolution: int, x_eps: float) -> tuple[np.ndarray, float]:
    """Uniform cell centers on cube; returns X shape (resolution^n, n), cell volume Δx^n."""
    edges = np.linspace(x_eps, 1.0 - x_eps, int(resolution) + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dx = float(edges[1] - edges[0])
    cell_vol = dx**n
    mesh = np.meshgrid(*[centers] * n, indexing="ij")
    X = np.stack([m.ravel() for m in mesh], axis=1)
    return X, cell_vol


def _normalize_density_from_logweights(logw: np.ndarray, cell_vol: float) -> np.ndarray:
    m = float(np.max(logw))
    w = np.exp(logw - m)
    z = float(np.sum(w) * cell_vol)
    return w / z


def _log_partition_sum_discrete(logw: np.ndarray, cell_vol: float) -> float:
    m = float(np.max(logw))
    w = np.exp(logw - m)
    return float(m + np.log(float(np.sum(w) * cell_vol)))


def free_fitness_and_kl_bits_haploid(
    M: np.ndarray,
    *,
    N_pop: float,
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    x_eps: float,
    resolution: int,
) -> tuple[float, float, float]:
    """
    Haploid, independence across loci; M is n×n symmetric, log ω(g)=g^T M g.

    Stationary masses p_eq ∝ exp(ℓ_k) Δ on uniform tensor grid (indexing_ij).
    Returns (F_N, KL bits, E_{p_eq}[W(x)]).
    """
    M = matrix_M_from_sigma(np.asarray(M, dtype=float))
    nloc = int(M.shape[0])
    if M.shape[1] != nloc:
        raise ValueError("M must be square")
    if u_vec.size != nloc or v_vec.size != nloc:
        raise ValueError("u_vec and v_vec must have length n")

    G = all_genotypes_haploid(nloc)
    log_omega_eq = log_fitness_haploid(G, M)
    log_omega_neutral = np.zeros_like(log_omega_eq)

    X, cell_vol = tensor_grid_centers_haploid(nloc, resolution, x_eps)

    ell_eq = ell_batch(X, G, log_omega_eq, N_pop, u_vec, v_vec, "haploid", x_eps)
    ell_neutral = ell_batch(X, G, log_omega_neutral, N_pop, u_vec, v_vec, "haploid", x_eps)

    rho_eq = _normalize_density_from_logweights(ell_eq, cell_vol)
    rho_neutral = _normalize_density_from_logweights(ell_neutral, cell_vol)

    p_eq = (rho_eq * cell_vol).astype(float)
    p_eq /= float(np.sum(p_eq))

    log_ratio = np.log(rho_eq + 1e-300) - np.log(rho_neutral + 1e-300)
    kl_bits = float(np.sum(p_eq * log_ratio) / np.log(2.0))

    log_z = _log_partition_sum_discrete(ell_eq, cell_vol)
    free_fitness = float(log_z / N_pop)

    entropy_disc = -float(np.sum(p_eq * np.log(p_eq + 1e-300)))
    free_from_decomp = float(np.sum(p_eq * (ell_eq.astype(float) / N_pop)) + (entropy_disc + np.log(cell_vol)) / N_pop)
    if not np.isclose(free_fitness, free_from_decomp, rtol=1e-10, atol=1e-9):
        raise RuntimeError(
            f"free fitness split-check failed: logZ/N={free_fitness!r} vs "
            f"E[ℓ]/N+(H+logΔ)/N={free_from_decomp!r}"
        )

    log_w = log_mean_fitness_batch(G, log_omega_eq, X, "haploid", x_eps)
    mwx = float(np.max(log_w))
    mean_w_eq = float(np.exp(mwx) * np.sum(p_eq * np.exp(log_w - mwx)))

    return free_fitness, kl_bits, mean_w_eq
