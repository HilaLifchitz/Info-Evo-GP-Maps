"""
equilibria_allele_freq.py — Equilibrium on the allele-frequency cube [0,1]^n

Stationary density (reversible Wright–Fisher–type diffusion with diagonal noise
B_ii = x_i(1-x_i) in detailed balance):

    ρ(x) ∝ exp(-V(x)),
    V(x) = -2N log W(x) + Σ_i (1 - 4N u_i) log x_i + Σ_i (1 - 4N v_i) log(1 - x_i),

so (up to an additive constant)

    ℓ(x) = -V(x) = 2N log W(x) + Σ_i (4N u_i - 1) log x_i + Σ_i (4N v_i - 1) log(1 - x_i).

    (Verify u/v ↔ boundary terms against your diffusion notation; Kimura often pairs
    v with log x and u with log(1−x) when u is 0→1 and v is 1→0.)

Mean fitness at allele frequencies x:

    W(x) = E[ ω(g) | x ],

with ω(g) = exp(g^T M g) (haploid, g ∈ {0,1}^n) or ω(g) = exp(φ(g)^T M φ(g)) (diploid),
independence across loci for P(g|x) (haploid Bernoulli; diploid Hardy–Weinberg).

Matrix M: M_ii = s_i, M_ij = ε_ij/2 for i≠j (same convention as equilibria_explored_*).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# numerics
# ---------------------------------------------------------------------------

#standard numerical stabilization for log(sum_i(exp(a_i)))
def logsumexp1d(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    m = float(np.max(a))
    if not np.isfinite(m):
        return m
    return m + math.log(float(np.sum(np.exp(a - m))))


# ---------------------------------------------------------------------------
# Genotypes and fitness
# ---------------------------------------------------------------------------

#builds all binary genotypes for haploid case
def all_genotypes_haploid(n: int) -> np.ndarray:
    """All g ∈ {0,1}^n as (2^n, n)."""
    indices = np.arange(2**n, dtype=np.uint64)[:, None]
    bit_positions = np.arange(n - 1, -1, -1, dtype=np.uint64)
    return ((indices >> bit_positions) & 1).astype(np.int8)

#builds all ternary genotypes for diploid case
def all_genotypes_diploid(n: int) -> np.ndarray:
    """All g ∈ {0,1,2}^n as (3^n, n). Base-3 enumeration."""
    k = 3**n
    G = np.zeros((k, n), dtype=np.int8)
    for idx in range(k):
        t = idx
        for j in range(n):
            G[idx, n - 1 - j] = t % 3
            t //= 3
    return G


def log_fitness_haploid(G: np.ndarray, M: np.ndarray) -> np.ndarray:
    """log ω(g) = g^T M g (column vector g)."""
    return np.einsum("bi,ij,bj->b", G.astype(float), M, G.astype(float))


def phi_diploid(G: np.ndarray, h: np.ndarray) -> np.ndarray:
    """φ_i(0)=0, φ_i(1)=h_i, φ_i(2)=1."""
    n = G.shape[1]
    h = np.asarray(h, dtype=float).reshape(1, n)
    out = np.zeros_like(G, dtype=float)
    for i in range(n):
        gi = G[:, i]
        out[:, i] = np.where(gi == 0, 0.0, np.where(gi == 1, h[0, i], 1.0))
    return out


def log_fitness_diploid(G: np.ndarray, M: np.ndarray, h: np.ndarray) -> np.ndarray:
    P = phi_diploid(G, h)
    return np.einsum("bi,ij,bj->b", P, M, P)


# ---------------------------------------------------------------------------
# Genotype probabilities given allele frequencies x
# ---------------------------------------------------------------------------
#
# For a point x = (x_1, ..., x_n) in the allele-frequency cube [0,1]^n,
# we define a probability law over full genotypes g across loci.
#
#     W(x) = E[ω(g) | x] = Σ_g ω(g) P(g | x).
#
# Haploid case
# ------------
# Here g_i ∈ {0,1}, and conditional on x we assume loci are independent with
#
#     P(g_i = 1 | x) = x_i,
#     P(g_i = 0 | x) = 1 - x_i.
#
# Therefore the full genotype probability factorizes as
#
#     P(g | x) = ∏_i x_i^{g_i} (1 - x_i)^{1 - g_i},
#
# and so
#
#     log P(g | x)
#       = Σ_i [ g_i log x_i + (1 - g_i) log(1 - x_i) ].
#
# The code rewrites this in a vectorized form:
#
#     log P(g | x)
#       = Σ_i g_i [log x_i - log(1 - x_i)] + Σ_i log(1 - x_i).
#
# Thus, if
#     Lx  = log X,
#     L1x = log(1 - X),
#     A   = Lx - L1x,
# then for a batch of allele-frequency vectors X and all genotypes G,
#
#     logp = A @ G.T + Σ_i log(1 - x_i).
#
# This gives a matrix of shape (batch_size, number_of_genotypes), where each
# entry is log P(g | x) for one sampled x and one genotype g.
#
# Diploid case
# ------------
# Here g_i ∈ {0,1,2}, interpreted as genotype dosage at locus i.
# Conditional on allele frequency x_i, we assume Hardy–Weinberg proportions:
#
#     P(g_i = 0 | x) = (1 - x_i)^2,
#     P(g_i = 1 | x) = 2 x_i (1 - x_i),
#     P(g_i = 2 | x) = x_i^2.
#
# Again loci are assumed independent conditional on x, so
#
#     P(g | x) = ∏_i P(g_i | x_i),
#
# and the log-probability is the sum over loci of the corresponding
# one-locus log-probabilities.
#
# Important modeling assumption
# -----------------------------
# In both the haploid and diploid versions, the genotype law P(g | x) is taken
# to be a product across loci once x is given. This is a strong assumption:
# it means there is no LD / cross-locus dependence in P(g | x) beyond what is
# encoded indirectly through the fitness function and the resulting equilibrium
# weighting. Computationally, this product structure is exactly what allows us
# to define and evaluate
#
#     W(x) = Σ_g ω(g) P(g | x)
#
# in a clean and explicit way.
def log_prob_haploid_batch(G: np.ndarray, X: np.ndarray, x_eps: float) -> np.ndarray:
    """
    X: (B, n) allele frequencies.
    (B = number of (x_1,...x_n) vectors - number of samples (batch size))
    X=[(x_1^1,x_2^1,...,x_n^1),
      (x_1^2,x_2^2,...,x_n^2),
      ...,
      (x_1^B,x_2^B,...,x_n^B)]
    (x_i^j = allele frequency of allele 1 at locus i for sample j)

    G = genotype matrix
    G=[(g_1^1,g_2^1,...,g_n^1),
      (g_1^2,g_2^2,...,g_n^2),
      ...,
      (g_1^k,g_2^k,...,g_n^k)]
      where k = number of genotypes=2^n for haploid case

     Returns log P(g|x) for each row, shape (B, 2^n).
    """
    X = np.clip(X, x_eps, 1.0 - x_eps)
    Lx = np.log(X)
    L1x = np.log(1.0 - X)
    A = Lx - L1x
    # (B, k) = (B, n) @ (n, k)
    logp = A @ G.astype(float).T + np.sum(L1x, axis=1, keepdims=True)
    return logp


def log_prob_diploid_batch(G: np.ndarray, X: np.ndarray, x_eps: float) -> np.ndarray:
    """
    Hardy-Weinberg: x_i = frequency of allele 1; P(0)=(1-x)^2, P(1)=2x(1-x), P(2)=x^2.
    Returns log P(g|x), shape (B, 3^n).
    """
    B, nloc = X.shape
    k = G.shape[0] #number of genotypes=3^n for diploid case
    X = np.clip(X, x_eps, 1.0 - x_eps) #clip allele frequencies to (x_eps, 1-x_eps) -- prevents log(0)
    lp = np.zeros((B, k), dtype=float) #initialize log probability matrix
    #loop over loci 
    for i in range(nloc):
        gc = G[None, :, i] #columns of G corresponding to locus i
        p = X[:, None, i] #columns of X corresponding to locus i
        lp += np.where( # --> log P(g|x)= sum_i P(log(g_i|x_i))
            gc == 0,#-- calculates P(g_i|x_i) for each locus i, where gc == 0, 2.0 * np.log(1.0 - p)
            2.0 * np.log(1.0 - p), #P(0)=(1-x)^2
            np.where(
                gc == 1,
                math.log(2.0) + np.log(p) + np.log(1.0 - p), #P(1)=2x(1-x)
                2.0 * np.log(p), #P(2)=x^2
            ),
        )
    return lp

# the real deal- comptues W(x) = E[ω(g) | x] = Σ_g ω(g) P(g | x)
def log_mean_fitness_batch(
    G: np.ndarray,
    log_omega: np.ndarray,
    X: np.ndarray,
    ploidy: str,
    x_eps: float,
) -> np.ndarray:
    """log W(x) for each row of X, shape (B,)."""
    if ploidy == "haploid":
        logp = log_prob_haploid_batch(G, X, x_eps)
    else:
        logp = log_prob_diploid_batch(G, X, x_eps)
    # log W = log sum_g exp(log ω_g + log p_g)
    B = X.shape[0] # --> number of x vectors (batch size)
    logW = np.empty(B, dtype=float)
    for b in range(B):
        logW[b] = logsumexp1d(log_omega + logp[b])
    return logW

# equilibria
def ell_batch(
    X: np.ndarray,
    G: np.ndarray,
    log_omega: np.ndarray,
    N_pop: float,
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    ploidy: str,
    x_eps: float,
) -> np.ndarray:
    """
    Unnormalized log density (up to constant):

        ℓ(x) = 2N log W(x) + Σ_i (4N u_i - 1) log x_i + Σ_i (4N v_i - 1) log(1 - x_i).

    Convention note: Kimura's mutation-only stationary often pairs u (0→1) with (1−x)
    and v (1→0) with x; if your u,v labels differ, swap the two sums accordingly.
    """
    Xs = np.clip(X, x_eps, 1.0 - x_eps)
    logW = log_mean_fitness_batch(G, log_omega, X, ploidy, x_eps)
    term_x = np.sum((4.0 * N_pop * u_vec - 1.0) * np.log(Xs), axis=1)
    term_1mx = np.sum((4.0 * N_pop * v_vec - 1.0) * np.log(1.0 - Xs), axis=1)
    return 2.0 * N_pop * logW + term_x + term_1mx


def V_batch(
    X: np.ndarray,
    G: np.ndarray,
    log_omega: np.ndarray,
    N_pop: float,
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    ploidy: str,
    x_eps: float,
) -> np.ndarray:
    """Potential V = -ℓ + const along the same kernel."""
    return -ell_batch(X, G, log_omega, N_pop, u_vec, v_vec, ploidy, x_eps)


# ---------------------------------------------------------------------------
# Build M (same as Σ elsewhere)
# ---------------------------------------------------------------------------

# enforces that M is symmetric, even if the user supplied a matrix that is slightly asymmetric
def matrix_M_from_sigma(Sigma: np.ndarray) -> np.ndarray:
    Sigma = np.asarray(Sigma, dtype=float)
    Sigma = (Sigma + Sigma.T) / 2.0
    return Sigma


# ---------------------------------------------------------------------------
# Exploration: Monte Carlo importance sample on uniform grid in cube
# ---------------------------------------------------------------------------


def explore_importance_uniform(
    G: np.ndarray,
    log_omega: np.ndarray,
    N_pop: float,
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    ploidy: str,
    n_samples: int,
    rng: np.random.Generator,
    x_eps: float,
) -> dict[str, Any]:
    """
    Draw x ~ Uniform(x_eps, 1-x_eps)^n, compute unnormalized weights w ∝ exp(ℓ(x)).
    Self-normalized estimates of E[x], Cov[x], entropy of discrete approximation.
    """
    n = G.shape[1]
    X = rng.uniform(x_eps, 1.0 - x_eps, size=(n_samples, n))
    ell = ell_batch(X, G, log_omega, N_pop, u_vec, v_vec, ploidy, x_eps)
    m = float(np.max(ell))
    w = np.exp(ell - m)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0:
        raise RuntimeError("All weights zero or invalid; check N, M, and mutation parameters.")
    p = w / s

    Ex = np.sum(p[:, None] * X, axis=0)
    Xc = X - Ex
    Cov = (p[:, None, None] * (Xc[:, :, None] * Xc[:, None, :])).sum(axis=0)

    H = -float(np.sum(p * np.log(p + 1e-300)))
    ess = float(1.0 / np.sum(p**2))

    imode = int(np.argmax(ell))
    x_mode = X[imode].copy()

    return {
        "X": X,
        "ell": ell,
        "weights": p,
        "E_x": Ex,
        "Cov_x": Cov,
        "entropy_weights": H,
        "ess": ess,
        "x_mode_argmax_ell": x_mode,
        "log_weight_scale": m,
    }


def explore_grid_small(
    G: np.ndarray,
    log_omega: np.ndarray,
    N_pop: float,
    u_vec: np.ndarray,
    v_vec: np.ndarray,
    ploidy: str,
    resolution: int,
    x_eps: float,
) -> dict[str, Any]:
    """Tensor grid on [x_eps,1-x_eps]^n (only for small n)."""
    n = G.shape[1]
    edges = np.linspace(x_eps, 1.0 - x_eps, resolution)
    grid = np.stack(np.meshgrid(*[edges] * n, indexing="ij"), axis=-1).reshape(-1, n)
    n_pts = grid.shape[0]
    ell = ell_batch(grid, G, log_omega, N_pop, u_vec, v_vec, ploidy, x_eps)
    m = float(np.max(ell))
    w = np.exp(ell - m)
    s = float(np.sum(w))
    p = w / s
    Ex = np.sum(p[:, None] * grid, axis=0)
    Xc = grid - Ex
    Cov = (p[:, None, None] * (Xc[:, :, None] * Xc[:, None, :])).sum(axis=0)
    H = -float(np.sum(p * np.log(p + 1e-300)))
    imode = int(np.argmax(ell))
    return {
        "grid": grid,
        "ell": ell,
        "weights": p,
        "E_x": Ex,
        "Cov_x": Cov,
        "entropy_weights": H,
        "ess": float(1.0 / np.sum(p**2)),
        "x_mode_argmax_ell": grid[imode].copy(),
        "log_weight_scale": m,
        "n_grid_points": n_pts,
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        if not HAS_YAML:
            raise ImportError("YAML config requires PyYAML: pip install pyyaml")
        return yaml.safe_load(text)
    return json.loads(text)


def _as_float_vec(v: Any, n: int, name: str) -> np.ndarray:
    if v is None:
        raise ValueError(f"Missing {name}")
    if np.isscalar(v):
        return np.full(n, float(v), dtype=float)
    a = np.asarray(v, dtype=float).ravel()
    if a.size != n:
        raise ValueError(f"{name}: expected length {n}, got {a.size}")
    return a


def apply_config_to_ns(args: argparse.Namespace, config: dict[str, Any]) -> None:
    for key in (
        "n_loci",
        "N",
        "ploidy",
        "u",
        "v",
        "alpha",
        "h",
        "seed",
        "out",
        "sigma_file",
    ):
        if key in config and config[key] is not None:
            setattr(args, key, config[key])

    if "sigma" in config and config["sigma"] is not None:
        args.sigma = np.array(config["sigma"], dtype=float)

    if "explore" in config and config["explore"]:
        ex = config["explore"]
        for k, attr in (
            ("method", "explore_method"),
            ("n_samples", "n_samples"),
            ("grid_resolution", "grid_resolution"),
            ("x_eps", "x_eps"),
        ):
            if k in ex and ex[k] is not None:
                setattr(args, attr, ex[k])


def parse_u_v_alpha(n: int, N_pop: float, u: Any, v: Any, alpha_cfg: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns u_vec, v_vec, alpha (length n)."""
    u_vec = _as_float_vec(u, n, "u")
    v_vec = _as_float_vec(v, n, "v")
    if alpha_cfg is not None:
        alpha = np.asarray(alpha_cfg, dtype=float).ravel()
        if alpha.size != n:
            raise ValueError(f"alpha: expected length {n}, got {alpha.size}")
    else:
        # Default: θ_i = 4 N u_i (same scaling spirit as genotype-simplex scripts)
        alpha = 4.0 * N_pop * u_vec
    return u_vec, v_vec, alpha


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def run_from_args(args: argparse.Namespace) -> None:
    n = int(args.n_loci)
    N_pop = float(args.N)
    ploidy = str(args.ploidy).lower()
    if ploidy not in ("haploid", "diploid"):
        raise ValueError("ploidy must be haploid or diploid")

    rng = np.random.default_rng(int(args.seed))

    # M matrix
    if getattr(args, "sigma", None) is not None:
        M = matrix_M_from_sigma(np.array(args.sigma, dtype=float))
    elif args.sigma_file:
        path = args.sigma_file
        if path.endswith(".npy"):
            M = np.load(path).astype(float)
        else:
            M = np.loadtxt(path).astype(float)
        M = matrix_M_from_sigma(M)
    else:
        raise ValueError("Provide sigma: in YAML or --sigma-file")

    if M.shape != (n, n):
        raise ValueError(f"M must be {n}×{n}, got {M.shape}")

    u_vec, v_vec, alpha = parse_u_v_alpha(n, N_pop, args.u, args.v, getattr(args, "alpha", None))

    if ploidy == "haploid":
        G = all_genotypes_haploid(n)
        log_omega = log_fitness_haploid(G, M)
    else:
        G = all_genotypes_diploid(n)
        h = _as_float_vec(getattr(args, "h", 0.5), n, "h")
        np.clip(h, 0.0, 1.0, out=h)
        log_omega = log_fitness_diploid(G, M, h)

    k = G.shape[0]
    x_eps = float(args.x_eps)

    print(f"n = {n} loci, ploidy = {ploidy}, k = {k} genotype states in sum for W(x)")
    print(f"N = {N_pop}")
    print(f"u (per locus): {u_vec}")
    print(f"v (per locus): {v_vec}")
    print(f"4N u (reference): {alpha}  (ℓ uses u,v on log x and log(1−x) separately)")
    print("M (s_i on diagonal, ε_ij/2 off-diagonal):")
    np.set_printoptions(precision=4, suppress=True)
    print(M)
    print()

    method = str(args.explore_method).lower()
    if method == "grid":
        res = explore_grid_small(
            G,
            log_omega,
            N_pop,
            u_vec,
            v_vec,
            ploidy,
            int(args.grid_resolution),
            x_eps,
        )
        print(f"Grid exploration: resolution {args.grid_resolution}^n = {res['n_grid_points']} points")
    else:
        res = explore_importance_uniform(
            G,
            log_omega,
            N_pop,
            u_vec,
            v_vec,
            ploidy,
            int(args.n_samples),
            rng,
            x_eps,
        )
        print(f"Monte Carlo exploration: {args.n_samples} uniform samples in cube")

    print("\n--- ρ(x) ∝ exp(-V(x)); importance weights ~ exp(ℓ(x)) on sampled points ---")
    print(f"E[x] (under self-normalized weights): {res['E_x']}")
    print(f"Std[x] (marginal, sqrt diag of Cov): {np.sqrt(np.clip(np.diag(res['Cov_x']), 0, np.inf))}")
    print(f"Entropy of discrete approximation H(p): {res['entropy_weights']:.6g} nats")
    print(f"ESS (1/Σ p_i^2): {res['ess']:.6g}")
    print(f"Argmax ℓ(x) in sample/grid (mode proxy): {res['x_mode_argmax_ell']}")

    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = res["Cov_x"][i, j]
            si = math.sqrt(max(res["Cov_x"][i, i], 1e-30))
            sj = math.sqrt(max(res["Cov_x"][j, j], 1e-30))
            corr[i, j] = d / (si * sj)
    print("Correlation Corr(x_i, x_j) under weights:")
    print(corr)

    if args.out:
        np.savez(
            args.out,
            M=M,
            alpha=alpha,
            u=u_vec,
            v=v_vec,
            N=N_pop,
            n=n,
            ploidy=ploidy,
            E_x=res["E_x"],
            Cov_x=res["Cov_x"],
            entropy=res["entropy_weights"],
            ess=res["ess"],
            x_mode=res["x_mode_argmax_ell"],
        )
        print(f"\nSaved summary to {args.out}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Equilibrium ρ ∝ exp(-V) on [0,1]^n; ℓ = 2N log W + Σ(4Nu_i−1)log x + Σ(4Nv_i−1)log(1−x)."
    )
    p.add_argument("--config", type=str, default="", help="YAML or JSON config path.")
    p.add_argument("--n-loci", type=int, default=2)
    p.add_argument("--N", type=float, default=50.0, help="Scaling N in V = -2N log W + …")
    p.add_argument("--ploidy", choices=["haploid", "diploid"], default="haploid")
    p.add_argument("--u", type=float, default=1e-4, help="Mutation 0→1 (scalar; use YAML for vector).")
    p.add_argument("--v", type=float, default=1e-4, help="Mutation 1→0 (scalar).")
    p.add_argument("--sigma-file", type=str, default="", help="n×n matrix .npy or .txt for M.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="", help="Optional .npz output.")
    p.add_argument("--explore-method", choices=["mc", "grid"], default="mc")
    p.add_argument("--n-samples", type=int, default=50_000)
    p.add_argument("--grid-resolution", type=int, default=15)
    p.add_argument("--x-eps", type=float, default=1e-8, dest="x_eps")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.sigma = None
    args.alpha = None
    args.h = 0.5

    if args.config:
        cfg = load_config(args.config)
        apply_config_to_ns(args, cfg)
        print(f"Config loaded from {args.config}\n")

    run_from_args(args)


if __name__ == "__main__":
    main()
