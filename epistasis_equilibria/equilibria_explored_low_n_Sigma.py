"""
equilibria_explored.py — Equilibrium distribution over the genotype simplex

Implements the stationary distribution
  ρ∞(x) ∝ W(x)^{2N} ∏_{i=1}^{2^n} x_i^{α_i - 1}

where:
  - x = (x_g) genotype frequencies on the (2^n - 1)-simplex
  - ω(g) = fitness of genotype g = 1 + g^T Σ g  (pairwise epistasis: Σ_ii = s_i, Σ_ij = ε_ij)
  - W(x) = mean fitness = ∑_g ω(g) x_g
  - α_i = mutation rate into genotype whose coordinate is i=  4N × μ_{→g_i},  μ_{→g} = u·(# of 1's) + v·(# of 0's)

State space: lattice points on simplex, coordinates in {0/N, 1/N, ..., N/N}, sum = 1.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import math

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False




########################## SMALL UTILITIES ###################################

# ---------------------------------------------------------------------------
# Simplex grid: lattice points with denominator N
# k = 2^n genotypes → (k-1)-simplex. Points x with sum=1, x_i ∈ {0,1/N,...,N/N}
# ---------------------------------------------------------------------------
"""
Comment: The number of lattice points on the simplex is equal to C(N + k - 1, k - 1).

We want all points x = (x_1, ..., x_k) such that:
    - x_i in {0/N, 1/N, ..., N/N}
    - sum_i x_i = 1

Equivalently, writing x_i = c_i / N, this is the same as counting all integer
tuples (c_1, ..., c_k) with:
    - c_i >= 0
    - c_1 + ... + c_k = N

So the question becomes: how many ways are there to distribute N identical
objects into k boxes, allowing empty boxes?

This is the classic "stars and bars" problem:
    - write N stars:      *****...
    - use k-1 bars to split them into k groups

For example, with k=4 and N=5:
    **|*||**
means
    (2,1,0,2)

Any nonnegative solution (c_1, ..., c_k) corresponds to exactly one arrangement
of N stars and k-1 bars, and vice versa.

So we are choosing positions for the k-1 bars among the total
    N + k - 1
slots.

Hence the number of lattice points is

    binomial(N + k - 1, k - 1)

equivalently binomial(N + k - 1, N).
-- this is the number of ectors that the code below returns.
"""

# Here k is the number of genotype coordinates (k = 2^n), so the simplex has dimension k - 1.
## small cases / plotting -- cause for large grid we dont wanna store all those points in memory.
def make_simplex_grid(k: int, N: int) -> np.ndarray:
    """
    Lattice points on the (k-1)-simplex with denominator N.
    Returns array of shape (n_points, k) where each row sums to 1.
    """
    def _gen(counts, remaining, dim):
        if dim == 1:
            yield counts + [remaining]
        else:
            for c in range(remaining + 1):
                yield from _gen(counts + [c], remaining - c, dim - 1)

    points = list(_gen([], N, k))
    return np.array(points, dtype=float) / N



# ---------------------------------------------------------------------------
# Simplex lattice generator: yields integer count vectors c with sum(c) = N
# ---------------------------------------------------------------------------

def simplex_count_generator(k: int, N: int):
    """
    Yield all integer vectors c = (c_1, ..., c_k) such that
        c_i >= 0 and sum_i c_i = N.

    These correspond to simplex points x = c / N.
    """
    def _gen(prefix, remaining, dim):
        if dim == 1:
            yield np.array(prefix + [remaining], dtype=np.int64)
        else:
            for c in range(remaining + 1):
                yield from _gen(prefix + [c], remaining - c, dim - 1)

    yield from _gen([], N, k)



######################## GENOTYPE ENUMERATION AND FITNESS COMPUTATION ###################################


# ---------------------------------------------------------------------------
# Fast generation of all genotypes and their fitness values
# ---------------------------------------------------------------------------

"""
Genotype enumeration and vectorized fitness computation
-------------------------------------------------------

For n loci there are

    k = 2^n

possible genotypes. Each genotype is represented as a binary vector

    g = (g_1, ..., g_n),   g_i ∈ {0,1}

where g_i indicates the allele at locus i.

To enumerate all genotypes we use the fact that integers

    i = 0, 1, ..., (2^n - 1)

have binary representations with n bits. Those bits directly correspond
to the genotype vector.

Example (n = 3):

    i = 5
    binary(i) = 101
    genotype  = [1, 0, 1]

Thus row i of the genotype matrix corresponds to the binary representation
of i written as a length-n vector.

Bit operations are used to extract the bits:

    i >> m
        shifts the binary representation of i to the right by m positions

    & 1
        extracts the least significant bit (0 or 1)

The expression

    (indices >> bit_positions) & 1

extracts every bit of every index simultaneously using NumPy vectorization.
This produces a matrix

    G ∈ {0,1}^{2^n × n}

where row i is the genotype corresponding to index i.

Fitness model
-------------

Genotype fitness is defined as

    ω(g) =  baseline + gᵀ Σ g

where

    Σ_ii = s_i        (additive effects)
    Σ_ij = ε_ij/2       (pairwise epistasis) -- so that summing over i and j gives the correct value.

Since we compute this for every genotype simultaneously, we use the
Einstein summation

    ω_g =  baseline + Σ_{i,j} g_i Σ_ij g_j  (where Σ_ij = Σ_ji)

implemented efficiently with numpy.einsum.
"""


def all_genotypes(n: int) -> np.ndarray:
    """
    Return all 2^n genotypes as an array of shape (2^n, n).
    Row i contains the binary representation of i as a genotype vector.
    """
    indices = np.arange(2**n, dtype=np.uint64)[:, None]
    bit_positions = np.arange(n - 1, -1, -1, dtype=np.uint64)
    return ((indices >> bit_positions) & 1).astype(np.int8)


def fitness_all_genotypes(
    G: np.ndarray,
    Sigma: np.ndarray,
    baseline: float = 1.0,
) -> np.ndarray:
    """
    Compute genotype fitness values

        ω(g) = baseline + gᵀ Σ g  (where Σ_ii = s_i, Σ_ij = ε_ij/2 for i != j)

    for every genotype g in the genotype matrix G.


    Parameters
    ----------
    G : array of shape (2^n, n)
        Genotype matrix where each row is a genotype vector.
    Sigma : array of shape (n, n)
        Epistasis matrix.
    baseline : float
        Constant fitness offset ensuring positive fitness values.

    Returns
    -------
    omega : array of shape (2^n,)
        Fitness value for each genotype.

    to be explicit:
    omega[0] → fitness of genotype [0,0,...,0]
    omega[1] → fitness of genotype [0,0,...,1]
    ...
    omega[k-1] → fitness of genotype [1,1,...,1]


    Comment: The einsum string commented below means:

        "bi,ij,bj->b"

    means:

        G[b,i] * Sigma[i,j] * G[b,j]

    summed over indices i and j, while keeping index b.

    So for each genotype b we compute

        ω_b = baseline + ∑_{i,j} G[b,i] Σ[i,j] G[b,j]

    which is exactly the quadratic form g_bᵀ Σ g_b.

    """
    return baseline + np.einsum("bi,ij,bj->b", G, Sigma, G)


# Example usage inside the main workflow:

# G = all_genotypes(n)                        # genotype matrix (k × n)
# omega = fitness_all_genotypes(G, Sigma)     # genotype fitness values

# # mutation parameters for each genotype
# n_ones = G.sum(axis=1)
# mu_to_g = u * n_ones + v * (n - n_ones)

# alpha = 4 * N_pop * mu_to_g




# ---------------------------------------------------------------------------
# mutation rate to genotype
# ---------------------------------------------------------------------------

# def genotype_from_index(i: int, n: int) -> np.ndarray:
#     """Genotype g_i as binary vector (locus 0 .. n-1)."""
#     return np.array([(i >> (n - 1 - j)) & 1 for j in range(n)])


# def mutation_rate_to_genotype(g: np.ndarray, u: float, v: float) -> float:
#     """μ_{→g} = u·(# of 1's) + v·(# of 0's)."""
#     n_ones = int(np.sum(g))
#     return u * n_ones + v * (len(g) - n_ones)

"""
Mutation parameter associated with each genotype
------------------------------------------------

Alleles mutate independently at loci:

    0 → 1 with rate u
    1 → 0 with rate v

For a genotype g = (y_1^g,...,y_n^g), consider the genotypes at Hamming
distance 1 from g (genotypes differing at exactly one locus). There are
exactly n such neighbors.

- If y_i^g = 1, the neighbor that mutates into g must have 0 at locus i,
  and the mutation 0→1 occurs at rate u.
- If y_i^g = 0, the neighbor must have 1 at locus i, and the mutation
  1→0 occurs at rate v.

Thus each 1 in g contributes rate u and each 0 contributes rate v.
Writing

    sum(g) = Σ_i y_i^g = number of 1's in g,

the mutation parameter is

    μ_{→g} = u·(#1's in g) + v·(#0's in g)
           = u Σ_i y_i^g + v (n − Σ_i y_i^g).

If G is the genotype matrix of shape (2^n, n), where row i is genotype g_i,
then G.sum(axis=1) gives the number of 1's in each genotype, so

    mu_to_g = u * n_ones + v * (n - n_ones)

computes μ_{→g} for all genotypes simultaneously.

The function below therefore returns the vector

    mu_to_g = ( μ_{→g_0}, μ_{→g_1}, ..., μ_{→g_{2^n-1}} )

of length 2^n.
"""
def mutation_rates_all_genotypes(G: np.ndarray, u: float, v: float) -> np.ndarray:
    """
    Compute μ_{→g} = u·(# of 1's) + v·(# of 0's) for every genotype in G.

    For each genotype g,
    0 → 1 with rate u
    1 → 0 with rate v
    Returns
    -------
    mu_to_g : ndarray of shape (2^n,)
        Entry i is the mutation parameter associated with genotype g_i.
    """
    n = G.shape[1]
    n_ones = G.sum(axis=1)
    return u * n_ones + v * (n - n_ones)



# ---------------------------------------------------------------------------
# Epistasis matrix Σ and fitness
# ---------------------------------------------------------------------------

def build_sigma(
    n: int,
    s: np.ndarray | None = None, 
    #syntax: s is optional. (| means "or")
    #If provided it must be a numpy array.
    #If not provided it will be None.
    eps: np.ndarray | None = None,
    s_dist: Callable[[], float] | None = None, 
    #a function that takes no arguments and returns a float
    # e.g. lambda: np.random.normal(0, 1)
    eps_dist: Callable[[], float] | None = None,
) -> np.ndarray:
    """
    Build n×n symmetric matrix Σ: Σ_ii = s_i, Σ_ij = ε_ij/2 for i≠j.
    If s_dist/eps_dist provided, sample. Else use s, eps arrays.
    """
    Sigma = np.zeros((n, n))
    if s is not None:
        for i in range(n):
            Sigma[i, i] = s[i]
    elif s_dist is not None:
        for i in range(n):
            Sigma[i, i] = s_dist()
    else:
        Sigma += np.eye(n)  # default s_i = 1

    if eps is not None:
        for i in range(n):
            for j in range(i + 1, n):
                Sigma[i, j] = Sigma[j, i] = eps[i, j]/2
    elif eps_dist is not None:
        for i in range(n):
            for j in range(i + 1, n):
                Sigma[i, j] = Sigma[j, i] = eps_dist()/2

    return Sigma


# def fitness_genotype(g: np.ndarray, Sigma: np.ndarray, baseline: float = 1.0) -> float:
#     """ω(g) = baseline +at(baseline + g @ Sigma @ g)


def mean_fitness(x: np.ndarray, omega: np.ndarray) -> float:
    """W(x) = ∑_g ω(g) x_g."""
    return float(np.dot(x, omega))



########################## equilibrium/statistics/diagnostics ###################################

"""
What is being computed here
---------------------------

We work with the equilibrium distribution on the simplex lattice:

    ρ∞(x) ∝ W(x)^(2N) ∏_i x_i^(α_i - 1),

where

    W(x) = Σ_i ω_i x_i

is the mean fitness of the population state x.

For each lattice point x, define its unnormalized log-weight

    ell(x) = -V(x) = log ρ∞(x) + constant
           = 2N log W(x) + Σ_i (α_i - 1) log x_i.

If we write

    Z = Σ_x exp(ell(x)),

then the normalized equilibrium probability is

    p(x) = exp(ell(x)) / Z.

The goal of this section is NOT to store all lattice points x or all
probabilities p(x), since for large grids that would use too much memory.
Instead, we stream over the lattice points one by one and accumulate only
the quantities needed for the summary statistics we care about.

Quantities we want to output
----------------------------

1. Equilibrium mean fitness:

       E_p[W] = Σ_x p(x) W(x)

2. Equilibrium Shannon entropy:

       H(p) = - Σ_x p(x) log p(x)

Key identity for the entropy
----------------------------

Since

    p(x) = exp(ell(x)) / Z,

we have

    log p(x) = ell(x) - log Z.

Substituting this into the entropy definition gives

    H(p)
      = - Σ_x p(x) log p(x)
      = - Σ_x p(x) (ell(x) - log Z)
      = - Σ_x p(x) ell(x) + log Z Σ_x p(x)
      = log Z - E_p[ell(x)],

because Σ_x p(x) = 1.

So instead of explicitly forming p(x), it is enough to compute:

    Z              = Σ_x exp(ell(x))
    Σ_x exp(ell(x)) W(x)
    Σ_x exp(ell(x)) ell(x)

Then

    E_p[W] = (Σ_x exp(ell(x)) W(x)) / Z

and

    H(p)   = log Z - (Σ_x exp(ell(x)) ell(x)) / Z.

Why this is useful computationally
----------------------------------

A direct computation of p(x) for all lattice points would require storing
the entire simplex grid and the full probability vector, which is expensive
or impossible on large grids.

The streaming approach avoids this: we loop over lattice points one at a
time, compute their log-weight ell(x), and update only a few running sums.
This keeps memory usage small while still giving the exact equilibrium mean
fitness and entropy.
"""
# ---------------------------------------------------------------------------
# Log-weight for one lattice point, using integer counts c instead of x = c/N
# ---------------------------------------------------------------------------

def compute_log_weight_from_counts(
    counts: np.ndarray,
    omega: np.ndarray,
    alpha: np.ndarray,
    N_pop: int,
    eps: float = 1e-300,
) -> tuple[float, float]:
    """
    For counts with sum(counts) = N_pop, compute:

        ell = -V = log unnormalized weight
        W   = mean fitness at x = counts / N_pop = mean_fitness(x, omega)

    Returns
    -------
    ell : float
        log unnormalized equilibrium weight
    W : float
        mean fitness at this simplex point
    """
    x = counts / N_pop
    W = float(x @ omega)

    if W <= 0:
        return -np.inf, W

    term1 = 2.0 * N_pop * math.log(W)

    # Use x_i = counts_i / N_pop. For zero counts, clip by eps.
    x_safe = np.maximum(x, eps)
    term2 = float(np.dot(alpha - 1.0, np.log(x_safe)))

    ell = term1 + term2
    return ell, W


# ---------------------------------------------------------------------------
# Streaming exact computation of Z, E[W], and entropy
# ---------------------------------------------------------------------------

def equilibrium_stats_streaming(
    k: int,
    N_pop: int,
    omega: np.ndarray,
    alpha: np.ndarray,
    eps: float = 1e-300,
    progress_every: int | None = None,
) -> tuple[float, float]:
    """
    Compute exact equilibrium mean fitness and entropy by streaming over all
    simplex lattice points, without storing the full grid or all probabilities.

    Returns
    -------
    E_W : float
        Expected mean fitness under the equilibrium distribution.
    H : float
        Shannon entropy of the equilibrium distribution over lattice points.
    """
    max_logw = -np.inf
    sum_exp = 0.0          # Σ exp(ell - max_logw)
    sum_exp_W = 0.0        # Σ exp(ell - max_logw) * W
    sum_exp_logw = 0.0     # Σ exp(ell - max_logw) * ell

    n_seen = 0

    for counts in simplex_count_generator(k, N_pop):
        ell, W = compute_log_weight_from_counts(counts, omega, alpha, N_pop, eps=eps)

        if not np.isfinite(ell):
            continue

        if ell > max_logw:
            # Rescale old sums to the new max for numerical stability
            scale = math.exp(max_logw - ell) if np.isfinite(max_logw) else 0.0
            sum_exp *= scale
            sum_exp_W *= scale
            sum_exp_logw *= scale
            max_logw = ell

        w = math.exp(ell - max_logw)
        sum_exp += w
        sum_exp_W += w * W
        sum_exp_logw += w * ell

        n_seen += 1
        if progress_every is not None and n_seen % progress_every == 0:
            print(f"Processed {n_seen} lattice points...")

    if sum_exp == 0.0:
        raise ValueError("All equilibrium weights vanished or were invalid.")

    logZ = max_logw + math.log(sum_exp)
    E_W = sum_exp_W / sum_exp
    E_logw = sum_exp_logw / sum_exp

    # H = -Σ p log p = log Z - E_p[ell]
    H = logZ - E_logw

    return E_W, H


# ---------------------------------------------------------------------------
# MCMC sampling for large grids (when exact enumeration is infeasible)
# ---------------------------------------------------------------------------

def mcmc_equilibrium_sampler(
    k: int,
    N_pop: int,
    omega: np.ndarray,
    alpha: np.ndarray,
    n_steps: int = 50_000,
    burn_in: int = 5_000,
    thin: int = 10,
    seed: int = 42,
    eps: float = 1e-300,
    progress_every: int | None = None,
):
    """
    MCMC sampler for the equilibrium distribution on the simplex lattice.
    Proposal: move one unit from genotype i to j (symmetric random walk).
    Yields (counts, x, W, log_weight) for each thinned post-burn-in step.
    """
    rng = np.random.default_rng(seed)

    # Start at uniform: each genotype gets N/k (approx) or spread evenly
    counts = np.zeros(k, dtype=np.int64)
    remainder = N_pop
    for i in range(k - 1):
        counts[i] = remainder // (k - i)
        remainder -= counts[i]
    counts[k - 1] = remainder
    if counts.sum() != N_pop:
        counts[k - 1] += N_pop - counts.sum()

    ell_cur, W_cur = compute_log_weight_from_counts(counts, omega, alpha, N_pop, eps=eps)
    if not np.isfinite(ell_cur):
        # Fallback: start at a random point
        counts = np.zeros(k, dtype=np.int64)
        counts[0] = N_pop
        ell_cur, W_cur = compute_log_weight_from_counts(counts, omega, alpha, N_pop, eps=eps)

    n_accepted = 0
    for step in range(n_steps + burn_in):
        # Proposal: pick i with c_i > 0 and j != i
        i_candidates = np.where(counts > 0)[0]
        if len(i_candidates) == 0:
            break
        i = rng.choice(i_candidates)
        j = rng.integers(0, k)
        if j == i:
            j = (j + 1) % k

        counts_new = counts.copy()
        counts_new[i] -= 1
        counts_new[j] += 1

        ell_new, W_new = compute_log_weight_from_counts(
            counts_new, omega, alpha, N_pop, eps=eps
        )
        if not np.isfinite(ell_new):
            continue

        log_accept = ell_new - ell_cur
        if log_accept >= 0 or rng.random() < math.exp(log_accept):
            counts = counts_new
            ell_cur, W_cur = ell_new, W_new
            n_accepted += 1

        if step >= burn_in and (step - burn_in) % thin == 0:
            x = counts.astype(float) / N_pop
            yield counts.copy(), x, W_cur, ell_cur

        if progress_every and (step + 1) % progress_every == 0:
            acc_rate = n_accepted / (step + 1)
            print(f"MCMC step {step + 1}/{n_steps + burn_in}, accept rate {acc_rate:.3f}")


def equilibrium_stats_mcmc(
    k: int,
    N_pop: int,
    omega: np.ndarray,
    alpha: np.ndarray,
    n_steps: int = 50_000,
    burn_in: int = 5_000,
    thin: int = 10,
    seed: int = 42,
    progress_every: int | None = None,
) -> dict:
    """
    Estimate equilibrium statistics via MCMC sampling.
    Returns dict with E_W, H (entropy), and geometry diagnostics.
    """
    samples_x: list[np.ndarray] = []
    samples_W: list[float] = []
    samples_ell: list[float] = []

    for counts, x, W, ell in mcmc_equilibrium_sampler(
        k, N_pop, omega, alpha,
        n_steps=n_steps, burn_in=burn_in, thin=thin, seed=seed,
        progress_every=progress_every,
    ):
        samples_x.append(x)
        samples_W.append(W)
        samples_ell.append(ell)

    if not samples_x:
        raise ValueError("MCMC produced no valid samples.")

    X = np.array(samples_x)
    W_arr = np.array(samples_W)
    ell_arr = np.array(samples_ell)

    # E[W]
    E_W = float(np.mean(W_arr))

    # Entropy: plugin estimator on empirical distribution over visited states
    from collections import Counter
    state_tuples = [tuple(x) for x in X]
    counts_emp = Counter(state_tuples)
    n_unique = len(counts_emp)
    n_samples = len(X)
    p_emp = np.array([c / n_samples for c in counts_emp.values()])
    p_emp = np.maximum(p_emp, 1e-300)
    H_proxy = float(-np.sum(p_emp * np.log(p_emp)))

    # Geometry: concentration near vertices
    max_x = np.max(X, axis=1)  # max genotype freq per sample
    E_max_x = float(np.mean(max_x))
    frac_near_vertex = float(np.mean(max_x > 0.9))   # mass in "near-vertex" region
    frac_dominant = float(np.mean(max_x > 0.5))     # mass with one genotype > 50%
    E_spread = float(np.mean(1.0 - max_x))           # 1 - max = "spread" (0 = at vertex)

    # Which vertex is most often dominant?
    dominant_idx = np.argmax(X, axis=1)
    mode_vertex = int(np.bincount(dominant_idx, minlength=k).argmax())

    return {
        "E_W": E_W,
        "H_proxy": H_proxy,
        "n_samples": len(samples_x),
        "n_unique_states": n_unique,
        "E_max_x": E_max_x,
        "frac_near_vertex": frac_near_vertex,
        "frac_dominant": frac_dominant,
        "E_spread": E_spread,
        "mode_vertex": mode_vertex,
    }


def equilibrium_distribution_grid(
    k: int,
    N_pop: int,
    omega: np.ndarray,
    alpha: np.ndarray,
    eps: float = 1e-300,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (points, probs) for the equilibrium distribution over the simplex grid.
    For visualization. If max_points is set and grid is larger, subsample uniformly.
    """
    points_list: list[np.ndarray] = []
    log_weights: list[float] = []
    max_logw = -np.inf

    for counts in simplex_count_generator(k, N_pop):
        ell, _ = compute_log_weight_from_counts(
            counts, omega, alpha, N_pop, eps=eps
        )
        if not np.isfinite(ell):
            continue
        points_list.append(counts.astype(float) / N_pop)
        if ell > max_logw:
            max_logw = ell
        log_weights.append(ell)

    if not points_list:
        raise ValueError("All equilibrium weights vanished or were invalid.")

    points = np.array(points_list)
    log_w = np.array(log_weights)
    logZ = max_logw + math.log(np.sum(np.exp(log_w - max_logw)))
    probs = np.exp(log_w - logZ)

    if max_points is not None and len(points) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(points), max_points, replace=False)
        points = points[idx]
        probs = probs[idx]
        probs = probs / probs.sum()

    return points, probs


# This is all fine just really doesnt optimize for memory
# ---------------------------------------------------------------------------
# Equilibrium distribution
# ---------------------------------------------------------------------------

# def compute_log_rho(
#     x: np.ndarray,
#     omega: np.ndarray,
#     alpha: np.ndarray,
#     N_pop: int,
#     eps: float = 1e-300,
# ) -> float:
#     """
#     log ρ(x) ∝ 2N·log W(x) + ∑_i (α_i - 1)·log(x_i)
#     Use eps for x_i = 0 to avoid -inf.
#     """
#     W = mean_fitness(x, omega)
#     if W <= 0:
#         return -np.inf
#     log_W = np.log(W + eps)
#     term1 = 2 * N_pop * log_W

#     x_safe = np.maximum(x, eps)
#     term2 = np.sum((alpha - 1) * np.log(x_safe))

#     return term1 + term2


# def compute_rho_grid(
#     points: np.ndarray,
#     omega: np.ndarray,
#     alpha: np.ndarray,
#     N_pop: int,
# ) -> np.ndarray:
#     """Unnormalized ρ at each lattice point. Returns log-values for stability."""
#     log_rho = np.array([
#         compute_log_rho(p, omega, alpha, N_pop) for p in points
#     ])
#     return log_rho


# def normalize_log_probs(log_probs: np.ndarray) -> np.ndarray:
#     """Convert log-probs to normalized probs via log-sum-exp."""
#     m = np.max(log_probs)
#     probs = np.exp(log_probs - m)
#     return probs / np.sum(probs)


# # ---------------------------------------------------------------------------
# # Diagnostics
# # ---------------------------------------------------------------------------

# def entropy(probs: np.ndarray, eps: float = 1e-300) -> float:
#     """H = -∑ p log p."""
#     p = np.maximum(probs, eps)
#     return float(-np.sum(p * np.log(p)))


# def mean_fitness_over_distribution(
#     points: np.ndarray,
#     probs: np.ndarray,
#     omega: np.ndarray,
# ) -> float:
#     """E_ρ[W] = ∑_x ρ(x) W(x)."""
#     W_vals = np.array([mean_fitness(p, omega) for p in points])
#     return float(np.dot(probs, W_vals))


# ---------------------------------------------------------------------------
# Config file loading (YAML preferred; JSON also supported)
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict[str, Any]:
    """Load config from YAML or JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        if not HAS_YAML:
            raise ImportError("YAML config requires PyYAML: pip install pyyaml")
        return yaml.safe_load(text)
    return json.loads(text)


def apply_config_to_args(args, config: dict[str, Any]) -> None:
    """Merge config into args. Config values override; CLI can override config."""
    # Top-level
    for key, attr in [
        ("n_loci", "n_loci"),
        ("N", "N"),
        ("u", "u"),
        ("v", "v"),
        ("seed", "seed"),
        ("out", "out"),
        ("progress_every", "progress_every"),
    ]:
        if key in config and config[key] is not None:
            setattr(args, attr, config[key])

    # Sigma: direct matrix or file
    if "sigma" in config and config["sigma"] is not None:
        setattr(args, "sigma", np.array(config["sigma"], dtype=float))
        setattr(args, "sigma_file", "")
    elif "sigma_file" in config and config["sigma_file"]:
        setattr(args, "sigma_file", config["sigma_file"])
        if hasattr(args, "sigma"):
            delattr(args, "sigma")

    # s block
    if "s" in config and config["s"]:
        s_cfg = config["s"]
        for k, attr in [
            ("dist", "s_dist"),
            ("fixed", "s_fixed"),
            ("mean", "s_mean"),
            ("std", "s_std"),
            ("var", "s_var"),
            ("low", "s_low"),
            ("high", "s_high"),
        ]:
            if k in s_cfg and s_cfg[k] is not None:
                setattr(args, attr, s_cfg[k])

    # eps block
    if "eps" in config and config["eps"]:
        eps_cfg = config["eps"]
        for k, attr in [
            ("dist", "eps_dist"),
            ("fixed", "eps_fixed"),
            ("mean", "eps_mean"),
            ("std", "eps_std"),
            ("var", "eps_var"),
            ("low", "eps_low"),
            ("high", "eps_high"),
        ]:
            if k in eps_cfg and eps_cfg[k] is not None:
                setattr(args, attr, eps_cfg[k])

##### this is for the viz -- for the plotting.

def build_sigma_omega_alpha(args) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build Sigma, omega, alpha from args. Returns (n, k, N_pop, Sigma, omega, alpha, G).
    """
    n = args.n_loci
    k = 1 << n
    N_pop = args.N
    rng = np.random.default_rng(args.seed)

    sigma_direct = getattr(args, "sigma", None)
    if sigma_direct is not None and isinstance(sigma_direct, np.ndarray):
        Sigma = sigma_direct.astype(float)
        if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
            raise ValueError(f"sigma: expected square matrix, got shape {Sigma.shape}")
        if Sigma.shape[0] != n:
            raise ValueError(f"sigma: matrix is {Sigma.shape[0]}×{Sigma.shape[1]}, n_loci={n} requires {n}×{n}")
        Sigma = (Sigma + Sigma.T) / 2.0
    elif args.sigma_file:
        path = args.sigma_file
        if path.endswith(".npy"):
            Sigma = np.load(path).astype(float)
        else:
            Sigma = np.loadtxt(path).astype(float)
        if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
            raise ValueError(f"--sigma-file: expected square matrix, got shape {Sigma.shape}")
        if Sigma.shape[0] != n:
            raise ValueError(f"--sigma-file: matrix is {Sigma.shape[0]}×{Sigma.shape[1]}, n_loci={n} requires {n}×{n}")
        Sigma = (Sigma + Sigma.T) / 2.0
    else:
        if args.s_dist == "normal":
            s_scale = math.sqrt(args.s_var) if args.s_var is not None else args.s_std
            s = rng.normal(args.s_mean, s_scale, n)
        elif args.s_dist == "uniform":
            s = rng.uniform(args.s_low, args.s_high, n)
        else:
            s = np.full(n, args.s_fixed, dtype=float)
        if args.eps_dist == "normal":
            eps_scale = math.sqrt(args.eps_var) if args.eps_var is not None else args.eps_std
            eps = rng.normal(args.eps_mean, eps_scale, (n, n))
        elif args.eps_dist == "uniform":
            eps = rng.uniform(args.eps_low, args.eps_high, (n, n))
        else:
            eps = np.full((n, n), args.eps_fixed, dtype=float)
        eps = (eps + eps.T) / 2.0
        np.fill_diagonal(eps, 0.0)
        Sigma = build_sigma(n, s=s, eps=eps)

    G = all_genotypes(n)
    omega = fitness_all_genotypes(G, Sigma)
    mu_to_g = mutation_rates_all_genotypes(G, args.u, args.v)
    alpha = 4.0 * N_pop * mu_to_g

    return n, k, N_pop, Sigma, omega, alpha, G

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args):
    n = args.n_loci
    k = 1 << n
    N_pop = args.N
    u = args.u
    v = args.v

    print(f"n={n} loci, k={k} genotypes, N={N_pop}")
    print(f"Mutation rates: u={u} (0→1), v={v} (1→0)")

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Build Σ: direct matrix, from file, or from s/eps distributions
    # ------------------------------------------------------------------
    sigma_direct = getattr(args, "sigma", None)
    if sigma_direct is not None and isinstance(sigma_direct, np.ndarray):
        # Σ provided directly (e.g. from config)
        Sigma = sigma_direct.astype(float)
        if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
            raise ValueError(f"sigma: expected square matrix, got shape {Sigma.shape}")
        if Sigma.shape[0] != n:
            raise ValueError(
                f"sigma: matrix is {Sigma.shape[0]}×{Sigma.shape[1]}, "
                f"but n_loci={n} requires {n}×{n}"
            )
        Sigma = (Sigma + Sigma.T) / 2.0
        s = np.diag(Sigma).copy()
        eps = Sigma.copy()
        np.fill_diagonal(eps, 0.0)
        print("Σ from config (direct matrix)")
    elif args.sigma_file:
        # Load custom Σ from file (.npy or .txt)
        path = args.sigma_file
        if path.endswith(".npy"):
            Sigma = np.load(path).astype(float)
        else:
            Sigma = np.loadtxt(path).astype(float)
        if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
            raise ValueError(f"--sigma-file: expected square matrix, got shape {Sigma.shape}")
        if Sigma.shape[0] != n:
            raise ValueError(
                f"--sigma-file: matrix is {Sigma.shape[0]}×{Sigma.shape[1]}, "
                f"but --n-loci={n} requires {n}×{n}"
            )
        # Symmetrize (in case user provided upper/lower triangular)
        Sigma = (Sigma + Sigma.T) / 2.0
        s = np.diag(Sigma).copy()
        eps = Sigma.copy()
        np.fill_diagonal(eps, 0.0)
        print("Σ loaded from file (custom matrix)")
    else:
        # Sample / set additive effects s_i
        if args.s_dist == "normal":
            s_scale = math.sqrt(args.s_var) if args.s_var is not None else args.s_std
            s = rng.normal(args.s_mean, s_scale, n)
        elif args.s_dist == "uniform":
            s = rng.uniform(args.s_low, args.s_high, n)
        else:
            s = np.full(n, args.s_fixed, dtype=float)

        # Sample / set pairwise epistasis coefficients ε_ij
        if args.eps_dist == "normal":
            eps_scale = math.sqrt(args.eps_var) if args.eps_var is not None else args.eps_std
            eps = rng.normal(args.eps_mean, eps_scale, (n, n))
        elif args.eps_dist == "uniform":
            eps = rng.uniform(args.eps_low, args.eps_high, (n, n))
        else:
            eps = np.full((n, n), args.eps_fixed, dtype=float)

        eps = (eps + eps.T) / 2.0
        np.fill_diagonal(eps, 0.0)
        Sigma = build_sigma(n, s=s, eps=eps)

    # ------------------------------------------------------------------
    # Print coefficient diagnostics and Σ matrix
    # ------------------------------------------------------------------
    print(f"s_i (diagonal): min={s.min():.3g}, max={s.max():.3g}, mean={s.mean():.3g}")

    if n >= 2:
        eps_vals = eps[np.triu_indices(n, 1)]
        print(
            f"ε_ij (off-diag): min={eps_vals.min():.3g}, "
            f"max={eps_vals.max():.3g}, "
            f"mean={eps_vals.mean():.3g}"
        )

    print("\nΣ (epistasis matrix, Σ_ii=s_i, Σ_ij=ε_ij/2):")
    np.set_printoptions(precision=4, suppress=True)
    print(Sigma)
    print()

    # ------------------------------------------------------------------
    # Build all genotypes and genotype-level quantities
    # ------------------------------------------------------------------
    G = all_genotypes(n)
    omega = fitness_all_genotypes(G, Sigma)
    mu_to_g = mutation_rates_all_genotypes(G, u, v)
    alpha = 4.0 * N_pop * mu_to_g

    print(f"ω(g): min={omega.min():.3g}, max={omega.max():.3g}, mean={omega.mean():.3g}")

    print("\nGenotypes and fitness ω(g) = baseline + gᵀ Σ g:")
    for i in range(k):
        g_str = "".join(str(int(b)) for b in G[i])
        print(f"  {g_str}  →  ω = {omega[i]:.6g}")
    print()

    print(f"μ_to_g: min={mu_to_g.min():.3g}, max={mu_to_g.max():.3g}")
    print(f"α_i: min={alpha.min():.3g}, max={alpha.max():.3g}")

    # Warn if some genotype fitnesses are nonpositive
    if np.any(omega <= 0):
        n_bad = int(np.sum(omega <= 0))
        print(
            f"Warning: {n_bad} genotype fitness values are nonpositive. "
            f"Some simplex points may therefore receive zero weight."
        )

    # ------------------------------------------------------------------
    # Lattice size and mode (exact vs MCMC)
    # ------------------------------------------------------------------
    n_pts = math.comb(N_pop + k - 1, k - 1)
    print(f"Lattice size: {n_pts} points")

    use_mcmc = False
    if args.mode == "mcmc":
        use_mcmc = True
    elif args.mode == "auto" and n_pts > args.max_grid:
        use_mcmc = True
        print(f"Grid exceeds {args.max_grid} points → using MCMC sampling")

    progress_every = args.progress_every if args.progress_every > 0 else None

    if use_mcmc:
        # ------------------------------------------------------------------
        # MCMC sampling (for large grids)
        # ------------------------------------------------------------------
        mcmc_stats = equilibrium_stats_mcmc(
            k=k,
            N_pop=N_pop,
            omega=omega,
            alpha=alpha,
            n_steps=args.mcmc_steps,
            burn_in=args.mcmc_burn_in,
            thin=args.mcmc_thin,
            seed=args.seed,
            progress_every=progress_every,
        )
        E_W = mcmc_stats["E_W"]
        H = mcmc_stats["H_proxy"]

        print("\n--- Equilibrium diagnostics (MCMC estimates) ---")
        print(f"E_ρ[W] (mean fitness) = {E_W:.6g}")
        print(f"log(unique states) ≈ {H:.4g}  (entropy proxy)")
        print(f"Samples: {mcmc_stats['n_samples']}, unique states visited: {mcmc_stats['n_unique_states']}")
        print("\n--- Geometry of the distribution ---")
        print(f"E[max_i x_i] = {mcmc_stats['E_max_x']:.4g}  (1 = concentrated at vertex)")
        print(f"Fraction of mass with max > 0.9 (near vertex): {mcmc_stats['frac_near_vertex']:.2%}")
        print(f"Fraction with one genotype > 50%: {mcmc_stats['frac_dominant']:.2%}")
        print(f"E[1 - max x] (spread): {mcmc_stats['E_spread']:.4g}  (0 = at vertex)")
        g_str = "".join(str(int(b)) for b in G[mcmc_stats["mode_vertex"]])
        print(f"Most often dominant genotype: {g_str} (index {mcmc_stats['mode_vertex']})")
    else:
        # ------------------------------------------------------------------
        # Streaming exact equilibrium statistics
        # ------------------------------------------------------------------
        E_W, H = equilibrium_stats_streaming(
            k=k,
            N_pop=N_pop,
            omega=omega,
            alpha=alpha,
            progress_every=progress_every,
        )

        print("\n--- Equilibrium diagnostics ---")
        print(f"E_ρ[W] (mean fitness) = {E_W:.6g}")
        print(f"H(ρ) (entropy) = {H:.6g}")

    # ------------------------------------------------------------------
    # Optional compact save
    # ------------------------------------------------------------------
    if args.out:
        np.savez(
            args.out,
            omega=omega,
            alpha=alpha,
            Sigma=Sigma,
            s=s,
            eps=eps,
            n=n,
            k=k,
            N=N_pop,
            u=u,
            v=v,
            E_W=E_W,
            H=H,
        )
        print(f"\nSaved to {args.out}")


def main():
    parser = argparse.ArgumentParser(
        description="Equilibrium distribution ρ∞ over genotype simplex."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to YAML config file (JSON also supported). All parameters can be specified there.",
    )
    parser.add_argument("--n-loci", type=int, default=2, help="Number of loci.")
    parser.add_argument("--N", type=int, default=20, help="Population size (grid denominator).")
    parser.add_argument("--u", type=float, default=1e-4, help="Mutation rate u (0→1).")
    parser.add_argument("--v", type=float, default=1e-4, help="Mutation rate v (1→0).")
    parser.add_argument("--seed", type=int, default=42)

    # Selection coefficients
    parser.add_argument("--s-dist", choices=["normal", "uniform", "fixed"], default="fixed")
    parser.add_argument("--s-fixed", type=float, default=0.1)
    parser.add_argument("--s-mean", type=float, default=0.1, help="Mean of s when --s-dist=normal")
    parser.add_argument("--s-std", type=float, default=0.05, help="Std of s when --s-dist=normal")
    parser.add_argument("--s-var", type=float, default=None, help="Variance of s (overrides --s-std when --s-dist=normal)")
    parser.add_argument("--s-low", type=float, default=-0.1)
    parser.add_argument("--s-high", type=float, default=0.2)

    # Epistasis
    parser.add_argument("--eps-dist", choices=["normal", "uniform", "fixed"], default="fixed")
    parser.add_argument("--eps-fixed", type=float, default=0.0)
    parser.add_argument("--eps-mean", type=float, default=0.0, help="Mean of ε when --eps-dist=normal")
    parser.add_argument("--eps-std", type=float, default=0.05, help="Std of ε when --eps-dist=normal")
    parser.add_argument("--eps-var", type=float, default=None, help="Variance of ε (overrides --eps-std when --eps-dist=normal)")
    parser.add_argument("--eps-low", type=float, default=-0.1)
    parser.add_argument("--eps-high", type=float, default=0.1)

    parser.add_argument(
        "--sigma-file",
        type=str,
        default="",
        help="Path to n×n matrix file (.npy or .txt). Overrides s/eps; must match --n-loci.",
    )
    parser.add_argument("--out", type=str, default="", help="Save results to .npz file.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every this many lattice points; 0 disables progress output.",
    )
    parser.add_argument(
        "--mode",
        choices=["exact", "mcmc", "auto"],
        default="auto",
        help="exact=full enumeration, mcmc=sample, auto=use mcmc when grid > --max-grid (default: auto).",
    )
    parser.add_argument(
        "--max-grid",
        type=int,
        default=500_000,
        help="Max lattice points for exact mode; above this, auto uses MCMC (default: 500000).",
    )
    parser.add_argument(
        "--mcmc-steps",
        type=int,
        default=50_000,
        help="MCMC steps (including burn-in) when using mcmc mode.",
    )
    parser.add_argument(
        "--mcmc-burn-in",
        type=int,
        default=5_000,
        help="MCMC burn-in steps.",
    )
    parser.add_argument(
        "--mcmc-thin",
        type=int,
        default=10,
        help="MCMC thinning (keep every Nth sample).",
    )
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        apply_config_to_args(args, config)
        print(f"Config loaded from {args.config}\n")

    run(args)


if __name__ == "__main__":
    main()


# =============================================================================
# USAGE EXAMPLES (commented out — run from the command line)
# =============================================================================
#
# This script computes the equilibrium distribution ρ∞ over the genotype simplex
# and reports mean fitness E_ρ[W] and entropy H(ρ). Below are example invocations
# for different choices of selection (s) and epistasis (ε) parameters.
#
# -----------------------------------------------------------------------------
# 0. CONFIG FILE (YAML)
# -----------------------------------------------------------------------------
# Specify all parameters in a YAML config file. Supports sigma (direct matrix),
# sigma_file (path), or s/eps distributions. Requires: pip install pyyaml
#
#   python equilibria_explored.py --config equilibria_config_example.yaml
#   python equilibria_explored.py --config equilibria_config_sigma_direct.yaml
#   python equilibria_explored.py --config equilibria_config_normal_dist.yaml
#
# See equilibria_config_example.yaml for the full schema (sigma, sigma_file, s, eps).
#
# -----------------------------------------------------------------------------
# 1. FIXED (constant) s and ε — default behaviour
# -----------------------------------------------------------------------------
# Use constant values for all selection coefficients and epistasis coefficients.
# Default: s_i = 0.1 for all loci, ε_ij = 0 for all pairs.
#
#   python equilibria_explored.py
#
# Custom fixed values:
#
#   python equilibria_explored.py --s-fixed 0.15 --eps-fixed 0.02
#
# -----------------------------------------------------------------------------
# 2. NORMAL distribution for s and/or ε
# -----------------------------------------------------------------------------
# Sample s_i and ε_ij from normal distributions. Use --s-mean, --s-std (or --s-var)
# for selection; --eps-mean, --eps-std (or --eps-var) for epistasis.
#
# s ~ Normal(mean=0.1, std=0.05):
#   python equilibria_explored.py --s-dist normal --s-mean 0.1 --s-std 0.05
#
# s ~ Normal(mean=0.2, variance=0.01) — using variance instead of std:
#   python equilibria_explored.py --s-dist normal --s-mean 0.2 --s-var 0.01
#
# ε ~ Normal(mean=0, std=0.05):
#   python equilibria_explored.py --eps-dist normal --eps-mean 0 --eps-std 0.05
#
# Both s and ε from normal distributions:
#   python equilibria_explored.py --s-dist normal --s-mean 0.1 --s-var 0.0025 \
#       --eps-dist normal --eps-mean 0 --eps-var 0.01
#
# -----------------------------------------------------------------------------
# 3. UNIFORM distribution for s and/or ε
# -----------------------------------------------------------------------------
# Sample s_i and ε_ij uniformly from [low, high].
#
# s ~ Uniform[-0.1, 0.2]:
#   python equilibria_explored.py --s-dist uniform --s-low -0.1 --s-high 0.2
#
# ε ~ Uniform[-0.05, 0.05]:
#   python equilibria_explored.py --eps-dist uniform --eps-low -0.05 --eps-high 0.05
#
# Both uniform:
#   python equilibria_explored.py --s-dist uniform --s-low 0 --s-high 0.2 \
#       --eps-dist uniform --eps-low -0.1 --eps-high 0.1
#
# -----------------------------------------------------------------------------
# 4. CUSTOM Σ from file (override s/eps)
# -----------------------------------------------------------------------------
# Provide your own n×n matrix. Supports .npy (NumPy binary) or .txt (space/comma
# separated). Matrix must match --n-loci (e.g. n=2 → 2×2, n=3 → 3×3).
#
# Create a 2×2 Sigma and save as .npy:
#   import numpy as np
#   Sigma = np.array([[0.1, 0.02], [0.02, 0.15]])  # diagonal=s_i, off-diag=ε_ij/2
#   np.save("my_sigma.npy", Sigma)
#
# Run with it:
#   python equilibria_explored.py --n-loci 2 --sigma-file my_sigma.npy
#
# Or use a .txt file (one row per line, space or comma separated):
#   echo "0.1 0.02" > sigma.txt
#   echo "0.02 0.15" >> sigma.txt
#   python equilibria_explored.py --n-loci 2 --sigma-file sigma.txt
#
# -----------------------------------------------------------------------------
# 5. MIXED: fixed s, distributed ε (or vice versa)
# -----------------------------------------------------------------------------
#   python equilibria_explored.py --s-fixed 0.1 --eps-dist normal --eps-mean 0 --eps-std 0.03
#   python equilibria_explored.py --s-dist uniform --s-low 0.05 --s-high 0.15 --eps-fixed 0
#
# -----------------------------------------------------------------------------
# 6. MUTATION RATES u and v
# -----------------------------------------------------------------------------
# u = rate 0→1, v = rate 1→0. Default: u = v = 1e-4.
#
#   python equilibria_explored.py --u 1e-5 --v 1e-5
#   python equilibria_explored.py --u 2e-4 --v 1e-4
#
# -----------------------------------------------------------------------------
# 7. GRID SIZE and LOCI
# -----------------------------------------------------------------------------
# --n-loci: number of loci (k = 2^n genotypes). --N: population size (grid denominator).
# Lattice size = C(N+k-1, k-1). For large n, use smaller N.
#
#   python equilibria_explored.py --n-loci 2 --N 20
#   python equilibria_explored.py --n-loci 3 --N 12
#
# -----------------------------------------------------------------------------
# 8. SAVE RESULTS and PROGRESS
# -----------------------------------------------------------------------------
#   python equilibria_explored.py --out results.npz
#   python equilibria_explored.py --progress-every 10000
#
# -----------------------------------------------------------------------------
# 9. REPRODUCIBILITY
# -----------------------------------------------------------------------------
#   python equilibria_explored.py --seed 123
#
# -----------------------------------------------------------------------------
# 10. FULL EXAMPLE
# -----------------------------------------------------------------------------
#   python equilibria_explored.py \
#       --n-loci 2 \
#       --N 15 \
#       --s-dist normal --s-mean 0.1 --s-var 0.01 \
#       --eps-dist normal --eps-mean 0 --eps-var 0.005 \
#       --u 1e-4 --v 1e-4 \
#       --seed 42 \
#       --out run_n2_N15.npz
#
# =============================================================================
# MODEL CODE: create custom Σ and save for --sigma-file (commented out)
# =============================================================================
#
# import numpy as np
#
# # --- Option A: save as .npy (NumPy binary) ---
# # n=2: 2×2 matrix. Diagonal = s_i, off-diagonal = ε_ij/2
# Sigma = np.array([
#     [0.1,  0.02],   # s_0=0.1, ε_01/2=0.02
#     [0.02, 0.15],   # ε_10/2=0.02, s_1=0.15
# ])
# np.save("my_sigma.npy", Sigma)
# # Then: python equilibria_explored.py --n-loci 2 --sigma-file my_sigma.npy
#
# # --- Option B: save as .txt (plain text) ---
# np.savetxt("my_sigma.txt", Sigma, fmt="%.6f")
# # Then: python equilibria_explored.py --n-loci 2 --sigma-file my_sigma.txt
#
# # --- Option C: 3×3 example ---
# # Sigma = np.array([
# #     [0.1,  0.01, 0.02],
# #     [0.01, 0.12, 0.015],
# #     [0.02, 0.015, 0.08],
# # ])
# # np.save("sigma_3x3.npy", Sigma)
# # Then: python equilibria_explored.py --n-loci 3 --sigma-file sigma_3x3.npy
#
# =============================================================================
# CONFIG FILE: run with --config (YAML)
# =============================================================================
#
#   python equilibria_explored.py --config equilibria_config_example.yaml
#   python equilibria_explored.py --config equilibria_config_sigma_direct.yaml
#   python equilibria_explored.py --config equilibria_config_normal_dist.yaml
