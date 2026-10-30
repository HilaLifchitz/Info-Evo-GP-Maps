#!/usr/bin/env python3

"""
Two-locus Wright-Fisher Markov chain utilities (independent loci).

This module leverages the single-locus `WrightFisherMarkovChain` from `Markov_1_locus.py`
to construct and analyze a two-locus process under the assumption that the loci evolve
independently (no epistasis or linkage used here). The joint distribution is thus the
outer product of identical marginals.
"""

from __future__ import annotations

import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import pandas as pd
from math import comb
from scipy.stats import multinomial as sp_multinomial
from scipy.sparse.linalg import LinearOperator, eigs
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

from Markov_1_locus import WrightFisherMarkovChain


##########################################
# Evolution of two independent loci
##########################################

class MarkovTwoLoci:
    """
    Two-locus wrapper using the single-locus Wright-Fisher Markov chain.

    Assumes both loci share N and μ, but can have distinct s1 and s2, and evolve independently.
    Provides:
    - Access to the single-locus transition matrix P and stationary distribution π
    - Evolution of a marginal distribution via repeated applications of P
    - Construction of a joint 2D distribution as outer product when independence holds
    - Plotting a 2D heatmap given a marginal π (outer product π ⊗ π)
    """

    def __init__(self, N: int, mu: float, s1: float, s2: float, quiet: bool = True):
        self.N = int(N)
        self.mu = float(mu)
        self.s1 = float(s1)
        self.s2 = float(s2)
        self.quiet = quiet

        self.chain1 = WrightFisherMarkovChain(N=self.N, s=self.s1, mu=self.mu, quiet=self.quiet)
        self.chain2 = WrightFisherMarkovChain(N=self.N, s=self.s2, mu=self.mu, quiet=self.quiet)
        self.P1 = None
        self.P2 = None
        self.pi1_star = None
        self.pi2_star = None

    def construct_transition_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        if self.P1 is None:
            self.P1 = self.chain1.construct_transition_matrix()
        if self.P2 is None:
            self.P2 = self.chain2.construct_transition_matrix()
        return self.P1, self.P2

    def stationary_distributions(self, method: str = 'direct') -> tuple[np.ndarray, np.ndarray]:
        if self.pi1_star is None or self.P1 is None:
            self.construct_transition_matrices()
            self.pi1_star = np.asarray(self.chain1.compute_stationary_distribution(method=method), dtype=float)
        if self.pi2_star is None or self.P2 is None:
            self.construct_transition_matrices()
            self.pi2_star = np.asarray(self.chain2.compute_stationary_distribution(method=method), dtype=float)
        return self.pi1_star, self.pi2_star

    # def evolve_marginals(self, pi1_0: np.ndarray, pi2_0: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray]:
    #     """Evolve both marginals independently for T steps: π_T = P^T π_0 for each locus."""
    #     P1, P2 = self.construct_transition_matrices()
    #     pi1 = np.asarray(pi1_0, dtype=float)
    #     pi2 = np.asarray(pi2_0, dtype=float)
    #     pi1 = pi1 / np.sum(pi1)
    #     pi2 = pi2 / np.sum(pi2)
    #     for _ in range(int(T)):
    #         pi1 = P1 @ pi1
    #         pi2 = P2 @ pi2
    #         pi1 = pi1 / np.sum(pi1)
    #         pi2 = pi2 / np.sum(pi2)
    #     return pi1, pi2

    def joint_from_marginals(self, pi1: np.ndarray, pi2: np.ndarray) -> np.ndarray:
        """Return joint distribution under independence: Π = π1 ⊗ π2 (outer product)."""
        pi1 = np.asarray(pi1, dtype=float)
        pi2 = np.asarray(pi2, dtype=float)
        pi1 = pi1 / np.sum(pi1)
        pi2 = pi2 / np.sum(pi2)
        return np.outer(pi1, pi2)

    def plot_joint_heatmap(self, pi1: np.ndarray, pi2: np.ndarray, title: str | None = None) -> plt.Figure:
        """Plot a 2D heatmap of Π = π1 ⊗ π2 (independent loci)."""
        sns.set_style("white")
        sns.set_palette("husl")

        joint = self.joint_from_marginals(pi1, pi2)
        states = np.arange(self.N + 1)

        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        im = ax.imshow(joint, cmap='Blues', origin='lower', aspect='equal', interpolation='nearest')
        ax.set_xlabel('Locus 1: number of "1" alleles')
        ax.set_ylabel('Locus 2: number of "1" alleles')
        if title is None:
            title = (
                f'Two-Locus Joint Distribution (independent)\n'
                f'N={self.N}, μ={self.mu}, s1={self.s1}, s2={self.s2}'
            )
        ax.set_title(title)

        # ticks
        tick_positions = np.arange(0, self.N + 1, max(1, self.N // 5))
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_positions)
        ax.set_yticklabels(tick_positions)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Joint probability')
        plt.tight_layout()
        return fig

    def plot_joint_3d(self, pi1: np.ndarray, pi2: np.ndarray, title: str | None = None,
                      elev: float = 30.0, azim: float = -60.0) -> plt.Figure:
        """
        Plot a 3D histogram (bar3d) of Π = π1 ⊗ π2.

        Parameters:
        -----------
        pi1, pi2 : np.ndarray
            Marginal distributions for locus 1 and locus 2 (length N+1 each)
        title : str | None
            Optional custom title. If None, a default title is constructed
        elev, azim : float
            Elevation and azimuth angles for the 3D view
        """
        joint = self.joint_from_marginals(pi1, pi2)
        states = np.arange(self.N + 1)
        X, Y = np.meshgrid(states, states)
        Xf = X.ravel()
        Yf = Y.ravel()
        Zf = joint.ravel()

        # Bar sizes
        dx = np.full_like(Xf, 0.8, dtype=float)
        dy = np.full_like(Yf, 0.8, dtype=float)
        z0 = np.zeros_like(Zf)

        # Colors based on height
        max_z = np.max(Zf) if np.max(Zf) > 0 else 1.0
        colors = cm.Blues(0.3 + 0.7 * (Zf / max_z))

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.bar3d(Xf - 0.4, Yf - 0.4, z0, dx, dy, Zf, shade=True, color=colors, alpha=0.95)

        ax.set_xlabel('Locus 1: number of "1" alleles')
        ax.set_ylabel('Locus 2: number of "1" alleles')
        ax.set_zlabel('Joint probability')

        if title is None:
            title = (
                f'Two-Locus Joint Distribution (3D)\n'
                f'N={self.N}, μ={self.mu}, s1={self.s1}, s2={self.s2}'
            )
        ax.set_title(title)

        ax.view_init(elev=elev, azim=azim)
        plt.tight_layout()
        return fig


def compute_phi_star(N: int, mu: float, method: str = 'direct') -> np.ndarray:
    """Return 1D stationary distribution at s=0 for given N, mu."""
    chain = WrightFisherMarkovChain(N=int(N), s=0.0, mu=float(mu), quiet=True)
    chain.construct_transition_matrix()
    phi = np.asarray(chain.compute_stationary_distribution(method=method), dtype=float)
    phi = phi / np.sum(phi)
    return phi


def evolve_p_list(P_list: list[np.ndarray], p0_list: list[np.ndarray], T: int) -> list[np.ndarray]:
    """
    Generalized evolution for multiple independent loci.
    For each k, returns p_k(T) = P_k^T @ p_k(0), normalized.

    Parameters:
    -----------
    P_list : list of (N+1)x(N+1) transition matrices
    p0_list : list of initial marginal distributions (length N+1 each)
    T : int generations
    """
    assert len(P_list) == len(p0_list), "P_list and p0_list must have same length"
    pT_list: list[np.ndarray] = []
    for P, p0 in zip(P_list, p0_list):
        p = np.asarray(p0, dtype=float)
        p = p / np.sum(p)
        PT = np.linalg.matrix_power(np.asarray(P, dtype=float), int(T))
        pT = PT @ p
        pT = pT / np.sum(pT)
        pT_list.append(pT)
    return pT_list


def evolve_p1_p2(P1: np.ndarray, P2: np.ndarray, p1_0: np.ndarray, p2_0: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible 2-locus wrapper around evolve_p_list."""
    p1_T, p2_T = evolve_p_list([P1, P2], [p1_0, p2_0], T)
    return p1_T, p2_T


def marginals_from_s(N: int, mu: float, s1: float, s2: float, T: int, method: str = 'direct') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function:
    - Builds P1 (s=s1) and P2 (s=s2) for shared (N, mu)
    - Uses compute_phi_star(N, mu) as initial condition for both marginals
    - Returns p1(T), p2(T) along with P1, P2
    """
    # Transition matrices for the two loci
    chain1 = WrightFisherMarkovChain(N=int(N), s=float(s1), mu=float(mu), quiet=True)
    chain2 = WrightFisherMarkovChain(N=int(N), s=float(s2), mu=float(mu), quiet=True)
    P1 = chain1.construct_transition_matrix()
    P2 = chain2.construct_transition_matrix()

    # Initial condition: neutral stationary distribution
    phi = compute_phi_star(N, mu, method=method)

    # Evolve to time T
    p1_T, p2_T = evolve_p1_p2(P1, P2, phi, phi, T)
    return p1_T, p2_T



# -----------------------------
# Simplex grid enumeration
# -----------------------------

def count_simplex_points(N: int, dim: int) -> int:
    # valid for N = grid size and dim = dimension of the simplex
    return comb(N +dim , dim)


def _compositions(n: int, k: int):
    """
    Yield all k-tuples of nonnegative integers summing to n (integer compositions with k parts).
    This is a generator with no filtering (stars-and-bars recursion).
    Mathematicaly, for any n >= 0 and k >= 2,
              C(n, k) = ⋃_{i=0}^n { (i, r1, ..., r_{k-1}) : (r1, ..., r_{k-1}) ∈ C(n - i, k - 1) }.
    """
    if k == 1:
        #Yield pauses a function's execution and returns a value temporarily
        # any function that contains yield is a generator function. open this as touples_list = list(_compositions(n, k))
        yield (n,) 
        return
    for i in range(n + 1):
        # (i,) is a one-element tuple; 'rest' is one of the (k-1)-tuples from the recursive call.
        # Tuple concatenation (i,) + rest prepends i to each smaller composition,
        # building full k-tuples like (i, r1, r2, ...).
        for rest in _compositions(n - i, k - 1):
            yield (i,) + rest


def simplex_grid_points(N: int, dim: int):
    """
    Iterate over the gridded simplex of dimension `dim` with grid size 1/N.

    - dim=1 → points on [0,1] as pairs (x, 1-x) with x ∈ {0,1/N,...,1}
    - dim=2 → triples (x,y,z) with x+y+z=1 on the 1/N grid
    - dim=3 → quadruples (A,B,C,D) with A+B+C+D=1 on the 1/N grid

    Yields tuples of length dim+1 with components in {0,1/N,...,1} summing to 1.
    """
    k = int(dim) + 1 # cause we have k+1 coordinates in a k-dimensional simplex

    for comp in _compositions(int(N), k): # comp = e.g. (2, 1, 3) 
        yield tuple(c / float(N) for c in comp)  # -> (2/N, 1/N, 3/N), tuple(...)builds a tuple from any iterable.

# shortcuts functions
def simplex2_grid(N: int): # dim=2, returning a list of tuples
    """Convenience wrapper for dim=2 (triples)."""
    return list(simplex_grid_points(N, dim=2))


def simplex3_grid(N: int): # dim=3, returning a list of tuples
    """Convenience wrapper for dim=3 (quadruples)."""
    return list(simplex_grid_points(N, dim=3))

#brut force stupid sanity check for the number of points in the 1/N grid simplex
def verify_simplex_triple_loop(N: int): # for dimension dim=3
    points = []
    for i in range(N + 1):          
        for j in range(N + 1):
            for l in range(N + 1):      
                for k in range(N + 1):  
                    if i + j + k +l== N:  # ⇐ equivalent to x+y+z == 1 without float error
                        points.append((i / N, j / N, k / N))
   # print(f"Found {len(points)} points; expected {math.comb(N+2, 2)}")
    return len(points)


#########################################################################################################################
# Building measure from P --> on X and on G
#########################################################################################################################
#########################################################################################################################   
# On X (simplex space X**3)
#########################################################################################################################

def build_simplex3_measure_from_marginals(p1: np.ndarray, p2: np.ndarray) -> dict[tuple[float, float, float, float], float]:
    """
    Construct a gridded probability measure on the 3D simplex (A,B,C,D) from independent marginals p1, p2.
    - in practice, this is a sub-simplex of dimension 2 embedded inside of a simplex of dimension 3
    Interpretation (genotypes):
      - A ≡ [0,0]
      - B ≡ [0,1]
      - C ≡ [1,0]
      - D ≡ [1,1]

    For each allele-frequency pair (x, y) with x ∈ {i/N} from p1 and y ∈ {j/N} from p2,
    map to simplex coordinates:
      A = (1 - x) * (1 - y)
      B = (1 - x) * y
      C = x * (1 - y)
      D = x * y

    The mass assigned to this simplex point is the product measure p1(x) * p2(y).

    Returns:
      dict mapping (A,B,C,D) → mass at that grid point. Grid granularity is 1/N where N = len(p1)-1 = len(p2)-1.
      # NOTICE: this dictionary is not over the entire simplex but only for points that we could reach via i,j in [0,1/N,...1]**2
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p1 = p1 / np.sum(p1)
    p2 = p2 / np.sum(p2)
    N1 = len(p1) - 1
    N2 = len(p2) - 1
    assert N1 == N2, "p1 and p2 must be over the same N grid"
    N = N1

    measure: dict[tuple[float, float, float, float], float] = {}
    for i in range(N + 1):
        x = i / N
        for j in range(N + 1):
            y = j / N
            # simplex coords
            A = (1.0 - x) * (1.0 - y)  # [0,0]
            B = (1.0 - x) * y          # [0,1]
            C = x * (1.0 - y)          # [1,0]
            D = x * y                  # [1,1]

            mass = p1[i] * p2[j]
            key = (A, B, C, D)
            measure[key] = measure.get(key, 0.0) + mass

    return measure


 
# This now completes the measure from above to be fully defines over all of the simplex by adding to the dictionary points with measure 0
def ensure_simplex3_grid_keys(p1: np.ndarray, p2: np.ndarray) -> dict[tuple[float, float, float, float], float]:
    """
    Build the simplex measure using p1 and p2 (independent marginals), then ensure that
    every 1/N*N-grid simplex point is present with mass 0.0 if it wasn't generated.
    """
    measure = build_simplex3_measure_from_marginals(p1, p2)
    for pt in simplex3_grid(N*N):
        if pt not in measure:
            measure[pt] = 0.0
    return measure


#########################################################################################################################
# On G
#########################################################################################################################

#using: Ex := E(P_1) ; Ey := E(P_2)
#     (1.0 - Ex) * (1.0 - Ey),  # [0,0]
#     (1.0 - Ex) * Ey,          # [0,1]
#     Ex * (1.0 - Ey),          # [1,0]
#     Ex * Ey,                  # [1,1] 
def genotype_distribution_from_marginals(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Compute genotype distribution G = [P([0,0]), P([0,1]), P([1,0]), P([1,1])]
    from independent marginals over allele counts p1, p2 on grid {0,1/N,...,1}.
    Uses means Ex, Ey (independence implies E[xy] = Ex*Ey).
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    N1 = len(p1) - 1
    N2 = len(p2) - 1
    assert N1 == N2 and N1 >= 0, "p1 and p2 must be over the same N grid"
    N = float(N1)
    x_vals = np.arange(N1 + 1, dtype=float) / N
    y_vals = x_vals  # same grid length
    Ex = float(np.sum(x_vals * p1))
    Ey = float(np.sum(y_vals * p2))
    G = np.array([
        (1.0 - Ex) * (1.0 - Ey),  # [0,0]
        (1.0 - Ex) * Ey,          # [0,1]
        Ex * (1.0 - Ey),          # [1,0]
        Ex * Ey,                  # [1,1]
    ], dtype=float)
    # numerical cleanup
    G = np.maximum(G, 0.0)
    s = G.sum()
    if s > 0:
        G = G / s
    else:
        G[:] = 0.25
    return G

# using law G =  E(X)
def genotype_distribution_from_simplex(measure: dict[tuple[float, float, float, float], float]) -> np.ndarray:
    """
    Compute genotype distribution G by integrating over the simplex measure:
      G = E[(A,B,C,D)] = sum_{(A,B,C,D)} (A,B,C,D) * mass / sum mass.
    """
    if not measure:
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    total = float(sum(measure.values()))
    if total <= 0:
        return np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    acc = np.zeros(4, dtype=float)
    for (A, B, C, D), w in measure.items():
        acc[0] += A * w
        acc[1] += B * w
        acc[2] += C * w
        acc[3] += D * w
    G = acc / total
    # numerical cleanup
    G = np.maximum(G, 0.0)
    s = G.sum()
    if s > 0:
        G = G / s
    else:
        G[:] = 0.25
    return G

# comparing both ways
def G_sanity_check(p1: np.ndarray, p2: np.ndarray) -> dict:
    """
    Compare G computed from P (marginals) vs G computed by integrating over X (simplex measure).

    - If measure is None, it is built from p1,p2 via build_simplex3_measure_from_marginals.
    - Returns a dict with G_from_P, G_from_X, and norms of their difference.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    
    measure = build_simplex3_measure_from_marginals(p1, p2)
    G_P = genotype_distribution_from_marginals(p1, p2)
    G_X = genotype_distribution_from_simplex(measure)
    diff = G_P - G_X
    return {
        'G_from_P': G_P,
        'G_from_X': G_X,
        'L1_diff': float(np.sum(np.abs(diff))),
        'L2_diff': float(np.linalg.norm(diff)),
        'max_diff': float(np.max(np.abs(diff))),
    }

#########################################################################################################################
# KL divergence functions
#########################################################################################################################
################ OVER P AND G
def kl_divergence(p: np.ndarray, q: np.ndarray) -> float: # over innocent 1D arrays of probabilities = P**1=X**1=G**(4)
    eps = 1e-16
    p_ = p + eps
    q_ = q + eps
    p_ = p_ / np.sum(p_)
    q_ = q_ / np.sum(q_)
    return float(np.sum(p_ * (np.log(p_) - np.log(q_))) / math.log(2.0))  # bits

#    ONLY WORKS WITH VERY SMALL N!! N=40 IS ALREADY TOO BIG :((((
# KL divergence on the 3D simplex
# def kl_on_simplex3_from_measures(measure_p: dict[tuple[float, float, float, float], float],
#                                  measure_q: dict[tuple[float, float, float, float], float],
#                                  N: int,
#                                  eps: float = 1e-12,
#                                  base: float = 2.0) -> float:
#     """

#     KL divergence KL(P || Q) on the 3-simplex grid (quadruples (A,B,C,D) summing to 1 with step 1/N).
#     - measure_p, measure_q: dict mapping (A,B,C,D) → mass; missing keys are treated as 0.
#     - Adds epsilon smoothing and renormalizes both distributions before KL.
#     - Returns KL in bits by default (base=2).
#     """
#     keys = simplex3_grid(N*N) # list of quadruples (A,B,C,D)
#     p_vec = np.array([float(measure_p.get(k, 0.0)) for k in keys], dtype=float)
#     q_vec = np.array([float(measure_q.get(k, 0.0)) for k in keys], dtype=float)

#     p_vec = p_vec + eps
#     q_vec = q_vec + eps
#     p_vec = p_vec / np.sum(p_vec)
#     q_vec = q_vec / np.sum(q_vec)

#     log_base = math.log(base)
#     kl = np.sum(p_vec * (np.log(p_vec) - np.log(q_vec))) / log_base
#     return float(kl)

################ OVER X

####  THIS WORKS: 2025-10-13
def kl_on_same_support_measures(measure_p: dict, measure_q: dict,
                                eps: float = 1e-12,
                                base: float = 2.0,
                                strict: bool = True) -> float:
    """
    KL(P || Q) over the existing support only (no grid expansion).

    - Assumes measure_p and measure_q are dicts mapping points (e.g., (A,B,C,D)) to masses.
    - If strict=True, require identical key sets; otherwise missing keys default to 0.
    - Adds epsilon smoothing, renormalizes, and returns KL in bits by default (base=2).
    """
    keys_p = set(measure_p.keys())
    keys_q = set(measure_q.keys())

    if strict and keys_p != keys_q:
        missing_in_q = list(keys_p - keys_q)[:5]
        missing_in_p = list(keys_q - keys_p)[:5]
        raise ValueError(f"Measures do not share identical support. Examples missing in Q: {missing_in_q}; missing in P: {missing_in_p}")

    # Use P's keys order deterministically
    keys = list(measure_p.keys())
    p_vec = np.array([float(measure_p.get(k, 0.0)) for k in keys], dtype=float)
    q_vec = np.array([float(measure_q.get(k, 0.0)) for k in keys], dtype=float)

    p_vec = p_vec + eps
    q_vec = q_vec + eps
    p_vec = p_vec / np.sum(p_vec)
    q_vec = q_vec / np.sum(q_vec)

    log_base = math.log(base)
    kl = np.sum(p_vec * (np.log(p_vec) - np.log(q_vec))) / log_base
    return float(kl)


#########################################################################################################################
# Information theory functions over P**2
#########################################################################################################################

def D(T,N,mu,s1,s2): #D(P**2(T)) the information stored by natural selection in the allele frequency space after T generations  
    p1_T, p2_T = marginals_from_s(N, mu, s1, s2, T)
    phi = compute_phi_star(N, mu) # phi is the stationary distribution of the neutral locus which is here always p1_0 = p2_0
    return kl_divergence(p1_T,phi)+kl_divergence(p2_T,phi) # we get to add the measures cause of their independece


def I(T,N,mu,s1,s2): #I(T) the free-fitness after T generations, I stands for Iwasa  
    p1_T, p2_T = marginals_from_s(N, mu, s1, s2, T)
    model = MarkovTwoLoci(N=N, mu=mu, s1=s1, s2=s2, quiet=True)
    psi_1, psi_2 = model.stationary_distributions(method='direct') # psi_1 and psi_2 are the stationary distributions of the two loci-- to which we are going to 
    return kl_divergence(p1_T,psi_1)+kl_divergence(p2_T,psi_2) # we get to add the measures cause of their independece


#######################################################

def plot_simplex3_measure_tetrahedron(measure: dict[tuple[float, float, float, float], float],
                                    title: str | None = None,
                                    figsize: tuple[int, int] = (10, 8),
                                    min_mass: float = 1e-6,
                                    cmap: str = 'viridis'):
    """
    Plot a probability measure over the 3-simplex (quadruples (A,B,C,D)) inside a tetrahedron.

    Vertex mapping (barycentric → 3D):
      - [1,0,0,0] → genotype [0,0]
      - [0,1,0,0] → genotype [0,1]
      - [0,0,1,0] → genotype [1,0]
      - [0,0,0,1] → genotype [1,1]

    Parameters:
      - measure: dict mapping (A,B,C,D) to mass (not necessarily normalized)
      - title: optional plot title
      - figsize: figure size
      - min_mass: filter out points with mass < min_mass
      - cmap: colormap for mass coloring
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Regular tetrahedron vertices (rows)
    V = np.array([
        [ 1.0,  1.0,  1.0],   # vertex for [1,0,0,0] ≡ [0,0]
        [-1.0, -1.0,  1.0],   # vertex for [0,1,0,0] ≡ [0,1]
        [-1.0,  1.0, -1.0],   # vertex for [0,0,1,0] ≡ [1,0]
        [ 1.0, -1.0, -1.0],   # vertex for [0,0,0,1] ≡ [1,1]
    ], dtype=float)

    def project(abcd: tuple[float, float, float, float]) -> np.ndarray:
        a, b, c, d = abcd
        return a * V[0] + b * V[1] + c * V[2] + d * V[3]

    # Collect points and weights
    items = list(measure.items())
    if min_mass > 0.0:
        items = [(k, v) for (k, v) in items if v >= min_mass]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    if not items:
        ax.set_axis_off()
        ax.set_title(title or 'Empty measure on simplex')
        return fig

    keys, weights = zip(*items)
    pts3d = np.array([project(k) for k in keys], dtype=float)
    weights = np.array(weights, dtype=float)

    # Normalize weights for color/size scaling
    wsum = weights.sum()
    if wsum > 0:
        wnorm = weights / wsum
    else:
        wnorm = np.ones_like(weights) / len(weights)

    # Draw points with size ∝ weight and color by weight
    sizes = 5.0 + 200.0 * wnorm  # tune as needed
    sc = ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], c=wnorm, s=sizes, cmap=cmap, alpha=0.8)

    # Draw tetrahedron faces and edges
    faces = [[V[0], V[1], V[2]], [V[0], V[1], V[3]], [V[0], V[2], V[3]], [V[1], V[2], V[3]]]
    poly3d = Poly3DCollection(faces, alpha=0.05, facecolor='gray', edgecolor='k')
    ax.add_collection3d(poly3d)
    for i in range(4):
        for j in range(i + 1, 4):
            ax.plot([V[i, 0], V[j, 0]], [V[i, 1], V[j, 1]], [V[i, 2], V[j, 2]], 'k-', lw=0.8)

    # Vertex labels with genotype mapping
    vertex_labels = ['[0,0]', '[0,1]', '[1,0]', '[1,1]']
    for i, label in enumerate(vertex_labels):
        ax.text(V[i, 0], V[i, 1], V[i, 2], label, color='black', fontsize=12,
                fontweight='bold', ha='center', va='center')

    ax.set_axis_off()
    ax.set_title(title or 'Measure on 3D Simplex (Tetrahedron)')

    # Colorbar for weight
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Normalized mass')

    plt.tight_layout()
    return fig


#########################################################################################################################
# Joint P plotting (3D histogram)
#########################################################################################################################

def plot_joint_P_3d(P: np.ndarray,
                    title: str | None = None,
                    elev: float = 30.0,
                    azim: float = -60.0) -> plt.Figure:
    """
    Plot a 3D histogram (bar3d) of a joint allele-frequency measure P over the (p1, p2) grid.
    - P: (N+1, N+1) matrix whose entries sum to 1 (will be normalized defensively).
    - Axes show counts 0..N for each locus; bars heights are joint probabilities.
    """
    P = np.asarray(P, dtype=float)
    assert P.ndim == 2 and P.shape[0] == P.shape[1], "P must be a square (N+1) x (N+1) matrix"

    # Defensive cleanup and normalization
    P = np.maximum(P, 0.0)
    s = float(P.sum())
    if s > 0.0:
        P = P / s
    else:
        n = P.shape[0]
        P = np.full((n, n), 1.0 / (n * n), dtype=float)

    N = P.shape[0] - 1
    states = np.arange(N + 1)
    X, Y = np.meshgrid(states, states)
    Xf = X.ravel()
    Yf = Y.ravel()
    Zf = P.ravel()

    # Bar sizes
    dx = np.full_like(Xf, 0.8, dtype=float)
    dy = np.full_like(Yf, 0.8, dtype=float)
    z0 = np.zeros_like(Zf)

    # Colors based on height
    max_z = float(np.max(Zf)) if float(np.max(Zf)) > 0.0 else 1.0
    colors = cm.Blues(0.3 + 0.7 * (Zf / max_z))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(Xf - 0.4, Yf - 0.4, z0, dx, dy, Zf, shade=True, color=colors, alpha=0.95)

    ax.set_xlabel('Locus 1: number of "1" alleles')
    ax.set_ylabel('Locus 2: number of "1" alleles')
    ax.set_zlabel('Joint probability')

    if title is None:
        title = f'Joint Distribution P (3D)\nN={N}'
    ax.set_title(title)

    # Ticks
    tick_positions = np.arange(0, N + 1, max(1, N // 5 or 1))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_zticks(np.linspace(0, max_z, 5))

    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    return fig


#########################################################################################################################
# Fast multi-T KL computation over X, P, G using LD model
#########################################################################################################################

def compute_D_I_series_LD(N: int,
                          s1: float,
                          s2: float,
                          r: float,
                          mu: float,
                          Ts: list[int] | np.ndarray,
                          selection_mode: str = "multiplicative",
                          modes_k: int = 30,
                          modes_tol: float = 1e-12,
                          modes_maxiter: int = 2000,
                          save_csv_path: str | None = None) -> pd.DataFrame:
    """
    Evolve the LD process from φ* (uniform over simplex states) at requested times Ts using spectral acceleration,
    then compute D(t) and I(t) (KL divergences) in three spaces X, P, and G:
      - D_X(t) = KL(X(t) || φ*)
      - D_P(t) = KL(P(t) || P(φ*))
      - D_G(t) = KL(G(t) || G(φ*))
      - I_X(t) = KL(X(t) || ψ*)
      - I_P(t) = KL(P(t) || P(ψ*))
      - I_G(t) = KL(G(t) || G(ψ*))

    Returns a tidy DataFrame with columns:
      T, D_X, D_P, D_G, I_X, I_P, I_G

    Notes:
      - X(t) is a probability vector over simplex grid states ordered as simplex3_grid(N).
      - P(t) is the pushforward of X(t) to the (p1, p2) grid (flattened in KL).
      - G(t) is the expectation of X(t) on simplex vertices (4-vector).
      - ψ* = power_stationary() is the stationary distribution in X under selection.
      - φ* is uniform over simplex states (neutral baseline in X).
    """
    # Model and baselines
    ld = TwoLocusLD(N=N, s1=s1, s2=s2, r=r, mu=mu, selection_mode=selection_mode)
    S = count_simplex_points(N, 3)
    phi_star = np.ones(S, dtype=float) / float(S)
    psi_star = ld.power_stationary()

    # Prepare spectral modes once; stream-evaluate v_T without storing all X_T
    Ts_arr = np.array(sorted(int(t) for t in Ts), dtype=int)
    lam, U = ld.leading_modes(k=modes_k, tol=modes_tol, maxiter=modes_maxiter, verbose=False)
    # Precompute least-squares coefficients for v0 in span(U)
    c, *_ = np.linalg.lstsq(U, phi_star, rcond=None)

    # Baselines in P and G
    P_phi = ld.pushforward_X_to_P(phi_star, normalize=True).ravel()
    P_psi = ld.pushforward_X_to_P(psi_star, normalize=True).ravel()
    G_phi = ld.pushforward_X_to_G(phi_star, normalize=True)
    G_psi = ld.pushforward_X_to_G(psi_star, normalize=True)

    # Collect KLs
    D_X, D_P, D_G = [], [], []
    I_X, I_P, I_G = [], [], []
    for T in Ts_arr:
        # Spectral evaluation: v_T = U @ ((lam ** T) * c)
        amp = (lam ** int(T)) * c
        xT = U @ amp
        xT = np.maximum(xT, 0.0)
        s = float(xT.sum())
        if s == 0.0:
            xT = np.full_like(xT, 1.0 / len(xT))
        else:
            xT /= s
        # X
        D_X.append(kl_divergence(xT, phi_star))
        I_X.append(kl_divergence(xT, psi_star))
        # P
        PT = ld.pushforward_X_to_P(xT, normalize=True).ravel()
        D_P.append(kl_divergence(PT, P_phi))
        I_P.append(kl_divergence(PT, P_psi))
        # G
        GT = ld.pushforward_X_to_G(xT, normalize=True)
        D_G.append(kl_divergence(GT, G_phi))
        I_G.append(kl_divergence(GT, G_psi))

    df = pd.DataFrame({
        "T": Ts_arr,
        "D_X": D_X, "D_P": D_P, "D_G": D_G,
        "I_X": I_X, "I_P": I_P, "I_G": I_G,
    })
    if save_csv_path:
        try:
            df.to_csv(save_csv_path, index=False)
        except Exception as e:
            print(f"Warning: could not save CSV to '{save_csv_path}': {e}")
    return df


#########################################################################################################################
# Plotting from DataFrame (LD series) – outside any class
#########################################################################################################################

def plot_D_I_series_from_df(df: pd.DataFrame, title: str | None = None, r: float | None = None):
    """
    Plot D and I series from a DataFrame returned by compute_D_I_series_LD.
    Expects columns:
      - T
      - D_P, D_X, D_G
      - I_P, I_X, I_G
    Produces two panels:
      Left: D_KL(ψ(t) || φ*) with lines KL_P, KL_X, KL_G
      Right: D_KL(ψ(t) || ψ*) with lines KL_P, KL_X, KL_G
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

    # Left: D
    axes[0].plot(df["T"], df["D_P"], label='KL_P', linewidth=2, marker='o', color='tab:blue')
    axes[0].plot(df["T"], df["D_X"], label='KL_X', linewidth=2, color='tab:red')
    axes[0].plot(df["T"], df["D_G"], label='KL_G', linewidth=2, color='tab:green')
    axes[0].set_title('Accumulation of information\nD_KL(ψ(t) || φ*)')
    axes[0].set_xlabel('Generations (T)')
    axes[0].set_ylabel('KL divergence [bits]')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best')

    # Right: I
    axes[1].plot(df["T"], df["I_P"], label='KL_P', linewidth=2, marker='o', color='tab:blue')
    axes[1].plot(df["T"], df["I_X"], label='KL_X', linewidth=2, color='tab:red')
    axes[1].plot(df["T"], df["I_G"], label='KL_G', linewidth=2, color='tab:green')
    axes[1].set_title("Iwasa's free fitness\nD_KL(ψ(t) || ψ*)")
    axes[1].set_xlabel('Generations (T)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best')

    if r is not None:
        title = (f"{title}, r={r}" if title else f"r={r}")
    if title:
        fig.suptitle(title, y=0.96)
    fig.tight_layout()
    fig.subplots_adjust(top=0.86 if title else 0.92)
    return fig, axes


#########################################################################################################################
# Dual-axis plotting from DataFrame (LD series) – separate scale for G
#########################################################################################################################

def plot_D_I_series_dual_axis_from_df(df: pd.DataFrame, title: str | None = None, r: float | None = None):
    """
    Plot D and I series using two y-axes per panel:
      - Left y-axis: KL_P, KL_X
      - Right y-axis: KL_G (separate scale)
    Expects columns: T, D_P, D_X, D_G, I_P, I_X, I_G.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    # Colors consistent with other plots
    color_P = 'tab:blue'
    color_X = 'tab:red'
    color_G = 'tab:green'

    # Left panel: D
    axL = axes[0]
    axR = axL.twinx()
    axL.plot(df["T"], df["D_P"], label='KL_P', linewidth=2, marker='o', color=color_P)
    axL.plot(df["T"], df["D_X"], label='KL_X', linewidth=2, color=color_X)
    axR.plot(df["T"], df["D_G"], label='KL_G', linewidth=2, color=color_G)
    axL.set_title('Accumulation of information\nD_KL(ψ(t) || φ*)')
    axL.set_xlabel('Generations (T)')
    axL.set_ylabel('KL_P, KL_X [bits]')
    axR.set_ylabel('KL_G [bits]')
    axL.grid(True, alpha=0.3)
    # Legends: primary and secondary
    leg1 = axL.legend(loc='upper left')
    leg2 = axR.legend(loc='upper right')
    axL.add_artist(leg1)

    # Right panel: I
    axL = axes[1]
    axR = axL.twinx()
    axL.plot(df["T"], df["I_P"], label='KL_P', linewidth=2, marker='o', color=color_P)
    axL.plot(df["T"], df["I_X"], label='KL_X', linewidth=2, color=color_X)
    axR.plot(df["T"], df["I_G"], label='KL_G', linewidth=2, color=color_G)
    axL.set_title("Iwasa's free fitness\nD_KL(ψ(t) || ψ*)")
    axL.set_xlabel('Generations (T)')
    axL.set_ylabel('KL_P, KL_X [bits]')
    axR.set_ylabel('KL_G [bits]')
    axL.grid(True, alpha=0.3)
    leg1 = axL.legend(loc='upper left')
    leg2 = axR.legend(loc='upper right')
    axL.add_artist(leg1)

    if r is not None:
        title = (f"{title}, r={r}" if title else f"r={r}")
    if title:
        fig.suptitle(title, y=0.96)
    fig.tight_layout()
    fig.subplots_adjust(top=0.86 if title else 0.92)
    return fig, axes


# Backwards-compatible short alias
def plot_D_I_series_dual(df: pd.DataFrame, title: str | None = None, r: float | None = None):
    return plot_D_I_series_dual_axis_from_df(df, title=title, r=r)


#########################################################################################################################
# Side-by-side comparison of D-series for two different μ values
#########################################################################################################################

def plot_compare_D_series_two_dfs(df_left: pd.DataFrame,
                                  df_right: pd.DataFrame,
                                  mu_left: float,
                                  mu_right: float,
                                  title: str | None = None,
                                  r: float | None = None):
    """
    Plot D-series (KL_P, KL_X, KL_G vs T) for two DataFrames side-by-side.
    - df_left, df_right: DataFrames with columns T, D_P, D_X, D_G (from compute_D_I_series_LD).
    - mu_left, mu_right: μ values corresponding to each DataFrame (for subplot titles).
    - title: optional super-title; if provided and r is not None, appends ', r=...'.
    - r: optional recombination rate to include in the super-title.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

    color_P = 'tab:blue'
    color_X = 'tab:red'
    color_G = 'tab:green'

    # Determine a common y-limit for fair comparison
    max_y = 0.0
    for df in (df_left, df_right):
        if len(df) == 0:
            continue
        max_y = max(max_y,
                    float(np.max(df["D_P"])) if "D_P" in df else 0.0,
                    float(np.max(df["D_X"])) if "D_X" in df else 0.0,
                    float(np.max(df["D_G"])) if "D_G" in df else 0.0)
    if max_y <= 0.0:
        max_y = 1.0

    def plot_panel(ax, df, mu_value):
        ax.plot(df["T"], df["D_P"], label='KL_P', linewidth=2, marker='o', color=color_P)
        ax.plot(df["T"], df["D_X"], label='KL_X', linewidth=2, color=color_X)
        ax.plot(df["T"], df["D_G"], label='KL_G', linewidth=2, color=color_G)
        ax.set_xlabel('Generations (T)')
        ax.set_ylabel('KL divergence [bits]')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'μ={mu_value}')

    plot_panel(axes[0], df_left, mu_left)
    plot_panel(axes[1], df_right, mu_right)
    axes[0].legend(loc='best')

    # Apply common y-limit with small headroom
    axes[0].set_ylim(0.0, max_y * 1.05)

    # Super-title
    if r is not None:
        title = (f"{title}, r={r}" if title else f"r={r}")
    if title:
        fig.suptitle(title, y=0.96)
    fig.tight_layout()
    fig.subplots_adjust(top=0.86 if title else 0.92)
    return fig, axes


#########################################################################################################################
# Comparing convergence on P**2 and X**3 and G**(4) -- all correspond to n = # loci = 2
#########################################################################################################################

# I stands for Iwasa- free fitness 

def compute_I_convergence_with_plot(N: int,
                           mu: float,
                           s1: float,
                           s2: float,
                           T_max: int = 5000,
                           step: int = 50,
                           eps: float = 1e-12,
                           base: float = 2.0,
                           title: str | None = None) -> dict:
    """
    Measure convergence of p1_T, p2_T to p1_* , p2_* over time T in two spaces:
      - P (allele space):   KL_P(T) = KL(psi_1_T || psi_1_*) * KL(psi_2_T || psi_2_*)
      - X (simplex space):  KL_X(T) = KL(psi_T || psi_*)
      - G (genotype space): KL_G(T) = KL(G(psi_T) || G(psi_*))

    Returns a dict with arrays and plots P, X, G in three panels.
    """
    model = MarkovTwoLoci(N=N, mu=mu, s1=s1, s2=s2, quiet=True)
    psi_1_star, psi_2_star = model.stationary_distributions(method='direct')

    Ts = list(range(0, int(T_max) + 1, int(step)))
    KL_P, KL_X, KL_G = [], [], []

    psi_star = build_simplex3_measure_from_marginals(psi_1_star, psi_2_star)
    G_psi_star = genotype_distribution_from_marginals(psi_1_star, psi_2_star)

    for T in Ts:
        psi_1_T, psi_2_T = marginals_from_s(N, mu, s1, s2, T)
        # P
        KL_P.append(kl_divergence(psi_1_T, psi_1_star) + kl_divergence(psi_2_T, psi_2_star))
        # X
        psi_T = build_simplex3_measure_from_marginals(psi_1_T, psi_2_T)
        KL_X.append(kl_on_same_support_measures(psi_T, psi_star, eps=eps, base=base, strict=True))
        # G
        G_T = genotype_distribution_from_marginals(psi_1_T, psi_2_T)
        KL_G.append(kl_divergence(G_T, G_psi_star))

    # Plot 1x3 panels
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)

    axes[0].plot(Ts, KL_P, label='KL on allele space P', color='tab:blue', linewidth=2)
    axes[0].set_title('Allele space P')
    axes[0].set_xlabel('Generations (T)')
    axes[0].set_ylabel('KL divergence [bits]')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best')

    axes[1].plot(Ts, KL_X, label='KL on simplex X', color='tab:red', linewidth=2)
    axes[1].set_title('Simplex X')
    axes[1].set_xlabel('Generations (T)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best')

    axes[2].plot(Ts, KL_G, label='KL on genotype space G', color='tab:green', linewidth=2)
    axes[2].set_title('Genotype space G')
    axes[2].set_xlabel('Generations (T)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='best')

    fig.suptitle(title or f'KL convergence: N={N}, μ={mu}, s1={s1}, s2={s2}, step={step}')
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)

    return {
        'T': np.array(Ts, dtype=int),
        'KL_P': np.array(KL_P, dtype=float),
        'KL_X': np.array(KL_X, dtype=float),
        'KL_G': np.array(KL_G, dtype=float),
    }

# D stands for D_kl - our "Miso" information
def compute_D_convergence(N: int, mu: float, s1: float, s2: float,
                           T_max: int = 5000, step: int = 50,
                           eps: float = 1e-12, base: float = 2.0) -> pd.DataFrame:
    """Return a tidy DataFrame with T, KL_P, KL_X, KL_G (no plotting)."""
    phi = compute_phi_star(N, mu)
    Ts = np.arange(0, int(T_max) + 1, int(step))
    phi_star = build_simplex3_measure_from_marginals(phi, phi)
    G_psi_star = genotype_distribution_from_marginals(phi, phi)

    KL_P, KL_X, KL_G = [], [], []
    for T in Ts:
        psi_1_T, psi_2_T = marginals_from_s(N, mu, s1, s2, T)
        KL_P.append(kl_divergence(psi_1_T, phi) + kl_divergence(psi_2_T, phi))
        psi_T = build_simplex3_measure_from_marginals(psi_1_T, psi_2_T)
        KL_X.append(kl_on_same_support_measures(psi_T, phi_star, eps=eps, base=base, strict=True))
        G_T = genotype_distribution_from_marginals(psi_1_T, psi_2_T)
        KL_G.append(kl_divergence(G_T, G_psi_star))

    return pd.DataFrame({"T": Ts, "KL_P": KL_P, "KL_X": KL_X, "KL_G": KL_G})


# def plot_D_convergence(df: pd.DataFrame, title: str | None = None):
#     fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)

#     axes[0].plot(df["T"], df["KL_P"], label='KL on allele space P', linewidth=2)
#     axes[0].set_title('Allele space P')
#     axes[0].set_xlabel('Generations (T)')
#     axes[0].set_ylabel('KL divergence [bits]')
#     axes[0].grid(True, alpha=0.3)
#     axes[0].legend(loc='best')

#     axes[1].plot(df["T"], df["KL_X"], label='KL on simplex X', linewidth=2)
#     axes[1].set_title('Simplex X')
#     axes[1].set_xlabel('Generations (T)')
#     axes[1].grid(True, alpha=0.3)
#     axes[1].legend(loc='best')

#     axes[2].plot(df["T"], df["KL_G"], label='KL on genotype space G', linewidth=2)
#     axes[2].set_title('Genotype space G')
#     axes[2].set_xlabel('Generations (T)')
#     axes[2].grid(True, alpha=0.3)
#     axes[2].legend(loc='best')

#     fig.suptitle(title, y=0.98)
#     fig.tight_layout()
#     return fig, axes

def compute_I_convergence(N: int, mu: float, s1: float, s2: float,
                           T_max: int = 5000, step: int = 50,
                           eps: float = 1e-12, base: float = 2.0) -> pd.DataFrame:
    """Return a tidy DataFrame with T, KL_P, KL_X, KL_G for free fitness (vs psi_*)."""
    model = MarkovTwoLoci(N=N, mu=mu, s1=s1, s2=s2, quiet=True)
    psi_1_star, psi_2_star = model.stationary_distributions(method='direct')
    Ts = np.arange(0, int(T_max) + 1, int(step))
    psi_star = build_simplex3_measure_from_marginals(psi_1_star, psi_2_star)
    G_star = genotype_distribution_from_marginals(psi_1_star, psi_2_star)

    KL_P, KL_X, KL_G = [], [], []
    for T in Ts:
        psi_1_T, psi_2_T = marginals_from_s(N, mu, s1, s2, T)
        KL_P.append(kl_divergence(psi_1_T, psi_1_star) + kl_divergence(psi_2_T, psi_2_star))
        psi_T = build_simplex3_measure_from_marginals(psi_1_T, psi_2_T)
        KL_X.append(kl_on_same_support_measures(psi_T, psi_star, eps=eps, base=base, strict=True))
        G_T = genotype_distribution_from_marginals(psi_1_T, psi_2_T)
        KL_G.append(kl_divergence(G_T, G_star))

    return pd.DataFrame({"T": Ts, "KL_P": KL_P, "KL_X": KL_X, "KL_G": KL_G})


def plot_final_convergence(N: int, mu: float, s1: float, s2: float,
                           T_max: int = 5000, step: int = 50,
                           eps: float = 1e-12, base: float = 2.0):
    """
    Final analysis plot: two panels summarizing KL convergence for D and I.
      Left: D_KL(psi(t), phi*) — accumulation of information
      Right: D_KL(psi(t), psi*) — Iwasa's free fitness

    Each panel shows three lines: KL_P, KL_X, KL_G.
    The figure title includes N, μ, s1, s2, step.
    """
    df_D = compute_D_convergence(N, mu, s1, s2, T_max=T_max, step=step, eps=eps, base=base)
    df_I = compute_I_convergence(N, mu, s1, s2, T_max=T_max, step=step, eps=eps, base=base)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

    # Left: D
    axes[0].plot(df_D["T"], df_D["KL_P"], label='KL_P', linewidth=2,marker='o',  color='tab:blue')
    axes[0].plot(df_D["T"], df_D["KL_X"], label='KL_X', linewidth=2, color='tab:red')
    axes[0].plot(df_D["T"], df_D["KL_G"], label='KL_G', linewidth=2, color='tab:green')
    axes[0].set_title('Accumulation of information\nD_KL(ψ(t) || φ*)')
    axes[0].set_xlabel('Generations (T)')
    axes[0].set_ylabel('KL divergence [bits]')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best')

    # Right: I
    axes[1].plot(df_I["T"], df_I["KL_P"], label='KL_P', linewidth=2, marker='o', color='tab:blue')
    axes[1].plot(df_I["T"], df_I["KL_X"], label='KL_X', linewidth=2, color='tab:red')
    axes[1].plot(df_I["T"], df_I["KL_G"], label='KL_G', linewidth=2, color='tab:green')
    axes[1].set_title("Iwasa's free fitness\nD_KL(ψ(t) || ψ*)")
    axes[1].set_xlabel('Generations (T)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best')

    fig.suptitle(f'N={N}, μ={mu}, s1={s1}, s2={s2}, step={step}', y=0.96)
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)

    return fig, axes


#########################################################################################################################
# Two-locus Wright–Fisher with LD (selection → recombination → mutation → sampling) HELPER FUNCTIONS
#########################################################################################################################

def two_locus_selection(x: np.ndarray, s1: float, s2: float, mode: str = "multiplicative") -> np.ndarray:
    """
    Apply genic selection on haplotype frequencies x = [x00, x01, x10, x11].

    - additive:       w = [1, 1+s2, 1+s1, 1+s1+s2]
    - multiplicative: w = [1, 1+s2, 1+s1, (1+s1)(1+s2)]  (default)
    Returns normalized post-selection frequencies x^(s).
    """
    x = np.asarray(x, dtype=float)
    s1 = float(s1)
    s2 = float(s2)
    if mode == "additive":
        w = np.array([1.0, 1.0 + s2, 1.0 + s1, 1.0 + s1 + s2], dtype=float)
    else:
        w = np.array([1.0, 1.0 + s2, 1.0 + s1, (1.0 + s1) * (1.0 + s2)], dtype=float)
    xw = x * w
    total = float(np.sum(xw))
    if total <= 0.0:
        return np.full(4, 0.25, dtype=float)
    return xw / total


def recombine_D_shift(x: np.ndarray, r: float) -> np.ndarray:
    """
    Recombination via LD D-shift on haplotypes x = [x00, x01, x10, x11]:
      D = x00*x11 - x01*x10
      x00 ← x00 - r D;
      x01 ← x01 + r D;
      x10 ← x10 + r D;
      x11 ← x11 - r D
    Preserves marginal allele frequencies. Returns normalized x^(r).
    """
    x = np.asarray(x, dtype=float)
    x00, x01, x10, x11 = x
    D = x00 * x11 - x01 * x10 # Linkage equilibrium of this population
    xr = np.array([x00 - r * D, x01 + r * D, x10 + r * D, x11 - r * D], dtype=float) #updating the population after recombination
    # Numerical cleanup
    xr = np.maximum(xr, 0.0) # doesnt allow negative numbers
    s = float(np.sum(xr)) # should be =1 -- summing all the frequencies in the population
    if s > 0.0:
        xr = xr / s # normalizing the population after recombination
    else:
        xr[:] = 0.25 # this means some mistake happens--> we dub every genotype as equally likely-- should never reach here
    return xr


def mutation_matrix_two_locus(mu: float) -> np.ndarray:
    """
    Build the 4x4 symmetric per-locus mutation matrix M(μ) = K ⊗ K with
      K = [[1-μ, μ], [μ, 1-μ]].
    Row-stochastic; rows are "from", columns are "to" in order [00, 01, 10, 11].
    """
    """
    Interlude:
    Kronecker product: definition, intuition, and multi-locus mutation matrices
    ----------------------------------------------------------------------------

    Mathematical definition
    -----------------------
    Let A be an (m×n) matrix and B be a (p×q) matrix. The Kronecker product
    A ⊗ B is the (mp × nq) block matrix defined by
        A ⊗ B = [ a_ij * B ]_(i=1..m, j=1..n)
    i.e., replace every entry a_ij of A by the scalar multiple a_ij * B.
    Dimensions multiply: (m×n) ⊗ (p×q) → (mp×nq).

    Key properties
    --------------
    • (A ⊗ B)(C ⊗ D) = (AC) ⊗ (BD)  (when dimensions match)
    • (A ⊗ B)^T = A^T ⊗ B^T
    • If A, B are row-stochastic (rows sum to 1 with nonnegative entries),
    then A ⊗ B is row-stochastic as well.

    Mutation: 1 locus → 2 loci via Kronecker
    ----------------------------------------
    Single-locus, symmetric mutation matrix (rows “from”, cols “to”):
        K = [[1-μ, μ],
            [ μ , 1-μ]]

    Two loci mutate independently ⇒ joint mutation matrix is
        M_2 = K ⊗ K   (size 4×4).

    If we order two-locus haplotypes as [00, 01, 10, 11], then
        M_2 =
        [[(1-μ)^2, (1-μ)μ, (1-μ)μ,  μ^2],
        [ (1-μ)μ, (1-μ)^2,   μ^2 , (1-μ)μ],
        [ (1-μ)μ,   μ^2 , (1-μ)^2, (1-μ)μ],
        [   μ^2 , (1-μ)μ, (1-μ)μ, (1-μ)^2]]

    Interpretation: entries factor by loci because mutations are independent.
    Example: P(00 → 11) = μ * μ = μ^2 (both loci flip).
            P(01 → 01) = (1-μ) * (1-μ) = (1-μ)^2 (no flips at either locus).
            P(00 → 01) = (1-μ) * μ (first stays 0, second flips 0→1).

    Generalization to 3 loci (and L loci)
    -------------------------------------
    For 3 independent loci, the joint mutation matrix is
        M_3 = K ⊗ K ⊗ K   (size 8×8).

    A convenient haplotype order is binary lexicographic:
        index 0..7 ↔ haplotype in {0,1}^3 as
        [000, 001, 010, 011, 100, 101, 110, 111].

    Because loci mutate independently, each entry is a product of per-locus
    probabilities. Example with symmetric μ:
        P(000 → 101) = (0→1) * (0→0) * (0→1) = μ * (1-μ) * μ = μ^2 (1-μ)
        P(111 → 011) = (1→0) * (1→1) * (1→1) = μ * (1-μ) * (1-μ) = μ (1-μ)^2

    Programmatically, you can build these with NumPy:
        # two loci
        M2 = np.kron(K, K)


    In general for L loci (independent and identically distributed):
        M_L = K ⊗ K ⊗ ... ⊗ K  (L times), shape (2^L × 2^L).
    """

    mu = float(mu)
    K = np.array([[1.0 - mu, mu], [mu, 1.0 - mu]], dtype=float)
    return np.kron(K, K)


def mutate_two_locus(x: np.ndarray, mu: float) -> np.ndarray:
    """Apply x^(m) = M(μ) @ x, where x=[x00,x01,x10,x11]."""
    M = mutation_matrix_two_locus(mu)
    xm = M @ np.asarray(x, dtype=float)
    xm = np.maximum(xm, 0.0) # making sure on negative values
    s = float(np.sum(xm))
    if s > 0.0:
        xm = xm / s
    else:
        xm[:] = 0.25
    return xm


def two_locus_deterministic_p(x: np.ndarray,
                              s1: float,
                              s2: float,
                              r: float,
                              mu: float,
                              selection_mode: str = "multiplicative") -> np.ndarray:
    """
    Deterministic update (expectation) for haplotype frequencies over one generation:
      x → x^(s) → x^(r) → x^(m) ≡ p
    Returns p = [p00,p01,p10,p11], normalized.
    Accepts either counts or frequencies; counts are internally normalized.
    """
    x = np.asarray(x, dtype=float)
    s = float(np.sum(x)) # normalizing
    if s <= 0.0:
        x = np.full(4, 0.25, dtype=float)
    else:
        x = x / s
    xs = two_locus_selection(x, s1=s1, s2=s2, mode=selection_mode)
    xr = recombine_D_shift(xs, r=r)
    xm = mutate_two_locus(xr, mu=mu)
    return xm

# THE PMF
def multinomial_logpmf(counts: np.ndarray, N: int, p: np.ndarray) -> float:
    """
    Log Multinomial PMF using lgamma: log(N!) - sum log(yi!) + sum yi*log(pi).
    counts: integer vector of length 4 summing to N; p: length 4, sum 1.
    """
    y = np.asarray(counts, dtype=int)
    p = np.asarray(p, dtype=float)
    if int(np.sum(y)) != int(N):
        return float('-inf')
    p = np.maximum(p, 1e-300)
    p = p / np.sum(p)
    return float(sp_multinomial.logpmf(y, n=int(N), p=p))

# gives transition prbability from x to y
def transition_probability_two_locus(x_counts: tuple[int, int, int, int],
                                     y_counts: tuple[int, int, int, int],
                                     N: int,
                                     s1: float,
                                     s2: float,
                                     r: float,
                                     mu: float,
                                     selection_mode: str = "multiplicative") -> float:
    """
    One-step transition probability Pr(x→y) for the two-locus WF with LD, where
      x, y are haplotype count 4-tuples summing to N in order [00,01,10,11].
    Deterministic p is computed via selection→recombination→mutation, then
      Pr(x→y) = Multinomial(N, p)[y].
    """
    x = np.array(x_counts, dtype=float)
    p = two_locus_deterministic_p(x, s1=s1, s2=s2, r=r, mu=mu, selection_mode=selection_mode)
    logp = multinomial_logpmf(np.array(y_counts, dtype=float), N=int(N), p=p)
    return float(np.exp(logp))

# full row in the trnasition matrix, given x
def transition_row_two_locus(x_counts: tuple[int, int, int, int],
                             N: int,
                             s1: float,
                             s2: float,
                             r: float,
                             mu: float,
                             selection_mode: str = "multiplicative",
                             return_log: bool = False) -> dict[tuple[int, int, int, int], float]:
    """
    Compute the full transition row Pr(x→·) as a dict mapping 4-tuples y to probabilities.
    WARNING: the state space size is comb(N+3, 3), which grows as O(N^3).
    """
    x = np.array(x_counts, dtype=float)
    p = two_locus_deterministic_p(x, s1=s1, s2=s2, r=r, mu=mu, selection_mode=selection_mode)
    out: dict[tuple[int, int, int, int], float] = {}
    from math import lgamma
    logNfact = lgamma(N + 1.0)
    # enumerate all integer compositions of N into 4 parts using existing helper
    for a, b, c, d in _compositions(N, 4):
        y = (a, b, c, d)
        # log multinomial pmf
        logp = (
            logNfact
            - (lgamma(a + 1.0) + lgamma(b + 1.0) + lgamma(c + 1.0) + lgamma(d + 1.0))
            + a * np.log(max(p[0], 1e-300))
            + b * np.log(max(p[1], 1e-300))
            + c * np.log(max(p[2], 1e-300))
            + d * np.log(max(p[3], 1e-300))
        )
        out[y] = float(logp if return_log else np.exp(logp))
    return out

#########################################################################################################################
# TwoLocusLD wrapper class (selection → recombination → mutation → sampling)
#########################################################################################################################


class TwoLocusLD:
    """
    Two-locus Wright-Fisher with LD and recombination.

    Keeps your existing functional implementation intact and provides an OO wrapper
    to get transition probabilities from a given haplotype count state.

    Parameters
    ----------
    N : int
        Population size (total haplotypes; counts sum to N)
    s1, s2 : float
        Genic selection coefficients for loci 1 and 2
    r : float
        Recombination rate
    mu : float
        Symmetric per-locus mutation rate
    selection_mode : {"multiplicative","additive"}
        Selection combination across loci; multiplicative by default
    """

    def __init__(self, N: int, s1: float, s2: float, r: float, mu: float, selection_mode: str = "multiplicative"):
        self.N = int(N)
        self.s1 = float(s1)
        self.s2 = float(s2)
        self.r = float(r)
        self.mu = float(mu)
        self.selection_mode = str(selection_mode)
        # lazy caches for state enumeration and factorial terms
        self._states_fracs: np.ndarray | None = None  # shape (S,4)
        self._states_counts: np.ndarray | None = None # shape (S,4) ints
        self._log_counts_fact_sum: np.ndarray | None = None # shape (S,)
        self._logNfact: float | None = None

    def p_from_counts(self, x_counts: tuple[int, int, int, int]) -> np.ndarray:
        """Return deterministic p = x^(s→r→m) from counts (normalized internally)."""
        return two_locus_deterministic_p(
            np.asarray(x_counts, dtype=float),
            s1=self.s1,
            s2=self.s2,
            r=self.r,
            mu=self.mu,
            selection_mode=self.selection_mode,
        )

    def transition_prob(self, x_counts: tuple[int, int, int, int], y_counts: tuple[int, int, int, int], return_log: bool = False) -> float:
        """Return Pr(x→y) (or its log) via the multinomial with p_from_counts(x)."""
        p = self.p_from_counts(x_counts)
        logp = multinomial_logpmf(np.asarray(y_counts, dtype=int), N=self.N, p=p)
        return float(logp) if return_log else float(np.exp(logp))

    def enumerate_states(self):
        """Yield all 4-tuples (a,b,c,d) of nonnegative integers summing to N in lexicographic order."""
        N = self.N
        for a in range(N + 1):
            for b in range(N - a + 1):
                for c in range(N - a - b + 1):
                    d = N - a - b - c
                    yield (a, b, c, d)

    def transition_row(self, x_counts: tuple[int, int, int, int], return_log: bool = False) -> dict[tuple[int, int, int, int], float]:
        """Return full transition row Pr(x→·) as a dict mapping 4-tuples to probabilities (or log-probabilities)."""
        return transition_row_two_locus(
            x_counts=x_counts,
            N=self.N,
            s1=self.s1,
            s2=self.s2,
            r=self.r,
            mu=self.mu,
            selection_mode=self.selection_mode,
            return_log=return_log,
        )

    def _ensure_state_cache(self):
        if self._states_fracs is not None:
            return
        # Enumerate states using simplex3_grid to get fractional coordinates in a consistent order
        fracs = np.array(simplex3_grid(self.N), dtype=float)  # shape (S,4), components are multiples of 1/N
        counts = (fracs * float(self.N) + 1e-12).astype(int)
        from math import lgamma
        log_counts_fact_sum = np.sum([[lgamma(int(c) + 1.0) for c in row] for row in counts], axis=1, dtype=float)
        self._states_fracs = fracs
        self._states_counts = counts
        self._log_counts_fact_sum = np.asarray(log_counts_fact_sum, dtype=float)
        self._logNfact = float(lgamma(self.N + 1.0))

    def transition_row_arrays(self,
                              x_state: tuple[int, int, int, int] | tuple[float, float, float, float] | np.ndarray,
                              return_log: bool = False,
                              states_as: str = "fractions") -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorized row computation aligned to the simplex grid order.

        - x_state can be counts (ints summing to N) or fractions (floats summing to 1 with step 1/N).
        - states_as: "fractions" or "counts" controls the returned state coordinates.
        - Returns (states, probs_or_logprobs).
        """
        self._ensure_state_cache()
        # Normalize input state
        x_state = np.asarray(x_state)
        if x_state.dtype.kind in ("f", "c"):
            if not np.isclose(np.sum(x_state), 1.0):
                raise ValueError("Fractional x_state must sum to 1.0")
            counts = (x_state * float(self.N) + 1e-12).astype(int)
        else:
            if int(np.sum(x_state)) != self.N:
                raise ValueError("Count x_state must sum to N")
            counts = x_state.astype(int)

        # Deterministic p from counts
        p = self.p_from_counts(tuple(int(c) for c in counts))
        p = np.maximum(p, 1e-300)
        logp = np.log(p)

        # Vectorized logpmf over all states y
        y_dot_logp = self._states_counts @ logp  # shape (S,)
        log_row = self._logNfact - self._log_counts_fact_sum + y_dot_logp
        if return_log:
            probs = log_row
        else:
            probs = np.exp(log_row)

        if states_as == "counts":
            states = self._states_counts.copy()
        else:
            states = self._states_fracs.copy()
        return states, probs

    def build_transition_matrix(self,
                                states_as: str = "fractions",
                                as_sparse: bool = True,
                                dtype: np.dtype = np.float64):
        """
        Build the full one-step transition matrix over the simplex 1/N grid ordering used by simplex3_grid.

        Returns (M, states):
          - M: scipy.sparse.csr_matrix if as_sparse, else dense np.ndarray (WARNING: dense is huge)
          - states: (S,4) array of state coordinates in fractions or counts per states_as
        """
        from scipy.sparse import csr_matrix
        self._ensure_state_cache()
        S = self._states_counts.shape[0]
        # Assemble rows one by one; the matrix is dense (multinomial support), so sparse may still be very large.
        if as_sparse:
            # build block in memory – will be large for N=40 (~1.5e8 entries) even in sparse form
            # We instead build in chunks to avoid holding entire dense matrix; but since rows are dense,
            # CSR will still store S^2 values. Use with caution.
            data = []
            indices = []
            indptr = [0]
            for i in range(S):
                x_counts = tuple(int(c) for c in self._states_counts[i])
                # vectorized row
                _, row = self.transition_row_arrays(x_counts, return_log=False, states_as="counts")
                # store
                data.extend(row.astype(dtype))
                indices.extend(range(S))
                indptr.append(len(data))
            M_row = csr_matrix((np.array(data, dtype=dtype), np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int32)), shape=(S, S))
        else:
            M_row = np.empty((S, S), dtype=dtype)
            for i in range(S):
                x_counts = tuple(int(c) for c in self._states_counts[i])
                _, row = self.transition_row_arrays(x_counts, return_log=False, states_as="counts")
                M_row[i, :] = row.astype(dtype)

        states = self._states_fracs.copy() if states_as == "fractions" else self._states_counts.copy()
        # Return column-stochastic matrix: P_col = (P_row)^T
        M_col = M_row.T
        return M_col, states

    def apply_P_colvec(self, v: np.ndarray, normalize: bool = True, use_log: bool = False) -> np.ndarray:
        """
        Compute v_next = P v for column-stochastic P without materializing P.

        TECHNICAL:
        - Iterates over source states j and accumulates v[j] * Pr(j→·), where Pr(j→·) is a dense row.
        - Each row is computed on-the-fly via a vectorized multinomial logpmf over all target states.
        - If use_log=True, we compute log-probabilities and exponentiate per row for improved stability.

        COST:
        - Complexity is O(S^2) per application because each row has full support in a multinomial.

        USAGE:
        - v must be ordered as simplex3_grid(N) states (same order as the internal state cache).
        - If normalize=True, the output is renormalized to sum to 1.
        """
        self._ensure_state_cache()
        v = np.asarray(v, dtype=float)
        S = self._states_counts.shape[0]
        if v.shape != (S,):
            raise ValueError(f"vector shape {v.shape} does not match number of states {S}")
        v_next = np.zeros(S, dtype=float)
        for j in range(S):
            if v[j] == 0.0:
                continue
            x_counts = tuple(int(c) for c in self._states_counts[j])
            # row contains Pr(j -> i) aligned with i order; this is exactly column j of P
            _, row = self.transition_row_arrays(x_counts, return_log=use_log, states_as="counts")
            if use_log:
                row = np.exp(row)
            v_next += v[j] * row
        if normalize:
            s = float(v_next.sum())
            if s > 0.0:
                v_next /= s
        return v_next

    def evolve_distribution(self, pi0: np.ndarray, T: int, normalize_each: bool = True) -> np.ndarray:
        """
        Return v_T = P^T v_0 via repeated applications of apply_P_colvec.

        NOTE:
        - Straightforward but expensive for large S or large T (O(T S^2)).
        - Prefer apply_P_colvec_fast for a single long evolution, or spectral methods when many T are needed.
        """
        pi = np.asarray(pi0, dtype=float)
        for _ in range(int(T)):
            pi = self.apply_P_colvec(pi, normalize=normalize_each)
        if not normalize_each:
            s = float(pi.sum())
            if s > 0.0:
                pi /= s
        return pi

    def apply_P_colvec_fast(self,
                             v: np.ndarray,
                             normalize: bool = True,
                             use_log: bool = False,
                             num_workers: int | None = None,
                             top_k: int | None = None,
                             tol: float | None = None,
                             chunk_size: int | None = None) -> np.ndarray:
        """
        Faster P v with multi-threading and optional truncation.

        RECOMMENDED FOR SPEED when computing a single P v or a single long P^T v:
        - Splits the sum over source indices into chunks processed in parallel threads.
        - Optional sparsification: only process the largest entries of v (top_k) or those above a threshold (tol).

        PARAMETERS:
        - num_workers: threads to use (default: os.cpu_count()).
        - top_k: only use the largest top_k entries of v (by absolute value).
        - tol: keep indices j with |v[j]| >= tol * max(|v|) (ignored if top_k set).
        - chunk_size: indices per task (default: ~ceil(len(J)/num_workers)).
        """
        import os
        self._ensure_state_cache()
        v = np.asarray(v, dtype=float)
        S = self._states_counts.shape[0]
        if v.shape != (S,):
            raise ValueError(f"vector shape {v.shape} does not match number of states {S}")

        # choose indices to process
        if top_k is not None and top_k < S:
            idx = np.argpartition(-np.abs(v), kth=min(top_k, S - 1))[:top_k]
            J = idx[np.argsort(-np.abs(v[idx]))]
        elif tol is not None:
            vmax = float(np.max(np.abs(v)))
            thresh = tol * vmax
            J = np.where(np.abs(v) >= thresh)[0]
        else:
            J = np.arange(S, dtype=int)

        if num_workers is None or num_workers <= 0:
            num_workers = max(1, os.cpu_count() or 1)
        if chunk_size is None or chunk_size <= 0:
            chunk_size = max(1, int(np.ceil(len(J) / num_workers)))

        # create chunks
        chunks = [J[i:i + chunk_size] for i in range(0, len(J), chunk_size)]

        def worker(idx_chunk: np.ndarray) -> np.ndarray:
            acc = np.zeros(S, dtype=float)
            for j in idx_chunk:
                w = v[j]
                if w == 0.0:
                    continue
                x_counts = tuple(int(c) for c in self._states_counts[j])
                _, row = self.transition_row_arrays(x_counts, return_log=use_log, states_as="counts")
                if use_log:
                    row = np.exp(row)
                acc += w * row
            return acc

        v_next = np.zeros(S, dtype=float)
        # threads are fine here; numpy releases GIL in heavy ops
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            for partial in ex.map(worker, chunks):
                v_next += partial

        if normalize:
            s = float(v_next.sum())
            if s > 0.0:
                v_next /= s
        return v_next

# getting the stationary distribution
    def power_stationary(self, tol: float = 1e-12, max_iter: int = 1000, verbose: bool = False) -> np.ndarray:
        """
        Compute stationary distribution (right eigenvector) by power iteration on column-stochastic P.

        RECOMMENDED FOR STATIONARY DISTRIBUTION in large-S settings:
        - Memory-light: uses only matvecs via apply_P_colvec.
        - Converges when there is a spectral gap; monitors L1 change per iteration.
        - Starts from uniform over states. Returns v such that P v ≈ v, sum=1.
        """
        self._ensure_state_cache()
        S = self._states_counts.shape[0]
        pi = np.full(S, 1.0 / S, dtype=float)
        for it in range(int(max_iter)):
            pi_next = self.apply_P_colvec(pi, normalize=True)
            diff = float(np.sum(np.abs(pi_next - pi)))
            if verbose:
                print(f"[power] iter={it} L1={diff:.3e}")
            if diff < tol:
                return pi_next
            pi = pi_next
        return pi
# an alternative to power_stationary
    def stationary_via_eigs(self, tol: float = 1e-12, maxiter: int = 1000, verbose: bool = False) -> np.ndarray:
        """
        Compute stationary distribution as the dominant right eigenvector using ARPACK (eigs),
        without materializing P. Uses a LinearOperator whose matvec is apply_P_colvec.

        DETAILS:
        - Finds the largest-magnitude eigenpair. For a stochastic matrix, λ≈1 mode is stationary.
        - May require careful tolerance/maxiter; more brittle than power iteration on tough spectra.

        USE WHEN:
        - You need eigen-information explicitly or want potentially faster convergence than power in some cases.
        - Otherwise, prefer power_stationary for robustness.
        """
        self._ensure_state_cache()
        S = self._states_counts.shape[0]

        def matvec(v):
            return self.apply_P_colvec(v, normalize=False)

        A = LinearOperator((S, S), matvec=matvec, dtype=np.float64)
        v0 = np.full(S, 1.0 / S, dtype=np.float64)
        vals, vecs = eigs(A, k=1, which='LM', v0=v0, tol=tol, maxiter=maxiter)
        v = np.real(vecs[:, 0])
        # enforce nonnegativity and normalize to simplex
        v = np.maximum(v, 0.0)
        s = float(v.sum())
        if s == 0.0:
            v[:] = 1.0 / S
        else:
            v /= s
        if verbose:
            lam = np.real(vals[0])
            print(f"[eigs] dominant eigenvalue ≈ {lam:.8f}")
        return v

    def leading_modes(self, k: int = 20, tol: float = 1e-12, maxiter: int = 2000, verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute k leading right eigenpairs (λ, U) of column-stochastic P using a LinearOperator.

        TECHNICAL:
        - Uses ARPACK to compute k largest-magnitude eigenpairs of the implicit operator v ↦ P v.
        - Returns (lam, U) with lam shape (k,), U shape (S, k). Real parts are taken.

        NOTE:
        - This is the setup step for fast multi-T evolution via spectral approximation.
        - k trades accuracy vs. cost; larger k captures more modes.
        """
        self._ensure_state_cache()
        S = self._states_counts.shape[0]

        def matvec(v):
            return self.apply_P_colvec_fast(v, normalize=False)

        A = LinearOperator((S, S), matvec=matvec, dtype=np.float64)
        v0 = np.full(S, 1.0 / S, dtype=np.float64)
        vals, vecs = eigs(A, k=min(k, S-2), which='LM', v0=v0, tol=tol, maxiter=maxiter)
        lam = np.real(vals)
        U = np.real(vecs)
        if verbose:
            print(f"[modes] obtained {len(lam)} modes; max |λ| = {np.max(np.abs(lam)):.6f}")
        return lam, U

    def approx_power_from_modes(self, v0: np.ndarray, lam: np.ndarray, U: np.ndarray, T: int) -> np.ndarray:
        """
        Approximate P^T v0 by projecting v0 into span(U) and scaling by λ^T.

        STEPS:
        - Solve least squares v0 ≈ U c (no regularization).
        - Apply spectral time evolution: U diag(λ^T) c.
        - Clip negatives to 0 and renormalize to a probability vector.

        PURPOSE:
        - Enables very fast evaluation of P^T v0 once U, λ are known.
        """
        v0 = np.asarray(v0, dtype=float)
        # Least-squares coefficients
        c, *_ = np.linalg.lstsq(U, v0, rcond=None)
        amp = (lam ** int(T)) * c
        vT = U @ amp
        vT = np.maximum(vT, 0.0)
        s = float(vT.sum())
        if s == 0.0:
            vT = np.full_like(vT, 1.0 / len(vT))
        else:
            vT /= s
        return vT

# ---------------------------------------------------------------------------
# evolve_many_via_modes
# ---------------------------------------------------------------------------
# Computes the approximate time evolution of a probability vector v0 under
# repeated application of the Markov transition matrix P:
#       v_T = P^T v0
#
# Instead of performing full matrix multiplications for each time T, this
# method uses the k leading eigenmodes (lam, U) of P for a fast spectral
# approximation:
#       P^T v0 ≈ U * (lam ** T) * (U⁻¹ v0)
#
# Steps:
#   1. Sort the requested timepoints Ts.
#   2. Compute the top-k eigenvalues and eigenvectors via self.leading_modes().
#   3. For each time T, call self.approx_power_from_modes(v0, lam, U, T)
#      to evaluate P^T v0 efficiently by scaling each modal component by lam_i^T.
#   4. Stack all resulting probability vectors into a single array.
#
# Returns:
#   Ts_sorted : (n_T,) int array
#       The sorted list of requested timepoints.
#   V : (n_T, S) float array
#       Each row V[i] is the approximate probability vector after Ts_sorted[i]
#       steps of the Markov process.
#
# Use when you need many timepoints — you pay the eigen-decomposition cost once,
# then evaluating each additional T is extremely cheap (O(kS) instead of O(S²)).
# ---------------------------------------------------------------------------

### USE THIS FOR SPEED- FOR A SERIES OF VALUES!!
    def evolve_many_via_modes(self, v0: np.ndarray, Ts: list[int], k: int = 20, tol: float = 1e-12, maxiter: int = 2000,
                               verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute [P^T v0] for multiple T using k leading modes once.

        RECOMMENDED FOR SPEED when evaluating many timepoints:
        - One-time cost: compute k modes (U, λ) via leading_modes.
        - Then P^T v0 is evaluated cheaply for each T by scaling coefficients with λ^T.

        RETURNS:
        - (Ts_sorted, V) where V has shape (len(Ts), S) and each row is a probability vector.
        """
        Ts_sorted = np.array(sorted(int(t) for t in Ts), dtype=int)
        lam, U = self.leading_modes(k=k, tol=tol, maxiter=maxiter, verbose=verbose)
        V = []
        for T in Ts_sorted:
            V.append(self.approx_power_from_modes(v0, lam, U, int(T)))
        return Ts_sorted, np.vstack(V)
#####################################################################################
# DEBUG: build X->P index map and print details once
    def build_X_to_P_index_map1(self, verbose: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[tuple[int, int], list[int]]]:
        """
        Vectorized construction of the pushforward index mapping from X states to P grid.

        DEFINITIONS:
        - X states are ordered as in simplex3_grid(N), cached in self._states_counts with columns [A,B,C,D].
        - P grid has integer coordinates (p1_count, p2_count) with 0..N for each axis.
          p1_count = C + D, p2_count = B + D (allele-1 counts at each locus).

        RETURNS:
        - p1_counts: (S,) integers
        - p2_counts: (S,) integers
        - idx: (S,) flattened target indices: idx = p1_counts*(N+1) + p2_counts
        - groups: dict mapping (p1_count, p2_count) -> list of X-state indices that map there

        If verbose=True, prints p1_counts, p2_counts, and for each (p1,p2) the count and indices.
        """
        self._ensure_state_cache()
        counts = self._states_counts  # (S,4) A,B,C,D
        p1_counts = counts[:, 2] + counts[:, 3]
        p2_counts = counts[:, 1] + counts[:, 3]
        idx = p1_counts * (self.N + 1) + p2_counts

        groups: dict[tuple[int, int], list[int]] = defaultdict(list)
        for j in range(len(idx)):
            key = (int(p1_counts[j]), int(p2_counts[j]))
            groups[key].append(j)

        if verbose:
            print("p1_counts:", p1_counts.tolist())
            print("p2_counts:", p2_counts.tolist())
            print("Mapping (p1,p2) -> indices [count]:")
            for (p1c, p2c) in sorted(groups.keys()):
                lst = groups[(p1c, p2c)]
                print(f"  ({p1c},{p2c}) -> {lst} [count={len(lst)}]")

        return p1_counts, p2_counts, idx, groups


#####################################################################################
    
    from collections import defaultdict
    from typing import Dict, List, Tuple
    import numpy as np

    def build_X_to_P_index_map(
        self,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[Tuple[int, int], List[int]]]:
        """
        Vectorized construction of the pushforward index mapping from X states to the P grid.

        DEFINITIONS
        ----------
        - X states are ordered as in simplex3_grid(N), cached in self._states_counts with columns [A, B, C, D].
        Each row sums to N.
        - The P grid is the lattice {(p1, p2): p1 in 0..N, p2 in 0..N}, where:
            p1_count = C + D   # allele-1 count at locus 1
            p2_count = B + D   # allele-1 count at locus 2

        RETURNS
        -------
        p1_counts : (S,) int ndarray
            Per-state allele-1 count at locus 1.
        p2_counts : (S,) int ndarray
            Per-state allele-1 count at locus 2.
        idx : (S,) int ndarray
            Flattened indices for a (N+1) x (N+1) grid in row-major order:
                idx = p1_counts * (N + 1) + p2_counts
        groups : dict[(int, int) -> list[int]]
            For each grid cell (p1, p2), the list of X-state indices j that map there.
            Lists are sorted in ascending order of j.

        NOTES
        -----
        - This implementation uses only NumPy vectorized operations.
        - Time complexity is O(S log S) due to sorting for grouping. In practice it is fast.
        - Memory is O(S) for the arrays, plus the final groups dictionary which stores S indices total.
        """

        # Ensure the cached state grid is populated.
        self._ensure_state_cache()

        # Alias for readability. Shape: (S, 4). Columns: [A, B, C, D].
        counts = self._states_counts

        # Compute allele-1 counts per locus for every X-state.
        # p1 = C + D, p2 = B + D
        p1_counts = counts[:, 2] + counts[:, 3]
        p2_counts = counts[:, 1] + counts[:, 3]

        # Flatten the 2D grid coordinate (p1, p2) to 1D row-major index for a (N+1) x (N+1) grid.
        # This allows fast grouping with np.unique.
        stride = self.N + 1
        idx = p1_counts * stride + p2_counts  # shape: (S,)

        # Group state indices by their flattened grid index.
        # - inv[j] is the group label of state j
        # - counts_uniq[g] is the size of group g
        # - uniq_idx[g] is the flattened grid index represented by group g
        uniq_idx, inv, counts_uniq = np.unique(idx, return_inverse=True, return_counts=True)

        # Sort states by their group labels so that states belonging to the same group are contiguous.
        # "stable" preserves the relative order of equal keys, which is not required here,
        # but it is often preferable for reproducibility.
        order = np.argsort(inv, kind="stable")

        # Compute split points to cut "order" into consecutive runs per group.
        # Example: counts_uniq = [3, 1, 2]  ->  split at [3, 4]
        split_points = np.cumsum(counts_uniq)[:-1]

        # Now "runs" is a list of arrays, each containing the original state indices for one group,
        # sorted ascending because "order" is a permutation of np.arange(S).
        runs = np.split(order, split_points)

        # Recover the (p1, p2) coordinate for each unique flattened index.
        # Shape: both (G,), where G = number of distinct occupied grid cells.
        uniq_p1 = uniq_idx // stride
        uniq_p2 = uniq_idx % stride

        # Assemble the dictionary that maps (p1, p2) -> list of X-state indices.
        # Convert each run to a Python list for the same return type as the original function.
        groups: Dict[Tuple[int, int], List[int]] = {
            (int(p1), int(p2)): run.astype(int).tolist()
            for (p1, p2), run in zip(zip(uniq_p1, uniq_p2), runs)
        }

        if verbose:
            # Diagnostic summary. Avoid printing huge arrays in very large problems.
            print("p1_counts:", p1_counts.tolist())
            print("p2_counts:", p2_counts.tolist())
            print("Mapping (p1,p2) -> indices [count]:")
            for (p1c, p2c) in sorted(groups.keys()):
                lst = groups[(p1c, p2c)]
                print(f"  ({p1c},{p2c}) -> {lst} [count={len(lst)}]")

        return p1_counts, p2_counts, idx, groups

#####################################################################################

    def _ensure_X_to_P_cached_idx(self) -> None:
        """Cache the flattened X→P target indices for reuse across many pushforwards."""
        if getattr(self, "_X2P_idx", None) is not None:
            return
        self._ensure_state_cache()
        counts = self._states_counts  # (S,4) A,B,C,D
        p1_counts = counts[:, 2] + counts[:, 3]
        p2_counts = counts[:, 1] + counts[:, 3]
        stride = self.N + 1
        self._X2P_idx = (p1_counts * stride + p2_counts).astype(int)

    def pushforward_X_to_P(self, w: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Map one X-distribution (shape (S,)) to a P matrix (shape (N+1, N+1)).
        Uses cached flattened indices and np.bincount for O(S) accumulation.
        """
        self._ensure_X_to_P_cached_idx()
        w = np.asarray(w, dtype=float).reshape(-1)
        # Validate length of w matches number of simplex states for this N
        S_expected = int(self._states_fracs.shape[0])
        if w.shape[0] != S_expected:
            raise ValueError(
                f"pushforward_X_to_P: length mismatch for X-distribution. "
                f"Expected {S_expected} states for N={self.N} (comb(N+3,3)), got {w.shape[0]}."
            )
        stride = self.N + 1
        P_flat = np.bincount(self._X2P_idx, weights=w, minlength=stride * stride)
        P = P_flat.reshape(stride, stride)
        if normalize:
            s = float(P.sum())
            if s > 0.0:
                P /= s
        return P

# ---------------------------------------------------------------------------
# pushforward_many_X_to_P
# ---------------------------------------------------------------------------
# Maps several probability distributions over X-space (the simplex of haplotype
# frequencies) into their corresponding joint distributions over P-space
# (the 2D allele-frequency grid for loci 1 and 2).
#
# INPUTS:
#   W : (K, S) float array
#       Each row W[k] is a probability distribution over all X-states
#       (S = number of simplex grid points).
#   Ts : list[int] or np.ndarray
#       The list of timepoints corresponding to each row in W.
#   normalize : bool, default=True
#       If True, each resulting P matrix is normalized to sum to 1.
#   as_dict : bool, default=True
#       If True, returns a dictionary mapping {T : P_matrix}.
#       If False, returns a 3D array of shape (K, N+1, N+1).
#   verbose : bool, default=False
#       If True, prints a short summary for each timepoint (sum, nonzeros).
#
# METHOD:
#   - Uses a precomputed flattened index mapping self._X2P_idx that says
#     for each X-state j which (p1, p2) cell it contributes to.
#   - For each timepoint k, it uses np.bincount to accumulate all state
#     probabilities W[k, j] that map to the same (p1, p2) cell.
#   - The flat array of length (N+1)^2 is reshaped into a square matrix P.
#   - Optionally normalized so that sum(P) = 1.
#
# RETURNS:
#   if as_dict:
#       dict[int, np.ndarray] mapping T -> P (each of shape (N+1, N+1))
#   else:
#       np.ndarray of shape (K, N+1, N+1) containing all P matrices stacked.
#
# PURPOSE:
#   Efficiently projects many evolving distributions in X-space (e.g. ψ_T)
#   into their allele-frequency representations in P-space, typically after
#   evolving them for multiple timepoints via spectral modes.
# ---------------------------------------------------------------------------

    def pushforward_many_X_to_P(self,
                                W: np.ndarray,
                                Ts: list[int] | np.ndarray,
                                normalize: bool = True,
                                as_dict: bool = True,
                                verbose: bool = False):
        """
        Map many X-distributions (shape (K,S)) to P matrices.
        Returns either a dict {T: P_matrix} or an array with shape (K, N+1, N+1).
        """
        self._ensure_X_to_P_cached_idx()
        W = np.asarray(W, dtype=float)
        Ts_arr = np.asarray(Ts, dtype=int)
        stride = self.N + 1
        mats = []
        for k in range(W.shape[0]):
            P_flat = np.bincount(self._X2P_idx, weights=W[k], minlength=stride * stride)
            P = P_flat.reshape(stride, stride)
            if normalize:
                s = float(P.sum())
                if s > 0.0:
                    P /= s
            if verbose:
                nz = int(np.count_nonzero(P))
                print(f"T={Ts_arr[k]}: sum={P.sum():.6f}, nonzeros={nz}")
            mats.append(P)
        if as_dict:
            return {int(Ts_arr[k]): mats[k] for k in range(len(mats))}
        return np.stack(mats, axis=0)

    def pushforward_X_to_G(self, w: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Map one X-distribution w (shape (S,)) to genotype distribution G (shape (4,)).
        Implements G = E_X[(A,B,C,D)] by averaging simplex coordinates with weights w.
        """
        self._ensure_state_cache()
        w = np.asarray(w, dtype=float).reshape(-1)
        S_expected = int(self._states_fracs.shape[0])
        if w.shape[0] != S_expected:
            raise ValueError(
                f"pushforward_X_to_G: length mismatch for X-distribution. "
                f"Expected {S_expected} states for N={self.N} (comb(N+3,3)), got {w.shape[0]}."
            )
        G = w @ self._states_fracs  # shape (4,)
        G = np.maximum(G, 0.0)
        if normalize:
            s = float(G.sum())
            if s > 0.0:
                G /= s
        return G

    def kl_P_from_X(self, w: np.ndarray, baseline_P: np.ndarray) -> float:
        """
        KL over P-space treating P as a measure on (p1,p2) grid.
        - w: X-distribution (shape (S,))
        - baseline_P: reference P measure, shape (N+1,N+1) or flattened ((N+1)^2,)
        """
        # Validates length inside pushforward_X_to_P
        P = self.pushforward_X_to_P(w, normalize=True).ravel()
        q = np.asarray(baseline_P, dtype=float).ravel()
        return kl_divergence(P, q)

    def kl_G_from_X(self, w: np.ndarray, baseline_G: np.ndarray) -> float:
        """
        KL over G-space where G is the average of X on the simplex vertices.
        - w: X-distribution (shape (S,))
        - baseline_G: reference 4-vector over genotypes [00,01,10,11]
        """
        # Validates length inside pushforward_X_to_G
        G = self.pushforward_X_to_G(w, normalize=True)
        q = np.asarray(baseline_G, dtype=float)
        return kl_divergence(G, q)

    def debug_check_G_pushforward(self, w: np.ndarray, verbose: bool = True) -> dict:
        """
        Sanity-check G computation from an X-distribution w:
          - G_fracs  = w @ states_fracs              (fractions per state)
          - G_counts = (w @ states_counts) / N       (expected counts, normalized)
        Both must match up to numerical noise.
        Returns a dict with both versions and their differences.
        """
        self._ensure_state_cache()
        w = np.asarray(w, dtype=float).reshape(-1)
        S_expected = int(self._states_fracs.shape[0])
        if w.shape[0] != S_expected:
            raise ValueError(
                f"debug_check_G_pushforward: length mismatch. Expected {S_expected} for N={self.N}, got {w.shape[0]}."
            )
        # Fractions-based expectation (primary implementation)
        G_fracs = self.pushforward_X_to_G(w, normalize=True)
        # Counts-based expectation, then normalize to sum to 1
        G_counts = (w @ self._states_counts) / float(self.N)
        G_counts = np.maximum(G_counts, 0.0)
        s = float(G_counts.sum())
        if s > 0.0:
            G_counts = G_counts / s
        else:
            G_counts = np.full(4, 0.25, dtype=float)
        # Differences
        diff = G_fracs - G_counts
        res = {
            "G_fracs": G_fracs,
            "G_counts": G_counts,
            "L1_diff": float(np.sum(np.abs(diff))),
            "L2_diff": float(np.linalg.norm(diff)),
            "max_diff": float(np.max(np.abs(diff))),
        }
        if verbose:
            print(f"[G-check] L1={res['L1_diff']:.3e}, L2={res['L2_diff']:.3e}, max={res['max_diff']:.3e}")
            print(f"[G-check] G_fracs = {G_fracs}")
            print(f"[G-check] G_counts= {G_counts}")
        return res

#########################################################################################################################
# main
#########################################################################################################################
    
if __name__ == '__main__':
    
    # PARAMETERS
    N = 30 # 40 -- CANONICAL
    #mu = 5e-4 # 5e-4 -- CANONICAL
    mu=0.01
    s1 = 0.005
    s2 = 0.10
    r = 0.005
    selection_mode = "multiplicative"   

    
    # CURRENT FOCUS:
    # Ts = [0, 10, 20, 50, 100, 200, 300, 400, 500, 600]
    # modes_k_val = 30
    # print("Computing D and I series for LD model... WISH US LUCK!")
    # import time
    # t0 = time.perf_counter()
    # df = compute_D_I_series_LD(N=N, s1=s1, s2=s2, r=r, mu=mu, Ts=Ts, modes_k=modes_k_val, selection_mode=selection_mode)
    # elapsed = time.perf_counter() - t0
    # print(f"[Timing] compute_D_I_series_LD: N={N}, |Ts|={len(Ts)}, modes_k={modes_k_val} -> {elapsed:.3f}s")
    # print(df)
    # _ = plot_D_I_series_from_df(df, title=f'N={N}, μ={mu}, s1={s1}, s2={s2}', r=r)
    # plt.show()
    # fig, axes = plot_D_I_series_dual_axis_from_df(df, title=f'N={N}, μ={mu}, s1={s1}, s2={s2}', r=r)
    # plt.show()

#########################################################################################################################
# Comparing D-series for two different μ values
#########################################################################################################################

    # Local helper for comparing D-series for two μ values (temporary, easy to delete)
    def plot_compare_D_series_two_dfs_local(df_left: pd.DataFrame,
                                            df_right: pd.DataFrame,
                                            mu_left: float,
                                            mu_right: float,
                                            title: str | None = None,
                                            r_val: float | None = None):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
        color_P, color_X, color_G = 'tab:blue', 'tab:red', 'tab:green'
        # Common y-limit
        max_y = 0.0
        for df_ in (df_left, df_right):
            if len(df_) == 0:
                continue
            max_y = max(max_y,
                        float(np.max(df_["D_P"])) if "D_P" in df_ else 0.0,
                        float(np.max(df_["D_X"])) if "D_X" in df_ else 0.0,
                        float(np.max(df_["D_G"])) if "D_G" in df_ else 0.0)
        if max_y <= 0.0:
            max_y = 1.0
        # Left panel
        ax = axes[0]
        ax.plot(df_left["T"], df_left["D_P"], label='KL_P', linewidth=2, marker='o', color=color_P)
        ax.plot(df_left["T"], df_left["D_X"], label='KL_X', linewidth=2, color=color_X)
        ax.plot(df_left["T"], df_left["D_G"], label='KL_G', linewidth=2, color=color_G)
        ax.set_xlabel('Generations (T)')
        ax.set_ylabel('KL divergence [bits]')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'μ={mu_left}')
        ax.legend(loc='best')
        ax.set_ylim(0.0, max_y * 1.05)
        # Right panel
        ax = axes[1]
        ax.plot(df_right["T"], df_right["D_P"], label='KL_P', linewidth=2, marker='o', color=color_P)
        ax.plot(df_right["T"], df_right["D_X"], label='KL_X', linewidth=2, color=color_X)
        ax.plot(df_right["T"], df_right["D_G"], label='KL_G', linewidth=2, color=color_G)
        ax.set_xlabel('Generations (T)')
        ax.set_ylabel('KL divergence [bits]')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'μ={mu_right}')
        # Super-title
        if r_val is not None:
            title = (f"{title}, r={r_val}" if title else f"r={r_val}")
        if title:
            fig.suptitle(title, y=0.96)
        fig.tight_layout()
        fig.subplots_adjust(top=0.86 if title else 0.92)
        return fig, axes

    # ----- HARD-CODED COMPARISON FROM PROVIDED DATA (two μ values) -----
    # Shared parameters: N=30, s1=0.005, s2=0.10, r=0.05 (μ differs)
    T_vals = [0, 10, 20, 50, 100, 200, 300, 400, 500, 600]

    # μ = 0.001
    D_X_mu1 = [0.157239, 1.874880, 4.011580, 7.303242, 8.394024, 8.675615, 8.786177, 8.858131, 8.905040, 8.935189]
    D_P_mu1 = [0.047020, 1.767381, 3.947357, 7.282551, 8.380071, 8.662614, 8.773361, 8.845415, 8.892382, 8.922566]
    D_G_mu1 = [0.004783, 0.102082, 0.227659, 0.448614, 0.558576, 0.669098, 0.744171, 0.794188, 0.826575, 0.847123]
    df_mu_0001 = pd.DataFrame({"T": T_vals, "D_X": D_X_mu1, "D_P": D_P_mu1, "D_G": D_G_mu1})

    # μ = 0.01
    D_X_mu2 = [0.012651, 0.746395, 1.584364, 2.691525, 3.032445, 3.081324, 3.082290, 3.082329, 3.082333, 3.082334]
    D_P_mu2 = [0.001533, 0.677543, 1.472491, 2.540047, 2.873431, 2.921459, 2.922408, 2.922447, 2.922451, 2.922451]
    D_G_mu2 = [0.000061, 0.066344, 0.169339, 0.367293, 0.450361, 0.463147, 0.463415, 0.463428, 0.463429, 0.463429]
    df_mu_001 = pd.DataFrame({"T": T_vals, "D_X": D_X_mu2, "D_P": D_P_mu2, "D_G": D_G_mu2})

    fig_cmp, axes_cmp = plot_compare_D_series_two_dfs_local(
        df_left=df_mu_0001,
        df_right=df_mu_001,
        mu_left=0.001,
        mu_right=0.01,
        title=f'Compare D-series: N=30, s1=0.005, s2=0.10',
        r_val=0.05
    )
    plt.show()
    # -------------------------------------------------------------------

    # s=0
    # T=500
    # step=10

    # model = TwoLocusLD(N=N, s1=s, s2=s, r=r, mu=mu, selection_mode=selection_mode)
    # n= count_simplex_points(N,3) # number of simplex poitns in the N-gridded 3D simplex
    # phi_star =  np.ones(n) / n # we checked- this is the stationary distribution without selection!

    # Example: compute psi_T for an array of times using spectral acceleration
    #Ts = [0, 10, 20, 50, 100, 150]
    #Ts_sorted, V = model.evolve_many_via_modes(phi_star, Ts, k=20, tol=1e-10, maxiter=2000, verbose=False)
    # KL over X (LD) vs neutral baseline
    #KL_X_LD = [kl_divergence(V[i], phi_star) for i in range(len(Ts_sorted))]
    #df_ld = pd.DataFrame({"T": Ts_sorted, "KL_X_LD": KL_X_LD})
    #print(df_ld)


    def compute_X_KL_LD_series(N: int, s1: float, s2: float, r: float, mu: float,
                            Ts: list[int], selection_mode: str = "multiplicative",
                            modes_k: int = 20, modes_tol: float = 1e-12, modes_maxiter: int = 2000) -> pd.DataFrame:
        """
        Compute KL_X_LD(T) over multiple times using spectral acceleration.
        Returns a DataFrame with columns [T, KL_X_LD].
        """
        ld = TwoLocusLD(N=N, s1=s1, s2=s2, r=r, mu=mu, selection_mode=selection_mode)
        S = count_simplex_points(N, 3)
        phi_star = np.ones(S, dtype=float) / S  # neutral baseline over simplex states
        psi_star = ld.power_stationary()

        Ts_sorted = sorted(int(t) for t in Ts)
        Ts_arr, p_T = ld.evolve_many_via_modes(phi_star, Ts_sorted, k=modes_k, tol=modes_tol, maxiter=modes_maxiter, verbose=False)
        D_kl = [kl_divergence(p_T[i], phi_star) for i in range(len(Ts_arr))]
        I_kl = [kl_divergence(p_T[i], psi_star) for i in range(len(Ts_arr))]
        return pd.DataFrame({"T": Ts_arr, "KL_X_LD": D_kl, "KL_I": I_kl})

    
   


    #df = compute_X_KL_LD_series(N=40, s1=s, s2=s, r=r, mu=mu, Ts=Ts, selection_mode=selection_mode)
    #print(df)
    #plt.plot(df["T"], df["KL_X_LD"])
    #plt.show()

    # DEBUG: build X->P index map and print details once
    #print("\n=== DEBUG: X -> P mapping (indices) ===")
    #p1c, p2c, idx_map, groups = model.build_X_to_P_index_map(verbose=False)

    # DEBUG: demonstrate pushforward for a couple of timepoints using spectral evolution
    #Ts_demo = [0, 10, 20]
    #Ts_sorted, V_demo = model.evolve_many_via_modes(phi_star, Ts_demo, k=10, tol=1e-10, maxiter=2000, verbose=False)
    #P_by_T = model.pushforward_many_X_to_P(V_demo, Ts_sorted, normalize=True, as_dict=True, verbose=True)
    #for T in Ts_sorted:
    #    P = P_by_T[int(T)]
    #    print(f"P[{T}].shape={P.shape}, sum={P.sum():.6f}")


    # things we did before
#########################################################################################################################
# PLOTTING THE CONVERGENCE OF D AND I 
#########################################################################################################################
    #fig=plot_final_convergence(N, mu, s1, s2, T_max=T, step=step, eps=1e-12, base=2.0)
    #plt.show()

    #information=compute_D_convergence(N, mu, s1, s2, T_max=T, step=50, eps=1e-12, base=2.0, title=f'D convergence: N={N}, μ={mu}, s1={s1}, s2={s2}')

    #free_fitness=analyze_I_convergence(N, mu, s1, s2, T_max=T, step=50, eps=1e-12, base=2.0, title=f'Free fitness convergence: N={N}, μ={mu}, s1={s1}, s2={s2}')

    #s1 = 0.005 -- CANONICAL
    #s2 = 0.010 -- CANONICAL
    # model = MarkovTwoLoci(N=N, mu=mu, s1=s1, s2=s2, quiet=True)
    # p1_star, p2_star= model.stationary_distributions(method='direct')
    # phi = compute_phi_star(N, mu)

    
    # p1_T, p2_T = marginals_from_s(N, mu, s1, s2, T) # on their way to stardom

    # measure_q = build_simplex3_measure_from_marginals(p1_star, p2_star)
    # measure_p = build_simplex3_measure_from_marginals(p1_T, p2_T)
    # kl = kl_on_same_support_measures(measure_p, measure_q)
    # print(kl)


#### This is an experiment that proves that the expected value of P and of X both lead to G 
# # --> Law G = E(X)=E(P) -- under independence

    # dist = []
    # print("G_sanity_check")
    # for t in range(0, 5000, 100):  
    #     p1_T, p2_T = marginals_from_s(N, mu, s1, s2, t)
    #     res = G_sanity_check(p1_T, p2_T)
    #     dist.append(res)
    #     print(f"T={t}: "
    #         f"L1={res['L1_diff']:.3e}, L2={res['L2_diff']:.3e}, max={res['max_diff']:.3e}")



### some plots
    #fig = plot_simplex3_measure_tetrahedron(measure_q, title='p1_star⊗p2_star on simplex')
    #fig.savefig("X_p1_star⊗p2_star_N40_mu5e-4_s10.005_s20.010.png", dpi=300, bbox_inches="tight")
    #plt.show()

    #print(kl_divergence(pi1_star,phi)*kl_divergence(pi2_star,phi))
    
    #fig_2d = model.plot_joint_heatmap(pi1_star, pi2_star)
    #plt.show()
    #fig_3d = model.plot_joint_3d(p1_star, p2_star)
    #fig_3d.savefig("P_p1_star⊗p2_star_N40_mu5e-4_s10.005_s20.010.png", dpi=300, bbox_inches="tight")
    #plt.show()

    # Nice, this works
    # D_t = []
    # for t in range(0, 5000, 100):
    #     D_t.append(D(t,N,mu,s1,s2))
    # plt.plot(D_t)
    # plt.show()


