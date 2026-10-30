#!/usr/bin/env python3

"""
Compare stationary distributions: discrete Wright-Fisher Markov chain vs. diffusion approximation.

This script computes:
- π_M(i): stationary distribution across states i=0..N from the finite Markov chain (existing class)
- π_D(i): diffusion stationary density f(x) ∝ x^{4N v - 1}(1-x)^{4N u - 1} exp[2 N s x]-- FOR DIPLOID POPULATION
- π_D(i): diffusion stationary density CONVENTION 1: f(x) ∝ x^{2N v - 1}(1-x)^{2N u - 1} exp[2 N s x]-- FOR HAPLOID POPULATION -- our case here
          diffusion stationary density CONVENTION 2: f(x) ∝ x^{2N v - 1}(1-x)^{2N u - 1} exp[N s x]-- FOR HAPLOID POPULATION -- our case here
          discretized to the same N+1 states by integrating over bins around i/N
          x: frequency of the favored A allele (fitness 1+s)
          v: mutation rate from a to A
          u: mutation rate from A to a


It then compares π_M and π_D using L1 distance and KL divergences and can optionally
search for an effective population size N_e for the diffusion that best matches π_M.

Notes
-----
- For symmetric mutation, set u=v=μ-- mutations are symmetric in this case.
- We integrate the unnormalized diffusion density with a log-offset for numerical stability;
   For numerical stability: instead of integrating f(x) directly (which can underflow if f is tiny),
   we work in log-space. Let log_f = ℓ(x) = log(f(x)), and pick an offset c = max(ℓ(x)) (or similar).
   Then we integrate exp(ℓ(x) - c) and multiply by exp(c) afterwards:
         I = e^c ∫ e^{ℓ(x) - c} dx
   This is the standard "log-sum-exp" trick—avoids overflow/underflow when dealing with very small densities.

- The Markov distribution is obtained from `WrightFisherMarkovChain` in `Markov_1_locus.py`.

Usage
-----
python diffusion_vs_markov.py --N 40 --mu 5e-4 --s 5e-3 --plot --optimize-ne
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from Markov_1_locus import WrightFisherMarkovChain, compute_stationary_quiet


# -----------------------------
# Diffusion stationary density
# -----------------------------

def _safe_log(x: float) -> float:
    if x <= 0.0:
        return -np.inf
    return math.log(x)


def diffusion_log_unnormalized(x: float, N_effective: float, s: float, u: float, v: float) -> float:
    """
    Log of unnormalized diffusion stationary density:
        f(x) ∝ x^{2N v - 1} (1-x)^{2N u - 1} exp[2 N s x]

    We compute in log-space for stability. Returns -inf at boundaries when appropriate.
    """
    if x <= 0.0:
        # Integrable if 4Nv > 0, otherwise zero mass at boundary bin by integration
        if 4.0 * N_effective * v - 1.0 > -1.0:
            return -np.inf
        return -np.inf
    if x >= 1.0:
        if 4.0 * N_effective * u - 1.0 > -1.0:
            return -np.inf
        return -np.inf

    a = 2.0 * N_effective * v - 1.0 # x**a
    b = 2.0 * N_effective * u - 1.0 # (1-x)**b
    term1 = a * _safe_log(x) # = log(x**a)
    term2 = b * _safe_log(1.0 - x) # = log((1-x)**b)
    ########################
    term3 = 2.0 * N_effective * s * x # = log(exp(2N*s*x))
    return term1 + term2 + term3 # = log(f(x))


def diffusion_log_unnormalized_v2(x: float, N_effective: float, s: float, u: float, v: float) -> float:
    """
    Log of unnormalized diffusion stationary density:
        f(x) ∝ x^{2N v - 1} (1-x)^{2N u - 1} exp[N s x]  -- SEEMS TO WORK LESS WEL!!

    We compute in log-space for stability. Returns -inf at boundaries when appropriate. 
    """
    if x <= 0.0:
        # Integrable if 4Nv > 0, otherwise zero mass at boundary bin by integration
        if 4.0 * N_effective * v - 1.0 > -1.0:
            return -np.inf
        return -np.inf
    if x >= 1.0:
        if 4.0 * N_effective * u - 1.0 > -1.0:
            return -np.inf
        return -np.inf

    a = 2.0 * N_effective * v - 1.0 # x**a
    b = 2.0 * N_effective * u - 1.0 # (1-x)**b
    term1 = a * _safe_log(x) # = log(x**a)
    term2 = b * _safe_log(1.0 - x) # = log((1-x)**b)
    term3 = 1.0 * N_effective * s * x # = log(exp(N*s*x))
    return term1 + term2 + term3 # = log(f(x))


def make_diffusion_integrand(N_effective: float, s: float, u: float, v: float) -> Tuple[Callable[[float], float], float]:
    """
    Create a numerically stable integrand g(x) = exp(log f(x) - c) and the offset c.

    We approximate c = max_x log f(x) on a dense grid for stability so integrals don't underflow.
    """
    xs = np.linspace(1e-8, 1 - 1e-8, 2000) # gridding the interval [0+eps,1-eps] with 2000 points   
    log_vals = np.array([diffusion_log_unnormalized(float(x), N_effective, s, u, v) for x in xs])
    c = float(np.max(log_vals))

    def g(x: float) -> float: # = exp(log f(x) - c) -- what we integrate
        val = diffusion_log_unnormalized(x, N_effective, s, u, v)
        return float(math.exp(val - c)) if np.isfinite(val) else 0.0

    return g, c


def discretize_diffusion_density(N_markov: int, N_effective: float, s: float, u: float, v: float) -> np.ndarray:
    """
    Discretize the diffusion stationary density onto N_markov+1 bins centered at i/N_markov.

    Each bin i integrates over [(i-0.5)/N, (i+0.5)/N], truncated to [0,1].
    We integrate the unnormalized density with a log offset and normalize afterwards.
    """
    integrand, c = make_diffusion_integrand(N_effective, s, u, v) # returns the integrand function and the offset c

    def integrate_interval(a: float, b: float) -> float:
        if b <= a:
            return 0.0
        # quad can handle mild endpoint singularities; provide tighter tolerances
        val, err = quad(integrand, a, b, epsabs=1e-12, epsrel=1e-10, limit=200) #: quad=SciPy’s adaptive 1D integrator. It integrates a callable over [a, b].
        return float(val) # notice--for normalized probabilities, the exp(c) factor cancels out when we divide by the total mass

    bin_masses = np.zeros(N_markov + 1, dtype=float) # initialize an array to store the masses of the bins

    for i in range(N_markov + 1):
        left = max(0.0, (i - 0.5) / N_markov)
        right = min(1.0, (i + 0.5) / N_markov)
        bin_masses[i] = integrate_interval(left, right)

    total = integrate_interval(0.0, 1.0)
    if total <= 0.0 or not np.isfinite(total):
        raise RuntimeError("Diffusion integral normalization failed (total mass non-positive).")

    probs = bin_masses / total # normalize by the total mass
    # Numerical cleanup
    probs = np.maximum(probs, 0.0) # ensure non-negative probabilities
    probs = probs / np.sum(probs) # normalize to sum to 1
    return probs


#Variant 2:  Each bin i integrates over [(i)/N, (i+1)/N], truncated to [0,1] for the markov value of i/N
def discretize_diffusion_density_v2(N_markov: int, N_effective: float, s: float, u: float, v: float) -> np.ndarray:
    """
    Discretize the diffusion stationary density onto N_markov+1 bins centered at i/N_markov.

    Each bin i integrates over [(i)/N, (i+1)/N], truncated to [0,1].
    We integrate the unnormalized density with a log offset and normalize afterwards.
    """
    integrand, c = make_diffusion_integrand(N_effective, s, u, v) # returns the integrand function and the offset c

    def integrate_interval(a: float, b: float) -> float:
        if b <= a:
            return 0.0
        # quad can handle mild endpoint singularities; provide tighter tolerances
        val, err = quad(integrand, a, b, epsabs=1e-12, epsrel=1e-10, limit=200) #: quad=SciPy’s adaptive 1D integrator. It integrates a callable over [a, b].
        return float(val) # notice--for normalized probabilities, the exp(c) factor cancels out when we divide by the total mass

    bin_masses = np.zeros(N_markov + 1, dtype=float) # initialize an array to store the masses of the bins

    for i in range(N_markov + 1):
        left = max(0.0, (i) / N_markov)
        right = min(1.0, (i + 1) / N_markov)
        bin_masses[i] = integrate_interval(left, right)

    total = integrate_interval(0.0, 1.0)
    if total <= 0.0 or not np.isfinite(total):
        raise RuntimeError("Diffusion integral normalization failed (total mass non-positive).")

    probs = bin_masses / total # normalize by the total mass
    # Numerical cleanup
    probs = np.maximum(probs, 0.0) # ensure non-negative probabilities
    probs = probs / np.sum(probs) # normalize to sum to 1
    return probs

# Variant 3: (evaluate f at x_i = i/N and normalize)
def discretize_diffusion_density_v3(N_markov: int, N_effective: float, s: float, u: float, v: float) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, N_markov + 1)
    # avoid exact 0/1 for numerical stability
    xs = np.clip(xs, 1e-12, 1 - 1e-12)
    log_vals = np.array([diffusion_log_unnormalized(float(x), N_effective, s, u, v) for x in xs])
    c = float(np.max(log_vals))
    vals = np.exp(log_vals - c)
    probs = vals / np.sum(vals)
    probs = np.maximum(probs, 0.0)
    probs = probs / np.sum(probs)
    return probs


# -----------------------------
# Comparison utilities
# -----------------------------

@dataclass
class ComparisonResult:
    L1: float
    KL_markov_to_diff: float
    KL_diff_to_markov: float
    N_effective: float
    KS_D: float


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-16
    p_ = p + eps
    q_ = q + eps
    p_ = p_ / np.sum(p_)
    q_ = q_ / np.sum(q_)
    return float(np.sum(p_ * (np.log(p_) - np.log(q_))) / math.log(2.0))  # bits


def compare_distributions(pi_markov: np.ndarray, pi_diffusion: np.ndarray, N_effective: float) -> ComparisonResult:
    L1 = float(np.sum(np.abs(pi_markov - pi_diffusion)))
    KL_md = kl_divergence(pi_markov, pi_diffusion)
    KL_dm = kl_divergence(pi_diffusion, pi_markov)
    D = ks_statistic(pi_markov, pi_diffusion)
    return ComparisonResult(L1=L1, KL_markov_to_diff=KL_md, KL_diff_to_markov=KL_dm, N_effective=N_effective, KS_D=D)


def ks_statistic(p: np.ndarray, q: np.ndarray) -> float:
    """Kolmogorov–Smirnov statistic D between two discrete distributions via CDFs."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / np.sum(p)
    q = q / np.sum(q)
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    D = float(np.max(np.abs(cdf_p - cdf_q)))
    return D


def find_best_Ne(
    pi_markov: np.ndarray,
    N_markov: int,
    s: float,
    mu: float,
    u: float | None = None,
    v: float | None = None,
    bounds: Tuple[float, float] | None = None,
    metric: str = "L1",
    discretizer: Optional[Callable[[int, float, float, float, float], np.ndarray]] = None,
) -> ComparisonResult:
    """
    Minimize distance between diffusion discretized distribution and π_M by varying N_effective.
    bounds: search interval for N_e (defaults to [0.25N, 4N]).
    metric: one of {"L1", "KL"} where KL means KL(π_M || π_D).
    """
    if u is None or v is None:
        u = mu
        v = mu

    if bounds is None:
        bounds = (max(1.0, 0.25 * N_markov), 4.0 * N_markov)

    if discretizer is None:
        discretizer = discretize_diffusion_density

    def objective(Ne: float) -> float:
        pi_diff = discretizer(N_markov, Ne, s, u, v)
        if metric == "L1":
            return float(np.sum(np.abs(pi_markov - pi_diff)))
        else:
            return kl_divergence(pi_markov, pi_diff)

    res = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-2})
    Ne_best = float(res.x)
    pi_diff_best = discretizer(N_markov, Ne_best, s, u, v)
    return compare_distributions(pi_markov, pi_diff_best, Ne_best)


# -----------------------------
# Plotting
# -----------------------------

def plot_markov_vs_diffusion(
    pi_markov: np.ndarray,
    pi_diffusion: np.ndarray,
    N: int,
    s: float,
    mu: float,
    Ne: float,
    title_suffix: str = "",
    metrics: Optional[ComparisonResult] = None,
):
  

    # High-contrast palette
    sns.set_style("white")
    colors = sns.color_palette("colorblind", 2)
    c_markov, c_diff = colors[0], colors[1]

    states = np.arange(N + 1)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    width = 0.4
    ax.bar(
        states - width / 2, pi_markov, width=width,
        label=f"Markov (N={N})",
        color=c_markov, edgecolor="none", alpha=0.9
    )
    ax.bar(
        states + width / 2, pi_diffusion, width=width,
        label=f"Diffusion (N_e={Ne:.2f})",
        color=c_diff, edgecolor="none", alpha=0.9
    )

    # Only two distributions are compared/visualized (Markov vs chosen diffusion)

    ax.set_xlabel('Number of "1" alleles')
    ax.set_ylabel('Probability')
    ax.set_title(f'Stationary distribution: Markov vs Diffusion\nN={N}, μ={mu}, s={s} {title_suffix}')
    ax.set_xlim(-0.5, N + 0.5)
    ax.set_xticks(np.arange(0, N + 1, max(1, N // 10)))
    # Metrics box (top-center)
    if metrics is not None:
        box_text = (
            f"KL(M||D): {metrics.KL_markov_to_diff:.4g} bits\n"
            f"L1: {metrics.L1:.4g}\n"
            f"KS D: {metrics.KS_D:.4g}"
        )
        ax.text(
            0.50,
            0.98,
            box_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=13,
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, linewidth=0.6),
        )

    # Legend at top-left
    ax.legend(loc='upper left', frameon=False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# -----------------------------
# Batch analysis function
# -----------------------------

def analyze_distances_vs_N(s: float, mu: float) -> pd.DataFrame:
    """
    Analyze how distances between Markov and diffusion distributions vary with population size N.
    
    Parameters:
    -----------
    s : float
        Selection coefficient
    mu : float
        Mutation rate
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: N, L1, KL_M_to_D, KL_D_to_M, KS_D
    """
    N_values = [50, 100, 150, 200, 500, 1000, 1500, 2000]
    results = []
    
    for N in N_values:
        # Compute Markov stationary distribution
        mc = WrightFisherMarkovChain(N=N, s=s, mu=mu, quiet=True)
        mc.construct_transition_matrix()
        pi_markov = np.asarray(mc.compute_stationary_distribution(method='direct'), dtype=float)
        
        # Compute diffusion stationary distribution (using N as Ne)
        pi_diffusion = discretize_diffusion_density(N, float(N), s, mu, mu)
        
        # Calculate distances
        result = compare_distributions(pi_markov, pi_diffusion, float(N))
        
        results.append({
            'N': N,
            'L1': result.L1,
            'KL_M_to_D': result.KL_markov_to_diff,
            'KL_D_to_M': result.KL_diff_to_markov,
            'KS_D': result.KS_D
        })
    
    return pd.DataFrame(results)


def plot_distances_vs_N(df: pd.DataFrame, s: float, mu: float):
    """
    Plot distance metrics vs population size N.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Output from analyze_distances_vs_N()
    s : float
        Selection coefficient (for title)
    mu : float
        Mutation rate (for title)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot all distance metrics
    ax.plot(df['N'], df['L1'], 'o-', label='L1', linewidth=2, markersize=6)
    ax.plot(df['N'], df['KL_M_to_D'], 's-', label='KL(M||D)', linewidth=2, markersize=6)
    ax.plot(df['N'], df['KL_D_to_M'], '^-', label='KL(D||M)', linewidth=2, markersize=6)
    ax.plot(df['N'], df['KS_D'], 'd-', label='KS D', linewidth=2, markersize=6)
    
    ax.set_xlabel('Population size N', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title(f'Markov vs Diffusion Distances\ns={s}, μ={mu}', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


# -----------------------------
# CLI
# -----------------------------

def run_comparison(
    N: int,
    s: float,
    mu: float,
    optimize_ne: bool = False,
    ne_bounds: Tuple[float, float] | None = None,
    plot: bool = True,
    diffusion_method: str = "v1",
) -> ComparisonResult:
    # 1) Markov stationary distribution
    mc = WrightFisherMarkovChain(N=N, s=s, mu=mu, quiet=True)
    mc.construct_transition_matrix()
    # Use direct eigen method for normal (non-optimized) run
    pi_markov = mc.compute_stationary_distribution(method='direct')
    pi_markov = np.asarray(pi_markov, dtype=float)

    # 2) Diffusion discretized distribution
    u = mu
    v = mu

    # Helper to select discretizer
    def _get_discretizer():
        if diffusion_method == "v2":
            return discretize_diffusion_density_v2
        if diffusion_method == "v3":
            return discretize_diffusion_density_v3
        return discretize_diffusion_density

    # Joint search: use the SAME Ne for Markov and Diffusion
    if optimize_ne:
        discretizer = _get_discretizer()
        # integer bounds for Ne search
        if ne_bounds is None:
            lo = max(2, int(max(1, 0.25 * N)))
            hi = int(max(lo, 4 * N))
        else:
            lo = max(2, int(np.floor(ne_bounds[0])))
            hi = int(np.ceil(ne_bounds[1]))
            if hi < lo:
                hi = lo

        best_res: Optional[ComparisonResult] = None
        best_pi_markov: Optional[np.ndarray] = None
        best_pi_diff: Optional[np.ndarray] = None
        best_Ne: Optional[int] = None

        for Ne_candidate in range(lo, hi + 1):
            mc_cand = WrightFisherMarkovChain(N=Ne_candidate, s=s, mu=mu, quiet=True)
            mc_cand.construct_transition_matrix()
            # Use power method during optimization for speed
            pi_markov_cand = np.asarray(mc_cand.compute_stationary_distribution(method='iterative'), dtype=float)
            pi_diff_cand = discretizer(Ne_candidate, float(Ne_candidate), s, u, v)
            res_cand = compare_distributions(pi_markov_cand, pi_diff_cand, float(Ne_candidate))
            if best_res is None or res_cand.L1 < best_res.L1:
                best_res = res_cand
                best_pi_markov = pi_markov_cand
                best_pi_diff = pi_diff_cand
                best_Ne = Ne_candidate

        assert best_res is not None and best_pi_markov is not None and best_pi_diff is not None and best_Ne is not None

        if plot:
            fig = plot_markov_vs_diffusion(
                best_pi_markov, best_pi_diff, best_Ne, s, mu, float(best_Ne),
                title_suffix=f"(best joint N_e, {diffusion_method})",
                metrics=best_res,
            )
            plt.show()
        return best_res

    Ne = float(N)
    pi_diff = _get_discretizer()(N, Ne, s, u, v)
    result = compare_distributions(pi_markov, pi_diff, Ne)
    if plot:
        fig = plot_markov_vs_diffusion(pi_markov, pi_diff, N, s, mu, Ne, metrics=result)
        plt.show()
    return result


N_default=100
s_default=0.005
mu_default=0.005

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare Markov stationary distribution to diffusion approximation.")
    p.add_argument('--N', type=int, default=N_default, help='Population size for Markov chain (states 0..N)')
    p.add_argument('--s', type=float, default=s_default, help='Selection coefficient s')
    p.add_argument('--mu', type=float, default=mu_default, help='Symmetric mutation rate μ (u=v=μ)')
    p.add_argument('--optimize-ne', action='store_true', help='Jointly search a single N_e used for both Markov and Diffusion')
    p.add_argument('--ne-min', type=float, default=None, help='Lower bound for N_e search (default 0.25N)')
    p.add_argument('--ne-max', type=float, default=None, help='Upper bound for N_e search (default 4N)')
    p.add_argument('--seed-by-diffusion', action='store_true', help='Seed joint search via diffusion-only coarse scan')
    p.add_argument('--no-plot', action='store_true', help='Disable plotting')
    return p.parse_args()


def main():
    args = parse_args()
    bounds = None
    if args.ne_min is not None or args.ne_max is not None:
        lo = args.ne_min if args.ne_min is not None else max(1.0, 0.25 * args.N)
        hi = args.ne_max if args.ne_max is not None else 4.0 * args.N
        bounds = (lo, hi)

    res = run_comparison(
        N=args.N,
        s=args.s,
        mu=args.mu,
        optimize_ne=args.optimize_ne,
        ne_bounds=bounds,
        plot=not args.no_plot,
    )

    print("=" * 80)
    print("MARKOV vs DIFFUSION STATIONARY DISTRIBUTION")
    print("=" * 80)
    print(f"Parameters (Markov):   N={args.N}, s={args.s}, μ={args.mu}")
    if args.optimize_ne:
        print(f"Best joint N_e:        {res.N_effective:.1f}")
    else:
        print(f"Diffusion N_e:         {res.N_effective:.1f}")
    print(f"L1 distance:           {res.L1:.6e}")
    print(f"KL(M || D) [bits]:     {res.KL_markov_to_diff:.6e}")
    #print(f"KL(D || M) [bits]:     {res.KL_diff_to_markov:.6e}")
    print(f"KS statistic D:        {res.KS_D:.6e}")


if __name__ == "__main__":
    main()


