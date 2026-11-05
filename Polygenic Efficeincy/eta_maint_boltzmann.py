#!/usr/bin/env python3
"""Boltzmann on the genotype-frequency simplex (not the allele cube).

This matches the potential on your slide (n-type Wright–Dirichlet):

    V(x) = −2N log W(x) + Σ_g (1 − α_g) log x_g
    α_g  = 4N μ_{→g}
    ρ(x) ∝ exp(−V(x)) = W(x)^{2N} Π_g x_g^{α_g − 1}

Hledík information is a KL on this simplex. Cost C(x) is already a function
of a genotype-frequency vector x, so no linkage-equilibrium conversion.

    conda activate miso
    python "Polygenic Efficeincy/eta_maint_boltzmann.py"
    python "Polygenic Efficeincy/eta_maint_boltzmann.py" --n 2 --N 50 --alpha 2 --Vs 0.0625

Alpha (diffusion, first version)
--------------------------------
We set α **directly** and use it on every genotype (symmetric Dirichlet):

    α_g ≡ α ,     μ_{→g} = α / (4N)   implied, not an independent input

    --alpha-mode direct   (default)  parent-independent α of the slide
    --alpha-mode cond     later: state-dependent cond-α from per-locus M

Neutral ρ_φ is Dirichlet(α,…,α). Because ω depends only on Hamming weight,
the observables (W, C, D, F, η) are exact on the (n+1)-layer Dirichlet

    X ~ Dirichlet(α C(n,0), …, α C(n,n)),   W = Σ_k X_k ω_k.

That is a reduction, not an approximation: we do not sample 2ⁿ genotypes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def log_binoms(n: int) -> np.ndarray:
    L = np.zeros(n + 1)
    for k in range(1, n + 1):
        L[k] = L[k - 1] + math.log(n - k + 1) - math.log(k)
    return L


def omega_layers(n: int, vs: float, z_star: float) -> np.ndarray:
    k = np.arange(n + 1, dtype=float)
    z = k / float(n)
    return np.exp(-((z - z_star) ** 2) / (2.0 * vs))


def sample_layer_dirichlet(
    log_alpha: float, logB: np.ndarray, rng: np.random.Generator, n_samples: int
) -> np.ndarray:
    """X ~ Dirichlet(α C(n,0), …, α C(n,n)), numerically stable for large n.

    log_alpha is log of the per-genotype Dirichlet exponent (log(α₀) − n log 2
    when α₀ = Kα is held fixed).
    """
    log_ak = log_alpha + logB
    m = logB.size
    logg = np.empty((n_samples, m))
    huge = log_ak > 700.0
    tiny = log_ak < -40.0
    big = (log_ak > 20.0) & ~huge
    small = ~huge & ~big & ~tiny
    if np.any(huge):
        logg[:, huge] = log_ak[huge]
    if np.any(tiny):
        logg[:, tiny] = -1e9
    if np.any(big):
        ak = np.exp(log_ak[big])
        g = ak + np.sqrt(ak) * rng.standard_normal((n_samples, int(ak.size)))
        logg[:, big] = np.log(np.maximum(g, 1e-300))
    if np.any(small):
        ak = np.exp(log_ak[small])
        g = rng.gamma(ak, size=(n_samples, int(ak.size)))
        logg[:, small] = np.log(np.maximum(g, 1e-300))
    logg -= np.max(logg, axis=1, keepdims=True)
    g = np.exp(logg)
    return g / np.sum(g, axis=1, keepdims=True)


def cost_bits_rows(x: np.ndarray, omega: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    """C(x) = D_KL(x′ ‖ x) in bits (log base 2); x shape (B, K).

    x'_g = x_g ω(g)/(x·ω) = x_g r_g, so
    C(x) = Σ_g x'_g log₂(x'_g / x_g) = Σ_g x_g r_g log₂(r_g).
    """
    p = np.maximum(x, eps)
    wbar = np.maximum(p @ omega, eps)
    r = np.maximum(omega / wbar[:, None], eps)
    xp = p * r
    return np.sum(xp * np.log2(np.maximum(xp, eps) / p), axis=1)


def logmeanexp(a: np.ndarray) -> float:
    m = float(np.max(a))
    return m + math.log(float(np.mean(np.exp(a - m))))


def expected_simpson(alpha: float, n: int, log_alpha: float | None = None) -> float:
    """E[Σ x_g²] = (α+1)/(Kα+1) for Dirichlet(α,…,α) on K=2ⁿ types.

    Pass log_alpha when α itself underflows (n ≳ 700 at fixed c).
    """
    la = math.log(alpha) if log_alpha is None else log_alpha
    if not math.isfinite(la):
        return float("nan")
    alpha0 = math.exp(la + n * math.log(2.0))
    a = math.exp(la)
    if not math.isfinite(alpha0):
        return 0.0
    if not (a > 0.0) or not math.isfinite(a):
        return 1.0 / (alpha0 + 1.0)
    return (a + 1.0) / (alpha0 + 1.0)


def log_alpha_from_simpson(c: float, n: int) -> float:
    """log α = log(1−c) − log(c 2ⁿ − 1). Requires 1/K < c < 1."""
    if n < 1:
        raise ValueError(f"n must be ≥ 1; got {n}")
    if not (0.0 < c < 1.0):
        raise ValueError(f"need 0 < c < 1; got c={c}")
    log_invK = -n * math.log(2.0)
    if c <= math.exp(log_invK):
        raise ValueError(f"need c > 1/K with K=2^{n}; got c={c}")
    log_cK = math.log(c) + n * math.log(2.0)
    if log_cK > 40.0:
        return math.log(1.0 - c) - log_cK
    den = c * math.exp(n * math.log(2.0)) - 1.0
    if den <= 0.0:
        raise ValueError(f"cK−1 must be > 0; got {den}")
    return math.log(1.0 - c) - math.log(den)


def alpha_from_simpson(c: float, n: int) -> float:
    """α(K) = (1−c)/(cK − 1) so E[S]=c. Requires 1/K < c < 1."""
    return math.exp(log_alpha_from_simpson(c, n))


def compute(
    n: int,
    N: float,
    vs: float,
    z_star: float,
    n_samples: int,
    seed: int,
    alpha: float,
    alpha_mode: str,
    alpha0: float | None = None,
    simpson_c: float | None = None,
) -> dict:
    if alpha_mode != "direct":
        raise NotImplementedError(
            "cond-α is not wired yet; use --alpha-mode direct and set --alpha."
        )
    if simpson_c is not None:
        log_alpha = log_alpha_from_simpson(simpson_c, n)
        alpha_used = math.exp(log_alpha)
        alpha0 = math.exp(log_alpha + n * math.log(2.0))
        scaling = "simpson_c_fixed"
    elif alpha0 is not None:
        if alpha0 <= 0:
            raise ValueError("α₀ must be > 0")
        log_alpha = math.log(alpha0) - n * math.log(2.0)
        alpha_used = math.exp(log_alpha)
        scaling = "alpha0_fixed"
    else:
        if alpha <= 0:
            raise ValueError("α must be > 0")
        log_alpha = math.log(alpha)
        alpha_used = alpha
        alpha0 = math.exp(log_alpha + n * math.log(2.0))
        scaling = "alpha_fixed"

    n_layers = n + 1
    omega = omega_layers(n, vs, z_star)
    logB = log_binoms(n)
    mu_in = alpha_used / (4.0 * N)

    rng = np.random.default_rng(seed)
    x = sample_layer_dirichlet(log_alpha, logB, rng, n_samples)
    W = np.maximum(x @ omega, 1e-300)
    logW = np.log(W)
    # ρ*/ρ_φ ∝ W^{2N}  (same law of W as the full 2ⁿ Dirichlet)
    log_tilt = (2.0 * N) * logW
    logZ = logmeanexp(log_tilt)  # log E_φ[W^{2N}]
    m = float(np.max(log_tilt))
    w = np.exp(log_tilt - m)
    p_star = w / float(np.sum(w))

    # D_KL(ρ*‖ρ_φ) in bits (log₂): (E_*[2N ln W] − ln Z) / ln 2
    D_nats = float(np.sum(p_star * log_tilt) - logZ)
    D_bits = D_nats / math.log(2.0)

    C = cost_bits_rows(x, omega)
    mean_C = float(np.sum(p_star * C))

    k_ploidy = 2.0  # slide uses 2N in V
    eta = float(D_bits / (k_ploidy * N * mean_C)) if mean_C > 0 else float("nan")

    z_layer = np.arange(n + 1, dtype=float) / float(n)
    mean_z = float(np.sum(p_star * (x @ z_layer)))
    mean_W = float(np.sum(p_star * W))
    mean_logW = float(np.sum(p_star * logW))  # ⟨log(x·ω)⟩_* nats
    # F* = ⟨log W⟩_* − D_nats/(2N) = log Z / (2N)
    F_star = float(logZ / (k_ploidy * N))
    ess = float(1.0 / np.sum(p_star**2))
    k_types = 2**n if n <= 40 else None

    return {
        "approx": "hamming_layer_dirichlet",
        "V": "−2N log W(x) + Σ (1−α_g) log x_g",
        "k_ploidy": 2,
        "n": n,
        "n_layers": n_layers,
        "K_genotypes": k_types,
        "N": N,
        "alpha_mode": alpha_mode,
        "scaling": scaling,
        "alpha": alpha_used,
        "log_alpha": log_alpha,
        "alpha0": alpha0,
        "simpson_c": expected_simpson(alpha_used, n, log_alpha=log_alpha),
        "mu_into_implied": mu_in,
        "Vs": vs,
        "z_star": z_star,
        "n_samples": n_samples,
        "ess": ess,
        "D_bits": D_bits,
        "mean_C_bits": mean_C,
        "eta_maint": eta,
        "mean_W_star": mean_W,
        "mean_logW_star": mean_logW,
        "F_star": F_star,
        "mean_z_star": mean_z,
        "omega_min": float(omega.min()),
        "omega_max": float(omega.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--N", type=float, default=50.0)
    p.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Dirichlet exponent α_g ≡ α (slide: α = 4N μ_→g). Direct mode.",
    )
    p.add_argument(
        "--alpha-mode",
        choices=("direct", "cond"),
        default="direct",
        help="direct: set α. cond: later, from per-locus mutation kernel.",
    )
    p.add_argument(
        "--alpha0",
        type=float,
        default=None,
        help="If set, hold α₀ = Kα = 2^n α fixed and use α(n)=α₀/2^n (overrides --alpha).",
    )
    p.add_argument(
        "--simpson-c",
        type=float,
        default=None,
        help="If set, hold E[S]=(α+1)/(Kα+1)=c and use α=(1-c)/(cK-1) (overrides --alpha/--alpha0).",
    )
    p.add_argument("--Vs", type=float, default=0.0625)
    p.add_argument("--z-star", type=float, default=0.5)
    p.add_argument("--samples", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    res = compute(
        n=args.n,
        N=args.N,
        vs=args.Vs,
        z_star=args.z_star,
        n_samples=args.samples,
        seed=args.seed,
        alpha=args.alpha,
        alpha_mode=args.alpha_mode,
        alpha0=args.alpha0,
        simpson_c=args.simpson_c,
    )
    print("Boltzmann on Hamming layers  ρ ∝ W^{2N} Π x_g^{α−1}  (exact reduction)")
    print(f"  n={res['n']}  layers={res['n_layers']}  K={res['K_genotypes']}  N={res['N']}")
    print(f"  scaling={res['scaling']}  α={res['alpha']:.6g}  α₀={res['alpha0']:.6g}  E[S]={res['simpson_c']:.6g}")
    print(f"  α-mode={res['alpha_mode']}  (μ_→g=α/(4N)={res['mu_into_implied']:.6g})")
    print(f"  Vs={res['Vs']}  z*={res['z_star']}  samples={res['n_samples']}  ESS={res['ess']:.4g}")
    print(f"  D      = KL(ρ* ‖ ρ_φ)   = {res['D_bits']:.6g} bits")
    print(f"  ⟨C⟩_*  = E_{{ρ*}}[C(x)]   = {res['mean_C_bits']:.6g} bits")
    print(f"  η_maint = D / (2 N ⟨C⟩_*) = {res['eta_maint']:.6g}   (k=2 from 2N in V)")
    print(f"  ⟨log W⟩_* = {res['mean_logW_star']:.6g} nats   ⟨W⟩_* = {res['mean_W_star']:.6g}")
    print(f"  F* = ⟨ln W⟩_* − D_nats/(2N) = log Z/(2N) = {res['F_star']:.6g} nats")
    print(f"  E[z]_* = {res['mean_z_star']:.6g}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2) + "\n")
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
