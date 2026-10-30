#!/usr/bin/env python3
"""F2b neutral references from cond-α with 2N exponents (W=1 on neutral simplex)."""

from __future__ import annotations

import numpy as np

from selection_diffusion_vs_chain import _normalize_log


def cond_alpha_exponents(
    X: np.ndarray,
    M_mut: np.ndarray,
    N: int,
    *,
    alpha_mode: str = "cond",
    factor: float = 2.0,
) -> np.ndarray:
    """α_g(x) with prefactor factor·N (default 2N for Wright exponents).

    X shape (..., 4). Returns same leading shape + (4,).
    """
    X = np.asarray(X, dtype=float)
    single = X.ndim == 1
    if single:
        X = X[None, :]
    XM = X @ M_mut
    diag_m = np.diag(M_mut)
    influx = XM - X * diag_m[None, :]
    scale = float(factor) * float(N)
    if alpha_mode == "cond":
        denom = np.clip(1.0 - X, 1e-300, None)
        alpha = scale * influx / denom
    elif alpha_mode == "influx":
        alpha = scale * influx
    else:
        raise ValueError(f"unknown alpha_mode {alpha_mode!r}")
    return alpha[0] if single else alpha


def _neutral_log_density_fixed_alpha(fracs: np.ndarray, alpha_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """log φ(x) = Σ_g (α_g* − 1) log x_g on interior (W=1)."""
    interior = np.all(fracs > 0.0, axis=1)
    log_p = np.full(fracs.shape[0], -np.inf, dtype=float)
    if np.any(interior):
        log_p[interior] = np.sum(
            (alpha_star[None, :] - 1.0) * np.log(fracs[interior]),
            axis=1,
        )
    return log_p, interior


def _neutral_log_density_state_alpha(fracs: np.ndarray, M_mut: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    """log φ(x) = Σ_g (α_g^cond(x) − 1) log x_g with 2N cond-α."""
    interior = np.all(fracs > 0.0, axis=1)
    log_p = np.full(fracs.shape[0], -np.inf, dtype=float)
    if np.any(interior):
        alpha = cond_alpha_exponents(fracs[interior], M_mut, N, factor=2.0)
        log_p[interior] = np.sum((alpha - 1.0) * np.log(fracs[interior]), axis=1)
    return log_p, interior


def build_phi_f2b_a(
    fracs: np.ndarray,
    M_mut: np.ndarray,
    N: int,
    phi_chain: np.ndarray,
) -> np.ndarray:
    """F2b-A: fixed exponents α_g* = α_g^cond( x̄ ) with x̄ = E_{φ_chain}[x]."""
    x_ref = np.asarray(phi_chain, dtype=float) @ np.asarray(fracs, dtype=float)
    alpha_star = cond_alpha_exponents(x_ref, M_mut, N, factor=2.0)
    log_p, interior = _neutral_log_density_fixed_alpha(fracs, alpha_star)
    if not np.any(interior):
        raise RuntimeError("F2b-A: no interior lattice points")
    q = np.zeros_like(log_p)
    q[interior] = _normalize_log(log_p, interior)[interior]
    s = float(q.sum())
    if s <= 0.0:
        raise RuntimeError("F2b-A reference has zero mass")
    return q / s


def build_phi_f2b_b(fracs: np.ndarray, M_mut: np.ndarray, N: int) -> np.ndarray:
    """F2b-B: normalized 2N cond-α neutral density on the lattice (state-dependent)."""
    log_p, interior = _neutral_log_density_state_alpha(fracs, M_mut, N)
    if not np.any(interior):
        raise RuntimeError("F2b-B: no interior lattice points")
    q = np.zeros_like(log_p)
    q[interior] = _normalize_log(log_p, interior)[interior]
    s = float(q.sum())
    if s <= 0.0:
        raise RuntimeError("F2b-B reference has zero mass")
    return q / s


def _normalize_pmf(p: np.ndarray) -> np.ndarray:
    s = float(np.sum(p))
    if s <= 0.0:
        raise RuntimeError("cannot normalize zero-mass pmf")
    return np.asarray(p, dtype=float) / s


def build_phi_f2b_b_epsilon(
    fracs: np.ndarray,
    M_mut: np.ndarray,
    N: int,
    *,
    eps: float | None = None,
) -> np.ndarray:
    """Cond-α density on the **full** lattice: zeros replaced by ε=1/N before log/α."""
    if eps is None:
        eps = 1.0 / float(N)
    x_eff = np.maximum(np.asarray(fracs, dtype=float), eps)
    alpha = cond_alpha_exponents(x_eff, M_mut, N, factor=2.0)
    log_p = np.sum((alpha - 1.0) * np.log(x_eff), axis=1)
    return _normalize_pmf(np.exp(log_p - np.max(log_p)))


def build_phi_boundary_smeared(
    fracs: np.ndarray,
    phi_chain: np.ndarray,
    q_interior: np.ndarray,
    P_col: np.ndarray,
) -> np.ndarray:
    """One neutral generation from interior cond-α mass; boundary from mutation influx.

    Interior mass and shape match q_interior scaled to chain interior weight;
    boundary mass matches chain boundary weight with shape from P_col @ ψ₀.
    """
    interior = np.all(fracs > 0.0, axis=1)
    boundary = ~interior
    m_int = float(phi_chain[interior].sum())
    m_bnd = float(phi_chain[boundary].sum())

    psi0 = np.zeros_like(q_interior)
    qi = np.asarray(q_interior, dtype=float)
    psi0[interior] = qi[interior]
    s0 = float(psi0.sum())
    if s0 <= 0.0:
        raise RuntimeError("interior reference has zero mass")
    psi0 /= s0

    smeared = np.asarray(P_col, dtype=float) @ psi0
    bnd_shape = np.clip(smeared[boundary], 0.0, None)
    sb = float(bnd_shape.sum())
    if sb <= 0.0:
        raise RuntimeError("mutation smear gives zero boundary mass")

    out = np.zeros_like(phi_chain)
    out[interior] = m_int * qi[interior] / float(qi[interior].sum())
    out[boundary] = m_bnd * bnd_shape / sb
    return _normalize_pmf(out)


def build_phi_shell_blend(
    fracs: np.ndarray,
    phi_chain: np.ndarray,
    q_interior: np.ndarray,
    P_col: np.ndarray,
    *,
    n_sweep: int = 1,
) -> np.ndarray:
    """Repeated neutral smearing: ψ ← P_col ψ starting from interior q (integrate toward boundary)."""
    interior = np.all(fracs > 0.0, axis=1)
    psi = np.zeros_like(q_interior)
    qi = np.asarray(q_interior, dtype=float)
    psi[interior] = qi[interior]
    psi = _normalize_pmf(psi)
    P = np.asarray(P_col, dtype=float)
    for _ in range(max(1, int(n_sweep))):
        psi = P @ psi
        s = float(psi.sum())
        if s > 0:
            psi /= s
    # Rescale interior/boundary totals to match chain
    boundary = ~interior
    out = np.zeros_like(phi_chain)
    qi_sum = float(qi[interior].sum())
    out[interior] = float(phi_chain[interior].sum()) * psi[interior] / max(float(psi[interior].sum()), 1e-300)
    bnd = np.clip(psi[boundary], 0.0, None)
    sb = float(bnd.sum())
    if sb > 0:
        out[boundary] = float(phi_chain[boundary].sum()) * bnd / sb
    else:
        out[boundary] = phi_chain[boundary]
    return _normalize_pmf(out)


def log_weights_cond_alpha(
    fracs: np.ndarray,
    omega: np.ndarray,
    M_mut: np.ndarray,
    N: int,
    *,
    eps: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Unnormalized log weight ℓ(x) = -V(x) on the genotype lattice.

    Continuous potential (cond-α, haploid N on selection, 2N on mutation):

        V(x) = -N \log W(x) + \sum_g \bigl(1 - \alpha_g^{\mathrm{cond}}(x)\bigr)\log x_g

    with W(x)=\sum_g x_g \omega_g and
    \alpha_g = 2N \cdot \mathrm{influx}_g / (1-x_g).

    Discrete partition function: Z \approx \sum_x \exp(\ell(x)) on lattice points.
    """
    fracs = np.asarray(fracs, dtype=float)
    omega = np.asarray(omega, dtype=float)
    if eps is None:
        interior = np.all(fracs > 0.0, axis=1)
        ell = np.full(fracs.shape[0], -np.inf, dtype=float)
        if not np.any(interior):
            return ell, interior
        xi = fracs[interior]
        alpha = cond_alpha_exponents(xi, M_mut, N, factor=2.0)
        w = np.clip(xi @ omega, 1e-300, None)
        ell[interior] = float(N) * np.log(w) + np.sum((alpha - 1.0) * np.log(xi), axis=1)
        return ell, interior

    x_eff = np.maximum(fracs, float(eps))
    alpha = cond_alpha_exponents(x_eff, M_mut, N, factor=2.0)
    w = np.clip(x_eff @ omega, 1e-300, None)
    ell = float(N) * np.log(w) + np.sum((alpha - 1.0) * np.log(x_eff), axis=1)
    valid = np.isfinite(ell)
    return ell, valid


def log_Z_discrete(ell: np.ndarray, mask: np.ndarray | None = None) -> float:
    """log Z = log sum_{x in mask} exp(ell(x)) (stable log-sum-exp)."""
    ell = np.asarray(ell, dtype=float)
    if mask is not None:
        ell = ell[np.asarray(mask, dtype=bool)]
    ell = ell[np.isfinite(ell)]
    if ell.size == 0:
        return float("-inf")
    m = float(np.max(ell))
    return m + float(np.log(np.sum(np.exp(ell - m))))


def log_Z_cond_alpha(
    fracs: np.ndarray,
    omega: np.ndarray,
    M_mut: np.ndarray,
    N: int,
    *,
    mode: str = "interior",
) -> dict[str, float]:
    """Compute log Z and log Z/N for cond-α Boltzmann weights on the lattice."""
    if mode == "interior":
        ell, mask = log_weights_cond_alpha(fracs, omega, M_mut, N, eps=None)
    elif mode == "epsilon":
        ell, mask = log_weights_cond_alpha(fracs, omega, M_mut, N, eps=1.0 / float(N))
    elif mode == "full":
        ell, mask = log_weights_cond_alpha(fracs, omega, M_mut, N, eps=1.0 / float(N))
        mask = np.ones(fracs.shape[0], dtype=bool)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    log_z = log_Z_discrete(ell, mask)
    return {
        "log_Z": log_z,
        "log_Z_over_N": log_z / float(N),
        "n_states": float(np.sum(mask)),
    }


def compare_phi_refs(
    phi_chain: np.ndarray,
    refs: dict[str, np.ndarray],
    fracs: np.ndarray,
) -> list[dict]:
    """TV/KL/corr on full lattice and interior-only renormalized."""
    from simplex import kl_divergence_bits

    interior = np.all(fracs > 0.0, axis=1)

    def renorm_int(p: np.ndarray) -> np.ndarray:
        q = np.zeros_like(p)
        q[interior] = p[interior]
        s = float(q.sum())
        return q / s if s > 0 else q

    pc = np.asarray(phi_chain, dtype=float)
    pc_int = renorm_int(pc)
    rows = []
    for name, ref in refs.items():
        pr = np.asarray(ref, dtype=float)
        pr = pr / pr.sum()
        rows.append({
            "name": name,
            "tv_full": 0.5 * float(np.sum(np.abs(pc - pr))),
            "kl_full_bits": kl_divergence_bits(pc, pr),
            "bnd_mass_chain": float(1.0 - pc[interior].sum()),
            "bnd_mass_ref": float(1.0 - pr[interior].sum()),
            "tv_int": 0.5 * float(np.sum(np.abs(pc_int - renorm_int(pr)))),
            "kl_int_bits": kl_divergence_bits(pc_int, renorm_int(pr)),
            "corr_int": float(np.corrcoef(pc_int[interior], renorm_int(pr)[interior])[0, 1]),
        })
    return rows


def kl_divergence_nats(p: np.ndarray, q: np.ndarray, eps: float = 1e-16) -> float:
    p_ = np.asarray(p, dtype=float) + eps
    q_ = np.asarray(q, dtype=float) + eps
    p_ /= p_.sum()
    q_ /= q_.sum()
    return float(np.sum(p_ * (np.log(p_) - np.log(q_))))


def expectation(rho: np.ndarray, values: np.ndarray) -> float:
    return float(np.sum(rho * values))


def free_f2b(
    psi: np.ndarray,
    wstar_ln: np.ndarray,
    phi_ref: np.ndarray,
    N: int,
) -> float:
    """F2b = ⟨ln W⟩_ψ − (1/N) D_KL(ψ||φ_ref) [nats]."""
    return expectation(psi, wstar_ln) - kl_divergence_nats(psi, phi_ref) / float(N)


def kl_to_ref_nats(psi: np.ndarray, phi_ref: np.ndarray) -> float:
    return kl_divergence_nats(psi, phi_ref)


def eps_star_dip_prediction(
    phi: np.ndarray,
    fracs: np.ndarray,
    s1: float,
    s2: float,
    *,
    eps_lin: float = 0.0,
) -> float:
    r"""Predicted KL-dip location (linearized):

    \[
    \varepsilon^* \approx
    -\frac{\mathrm{Cov}_\varphi\!\bigl[\sum_g x_g \omega_g
    \;\;(\omega_{11}\to\omega_{11}/e^{\varepsilon}),\, x_{11}\bigr]}
    {\mathrm{Var}_\varphi(x_{11})}
    \]

    Evaluate the bracket at ``eps_lin`` (default 0) under neutral ``phi``.
    """
    from simplex import build_sigma, fitness_exp

    omega = fitness_exp(build_sigma(s1, s2, float(eps_lin)))
    x11 = fracs[:, 3]
    # All four haplotype fractions; g11 weight uses ω_11/e^ε (epistasis split).
    w = omega.copy()
    w[3] = omega[3] / np.exp(float(eps_lin))
    B = np.asarray(fracs, dtype=float) @ w
    Eb = float(np.sum(phi * B))
    Ex = float(np.sum(phi * x11))
    cov = float(np.sum(phi * (B - Eb) * (x11 - Ex)))
    var = float(np.sum(phi * (x11 - Ex) ** 2))
    if var <= 0.0:
        return float("nan")
    return -cov / var
