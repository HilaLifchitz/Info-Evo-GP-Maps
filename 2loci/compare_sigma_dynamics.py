#!/usr/bin/env python3
r"""
Compare two 2×2 Sigma matrices on the discrete simplex chain.

For each ψ(t) (distribution on the N-locus simplex grid), tracks:

  ⟨W*⟩_ψ = E_ψ[ln(x·ω_Σ)]   natural log mean fitness under Σ [nats]
  KL(ψ||φ*)        bits, distance from neutral equilibrium φ*
  KL(ψ||ψ*)       bits, distance from selection equilibrium
  F₁(ψ)           ⟨W⟩_ψ + H(ψ)/N       (W incl. mutation; nats, haploid /N)
  F₂(ψ)           ⟨W*⟩_ψ − KL(ψ||φ*)/N     (φ* = chain neutral equilibrium)
  F₃(ψ)           ⟨W*⟩_ψ − KL(ψ||q_cond)/N  (q_cond = cond-α diffusion ref, Σ=0)
  F₄(ψ)           −(1/N) D_KL(ψ||ψ*) [nats]  (negative distance to selection equilibrium)
  C(ψ)              ⟨C(x)⟩_ψ  with  C(x) = Σ_g x_g · r_g · log₂(r_g),
                    r_g = ω(g) / (x·ω)   where x·ω = mean fitness at x

Also rates ΔKL/Δt, ΔF₁/Δt and normalized by C(t):  (ΔD/C), (ΔF₁/C).

Run:
  cd 2loci
  python compare_sigma_dynamics.py
  python compare_sigma_dynamics.py --config compare_sigma_config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from simplex import (  # noqa: E402
    build_transition_matrix_row_stochastic,
    column_stochastic,
    compute_phi,
    fitness_exp,
    log_mean_fitness_genotype_simplex,
    kl_divergence_bits,
    mutation_matrix_asymmetric,
    params_from_config,
    try_selection_equilibrium,
    load_config,
    consecutive_rates,
    w_potential_per_state,
    free_fitness_1,
)
from selection_diffusion_vs_chain import diffusion_logdensity, _normalize_log  # noqa: E402
from selection_diffusion_vs_chain import Setting  # noqa: E402

# =============================================================================
# Default: two landscapes to compare (edit or use YAML)
# =============================================================================
N_MARKOV = 40
R = 0.0
U = V = 1.0e-2
TIMES = list(range(0, 101, 5)) + [150, 200, 300, 500]

SIGMA_A = np.array([[1.0e-2, 1.0e-3], [1.0e-3, -1.0e-2]], dtype=float)
SIGMA_B = np.array([[1.0e-2, -1.0e-3], [-1.0e-3, -1.0e-2]], dtype=float)
LABEL_A = "Sigma_A (+)"
LABEL_B = "Sigma_B (-)"

OUT_DIR = SCRIPT_DIR / "plots" / "compare_sigma"
PERGEN_MAX_T = 200


@dataclass
class Trajectory:
    label: str
    sigma: np.ndarray
    times: list[int]
    kl_eq_bits: float
    mean_log_fit: np.ndarray
    kl_phi_bits: np.ndarray
    kl_star_bits: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    f3: np.ndarray
    f4: np.ndarray
    c_cost: np.ndarray
    kl_phi_over_c: np.ndarray
    rate_times: np.ndarray
    rate_kl_phi: np.ndarray
    rate_kl_star: np.ndarray
    rate_f1: np.ndarray
    rate_f2: np.ndarray
    rate_f3: np.ndarray
    rate_kl_phi_over_c: np.ndarray
    rate_kl_star_over_c: np.ndarray
    rate_f1_over_c: np.ndarray
    rate_f2_over_c: np.ndarray
    rate_f3_over_c: np.ndarray
    pergen_times: np.ndarray
    pergen_kl_phi_bits: np.ndarray
    pergen_c_cost: np.ndarray


def log_mean_fitness_nats(fracs: np.ndarray, omega: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    """W*(x) = ln(x·ω) — natural log of mean genotype fitness at x."""
    return log_mean_fitness_genotype_simplex(fracs, omega, eps=eps)


def kl_divergence_nats(p: np.ndarray, q: np.ndarray, eps: float = 1e-16) -> float:
    """D_KL(p||q) in nats (natural log)."""
    p_ = np.asarray(p, dtype=float) + eps
    q_ = np.asarray(q, dtype=float) + eps
    p_ /= p_.sum()
    q_ /= q_.sum()
    return float(np.sum(p_ * (np.log(p_) - np.log(q_))))


def cond_alpha_reference_pmf(
    fracs: np.ndarray,
    M_mut: np.ndarray,
    N: int,
    *,
    alpha_mode: str = "cond",
) -> np.ndarray:
    """Normalized cond-α reference q_cond on the lattice (neutral W=1, haploid N in α)."""
    omega0 = fitness_exp(np.zeros((2, 2)))
    st = Setting("neutral", N=N, s1=0.0, s2=0.0, eps12=0.0, u=0.0, v=0.0, r=0.0)
    log_p, interior = diffusion_logdensity(st, fracs, omega0, M_mut, alpha_mode)
    q = np.zeros_like(log_p)
    if np.any(interior):
        q[interior] = _normalize_log(log_p, interior)[interior]
    s = float(q.sum())
    if s <= 0.0:
        raise RuntimeError("cond-α reference has zero mass on interior")
    q /= s
    return q


def free_f2(psi: np.ndarray, wstar_ln: np.ndarray, phi: np.ndarray, N: int) -> float:
    """F₂ = ⟨W*⟩_ψ − KL(ψ||φ*)/N; W* = ln(x·ω_Σ), φ* = neutral equilibrium."""
    return expectation(psi, wstar_ln) - kl_divergence_nats(psi, phi) / float(N)


def free_f3(psi: np.ndarray, wstar_ln: np.ndarray, q_cond: np.ndarray, N: int) -> float:
    """F₃ = ⟨W*⟩_ψ − KL(ψ||q_cond)/N; q_cond = cond-α diffusion reference (not chain φ*)."""
    return expectation(psi, wstar_ln) - kl_divergence_nats(psi, q_cond) / float(N)


def free_f4(psi: np.ndarray, psi_star: np.ndarray, N: int) -> float:
    """F₄ = −(1/N) D_KL(ψ||ψ*) [nats]; maximized (→ 0) at selection equilibrium."""
    return -kl_divergence_nats(psi, psi_star) / float(N)


def _monotonicity_report(name: str, values: np.ndarray) -> None:
    d = np.diff(values)
    ok = bool(np.all(d >= -1e-12))
    status = "monotone ↑" if ok else f"NOT monotone ({int(np.sum(d < -1e-12))} decreases)"
    print(f"  {name}: {status}", flush=True)
    if not ok and name.endswith("F2"):
        worst = int(np.argmin(d))
        print(
            f"    largest F2 drop between sampled times: "
            f"t={worst}→{worst + 1}, ΔF2={d[worst]:.3e}",
            flush=True,
        )


def _monotonicity_report_decreasing(name: str, values: np.ndarray) -> None:
    d = np.diff(values)
    ok = bool(np.all(d <= 1e-12))
    status = "monotone ↓" if ok else f"NOT monotone ({int(np.sum(d > 1e-12))} increases)"
    print(f"  {name}: {status}", flush=True)


def cost_per_state(fracs: np.ndarray, omega: np.ndarray, eps: float = 1e-300) -> np.ndarray:
    """C(x) = Σ_g x_g r_g log₂(r_g),  r_g = ω(g) / (x·ω).

    x·ω = Σ_g x_g ω(g) is the **mean fitness** of genotype distribution x.
    r_g is each genotype's fitness relative to that mean.
    """
    x = np.maximum(fracs, eps)
    wbar = np.maximum(x @ omega, eps)
    r = omega / wbar[:, None]
    r = np.maximum(r, eps)
    return np.sum(x * r * np.log2(r), axis=1)


def expectation(rho: np.ndarray, values: np.ndarray) -> float:
    return float(np.sum(rho * values))


def track_d_c_per_generation(
    psi0: np.ndarray,
    P_col: np.ndarray,
    phi: np.ndarray,
    c_state: np.ndarray,
    max_t: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D(t)=KL(ψ(t)||φ*) and C(t) at every generation t=0..max_t."""
    d = np.empty(max_t + 1, dtype=float)
    c = np.empty(max_t + 1, dtype=float)
    psi = psi0.copy()
    d[0] = kl_divergence_bits(psi, phi)
    c[0] = expectation(psi, c_state)
    for t in range(max_t):
        psi = P_col @ psi
        s = float(psi.sum())
        if s > 0:
            psi /= s
        d[t + 1] = kl_divergence_bits(psi, phi)
        c[t + 1] = expectation(psi, c_state)
    return np.arange(max_t + 1, dtype=float), d, c


def _make_cfg(
    sigma: np.ndarray,
    N: int,
    u: float,
    v: float,
    r: float,
    times: list[int],
) -> dict:
    return {
        "_config_dir": str(SCRIPT_DIR),
        "N": N,
        "r": float(r),
        "u": [float(u), float(u)],
        "v": [float(v), float(v)],
        "sigma": sigma.tolist(),
        "times": times,
        "n_jobs": -1,
        "neutral_cache_dir": "cache",
        "neutral": False,
        "top_k": 3,
        "plot_dynamics": False,
        "compute_f2": False,
        "ld_top_k_times": "first_last",
        "debug_kl": False,
    }


def _rates_over_c(
    times: list[int],
    values: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discrete rate Δvalue/Δt and same rate divided by C at interval end."""
    rt, rate = consecutive_rates(times, values)
    t_to_i = {int(t): i for i, t in enumerate(times)}
    c_end = np.array([c[t_to_i[int(t)]] for t in rt])
    with np.errstate(divide="ignore", invalid="ignore"):
        rate_over_c = np.where(c_end > 1e-300, rate / c_end, np.nan)
    return rt, rate, rate_over_c


def run_one_sigma(
    label: str,
    sigma: np.ndarray,
    *,
    N: int,
    u: float,
    v: float,
    r: float,
    times: list[int],
    pergen_max_t: int = PERGEN_MAX_T,
) -> Trajectory:
    sigma = np.asarray(sigma, dtype=float)
    if sigma.shape != (2, 2):
        raise ValueError(f"{label}: Sigma must be 2×2, got {sigma.shape}")

    times_u = sorted(set(int(t) for t in times))
    if 0 not in times_u:
        times_u = [0] + times_u

    M_mut = mutation_matrix_asymmetric(np.array([u, u]), np.array([v, v]))
    params0 = params_from_config(_make_cfg(np.zeros((2, 2)), N, u, v, r, [0]), SCRIPT_DIR)
    phi, _, fracs, counts, _ = compute_phi(params0, M_mut)

    params = params_from_config(_make_cfg(sigma, N, u, v, r, times_u), SCRIPT_DIR)
    omega = fitness_exp(params.Sigma)
    wstar = log_mean_fitness_nats(fracs, omega)
    w_ps = w_potential_per_state(fracs, omega, M_mut, N)
    q_cond = cond_alpha_reference_pmf(fracs, M_mut, N)
    c_state = cost_per_state(fracs, omega)

    print(f"\n--- {label}: building transition matrix ---", flush=True)
    T_row, _, _ = build_transition_matrix_row_stochastic(
        N, omega, M_mut, r, n_jobs=-1, label=label[:20], verbose=True,
    )
    P_col = column_stochastic(T_row)

    psi_star, resid, ok = try_selection_equilibrium(P_col)
    if not ok or psi_star is None:
        raise RuntimeError(f"{label}: selection equilibrium ψ* not found (residual={resid})")
    print(f"  ψ* residual ||Pψ*−ψ*||₁ = {resid:.3e}", flush=True)
    kl_eq_bits = kl_divergence_bits(psi_star, phi)

    mean_log_fit: list[float] = []
    kl_phi: list[float] = []
    kl_star: list[float] = []
    f1: list[float] = []
    f2: list[float] = []
    f3: list[float] = []
    f4: list[float] = []
    c_vals: list[float] = []

    psi = phi.copy()
    prev_t = 0
    for t in times_u:
        if t > prev_t:
            for _ in range(t - prev_t):
                psi = P_col @ psi
                s = float(psi.sum())
                if s > 0:
                    psi /= s
            prev_t = t
        mean_log_fit.append(expectation(psi, wstar))
        kl_phi.append(kl_divergence_bits(psi, phi))
        kl_star.append(kl_divergence_bits(psi, psi_star))
        f1.append(free_fitness_1(psi, w_ps, N))
        f2.append(free_f2(psi, wstar, phi, N))
        f3.append(free_f3(psi, wstar, q_cond, N))
        f4.append(free_f4(psi, psi_star, N))
        c_vals.append(expectation(psi, c_state))

    pg_times, pg_d, pg_c = track_d_c_per_generation(phi.copy(), P_col, phi, c_state, pergen_max_t)

    t_arr = times_u
    ml = np.asarray(mean_log_fit)
    kp = np.asarray(kl_phi)
    ks = np.asarray(kl_star)
    f1a = np.asarray(f1)
    f2a = np.asarray(f2)
    f3a = np.asarray(f3)
    f4a = np.asarray(f4)
    ca = np.asarray(c_vals)
    kl_over_c = np.where(ca > 1e-300, kp / ca, np.nan)

    rt, r_kp, r_kp_c = _rates_over_c(t_arr, kp, ca)
    _, r_ks, r_ks_c = _rates_over_c(t_arr, ks, ca)
    _, r_f1, r_f1_c = _rates_over_c(t_arr, f1a, ca)
    _, r_f2, r_f2_c = _rates_over_c(t_arr, f2a, ca)
    _, r_f3, r_f3_c = _rates_over_c(t_arr, f3a, ca)

    _monotonicity_report(f"{label} F1", f1a)
    _monotonicity_report(f"{label} F2", f2a)
    _monotonicity_report(f"{label} F3", f3a)
    _monotonicity_report(f"{label} F4", f4a)

    print(
        f"  KL_eq=KL(ψ*||φ)={kl_eq_bits:.4f} bits | "
        f"C(ψ*)={expectation(psi_star, c_state):.4g} | "
        f"KL(φ)=0, KL(ψ*)={kl_star[0]:.4g} at t=0",
        flush=True,
    )

    return Trajectory(
        label=label,
        sigma=sigma,
        times=t_arr,
        kl_eq_bits=kl_eq_bits,
        mean_log_fit=ml,
        kl_phi_bits=kp,
        kl_star_bits=ks,
        f1=f1a,
        f2=f2a,
        f3=f3a,
        f4=f4a,
        c_cost=ca,
        kl_phi_over_c=kl_over_c,
        rate_times=rt,
        rate_kl_phi=r_kp,
        rate_kl_star=r_ks,
        rate_f1=r_f1,
        rate_f2=r_f2,
        rate_f3=r_f3,
        rate_kl_phi_over_c=r_kp_c,
        rate_kl_star_over_c=r_ks_c,
        rate_f1_over_c=r_f1_c,
        rate_f2_over_c=r_f2_c,
        rate_f3_over_c=r_f3_c,
        pergen_times=pg_times,
        pergen_kl_phi_bits=pg_d,
        pergen_c_cost=pg_c,
    )


def _write_csv(path: Path, traj: Trajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t", "wstar_nats", "kl_phi_bits", "kl_star_bits",
            "f1", "f2", "f3", "f4", "c_cost", "kl_phi_over_c", "kl_eq_bits",
        ])
        for i, t in enumerate(traj.times):
            w.writerow([
                t, traj.mean_log_fit[i], traj.kl_phi_bits[i], traj.kl_star_bits[i],
                traj.f1[i], traj.f2[i], traj.f3[i], traj.f4[i], traj.c_cost[i],
                traj.kl_phi_over_c[i], traj.kl_eq_bits,
            ])
    print(f"  wrote {path}", flush=True)


def _plot_delta_d_and_c(
    ax,
    traj: Trajectory,
    *,
    N: int,
    color_d: str,
    color_c: str,
    set_ylim: bool = True,
) -> tuple[float, float]:
    """Per-generation ΔD and N·C(t) on one shared y-axis. Returns (ymin, ymax) used."""
    d = traj.pergen_kl_phi_bits
    t = traj.pergen_times
    delta_d = np.diff(d)
    t_delta = t[1:]
    c_scaled = N * traj.pergen_c_cost

    line_d, = ax.plot(
        t_delta, delta_d, "-", lw=1.2, color=color_d,
        label=r"$\Delta D = D(t\!+\!1)-D(t)$ [bits/gen]",
    )
    line_c, = ax.plot(
        t, c_scaled, "--", lw=1.2, color=color_c,
        label=rf"$N\,C(t)$",
    )
    ax.axhline(0.0, color="0.5", ls=":", lw=0.8, alpha=0.5)
    ax.set_ylabel("value", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=[line_d, line_c], fontsize=7, loc="upper right")
    ax.set_xlabel("generation $t$")
    ax.set_title(traj.label, fontsize=10)
    ax.set_xlim(t[0], t[-1])
    ylo = min(float(np.min(delta_d)), float(np.min(c_scaled)))
    yhi = max(float(np.max(delta_d)), float(np.max(c_scaled)))
    pad = 0.05 * (yhi - ylo) if yhi > ylo else 1e-6
    ylo, yhi = ylo - pad, yhi + pad
    if set_ylim:
        ax.set_ylim(ylo, yhi)
    return ylo, yhi


def plot_comparison(
    tr_a: Trajectory, tr_b: Trajectory, out_dir: Path,
    *, N: int, u: float, r: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 2, figsize=(12, 13))

    ax = axes[0, 0]
    ax.plot(tr_a.times, tr_a.mean_log_fit, "o-", lw=1.5, ms=3, label=tr_a.label)
    ax.plot(tr_b.times, tr_b.mean_log_fit, "s--", lw=1.5, ms=3, label=tr_b.label)
    ax.set_ylabel(r"$\langle W^* \rangle_\psi = E_\psi[\ln(x\!\cdot\!\omega_\Sigma)]$ [nats]", fontsize=9)
    ax.set_xlabel("generation $t$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(tr_a.times, tr_a.kl_phi_bits, "o-", label=tr_a.label)
    ax.plot(tr_b.times, tr_b.kl_phi_bits, "s--", label=tr_b.label)
    ax.axhline(tr_a.kl_eq_bits, color="C0", ls=":", lw=1.2,
               label=f"{tr_a.label} $KL_{{eq}}$={tr_a.kl_eq_bits:.4f}")
    ax.axhline(tr_b.kl_eq_bits, color="C1", ls=":", lw=1.2,
               label=f"{tr_b.label} $KL_{{eq}}$={tr_b.kl_eq_bits:.4f}")
    ax.set_ylabel(r"$KL(\psi\|\varphi)$ [bits]", fontsize=9)
    ax.set_xlabel("generation $t$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)

    for ax, tr_a_s, tr_b_s, ylab in [
        (axes[1, 0], tr_a.f1, tr_b.f1,
         r"$F_1 = \langle W\rangle_\psi + H(\psi)/N$"),
        (axes[1, 1], tr_a.f2, tr_b.f2,
         r"$F_2 = \langle W^* \rangle_\psi - \frac{KL(\psi\|\varphi^*)}{N}$"),
        (axes[2, 0], tr_a.f3, tr_b.f3,
         r"$F_3 = \langle W^* \rangle_\psi - \frac{KL(\psi\|q_{\mathrm{cond}})}{N}$"),
        (axes[2, 1], tr_a.c_cost, tr_b.c_cost, r"$C(t)=\langle C(x)\rangle_\psi$"),
    ]:
        ax.plot(tr_a.times, tr_a_s, "o-", lw=1.5, ms=3, label=tr_a.label)
        ax.plot(tr_b.times, tr_b_s, "s--", lw=1.5, ms=3, label=tr_b.label)
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_xlabel("generation $t$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    ax = axes[3, 0]
    ax.plot(tr_a.times, tr_a.kl_phi_over_c, "o-", lw=1.5, ms=3, label=tr_a.label)
    ax.plot(tr_b.times, tr_b.kl_phi_over_c, "s--", lw=1.5, ms=3, label=tr_b.label)
    ax.set_ylabel(r"$KL(\psi\|\varphi)\,/\,C(t)$", fontsize=9)
    ax.set_xlabel("generation $t$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[3, 1]
    ax.axis("off")

    fig.suptitle(rf"Sigma comparison ($N={N}$, $u=v={u:g}$, $r={r:g}$)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p1 = out_dir / "compare_trajectories.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"  wrote {p1}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(tr_a.times, tr_a.kl_phi_over_c, "o-", lw=1.8, ms=4, label=tr_a.label)
    ax.plot(tr_b.times, tr_b.kl_phi_over_c, "s--", lw=1.8, ms=4, label=tr_b.label)
    ax.set_xlabel("generation $t$")
    ax.set_ylabel(r"$KL(\psi\|\varphi)\,/\,C(t)$")
    ax.set_title(rf"Information per unit cost ($N={N}$, $u=v={u:g}$)", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p_kl = out_dir / "compare_kl_over_c.png"
    fig.savefig(p_kl, dpi=150)
    plt.close(fig)
    print(f"  wrote {p_kl}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, ra, rb, ylab in [
        (axes[0], tr_a.rate_f1_over_c, tr_b.rate_f1_over_c, r"$(\Delta F_1/\Delta t)\,/\,C(t)$"),
        (axes[1], tr_a.rate_f2_over_c, tr_b.rate_f2_over_c, r"$(\Delta F_2/\Delta t)\,/\,C(t)$"),
    ]:
        ax.plot(tr_a.rate_times, ra, "o-", lw=1.5, ms=3, label=tr_a.label)
        ax.plot(tr_b.rate_times, rb, "s--", lw=1.5, ms=3, label=tr_b.label)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_xlabel("$t$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(r"$F_1$, $F_2$ rates normalized by $C(t)$", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p2 = out_dir / "compare_rates_over_c.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  wrote {p2}", flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for ax, ra, rb, ylab in [
        (axes[0], tr_a.rate_kl_phi_over_c, tr_b.rate_kl_phi_over_c,
         r"$(\Delta KL(\psi\|\varphi)/\Delta t)\,/\,C(t)$"),
        (axes[1], tr_a.rate_kl_star_over_c, tr_b.rate_kl_star_over_c,
         r"$(\Delta KL(\psi\|\psi^*)/\Delta t)\,/\,C(t)$"),
        (axes[2], tr_a.rate_f1_over_c, tr_b.rate_f1_over_c,
         r"$(\Delta F_1/\Delta t)\,/\,C(t)$"),
    ]:
        ax.plot(tr_a.rate_times, ra, "o-", lw=1.5, ms=3, label=tr_a.label)
        ax.plot(tr_b.rate_times, rb, "s--", lw=1.5, ms=3, label=tr_b.label)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_xlabel("$t$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("KL rates normalized by cost $C(t)$", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p3 = out_dir / "compare_kl_rates_over_c.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"  wrote {p3}", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, va, vb, ylab in [
        (axes[0, 0], tr_a.f1, tr_b.f1, r"$F_1$"),
        (axes[0, 1], tr_a.f2, tr_b.f2, r"$F_2$"),
        (axes[1, 0], tr_a.f3, tr_b.f3, r"$F_3$"),
        (axes[1, 1], tr_a.f4, tr_b.f4,
         r"$F_4 = -\frac{1}{N}D_{\mathrm{KL}}(\psi\|\psi^*)$ [nats]"),
    ]:
        ax.plot(tr_a.times, va, "o-", lw=1.8, ms=4, label=tr_a.label)
        ax.plot(tr_b.times, vb, "s--", lw=1.8, ms=4, label=tr_b.label)
        ax.set_ylabel(ylab, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    for ax in axes[1, :]:
        ax.set_xlabel("generation $t$")
    fig.suptitle(rf"$F_1$–$F_4$ ($N={N}$, $u=v={u:g}$)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p_f123 = out_dir / "compare_f1_f2_f3_f4.png"
    fig.savefig(p_f123, dpi=150)
    plt.close(fig)
    print(f"  wrote {p_f123}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    ylo_a, yhi_a = _plot_delta_d_and_c(
        axes[0], tr_a, N=N, color_d="C0", color_c="C3", set_ylim=False,
    )
    ylo_b, yhi_b = _plot_delta_d_and_c(
        axes[1], tr_b, N=N, color_d="C1", color_c="C4", set_ylim=False,
    )
    ylo, yhi = min(ylo_a, ylo_b), max(yhi_a, yhi_b)
    for ax in axes:
        ax.set_ylim(ylo, yhi)
    fig.suptitle(
        rf"Per-gen $\Delta D$ and $N\,C(t)$ "
        rf"($N={N}$, $u=v={u:g}$, $t=0\ldots{int(tr_a.pergen_times[-1])}$)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p_dc = out_dir / "compare_delta_d_and_c.png"
    fig.savefig(p_dc, dpi=150)
    plt.close(fig)
    print(f"  wrote {p_dc}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for traj, style, color in [(tr_a, "o-", "C0"), (tr_b, "s--", "C1")]:
        t = traj.pergen_times
        delta_d = np.diff(traj.pergen_kl_phi_bits)
        t_delta = t[1:]
        nc = N * traj.pergen_c_cost[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(nc > 1e-300, delta_d / nc, np.nan)
        ax.plot(t_delta, ratio, style, lw=1.5, ms=3, color=color, label=traj.label)
    ax.axhline(1.0, color="0.5", ls=":", lw=0.8, alpha=0.6)
    ax.set_ylabel(r"$\Delta D / (N\,C(t))$", fontsize=10)
    ax.set_xlabel("generation $t$")
    ax.set_title(
        rf"$\Delta D / (N\,C(t))$ per generation "
        rf"($N={N}$, $u=v={u:g}$, $t=1\ldots{int(tr_a.pergen_times[-1])}$)",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlim(tr_a.pergen_times[0], tr_a.pergen_times[-1])
    fig.tight_layout()
    p_ratio = out_dir / "compare_delta_d_over_nc.png"
    fig.savefig(p_ratio, dpi=150)
    plt.close(fig)
    print(f"  wrote {p_ratio}", flush=True)


def _load_yaml_config(path: Path) -> dict:
    cfg = load_config(path)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two Sigma landscapes on simplex dynamics")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    if args.config is not None:
        cfg = _load_yaml_config(args.config.resolve())
        N = int(cfg.get("N", N_MARKOV))
        r = float(cfg.get("r", R))
        u = float(cfg["u"][0]) if isinstance(cfg.get("u"), list) else float(cfg.get("u", U))
        v = float(cfg["v"][0]) if isinstance(cfg.get("v"), list) else float(cfg.get("v", V))
        times = [int(t) for t in cfg.get("times", TIMES)]
        pergen_max_t = int(cfg.get("pergen_max_t", PERGEN_MAX_T))
        runs = cfg["compare"]
        sigmas = [(r["label"], np.array(r["sigma"], dtype=float)) for r in runs]
    else:
        N, r, u, v, times = N_MARKOV, R, U, V, TIMES
        pergen_max_t = PERGEN_MAX_T
        sigmas = [(LABEL_A, SIGMA_A), (LABEL_B, SIGMA_B)]

    if len(sigmas) != 2:
        raise ValueError("Exactly two Sigma matrices required in compare list")

    trajs = [
        run_one_sigma(lab, sig, N=N, u=u, v=v, r=r, times=times, pergen_max_t=pergen_max_t)
        for lab, sig in sigmas
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "N": N, "r": r, "u": u, "v": v,
        "runs": [{"label": t.label, "sigma": t.sigma.tolist(), "kl_eq_bits": t.kl_eq_bits} for t in trajs],
    }
    (args.out_dir / "settings.json").write_text(json.dumps(meta, indent=2))
    for i, t in enumerate(trajs):
        _write_csv(args.out_dir / f"trajectory_{i}.csv", t)

    plot_comparison(trajs[0], trajs[1], args.out_dir, N=N, u=u, r=r)
    print(f"\nDone. Outputs in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
