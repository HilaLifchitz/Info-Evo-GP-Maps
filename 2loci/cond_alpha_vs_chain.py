#!/usr/bin/env python3
r"""
Compare discrete chain equilibria to the cond-alpha diffusion surrogate

    log p(x) = N log W(x) + sum_i (alpha_i(x) - 1) log x_i

with alpha_i from "cond" or "influx" (see COND_ALPHA.md).

For neutral: W=1, compare phi* from the chain to the surrogate.
For selection: compare psi* to the surrogate with W(x)=x·omega.

Also checks:
  - per-locus marginals of phi* vs Beta(2Nu, 2Nv)
  - x·omega = sum_g x_g omega(g) indexing (00,01,10,11)

Run:
  cd 2loci
  python cond_alpha_vs_chain.py
  python cond_alpha_vs_chain.py --config compare_sigma_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import simplex as sp  # noqa: E402
from selection_diffusion_vs_chain import diffusion_logdensity, _normalize_log, _tv_kl  # noqa: E402
from selection_diffusion_vs_chain import Setting  # noqa: E402
from explore_neutral_phi import diffusion_marginal, _tv_kl as marginal_tv_kl  # noqa: E402


def verify_x_dot_omega(fracs: np.ndarray, omega: np.ndarray) -> float:
    """Max |x@omega - explicit sum_g x_g omega_g| over lattice."""
    explicit = fracs @ omega
    manual = sum(fracs[:, g] * omega[g] for g in range(4))
    return float(np.max(np.abs(explicit - manual)))


def chain_neutral_phi(N: int, u: np.ndarray, v: np.ndarray, r: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    M = sp.mutation_matrix_asymmetric(u, v)
    fracs = sp.make_simplex_grid(N)
    counts = sp.make_simplex_counts(N)
    omega0 = sp.fitness_exp(np.zeros((2, 2)))
    T, _, _ = sp.build_transition_matrix_row_stochastic(
        N, omega0, M, r, n_jobs=-1, label="neutral", verbose=False
    )
    P = sp.column_stochastic(T)
    phi = sp.stationary_fast(P)
    phi = np.clip(phi, 0.0, None)
    phi /= phi.sum()
    return phi, fracs, counts


def chain_psi_star(N: int, sigma: np.ndarray, u: np.ndarray, v: np.ndarray, r: float):
    M = sp.mutation_matrix_asymmetric(u, v)
    fracs = sp.make_simplex_grid(N)
    counts = sp.make_simplex_counts(N)
    omega = sp.fitness_exp(sigma)
    T, _, _ = sp.build_transition_matrix_row_stochastic(
        N, omega, M, r, n_jobs=-1, label="sel", verbose=False
    )
    P = sp.column_stochastic(T)
    psi, resid, ok = sp.try_selection_equilibrium(P)
    if not ok or psi is None:
        raise RuntimeError(f"psi* not found, residual={resid}")
    return psi, fracs, counts, omega, M, float(resid)


def diffusion_pmf(st: Setting, fracs: np.ndarray, omega: np.ndarray, M: np.ndarray, alpha_mode: str) -> np.ndarray:
    log_p, interior = diffusion_logdensity(st, fracs, omega, M, alpha_mode)
    return _normalize_log(log_p, interior)


def compare_distribution(label: str, chain_p: np.ndarray, diff_p: np.ndarray, fracs: np.ndarray) -> dict:
    interior = np.all(fracs > 0, axis=1)
    tv_full, kl_full = _tv_kl(chain_p, diff_p)
    c_int = np.zeros_like(chain_p)
    c_int[interior] = chain_p[interior]
    c_int /= c_int.sum()
    d_int = np.zeros_like(diff_p)
    d_int[interior] = diff_p[interior]
    d_int /= d_int.sum()
    tv_int, kl_int = _tv_kl(c_int, d_int)
    return {
        "label": label,
        "tv_full": tv_full,
        "kl_full_bits": kl_full,
        "tv_interior": tv_int,
        "kl_interior_bits": kl_int,
        "chain_boundary_mass": float(chain_p[~interior].sum()),
    }


def marginal_beta_check(phi: np.ndarray, counts: np.ndarray, N: int, u: np.ndarray, v: np.ndarray) -> dict:
    nA = counts[:, 2] + counts[:, 3]
    nB = counts[:, 1] + counts[:, 3]
    margA = np.bincount(nA, weights=phi, minlength=N + 1)
    margB = np.bincount(nB, weights=phi, minlength=N + 1)
    diffA = diffusion_marginal(N, 2.0 * N * u[0], 2.0 * N * v[0])
    diffB = diffusion_marginal(N, 2.0 * N * u[1], 2.0 * N * v[1])
    tvA, klA = marginal_tv_kl(margA, diffA)
    tvB, klB = marginal_tv_kl(margB, diffB)
    return {"tv_locus1": tvA, "kl_locus1_bits": klA, "tv_locus2": tvB, "kl_locus2_bits": klB}


def run(config_path: Path | None, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if config_path is not None:
        cfg = yaml.safe_load(config_path.read_text())
        N = int(cfg["N"])
        r = float(cfg["r"])
        u = np.array(cfg["u"], dtype=float)
        v = np.array(cfg["v"], dtype=float)
        runs = cfg.get("compare", [])
    else:
        N, r = 30, 0.0
        u = v = np.array([0.01, 0.01])
        runs = []

    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "cond-alpha vs chain equilibrium comparison",
        f"N={N}, u={u.tolist()}, v={v.tolist()}, r={r}",
        "",
    ]

    phi, fracs, counts = chain_neutral_phi(N, u, v, r)
    omega0 = sp.fitness_exp(np.zeros((2, 2)))
    M = sp.mutation_matrix_asymmetric(u, v)
    st_neutral = Setting("neutral", N=N, s1=0.0, s2=0.0, eps12=0.0, u=float(u[0]), v=float(v[0]), r=r)

    xdot_err = verify_x_dot_omega(fracs, omega0)
    lines.append(f"x·omega indexing check (max error): {xdot_err:.3e}")
    lines.append("  (x@omega = x_00*omega_00 + ... + x_11*omega_11, order 00,01,10,11)")
    lines.append("")

    mb = marginal_beta_check(phi, counts, N, u, v)
    lines.append("Neutral phi* marginals vs Beta(2Nu, 2Nv) per locus:")
    lines.append(f"  locus1: TV={mb['tv_locus1']:.4f}, KL={mb['kl_locus1_bits']:.4f} bits")
    lines.append(f"  locus2: TV={mb['tv_locus2']:.4f}, KL={mb['kl_locus2_bits']:.4f} bits")
    lines.append("")

    lines.append("Neutral phi* vs cond-alpha diffusion (W=1):")
    for mode in ("cond", "influx"):
        diff_p = diffusion_pmf(st_neutral, fracs, omega0, M, mode)
        stats = compare_distribution(mode, phi, diff_p, fracs)
        lines.append(
            f"  [{mode:6s}] TV_full={stats['tv_full']:.4f}, KL_full={stats['kl_full_bits']:.4f} bits | "
            f"TV_int={stats['tv_interior']:.4f}, KL_int={stats['kl_interior_bits']:.4f} bits | "
            f"chain boundary mass={stats['chain_boundary_mass']:.4f}"
        )
    lines.append("")

    for run in runs:
        label = run["label"]
        sigma = np.asarray(run["sigma"], dtype=float)
        psi, fracs, counts, omega, M, resid = chain_psi_star(N, sigma, u, v, r)
        s1, s2 = float(sigma[0, 0]), float(sigma[1, 1])
        eps12 = float(sigma[0, 1] + sigma[1, 0])
        st = Setting(label, N=N, s1=s1, s2=s2, eps12=eps12, u=float(u[0]), v=float(v[0]), r=r)
        lines.append(f"Selection psi* ({label}), ||P psi*-psi*||_1={resid:.3e}:")
        for mode in ("cond", "influx"):
            diff_p = diffusion_pmf(st, fracs, omega, M, mode)
            stats = compare_distribution(mode, psi, diff_p, fracs)
            lines.append(
                f"  [{mode:6s}] TV_full={stats['tv_full']:.4f}, KL_full={stats['kl_full_bits']:.4f} bits | "
                f"TV_int={stats['tv_interior']:.4f}, KL_int={stats['kl_interior_bits']:.4f} bits"
            )
        lines.append("")

    report = out_dir / "cond_alpha_vs_chain.txt"
    report.write_text("\n".join(lines) + "\n")
    print(report.read_text())

    # Plot: neutral phi vs cond diffusion on interior scatter (top 500 points by phi mass)
    diff_cond = diffusion_pmf(st_neutral, fracs, omega0, M, "cond")
    idx = np.argsort(phi)[::-1][: min(800, phi.size)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(phi[idx], diff_cond[idx], s=8, alpha=0.5)
    mx = max(phi[idx].max(), diff_cond[idx].max()) * 1.05
    axes[0].plot([0, mx], [0, mx], "k--", lw=1)
    axes[0].set_xlabel(r"chain $\varphi^*(x)$")
    axes[0].set_ylabel(r"cond-$\alpha$ diffusion $p(x)$")
    axes[0].set_title("Neutral: mass on lattice points")
    axes[0].grid(True, alpha=0.3)

    nA = counts[:, 2] + counts[:, 3]
    margA = np.bincount(nA, weights=phi, minlength=N + 1)
    diffA = diffusion_marginal(N, 2.0 * N * u[0], 2.0 * N * v[0])
    xs = np.arange(N + 1) / N
    axes[1].plot(xs, margA, "o-", label=r"chain $\varphi^*$, locus 1")
    axes[1].plot(xs, diffA, "s--", label=rf"Beta$(2Nu,2Nv)$, $2Nu={2*N*u[0]:g}$")
    axes[1].set_xlabel("allele frequency")
    axes[1].set_ylabel("mass")
    axes[1].set_title("Per-locus marginal (neutral)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(rf"$\varphi^*$ vs cond-$\alpha$ surrogate ($N={N}$)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "cond_alpha_vs_chain_neutral.png", dpi=150)
    plt.close(fig)
    print(f"wrote {report}")
    print(f"wrote {out_dir / 'cond_alpha_vs_chain_neutral.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare chain equilibria to cond-alpha diffusion")
    p.add_argument("--config", type=Path, default=SCRIPT_DIR / "compare_sigma_config.yaml")
    p.add_argument("--out", type=Path, default=SCRIPT_DIR / "plots" / "cond_alpha_check")
    args = p.parse_args()
    run(args.config if args.config.is_file() else None, args.out)


if __name__ == "__main__":
    main()
