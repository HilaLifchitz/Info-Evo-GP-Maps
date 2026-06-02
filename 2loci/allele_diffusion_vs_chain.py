#!/usr/bin/env python3
r"""
Compare the Markov-chain selection equilibrium's ALLELE-FREQUENCY marginals to
the haploid allele-frequency DIFFUSION equilibrium (LE + HWE assumption):

    rho(x1, x2) ∝ W(x1, x2)^N * prod_{i=1,2} x_i^{2N u_i - 1} (1 - x_i)^{2N v_i - 1}

where x_i = derived-allele frequency at locus i, and W is the mean fitness
computed over genotypes assuming LINKAGE EQUILIBRIUM (loci independent) and HWE:
    f00=(1-x1)(1-x2), f01=(1-x1)x2, f10=x1(1-x2), f11=x1 x2,
    W(x1,x2) = sum_g f_g * omega(g),   omega(g)=exp(g^T Sigma g).

This formula uses the CLEAN per-locus Wright mutation exponents (2N u_i - 1),
so it avoids the ambiguous "alpha = mutation rate into genotype i" surrogate of
the genotype-simplex formula. Because it assumes LE, it should match chains with
HIGHER recombination r better.

For each (mutation, r) at fixed N we report, on the INTERIOR allele lattice
(both loci polymorphic), the interior TV / KL between:
  * the chain psi* projected to the (n_A, n_B) allele-count joint, and
  * this allele diffusion.
We also report, for the same settings, the genotype-simplex/alpha fit
(W^N, alpha="cond") for direct comparison.

Run:
  python3 allele_diffusion_vs_chain.py
  python3 allele_diffusion_vs_chain.py --out results/allele_diff
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

import simplex as sp
from selection_diffusion_vs_chain import (
    Setting,
    diffusion_logdensity,       # genotype-simplex (alpha) version, for comparison
    discrete_psi_star,
    _normalize_log,
    _tv_kl,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# fixed selection; sweep mutation (incl. very small) and r at this N
BASE = Setting("sel", N=45, s1=0.08, s2=-0.04, eps12=0.15, u=0.05, v=0.05)
N_POP = 45
U_VALUES = [0.05, 0.01, 0.001]          # includes very small (non-zero) mutation
R_VALUES = [0.0, 0.25, 0.5, 1.0]


def chain_allele_joint(psi: np.ndarray, counts: np.ndarray, N: int) -> np.ndarray:
    """Project genotype-simplex psi to the (n_A, n_B) allele-count joint, (N+1,N+1)."""
    nA = counts[:, 2] + counts[:, 3]    # locus-1 derived count
    nB = counts[:, 1] + counts[:, 3]    # locus-2 derived count
    H = np.zeros((N + 1, N + 1), dtype=float)
    np.add.at(H, (nA, nB), psi)
    return H


def allele_diffusion(N: int, omega: np.ndarray, u: float, v: float):
    """
    log rho(x1,x2) on the (N+1)x(N+1) allele-frequency lattice (x=a/N).
    Returns (log_rho 2D, interior_mask 2D). W via LE+HWE.
    """
    a = np.arange(N + 1, dtype=float) / N
    X1, X2 = np.meshgrid(a, a, indexing="ij")
    f00 = (1 - X1) * (1 - X2)
    f01 = (1 - X1) * X2
    f10 = X1 * (1 - X2)
    f11 = X1 * X2
    W = f00 * omega[0] + f01 * omega[1] + f10 * omega[2] + f11 * omega[3]

    a_exp = 2.0 * N * u - 1.0      # x_i exponent
    b_exp = 2.0 * N * v - 1.0      # (1-x_i) exponent
    interior = (X1 > 0) & (X1 < 1) & (X2 > 0) & (X2 < 1)
    log_rho = np.full((N + 1, N + 1), -np.inf, dtype=float)
    Wi = np.clip(W, 1e-300, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = (N * np.log(Wi)
               + a_exp * np.log(np.clip(X1, 1e-300, None))
               + b_exp * np.log(np.clip(1 - X1, 1e-300, None))
               + a_exp * np.log(np.clip(X2, 1e-300, None))
               + b_exp * np.log(np.clip(1 - X2, 1e-300, None)))
    log_rho[interior] = val[interior]
    return log_rho, interior


def _norm_interior_2d(arr2d: np.ndarray, interior: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr2d)
    m = arr2d[interior].sum()
    if m > 0:
        out[interior] = arr2d[interior] / m
    return out


def _norm_logdensity_2d(log_rho: np.ndarray, interior: np.ndarray) -> np.ndarray:
    out = np.zeros_like(log_rho)
    lm = log_rho[interior]
    mx = float(np.max(lm))
    w = np.exp(lm - mx)
    out[interior] = w / w.sum()
    return out


def genotype_fit(st: Setting, psi, fracs, counts, omega, M_mut) -> tuple[float, float]:
    """Interior TV/KL for the genotype-simplex (alpha='cond') diffusion."""
    log_p, interior = diffusion_logdensity(st, fracs, omega, M_mut, "cond")
    psi_int = np.zeros_like(psi)
    psi_int[interior] = psi[interior]
    s = psi_int.sum()
    if s > 0:
        psi_int /= s
    pdiff = _normalize_log(log_p, interior)
    return _tv_kl(psi_int, pdiff)


def run(out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9)
    except ImportError:
        pass
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "figure.facecolor": "white"})
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"allele-frequency diffusion vs chain   N={N_POP}")
    print(f"  selection s1={BASE.s1}, s2={BASE.s2}, eps12={BASE.eps12}")
    print(f"  u=v in {U_VALUES},  r in {R_VALUES}")
    print("=" * 72, flush=True)

    data: dict[tuple[float, float], dict] = {}
    for u in U_VALUES:
        for r in R_VALUES:
            t0 = time.perf_counter()
            st = replace(BASE, N=N_POP, u=u, v=u, r=r)
            psi, fracs, counts, omega, M_mut = discrete_psi_star(st)

            # allele-frequency comparison
            H = chain_allele_joint(psi, counts, N_POP)
            log_rho, interior2d = allele_diffusion(N_POP, omega, u, u)
            H_int = _norm_interior_2d(H, interior2d)
            rho_int = _norm_logdensity_2d(log_rho, interior2d)
            tv_a, kl_a = _tv_kl(H_int.ravel(), rho_int.ravel())
            boundary_allele = 1.0 - float(H[interior2d].sum())

            # genotype-simplex (alpha) comparison, same settings
            tv_g, kl_g = genotype_fit(st, psi, fracs, counts, omega, M_mut)

            data[(u, r)] = {
                "tv_a": tv_a, "kl_a": kl_a, "tv_g": tv_g, "kl_g": kl_g,
                "boundary_allele": boundary_allele,
                "H_int": H_int, "rho_int": rho_int,
            }
            print(f"  u={u:<6g} r={r:<4g} | ALLELE TV={tv_a:.4f} KL={kl_a:.4f} | "
                  f"GENO TV={tv_g:.4f} KL={kl_g:.4f} | bdry={boundary_allele:.3f} "
                  f"({time.perf_counter()-t0:.1f}s)", flush=True)

    # ---- summary: TV vs r, allele (solid) vs genotype (dashed), per mutation ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for j, u in enumerate(U_VALUES):
        c = colors[j % len(colors)]
        tv_a = [data[(u, r)]["tv_a"] for r in R_VALUES]
        tv_g = [data[(u, r)]["tv_g"] for r in R_VALUES]
        kl_a = [data[(u, r)]["kl_a"] for r in R_VALUES]
        kl_g = [data[(u, r)]["kl_g"] for r in R_VALUES]
        axes[0].plot(R_VALUES, tv_a, "o-", color=c, lw=1.9, ms=6, label=f"allele, u={u:g}")
        axes[0].plot(R_VALUES, tv_g, "s--", color=c, lw=1.4, ms=5, alpha=0.7, label=f"genotype, u={u:g}")
        axes[1].plot(R_VALUES, kl_a, "o-", color=c, lw=1.9, ms=6, label=f"allele, u={u:g}")
        axes[1].plot(R_VALUES, kl_g, "s--", color=c, lw=1.4, ms=5, alpha=0.7, label=f"genotype, u={u:g}")
    axes[0].set_xlabel(r"recombination rate $r$"); axes[0].set_ylabel("interior TV")
    axes[1].set_xlabel(r"recombination rate $r$"); axes[1].set_ylabel("interior KL [bits]")
    axes[0].set_title("Total-variation distance"); axes[1].set_title("KL divergence")
    for ax in axes:
        ax.legend(fontsize=7.5, ncol=1); ax.grid(True, alpha=0.3)
    fig.suptitle(rf"Allele-freq diffusion (solid) vs genotype/$\alpha$ (dashed),  $N$={N_POP}, "
                 rf"$s_1$={BASE.s1}, $s_2$={BASE.s2}, $\epsilon_{{12}}$={BASE.eps12}", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p_sum = out_dir / "tv_kl_vs_r.png"
    fig.savefig(p_sum, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {p_sum}")

    # ---- per-locus marginals for moderate mutation across r (LE hypothesis) ----
    u_show = U_VALUES[0]
    r_show = [r for r in (0.0, 0.5, 1.0) if r in R_VALUES]
    fig2, axes2 = plt.subplots(2, len(r_show), figsize=(4.6 * len(r_show), 7.6), squeeze=False)
    xs = np.arange(N_POP + 1) / N_POP
    for ci, r in enumerate(r_show):
        d = data[(u_show, r)]
        H_int = d["H_int"]; rho_int = d["rho_int"]
        for li in range(2):
            ax = axes2[li][ci]
            chain_marg = H_int.sum(axis=1 - li)
            diff_marg = rho_int.sum(axis=1 - li)
            ax.plot(xs, chain_marg, "o", ms=3.5, color="C0", label=r"chain $\psi^*$")
            ax.plot(xs, diff_marg, "-", lw=1.7, color="C3", alpha=0.85, label="allele diffusion")
            ax.set_title(f"locus {li+1},  r={r:g}  (TV$_{{2D}}$={d['tv_a']:.3f})", fontsize=9)
            ax.set_xlabel("derived-allele frequency"); ax.set_ylabel("interior marginal")
            ax.grid(True, alpha=0.3)
            if li == 0 and ci == 0:
                ax.legend(fontsize=8)
    fig2.suptitle(rf"Allele-frequency marginals: chain vs LE diffusion  ($N$={N_POP}, $u$={u_show:g})",
                  y=1.0, fontsize=12)
    fig2.tight_layout(rect=(0, 0, 1, 0.97))
    p_marg = out_dir / "allele_marginals_vs_r.png"
    fig2.savefig(p_marg, dpi=170, bbox_inches="tight"); plt.close(fig2)
    print(f"Wrote {p_marg}")

    # CSV
    csv = out_dir / "allele_diff.csv"
    with csv.open("w", encoding="utf-8") as f:
        f.write("u,r,tv_allele,kl_allele,tv_genotype,kl_genotype,boundary_allele\n")
        for u in U_VALUES:
            for r in R_VALUES:
                d = data[(u, r)]
                f.write(f"{u},{r},{d['tv_a']:.6g},{d['kl_a']:.6g},"
                        f"{d['tv_g']:.6g},{d['kl_g']:.6g},{d['boundary_allele']:.6g}\n")
    print(f"Wrote {csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or (SCRIPT_DIR / "results" /
                       f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_allele_diff")
    run(out)
    print(f"\nAll outputs in: {out}")


if __name__ == "__main__":
    main()
