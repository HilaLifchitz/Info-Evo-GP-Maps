#!/usr/bin/env python3
r"""
SANITY CHECK: discrete SELECTION equilibrium  vs  diffusion selection equilibrium
on the 2-locus genotype simplex (4 genotypes 00, 01, 10, 11).

We pick a selection matrix Sigma (s1, s2, eps12), N=30, mutation u/v, and r=0,
then compare two equilibria over the SAME simplex lattice:

  (1) MARKOV CHAIN psi*  — stationary distribution of the discrete Wright-Fisher
      chain with selection omega(g)=exp(g^T Sigma g), mutation M, drift, r=0.

  (2) DIFFUSION  p(x) = exp(-V(x)) / Z   (Boltzmann form), with
        V(x) = -N log W(x) + sum_i (1 - alpha_i) log x_i
      where
        x = (x00, x01, x10, x11) is a point on the genotype simplex,
        W(x) = sum_i x_i * omega(g_i)            (mean fitness at x),
        alpha_i = 2N * (mutation rate INTO genotype i).
      => p(x) ∝ W(x)^{N} * prod_i x_i^{alpha_i - 1}.

The 4-allele Wright form assumes parent-independent (reversible) mutation, which
our M is NOT. We therefore approximate alpha_i with a LOCAL, x-dependent rate.
Two choices are computed (see ALPHA_MODE):

  "cond"   alpha_i(x) = 2N * [sum_{j!=i} x_j M_ji] / [sum_{j!=i} x_j]
           (per-capita rate at which a non-i copy mutates into i; this reduces
            EXACTLY to the 2-allele Wright exponent 2Nu for a single locus.)
  "influx" alpha_i(x) = 2N *  sum_{j!=i} x_j M_ji
           (raw mutational inflow into i, not per-capita.)

Because the diffusion density has integrable singularities on the simplex
boundary (x_i -> 0 with alpha_i < 1), the discrete comparison is made on the
INTERIOR support (all four genotypes present); psi*'s boundary mass is reported
separately.

Run:
  python3 selection_diffusion_vs_chain.py
  python3 selection_diffusion_vs_chain.py --out results/sel_diff_check
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

import simplex as sp

SCRIPT_DIR = Path(__file__).resolve().parent
GENO_LABELS = ["00", "01", "10", "11"]


# =============================================================================
# SETTINGS — selection, population, mutation. Edit here.
# =============================================================================
@dataclass
class Setting:
    label: str
    N: int
    s1: float
    s2: float
    eps12: float
    u: float
    v: float
    r: float = 0.0


SETTINGS: list[Setting] = [
    # moderate selection, moderate-strong mutation (keeps mass polymorphic)
    Setting("moderate sel + mut", N=40, s1=0.08, s2=-0.04, eps12=0.15, u=0.08, v=0.08),
    # stronger selection, strong mutation
    Setting("stronger sel",       N=40, s1=0.15, s2=-0.10, eps12=0.30, u=0.10, v=0.10),
]

ALPHA_MODE_DEFAULT = "cond"   # which alpha definition the PLOTS use ("cond" or "influx")


def _stationary_fast(P_col: np.ndarray) -> np.ndarray:
    S = P_col.shape[0]
    try:
        from scipy.sparse.linalg import eigs, LinearOperator
        op = LinearOperator((S, S), matvec=lambda x: P_col @ x, dtype=float)
        _, vecs = eigs(op, k=1, which="LM", maxiter=8000, tol=1e-11)
        pi = np.real(vecs[:, 0])
        if pi.sum() < 0:
            pi = -pi
        pi = np.clip(pi, 0.0, None)
        s = float(pi.sum())
        if s > 0 and np.isfinite(s):
            pi /= s
            if float(np.linalg.norm(P_col @ pi - pi, ord=1)) < 1e-6:
                return pi
    except Exception:
        pass
    return sp.stationary_from_matrix(P_col, tol=1e-12, max_iter=40000)


def discrete_psi_star(st: Setting):
    """Discrete selection equilibrium psi* and the simplex lattice."""
    Sigma = sp.build_sigma(st.s1, st.s2, st.eps12)
    omega = sp.fitness_exp(Sigma)                       # length-4: [00,01,10,11]
    M_mut = sp.mutation_matrix_asymmetric(np.array([st.u, st.u]), np.array([st.v, st.v]))
    fracs = sp.make_simplex_grid(st.N)
    counts = sp.make_simplex_counts(st.N)
    T_row, _, _ = sp.build_transition_matrix_row_stochastic(
        st.N, omega, M_mut, st.r, n_jobs=-1, label=st.label, verbose=False,
    )
    P_col = sp.column_stochastic(T_row)
    psi = _stationary_fast(P_col)
    psi = np.clip(psi, 0.0, None)
    psi /= psi.sum()
    return psi, fracs, counts, omega, M_mut


def diffusion_logdensity(st: Setting, fracs: np.ndarray, omega: np.ndarray,
                         M_mut: np.ndarray, alpha_mode: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Unnormalized log p(x) = N log W(x) + sum_i (alpha_i(x) - 1) log x_i,
    evaluated on the lattice. Returns (log_p, interior_mask). Non-interior
    points (some x_i == 0) get log_p = -inf.
    """
    N = st.N
    X = np.asarray(fracs, dtype=float)                  # (S,4)
    interior = np.all(X > 0.0, axis=1)

    W = X @ omega                                       # (S,) mean fitness
    log_W = np.log(np.clip(W, 1e-300, None))

    XM = X @ M_mut                                      # post-mutation freq
    diagM = np.diag(M_mut)
    influx = XM - X * diagM[None, :]                    # sum_{j!=i} x_j M_ji
    if alpha_mode == "cond":
        denom = np.clip(1.0 - X, 1e-300, None)          # sum_{j!=i} x_j = 1 - x_i
        alpha = 2.0 * N * influx / denom
    elif alpha_mode == "influx":
        alpha = 2.0 * N * influx
    else:
        raise ValueError(f"unknown alpha_mode {alpha_mode!r}")

    log_p = np.full(X.shape[0], -np.inf, dtype=float)
    Xi = X[interior]
    log_p[interior] = (N * log_W[interior]
                       + np.sum((alpha[interior] - 1.0) * np.log(Xi), axis=1))
    return log_p, interior


def _normalize_log(log_p: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax over masked entries -> pmf supported on mask (others 0)."""
    p = np.zeros_like(log_p)
    lm = log_p[mask]
    m = float(np.max(lm))
    w = np.exp(lm - m)
    p[mask] = w / w.sum()
    return p


def _tv_kl(p: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    tv = 0.5 * float(np.sum(np.abs(p - q)))
    pe = p + 1e-15
    qe = q + 1e-15
    kl = float(np.sum(pe * (np.log(pe) - np.log(qe))) / np.log(2.0))
    return tv, kl


def genotype_marginals(p: np.ndarray, counts: np.ndarray, N: int) -> list[np.ndarray]:
    """Per-genotype count distributions P(n_g = k), k=0..N, for g=0..3."""
    return [np.bincount(counts[:, g], weights=p, minlength=N + 1) for g in range(4)]


def run(out_dir: Path, show: bool = False) -> None:
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9)
    except ImportError:
        pass
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "figure.facecolor": "white"})
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    for st in SETTINGS:
        print(f"\n### {st.label}:  N={st.N}, s1={st.s1}, s2={st.s2}, eps12={st.eps12}, "
              f"u=v={st.u}, r={st.r}", flush=True)
        psi, fracs, counts, omega, M_mut = discrete_psi_star(st)
        interior_full = np.all(fracs > 0.0, axis=1)
        psi_boundary_mass = float(psi[~interior_full].sum())

        block = [
            f"=== {st.label} ===",
            f"  N={st.N}, s1={st.s1}, s2={st.s2}, eps12={st.eps12}, u=v={st.u}, r={st.r}",
            f"  omega (fitness 00,01,10,11) = {np.round(omega,4).tolist()}",
            f"  psi* boundary mass (>=1 genotype absent) = {psi_boundary_mass:.4f}",
            f"  psi* interior mass                       = {1.0 - psi_boundary_mass:.4f}",
        ]

        results_for_plot = {}
        for amode in ("cond", "influx"):
            log_p, interior = diffusion_logdensity(st, fracs, omega, M_mut, amode)
            # restrict BOTH distributions to interior, renormalize
            psi_int = np.zeros_like(psi)
            psi_int[interior] = psi[interior]
            si = psi_int.sum()
            psi_int = psi_int / si if si > 0 else psi_int
            pdiff_int = _normalize_log(log_p, interior)
            tv, kl = _tv_kl(psi_int, pdiff_int)
            block.append(f"  [alpha={amode:6s}] interior TV(psi*,diff)={tv:.3e}  "
                         f"KL(psi*||diff)={kl:.3e} bits")
            results_for_plot[amode] = (psi_int, pdiff_int, interior)
        log_lines += block + [""]
        print("\n".join(block), flush=True)

        # ---- plots use ALPHA_MODE_DEFAULT ----
        psi_int, pdiff_int, interior = results_for_plot[ALPHA_MODE_DEFAULT]
        N = st.N
        tag = st.label.replace(" ", "_").replace("+", "")

        # (A) scatter: diffusion vs chain over interior lattice points (log-log)
        fig1, ax1 = plt.subplots(figsize=(6.4, 6.0))
        xs = pdiff_int[interior]
        ys = psi_int[interior]
        ax1.scatter(xs, ys, s=14, alpha=0.5, edgecolor="none", color="#3b528b")
        lo = min(xs[xs > 0].min(), ys[ys > 0].min())
        hi = max(xs.max(), ys.max())
        ax1.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.set_xlabel(r"diffusion $p(x)=e^{-V(x)}$ (interior, renorm.)")
        ax1.set_ylabel(r"chain $\psi^*(x)$ (interior, renorm.)")
        ax1.set_title(f"{st.label}: per-state mass, chain vs diffusion\n"
                      rf"$\alpha$={ALPHA_MODE_DEFAULT}, $N$={N}, $s_1$={st.s1}, $s_2$={st.s2}, "
                      rf"$\epsilon_{{12}}$={st.eps12}, $u$={st.u}", fontsize=9)
        ax1.legend(); ax1.grid(True, which="both", alpha=0.3)
        p1 = out_dir / f"scatter_{tag}.png"
        fig1.tight_layout(); fig1.savefig(p1, dpi=170, bbox_inches="tight"); plt.close(fig1)
        print(f"Wrote {p1}")

        # (B) per-genotype marginals (interior-restricted), chain vs diffusion
        mp = genotype_marginals(psi_int, counts, N)
        md = genotype_marginals(pdiff_int, counts, N)
        ks = np.arange(N + 1) / N
        fig2, axes2 = plt.subplots(2, 2, figsize=(11, 8))
        for g in range(4):
            ax = axes2[g // 2][g % 2]
            ax.plot(ks, mp[g], "o", ms=3.5, color="C0", label=r"chain $\psi^*$")
            ax.plot(ks, md[g], "-", lw=1.6, color="C3", alpha=0.85, label="diffusion")
            ax.set_title(rf"genotype {GENO_LABELS[g]}  ($\omega$={omega[g]:.3f})", fontsize=9)
            ax.set_xlabel("genotype frequency"); ax.set_ylabel("marginal prob.")
            ax.grid(True, alpha=0.3)
            if g == 0:
                ax.legend(fontsize=8)
        fig2.suptitle(f"{st.label}: per-genotype marginals (interior), "
                      rf"$\alpha$={ALPHA_MODE_DEFAULT}", y=0.99, fontsize=12)
        fig2.tight_layout(rect=(0, 0, 1, 0.97))
        p2 = out_dir / f"marginals_{tag}.png"
        fig2.savefig(p2, dpi=170, bbox_inches="tight"); plt.close(fig2)
        print(f"Wrote {p2}")

        if show:
            plt.show()

    (out_dir / "sel_diff_check.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_dir / 'sel_diff_check.log'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Discrete vs diffusion SELECTION equilibrium on the genotype simplex.")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or (SCRIPT_DIR / "results" /
                       f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_sel_diff_check")
    run(out, show=args.show)
    print(f"\nAll outputs in: {out}")


if __name__ == "__main__":
    main()
