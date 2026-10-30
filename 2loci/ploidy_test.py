#!/usr/bin/env python3
r"""
Is the discrete 2-locus chain haploid (n_c = N gene copies) or diploid (n_c = 2N)?

Two complementary tests, both against the SAME discrete Wright-Fisher chain
(Multinomial(N, p), p(g) ∝ x_g omega(g)):

TEST A (selection-free, decisive on copy number)
  At neutrality (Sigma = 0), the single-locus allele-frequency marginal of an
  n_c-copy WF with symmetric mutation u=v is  Beta(2 n_c u, 2 n_c u).
  We compare the chain's locus-1 allele-freq marginal to
     haploid : Beta(2Nu, 2Nu)
     diploid : Beta(4Nu, 4Nu)

TEST B (selection term factor)
  Compare the chain's selection equilibrium psi* (interior) to three diffusion
  densities  p(x) ∝ W(x)^P * prod_i x_i^{alpha_i - 1},  alpha_i = A*N*influx/denom:
     (a) current   P = N,    A = 2     (W^N,  2Nu)
     (b) fix-sel   P = 2N,   A = 2     (W^2N, 2Nu)   <- n_c=N haploid, consistent
     (c) diploid   P = 2N,   A = 4     (W^2N, 4Nu)   <- n_c=2N (the 'N->2N' idea)

Run:
  python3 ploidy_test.py --out results/ploidy_test
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

import simplex as sp
from selection_diffusion_vs_chain import (
    Setting, discrete_psi_star, _normalize_log, _tv_kl, genotype_marginals,
)

SCRIPT_DIR = Path(__file__).resolve().parent
GENO_LABELS = ["00", "01", "10", "11"]

N = 40
U = V = 0.1
R = 0.0

# positive-epistasis Sigma direction, scaled to a few N*s_eff targets for TEST B
DIR_S1, DIR_S2, DIR_EPS = 0.08, -0.04, 0.15
B_TARGETS = [0.5, 1.0, 2.0, 3.0]

# diffusion variants: (label, selection power factor on N, alpha coefficient)
VARIANTS = [
    ("current  W^N, 2Nu",  1.0, 2.0),
    ("fix-sel  W^2N, 2Nu", 2.0, 2.0),
    ("diploid  W^2N, 4Nu", 2.0, 4.0),
]


def diffusion_logdensity_general(st, fracs, omega, M_mut, sel_pow_factor, alpha_coef):
    """log p = (sel_pow_factor*N) log W + sum_i (alpha_i - 1) log x_i,
    alpha_i = alpha_coef * N * influx_i / (1 - x_i)  (the 'cond' surrogate)."""
    N_ = st.N
    X = np.asarray(fracs, float)
    interior = np.all(X > 0.0, axis=1)
    W = X @ omega
    log_W = np.log(np.clip(W, 1e-300, None))
    XM = X @ M_mut
    influx = XM - X * np.diag(M_mut)[None, :]
    denom = np.clip(1.0 - X, 1e-300, None)
    alpha = alpha_coef * N_ * influx / denom
    log_p = np.full(X.shape[0], -np.inf)
    Xi = X[interior]
    log_p[interior] = (sel_pow_factor * N_ * log_W[interior]
                       + np.sum((alpha[interior] - 1.0) * np.log(Xi), axis=1))
    return log_p, interior


def _scaled_setting(target):
    base = {"01": DIR_S2, "10": DIR_S1, "11": DIR_S1 + DIR_S2 + DIR_EPS}
    c = target / (N * max(abs(v) for v in base.values()))
    return Setting(f"Nseff={target:g}", N=N, s1=c * DIR_S1, s2=c * DIR_S2,
                   eps12=c * DIR_EPS, u=U, v=V, r=R)


def discretized_beta(a, b, N_):
    """P(round(x*N)=k) proxy: integrate Beta over each 1/N cell, k=0..N."""
    edges = (np.arange(N_ + 2) - 0.5) / N_
    edges = np.clip(edges, 0.0, 1.0)
    cdf = beta_dist.cdf(edges, a, b)
    pmf = np.diff(cdf)
    return pmf / pmf.sum()


def run(out_dir: Path):
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9)
    except ImportError:
        pass
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "figure.facecolor": "white"})
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []

    # ----------------------------- TEST A -----------------------------
    st0 = Setting("neutral", N=N, s1=0.0, s2=0.0, eps12=0.0, u=U, v=V, r=R)
    psi0, fracs0, counts0, omega0, _ = discrete_psi_star(st0)
    # locus-1 allele-1 count = genotypes 10 + 11  (columns 2,3)
    a1_count = counts0[:, 2] + counts0[:, 3]
    marg = np.bincount(a1_count, weights=psi0, minlength=N + 1)
    marg = marg / marg.sum()
    ks = np.arange(N + 1) / N

    hap = discretized_beta(2 * N * U, 2 * N * V, N)      # Beta(2Nu,2Nv)
    dip = discretized_beta(4 * N * U, 4 * N * V, N)      # Beta(4Nu,4Nv)
    tv_hap = 0.5 * np.sum(np.abs(marg - hap))
    tv_dip = 0.5 * np.sum(np.abs(marg - dip))
    # variance comparison (sharper summary)
    mean_chain = np.sum(ks * marg)
    var_chain = np.sum((ks - mean_chain) ** 2 * marg)
    var_hap = (2 * N * U) * (2 * N * V) / ((4 * N * U) ** 2 * (4 * N * U + 1))  # Beta var a=b
    var_dip = (4 * N * U) * (4 * N * V) / ((8 * N * U) ** 2 * (8 * N * U + 1))

    log += ["=== TEST A: neutral locus-1 allele-freq marginal ===",
            f"  N={N}, u=v={U}   2Nu={2*N*U:g} (haploid)   4Nu={4*N*U:g} (diploid)",
            f"  TV(chain, Beta(2Nu))={tv_hap:.4f}   TV(chain, Beta(4Nu))={tv_dip:.4f}",
            f"  chain var={var_chain:.5f}   Beta(2Nu) var={var_hap:.5f}   "
            f"Beta(4Nu) var={var_dip:.5f}",
            f"  --> chain is {'HAPLOID (n_c=N)' if tv_hap < tv_dip else 'DIPLOID (n_c=2N)'}", ""]
    print("\n".join(log[-6:]), flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, marg, "o", ms=4, color="k", label="chain (neutral)")
    ax.plot(ks, hap, "-", lw=2, color="#3b528b", label=rf"haploid Beta($2Nu$={2*N*U:g})")
    ax.plot(ks, dip, "-", lw=2, color="#b5367a", label=rf"diploid Beta($4Nu$={4*N*U:g})")
    ax.set_xlabel("locus-1 allele frequency"); ax.set_ylabel("marginal prob.")
    ax.set_title(f"TEST A (neutral, N={N}, u=v={U}): copy-number discriminator\n"
                 f"TV vs haploid={tv_hap:.3f}, vs diploid={tv_dip:.3f}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); p = out_dir / "testA_neutral_marginal.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {p}")

    # ----------------------------- TEST B -----------------------------
    log += ["=== TEST B: selection equilibrium, 3 diffusion variants ==="]
    table = {lab: [] for (lab, _, _) in VARIANTS}
    for target in B_TARGETS:
        st = _scaled_setting(target)
        psi, fracs, counts, omega, M_mut = discrete_psi_star(st)
        interior = np.all(fracs > 0.0, axis=1)
        psi_int = np.zeros_like(psi); psi_int[interior] = psi[interior]
        psi_int /= psi_int.sum()
        line = [f"  N*s_eff={target:g}: "]
        for (lab, spf, ac) in VARIANTS:
            log_p, _ = diffusion_logdensity_general(st, fracs, omega, M_mut, spf, ac)
            pdiff = _normalize_log(log_p, interior)
            tv, kl = _tv_kl(psi_int, pdiff)
            table[lab].append((target, tv, kl))
            line.append(f"[{lab}] TV={tv:.4f} KL={kl:.4f}  ")
        log.append("".join(line))
        print("".join(line), flush=True)

    fig2, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"current  W^N, 2Nu": "#3b528b", "fix-sel  W^2N, 2Nu": "#21908d",
              "diploid  W^2N, 4Nu": "#b5367a"}
    for lab in table:
        arr = np.array(table[lab])
        axL.plot(arr[:, 0], arr[:, 1], "o-", color=colors[lab], label=lab)
        axR.plot(arr[:, 0], arr[:, 2], "s-", color=colors[lab], label=lab)
    for ax, ylab in ((axL, "interior TV"), (axR, "interior KL (bits)")):
        ax.set_xlabel(r"$N\,s_{eff}$(11)"); ax.set_ylabel(ylab)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    axL.set_title(f"TEST B: chain vs diffusion variants (N={N}, u=v={U})")
    fig2.tight_layout(); p2 = out_dir / "testB_variants.png"
    fig2.savefig(p2, dpi=170, bbox_inches="tight"); plt.close(fig2)
    print(f"Wrote {p2}")

    (out_dir / "ploidy_test.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'ploidy_test.log'}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out = args.out or (SCRIPT_DIR / "results" /
                       f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_ploidy_test")
    run(out)
    print(f"\nAll outputs in: {out}")


if __name__ == "__main__":
    main()
