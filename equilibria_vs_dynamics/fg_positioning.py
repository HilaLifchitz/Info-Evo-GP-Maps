"""Add F and G to the (corrected) Panel C positioning plot.

Panel C needs only equilibrium scalars -- D_eq, F*, tau_spec, mlf -- which are cheap
(stationary distribution + one spectral-gap eigenvalue; no trajectory).  We compute
them for F,G at the same 3 a-values x 4 clocks as the existing A,B,C,D rows, then
rebuild the 6-case positioning figure.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from node_jump_analysis import build_chain, Landscape, selection_P
from adaptation_per_a import CASE_DEFS, CASE_COLORS
from info_speed_experiment import spectral_gap
from compare_two_cases import free_fitness5_asymptote
from simplex import (build_sigma, fitness_exp, kl_divergence_bits,
                     log_mean_fitness_genotype_simplex,
                     mutation_matrix_asymmetric, stationary_fast)

SRC = Path(__file__).resolve().parent / "results" / "info_speed_20260727"
CASES = ["A", "B", "C", "D", "F", "G"]
NEW = ["F", "G"]
LABEL = {s: f"{s}: {CASE_DEFS[s][0]}" for s in CASES}
AVALS = [0.05, 0.1, 0.2]
# (regime tag, u, v, r) matching the existing AE rows
CLOCKS = [("N30_u0.005_v0.005_r0", 5e-3, 5e-3, 0.0),
          ("N30_u0.02_v0.02_r0", 2e-2, 2e-2, 0.0),
          ("N30_u0.005_v0.005_r0.5", 5e-3, 5e-3, 0.5),
          ("N30_u0.02_v0.005_r0", 2e-2, 5e-3, 0.0)]
N = 30


def eq_scalars(chain, L):
    """D_eq, F*, tau_spec, mlf at equilibrium -- no trajectory needed."""
    P = selection_P(chain, L, 2)
    psi = stationary_fast(P)
    omega = fitness_exp(build_sigma(L.s1, L.s2, L.eps12))
    lmf = log_mean_fitness_genotype_simplex(chain.fracs, omega)
    mlf = float(psi @ lmf)
    D_eq = kl_divergence_bits(psi, chain.phi)
    M = mutation_matrix_asymmetric(np.array([chain.u, chain.u]),
                                   np.array([chain.v, chain.v]))
    Fstar, _ = free_fitness5_asymptote(psi, chain.phi, chain.fracs, omega, M, N)
    lam2 = spectral_gap(P)
    tau = (-1.0 / np.log(lam2)) if (0 < lam2 < 1) else 200.0
    return dict(D_eq=D_eq, Fstar=Fstar, tau_spec=tau, mlf=mlf)


def main():
    rows = []
    for tag, u, v, r in CLOCKS:
        chain = build_chain(N, u, v, r, 2)
        for s in NEW:
            _, fn = CASE_DEFS[s]
            for a in AVALS:
                s1, s2, eps = fn(a)
                L = Landscape(f"{s}", s, s1=s1, s2=s2, eps12=eps, blurb="")
                m = eq_scalars(chain, L)
                m.update(dict(case=s, a=a, regime=tag))
                rows.append(m)
        print(f"  {tag} done", flush=True)
    fg = pd.DataFrame(rows)
    fg.to_csv(SRC / "fg_metrics.csv", index=False)

    df = pd.read_csv(SRC / "info_speed.csv")
    abcd = df[(df.family == "AE") & (df.case.isin(["A", "B", "C", "D"]))][
        ["case", "a", "regime", "D_eq", "Fstar", "tau_spec", "mlf"]]
    allc = pd.concat([abcd, fg], ignore_index=True)

    def ms(col):
        g = allc.groupby("case")[col]
        return g.mean(), g.sem()

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6))
    for a_, xcol, xl in [(ax[0], "D_eq", r"mean $D_{eq}$ (info, bits)"),
                         (ax[1], "Fstar", r"mean $F^*$ (free fitness, nats)")]:
        mx, sx = ms(xcol); my, sy = ms("tau_spec")
        for s in CASES:
            a_.errorbar(mx[s], my[s], xerr=sx[s], yerr=sy[s], fmt="o", ms=14,
                        c=CASE_COLORS[s], capsize=4, label=LABEL[s])
        a_.set_yscale("log"); a_.set_xlabel(xl)
        a_.set_ylabel(r"mean $\tau_{spec}$ (slow $\to$, log)")
        a_.grid(True, which="both", alpha=.25)
    ax[0].set_title("(i) information vs relaxation time")
    ax[1].set_title("(ii) free fitness vs relaxation time")
    ax[0].legend(fontsize=9, loc="best")
    fig.suptitle("Panel C (6 canonical strategies): equilibrium information / free "
                 "fitness  vs  spectral relaxation time\n"
                 "points = case means +/- SEM over 3 a-values x 4 clocks   "
                 "(N=30)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(SRC / "fig_ABCDFG_positioning.png", dpi=130)
    print("wrote", SRC / "fig_ABCDFG_positioning.png")
    print("\ncase means (pooled):")
    print(allc.groupby("case")[["D_eq", "tau_spec", "Fstar", "mlf"]]
          .mean().round(3).reindex(CASES))


if __name__ == "__main__":
    main()
