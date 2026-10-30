#!/usr/bin/env python3
r"""
Controlled comparison of two fitness landscapes with the SAME endpoints.

  Case A (sign epistasis):   s1 = a,  s2 = -a,  eps12 = 2a
      -> log-fitness  [00, 01, 10, 11] = [0, -a, +a, +2a]
      The locus-2 mutation is DELETERIOUS on the 0 background (00->01 = -a)
      but BENEFICIAL on the 1 background (10->11 = +a): one sign flip.
      Only ONE uphill route exists: 00 -> 10 -> 11.  The 00->01 route is a
      valley step.  Note the ADDITIVE part s1+s2 = 0 -- all of 11's fitness is
      epistatic.

  Case B (additive):         s1 = a,  s2 = a,   eps12 = 0
      -> log-fitness  [00, 01, 10, 11] = [0, +a, +a, +2a]
      Both single steps uphill; TWO equivalent monotone routes to 11.

Both have the SAME base (00 = 0) and the SAME global optimum (11 = 2a), so the
equilibrium is nearly identical -- the interest is entirely in the PATH and the
SPEED of getting there.

Outputs (results/compare_cases_<date>/):
  concentration_AB.png     corner mass vs t, A vs B
  basin_from_phi_AB.png    corner-basin mass vs t (start phi), A vs B
  basin_from_00_AB.png     corner-basin mass vs t (start pure 00), A vs B
  jumps_A.png / jumps_B.png   node-jump graphs (empirical Q vs SSWM)
  adaptation_AB.png        KL(t) and mean log-fitness(t), A vs B
  summary.txt

Run:
  conda activate miso
  cd equilibria_vs_dynamics
  python compare_two_cases.py --a 0.1 --N 24 --u 2e-4 --tmax 20000
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from node_jump_analysis import (  # noqa: E402
    GENO,
    Landscape,
    basin_mass,
    build_chain,
    class_mass,
    detailed_balance_residual,
    evolve_many,
    lumped_Q,
    plot_jump_graph,
    sample_times,
    selection_P,
    sswm_rates,
)
from scipy.special import gammaln  # noqa: E402

from simplex import (  # noqa: E402
    build_sigma,
    fitness_exp,
    free_fitness_1,
    kl_divergence_bits,
    log_mean_fitness_genotype_simplex,
    mutation_matrix_asymmetric,
    stationary_fast,
)
from f2b_reference import (  # noqa: E402
    cond_alpha_exponents,
    kl_divergence_nats,
    log_weights_cond_alpha,
    log_Z_discrete,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def mfpt_to_basin(P_col: np.ndarray, basin: np.ndarray, target_g: int) -> np.ndarray:
    """Expected #generations to first reach the target corner basin, per start state.

    P_col is column-stochastic (P_col[j,i] = prob i->j); we use the row-stochastic
    form P_row[i,j] = prob i->j and solve  (I - P_row[T,T]) h = 1  on transient T.
    """
    P_row = P_col.T
    T = np.where(basin != target_g)[0]
    A = np.eye(len(T)) - P_row[np.ix_(T, T)]
    h_T = np.linalg.solve(A, np.ones(len(T)))
    h = np.zeros(P_col.shape[0])
    h[T] = h_T
    return h


def four_state_stationary(Q: np.ndarray) -> np.ndarray:
    """Stationary distribution of the 4-node lumped chain (rows=from)."""
    # Q is row-stochastic-ish (rows sum ~1). Left eigenvector for eigenvalue 1.
    A = np.vstack([Q.T - np.eye(4), np.ones(4)])
    b = np.array([0, 0, 0, 0, 1.0])
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    return p


def cycle_currents(Q: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Net probability currents on the square edges: J(g->h) = p_g Q_gh - p_h Q_hg.

    Square cycle order: 00(0) -> 10(2) -> 11(3) -> 01(1) -> 00(0).
    A reversible chain has all J = 0; a nonzero common loop value = circulation.
    """
    cyc = [(0, 2), (2, 3), (3, 1), (1, 0)]
    out = {}
    for g, h in cyc:
        out[f"{GENO[g]}->{GENO[h]}"] = float(p[g] * Q[g, h] - p[h] * Q[h, g])
    return out


def cycle_affinity(Q: np.ndarray) -> float:
    r"""Scale-free (Kolmogorov) circulation of the 4-node square cycle.

        A = ln [ Q(00->10) Q(10->11) Q(11->01) Q(01->00)
               / Q(00->01) Q(01->11) Q(11->10) Q(10->00) ]

    Kolmogorov's criterion: the lumped chain is reversible on this cycle iff
    A = 0 (product of jump rates the same both ways round the loop).  A != 0 is
    a genuine circulating tendency, independent of the stationary weights p;
    sign gives the preferred rotation sense.
    """
    fwd = Q[0, 2] * Q[2, 3] * Q[3, 1] * Q[1, 0]
    rev = Q[0, 1] * Q[1, 3] * Q[3, 2] * Q[2, 0]
    if fwd <= 0 or rev <= 0:
        return float("nan")
    return float(np.log(fwd / rev))


def free_fitness_series(snaps: np.ndarray, w_per_state: np.ndarray, N: int,
                        col: int) -> np.ndarray:
    """F_N(psi(t)) = <W>_psi + H(psi)/N  over time (simplex.free_fitness_1)."""
    return np.array([free_fitness_1(snaps[k, :, col], w_per_state, N)
                     for k in range(snaps.shape[0])])


def w_potential_mu(fracs: np.ndarray, omega: np.ndarray, M_mut: np.ndarray,
                   N: int, eps: float | None = None) -> np.ndarray:
    r"""Per-state free-fitness potential with the FULL mutation term (raw influx).

        W(x) = log( sum_g x_g w_g )  -  (1/N) sum_g (1 - 2N * influx_g) * log x_g

    influx_g = (x@M)_g - x_g M_gg  is the mutational flow INTO genotype g from the
    other genotypes.  This is the (1 - 2N mu) structure that was missing from the
    old potential (old code kept only the -(1/N) (x@M)_g log x_g piece and dropped
    the standalone -(1/N) log x_g term).  Boundary floored at eps = 1/N.
    """
    eps = 1.0 / float(N) if eps is None else eps
    x_eff = np.maximum(fracs, eps)
    a = cond_alpha_exponents(x_eff, M_mut, N, alpha_mode="influx", factor=2.0)
    logw = np.log(np.maximum(x_eff @ omega, 1e-300))
    return logw - np.sum((1.0 - a) * np.log(x_eff), axis=1) / float(N)


def w_potential_condalpha(fracs: np.ndarray, omega: np.ndarray, M_mut: np.ndarray,
                          N: int) -> np.ndarray:
    r"""Same potential but with the CONDITIONED exponent a_g = alpha_g.

        alpha_g = 2N influx_g / (1 - x_g)
        W(x) = log( sum_g x_g w_g ) - (1/N) sum_g (1 - alpha_g) log x_g = ell(x)/N

    So free_fitness_1 with this potential is IDENTICALLY F_boltz(psi).
    """
    ell, valid = log_weights_cond_alpha(fracs, omega, M_mut, N, eps=1.0 / float(N))
    w = ell / float(N)
    w[~valid] = -np.inf
    return w


def dirichlet_log_beta(alpha: np.ndarray) -> float:
    """log B(alpha) = sum_i logGamma(alpha_i) - logGamma(sum_i alpha_i)."""
    a = np.clip(np.asarray(alpha, dtype=float), 1e-9, None)
    return float(gammaln(a).sum() - gammaln(a.sum()))


def free_fitness5_asymptote(psi_star: np.ndarray, phi: np.ndarray, fracs: np.ndarray,
                            omega: np.ndarray, M_mut: np.ndarray, N: int,
                            include_logB: bool = False) -> tuple[float, dict]:
    r"""Equilibrium free-fitness VALUE at the true 2-loci psi* (picture 2):

        F* = <log W>_{psi*}  -  (1/N) D_KL(psi* || phi)  [ - (1/N) log B(alpha) ]

    <log W>_{psi*} uses log mean fitness log(sum_g x_g w_g); phi is the neutral
    stationary.  The Dirichlet-normalizer log B(alpha) (alpha = cond-alpha at the
    mean composition E_{psi*}[x]) is a nuisance offset that is NOT shared across
    landscapes (it depends on how psi* concentrates), so by default it is EXCLUDED:
    then F* is exactly <log W> - Info/N, which ranks landscapes by adaptation minus
    information.  It is still computed and reported for reference.
    """
    logmeanfit = log_mean_fitness_genotype_simplex(fracs, omega)   # (S,)
    mean_logW = float(psi_star @ logmeanfit)
    dkl = kl_divergence_nats(psi_star, phi)
    xbar = np.maximum(psi_star @ fracs, 1.0 / float(N))
    alpha = cond_alpha_exponents(xbar, M_mut, N, alpha_mode="cond", factor=2.0)
    logB = dirichlet_log_beta(alpha)
    Fstar = mean_logW - dkl / float(N)
    if include_logB:
        Fstar -= logB / float(N)
    return Fstar, dict(mean_logW=mean_logW, dkl=dkl, logB=logB, alpha=alpha,
                       include_logB=include_logB)


def free_fitness5_series(snaps: np.ndarray, psi_star: np.ndarray, Fstar: float,
                         N: int, col: int) -> np.ndarray:
    r"""F5(psi(t)) = F*  -  (1/N) D_KL(psi(t) || psi*_chain).

    Because D_KL(psi(t)||psi*) is a non-increasing Lyapunov functional for the
    Markov chain, F5 is guaranteed MONOTONICALLY INCREASING up to F*.
    """
    return np.array([Fstar - kl_divergence_nats(snaps[k, :, col], psi_star) / float(N)
                     for k in range(snaps.shape[0])])


def boltz_reference(fracs: np.ndarray, omega: np.ndarray, M_mut: np.ndarray,
                    N: int) -> tuple[np.ndarray, float]:
    """Cond-alpha Boltzmann reference psi* = exp(ell)/Z and log Z (f2b_reference).

        ell(x) = N log W(x) + sum_g (alpha_g - 1) log x_g ,   alpha_g = 2N influx/(1-x_g)

    Boundary handled by eps = 1/N (so corners are included).  psi* is the Gibbs
    equilibrium of this potential; F_boltz is maximized (= logZ/N) at psi*.
    """
    ell, valid = log_weights_cond_alpha(fracs, omega, M_mut, N, eps=1.0 / float(N))
    logZ = log_Z_discrete(ell, valid)
    psi_star = np.zeros_like(ell, dtype=float)
    psi_star[valid] = np.exp(ell[valid] - logZ)
    s = float(psi_star.sum())
    if s > 0:
        psi_star /= s
    return psi_star, float(logZ)


def boltz_free_fitness_series(snaps: np.ndarray, psi_star_boltz: np.ndarray,
                              logZ: float, N: int, col: int) -> np.ndarray:
    """F_boltz(psi(t)) = logZ/N - (1/N) D_KL(psi || psi*_boltz)   [nats]."""
    return np.array([logZ / N - kl_divergence_nats(snaps[k, :, col], psi_star_boltz) / N
                     for k in range(snaps.shape[0])])


def mean_logfit_series(snaps: np.ndarray, fracs: np.ndarray, logfit: np.ndarray,
                       col: int) -> np.ndarray:
    """E_psi[ population mean log-fitness ] over time for one column.

    population mean log-fitness at state x = sum_g x_g * logfit_g.
    """
    per_state = fracs @ logfit          # (S,)
    return snaps[:, :, col] @ per_state


def kl_series(snaps: np.ndarray, phi: np.ndarray, col: int) -> np.ndarray:
    return np.array([kl_divergence_bits(snaps[k, :, col], phi)
                     for k in range(snaps.shape[0])])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float, default=0.1, help="s1 magnitude")
    ap.add_argument("--N", type=int, default=24)
    ap.add_argument("--u", type=float, default=2e-4)
    ap.add_argument("--v", type=float, default=2e-4)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--tmax", type=int, default=20000)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", type=Path,
                    default=SCRIPT_DIR / "results" /
                    f"compare_cases_{datetime.now():%Y%m%d}")
    args = ap.parse_args()
    a = args.a
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    LA = Landscape("Case A: sign epistasis", "A", s1=a, s2=-a, eps12=2 * a,
                   blurb=f"logw=[0,-{a:g},+{a:g},+{2*a:g}]; only route 00->10->11")
    LB = Landscape("Case B: additive", "B", s1=a, s2=a, eps12=0.0,
                   blurb=f"logw=[0,+{a:g},+{a:g},+{2*a:g}]; both routes uphill")

    print(f"Building neutral chain N={args.N}, u=v={args.u}, r={args.r} ...", flush=True)
    chain = build_chain(args.N, args.u, args.v, args.r, args.n_jobs)
    S = len(chain.phi)
    pure_idx = {g: int(np.where(chain.counts[:, g] == chain.N)[0][0]) for g in range(4)}
    print(f"  S={S}. phi corner mass={chain.phi[chain.cls==1].sum():.4f}  "
          f"(built {time.perf_counter()-t0:.1f}s)")

    times = sample_times(args.tmax)
    lines: list[str] = [f"a={a} N={args.N} u=v={args.u} r={args.r} tmax={args.tmax}",
                        f"phi corner mass = {chain.phi[chain.cls==1].sum():.6f}", ""]

    results = {}
    for L in (LA, LB):
        print(f"\n=== {L.name}  logfit={dict(zip(GENO, L.logfit))} ===", flush=True)
        P = selection_P(chain, L, args.n_jobs)
        psi_star = stationary_fast(P)
        star_basin = np.array([psi_star[chain.basin == g].sum() for g in range(4)])
        kl_eq = kl_divergence_bits(psi_star, chain.phi)

        # evolve from phi and from pure 00
        Psi0 = np.zeros((S, 2))
        Psi0[:, 0] = chain.phi
        Psi0[pure_idx[0], 1] = 1.0
        snaps = evolve_many(P, Psi0, times)

        Q = lumped_Q(P, chain)
        K = sswm_rates(L, chain.N, chain.u)
        p4 = four_state_stationary(Q)
        cyc = cycle_currents(Q, p4)
        affinity = cycle_affinity(Q)
        db = detailed_balance_residual(P, psi_star)

        # free-fitness potentials per state (uses this landscape's omega + M)
        omega = fitness_exp(build_sigma(L.s1, L.s2, L.eps12))
        M_mut = mutation_matrix_asymmetric(np.array([chain.u, chain.u]),
                                           np.array([chain.v, chain.v]))
        # (1) corrected potential -- full (1 - 2N influx) mutation term
        w_mu = w_potential_mu(chain.fracs, omega, M_mut, chain.N)
        # cond-alpha exponent variant (equals the F_boltz potential ell/N)
        w_ca = w_potential_condalpha(chain.fracs, omega, M_mut, chain.N)

        # cond-alpha Boltzmann reference: F_boltz = logZ/N - (1/N) KL(psi||psi*_boltz)
        psi_star_boltz, logZ = boltz_reference(chain.fracs, omega, M_mut, chain.N)

        # (2) equilibrium free-fitness value F* at the TRUE 2-loci psi* (picture 2)
        Fstar, Fstar_parts = free_fitness5_asymptote(
            psi_star, chain.phi, chain.fracs, omega, M_mut, chain.N)

        # mean first-passage time (generations) to the 11 basin
        h = mfpt_to_basin(P, chain.basin, target_g=3)
        mfpt_00 = float(h[pure_idx[0]])
        mfpt_phi = float(chain.phi @ h)

        results[L.slug] = dict(
            L=L, P=P, snaps=snaps, Q=Q, K=K, psi_star=psi_star,
            star_basin=star_basin, kl_eq=kl_eq, cyc=cyc, db=db,
            mfpt_00=mfpt_00, mfpt_phi=mfpt_phi, affinity=affinity,
            w_mu=w_mu, w_ca=w_ca,
            psi_star_boltz=psi_star_boltz, logZ=logZ,
            Fstar=Fstar, Fstar_parts=Fstar_parts,
        )
        plot_jump_graph(L, Q, K, chain.N, out / f"jumps_{L.slug}.png")

        print(f"  psi* basins {dict(zip(GENO, np.round(star_basin,4)))}")
        print(f"  KL_eq(psi*||phi) = {kl_eq:.4f} bits")
        print(f"  F_N(psi*)  mu-variant       = {free_fitness_1(psi_star, w_mu, chain.N):.5f}")
        print(f"  F_N(psi*)  cond-alpha       = {free_fitness_1(psi_star, w_ca, chain.N):.5f}")
        print(f"  logZ/N (cond-alpha)         = {logZ/chain.N:.5f}  = F_boltz(psi*_boltz)")
        print(f"  F* (asymptote, logB excluded)= {Fstar:.5f}  "
              f"[= <logW> - Info/N]")
        print(f"       <logW>_psi*={Fstar_parts['mean_logW']:.5f}  "
              f"Info=D_KL(psi*||phi)/N={Fstar_parts['dkl']/chain.N:.5f} nats  "
              f"(logB/N={Fstar_parts['logB']/chain.N:.5f}, excluded)")
        print(f"  MFPT 00 -> 11 basin: {mfpt_00:,.0f} gen   (from phi: {mfpt_phi:,.0f})")
        print(f"  detailed-balance rel asym = {db['rel_max']:.3e}")
        print(f"  cycle AFFINITY (Kolmogorov) = {affinity:+.4f}  "
              f"(0 = reversible loop; sign = rotation sense)")
        print(f"  square-cycle net currents (00->10->11->01->00):")
        for k, val in cyc.items():
            print(f"      {k}: {val:+.3e}")
        print("  lumped Q (rows from, cols to 00,01,10,11):")
        for g in range(4):
            print("     " + GENO[g] + " " + "  ".join(f"{Q[g,h]:.3e}" for h in range(4)))

        lines += [
            f"[{L.slug}] {L.name}",
            f"  logfit = {dict(zip(GENO, np.round(L.logfit,3)))}",
            f"  psi* basins = {np.round(star_basin,4).tolist()}",
            f"  KL_eq = {kl_eq:.4f} bits",
            f"  MFPT(00->11) = {mfpt_00:.1f} gen ; MFPT(phi->11) = {mfpt_phi:.1f} gen",
            f"  DB rel asym = {db['rel_max']:.3e}",
            f"  cycle affinity = {affinity:+.4f}",
            f"  cycle currents = {cyc}",
            "",
        ]

    # ------- comparison plots -------
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    # concentration (corner mass) A vs B
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for L, ls in [(LA, "-"), (LB, "--")]:
        cm = class_mass(results[L.slug]["snaps"], chain.cls, 0)
        ax.plot(times, cm[1], ls, color="#1f77b4", lw=2,
                label=f"{L.slug}: monomorphic")
        ax.plot(times, cm[2], ls, color="#ff7f0e", lw=1.6,
                label=f"{L.slug}: edges")
    ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("generation $t$"); ax.set_ylabel("mass by support size")
    ax.set_title("Corner concentration through the process (start $\\varphi$)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "concentration_AB.png", dpi=150)
    plt.close(fig)

    # basin mass from phi and from pure 00, A vs B (2x2)
    for startcol, tag, title in [(0, "phi", "start $\\psi(0)=\\varphi$"),
                                 (1, "00", "start = pure 00")]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        for ax, L in zip(axes, (LA, LB)):
            bm = basin_mass(results[L.slug]["snaps"], chain.basin, startcol)
            for g in range(4):
                ax.plot(times, bm[:, g], "-", lw=2, color=colors[g],
                        label=f"{GENO[g]} (logw={L.logfit[g]:+.2f})")
            ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel("generation $t$")
            mf = results[L.slug]["mfpt_00" if tag == "00" else "mfpt_phi"]
            ax.set_title(f"{L.name}\nMFPT to 11 basin = {mf:,.0f} gen", fontsize=10)
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        axes[0].set_ylabel("corner-basin mass")
        fig.suptitle(f"Flow between nodes, {title}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out / f"basin_from_{tag}_AB.png", dpi=150)
        plt.close(fig)

    # adaptation curves (1x3):
    #   1 information  D_KL(psi||phi)
    #   2 mean log-fitness (raw)
    #   3 free fitness  F(psi) = F* - (1/N) D_KL(psi||psi*),
    #                   F* = <logW>_{psi*} - (1/N) D_KL(psi*||phi)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 4.8))
    for L, c in [(LA, "#c44e52"), (LB, "#4c72b0")]:
        R = results[L.slug]
        snaps = R["snaps"]
        klt = kl_series(snaps, chain.phi, 0)
        mlf = mean_logfit_series(snaps, chain.fracs, L.logfit, 0)
        f5 = free_fitness5_series(snaps, R["psi_star"], R["Fstar"], chain.N, 0)

        ax1.plot(times, klt, "-", lw=2, color=c, label=L.name)
        ax2.plot(times, mlf, "-", lw=2, color=c, label=L.name)
        ax3.plot(times, f5, "-", lw=2, color=c, label=L.name)
        ax3.axhline(R["Fstar"], ls=":", lw=1.0, color=c, alpha=0.7)
    for ax in (ax1, ax2, ax3):
        ax.set_xscale("log"); ax.set_xlabel("generation $t$")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    ax1.set_ylabel(r"$D_{KL}(\psi(t)\,\|\,\varphi)$ [bits]")
    ax1.set_title("Information gain vs time")
    ax2.set_ylabel(r"mean log-fitness $\langle \sum_g x_g \log w_g\rangle$")
    ax2.set_title("Mean log-fitness (raw)")
    ax3.set_ylabel(r"$\mathcal{F}(\psi)=F^*-\frac{1}{N}D_{KL}(\psi\|\psi^*)$ [nats]")
    ax3.set_title(r"Free fitness (dotted $=F^*=\langle\log W\rangle_{\psi^*}"
                  r"-\frac{1}{N}D_{KL}(\psi^*\|\varphi)$)")
    fig.suptitle("Adaptation: same endpoints, different path "
                 "(A = sign epistasis, B = additive)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "adaptation_AB.png", dpi=150)
    plt.close(fig)

    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nDone in {time.perf_counter()-t0:.1f}s. Outputs in {out}")


if __name__ == "__main__":
    main()
