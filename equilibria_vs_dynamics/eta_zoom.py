#!/usr/bin/env python3
r"""
Dense per-generation zoom on the early adaptation window for all five cases
(A sign, B additive, C reciprocal, D neutral, E multiple peaks).

Unlike adaptation_per_{a,u}.py (log-spaced snapshots), this evolves EVERY
generation up to --tmax (default 100) so the efficiency curve

    eta(t) = Delta D / (N C(t)),   Delta D = D(t) - D(t-1),

is resolved at full time resolution.  D(t)=KL(psi(t)||phi) [bits],
C(t)=<C(x)>_psi [bits] is the per-generation selective cost.

Saves the raw arrays to <out>/eta_zoom_data.npz and writes a 3-panel figure
(information gain, cost, efficiency) on a LINEAR time axis.

Run:
  conda activate miso
  cd equilibria_vs_dynamics
  python eta_zoom.py --N 30 --a 0.05 --u 0.01 --tmax 100
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

from node_jump_analysis import build_chain, selection_P  # noqa: E402
from adaptation_per_a import (  # noqa: E402
    CASE_COLORS, CASE_DEFS, CASE_ORDER, all_landscapes, cost_per_state,
)
from compare_two_cases import free_fitness5_asymptote  # noqa: E402
from f2b_reference import kl_divergence_nats  # noqa: E402
from simplex import (  # noqa: E402
    build_sigma, fitness_exp, kl_divergence_bits, mutation_matrix_asymmetric,
    stationary_fast,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def dense_track(chain, L, tmax: int, n_jobs: int) -> dict:
    """Per-generation D(t), C(t), mean-log-fitness(t), free-fitness F5(t)."""
    P = selection_P(chain, L, n_jobs)
    psi_star = stationary_fast(P)
    omega = fitness_exp(build_sigma(L.s1, L.s2, L.eps12))
    c_state = cost_per_state(chain.fracs, omega)          # (S,) bits
    per_logfit = chain.fracs @ L.logfit                   # (S,)
    M_mut = mutation_matrix_asymmetric(np.array([chain.u, chain.u]),
                                       np.array([chain.v, chain.v]))
    Fstar, _ = free_fitness5_asymptote(psi_star, chain.phi, chain.fracs,
                                       omega, M_mut, chain.N)
    psi = chain.phi.astype(float).copy()
    T = tmax + 1
    D = np.empty(T); C = np.empty(T); M = np.empty(T); F = np.empty(T)
    D[0] = kl_divergence_bits(psi, chain.phi)
    C[0] = float(psi @ c_state)
    M[0] = float(psi @ per_logfit)
    F[0] = Fstar - kl_divergence_nats(psi, psi_star) / float(chain.N)
    for t in range(1, T):
        psi = P @ psi
        s = float(psi.sum())
        if s > 0:
            psi /= s
        D[t] = kl_divergence_bits(psi, chain.phi)
        C[t] = float(psi @ c_state)
        M[t] = float(psi @ per_logfit)
        F[t] = Fstar - kl_divergence_nats(psi, psi_star) / float(chain.N)
    # eta defined at t=1..tmax : Delta D / (N C(t))
    eta = np.full(T, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta[1:] = np.where(C[1:] > 1e-15, np.diff(D) / (float(chain.N) * C[1:]), np.nan)
    return dict(D=D, C=C, M=M, F=F, eta=eta, Fstar=Fstar)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float, default=0.05)
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--u", type=float, default=0.01)
    ap.add_argument("--v", type=float, default=None)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--tmax", type=int, default=100)
    ap.add_argument("--cases", type=str, nargs="+", default=CASE_ORDER,
                    help="subset/order of case slugs to plot (default all)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", type=Path,
                    default=SCRIPT_DIR / "results" /
                    f"eta_zoom_{datetime.now():%Y%m%d}")
    args = ap.parse_args()
    cases = [s for s in args.cases if s in CASE_DEFS]
    v = args.u if args.v is None else args.v
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    print(f"# build chain N={args.N}, u={args.u:g}, v={v:g}, r={args.r:g} ...", flush=True)
    chain = build_chain(args.N, args.u, v, args.r, args.n_jobs)
    cm = float(chain.phi[chain.cls == 1].sum())
    print(f"  S={len(chain.phi)}  phi corner={cm:.4f}  ({time.perf_counter()-t0:.1f}s)",
          flush=True)

    t = np.arange(args.tmax + 1, dtype=float)
    res = {L.slug: dense_track(chain, L, args.tmax, args.n_jobs)
           for L in all_landscapes(args.a) if L.slug in cases}
    for s in cases:
        pk = np.nanargmax(res[s]["eta"])
        print(f"  {s} ({CASE_DEFS[s][0]:>20s}):  peak eta={res[s]['eta'][pk]:.4f} "
              f"at t={int(t[pk])}   D(T)={res[s]['D'][-1]:.4f} bits", flush=True)

    # save raw data
    tag = "".join(cases)
    npz = out / f"eta_zoom_data_u{args.u:g}_a{args.a:g}_N{args.N}_{tag}.npz"
    save = {"t": t}
    for s in cases:
        for k in ("D", "C", "M", "F", "eta"):
            save[f"{s}_{k}"] = res[s][k]
    np.savez(npz, **save)
    print(f"  saved data -> {npz}", flush=True)

    labels = {s: f"{s}: {CASE_DEFS[s][0]}" for s in cases}
    # layout: row 1 = 3 panels (info, cost, eta), row 2 = 2 panels (mlf, free fit)
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 9.2))
    ax_i, ax_c, ax_e = axes[0]
    ax_m, ax_f, ax_dead = axes[1]
    ax_dead.axis("off")
    for s in cases:
        c = CASE_COLORS[s]
        ax_i.plot(t, res[s]["D"], "-", lw=1.9, color=c, label=labels[s])
        ax_c.plot(t, res[s]["C"], "-", lw=1.9, color=c, label=labels[s])
        ax_e.plot(t, res[s]["eta"], "-", lw=1.9, color=c, label=labels[s])
        ax_m.plot(t, res[s]["M"], "-", lw=1.9, color=c, label=labels[s])
        ax_f.plot(t, res[s]["F"], "-", lw=1.9, color=c, label=labels[s])
        ax_f.axhline(res[s]["Fstar"], ls=":", lw=0.9, color=c, alpha=0.6)
    ax_e.axhline(1.0, color="0.5", ls=":", lw=0.9, alpha=0.7, label=r"bound $\eta=1$")
    for ax in (ax_i, ax_c, ax_e, ax_m, ax_f):
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("generation $t$")
        ax.set_xlim(0, args.tmax)
        ax.legend(fontsize=8)
    ax_i.set_ylabel(r"$D(t)=D_{KL}(\psi\|\varphi)$ [bits]")
    ax_i.set_title("Information gain (per generation)")
    ax_c.set_ylabel(r"$C(t)=\langle C(x)\rangle_\psi$ [bits]")
    ax_c.set_title("Cost per generation")
    ax_e.set_ylabel(r"$\eta(t)=\Delta D/(N\,C(t))$")
    ax_e.set_title(r"Efficiency $\eta$")
    ax_m.set_ylabel(r"mean log-fitness $\langle\sum_g x_g\log w_g\rangle$")
    ax_m.set_title("Mean log-fitness")
    ax_f.set_ylabel(r"$F^*-\frac{1}{N}D_{KL}(\psi\|\psi^*)$ [nats]")
    ax_f.set_title(r"Free fitness (dotted $=F^*$)")
    blurb = ", ".join(f"{s}={CASE_DEFS[s][0]}" for s in cases)
    fig.suptitle(f"Early-window zoom (dense per-generation)  "
                 f"u=v={args.u:g}, a={args.a:g}, N={args.N}   {blurb}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = out / f"eta_zoom_u{args.u:g}_a{args.a:g}_N{args.N}_t{args.tmax}_{tag}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"\nDone in {time.perf_counter()-t0:.1f}s. Figure: {png}")


if __name__ == "__main__":
    main()
