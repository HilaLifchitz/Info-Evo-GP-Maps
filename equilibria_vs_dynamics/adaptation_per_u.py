#!/usr/bin/env python3
r"""
Three-panel adaptation curves (information gain, mean log-fitness, free fitness)
for A (sign epistasis) vs B (additive), FIXED selection strength `a`, one row per
mutation rate u=v.  N fixed.

The neutral chain depends on u, so it is rebuilt per row.  tmax is scaled per row
(~ 1/u) so every row reaches equilibrium.  Evolution starts from phi.

Free fitness (panel 3):
    F(psi) = F* - (1/N) D_KL(psi || psi*),
    F*     = <log W>_{psi*} - (1/N) D_KL(psi* || phi)         (logB(alpha) excluded)

Run:
  conda activate miso
  cd equilibria_vs_dynamics
  python adaptation_per_u.py --a 0.05 --N 30 --u 1e-4 2e-4 5e-4 1e-3 2e-3
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

from node_jump_analysis import build_chain, sample_times  # noqa: E402
from adaptation_per_a import (  # noqa: E402
    CASE_COLORS, CASE_DEFS, CASE_ORDER, all_landscapes, curves_for,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float, default=0.05, help="fixed s1 magnitude")
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--u", type=float, nargs="+",
                    default=[1e-4, 2e-4, 5e-4, 1e-3, 2e-3])
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--tmax-base", type=int, default=20000)
    ap.add_argument("--tmax", type=int, default=0,
                    help="if >0, use this fixed tmax for every row (overrides ~4/u)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", type=Path,
                    default=SCRIPT_DIR / "results" /
                    f"adaptation_per_u_{datetime.now():%Y%m%d}")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    a = args.a
    labels = {s: f"{s}: {CASE_DEFS[s][0]}" for s in CASE_ORDER}

    res = {}   # u -> (row{slug:curves}, times, phi_corner)
    for u in args.u:
        # slower mixing at small u -> longer horizon (~ 4/u), floored at tmax_base;
        # or a fixed override via --tmax
        tmax = int(args.tmax) if args.tmax > 0 else int(max(args.tmax_base,
                                                            round(4.0 / u)))
        print(f"# u=v={u:g}  build chain (N={args.N}), tmax={tmax} ...", flush=True)
        chain = build_chain(args.N, u, u, args.r, args.n_jobs)
        cm = float(chain.phi[chain.cls == 1].sum())
        times = sample_times(tmax)
        row = {L.slug: curves_for(chain, L, times, args.n_jobs)
               for L in all_landscapes(a)}
        res[u] = (row, times, cm)
        fs = "  ".join(f"F*_{s}={row[s]['Fstar']:.5f}" for s in CASE_ORDER)
        print(f"   S={len(chain.phi)}  phi corner={cm:.4f}  {fs}  "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)

    nu = len(args.u)
    col_titles = ["Information gain vs time", "Mean log-fitness", "Free fitness",
                  "Cost per generation", "Efficiency $\\eta$"]
    cases_blurb = "A=sign, B=additive, C=reciprocal, D=neutral, E=multiple peaks"

    # combined figure: one row per u, five columns, five case-lines each
    fig, axes = plt.subplots(nu, 5, figsize=(26.0, 3.1 * nu), squeeze=False)
    for i, u in enumerate(args.u):
        row, times, cm = res[u]
        ax_i, ax_m, ax_f, ax_c, ax_e = axes[i]
        for s in CASE_ORDER:
            c = CASE_COLORS[s]
            ax_i.plot(times, row[s]["info"], "-", lw=1.8, color=c, label=labels[s])
            ax_m.plot(times, row[s]["mlf"], "-", lw=1.8, color=c)
            ax_f.plot(times, row[s]["F"], "-", lw=1.8, color=c)
            ax_f.axhline(row[s]["Fstar"], ls=":", lw=0.9, color=c, alpha=0.6)
            ax_c.plot(times, row[s]["cost"], "-", lw=1.8, color=c)
            ax_e.plot(times, row[s]["eta"], "-", lw=1.8, color=c)
        ax_e.axhline(1.0, color="0.5", ls=":", lw=0.9, alpha=0.7)
        for ax in (ax_i, ax_m, ax_f, ax_c, ax_e):
            ax.set_xscale("log"); ax.grid(True, alpha=0.3)
            if i == nu - 1:
                ax.set_xlabel("generation $t$")
        ax_i.set_ylabel(f"$u=v={u:g}$\n(corner {cm:.2f})\n"
                        r"$D_{KL}(\psi\|\varphi)$ [bits]")
        ax_m.set_ylabel(r"$\langle\sum_g x_g\log w_g\rangle$")
        ax_f.set_ylabel(r"$F^*-\frac{1}{N}D_{KL}(\psi\|\psi^*)$ [nats]")
        ax_c.set_ylabel(r"$C(t)=\langle C(x)\rangle_\psi$ [bits]")
        ax_e.set_ylabel(r"$\eta(t)=\frac{dD/dt}{N\,C(t)}$")
        if i == 0:
            ax_i.legend(fontsize=7, ncol=2)
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=12)
    fig.suptitle(f"Adaptation curves per mutation rate $u=v$  (a={a:g}, N={args.N})"
                 f"   {cases_blurb}",
                 fontsize=13, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    combined = out / f"adaptation_per_u_a{a:g}_N{args.N}.png"
    fig.savefig(combined, dpi=150)
    plt.close(fig)

    # standalone 5-panel per u
    for u in args.u:
        row, times, cm = res[u]
        fig, (ax_i, ax_m, ax_f, ax_c, ax_e) = plt.subplots(1, 5, figsize=(26.0, 4.6))
        for s in CASE_ORDER:
            c = CASE_COLORS[s]
            ax_i.plot(times, row[s]["info"], "-", lw=1.8, color=c, label=labels[s])
            ax_m.plot(times, row[s]["mlf"], "-", lw=1.8, color=c, label=labels[s])
            ax_f.plot(times, row[s]["F"], "-", lw=1.8, color=c, label=labels[s])
            ax_f.axhline(row[s]["Fstar"], ls=":", lw=0.9, color=c, alpha=0.6)
            ax_c.plot(times, row[s]["cost"], "-", lw=1.8, color=c, label=labels[s])
            ax_e.plot(times, row[s]["eta"], "-", lw=1.8, color=c, label=labels[s])
        ax_e.axhline(1.0, color="0.5", ls=":", lw=0.9, alpha=0.7, label=r"bound $\eta=1$")
        for ax in (ax_i, ax_m, ax_f, ax_c, ax_e):
            ax.set_xscale("log"); ax.grid(True, alpha=0.3)
            ax.set_xlabel("generation $t$"); ax.legend(fontsize=8)
        ax_i.set_ylabel(r"$D_{KL}(\psi(t)\|\varphi)$ [bits]")
        ax_i.set_title("Information gain vs time")
        ax_m.set_ylabel(r"mean log-fitness $\langle\sum_g x_g\log w_g\rangle$")
        ax_m.set_title("Mean log-fitness")
        ax_f.set_ylabel(r"$F^*-\frac{1}{N}D_{KL}(\psi\|\psi^*)$ [nats]")
        ax_f.set_title("Free fitness")
        ax_c.set_ylabel(r"$C(t)=\langle C(x)\rangle_\psi$ [bits]")
        ax_c.set_title("Cost per generation")
        ax_e.set_ylabel(r"$\eta(t)=\Delta D/(N\,C(t))$")
        ax_e.set_title(r"Efficiency $\eta$")
        fig.suptitle(f"u=v={u:g}  (a={a:g}, N={args.N})", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(out / f"adaptation_u{u:g}_a{a:g}_N{args.N}.png", dpi=150)
        plt.close(fig)

    print(f"\nDone in {time.perf_counter()-t0:.1f}s. Combined: {combined}")


if __name__ == "__main__":
    main()
