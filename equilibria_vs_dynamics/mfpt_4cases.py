#!/usr/bin/env python3
r"""
Mean first-passage time (MFPT) to the 11 basin for the four landscapes
(A sign, B additive, C reciprocal sign, D neutral intermediates).

"Time till the population is in the 11 blob": expected number of generations for
a population to first enter the set of states where 11 is the plurality genotype,
starting from the pure-00 corner (and, for reference, from the neutral phi).
Computed by the linear system  (I - P_TT) h = 1  on the transient set T (no time
integration), so it is fast and exact.

Uses a relatively high mutation rate so the reciprocal-sign valley crossing (C)
does not blow up to astronomical times.

Run:
  conda activate miso
  cd equilibria_vs_dynamics
  python mfpt_4cases.py --N 30 --u 2e-3 --a 0.02 0.05 0.1 0.15 0.2 0.3
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from node_jump_analysis import build_chain, selection_P  # noqa: E402
from adaptation_per_a import CASE_COLORS, CASE_DEFS, CASE_ORDER, all_landscapes  # noqa: E402
from compare_two_cases import mfpt_to_basin  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--u", type=float, default=2e-3, help="high mutation rate u=v")
    ap.add_argument("--a", type=float, nargs="+",
                    default=[0.02, 0.05, 0.1, 0.15, 0.2, 0.3])
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", type=Path,
                    default=SCRIPT_DIR / "results" /
                    f"mfpt_4cases_{datetime.now():%Y%m%d}")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"Building neutral chain N={args.N}, u=v={args.u:g} ...", flush=True)
    chain = build_chain(args.N, args.u, args.u, args.r, args.n_jobs)
    cm = float(chain.phi[chain.cls == 1].sum())
    pure00 = int(np.where(chain.counts[:, 0] == chain.N)[0][0])
    print(f"  S={len(chain.phi)}  phi corner mass={cm:.4f}", flush=True)

    mfpt00 = {s: [] for s in CASE_ORDER}
    mfptphi = {s: [] for s in CASE_ORDER}
    rows = []
    for a in args.a:
        line = []
        for L in all_landscapes(a):
            P = selection_P(chain, L, args.n_jobs)
            h = mfpt_to_basin(P, chain.basin, target_g=3)
            m00, mphi = float(h[pure00]), float(chain.phi @ h)
            mfpt00[L.slug].append(m00); mfptphi[L.slug].append(mphi)
            line.append(f"{L.slug}={m00:,.0f}")
            rows.append(dict(a=a, case=L.slug, u=args.u, N=args.N,
                             mfpt_from00=m00, mfpt_from_phi=mphi))
        print(f"  a={a:.3f}: MFPT(00->11 basin)  " + "  ".join(line), flush=True)

    with (out / "mfpt.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    labels = {s: f"{s}: {CASE_DEFS[s][0]}" for s in CASE_ORDER}
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for s in CASE_ORDER:
        c = CASE_COLORS[s]
        ax0.plot(args.a, mfpt00[s], "o-", color=c, lw=2, label=labels[s])
        ax1.plot(args.a, mfptphi[s], "o-", color=c, lw=2, label=labels[s])
    for ax, ttl in ((ax0, "from pure 00"), (ax1, r"from neutral $\varphi$")):
        ax.set_yscale("log")
        ax.set_xlabel(r"selection strength $a=s_1$")
        ax.set_ylabel("MFPT to 11 basin [generations]")
        ax.set_title(f"Time to reach the 11 blob ({ttl})")
        ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=9)
    fig.suptitle(f"Mean first-passage time to the 11 basin  "
                 f"(N={args.N}, u=v={args.u:g}, corner mass {cm:.2f})",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / f"mfpt_4cases_N{args.N}_u{args.u:g}.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved {out / f'mfpt_4cases_N{args.N}_u{args.u:g}.png'}")


if __name__ == "__main__":
    main()
