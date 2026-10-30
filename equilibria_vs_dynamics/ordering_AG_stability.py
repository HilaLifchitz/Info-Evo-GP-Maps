#!/usr/bin/env python3
"""Ordering-stability analysis for the 6 iso-peak strategies A,B,C,D,F,G.

IMPORTANT naming:
  F = synergy iso-peak  Sigma=[[a/2,a/2],[a/2,a/2]]  lf=[0,a/2,a/2,2a]
  G = additive asymmetric
  old case E (multiple peaks) is EXCLUDED — do not confuse with F.

Computes diffusion-limit equilibrium metrics over (a,u,v,r) slices, then
replicates the earlier ordering_slices / bumpchart analysis for these 6 cases.
"""
from __future__ import annotations

from collections import Counter
from math import log
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from node_jump_analysis import build_chain, Landscape
from adaptation_per_a import CASE_DEFS, CASE_COLORS
from compare_two_cases import boltz_reference
from simplex import (
    build_sigma, fitness_exp, log_mean_fitness_genotype_simplex,
    mutation_matrix_asymmetric,
)
from f2b_reference import kl_divergence_nats

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "results" / "ordering_AG_20260729"
CANON = ["A", "B", "C", "D", "F", "G"]   # no E
LN2 = log(2.0)
METRICS = [
    ("info_diff", "information"),
    ("mlf_diff", "mean log-fitness"),
    ("F_diff", "free fitness F"),
]


def landscapes(a: float):
    out = []
    for s in CANON:
        name, fn = CASE_DEFS[s]
        s1, s2, eps = fn(a)
        out.append(Landscape(f"{s}: {name}", s, s1=s1, s2=s2, eps12=eps, blurb=""))
    return out


def build_medium_grid():
    a_vals = [0.05, 0.08, 0.11, 0.14, 0.18, 0.22, 0.28, 0.35, 0.42]
    sym = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2]
    asym = [(2e-2, 2e-3), (2e-3, 2e-2), (1e-2, 1e-3), (5e-3, 5e-2)]
    r_vals = [0.0, 0.1, 0.5]
    combos = []
    for r in r_vals:
        for u in sym:
            combos.append((u, u, r, "sym"))
        for u, v in asym:
            combos.append((u, v, r, "asym"))
    return a_vals, combos


def sweep(N: int = 30) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    a_vals, combos = build_medium_grid()
    rows = []
    t0 = time.perf_counter()
    n_total = len(combos) * len(a_vals) * len(CANON)
    print(f"# AG ordering sweep  N={N}  combos={len(combos)}  a={len(a_vals)}  "
          f"cases={CANON}  -> {n_total} rows", flush=True)
    for ci, (u, v, r, kind) in enumerate(combos):
        chain = build_chain(N, u, v, r, 2)
        fracs = chain.fracs
        M = mutation_matrix_asymmetric(np.array([u, u]), np.array([v, v]))
        phi_b, _ = boltz_reference(fracs, fitness_exp(np.zeros((2, 2))), M, N)
        for a in a_vals:
            for L in landscapes(a):
                omega = fitness_exp(build_sigma(L.s1, L.s2, L.eps12))
                logmf = log_mean_fitness_genotype_simplex(fracs, omega)
                psm = fracs @ L.logfit
                psi, logZ = boltz_reference(fracs, omega, M, N)
                dkl = kl_divergence_nats(psi, phi_b)
                lwb = float(psi @ logmf)
                mlf = float(psi @ psm)
                rows.append(dict(
                    N=N, u=u, v=v, r=r, mut_kind=kind, a=a, case=L.slug,
                    case_name=CASE_DEFS[L.slug][0],
                    lf00=float(L.logfit[0]), lf01=float(L.logfit[1]),
                    lf10=float(L.logfit[2]), lf11=float(L.logfit[3]),
                    mlf_diff=mlf, lwb_diff=lwb, info_diff=dkl / LN2,
                    F_diff=lwb - dkl / N, G_diff=mlf - dkl / N,
                    Fboltz=logZ / N,
                ))
        pd.DataFrame(rows).to_csv(OUT / "sweep_AG.csv", index=False)
        print(f"  [{ci+1}/{len(combos)}] u={u:g} v={v:g} r={r:g} ({kind})  "
              f"rows={len(rows)}  ({time.perf_counter()-t0:.1f}s)", flush=True)
    return pd.DataFrame(rows)


def ordering(sub: pd.DataFrame, metric: str) -> tuple:
    vals = {c: sub[sub.case == c][metric].mean() for c in CANON}
    return tuple(sorted(CANON, key=lambda c: -vals[c]))


def analyse_slice(df, fixed, vary, metric):
    axis_cols = fixed + (["u", "v"] if vary == "uv" else [vary])
    modal_fracs, n_distinct, perfect = [], [], 0
    modal_orders = Counter()
    n_groups = 0
    for _, g in df.groupby(fixed):
        pts = g.groupby(axis_cols)
        if pts.ngroups < 3:
            continue
        orders = [ordering(sub, metric) for _, sub in pts]
        cnt = Counter(orders)
        top, topn = cnt.most_common(1)[0]
        modal_fracs.append(topn / len(orders))
        n_distinct.append(len(cnt))
        perfect += int(len(cnt) == 1)
        modal_orders[top] += 1
        n_groups += 1
    if n_groups == 0:
        return None
    return dict(
        mean_modal_frac=float(np.mean(modal_fracs)),
        perfect_frac=perfect / n_groups,
        mean_n_distinct=float(np.mean(n_distinct)),
        n_groups=n_groups,
        top_ordering=modal_orders.most_common(1)[0][0],
        modal_orders=modal_orders,
    )


def ranks_along(df, axis, metric, fixed: dict):
    sub = df.copy()
    for k, v in fixed.items():
        sub = sub[np.isclose(sub[k], v)]
    xvals = np.sort(sub[axis].unique())
    ranks = {c: [] for c in CANON}
    for x in xvals:
        s = sub[np.isclose(sub[axis], x)]
        vals = {c: s[s.case == c][metric].mean() for c in CANON}
        order = sorted(CANON, key=lambda c: -vals[c])
        for rank, c in enumerate(order, start=1):
            ranks[c].append(rank)
    return xvals, ranks


def report_and_figs(df: pd.DataFrame):
    SLICES = [
        ("vary u=v (sym), fix a,r", ["a", "r"], "u", dict(mut_kind="sym")),
        ("vary u,v (asym), fix a,r", ["a", "r"], "uv", dict(mut_kind="asym")),
        ("vary r, fix a,u,v", ["a", "u", "v"], "r", {}),
        ("vary a, fix u,v,r", ["u", "v", "r"], "a", {}),
    ]
    lines = []
    p = lines.append
    p("=" * 78)
    p("ORDERING STABILITY: A,B,C,D,F,G  (corrected iso-peak F; old E excluded)")
    p("  F = synergy [[a/2,a/2],[a/2,a/2]] -> lf [0,a/2,a/2,2a]")
    p("  G = additive asymmetric [[a/2,0],[0,3a/2]]")
    p("  metrics from diffusion / Boltzmann equilibrium (N=30)")
    p("=" * 78)

    heat = []  # rows for heatmap
    for slab, fixed, vary, restrict in SLICES:
        src = df.copy()
        for k, v in restrict.items():
            src = src[src[k] == v]
        p(f"\n--- {slab} ---")
        for mcol, mlab in METRICS:
            res = analyse_slice(src, fixed, vary, mcol)
            if res is None:
                p(f"  {mlab:20s}  (too few groups)")
                continue
            top = "".join(res["top_ordering"])
            p(f"  {mlab:20s}  modal_frac={res['mean_modal_frac']:.3f}  "
              f"perfect={res['perfect_frac']:.3f}  "
              f"n_distinct~{res['mean_n_distinct']:.2f}  "
              f"n_groups={res['n_groups']}  top={top}")
            heat.append(dict(slice=slab, metric=mlab,
                             modal_frac=res["mean_modal_frac"],
                             perfect=res["perfect_frac"],
                             top=top))

    # global modal ordering per metric (all rows as independent points)
    p("\n--- global modal ordering (every (u,v,r,a) as one point) ---")
    for mcol, mlab in METRICS:
        pts = df.groupby(["u", "v", "r", "a"])
        orders = [ordering(sub, mcol) for _, sub in pts]
        cnt = Counter(orders)
        top, n = cnt.most_common(1)[0]
        p(f"  {mlab:20s}  top={''.join(top)}  share={n/len(orders):.3f}  "
          f"n_distinct={len(cnt)}  / {len(orders)} points")
        # top 3
        for ord_, n2 in cnt.most_common(3):
            p(f"      {n2:4d}  {''.join(ord_)}")

    report = "\n".join(lines)
    (OUT / "ordering_AG_report.txt").write_text(report + "\n")
    print(report)

    hdf = pd.DataFrame(heat)
    if len(hdf):
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        piv = hdf.pivot(index="slice", columns="metric", values="modal_frac")
        piv = piv[[m for _, m in METRICS]]
        im = ax.imshow(piv.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, fontsize=10)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=8)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center",
                        fontsize=9)
        fig.colorbar(im, ax=ax, label="mean modal-ordering fraction")
        ax.set_title("Ordering stability (A–G, no old E): fraction of slice "
                     "matching its modal ordering\n1.0 = perfectly stable")
        fig.tight_layout()
        fig.savefig(OUT / "fig_AG_ordering_heatmap.png", dpi=140)
        print("wrote", OUT / "fig_AG_ordering_heatmap.png")

    # bump charts
    axes_spec = [
        ("a", dict(u=0.005, v=0.005, r=0.0), False,
         "selection a  (u=v=0.005, r=0)"),
        ("u", dict(a=0.18, r=0.0), True,
         "mutation u=v  (a=0.18, r=0, sym)"),
        ("r", dict(a=0.18, u=0.005, v=0.005), False,
         "recombination r  (a=0.18, u=v=0.005)"),
    ]
    fig, axs = plt.subplots(len(METRICS), len(axes_spec),
                            figsize=(15.5, 10.5), squeeze=False)
    for i, (mcol, mlab) in enumerate(METRICS):
        for j, (axis, fixed, logx, albl) in enumerate(axes_spec):
            ax = axs[i][j]
            src = df[df.mut_kind == "sym"] if axis == "u" else df
            xvals, ranks = ranks_along(src, axis, mcol, fixed)
            for c in CANON:
                jit = (CANON.index(c) - 2.5) * 0.04
                ax.plot(xvals, np.array(ranks[c]) + jit, "-o", ms=4.5, lw=1.8,
                        color=CASE_COLORS[c], label=c if i == 0 and j == 0 else None)
            if logx:
                ax.set_xscale("log")
            ax.set_ylim(6.6, 0.4)
            ax.set_yticks([1, 2, 3, 4, 5, 6])
            if j == 0:
                ax.set_ylabel(f"{mlab}\nrank (1=highest)")
            if i == 0:
                ax.set_title(albl, fontsize=9)
            if i == len(METRICS) - 1:
                ax.set_xlabel(axis)
            ax.grid(True, alpha=0.3)
    axs[0][0].legend(fontsize=8, ncol=3, loc="lower left")
    fig.suptitle("Bump charts: A,B,C,D,F,G ranks along 1-D slices  "
                 "(F = iso-peak synergy; old E excluded)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "fig_AG_bumpcharts.png", dpi=140)
    print("wrote", OUT / "fig_AG_bumpcharts.png")

    # pairwise dominance consistency (how often case X > case Y)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (mcol, mlab) in zip(axes, METRICS):
        M = np.zeros((6, 6))
        pts = list(df.groupby(["u", "v", "r", "a"]))
        for _, sub in pts:
            vals = {c: sub[sub.case == c][mcol].mean() for c in CANON}
            for i, ci in enumerate(CANON):
                for j, cj in enumerate(CANON):
                    if ci == cj:
                        continue
                    M[i, j] += float(vals[ci] > vals[cj])
        M /= len(pts)
        im = ax.imshow(M, vmin=0, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.set_xticklabels(CANON); ax.set_yticklabels(CANON)
        ax.set_title(mlab)
        for i in range(6):
            for j in range(6):
                if i != j:
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=7)
        ax.set_xlabel("loses to →"); ax.set_ylabel("row beats col")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85,
                 label="P(row > col) over param points")
    fig.suptitle("Pairwise dominance consistency across parameter space "
                 "(A–G, diffusion)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "fig_AG_dominance.png", dpi=140)
    print("wrote", OUT / "fig_AG_dominance.png")


def main():
    csv = OUT / "sweep_AG.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        # verify F is iso-peak (lf11 should equal 2a)
        f = df[df.case == "F"].iloc[0]
        assert np.isclose(f.lf11, 2 * f.a), (f.lf11, f.a)
        print(f"loaded existing {csv} ({len(df)} rows); F iso-peak OK")
    else:
        df = sweep(30)
    report_and_figs(df)


if __name__ == "__main__":
    main()
