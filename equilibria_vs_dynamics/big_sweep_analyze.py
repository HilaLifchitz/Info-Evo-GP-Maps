#!/usr/bin/env python3
r"""
Analyze + plot the big_sweep.py output (works on partial or complete CSV).

Produces, in <out> (default: alongside the data file):
  ordering_report.txt   text report: modal orderings, mean ranks,
                        diffusion-vs-chain ordering match rates, rank correlations
  fig_meanrank.png      heatmap: mean rank of each case per metric
  fig_metric_vs_a.png   info / free-fitness / mean-logfit / concentration vs a,
                        faceted by r (representative symmetric u)
  fig_diff_vs_chain.png scatter F_diff vs F_chain and info_diff vs info_chain
  fig_match_vs_r.png    diffusion==chain ordering match-rate vs r
  fig_u_effect.png      info & concentration vs u at fixed a, per case (r=0)

Run:
  conda activate miso
  cd equilibria_vs_dynamics
  python big_sweep_analyze.py --data results/big_sweep_20260719/sweep_data.csv
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

CASE_ORDER = ["A", "B", "C", "D", "E"]
CASE_COLORS = {"A": "#c44e52", "B": "#4c72b0", "C": "#55a868", "D": "#8172b3",
               "E": "#dd8452"}
CASE_NAME = {"A": "sign", "B": "additive", "C": "reciprocal", "D": "neutral",
             "E": "multi-peak"}
GROUP = ["u", "v", "r", "a"]

# metrics to rank (higher = "better"); (column, pretty label)
METRICS = [
    ("info_diff", "information (diffusion)"),
    ("info_chain", "information (chain)"),
    ("mlf_diff", "mean log-fit (diffusion)"),
    ("mlf_chain", "mean log-fit (chain)"),
    ("F_diff", "free fitness F (diffusion)"),
    ("F_chain", "free fitness F (chain)"),
    ("G_diff", "free fitness G (diffusion)"),
    ("Fboltz", "free fitness logZ/N (Boltzmann)"),
    ("domf_diff", "concentration domf (diffusion)"),
    ("domf_chain", "concentration domf (chain)"),
]


def ordering(sub: pd.DataFrame, metric: str) -> tuple:
    return tuple(sub.sort_values(metric, ascending=False)["case"].tolist())


def rank_map(sub: pd.DataFrame, metric: str) -> dict:
    order = ordering(sub, metric)
    return {c: i + 1 for i, c in enumerate(order)}


def analyze(df: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    groups = list(df.groupby(GROUP))
    n_groups = len(groups)
    L = [f"big_sweep analysis  ({len(df)} rows, {n_groups} (u,v,r,a) groups)",
         f"N values: {sorted(df['N'].unique())}",
         f"r values: {sorted(df['r'].unique())}",
         f"a values: {sorted(df['a'].unique())}",
         f"symmetric u=v: {sorted(df.loc[df.mut_kind=='sym','u'].unique())}",
         "=" * 78]

    # ---- modal orderings + mean rank per metric ----
    meanrank = {m: {c: [] for c in CASE_ORDER} for m, _ in METRICS}
    for metric, label in METRICS:
        counts = Counter()
        for _, sub in groups:
            if sub[metric].isna().any():
                continue
            counts[ordering(sub, metric)] += 1
            rm = rank_map(sub, metric)
            for c in CASE_ORDER:
                meanrank[metric][c].append(rm[c])
        top = counts.most_common(3)
        L.append(f"\n### {label}  [{metric}]")
        for order, n in top:
            L.append(f"    {' > '.join(order)}   in {n}/{n_groups} groups "
                     f"({100*n/n_groups:.0f}%)")
        mr = {c: float(np.mean(meanrank[metric][c])) for c in CASE_ORDER}
        L.append("    mean rank (1=highest): " +
                 "  ".join(f"{c}={mr[c]:.2f}" for c in CASE_ORDER))

    # ---- diffusion vs chain ordering match ----
    L.append("\n" + "=" * 78)
    L.append("Diffusion vs chain ORDERING match (per (u,v,r,a) group):")
    for base in ["info", "mlf", "F", "G"]:
        cd, cc = f"{base}_diff", f"{base}_chain"
        if cd not in df.columns or cc not in df.columns:
            continue
        match = tot = 0
        by_r = {}
        for (u, v, r, a), sub in groups:
            if sub[cd].isna().any() or sub[cc].isna().any():
                continue
            ok = ordering(sub, cd) == ordering(sub, cc)
            match += ok; tot += 1
            by_r.setdefault(r, [0, 0])
            by_r[r][0] += ok; by_r[r][1] += 1
        if tot:
            L.append(f"  {base}: {match}/{tot} groups identical "
                     f"({100*match/tot:.1f}%)")
            L.append("      by r: " + "  ".join(
                f"r={r:g}:{100*m/n:.0f}%" for r, (m, n) in sorted(by_r.items())))

    # ---- rank correlation diffusion vs chain (row-wise) ----
    L.append("\nRank correlation diffusion vs chain (Spearman, all rows):")
    for base in ["info", "mlf", "F", "G"]:
        cd, cc = f"{base}_diff", f"{base}_chain"
        if cd in df.columns and cc in df.columns:
            m = df[[cd, cc]].dropna()
            if len(m) > 2:
                rho, _ = spearmanr(m[cd], m[cc])
                L.append(f"  {base}: rho={rho:.4f}  (n={len(m)})")

    (out / "ordering_report.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n[wrote {out/'ordering_report.txt'}]")

    _fig_meanrank(meanrank, out)
    _fig_metric_vs_a(df, out)
    _fig_diff_vs_chain(df, out)
    _fig_match_vs_r(df, groups, out)
    _fig_u_effect(df, out)


def _fig_meanrank(meanrank, out):
    metrics = [m for m, _ in METRICS]
    labels = [lab for _, lab in METRICS]
    M = np.array([[np.mean(meanrank[m][c]) if meanrank[m][c] else np.nan
                   for c in CASE_ORDER] for m in metrics])
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=1, vmax=5, aspect="auto")
    ax.set_xticks(range(len(CASE_ORDER)))
    ax.set_xticklabels([f"{c}\n{CASE_NAME[c]}" for c in CASE_ORDER])
    ax.set_yticks(range(len(metrics))); ax.set_yticklabels(labels, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean rank (1=highest)")
    ax.set_title("Mean rank of each case per metric (averaged over all u,v,r,a)")
    fig.tight_layout(); fig.savefig(out / "fig_meanrank.png", dpi=150)
    plt.close(fig)


def _rep_u(df):
    syms = sorted(df.loc[df.mut_kind == "sym", "u"].unique())
    # pick a middle symmetric u
    return syms[len(syms) // 2] if syms else df["u"].iloc[0]


def _fig_metric_vs_a(df, out):
    u = _rep_u(df)
    d = df[(df.mut_kind == "sym") & (np.isclose(df.u, u))]
    rs = sorted(d["r"].unique())
    rows = [("info_diff", "information [bits] (diffusion)"),
            ("F_diff", "free fitness F [nats] (diffusion)"),
            ("mlf_diff", "mean log-fitness (diffusion)"),
            ("domf_diff", "dominant mean freq (diffusion)")]
    fig, axes = plt.subplots(len(rows), len(rs),
                             figsize=(3.4 * len(rs), 3.0 * len(rows)),
                             squeeze=False, sharex=True)
    for j, r in enumerate(rs):
        dr = d[np.isclose(d.r, r)]
        for i, (col, ylab) in enumerate(rows):
            ax = axes[i][j]
            for c in CASE_ORDER:
                dc = dr[dr.case == c].sort_values("a")
                ax.plot(dc["a"], dc[col], "-o", ms=3, lw=1.6,
                        color=CASE_COLORS[c], label=c)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(f"r={r:g}", fontsize=10)
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8)
            if i == len(rows) - 1:
                ax.set_xlabel("a")
    axes[0][0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Equilibrium metrics vs selection strength a  "
                 f"(symmetric u=v={u:g}, N={df['N'].iloc[0]})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "fig_metric_vs_a.png", dpi=150)
    plt.close(fig)


def _fig_diff_vs_chain(df, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, base, lab in [(axes[0], "F", "free fitness F [nats]"),
                          (axes[1], "info", "information [bits]")]:
        cd, cc = f"{base}_diff", f"{base}_chain"
        for c in CASE_ORDER:
            dc = df[df.case == c]
            ax.scatter(dc[cd], dc[cc], s=6, alpha=0.35, color=CASE_COLORS[c],
                       label=c, edgecolors="none")
        lo = float(np.nanmin([df[cd].min(), df[cc].min()]))
        hi = float(np.nanmax([df[cd].max(), df[cc].max()]))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
        rho, _ = spearmanr(df[[cd, cc]].dropna().values)
        ax.set_xlabel(f"{lab} — diffusion")
        ax.set_ylabel(f"{lab} — chain")
        ax.set_title(f"{lab}   (Spearman rho={rho:.3f})")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_diff_vs_chain.png", dpi=150)
    plt.close(fig)


def _fig_match_vs_r(df, groups, out):
    bases = ["info", "F", "mlf"]
    rs = sorted(df["r"].unique())
    rate = {b: {r: [0, 0] for r in rs} for b in bases}
    for (u, v, r, a), sub in groups:
        for b in bases:
            cd, cc = f"{b}_diff", f"{b}_chain"
            if sub[cd].isna().any() or sub[cc].isna().any():
                continue
            rate[b][r][0] += ordering(sub, cd) == ordering(sub, cc)
            rate[b][r][1] += 1
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.25
    x = np.arange(len(rs))
    for k, b in enumerate(bases):
        ys = [100 * rate[b][r][0] / rate[b][r][1] if rate[b][r][1] else np.nan
              for r in rs]
        ax.bar(x + (k - 1) * width, ys, width, label=b)
    ax.set_xticks(x); ax.set_xticklabels([f"r={r:g}" for r in rs])
    ax.set_ylabel("% of (u,v,a) groups with identical diffusion/chain ordering")
    ax.set_ylim(0, 105); ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    ax.set_title("Diffusion-vs-chain ordering agreement vs recombination r")
    fig.tight_layout(); fig.savefig(out / "fig_match_vs_r.png", dpi=150)
    plt.close(fig)


def _fig_u_effect(df, out):
    d = df[(df.mut_kind == "sym") & np.isclose(df.r, 0.0)]
    avals = sorted(d["a"].unique())
    a_rep = avals[len(avals) // 2]
    da = d[np.isclose(d.a, a_rep)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, ylab in [(axes[0], "info_diff", "information [bits] (diffusion)"),
                          (axes[1], "domf_diff", "dominant mean freq (diffusion)")]:
        for c in CASE_ORDER:
            dc = da[da.case == c].sort_values("u")
            ax.plot(dc["u"], dc[col], "-o", ms=4, color=CASE_COLORS[c], label=c)
        ax.set_xscale("log"); ax.set_xlabel("u=v"); ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(f"Mutation-rate effect at a={a_rep:g}, r=0, N={df['N'].iloc[0]}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / "fig_u_effect.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    df = pd.read_csv(args.data)
    out = args.out or args.data.parent / "analysis"
    print(f"loaded {len(df)} rows from {args.data}")
    analyze(df, out)
    print(f"\nFigures + report in {out}")


if __name__ == "__main__":
    main()
