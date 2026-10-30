#!/usr/bin/env python3
r"""Alternative displays of the ordering / ranking information from big_sweep.

(1) fig_bumpcharts.png : rank (1..5) of each case vs one varied axis, for the
    three metrics x three axes.  Line crossings = ordering swaps -> you literally
    see the (in)stability the heatmap summarised.
(2) fig_phase_AD.png   : 2D (a, u=v) maps of which case is higher for the pairs
    that actually contest a rank (A vs D on information; A vs D on mean-logfit),
    plus the outright information winner.  Shows *where* in parameter space swaps
    live.
(3) fig_ordering_bars.png : stacked bars of how the axis-sweep's orderings are
    distributed (how many distinct orderings, and their shares) per slice+metric.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptation_per_a import CASE_ORDER, CASE_COLORS, CASE_DEFS

SCRIPT_DIR = Path(__file__).resolve().parent
DATA = SCRIPT_DIR / "results" / "big_sweep_20260719" / "sweep_data.csv"
OUT = DATA.parent / "analysis"
METRICS = [("info_diff", "information"),
           ("mlf_diff", "mean log-fitness"),
           ("F_diff", "free fitness F")]


def ranks_along(df, axis, metric, fixed: dict):
    """Return (xvals, {case: [rank per xval]}) sweeping `axis` with `fixed` held."""
    sub = df.copy()
    for k, v in fixed.items():
        sub = sub[np.isclose(sub[k], v)]
    xvals = np.sort(sub[axis].unique())
    ranks = {c: [] for c in CASE_ORDER}
    for x in xvals:
        s = sub[np.isclose(sub[axis], x)]
        vals = {c: s[s.case == c][metric].mean() for c in CASE_ORDER}
        order = sorted(CASE_ORDER, key=lambda c: -vals[c])   # high->low
        for rank, c in enumerate(order, start=1):
            ranks[c].append(rank)
    return xvals, ranks


def fig_bumpcharts(df):
    # axis, fixed params, logx?, label
    axes_spec = [
        ("a", dict(u=0.005, v=0.005, r=0.0), False, "selection a  (u=v=0.005, r=0)"),
        ("u", dict(a=0.17, r=0.0), True, "mutation u=v  (a=0.17, r=0)"),
        ("r", dict(a=0.17, u=0.005, v=0.005), False, "recombination r  (a=0.17, u=v=0.005)"),
    ]
    fig, axs = plt.subplots(len(METRICS), len(axes_spec),
                            figsize=(15, 10), squeeze=False)
    for i, (mcol, mlab) in enumerate(METRICS):
        for j, (axis, fixed, logx, albl) in enumerate(axes_spec):
            ax = axs[i][j]
            # sym only for u sweep
            src = df[df.mut_kind == "sym"] if axis == "u" else df
            xvals, ranks = ranks_along(src, axis, mcol, fixed)
            for c in CASE_ORDER:
                jit = (CASE_ORDER.index(c) - 2) * 0.05     # tiny offset to see overlaps
                ax.plot(xvals, np.array(ranks[c]) + jit, "-o", ms=4, lw=1.8,
                        color=CASE_COLORS[c], label=c, alpha=0.9)
            if logx:
                ax.set_xscale("log")
            ax.set_ylim(5.5, 0.5)                            # rank 1 on top
            ax.set_yticks([1, 2, 3, 4, 5])
            if j == 0:
                ax.set_ylabel(f"{mlab}\nrank (1=highest)")
            if i == 0:
                ax.set_title(albl, fontsize=10)
            if i == len(METRICS) - 1:
                ax.set_xlabel(axis)
            ax.grid(True, alpha=0.3)
    handles = [plt.Line2D([0], [0], color=CASE_COLORS[c], marker="o", lw=1.8,
                          label=f"{c}: {CASE_DEFS[c][0]}") for c in CASE_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Rank of each case along one axis (diffusion). Crossings = ordering swaps",
                 y=0.965, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT / "fig_bumpcharts.png", dpi=150)
    plt.close(fig)


def fig_phase(df):
    """2D (a, u=v) winner/contest maps at r=0, symmetric mutation."""
    sub = df[(df.mut_kind == "sym") & np.isclose(df.r, 0.0)]
    avals = np.sort(sub.a.unique())
    uvals = np.sort(sub.u.unique())

    def grid_of(func):
        g = np.full((len(uvals), len(avals)), np.nan)
        for iu, u in enumerate(uvals):
            for ia, a in enumerate(avals):
                s = sub[np.isclose(sub.a, a) & np.isclose(sub.u, u)]
                if len(s) == 5:
                    g[iu, ia] = func(s)
        return g

    def winner_info(s):
        vals = {c: s[s.case == c]["info_diff"].mean() for c in CASE_ORDER}
        return CASE_ORDER.index(max(vals, key=vals.get))

    def AmD_info(s):
        return (s[s.case == "A"]["info_diff"].mean()
                - s[s.case == "D"]["info_diff"].mean())

    def AmD_mlf(s):
        return (s[s.case == "A"]["mlf_diff"].mean()
                - s[s.case == "D"]["mlf_diff"].mean())

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    ext = [avals.min(), avals.max(), 0, len(uvals)]

    # panel 1: outright information winner
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap([CASE_COLORS[c] for c in CASE_ORDER])
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)
    im0 = axs[0].imshow(grid_of(winner_info), aspect="auto", origin="lower",
                        extent=ext, cmap=cmap, norm=norm)
    axs[0].set_title("information: outright winner")
    cb = fig.colorbar(im0, ax=axs[0], ticks=range(5))
    cb.ax.set_yticklabels(CASE_ORDER)

    # panels 2,3: signed contests (diverging)
    for ax, g, title in [
        (axs[1], grid_of(AmD_info), "info(A) - info(D)\n(red: A>D, blue: D>A)"),
        (axs[2], grid_of(AmD_mlf), "mlf(A) - mlf(D)\n(red: A>D, blue: D>A)")]:
        vmax = np.nanmax(np.abs(g))
        im = ax.imshow(g, aspect="auto", origin="lower", extent=ext,
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        cs = ax.contour(np.linspace(ext[0], ext[1], g.shape[1]),
                        np.arange(len(uvals)), g, levels=[0], colors="k", linewidths=1.5)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axs:
        ax.set_xlabel("selection a")
        ax.set_yticks(np.arange(len(uvals)) + 0.0)
        ax.set_yticklabels([f"{u:g}" for u in uvals], fontsize=7)
        axs[0].set_ylabel("u = v")
    axs[0].set_ylabel("u = v")
    fig.suptitle("Where in (a, u=v) space cases contest a rank  (r=0, diffusion)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "fig_phase_AD.png", dpi=150)
    plt.close(fig)


def fig_ordering_bars(df):
    """Stacked bars: distribution of distinct orderings per slice family+metric."""
    slices = [
        ("vary u=v", ["a", "r"], dict(mut_kind="sym")),
        ("vary u,v asym", ["a", "r"], dict(mut_kind="asym")),
        ("vary r", ["a", "u", "v"], {}),
        ("vary a", ["u", "v", "r"], {}),
    ]
    fig, axs = plt.subplots(len(METRICS), 1, figsize=(11, 11))
    for mi, (mcol, mlab) in enumerate(METRICS):
        ax = axs[mi]
        ypos = []
        ylabels = []
        y = 0
        for slab, fixed, restrict in slices:
            sub = df.copy()
            for k, v in restrict.items():
                sub = sub[sub[k] == v]
            # dominant ordering per group
            doms = []
            for _, g in sub.groupby(fixed):
                axis_cols = fixed + (["u", "v"])
                pts = g.groupby([c for c in g.columns
                                 if c in (fixed + ["u", "v", "r", "a"])])
                # ordering per point, dominant per group
                oc = Counter()
                for _, s in g.groupby([c for c in ["a", "u", "v", "r"]
                                       if c not in fixed]):
                    if s.case.nunique() < 5:
                        continue
                    vals = {c: s[s.case == c][mcol].mean() for c in CASE_ORDER}
                    oc[tuple(sorted(CASE_ORDER, key=lambda c: -vals[c]))] += 1
                if oc:
                    doms.append(oc.most_common(1)[0][0])
            cnt = Counter(doms)
            total = sum(cnt.values())
            left = 0
            # color distinct orderings by a stable palette
            palette = plt.cm.tab20(np.linspace(0, 1, 20))
            for k, (o, n) in enumerate(cnt.most_common()):
                ax.barh(y, n / total, left=left, height=0.7,
                        color=palette[k % 20], edgecolor="white")
                if n / total > 0.08:
                    ax.text(left + n / total / 2, y, " > ".join(o),
                            ha="center", va="center", fontsize=7)
                left += n / total
            ypos.append(y); ylabels.append(f"{slab}\n({total} groups)")
            y += 1
        ax.set_yticks(ypos); ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xlim(0, 1); ax.set_xlabel("share of groups")
        ax.set_title(f"dominant orderings — {mlab}", fontsize=11)
    fig.suptitle("Distribution of dominant A..E orderings across groups (diffusion)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "fig_ordering_bars.png", dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA)
    OUT.mkdir(exist_ok=True)
    fig_bumpcharts(df)
    fig_phase(df)
    fig_ordering_bars(df)
    print("wrote fig_bumpcharts.png, fig_phase_AD.png, fig_ordering_bars.png to", OUT)


if __name__ == "__main__":
    main()
