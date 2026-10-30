#!/usr/bin/env python3
r"""Slice-based ordering-consistency analysis of the big sweep.

For a chosen metric (information / mean-log-fitness / free fitness) we ask:
if we FIX some parameters and VARY one axis, how stable is the A..E ordering?

Slice families (each = "fix these, vary that"):
  vary u=v (sym)   fixing (a, r)
  vary (u,v) asym  fixing (a, r)
  vary r           fixing (a, u, v)
  vary a           fixing (u, v, r)

Per slice-group we record: the full ordering at every point on the varied axis,
the modal (most common) ordering, the fraction of points matching it
(1.0 = perfectly stable), and the number of distinct orderings seen.
Aggregated over all groups -> a consistency table + heatmap.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from adaptation_per_a import CASE_ORDER

SCRIPT_DIR = Path(__file__).resolve().parent
METRICS = [("info_diff", "information"),
           ("mlf_diff", "mean log-fitness"),
           ("F_diff", "free fitness F")]
# fixed-cols, vary-col, restrict (col->value) ; label
SLICES = [
    ("vary u=v (sym), fix a,r", ["a", "r"], "u",
     dict(mut_kind="sym")),
    ("vary u,v (asym), fix a,r", ["a", "r"], "uv",
     dict(mut_kind="asym")),
    ("vary r, fix a,u,v", ["a", "u", "v"], "r", {}),
    ("vary a, fix u,v,r", ["u", "v", "r"], "a", {}),
]


def ordering(sub: pd.DataFrame, metric: str) -> tuple:
    """Cases sorted high->low by metric within one point (unique param combo)."""
    vals = {c: sub[sub.case == c][metric].mean() for c in CASE_ORDER}
    return tuple(sorted(CASE_ORDER, key=lambda c: -vals[c]))


def analyse_slice(df: pd.DataFrame, fixed: list[str], vary: str,
                  metric: str) -> dict:
    """Return per-group consistency stats for one slice family + metric."""
    # each "point" on the varied axis is a unique combo of (fixed + vary axis)
    axis_cols = fixed + (["u", "v"] if vary == "uv" else [vary])
    modal_fracs, n_distinct, perfect = [], [], 0
    modal_orders = Counter()
    n_groups = 0
    for _, g in df.groupby(fixed):
        pts = g.groupby(axis_cols)
        if pts.ngroups < 3:                       # need a real axis to vary
            continue
        orders = [ordering(sub, metric) for _, sub in pts]
        cnt = Counter(orders)
        top, topn = cnt.most_common(1)[0]
        modal_fracs.append(topn / len(orders))
        n_distinct.append(len(cnt))
        perfect += int(len(cnt) == 1)
        modal_orders[top] += 1
        n_groups += 1
    return dict(
        mean_modal_frac=float(np.mean(modal_fracs)),
        perfect_frac=perfect / n_groups,
        mean_n_distinct=float(np.mean(n_distinct)),
        n_groups=n_groups,
        top_ordering=modal_orders.most_common(1)[0][0],
        modal_orders=modal_orders,
    )


def fmt(o: tuple) -> str:
    return " > ".join(o)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=SCRIPT_DIR / "results" / "big_sweep_20260719" /
                    "sweep_data.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.data)
    out = args.data.parent / "analysis"
    out.mkdir(exist_ok=True)

    lines = [f"Ordering-consistency by slice  ({len(df)} rows)\n"]
    # consistency grid: rows = slice family, cols = metric ; value = mean modal frac
    grid = np.zeros((len(SLICES), len(METRICS)))
    perfect_grid = np.zeros_like(grid)
    for si, (slab, fixed, vary, restrict) in enumerate(SLICES):
        sub = df
        for k, v in restrict.items():
            sub = sub[sub[k] == v]
        lines.append("=" * 78)
        lines.append(f"### SLICE: {slab}   (n_rows={len(sub)})")
        for mi, (mcol, mlab) in enumerate(METRICS):
            r = analyse_slice(sub, fixed, vary, mcol)
            grid[si, mi] = r["mean_modal_frac"]
            perfect_grid[si, mi] = r["perfect_frac"]
            lines.append(
                f"  [{mlab:16s}] mean modal-frac={r['mean_modal_frac']:.3f}  "
                f"perfectly-stable groups={r['perfect_frac']*100:4.0f}%  "
                f"mean #distinct orderings={r['mean_n_distinct']:.2f}  "
                f"(over {r['n_groups']} groups)")
            lines.append(f"        dominant ordering: {fmt(r['top_ordering'])}")
            # show the 3 commonest orderings for this slice+metric
            for o, n in r["modal_orders"].most_common(3):
                lines.append(f"          {fmt(o):28s}  dominant in {n:3d} groups")
        lines.append("")

    report = "\n".join(lines)
    (out / "ordering_slices_report.txt").write_text(report)
    print(report)

    # heatmap
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, data, title in [
            (axes[0], grid, "mean modal-ordering fraction\n(1 = ordering never changes along axis)"),
            (axes[1], perfect_grid, "fraction of groups perfectly stable\n(single ordering along whole axis)")]:
        im = ax.imshow(data, cmap="RdYlGn", vmin=0.3, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(METRICS)))
        ax.set_xticklabels([m[1] for m in METRICS], rotation=20, ha="right")
        ax.set_yticks(range(len(SLICES)))
        ax.set_yticklabels([s[0] for s in SLICES])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                        fontsize=10)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Ordering stability of A..E along each parameter axis (diffusion)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out / "fig_ordering_consistency.png", dpi=150)
    plt.close(fig)
    print(f"\n[wrote {out/'ordering_slices_report.txt'}]")
    print(f"[wrote {out/'fig_ordering_consistency.png'}]")


if __name__ == "__main__":
    main()
