#!/usr/bin/env python3
r"""
Overlay path efficiency and peak eta vs KL_eq across recombination rates.

Reads eps12_metrics.csv from each run directory and writes comparison plots to
plots/eps12_delta_d_r_compare/.

Run (after sweeps exist):
  cd 2loci
  python eps12_eta_r_overlay.py
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS = [
    ("r=0", 0.0, SCRIPT_DIR / "plots" / "eps12_delta_d"),
    ("r=0.01", 0.01, SCRIPT_DIR / "plots" / "eps12_delta_d_r0p01"),
    ("r=0.1", 0.1, SCRIPT_DIR / "plots" / "eps12_delta_d_r0p1"),
    ("r=0.5", 0.5, SCRIPT_DIR / "plots" / "eps12_delta_d_r0p5"),
]
OUT_DIR = SCRIPT_DIR / "plots" / "eps12_delta_d_r_compare"


@dataclass
class Row:
    eps12: float
    kl_eq: float
    path_efficiency: float
    peak_eta: float
    t_90: float


def load_metrics(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open() as f:
        for rec in csv.DictReader(f):
            rows.append(Row(
                eps12=float(rec["eps12"]),
                kl_eq=float(rec["kl_eq"]),
                path_efficiency=float(rec["path_efficiency"]),
                peak_eta=float(rec["peak_eta"]),
                t_90=float(rec["t_90"]) if rec["t_90"] else float("nan"),
            ))
    return rows


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xr = np.argsort(np.argsort(x[mask]))
    yr = np.argsort(np.argsort(y[mask]))
    return float(np.corrcoef(xr, yr)[0, 1])


def _branch(rows: list[Row], branch: str) -> list[Row]:
    if branch == "neg":
        return [r for r in rows if r.eps12 < -1e-12]
    if branch == "pos":
        return [r for r in rows if r.eps12 > 1e-12]
    return rows


def plot_overlay(
    runs: list[tuple[str, float, list[Row]]],
    out_path: Path,
    *,
    branch: str,
    y_attr: str,
    ylabel: str,
    title: str,
    N: int,
    u: float,
    s1: float,
    s2: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    markers = ["o", "s", "^", "D", "v", "P"]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    summary: dict[str, dict[str, float]] = {}

    for i, (label, r_val, rows) in enumerate(runs):
        sub = _branch(rows, branch)
        if not sub:
            continue
        kl = np.array([r.kl_eq for r in sub])
        y = np.array([getattr(r, y_attr) for r in sub])
        order = np.argsort(kl)
        c = colors[i % len(colors)]
        mk = markers[i % len(markers)]
        ax.plot(
            kl[order], y[order], f"{mk}-", color=c, lw=1.8, ms=6,
            label=label, alpha=0.92,
        )
        summary[label] = {
            "r": r_val,
            "n": len(sub),
            "spearman_kl_eq": _spearman(kl, y),
            f"mean_{y_attr}": float(np.nanmean(y)),
        }

    ax.set_xlabel(r"$\mathrm{KL}_{\mathrm{eq}}(\psi^*\|\varphi)$ [bits]")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="best")
    fig.suptitle(
        rf"$N={N}$, $u=v={u:g}$, $s_1={s1:g}$, $s_2={s2:g}$",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)
    return summary


def plot_bar_summary(
    runs: list[tuple[str, float, list[Row]]],
    out_path: Path,
    *,
    branch: str,
    N: int,
    u: float,
    s1: float,
    s2: float,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [lab for lab, _, _ in runs]
    mean_eta = []
    mean_peak = []
    for _, _, rows in runs:
        sub = _branch(rows, branch)
        mean_eta.append(float(np.mean([r.path_efficiency for r in sub])))
        mean_peak.append(float(np.mean([r.peak_eta for r in sub])))

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, mean_eta, w, label=r"mean $\bar{\eta}$", color="C2", alpha=0.85)
    ax.bar(x + w / 2, mean_peak, w, label=r"mean peak $\eta$", color="C4", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("efficiency")
    branch_title = {"neg": r"$\varepsilon_{12}<0$", "pos": r"$\varepsilon_{12}>0$", "all": "all"}[branch]
    ax.set_title(rf"Mean efficiency by $r$ ({branch_title})", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        rf"$N={N}$, $u=v={u:g}$, $s_1={s1:g}$, $s_2={s2:g}$",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay eta vs KL_eq across r values")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    loaded: list[tuple[str, float, list[Row]]] = []
    meta = None
    for label, r_val, run_dir in DEFAULT_RUNS:
        csv_path = run_dir / "eps12_metrics.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing {csv_path} — run sweep for {label} first")
        loaded.append((label, r_val, load_metrics(csv_path)))
        settings_path = run_dir / "settings.json"
        if settings_path.is_file() and meta is None:
            meta = json.loads(settings_path.read_text())

    N = int(meta.get("N", 30)) if meta else 30
    u = float(meta.get("u", 0.01)) if meta else 0.01
    s1 = float(meta.get("s1", 0.001)) if meta else 0.001
    s2 = float(meta.get("s2", -0.001)) if meta else -0.001

    all_summary: dict = {"runs": [lab for lab, _, _ in loaded], "panels": {}}

    for branch in ("neg", "pos"):
        bt = r"$\varepsilon_{12}<0$" if branch == "neg" else r"$\varepsilon_{12}>0$"
        for y_attr, ylabel, stem in [
            ("path_efficiency", r"$\bar{\eta}=\sum_t \Delta D / \sum_t N C(t)$", "path_efficiency"),
            ("peak_eta", r"$\max_t \Delta D/(N C(t))$", "peak_eta"),
        ]:
            key = f"{stem}_{branch}"
            summ = plot_overlay(
                loaded,
                args.out_dir / f"compare_{stem}_vs_kl_eq_{branch}.png",
                branch=branch,
                y_attr=y_attr,
                ylabel=ylabel,
                title=rf"{ylabel} vs $\mathrm{{KL}}_{{\mathrm{{eq}}}}$ ({bt})",
                N=N, u=u, s1=s1, s2=s2,
            )
            all_summary["panels"][key] = summ
        plot_bar_summary(
            loaded, args.out_dir / f"compare_mean_efficiency_by_r_{branch}.png",
            branch=branch, N=N, u=u, s1=s1, s2=s2,
        )

    (args.out_dir / "overlay_summary.json").write_text(json.dumps(all_summary, indent=2))
    print(f"\nDone. Outputs in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
