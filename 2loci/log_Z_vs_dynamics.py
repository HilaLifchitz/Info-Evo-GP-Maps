#!/usr/bin/env python3
r"""
Compare cond-α log Z/N to dynamics / information metrics from the eps12 sweep.

Loads log Z from log_Z_eps12_sweep.csv (or recomputes).
Runs (or loads) per-eps12 trajectories and computes:
  KL_eq, d_final, path_efficiency (bar η), peak η,
  F4 slopes, F4+logZ/N slopes, F4 at start and near equilibrium.

Run:
  cd 2loci
  python log_Z_vs_dynamics.py --config eps12_delta_d_config.yaml
  python log_Z_vs_dynamics.py --metrics-csv plots/eps12_delta_d/eps12_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eps12_delta_d_sweep import (  # noqa: E402
    Eps12Metrics,
    analyze_run,
    run_one_eps12,
    _eps12_grid,
    _make_cfg,
)
from f2b_reference import log_Z_cond_alpha  # noqa: E402
from compare_sigma_dynamics import cond_alpha_reference_pmf  # noqa: E402
from simplex import compute_phi, fitness_exp, load_config, make_simplex_grid, mutation_matrix_asymmetric, params_from_config  # noqa: E402

OUT = SCRIPT_DIR / "plots" / "f2b"

BRANCH_TITLES = {
    "pos": r"$\varepsilon_{12} > 0$",
    "neg": r"$\varepsilon_{12} < 0$",
}

CORR_KEYS = [
    ("kl_eq_bits", r"$KL_{\mathrm{eq}}$ [bits]"),
    ("d_final_bits", r"$D(t_{\mathrm{final}})$ [bits]"),
    ("path_efficiency", r"$\bar\eta$"),
    ("peak_eta", r"peak $\eta$"),
    ("f4_slope_early", r"slope $F_4$ (early)"),
    ("F4_plus_logZ_at_phi0", r"$F_4+\log Z/N$ at $\varphi$"),
]


def _filter_branch_rows(rows: list[dict], branch: str) -> list[dict]:
    if branch == "pos":
        return [r for r in rows if r["eps12"] > 1e-12]
    if branch == "neg":
        return [r for r in rows if r["eps12"] < -1e-12]
    raise ValueError(f"branch must be 'pos' or 'neg', got {branch!r}")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok]))
    ry = np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1])


def _slope(t: np.ndarray, y: np.ndarray, t_lo: float, t_hi: float) -> float:
    m = (t >= t_lo) & (t <= t_hi) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    a, b = np.polyfit(t[m], y[m], 1)
    return float(a)


def _load_log_z_csv(path: Path) -> dict[float, dict]:
    by_eps: dict[float, dict] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            eps = float(row["eps12"])
            by_eps[eps] = row
    return by_eps


def _load_metrics_csv(path: Path) -> list[Eps12Metrics]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append(Eps12Metrics(
                eps12=float(row["eps12"]),
                kl_eq=float(row["kl_eq"]),
                f1_eq=float(row["f1_eq"]),
                f2_eq=float(row["f2_eq"]),
                f3_eq=float(row["f3_eq"]),
                wstar_eq=float(row["wstar_eq"]),
                path_efficiency=float(row["path_efficiency"]),
                peak_eta=float(row["peak_eta"]),
                t_peak_eta=float(row["t_peak_eta"]),
                t_half=float(row["t_half"]),
                t_90=float(row["t_90"]),
                t_95=float(row["t_95"]),
                t_wstar_90=float(row["t_wstar_90"]),
                d_final=float(row["d_final"]),
                frac_equilibrium=float(row["frac_equilibrium"]),
                sum_delta_d=float(row["sum_delta_d"]),
                sum_nc=float(row["sum_nc"]),
                frac_info_driven_steps=float(row["frac_info_driven_steps"]),
                frac_info_driven_after_wstar_90=float(row["frac_info_driven_after_wstar_90"]),
                frac_f2_down_wstar_flat=float(row["frac_f2_down_wstar_flat"]),
            ))
    return rows


def _run_dynamics(cfg: dict, *, pergen_max_t: int | None = None) -> tuple[list, list[Eps12Metrics]]:
    N = int(cfg["N"])
    r = float(cfg.get("r", 0.0))
    u = float(cfg["u"][0]) if isinstance(cfg.get("u"), list) else float(cfg.get("u", 0.01))
    v = float(cfg["v"][0]) if isinstance(cfg.get("v"), list) else float(cfg.get("v", u))
    s1 = float(cfg.get("s1", 0.001))
    s2 = float(cfg.get("s2", -0.001))
    max_t = int(pergen_max_t if pergen_max_t is not None else cfg.get("pergen_max_t", 500))
    eps_grid = _eps12_grid(cfg)

    M_mut = mutation_matrix_asymmetric(np.array([u, u]), np.array([v, v]))
    params0 = params_from_config(_make_cfg(N=N, r=r, u=u, v=v, s1=s1, s2=s2, eps12=0.0), SCRIPT_DIR)
    phi, _, fracs, _, _ = compute_phi(params0, M_mut)
    q_cond = cond_alpha_reference_pmf(fracs, M_mut, N)

    runs = []
    metrics = []
    for i, eps in enumerate(eps_grid, 1):
        print(f"  dynamics [{i}/{len(eps_grid)}] eps12={eps:+.3f}", flush=True)
        run = run_one_eps12(
            float(eps), N=N, r=r, u=u, v=v, s1=s1, s2=s2,
            phi=phi, fracs=fracs, M_mut=M_mut, q_cond=q_cond,
            pergen_max_t=max_t, verbose=False,
        )
        runs.append(run)
        metrics.append(analyze_run(run, N))
    return runs, metrics


def _merge_rows(
    cfg: dict,
    log_z_by_eps: dict[float, dict],
    runs: list,
    metrics: list[Eps12Metrics],
) -> list[dict]:
    N = int(cfg["N"])
    fracs = make_simplex_grid(N)
    u = float(cfg["u"][0]) if isinstance(cfg.get("u"), list) else float(cfg.get("u", 0.01))
    v = float(cfg["v"][0]) if isinstance(cfg.get("v"), list) else float(cfg.get("v", u))
    M_mut = mutation_matrix_asymmetric(np.array([u, u]), np.array([v, v]))
    omega0 = fitness_exp(np.zeros((2, 2)))
    z0 = log_Z_cond_alpha(fracs, omega0, M_mut, N, mode="interior")
    log_z0 = z0["log_Z_over_N"]

    rows = []
    run_by_eps = {r.eps12: r for r in runs}
    for m in metrics:
        lz = log_z_by_eps.get(m.eps12, {})
        log_z_n = float(lz.get("log_Z_over_N_interior", np.nan))
        run = run_by_eps[m.eps12]
        t = run.pergen_times
        f4 = run.pergen_f4
        f4_plus = f4 + log_z_n
        f4_0 = float(f4[0])
        f4_end = float(f4[-1])
        rows.append({
            "eps12": m.eps12,
            "log_Z_over_N": log_z_n,
            "log_Z0_over_N": log_z0,
            "delta_log_Z_over_N": log_z_n - log_z0,
            "kl_eq_bits": m.kl_eq,
            "kl_eq_nats": m.kl_eq * np.log(2),
            "d_final_bits": m.d_final,
            "path_efficiency": m.path_efficiency,
            "peak_eta": m.peak_eta,
            "t_peak_eta": m.t_peak_eta,
            "t_90": m.t_90,
            "f2_eq": m.f2_eq,
            "f4_eq": f4_end,
            "f4_at_phi0": f4_0,
            "f4_rise": f4_end - f4_0,
            "F4_plus_logZ_at_phi0": f4_0 + log_z_n,
            "F4_plus_logZ_at_end": f4_end + log_z_n,
            "f4_slope_early": _slope(t, f4, 0, 50),
            "f4_slope_late": _slope(t, f4, max(0, t[-1] - 100), t[-1]),
            "f4_plus_logZ_slope_early": _slope(t, f4_plus, 0, 50),
            "f4_plus_logZ_slope_late": _slope(t, f4_plus, max(0, t[-1] - 100), t[-1]),
            "frac_equilibrium": m.frac_equilibrium,
        })
    return rows


def _load_merged_csv(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rec: dict = {}
            for k, v in row.items():
                if not v:
                    rec[k] = float("nan")
                else:
                    rec[k] = float(v)
            rows.append(rec)
    return rows


def _branch_correlations(rows: list[dict]) -> dict[str, float]:
    x = np.array([r["log_Z_over_N"] for r in rows])
    corrs = {}
    for key, _ in CORR_KEYS:
        y = np.array([r[key] for r in rows])
        corrs[key] = _spearman(x, y)
    return corrs


def _plot_branch_correlations(
    rows_all: list[dict],
    out_dir: Path,
    *,
    N: int,
    s1: float,
    s2: float,
) -> dict:
    """Pos/neg branch scatter panels and correlation summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_corrs: dict = {"all": _branch_correlations(rows_all)}

    fig, axes = plt.subplots(2, len(CORR_KEYS), figsize=(3.2 * len(CORR_KEYS), 6.5), sharey="row")
    if len(CORR_KEYS) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for j, (key, ylab) in enumerate(CORR_KEYS):
        for i, (branch, color) in enumerate((("neg", "C0"), ("pos", "C3"))):
            sub = _filter_branch_rows(rows_all, branch)
            all_corrs[branch] = _branch_correlations(sub)
            ax = axes[i, j]
            x = np.array([r["log_Z_over_N"] for r in sub])
            y = np.array([r[key] for r in sub])
            o = np.argsort(x)
            ax.plot(x[o], y[o], "-", color=color, lw=1.0, alpha=0.5)
            ax.scatter(x, y, c=color, s=55, edgecolors="white", linewidths=0.4)
            rho = all_corrs[branch][key]
            ax.set_title(rf"{BRANCH_TITLES[branch]}  $\rho_s={rho:+.2f}$", fontsize=8)
            if i == 1:
                ax.set_xlabel(r"$\log Z/N$")
            if j == 0:
                ax.set_ylabel(ylab, fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        rf"$\log Z/N$ correlations by epistasis sign ($N={N}$, $s_1={s1:g}$, $s_2={s2:g}$)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out_dir / "log_Z_vs_dynamics_by_branch.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # bar chart: all vs pos vs neg
    fig, ax = plt.subplots(figsize=(10, 4.5))
    labels = [k for k, _ in CORR_KEYS]
    xpos = np.arange(len(labels))
    w = 0.25
    for i, (name, col) in enumerate((("all", "0.4"), ("neg", "C0"), ("pos", "C3"))):
        ys = [all_corrs[name].get(k, float("nan")) for k in labels]
        ax.bar(xpos + (i - 1) * w, ys, width=w, label=name, color=col if col != "0.4" else col)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels([lab.replace("$", "") for _, lab in CORR_KEYS], rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(r"Spearman $\rho$ with $\log Z/N$")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_title("Correlation strength: all vs negative vs positive $\\varepsilon_{12}$")
    fig.tight_layout()
    p2 = out_dir / "log_Z_vs_dynamics_branch_rho_bars.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return all_corrs


def _plot_vs_eps12(
    rows: list[dict],
    out_dir: Path,
    *,
    N: int,
    s1: float,
    s2: float,
) -> None:
    """All key metrics vs epsilon12 on shared x-axis (multi-panel)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = np.array([r["eps12"] for r in rows])
    o = np.argsort(eps)
    eps = eps[o]
    kl = np.array([rows[i]["kl_eq_bits"] for i in o])
    eta_bar = np.array([rows[i]["path_efficiency"] for i in o])
    eta_peak = np.array([rows[i]["peak_eta"] for i in o])
    log_z = np.array([rows[i]["log_Z_over_N"] for i in o])
    log_z0 = rows[0]["log_Z0_over_N"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    def _branch_vline(ax):
        ax.axvline(0, color="0.35", ls=":", lw=1.0, zorder=0)
        ax.axvspan(eps.min(), 0, alpha=0.04, color="C0", zorder=0)
        ax.axvspan(0, eps.max(), alpha=0.04, color="C3", zorder=0)

    ax = axes[0]
    ax.plot(eps, kl, "o-", color="C0", lw=1.8, ms=5, label=r"$KL_{\mathrm{eq}}=D(\psi^*\|\varphi^*)$ [bits]")
    ax.set_ylabel(r"information [bits]")
    ax.legend(loc="upper center", fontsize=9)
    ax.grid(True, alpha=0.3)
    _branch_vline(ax)
    ax.set_title(rf"Equilibrium & path metrics vs $\varepsilon_{{12}}$ ($N={N}$, $s_1={s1:g}$, $s_2={s2:g}$)")

    ax = axes[1]
    ax.plot(eps, eta_bar, "o-", color="C2", lw=1.8, ms=5, label=r"$\bar\eta = \sum \Delta D / \sum NC$")
    ax.set_ylabel(r"$\bar\eta$")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    _branch_vline(ax)

    ax = axes[2]
    ax.plot(eps, eta_peak, "s-", color="C3", lw=1.8, ms=5, label=r"peak $\eta = \max \Delta D/(NC)$")
    ax.set_ylabel(r"peak $\eta$")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    _branch_vline(ax)

    ax = axes[3]
    ax.plot(eps, log_z, "D-", color="C4", lw=1.8, ms=5, label=r"$\log Z/N$ (cond-$\alpha$)")
    ax.axhline(log_z0, color="C4", ls="--", lw=1.0, alpha=0.7,
               label=rf"neutral $\log Z_0/N={log_z0:.3f}$")
    ax.set_ylabel(r"$\log Z/N$ [nats]")
    ax.set_xlabel(r"$\varepsilon_{12}$")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    _branch_vline(ax)

    fig.tight_layout()
    p = out_dir / "log_Z_metrics_vs_eps12.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # overlay: normalize each curve to [0,1] for shape comparison
    fig, ax = plt.subplots(figsize=(10, 5.5))

    def _norm(y: np.ndarray) -> np.ndarray:
        lo, hi = float(np.min(y)), float(np.max(y))
        if hi - lo < 1e-15:
            return np.zeros_like(y)
        return (y - lo) / (hi - lo)

    ax.plot(eps, _norm(kl), "o-", lw=1.6, ms=4, label=r"$KL_{\mathrm{eq}}$ (norm.)")
    ax.plot(eps, _norm(eta_bar), "s-", lw=1.6, ms=4, label=r"$\bar\eta$ (norm.)")
    ax.plot(eps, _norm(eta_peak), "^-", lw=1.6, ms=4, label=r"peak $\eta$ (norm.)")
    ax.plot(eps, _norm(log_z), "D-", lw=1.6, ms=4, label=r"$\log Z/N$ (norm.)")
    ax.axvline(0, color="0.35", ls=":", lw=1.0)
    ax.set_xlabel(r"$\varepsilon_{12}$")
    ax.set_ylabel("normalized value")
    ax.set_title(rf"Shape comparison (each curve rescaled to $[0,1]$)  ($N={N}$)")
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = out_dir / "log_Z_metrics_vs_eps12_normalized.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # single figure, dual y-axes: left = KL + log Z, right = etas
    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(eps, kl, "o-", color="C0", lw=1.8, ms=5, label=r"$KL_{\mathrm{eq}}$ [bits]")
    ax1.plot(eps, log_z, "D-", color="C4", lw=1.8, ms=5, label=r"$\log Z/N$ [nats]")
    ax1.axvline(0, color="0.35", ls=":", lw=1.0)
    ax1.set_xlabel(r"$\varepsilon_{12}$")
    ax1.set_ylabel(r"$KL_{\mathrm{eq}}$ [bits] / $\log Z/N$ [nats]")
    ax2 = ax1.twinx()
    ax2.plot(eps, eta_bar, "s--", color="C2", lw=1.5, ms=4, label=r"$\bar\eta$")
    ax2.plot(eps, eta_peak, "^--", color="C3", lw=1.5, ms=4, label=r"peak $\eta$")
    ax2.set_ylabel(r"$\bar\eta$ / peak $\eta$")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper center", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(rf"All metrics vs $\varepsilon_{{12}}$ ($N={N}$, $s_1={s1:g}$, $s_2={s2:g}$)")
    fig.tight_layout()
    p3 = out_dir / "log_Z_metrics_vs_eps12_combined.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_correlations(rows: list[dict], out_dir: Path, *, N: int, s1: float, s2: float) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    x = np.array([r["log_Z_over_N"] for r in rows])
    pairs = [
        ("kl_eq_bits", r"$KL_{\mathrm{eq}}$ [bits]", "C0"),
        ("d_final_bits", r"$D(t_{\mathrm{final}})$ [bits]", "C1"),
        ("path_efficiency", r"$\bar\eta$", "C2"),
        ("peak_eta", r"peak $\eta$", "C3"),
        ("f4_slope_early", r"slope $F_4$ (early)", "C4"),
        ("f4_plus_logZ_slope_early", r"slope $F_4+\log Z/N$ (early)", "C5"),
        ("F4_plus_logZ_at_phi0", r"$F_4+\log Z/N$ at $\varphi$ start", "C6"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    corrs = {}
    for ax, (key, ylab, col) in zip(axes.flat, pairs):
        y = np.array([r[key] for r in rows])
        eps = np.array([r["eps12"] for r in rows])
        sc = ax.scatter(x, y, c=eps, cmap="coolwarm", s=45, edgecolors="0.3", linewidths=0.3)
        rho = _spearman(x, y)
        corrs[f"spearman_logZ_vs_{key}"] = rho
        ax.set_xlabel(r"$\log Z / N$ [nats]")
        ax.set_ylabel(ylab)
        ax.set_title(rf"$\rho_s={rho:+.2f}$", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes.flat[-1].axis("off")
    fig.colorbar(sc, ax=axes, label=r"$\varepsilon_{12}$", shrink=0.6)
    fig.suptitle(
        rf"Cond-$\alpha$ $\log Z/N$ vs dynamics ($N={N}$, $s_1={s1:g}$, $s_2={s2:g}$)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out_dir / "log_Z_vs_dynamics_scatter.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # aligned vs eps12
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    eps = np.array([r["eps12"] for r in rows])
    o = np.argsort(eps)
    axes[0].plot(eps[o], x[o], "o-", label=r"$\log Z/N$")
    axes[0].plot(eps[o], [rows[i]["kl_eq_bits"] for i in o], "s--", label=r"$KL_{\mathrm{eq}}$ [bits]")
    axes[0].set_ylabel("nats / bits")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Shape vs $\\varepsilon_{12}$")

    axes[1].plot(eps[o], [rows[i]["path_efficiency"] for i in o], "o-", color="C2", label=r"$\bar\eta$")
    axr = axes[1].twinx()
    axr.plot(eps[o], [rows[i]["peak_eta"] for i in o], "s--", color="C3", label="peak η")
    axes[1].set_ylabel(r"$\bar\eta$", color="C2")
    axr.set_ylabel("peak η", color="C3")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(eps[o], [rows[i]["f4_slope_early"] for i in o], "o-", label=r"slope $F_4$ early")
    axes[2].plot(eps[o], [rows[i]["f4_plus_logZ_slope_early"] for i in o], "s--",
                 label=r"slope $F_4+\log Z/N$ early")
    axes[2].set_xlabel(r"$\varepsilon_{12}$")
    axes[2].set_ylabel("slope [nats/gen]")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    p2 = out_dir / "log_Z_vs_dynamics_aligned.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return corrs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=SCRIPT_DIR / "eps12_delta_d_config.yaml")
    p.add_argument("--log-z-csv", type=Path, default=OUT / "log_Z_eps12_sweep.csv")
    p.add_argument("--metrics-csv", type=Path, default=None,
                   help="skip dynamics if present (from eps12_delta_d_sweep)")
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--pergen-max-t", type=int, default=None)
    p.add_argument("--from-csv", type=Path, default=None,
                   help="reuse merged CSV (skip dynamics rerun)")
    args = p.parse_args()

    cfg = load_config(args.config.resolve())
    log_z_by_eps = _load_log_z_csv(args.log_z_csv)

    merged_path = args.from_csv or (args.out_dir / "log_Z_vs_dynamics.csv")
    if merged_path.exists():
        print(f"Loading merged rows from {merged_path}", flush=True)
        rows = _load_merged_csv(merged_path)
    else:
        metrics_path = args.metrics_csv or (SCRIPT_DIR / "plots" / "eps12_delta_d" / "eps12_metrics.csv")
        if metrics_path.exists():
            print(f"Loading metrics from {metrics_path}", flush=True)
            metrics = _load_metrics_csv(metrics_path)
            print("Re-running trajectories for F4 slopes...", flush=True)
            runs, _ = _run_dynamics(cfg, pergen_max_t=args.pergen_max_t)
        else:
            print("Running full dynamics sweep...", flush=True)
            runs, metrics = _run_dynamics(cfg, pergen_max_t=args.pergen_max_t)

        rows = _merge_rows(cfg, log_z_by_eps, runs, metrics)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = args.out_dir / "log_Z_vs_dynamics.csv"
    with csv_out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    N = int(cfg["N"])
    s1 = float(cfg["s1"])
    s2 = float(cfg["s2"])
    corrs = _plot_correlations(rows, args.out_dir, N=N, s1=s1, s2=s2)
    branch_corrs = _plot_branch_correlations(rows, args.out_dir, N=N, s1=s1, s2=s2)
    _plot_vs_eps12(rows, args.out_dir, N=N, s1=s1, s2=s2)
    (args.out_dir / "log_Z_vs_dynamics_correlations.json").write_text(json.dumps(corrs, indent=2))
    (args.out_dir / "log_Z_vs_dynamics_correlations_by_branch.json").write_text(
        json.dumps(branch_corrs, indent=2),
    )

    print("\n=== Spearman(log Z/N, · ) — ALL ===")
    for k, v in sorted(corrs.items()):
        print(f"  {k.replace('spearman_logZ_vs_', ''):30s}  rho = {v:+.3f}")

    for branch in ("neg", "pos"):
        sub = _filter_branch_rows(rows, branch)
        print(f"\n=== {BRANCH_TITLES[branch]}  (n={len(sub)}) ===")
        for key, _ in CORR_KEYS:
            print(f"  {key:30s}  rho = {branch_corrs[branch][key]:+.3f}")

    print(f"\nSample (sorted by eps12):")
    print(f"{'eps12':>7} {'logZ/N':>8} {'KL_eq':>7} {'bar_eta':>8} {'peak_eta':>9} {'slope_F4':>9}")
    for r in sorted(rows, key=lambda z: z["eps12"]):
        print(f"{r['eps12']:7.2f} {r['log_Z_over_N']:8.4f} {r['kl_eq_bits']:7.4f} "
              f"{r['path_efficiency']:8.5f} {r['peak_eta']:9.5f} {r['f4_slope_early']:9.2e}")

    print(f"\nwrote {csv_out}")
    print(f"wrote {args.out_dir / 'log_Z_vs_dynamics_scatter.png'}")
    print(f"wrote {args.out_dir / 'log_Z_vs_dynamics_by_branch.png'}")
    print(f"wrote {args.out_dir / 'log_Z_metrics_vs_eps12.png'}")
    print(f"wrote {args.out_dir / 'log_Z_metrics_vs_eps12_normalized.png'}")
    print(f"wrote {args.out_dir / 'log_Z_metrics_vs_eps12_combined.png'}")


if __name__ == "__main__":
    main()
