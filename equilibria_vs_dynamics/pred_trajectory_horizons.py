#!/usr/bin/env python3
r"""
Trajectory + multi-horizon predictiveness for user-designed (s1,s2) backgrounds.

Runs discrete-chain dynamics from psi(0)=phi, saves KL(t) and rates on a dense
grid t=1..100, then evaluates Spearman rho(KL_eq, ·) across eps12 for several
horizons (avg rate till t, KL(t), init slope).

Run:
  cd equilibria_vs_dynamics
  python pred_trajectory_horizons.py
  python pred_trajectory_horizons.py --replot
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
TWOLOCI_DIR = SCRIPT_DIR.parent / "2loci"
if str(TWOLOCI_DIR) not in sys.path:
    sys.path.insert(0, str(TWOLOCI_DIR))

from simplex import (  # noqa: E402
    build_sigma,
    build_transition_matrix_row_stochastic,
    column_stochastic,
    compute_phi,
    dynamics_series,
    fitness_exp,
    free_fitness_1,
    kl_divergence_bits,
    linkage_disequilibrium_per_state,
    mutation_matrix_asymmetric,
    params_from_config,
    try_selection_equilibrium,
    w_potential_per_state,
)

# =============================================================================
# USER SETTINGS
# =============================================================================
N_MARKOV = 30
R_FIXED = 0.0
U_VALUES = [1.0e-4, 1.0e-2]
N_JOBS = -1

EPS12_MIN, EPS12_MAX, EPS12_STEP = -0.02, 0.02, 0.004

# Dense early times + long horizons for Spearman
TIMES_DENSE = list(range(1, 101))
HORIZONS_EXTRA = [150, 200]
TIMES_ALL = sorted(set(TIMES_DENSE + HORIZONS_EXTRA))

SPEARMAN_HORIZONS = [20, 50, 100, 200]
KL_AT_TIMES = sorted(set(SPEARMAN_HORIZONS + [80]))  # dense grid covers t=80
AVG_RATE_HORIZONS = list(SPEARMAN_HORIZONS)
MA_WINDOW = 5  # moving-average rate window (generations)

# (label_prefix, |s1|, |s2|, sign tags: p=+, m=-)
MAG_GROUPS: list[tuple[str, float, float, tuple[str, ...]]] = [
    ("13_53", 1.0e-3, 5.0e-3, ("pp", "pm", "mp")),
    ("10_01", 1.0e-2, 1.0e-3, ("pp", "pm", "mp")),
    ("01_01", 1.0e-4, 1.0e-4, ("pp", "mp")),
]

OUT_ROOT = SCRIPT_DIR / "results" / "pred_traj_design"


@dataclass(frozen=True)
class Background:
    label: str
    s1: float
    s2: float


def _make_backgrounds() -> list[Background]:
    out: list[Background] = []
    for prefix, m1, m2, tags in MAG_GROUPS:
        for tag in tags:
            sg1 = 1.0 if tag[0] == "p" else -1.0
            sg2 = 1.0 if tag[1] == "p" else -1.0
            out.append(Background(f"{prefix}_{tag}", sg1 * m1, sg2 * m2))
    return out


BACKGROUNDS = _make_backgrounds()
BG_BY_LABEL = {b.label: b for b in BACKGROUNDS}


def _sign_sym(x: float) -> str:
    return "+" if x >= 0 else "−"


def _bg_panel_title(bg: Background, extra: str = "") -> str:
    s1s, s2s = _sign_sym(bg.s1), _sign_sym(bg.s2)
    m1, m2 = abs(bg.s1), abs(bg.s2)
    title = rf"$s_1={s1s}{m1:g}$, $s_2={s2s}{m2:g}$  $\mathbf{{({s1s},{s2s})}}$"
    return f"{title}  {extra}" if extra else title


def _bg_axis_label(bg: Background) -> str:
    s1s, s2s = _sign_sym(bg.s1), _sign_sym(bg.s2)
    m1, m2 = abs(bg.s1), abs(bg.s2)
    return rf"$s_1={s1s}{m1:g}$" + "\n" + rf"$s_2={s2s}{m2:g}$" + "\n" + rf"$\mathbf{{({s1s},{s2s})}}$"


def _u_dir_tag(u: float) -> str:
    """Filesystem-safe folder name: u0p01 means u=v=0.01 (p = decimal point)."""
    return f"u{u:g}".replace(".", "p")


@dataclass
class TrajPoint:
    eps12: float
    kl_eq: float
    ff_eq: float
    times: np.ndarray
    kl: np.ndarray
    rate_times: np.ndarray
    rate_kl: np.ndarray
    rate_kl_ma: np.ndarray  # moving average on uniform grid


def _eps12_grid() -> np.ndarray:
    n = int(round((EPS12_MAX - EPS12_MIN) / EPS12_STEP)) + 1
    grid = EPS12_MIN + EPS12_STEP * np.arange(n)
    return np.unique(np.round(grid, 10))


def _make_params(u: float, v: float, r: float, s1: float, s2: float, eps12: float):
    cfg = {
        "_config_dir": str(TWOLOCI_DIR),
        "N": N_MARKOV,
        "r": float(r),
        "u": [float(u), float(u)],
        "v": [float(v), float(v)],
        "sigma": build_sigma(s1, s2, float(eps12)).tolist(),
        "times": TIMES_ALL,
        "n_jobs": N_JOBS,
        "neutral_cache_dir": "cache",
        "neutral": False,
        "top_k": 3,
        "plot_dynamics": False,
        "compute_f2": False,
        "ld_top_k_times": "first_last",
        "debug_kl": False,
    }
    return params_from_config(cfg, config_dir=TWOLOCI_DIR)


def _moving_avg_rate(times: np.ndarray, kl: np.ndarray, window: int) -> np.ndarray:
    """Rate from sliding window [t-window, t] on integer generation grid."""
    times_i = np.round(times).astype(int)
    t_to_kl = {int(t): float(k) for t, k in zip(times_i, kl)}
    t_max = int(times_i.max())
    rate = np.full(t_max, np.nan)
    for t in range(1, t_max + 1):
        if t not in t_to_kl:
            continue
        t0 = max(1, t - window)
        while t0 < t and t0 not in t_to_kl:
            t0 += 1
        if t0 >= t:
            continue
        rate[t - 1] = (t_to_kl[t] - t_to_kl[t0]) / (t - t0)
    return rate


def _kl_at_time(times: np.ndarray, kl: np.ndarray, t: int) -> float:
    idx = int(np.where(times == t)[0][0])
    return float(kl[idx])


def _avg_rate_until(rate_times: np.ndarray, rate_kl: np.ndarray, t_horizon: int) -> float:
    m = rate_times <= t_horizon
    if not m.any():
        return float("nan")
    return float(np.mean(rate_kl[m]))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x[m], y[m]).correlation)
    except Exception:
        rx = np.argsort(np.argsort(x[m])).astype(float)
        ry = np.argsort(np.argsort(y[m])).astype(float)
        if rx.std() == 0 or ry.std() == 0:
            return float("nan")
        return float(np.corrcoef(rx, ry)[0, 1])


def _load_cache(cache_path: Path) -> tuple[list[TrajPoint], dict]:
    points: list[TrajPoint] = []
    cache: dict = {"eps12": [], "kl_eq": [], "ff_eq": []}
    if not cache_path.exists():
        return points, cache
    d = np.load(cache_path, allow_pickle=True)
    for i in range(len(d["eps12"])):
        points.append(TrajPoint(
            eps12=float(d["eps12"][i]),
            kl_eq=float(d["kl_eq"][i]),
            ff_eq=float(d["ff_eq"][i]),
            times=d[f"times_{i}"],
            kl=d[f"kl_{i}"],
            rate_times=d[f"rate_times_{i}"],
            rate_kl=d[f"rate_kl_{i}"],
            rate_kl_ma=d[f"rate_ma_{i}"],
        ))
        cache["eps12"].append(float(d["eps12"][i]))
        cache["kl_eq"].append(float(d["kl_eq"][i]))
        cache["ff_eq"].append(float(d["ff_eq"][i]))
        cache[f"times_{i}"] = d[f"times_{i}"]
        cache[f"kl_{i}"] = d[f"kl_{i}"]
        cache[f"rate_times_{i}"] = d[f"rate_times_{i}"]
        cache[f"rate_kl_{i}"] = d[f"rate_kl_{i}"]
        cache[f"rate_ma_{i}"] = d[f"rate_ma_{i}"]
    return points, cache


def run_one_background(
    bg: Background, u: float, v: float, eps_grid: np.ndarray, cache_path: Path,
) -> list[TrajPoint]:
    points, cache = _load_cache(cache_path)
    done_eps = {p.eps12 for p in points}
    if len(done_eps) == len(eps_grid):
        return points

    M_mut = mutation_matrix_asymmetric(np.array([u, u]), np.array([v, v]))
    params0 = _make_params(u, v, R_FIXED, 0.0, 0.0, 0.0)
    phi, _, fracs, counts, _ = compute_phi(params0, M_mut)
    d_per_state = linkage_disequilibrium_per_state(fracs)

    for j, eps in enumerate(eps_grid, start=1):
        eps = float(eps)
        if eps in done_eps:
            continue
        params = _make_params(u, v, R_FIXED, bg.s1, bg.s2, eps)
        omega = fitness_exp(params.Sigma)
        w_ps = w_potential_per_state(fracs, omega, M_mut, params.N)
        T_row, _, _ = build_transition_matrix_row_stochastic(
            params.N, omega, M_mut, params.r,
            n_jobs=params.n_jobs, label=f"{bg.label}_eps{eps:+.3g}", verbose=False,
        )
        P_col = column_stochastic(T_row)
        dyn = dynamics_series(
            P_col, phi, TIMES_ALL, w_ps, d_per_state,
            counts, fracs, params.N, params.top_k, params.ld_top_k_times, psi_star=None,
        )
        times = np.asarray(dyn.times, dtype=float)
        kl = np.asarray(dyn.kl_phi_bits, dtype=float)
        rate_times = np.asarray(dyn.rate_times, dtype=float)
        rate_kl = np.asarray(dyn.rate_kl_bits, dtype=float)
        rate_ma = _moving_avg_rate(times, kl, MA_WINDOW)

        psi_star, _, ok = try_selection_equilibrium(P_col)
        kl_eq = kl_divergence_bits(psi_star, phi) if psi_star is not None else float("nan")
        ff_eq = free_fitness_1(psi_star, w_ps, params.N) if psi_star is not None else float("nan")

        pt = TrajPoint(eps, kl_eq, ff_eq, times, kl, rate_times, rate_kl, rate_ma)
        points.append(pt)

        cache["eps12"].append(eps)
        cache["kl_eq"].append(kl_eq)
        cache["ff_eq"].append(ff_eq)
        i = len(cache["eps12"]) - 1
        cache[f"times_{i}"] = times
        cache[f"kl_{i}"] = kl
        cache[f"rate_times_{i}"] = rate_times
        cache[f"rate_kl_{i}"] = rate_kl
        cache[f"rate_ma_{i}"] = rate_ma
        np.savez(cache_path, **cache)

        print(f"    [{j:2d}/{len(eps_grid)}] eps={eps:+.3f} KL_eq={kl_eq:.4f}", flush=True)
    points.sort(key=lambda p: p.eps12)
    return points


def _metrics_row(bg: Background, u: float, pt: TrajPoint) -> dict:
    row = {
        "background": bg.label, "u": u, "eps12": pt.eps12,
        "kl_eq": pt.kl_eq, "ff_eq": pt.ff_eq,
        "init_rate": float(pt.rate_kl[0]) if len(pt.rate_kl) else float("nan"),
    }
    for h in AVG_RATE_HORIZONS:
        row[f"avg_rate_{h}"] = _avg_rate_until(pt.rate_times, pt.rate_kl, h)
        row[f"avg_rate_ma{MA_WINDOW}_{h}"] = float(np.nanmean(pt.rate_kl_ma[:h]))
    for t in KL_AT_TIMES:
        row[f"kl_at_{t}"] = _kl_at_time(pt.times, pt.kl, t)
    return row


def _spearman_summary(rows: list[dict]) -> dict:
    kl_eq = np.array([r["kl_eq"] for r in rows])
    out = {"background": rows[0]["background"], "u": rows[0]["u"]}
    out["rho_init"] = _spearman(kl_eq, [r["init_rate"] for r in rows])
    for h in AVG_RATE_HORIZONS:
        out[f"rho_avg_rate_{h}"] = _spearman(kl_eq, [r[f"avg_rate_{h}"] for r in rows])
        out[f"rho_ma{MA_WINDOW}_{h}"] = _spearman(
            kl_eq, [r[f"avg_rate_ma{MA_WINDOW}_{h}"] for r in rows],
        )
    for t in KL_AT_TIMES:
        out[f"rho_kl_at_{t}"] = _spearman(kl_eq, [r[f"kl_at_{t}"] for r in rows])
    eps = np.array([r["eps12"] for r in rows])
    slope = np.array([r["init_rate"] for r in rows])
    out["eps_at_min_kl_eq"] = float(eps[np.nanargmin(kl_eq)])
    out["eps_at_min_init_rate"] = float(eps[np.nanargmin(slope)])
    return out


def _grid_shape(n: int) -> tuple[int, int]:
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    return nrow, ncol


def _curve_colors(sc: np.ndarray, cmap_name: str = "turbo"):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    n = len(sc)
    order = np.argsort(sc)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    colors = [cmap(ranks[i] / max(n - 1, 1)) for i in range(n)]
    return colors, cmap, ranks, order


def plot_traj_panel(
    points: list[TrajPoint],
    bg: Background,
    u: float,
    out_path: Path,
    *,
    y_attr: str,
    x_attr: str,
    ylabel: str,
    t_max: int | None = 100,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    pts = [p for p in points if np.isfinite(p.kl_eq)]
    sc = np.array([p.kl_eq for p in pts])
    colors, cmap, _, _ = _curve_colors(sc)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, p in enumerate(pts):
        x = getattr(p, x_attr)
        y = getattr(p, y_attr)
        if t_max is not None:
            m = x <= t_max
            x, y = x[m], y[m]
        ax.plot(x, y, color=colors[i], lw=1.7, alpha=0.95,
                label=rf"$\varepsilon_{{12}}={p.eps12:+.3g}$")
    ax.set_xlabel("generation $t$")
    ax.set_ylabel(ylabel)
    ax.set_title(
        _bg_panel_title(bg) + rf"   $N={N_MARKOV}$, $u=v={u:g}$, $r={R_FIXED:g}$",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    if t_max is not None:
        ax.set_xlim(0, t_max)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r"$KL_{eq}$ [bits]")
    ncol = min(6, len(pts))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=ncol, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def plot_rate_panel(
    points: list[TrajPoint], bg: Background, u: float, out_path: Path, t_max: int = 100,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    pts = [p for p in points if np.isfinite(p.kl_eq)]
    sc = np.array([p.kl_eq for p in pts])
    cmap = plt.get_cmap("turbo")
    n = len(sc)
    order = np.argsort(sc)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    colors = [cmap(ranks[i] / max(n - 1, 1)) for i in range(n)]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for i, p in enumerate(pts):
        m = p.times <= t_max
        axes[0].plot(p.times[m], p.kl[m], color=colors[i], lw=1.5, alpha=0.9)
        axes[1].plot(np.arange(1, t_max + 1), p.rate_kl_ma[:t_max],
                     color=colors[i], lw=1.5, alpha=0.9,
                     label=rf"$\varepsilon_{{12}}={p.eps12:+.3g}$")
    axes[0].set_ylabel(r"$\mathrm{KL}(\psi(t)\|\varphi)$ [bits]")
    axes[0].set_title(
        _bg_panel_title(bg) + rf"   $N={N_MARKOV}$, $u=v={u:g}$, $r={R_FIXED:g}$",
        fontsize=10,
    )
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel(rf"MA rate (window={MA_WINDOW}) [bits/gen]")
    axes[1].set_xlabel("generation $t$")
    axes[1].set_xlim(0, t_max)
    axes[1].grid(True, alpha=0.3)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, 1))
    cbar = fig.colorbar(sm, ax=axes, pad=0.02, fraction=0.02)
    cbar.set_label(r"$KL_{eq}$ [bits]")
    ncol = min(6, len(pts))
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=ncol, fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def plot_scatter_grid(
    by_bg: dict[str, list[TrajPoint]],
    u: float,
    path: Path,
    *,
    y_values: dict[str, np.ndarray],
    ylabel: str,
    suptitle: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [bg.label for bg in BACKGROUNDS if bg.label in by_bg]
    nrow, ncol = _grid_shape(len(labels))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow), squeeze=False)
    for k, label in enumerate(labels):
        ax = axes[k // ncol][k % ncol]
        pts = by_bg[label]
        kl_eq = np.array([p.kl_eq for p in pts])
        y = y_values[label]
        eps = np.array([p.eps12 for p in pts])
        sc = ax.scatter(kl_eq, y, c=eps, cmap="coolwarm", s=42, edgecolor="k", linewidth=0.4, zorder=2)
        ax.autoscale(enable=True, axis="both", tight=True)
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        d0 = max(x0, y0)
        d1 = min(x1, y1)
        if d1 > d0:
            ax.plot([d0, d1], [d0, d1], "k:", lw=1.3, alpha=0.55, zorder=1)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        bg = BG_BY_LABEL[label]
        rho = _spearman(kl_eq, y)
        ax.set_title(_bg_panel_title(bg, extra=rf"$\rho={rho:+.2f}$"), fontsize=9)
        ax.set_xlabel(r"$KL_{eq}(\psi^*\|\varphi)$ [bits]", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r"$\varepsilon_{12}$", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    for k in range(len(labels), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def _scatter_y_init(by_bg: dict[str, list[TrajPoint]]) -> dict[str, np.ndarray]:
    return {
        label: np.array([p.rate_kl[0] if len(p.rate_kl) else np.nan for p in pts])
        for label, pts in by_bg.items()
    }


def _scatter_y_avg_rate(by_bg: dict[str, list[TrajPoint]], h: int) -> dict[str, np.ndarray]:
    return {
        label: np.array([_avg_rate_until(p.rate_times, p.rate_kl, h) for p in pts])
        for label, pts in by_bg.items()
    }


def _scatter_y_kl_at(by_bg: dict[str, list[TrajPoint]], t: int) -> dict[str, np.ndarray]:
    return {
        label: np.array([_kl_at_time(p.times, p.kl, t) for p in pts])
        for label, pts in by_bg.items()
    }


def plot_all_scatter_grids(by_bg: dict[str, list[TrajPoint]], u: float, u_dir: Path) -> None:
    u_tag = rf"$u=v={u:g}$"
    plot_scatter_grid(
        by_bg, u, u_dir / "scatter_grid.png",
        y_values=_scatter_y_init(by_bg),
        ylabel=r"initial $d\,KL/dt$ [bits/gen]",
        suptitle=rf"$KL_{{eq}}$ vs initial accumulation rate ({u_tag})",
    )
    for h in AVG_RATE_HORIZONS:
        plot_scatter_grid(
            by_bg, u, u_dir / f"scatter_avg_rate_t{h}.png",
            y_values=_scatter_y_avg_rate(by_bg, h),
            ylabel=rf"avg $d\,KL/dt$ for $t \leq {h}$ [bits/gen]",
            suptitle=rf"$KL_{{eq}}$ vs average accumulation rate until $t={h}$ ({u_tag})",
        )
    for t in KL_AT_TIMES:
        plot_scatter_grid(
            by_bg, u, u_dir / f"scatter_kl_at_t{t}.png",
            y_values=_scatter_y_kl_at(by_bg, t),
            ylabel=rf"$KL(\psi({t})\|\varphi)$ [bits]",
            suptitle=rf"$KL_{{eq}}$ vs $KL(\psi({t})\|\varphi)$ ({u_tag})",
        )



def plot_dip_grid(by_bg: dict[str, list[TrajPoint]], u: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [bg.label for bg in BACKGROUNDS if bg.label in by_bg]
    nrow, ncol = _grid_shape(len(labels))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.0 * nrow), squeeze=False)
    for k, label in enumerate(labels):
        ax = axes[k // ncol][k % ncol]
        pts = by_bg[label]
        eps = np.array([p.eps12 for p in pts])
        kl_eq = np.array([p.kl_eq for p in pts])
        slope = np.array([p.rate_kl[0] if len(p.rate_kl) else np.nan for p in pts])
        ax.plot(eps, kl_eq, "o-", color="C0", lw=1.6, ms=4)
        ax.set_xlabel(r"$\varepsilon_{12}$", fontsize=8)
        ax.set_ylabel(r"$KL_{eq}$ [bits]", color="C0", fontsize=8)
        ax.tick_params(axis="y", labelcolor="C0", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax2 = ax.twinx()
        ax2.plot(eps, slope, "s--", color="C3", lw=1.4, ms=3.5)
        ax2.set_ylabel(r"init $d\,KL/dt$", color="C3", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="C3", labelsize=7)
        ax.set_title(_bg_panel_title(BG_BY_LABEL[label]), fontsize=9)
    for k in range(len(labels), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(rf"$KL_{{eq}}$ and initial rate vs $\varepsilon_{{12}}$ (u=v={u:g})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def plot_spearman_bars(summaries: list[dict], u: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [s for s in summaries if s["u"] == u]
    labels = [_bg_axis_label(BG_BY_LABEL[r["background"]]) for r in rows]
    x = np.arange(len(labels))
    w = 0.18
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 3, 4.8))
    ax.bar(x - 1.5 * w, [r["rho_init"] for r in rows], w, label="vs init rate", color="C0")
    ax.bar(x - 0.5 * w, [r["rho_avg_rate_20"] for r in rows], w, label="vs avg rate t<=20", color="C2")
    ax.bar(x + 0.5 * w, [r["rho_kl_at_20"] for r in rows], w, label=r"vs $KL(20)$", color="C1")
    ax.bar(x + 1.5 * w, [r["rho_avg_rate_100"] for r in rows], w, label="vs avg rate t<=100", color="C4")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(r"Spearman $\rho$ ($KL_{eq}$ vs rate)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(rf"Predictiveness by background (u=v={u:g})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def plot_spearman_horizon_grid(summaries: list[dict], u: float, path: Path) -> None:
    """Per-background rho vs horizon t (cleaner than one crowded line plot)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [s for s in summaries if s["u"] == u]
    labels = [_bg_axis_label(BG_BY_LABEL[r["background"]]) for r in rows]
    nrow, ncol = _grid_shape(len(labels))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow), squeeze=False)
    for k, row in enumerate(rows):
        ax = axes[k // ncol][k % ncol]
        hs = SPEARMAN_HORIZONS
        ax.plot(hs, [row[f"rho_avg_rate_{h}"] for h in hs], "o-", color="C0", lw=1.5, ms=4, label="avg rate")
        ax.plot(hs, [row[f"rho_kl_at_{h}"] for h in hs], "s--", color="C1", lw=1.4, ms=4, label=r"$KL(t)$")
        ax.set_ylim(0.8, 1.05)
        ax.set_title(_bg_panel_title(BG_BY_LABEL[row["background"]]), fontsize=9)
        ax.set_xlabel("$t$", fontsize=8)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=7)
    for k in range(len(labels), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(rf"$\rho(KL_{{eq}}, \cdot)$ vs horizon (u=v={u:g})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def plot_spearman_curves(summaries: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for row in summaries:
        hs = SPEARMAN_HORIZONS
        rhos = [row[f"rho_avg_rate_{h}"] for h in hs]
        ax.plot(hs, rhos, "o-", lw=1.5, ms=5,
                label=f"{row['background']} u={row['u']:g}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("horizon $t$ (avg rate till $t$)")
    ax.set_ylabel(r"Spearman $\rho(KL_{eq}$, avg rate)")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_title(r"Predictiveness vs accumulation horizon")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replot", action="store_true", help="plots only from cached npz")
    ap.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "N": N_MARKOV, "r": R_FIXED, "u_values": U_VALUES,
        "backgrounds": [{"label": b.label, "s1": b.s1, "s2": b.s2} for b in BACKGROUNDS],
        "eps12": _eps12_grid().tolist(),
        "times_dense": TIMES_DENSE,
        "horizons": SPEARMAN_HORIZONS,
        "kl_at_times": KL_AT_TIMES,
        "avg_rate_horizons": AVG_RATE_HORIZONS,
    }
    (args.out_dir / "settings.json").write_text(json.dumps(meta, indent=2))

    eps_grid = _eps12_grid()
    all_rows: list[dict] = []
    all_summaries: list[dict] = []

    print("Backgrounds:", flush=True)
    for b in BACKGROUNDS:
        print(f"  {b.label}: s1={b.s1:g}, s2={b.s2:g}", flush=True)

    for u in U_VALUES:
        u_tag = _u_dir_tag(u)
        u_dir = args.out_dir / u_tag
        u_dir.mkdir(parents=True, exist_ok=True)
        by_bg: dict[str, list[TrajPoint]] = {}
        u_summaries: list[dict] = []

        for bg in BACKGROUNDS:
            cache_path = u_dir / f"cache_{bg.label}.npz"
            print(f"\n=== u=v={u:g}  {bg.label} ===", flush=True)
            if args.replot:
                points, _ = _load_cache(cache_path)
                if not points:
                    print(f"  skip {bg.label}: no cache", flush=True)
                    continue
            else:
                points = run_one_background(bg, u, u, eps_grid, cache_path)

            by_bg[bg.label] = points
            plot_traj_panel(
                points, bg, u, u_dir / f"info_kl_vs_t_{bg.label}.png",
                y_attr="kl", x_attr="times",
                ylabel=r"$\mathrm{KL}(\psi(t)\|\varphi)$ [bits]",
            )
            plot_traj_panel(
                points, bg, u, u_dir / f"info_kl_rate_vs_t_{bg.label}.png",
                y_attr="rate_kl", x_attr="rate_times",
                ylabel=r"$\mathrm{d\,KL}/\mathrm{d}t$ [bits/gen]",
            )
            plot_rate_panel(points, bg, u, u_dir / f"rate_kl_ma_{bg.label}.png")
            rows = [_metrics_row(bg, u, p) for p in points]
            all_rows.extend(rows)
            u_summaries.append(_spearman_summary(rows))
            all_summaries.append(u_summaries[-1])

        if by_bg:
            plot_all_scatter_grids(by_bg, u, u_dir)
            plot_dip_grid(by_bg, u, u_dir / "dip_grid.png")
            plot_spearman_bars(u_summaries, u, u_dir / "spearman_bars.png")
            plot_spearman_horizon_grid(u_summaries, u, u_dir / "spearman_horizon_grid.png")

    # CSV outputs
    if all_rows:
        fields = list(all_rows[0].keys())
        with (args.out_dir / "metrics_all.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)

    if all_summaries:
        fields = list(all_summaries[0].keys())
        with (args.out_dir / "spearman_horizons.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_summaries)
        plot_spearman_curves(all_summaries, args.out_dir / "spearman_vs_horizon.png")

    print(f"\nDone. Outputs in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
