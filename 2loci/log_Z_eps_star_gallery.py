#!/usr/bin/env python3
r"""
Per-parameter-set normalized metric plots vs $\varepsilon_{12}$, focused on shifted dips.

For each $(N, s_1, s_2, u, v, r)$:
  - predict dip $\varepsilon^* = -\mathrm{Cov}_\varphi(B,x_{11})/\mathrm{Var}_\varphi(x_{11})$
    with $B=s_1 p_1 + s_2 p_2$ (same as ``eps12_dip_cov_prediction``)
  - scan $\varepsilon_{12}$ on a grid **centered on $\varepsilon^*$** (not on 0)
  plots/f2b/eps_star_gallery/<label>/
    log_Z_metrics_vs_eps12_normalized.png
    C_vs_time_by_kl_eq.png
    C_vs_time_by_eps12.png
    trajectories.npz

Vertical lines on each plot:
  - $\varepsilon_{12}=0$ (gray dotted)
  - predicted $\varepsilon^*$ (red dashed)
  - fitted KL minimum (blue dashed)

Run:
  cd 2loci
  python log_Z_eps_star_gallery.py
  python log_Z_eps_star_gallery.py --reuse
  python log_Z_eps_star_gallery.py --plot-only
  python log_Z_eps_star_gallery.py --include-zero   # also s1+s2=0 control
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from eps12_dip_cov_prediction import cov_prediction  # noqa: E402
from eps12_delta_d_sweep import (  # noqa: E402
    Eps12Run,
    _add_eps12_colorbar,
    _add_score_colorbar,
    _format_scalar,
    _score_colors,
)
from log_Z_param_sweep import (  # noqa: E402
    Scenario,
    _cfg_from_scenario,
    _make_cfg_vec,
    _run_one_eps12_vec,
    _run_scenario,
    _write_csv,
)
from log_Z_vs_dynamics import _slope  # noqa: E402
from compare_sigma_dynamics import cond_alpha_reference_pmf  # noqa: E402
from eps12_delta_d_sweep import analyze_run  # noqa: E402
from f2b_reference import log_Z_cond_alpha  # noqa: E402
from simplex import (  # noqa: E402
    allele_freqs_from_genotype_fracs,
    build_sigma,
    compute_phi,
    fitness_exp,
    make_simplex_grid,
    mutation_matrix_asymmetric,
    params_from_config,
)

DEFAULT_OUT = SCRIPT_DIR / "plots" / "f2b" / "eps_star_gallery"

# (tag, s1, s2) — sign panel + sum-zero control
S_PAIRS: tuple[tuple[str, float, float], ...] = (
    ("mix_pp", 2.0e-3, -5.0e-4),
    ("same_pp", 1.2e-3, 4.0e-4),
    ("mix_pp_sm", 1.0e-3, -5.0e-4),
    ("same_mm", -4.0e-4, -1.2e-3),
    ("equal_mm", -5.0e-4, -5.0e-4),
    ("mix_mp", -1.0e-3, 5.0e-4),
    ("sum0", 1.0e-3, -1.0e-3),
)

NS_PAIRS = (
    (1, -2), (2, -3), (3, -2), (2, -1), (3, -1),
    (-2, 1), (-1, 2), (2, 1), (-3, 2),
)

MUT_CFGS: tuple[tuple[tuple[float, float], tuple[float, float], float, str], ...] = (
    ((1e-3, 1e-3), (1e-3, 1e-3), 1e-3, "ur1e3"),
    ((1e-2, 1e-2), (1e-2, 1e-2), 0.0, "u1e2"),
    ((1e-3, 1e-3), (1e-3, 1e-3), 0.0, "u1e3"),
    ((2e-2, 1e-2), (1e-2, 2e-2), 0.0, "asym"),
    ((1e-3, 1e-3), (1e-3, 1e-3), 1e-2, "r1e2"),
    ((5e-3, 5e-3), (5e-3, 5e-3), 0.0, "u5e3"),
    ((1e-4, 1e-4), (1e-4, 1e-4), 0.0, "u1e4"),
    ((3e-3, 3e-3), (3e-3, 3e-3), 1e-3, "u3e3_r1e3"),
    ((1e-2, 5e-3), (5e-3, 1e-2), 0.0, "asym_nor"),
    ((1e-3, 1e-3), (5e-3, 5e-3), 0.0, "v5e3"),
    ((5e-3, 1e-3), (1e-3, 5e-3), 0.0, "uv_x"),
    ((2e-3, 2e-3), (2e-3, 2e-3), 5e-3, "u2e3_r5e3"),
)


@dataclass(frozen=True)
class GalleryCase:
    label: str
    title: str
    scenario: Scenario
    eps_star: float


def _fmt_float(x: float) -> str:
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1:
        return f"{x:g}".replace(".", "p").replace("-", "m")
    exp = int(np.floor(np.log10(ax)))
    mant = x / 10.0 ** exp
    m = f"{mant:g}".replace(".", "p")
    return f"{m}e{exp}" if x > 0 else f"m{m}e{exp}"


def _eps12_grid_centered(eps_star: float, *, s1: float, s2: float, step: float) -> tuple[float, float, float]:
    """Grid centered on predicted dip; wide enough to see 0 and the minimum."""
    sel = max(abs(s1), abs(s2))
    half = max(2.0e-3, 8.0 * step, 6.0 * abs(eps_star))
    if sel >= 0.02:
        half = max(half, 0.06, 3.0 * sel)
    elif sel >= 0.005:
        half = max(half, 0.015)
    center = float(eps_star) if np.isfinite(eps_star) else 0.0
    lo = center - half
    hi = center + half
    # always bracket 0 and center for shape comparison
    lo = min(lo, -half * 0.15)
    hi = max(hi, half * 0.15)
    return lo, hi, step


def _build_eps12_grid(
    lo: float,
    hi: float,
    step: float,
    *,
    eps_star: float,
) -> np.ndarray:
    """Uniform grid on [lo, hi] with mandatory anchors at 0 and eps*."""
    grid = np.arange(lo, hi + 0.5 * step, step)
    anchors = [0.0]
    if np.isfinite(eps_star):
        anchors.append(float(eps_star))
    return np.sort(np.unique(np.round(np.concatenate([grid, anchors]), 12)))


def predict_eps_star(
    N: int, s1: float, s2: float,
    u: tuple[float, float], v: tuple[float, float], r: float,
) -> float:
    """Cov/Var dip prediction; falls back to on-the-fly phi if cache missing."""
    u0 = 0.5 * (u[0] + u[1])
    try:
        return float(cov_prediction(s1, s2, u0, N_pop=N, r=r))
    except FileNotFoundError:
        u_arr = np.asarray(u, dtype=float)
        v_arr = np.asarray(v, dtype=float)
        M = mutation_matrix_asymmetric(u_arr, v_arr)
        fracs = make_simplex_grid(N)
        params0 = params_from_config(
            _make_cfg_vec(N=N, r=r, u=u_arr, v=v_arr, s1=s1, s2=s2, eps12=0.0),
            SCRIPT_DIR,
        )
        phi, _, _, _, _ = compute_phi(params0, M)
        p1, p2 = allele_freqs_from_genotype_fracs(fracs)
        x11 = fracs[:, 3]
        B = s1 * p1 + s2 * p2
        Eb = float(np.sum(phi * B))
        Ex = float(np.sum(phi * x11))
        cov = float(np.sum(phi * (B - Eb) * (x11 - Ex)))
        var = float(np.sum(phi * (x11 - Ex) ** 2))
        if var <= 0.0:
            return float("nan")
        return -cov / var


def _has_eps12(rows: list[dict], target: float, *, tol: float = 5e-7) -> bool:
    return any(abs(float(r["eps12"]) - target) <= tol for r in rows)


def _append_eps12_points(
    case: GalleryCase,
    rows: list[dict],
    runs: list[Eps12Run],
    eps_targets: list[float],
    *,
    pergen_max_t: int,
) -> tuple[list[dict], list[Eps12Run]]:
    """Run only missing anchor eps12 values and merge into cached results."""
    missing = [float(e) for e in eps_targets if not _has_eps12(rows, float(e))]
    if not missing:
        return rows, runs

    sc = case.scenario
    u = np.asarray(sc.u, dtype=float)
    v = np.asarray(sc.v, dtype=float)
    N = sc.N
    M_mut = mutation_matrix_asymmetric(u, v)
    fracs = make_simplex_grid(N)
    params0 = params_from_config(
        _make_cfg_vec(N=N, r=sc.r, u=u, v=v, s1=sc.s1, s2=sc.s2, eps12=0.0),
        SCRIPT_DIR,
    )
    phi, _, _, _, _ = compute_phi(params0, M_mut)
    q_cond = cond_alpha_reference_pmf(fracs, M_mut, N)
    omega0 = fitness_exp(np.zeros((2, 2)))
    log_z0 = log_Z_cond_alpha(fracs, omega0, M_mut, N, mode="interior")["log_Z_over_N"]

    new_rows = list(rows)
    new_runs = list(runs)
    for eps12 in missing:
        print(f"    [append] eps12={eps12:+.6g}", flush=True)
        run = _run_one_eps12_vec(
            eps12, sc=sc, phi=phi, fracs=fracs, M_mut=M_mut,
            q_cond=q_cond, u=u, v=v, pergen_max_t=pergen_max_t,
        )
        m = analyze_run(run, N)
        t = run.pergen_times
        f4 = run.pergen_f4
        new_rows.append({
            "scenario": sc.label,
            "N": N,
            "s1": sc.s1,
            "s2": sc.s2,
            "u0": sc.u[0],
            "u1": sc.u[1],
            "v0": sc.v[0],
            "v1": sc.v[1],
            "r": sc.r,
            "eps12": float(eps12),
            "log_Z_over_N": float(log_Z_cond_alpha(
                fracs, fitness_exp(build_sigma(sc.s1, sc.s2, float(eps12))), M_mut, N,
                mode="interior",
            )["log_Z_over_N"]),
            "log_Z0_over_N": float(log_z0),
            "kl_eq_bits": m.kl_eq,
            "path_efficiency": m.path_efficiency,
            "peak_eta": m.peak_eta,
            "mean_c_cost": float(np.mean(run.pergen_c_cost)),
            "d_final_bits": m.d_final,
            "t_90": m.t_90,
            "f4_slope_early": _slope(t, f4, 0, 50),
        })
        new_runs.append(run)

    order = np.argsort([r["eps12"] for r in new_rows])
    new_rows = [new_rows[i] for i in order]
    new_runs = [new_runs[i] for i in order]
    return new_rows, new_runs


def build_cases(
    *,
    quick: bool,
    include_zero: bool,
    min_abs_eps: float,
    only_n: tuple[int, ...] | None = None,
) -> tuple[GalleryCase, ...]:
    out: list[GalleryCase] = []
    if only_n:
        n_list = list(only_n)
    elif quick:
        n_list = [20, 30, 50]
    else:
        n_list = [10, 15, 20, 30, 40, 50]
    mut_cfgs = MUT_CFGS[:3] if quick else MUT_CFGS
    s_pairs = S_PAIRS[:3] if quick else S_PAIRS

    def try_add(
        N: int, s1: float, s2: float,
        u: tuple[float, float], v: tuple[float, float], r: float,
        tag: str, stitle: str,
    ) -> None:
        eps_star = predict_eps_star(N, s1, s2, u, v, r)
        if not np.isfinite(eps_star):
            return
        if not include_zero and abs(s1 + s2) < 1e-15:
            return
        if abs(eps_star) < min_abs_eps and not include_zero:
            return
        label = f"N{N}_{tag}_s1{_fmt_float(s1)}_s2{_fmt_float(s2)}"
        title = stitle + rf"  ($N={N}$, $s_1={s1:g}$, $s_2={s2:g}$)"
        sc = Scenario(label, title, N, s1, s2, u, v, r)
        out.append(GalleryCase(label, title, sc, float(eps_star)))

    for N in n_list:
        for stag, s1, s2 in s_pairs:
            for u, v, r, mtag in mut_cfgs:
                try_add(N, s1, s2, u, v, r, f"{stag}_{mtag}", stag.replace("_", " "))

    if not quick:
        ns_n = list(only_n) if only_n else [10, 15, 20, 30, 50]
        for N in ns_n:
            for ns1, ns2 in NS_PAIRS:
                if max(abs(ns1), abs(ns2)) > min(8, N // 2):
                    continue
                s1, s2 = ns1 / N, ns2 / N
                for u, v, r, mtag in mut_cfgs[:6]:
                    try_add(
                        N, s1, s2, u, v, r,
                        f"Ns{ns1}m{abs(ns2)}_{mtag}",
                        rf"$N s_1={ns1}$, $N s_2={ns2}$",
                    )

    seen: set[str] = set()
    uniq: list[GalleryCase] = []
    for c in out:
        if c.label in seen:
            continue
        seen.add(c.label)
        uniq.append(c)
    return tuple(sorted(uniq, key=lambda c: (c.scenario.N, abs(c.eps_star))))


def _load_csv(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rec: dict = {}
            for k, v in row.items():
                if k in ("scenario", "scenario_title", "label") or not v:
                    rec[k] = v
                else:
                    rec[k] = float(v)
            rows.append(rec)
    return rows if len(rows) >= 5 else None


def _observed_dip(rows: list[dict]) -> float:
    eps = np.array([r["eps12"] for r in rows])
    kl = np.array([r["kl_eq_bits"] for r in rows])
    j = int(np.argmin(kl))
    # local quadratic if enough points
    lo = max(0, j - 2)
    hi = min(len(eps), j + 3)
    if hi - lo >= 3:
        a, b, c = np.polyfit(eps[lo:hi], kl[lo:hi], 2)
        if abs(a) > 1e-15:
            est = -b / (2.0 * a)
            if eps.min() <= est <= eps.max():
                return float(est)
    return float(eps[j])


def _plot_normalized(
    rows: list[dict],
    case: GalleryCase,
    out_dir: Path,
    *,
    eps_fit: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sc = case.scenario
    eps = np.array([r["eps12"] for r in rows])
    o = np.argsort(eps)
    eps = eps[o]
    kl = np.array([rows[i]["kl_eq_bits"] for i in o])
    pe = np.array([rows[i]["path_efficiency"] for i in o])
    pk = np.array([rows[i]["peak_eta"] for i in o])
    logz = np.array([rows[i]["log_Z_over_N"] for i in o])
    mc = np.array([rows[i].get("mean_c_cost", float("nan")) for i in o])

    def norm(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-15:
            return np.zeros_like(y)
        out = (y - lo) / (hi - lo)
        return np.where(np.isfinite(y), out, np.nan)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(eps, norm(kl), "o-", lw=1.6, ms=4, label=r"$KL_{\mathrm{eq}}$ (norm.)")
    ax.plot(eps, norm(pe), "s-", lw=1.6, ms=4, label=r"$\overline{\Delta D/(N\langle C\rangle)}$ (norm.)")
    ax.plot(eps, norm(pk), "^-", lw=1.6, ms=4, label=r"$\max \Delta D/(N\langle C\rangle)$ (norm.)")
    ax.plot(eps, norm(logz), "D-", lw=1.6, ms=4, label=r"$\log Z/N$ (norm.)")
    if np.isfinite(mc).any():
        ax.plot(eps, norm(mc), "v-", lw=1.6, ms=4, color="C5", label=r"$\langle C\rangle$ (norm.)")
    if "t_90" in rows[o[0]]:
        t90 = np.array([rows[i].get("t_90", float("nan")) for i in o])
        ok = np.isfinite(t90)
        if ok.any():
            ax.plot(eps[ok], norm(t90)[ok], "P-", lw=1.6, ms=4, color="C4",
                    label=r"$t_{90}$ (norm.)")

    eps_span = float(eps.max() - eps.min()) if eps.size > 1 else 0.01
    ax.axvline(0.0, color="0.35", ls=":", lw=1.0, label=r"$\varepsilon_{12}=0$")
    ax.axvline(case.eps_star, color="C3", ls="--", lw=1.2,
               label=rf"pred $\varepsilon^*={_format_scalar(case.eps_star, signed=True, ref_span=eps_span)}$")
    ax.axvline(eps_fit, color="C0", ls="--", lw=1.2,
               label=rf"fit KL min ${_format_scalar(eps_fit, signed=True, ref_span=eps_span)}$")

    ax.set_xlabel(r"$\varepsilon_{12}$")
    ax.set_ylabel("normalized value")
    ax.set_title(
        rf"Shape comparison (each curve rescaled to $[0,1]$)  ($N={sc.N}$, "
        rf"$s_1={sc.s1:g}$, $s_2={sc.s2:g}$)" + "\n"
        rf"$u=({sc.u[0]:g},{sc.u[1]:g})$, $v=({sc.v[0]:g},{sc.v[1]:g})$, $r={sc.r:g}$  |  "
        r"$\log Z/N$ = free fitness at equilibrium",
        fontsize=9,
    )
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "log_Z_metrics_vs_eps12_normalized.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _dip_vlines(ax, eps: np.ndarray, case: GalleryCase, eps_fit: float) -> None:
    eps_span = float(eps.max() - eps.min()) if eps.size > 1 else 0.01
    ax.axvline(0.0, color="0.35", ls=":", lw=1.0, zorder=0, label="_nolegend_")
    ax.axvline(case.eps_star, color="C3", ls="--", lw=1.0, alpha=0.85, zorder=0, label="_nolegend_")
    ax.axvline(eps_fit, color="C0", ls="--", lw=1.0, alpha=0.85, zorder=0, label="_nolegend_")


def _legend_data_lines(*axes) -> list:
    """Return plot lines with user-visible labels (exclude axvline auto-labels)."""
    out = []
    for ax in axes:
        for ln in ax.get_lines():
            lab = ln.get_label()
            if lab and not lab.startswith("_"):
                out.append(ln)
    return out


def _plot_raw_stacked(
    rows: list[dict],
    case: GalleryCase,
    out_dir: Path,
    *,
    eps_fit: float,
) -> None:
    """Four raw metric panels vs eps12 (not normalized)."""
    sc = case.scenario
    eps = np.array([r["eps12"] for r in rows])
    o = np.argsort(eps)
    eps = eps[o]
    kl = np.array([rows[i]["kl_eq_bits"] for i in o])
    pe = np.array([rows[i]["path_efficiency"] for i in o])
    pk = np.array([rows[i]["peak_eta"] for i in o])
    logz = np.array([rows[i]["log_Z_over_N"] for i in o])
    logz0 = float(rows[o[0]].get("log_Z0_over_N", np.nan))
    mc = np.array([rows[i].get("mean_c_cost", float("nan")) for i in o])
    has_mc = np.isfinite(mc).any()
    n_panels = 5 if has_mc else 4
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 2.6 * n_panels + 1.5), sharex=True)
    if n_panels == 1:
        axes = [axes]

    panels = [
        (kl, r"$KL_{\mathrm{eq}}$ [bits]", "C0", r"$KL_{\mathrm{eq}}=D(\psi^*\|\varphi^*)$"),
        (pe, r"$\overline{\Delta D/(N\langle C\rangle)}$", "C2", None),
        (pk, r"$\max \Delta D/(N\langle C\rangle)$", "C3", None),
        (logz, r"$\log Z/N$ [nats]", "C4", r"$\log Z/N$ (cond-$\alpha$)"),
    ]
    for ax, (y, ylab, col, lab) in zip(axes[:4], panels):
        ax.plot(eps, y, "o-", color=col, lw=1.6, ms=4, label=lab or ylab)
        if ylab.startswith(r"$\log") and np.isfinite(logz0):
            ax.axhline(logz0, color=col, ls="--", lw=1.0, alpha=0.65,
                       label=rf"neutral $\log Z_0/N={logz0:.4f}$")
        _dip_vlines(ax, eps, case, eps_fit)
        ax.set_ylabel(ylab, fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    if has_mc:
        ax = axes[4]
        ax.plot(eps, mc, "v-", color="C5", lw=1.6, ms=4, label=r"$\langle C\rangle_t$")
        _dip_vlines(ax, eps, case, eps_fit)
        ax.set_ylabel(r"$\langle C\rangle_t$", fontsize=9)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(r"$\varepsilon_{12}$")

    fig.suptitle(
        rf"Raw metrics vs $\varepsilon_{{12}}$ ($N={sc.N}$, $s_1={sc.s1:g}$, $s_2={sc.s2:g}$)" + "\n"
        rf"$u=({sc.u[0]:g},{sc.u[1]:g})$, $v=({sc.v[0]:g},{sc.v[1]:g})$, $r={sc.r:g}$  |  "
        r"dashed: pred $\varepsilon^*$, fit KL min",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "log_Z_metrics_vs_eps12_raw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_t90_vs_eps(
    rows: list[dict],
    case: GalleryCase,
    out_dir: Path,
    *,
    eps_fit: float,
) -> None:
    sc = case.scenario
    eps = np.array([r["eps12"] for r in rows])
    o = np.argsort(eps)
    eps = eps[o]
    if "t_90" not in rows[o[0]]:
        return
    t90 = np.array([rows[i].get("t_90", float("nan")) for i in o])
    kl = np.array([rows[i]["kl_eq_bits"] for i in o])
    if not np.isfinite(t90).any():
        return

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(eps, t90, "o-", color="C4", lw=1.8, ms=5,
             label=r"$t_{90}$ [generations]")
    _dip_vlines(ax1, eps, case, eps_fit)
    ax1.set_xlabel(r"$\varepsilon_{12}$")
    ax1.set_ylabel(r"$t_{90}$ [generations]", color="C4")
    ax1.tick_params(axis="y", labelcolor="C4")
    ax2 = ax1.twinx()
    ax2.plot(eps, kl, "s--", color="C0", lw=1.2, ms=3, alpha=0.75,
             label=r"$KL_{\mathrm{eq}}$ [bits]")
    ax2.set_ylabel(r"$KL_{\mathrm{eq}}$ [bits]", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")
    lines = _legend_data_lines(ax1, ax2)
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper left", fontsize=8)
    ax1.set_title(
        rf"$t_{{90}}$: first $t$ with $D(t)\geq 0.9\,KL_{{\mathrm{{eq}}}}$ vs $\varepsilon_{{12}}$"
        f"  ($N={sc.N}$, $s_1={sc.s1:g}$, $s_2={sc.s2:g}$)",
        fontsize=9,
    )
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "t90_vs_eps12.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_trajectories(runs: list[Eps12Run], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        eps12=np.array([r.eps12 for r in runs]),
        kl_eq=np.array([r.kl_eq for r in runs]),
        times=runs[0].pergen_times,
        c_cost=np.stack([r.pergen_c_cost for r in runs]),
        d_kl=np.stack([r.pergen_kl_phi_bits for r in runs]),
    )


def _load_trajectories(path: Path) -> list[Eps12Run]:
    d = np.load(path)
    times = d["times"]
    has_d = "d_kl" in d.files
    runs: list[Eps12Run] = []
    for i, eps in enumerate(d["eps12"]):
        d_kl = d["d_kl"][i] if has_d else np.full_like(times, np.nan)
        runs.append(Eps12Run(
            eps12=float(eps),
            kl_eq=float(d["kl_eq"][i]),
            f1_eq=float("nan"),
            f2_eq=float("nan"),
            f3_eq=float("nan"),
            wstar_eq=float("nan"),
            pergen_times=times,
            pergen_kl_phi_bits=d_kl,
            pergen_c_cost=d["c_cost"][i],
            pergen_f1=np.full_like(times, np.nan),
            pergen_f2=np.full_like(times, np.nan),
            pergen_f3=np.full_like(times, np.nan),
            pergen_f4=np.full_like(times, np.nan),
            pergen_mean_wstar=np.full_like(times, np.nan),
            pergen_info_f2=np.full_like(times, np.nan),
            pergen_info_f3=np.full_like(times, np.nan),
        ))
    return runs


def _plot_c_eta_vs_time_by_kl(
    runs: list[Eps12Run],
    case: GalleryCase,
    out_dir: Path,
) -> None:
    """C(t) and eta(t) trajectories for every eps12 in the grid, colored by KL_eq."""
    sc = case.scenario
    runs_f = [r for r in runs if np.isfinite(r.kl_eq)]
    if not runs_f:
        return
    colors, cbar_cmap, cbar_norm, cbar_ticks = _score_colors(runs_f)

    fig, (ax_c, ax_eta) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for run, color in zip(runs_f, colors):
        ax_c.plot(run.pergen_times, run.pergen_c_cost, "-", lw=1.2, color=color, alpha=0.92)
        t_eta, eta = _eta_trajectory(run, sc.N)
        ax_eta.plot(t_eta, eta, "-", lw=1.1, color=color, alpha=0.92)

    ax_c.set_ylabel(r"$C(t)=\langle C(x)\rangle_\psi$")
    ax_c.set_title(
        rf"$C(t)$ and $\eta(t)$ on fine $\varepsilon_{{12}}$ grid, colored by $KL_{{\mathrm{{eq}}}}$"
        f"  ($N={sc.N}$, $s_1={sc.s1:g}$, $s_2={sc.s2:g}$, "
        rf"$u=({sc.u[0]:g},{sc.u[1]:g})$, $v=({sc.v[0]:g},{sc.v[1]:g})$, $r={sc.r:g}$)",
        fontsize=9,
    )
    ax_c.grid(True, alpha=0.3)
    ax_eta.set_xlabel("generation $t$")
    ax_eta.set_ylabel(r"$\eta(t)=\Delta D/(N\,C(t))$")
    ax_eta.grid(True, alpha=0.3)
    ax_c.set_xlim(runs_f[0].pergen_times[0], runs_f[0].pergen_times[-1])
    _add_score_colorbar(fig, ax_c, cbar_cmap, cbar_norm, cbar_ticks)
    fig.tight_layout()
    fig.savefig(out_dir / "C_vs_time_by_kl_eq.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_c_vs_time_by_eps(
    runs: list[Eps12Run],
    case: GalleryCase,
    out_dir: Path,
) -> None:
    import matplotlib as mpl

    sc = case.scenario
    runs_f = sorted(runs, key=lambda r: r.eps12)
    eps_vals = np.array([r.eps12 for r in runs_f])
    cmap = plt.get_cmap("coolwarm")
    norm = mpl.colors.Normalize(vmin=float(eps_vals.min()), vmax=float(eps_vals.max()))

    fig, ax = plt.subplots(figsize=(9, 5))
    for run in runs_f:
        color = cmap(norm(run.eps12))
        ax.plot(
            run.pergen_times, run.pergen_c_cost, "-", lw=1.3, color=color, alpha=0.92,
        )
    ax.set_xlabel("generation $t$")
    ax.set_ylabel(r"$C(t)=\langle C(x)\rangle_\psi$")
    ax.set_title(
        rf"$C(t)$ colored by $\varepsilon_{{12}}$  ($N={sc.N}$, "
        rf"$s_1={sc.s1:g}$, $s_2={sc.s2:g}$, $u=({sc.u[0]:g},{sc.u[1]:g})$, "
        rf"$v=({sc.v[0]:g},{sc.v[1]:g})$, $r={sc.r:g}$)",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(runs_f[0].pergen_times[0], runs_f[0].pergen_times[-1])
    _add_eps12_colorbar(fig, ax, cmap, norm)
    fig.tight_layout()
    fig.savefig(out_dir / "C_vs_time_by_eps12.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _eta_trajectory(run: Eps12Run, N: int) -> tuple[np.ndarray, np.ndarray]:
    t = run.pergen_times
    d = run.pergen_kl_phi_bits
    c = run.pergen_c_cost
    delta_d = np.diff(d)
    nc = N * c[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(nc > 1e-300, delta_d / nc, np.nan)
    return t[1:], eta


def _plot_eta_vs_time_by_kl(runs: list[Eps12Run], case: GalleryCase, out_dir: Path) -> None:
    sc = case.scenario
    runs_f = [r for r in runs if np.isfinite(r.kl_eq) and np.isfinite(r.pergen_kl_phi_bits).any()]
    if not runs_f:
        return
    colors, cbar_cmap, cbar_norm, cbar_ticks = _score_colors(runs_f)
    fig, ax = plt.subplots(figsize=(9, 5))
    for run, color in zip(runs_f, colors):
        t_eta, eta = _eta_trajectory(run, sc.N)
        ax.plot(t_eta, eta, "-", lw=1.2, color=color, alpha=0.92)
    ax.set_xlabel("generation $t$")
    ax.set_ylabel(r"$\eta(t)=\Delta D/(N\,C(t))$")
    ax.set_title(
        rf"$\eta(t)$ colored by $KL_{{\mathrm{{eq}}}}$  ($N={sc.N}$, "
        rf"$s_1={sc.s1:g}$, $s_2={sc.s2:g}$)",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    _add_score_colorbar(fig, ax, cbar_cmap, cbar_norm, cbar_ticks)
    fig.tight_layout()
    fig.savefig(out_dir / "eta_vs_time_by_kl_eq.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_eta_vs_time_by_eps(runs: list[Eps12Run], case: GalleryCase, out_dir: Path) -> None:
    import matplotlib as mpl

    sc = case.scenario
    runs_f = sorted(
        [r for r in runs if np.isfinite(r.pergen_kl_phi_bits).any()],
        key=lambda r: r.eps12,
    )
    if not runs_f:
        return
    eps_vals = np.array([r.eps12 for r in runs_f])
    cmap = plt.get_cmap("coolwarm")
    norm = mpl.colors.Normalize(vmin=float(eps_vals.min()), vmax=float(eps_vals.max()))
    fig, ax = plt.subplots(figsize=(9, 5))
    for run in runs_f:
        t_eta, eta = _eta_trajectory(run, sc.N)
        ax.plot(t_eta, eta, "-", lw=1.2, color=cmap(norm(run.eps12)), alpha=0.92)
    ax.set_xlabel("generation $t$")
    ax.set_ylabel(r"$\eta(t)=\Delta D/(N\,C(t))$")
    ax.set_title(
        rf"$\eta(t)$ colored by $\varepsilon_{{12}}$  ($N={sc.N}$, "
        rf"$s_1={sc.s1:g}$, $s_2={sc.s2:g}$)",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    _add_eps12_colorbar(fig, ax, cmap, norm)
    fig.tight_layout()
    fig.savefig(out_dir / "eta_vs_time_by_eps12.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_case_figures(
    rows: list[dict],
    runs: list[Eps12Run] | None,
    case: GalleryCase,
    out_dir: Path,
    *,
    eps_fit: float,
    minimal_plots: bool = False,
) -> None:
    _plot_normalized(rows, case, out_dir, eps_fit=eps_fit)
    if minimal_plots:
        if runs:
            _plot_c_eta_vs_time_by_kl(runs, case, out_dir)
        return
    _plot_raw_stacked(rows, case, out_dir, eps_fit=eps_fit)
    _plot_t90_vs_eps(rows, case, out_dir, eps_fit=eps_fit)
    if runs:
        _plot_c_eta_vs_time_by_kl(runs, case, out_dir)
        _plot_c_vs_time_by_eps(runs, case, out_dir)
        if np.isfinite(runs[0].pergen_kl_phi_bits).any():
            _plot_eta_vs_time_by_kl(runs, case, out_dir)
            _plot_eta_vs_time_by_eps(runs, case, out_dir)


def _run_case(
    case: GalleryCase,
    out_root: Path,
    *,
    step: float,
    pergen_max_t: int,
    reuse: bool,
    minimal_plots: bool = False,
) -> tuple[list[dict], list[Eps12Run] | None]:
    out_dir = out_root / case.label
    cache = out_dir / "data.csv"
    traj_path = out_dir / "trajectories.npz"
    if reuse and (cached := _load_csv(cache)) and traj_path.exists():
        if "mean_c_cost" in cached[0] and "d_kl" in np.load(traj_path, allow_pickle=False).files:
            rows, runs = cached, _load_trajectories(traj_path)
            rows, runs = _append_eps12_points(
                case, rows, runs, [0.0], pergen_max_t=pergen_max_t,
            )
            if rows is not cached:
                eps_fit = _observed_dip(rows)
                for row in rows:
                    row["eps_star_pred"] = case.eps_star
                    row["eps_fit_kl"] = eps_fit
                    row["scenario_title"] = case.title
                _write_csv(rows, cache)
                _save_trajectories(runs, traj_path)
                _plot_case_figures(rows, runs, case, out_dir, eps_fit=eps_fit, minimal_plots=minimal_plots)
            else:
                print(f"  {case.label}  (cached)", flush=True)
            return rows, runs
        print(f"  {case.label}  (cached, missing mean C — rerunning)", flush=True)
    elif reuse and (cached := _load_csv(cache)) and "mean_c_cost" in cached[0]:
        print(f"  {case.label}  (cached, missing trajectories — rerunning)", flush=True)

    lo, hi, st = _eps12_grid_centered(case.eps_star, s1=case.scenario.s1, s2=case.scenario.s2, step=step)
    eps_values = _build_eps12_grid(lo, hi, st, eps_star=case.eps_star)
    cfg = _cfg_from_scenario(
        case.scenario,
        eps12_min=lo, eps12_max=hi, eps12_step=st,
        pergen_max_t=pergen_max_t,
    )
    cfg["eps12_values"] = eps_values
    print(
        f"  {case.label}  eps*={case.eps_star:+.2e}  scan [{lo:+.3g},{hi:+.3g}]"
        f" step={st:g}  n={len(eps_values)} (incl. eps=0)",
        flush=True,
    )
    rows, runs = _run_scenario(case.scenario, cfg, return_runs=True)
    eps_fit = _observed_dip(rows)
    for row in rows:
        row["eps_star_pred"] = case.eps_star
        row["eps_fit_kl"] = eps_fit
        row["scenario_title"] = case.title
    _write_csv(rows, cache)
    _save_trajectories(runs, traj_path)
    _plot_case_figures(rows, runs, case, out_dir, eps_fit=eps_fit, minimal_plots=minimal_plots)
    meta = {
        "label": case.label,
        "N": case.scenario.N,
        "s1": case.scenario.s1,
        "s2": case.scenario.s2,
        "u": list(case.scenario.u),
        "v": list(case.scenario.v),
        "r": case.scenario.r,
        "eps_star_pred": case.eps_star,
        "eps_fit_kl": eps_fit,
        "eps12_lo": lo,
        "eps12_hi": hi,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return rows, runs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--include-zero", action="store_true", help="include s1+s2=0 control")
    p.add_argument("--min-abs-eps", type=float, default=2e-4,
                   help="skip cases with |eps*| below this (default 2e-4)")
    p.add_argument("--step", type=float, default=None)
    p.add_argument("--pergen-max-t", type=int, default=500)
    p.add_argument("--case", type=str, default=None, help="run/replot single case label only")
    p.add_argument("--only-n", type=int, nargs="+", default=None,
                   help="restrict to these N values (e.g. --only-n 30 40)")
    p.add_argument("--out-dir", type=Path, default=None, help="output root directory")
    p.add_argument("--max-cases", type=int, default=None, help="limit number of cases")
    p.add_argument(
        "--minimal-plots",
        action="store_true",
        help="only normalized metrics (+ t90 curve) and combined C/eta vs time by KL_eq",
    )
    args = p.parse_args()

    step = args.step if args.step is not None else (5e-4 if not args.quick else 1e-3)
    only_n = tuple(args.only_n) if args.only_n else None
    out_root = args.out_dir
    if out_root is None:
        if only_n == (30, 40) or only_n == (40, 30):
            out_root = SCRIPT_DIR / "plots" / "f2b" / "eps_star_gallery_N30_N40"
        elif only_n == (40,):
            out_root = SCRIPT_DIR / "plots" / "f2b" / "eps_star_gallery_N40"
        else:
            out_root = DEFAULT_OUT
    if args.case:
        out_root.mkdir(parents=True, exist_ok=True)
        case_dir = out_root / args.case
        meta_path = case_dir / "meta.json"
        if meta_path.exists():
            import json as _json
            meta = _json.loads(meta_path.read_text())
            sc = Scenario(
                meta["label"], meta["label"], int(meta["N"]),
                float(meta["s1"]), float(meta["s2"]),
                tuple(meta["u"]), tuple(meta["v"]), float(meta["r"]),
            )
            eps_star = float(meta.get("eps_star_pred", predict_eps_star(
                sc.N, sc.s1, sc.s2, sc.u, sc.v, sc.r,
            )))
            cases = (GalleryCase(sc.label, sc.title, sc, eps_star),)
        else:
            cases = tuple(c for c in build_cases(
                quick=False, include_zero=True, min_abs_eps=0.0, only_n=only_n,
            ) if c.label == args.case)
            if not cases:
                raise SystemExit(f"unknown case label: {args.case}")
    else:
        cases = build_cases(
            quick=args.quick,
            include_zero=args.include_zero,
            min_abs_eps=args.min_abs_eps,
            only_n=only_n,
        )
    if args.max_cases:
        cases = cases[: args.max_cases]
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"{len(cases)} gallery cases  (|eps*|>={args.min_abs_eps:g}, step={step:g})\n", flush=True)

    summary: list[dict] = []
    for ci, case in enumerate(cases, 1):
        print(f"[{ci}/{len(cases)}]", flush=True)
        out_dir = out_root / case.label
        traj_path = out_dir / "trajectories.npz"
        if args.plot_only:
            rows = _load_csv(out_dir / "data.csv")
            if not rows:
                continue
            eps_fit = float(rows[0].get("eps_fit_kl", _observed_dip(rows)))
            runs = _load_trajectories(traj_path) if traj_path.exists() else None
        else:
            rows, runs = _run_case(
                case, out_root,
                step=step, pergen_max_t=args.pergen_max_t, reuse=args.reuse,
                minimal_plots=args.minimal_plots,
            )
            eps_fit = float(rows[0]["eps_fit_kl"])
        _plot_case_figures(
            rows, runs, case, out_dir, eps_fit=eps_fit, minimal_plots=args.minimal_plots,
        )
        summary.append({
            "label": case.label,
            "eps_star": case.eps_star,
            "eps_fit": eps_fit,
            "delta": eps_fit - case.eps_star,
            "N": case.scenario.N,
            "s1": case.scenario.s1,
            "s2": case.scenario.s2,
        })

    (out_root / "index.json").write_text(json.dumps({"n": len(summary), "cases": summary}, indent=2))
    print(f"\nwrote {out_root}/index.json  ({len(summary)} cases × 3 figures in {out_root}/<label>/)")


if __name__ == "__main__":
    main()
