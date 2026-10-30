#!/usr/bin/env python3
r"""
ε₁₂ sweep with coupled mutation model (``simplex_not_sym``).

Goal: test whether peak η still occurs at ε=0 when mutation is non-independent
(ux: 00↔11, vx: 01↔10, order of magnitude above u, v).

Produces gallery-style normalized metric plot plus raw η vs ε₁₂ panel.

Run:
  cd 2loci
  python eps12_not_sym_investigation.py
  python eps12_not_sym_investigation.py --config eps12_not_sym_config.yaml --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_sigma_dynamics import cond_alpha_reference_pmf  # noqa: E402
from eps12_delta_d_sweep import (  # noqa: E402
    Eps12Metrics,
    analyze_run,
    run_one_eps12,
    _eps12_grid,
    _format_scalar,
)
from f2b_reference import eps_star_dip_prediction, log_Z_cond_alpha  # noqa: E402
from simplex import (  # noqa: E402
    build_sigma,
    compute_phi,
    fitness_exp,
    load_config,
    mutation_matrix_asymmetric,
    params_from_config as params_from_config_sym,
)
from simplex_not_sym import (  # noqa: E402
    build_mutation_matrix,
    params_from_config as params_from_config_not_sym,
    print_mutation_matrix_coupled,
)


def _make_cfg(
    *,
    N: int,
    r: float,
    u: list[float],
    v: list[float],
    s1: float,
    s2: float,
    eps12: float,
    ux: list[float] | None = None,
    vx: list[float] | None = None,
    coupled_factor: float = 10.0,
) -> dict:
    cfg = {
        "_config_dir": str(SCRIPT_DIR),
        "N": N,
        "r": float(r),
        "u": [float(u[0]), float(u[1])],
        "v": [float(v[0]), float(v[1])],
        "sigma": build_sigma(s1, s2, eps12).tolist(),
        "times": [0],
        "n_jobs": -1,
        "neutral_cache_dir": "cache",
        "neutral": False,
        "top_k": 3,
        "plot_dynamics": False,
        "compute_f2": False,
        "ld_top_k_times": "first_last",
        "debug_kl": False,
        "coupled_factor": coupled_factor,
    }
    if ux is not None:
        cfg["ux"] = [float(ux[0]), float(ux[1])]
    if vx is not None:
        cfg["vx"] = [float(vx[0]), float(vx[1])]
    return cfg


def _run_sweep(
    *,
    label: str,
    M_mut: np.ndarray,
    N: int,
    r: float,
    u: list[float],
    v: list[float],
    s1: float,
    s2: float,
    eps_grid: np.ndarray,
    pergen_max_t: int,
    params_fn,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    params0 = params_fn(
        _make_cfg(N=N, r=r, u=u, v=v, s1=s1, s2=s2, eps12=0.0),
        SCRIPT_DIR,
    )
    phi, _, fracs, _, src = compute_phi(params0, M_mut)
    print(f"  [{label}] phi: {src}", flush=True)
    q_cond = cond_alpha_reference_pmf(fracs, M_mut, N)

    omega0 = fitness_exp(build_sigma(s1, s2, 0.0))
    log_z0 = log_Z_cond_alpha(fracs, omega0, M_mut, N, mode="interior")["log_Z_over_N"]

    rows: list[dict] = []
    for i, eps in enumerate(eps_grid, start=1):
        print(f"  [{label}] [{i}/{len(eps_grid)}] eps12={eps:+.4g}", flush=True)
        run = run_one_eps12(
            float(eps),
            N=N, r=r, u=u[0], v=v[0], s1=s1, s2=s2,
            phi=phi, fracs=fracs, M_mut=M_mut, q_cond=q_cond,
            pergen_max_t=pergen_max_t, verbose=False,
        )
        m: Eps12Metrics = analyze_run(run, N)
        omega = fitness_exp(build_sigma(s1, s2, float(eps)))
        log_z = log_Z_cond_alpha(fracs, omega, M_mut, N, mode="interior")["log_Z_over_N"]
        c_mean = float(np.mean(run.pergen_c_cost))
        rows.append({
            "model": label,
            "eps12": float(eps),
            "kl_eq_bits": m.kl_eq,
            "path_efficiency": m.path_efficiency,
            "peak_eta": m.peak_eta,
            "t_peak_eta": m.t_peak_eta,
            "t_90": m.t_90,
            "mean_c_cost": c_mean,
            "log_Z_over_N": float(log_z),
            "log_Z0_over_N": float(log_z0),
            "wstar_eq": m.wstar_eq,
            "d_final": m.d_final,
        })
    return rows, phi, fracs


def _norm(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-15:
        return np.zeros_like(y)
    out = (y - lo) / (hi - lo)
    return np.where(np.isfinite(y), out, np.nan)


def _peak_eps(eps: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(y)
    if not ok.any():
        return float("nan")
    return float(eps[ok][int(np.nanargmax(y[ok]))])


def _min_eps(eps: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(y)
    if not ok.any():
        return float("nan")
    return float(eps[ok][int(np.nanargmin(y[ok]))])


def _eps_xlim(eps: np.ndarray, *, pad_frac: float = 0.06) -> tuple[float, float]:
    lo, hi = float(np.min(eps)), float(np.max(eps))
    span = hi - lo
    pad = pad_frac * span if span > 0 else max(abs(lo), abs(hi), 1e-4) * 0.15
    return lo - pad, hi + pad


def _format_eps_axis(ax, eps: np.ndarray) -> None:
    """Keep x-axis on the sweep range; readable ticks for small |ε|."""
    lo, hi = _eps_xlim(eps)
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _pos: _fmt_eps(float(x))),
    )


def _fit_kl_min(rows: list[dict]) -> float:
    eps = np.array([r["eps12"] for r in rows], dtype=float)
    kl = np.array([r["kl_eq_bits"] for r in rows], dtype=float)
    ok = np.isfinite(kl)
    if ok.sum() < 3:
        return float("nan")
    eps, kl = eps[ok], kl[ok]
    j = int(np.argmin(kl))
    lo, hi = max(0, j - 2), min(len(eps), j + 3)
    if hi - lo >= 3:
        a, b, c = np.polyfit(eps[lo:hi], kl[lo:hi], 2)
        if abs(a) > 1e-15:
            est = -b / (2.0 * a)
            if eps.min() <= est <= eps.max():
                return float(est)
    return float(eps[j])


def _build_eps_grid(
    cfg: dict,
    *,
    eps_stars: list[float],
) -> np.ndarray:
    """Uniform grid from config; optional dense sampling near each predicted ε*."""
    base = _eps12_grid(cfg)
    step = float(cfg.get("eps12_step", 0.00015))
    parts = [base, np.array([0.0], dtype=float)]

    fine_step = cfg.get("eps12_fine_step")
    if fine_step is not None:
        fine_step = float(fine_step)
        fine_half = float(cfg.get("eps12_fine_half", max(4.0 * step, 0.0004)))
        for eps_star in eps_stars:
            if not np.isfinite(eps_star):
                continue
            lo = float(eps_star) - fine_half
            hi = float(eps_star) + fine_half
            parts.append(np.arange(lo, hi + 0.5 * fine_step, fine_step))

    for eps_star in eps_stars:
        if np.isfinite(eps_star):
            parts.append(np.array([float(eps_star)], dtype=float))

    merged = np.concatenate(parts)
    return np.sort(np.unique(np.round(merged, 12)))


def _predict_eps_star(phi: np.ndarray, fracs: np.ndarray, s1: float, s2: float) -> float:
    """Cov/Var dip location under neutral φ (see ``f2b_reference.eps_star_dip_prediction``)."""
    return float(eps_star_dip_prediction(phi, fracs, s1, s2))


def _fmt_eps(v: float) -> str:
    if not np.isfinite(v):
        return "nan"
    av = abs(v)
    if av == 0:
        return "0"
    if av < 1e-3:
        return f"{v:+.3e}"
    return f"{v:+.4g}"


def _dip_vlines(
    ax,
    eps: np.ndarray,
    *,
    eps_star_pred: float,
    eps_fit_kl: float | None = None,
) -> None:
    """Vertical guides; only draw markers inside the sweep window."""
    xlo, xhi = _eps_xlim(eps)
    if xlo <= 0.0 <= xhi:
        ax.axvline(0.0, color="0.35", ls=":", lw=1.0, zorder=0, label=r"$\varepsilon_{12}=0$")
    if np.isfinite(eps_star_pred) and xlo <= eps_star_pred <= xhi:
        ax.axvline(
            eps_star_pred, color="C3", ls="--", lw=1.2, zorder=0,
            label=rf"pred $\varepsilon^*$ (Cov/Var) = {_fmt_eps(eps_star_pred)}",
        )
    if eps_fit_kl is not None and np.isfinite(eps_fit_kl) and xlo <= eps_fit_kl <= xhi:
        ax.axvline(
            eps_fit_kl, color="C0", ls="--", lw=1.2, zorder=0,
            label=rf"fit $KL_{{\mathrm{{eq}}}}$ min = {_fmt_eps(eps_fit_kl)}",
        )


def _plot_eta_raw(
    rows_not_sym: list[dict],
    rows_sym: list[dict] | None,
    out_dir: Path,
    *,
    eps_star_sym: float,
    eps_star_not_sym: float,
    eps_fit_kl_sym: float | None,
    eps_fit_kl_not_sym: float | None,
    metric_key: str,
    ylabel: str,
    title_metric: str,
    outfile: str,
    suptitle: str,
    extremum: dict[str, str] | None = None,
) -> None:
    extremum = extremum or {}
    dip = {
        "symmetric": (eps_star_sym, eps_fit_kl_sym),
        "coupled": (eps_star_not_sym, eps_fit_kl_not_sym),
    }
    fig, axes = plt.subplots(1, 2 if rows_sym else 1, figsize=(14 if rows_sym else 8, 4.5), squeeze=False)

    for ax, rows, lab in (
        (axes[0, 0], rows_sym, "symmetric"),
        (axes[0, 1] if rows_sym else None, rows_not_sym, "coupled"),
    ):
        if rows is None:
            continue
        eps = np.array([r["eps12"] for r in rows])
        o = np.argsort(eps)
        eps = eps[o]
        y = np.array([rows[i][metric_key] for i in o])
        est_star, fit_kl = dip[lab]
        _dip_vlines(ax, eps, eps_star_pred=est_star, eps_fit_kl=fit_kl)
        ax.plot(eps, y, "o-", ms=5, lw=1.8, label=rf"{title_metric} ({lab})")
        mode = extremum.get(lab, "max")
        xlo, xhi = _eps_xlim(eps)
        if mode in ("max", "both"):
            ep = _peak_eps(eps, y)
            if np.isfinite(ep) and xlo <= ep <= xhi:
                ax.axvline(ep, color="C2", ls="-.", alpha=0.7,
                           label=rf"argmax = {_fmt_eps(ep)} ({lab})")
        if mode in ("min", "both"):
            em = _min_eps(eps, y)
            if np.isfinite(em) and xlo <= em <= xhi:
                ax.axvline(em, color="C1", ls="-.", alpha=0.7,
                           label=rf"argmin = {_fmt_eps(em)} ({lab})")
        _format_eps_axis(ax, eps)
        ax.set_xlabel(r"$\varepsilon_{12}$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_metric} — {lab}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_peak_eta_raw(
    rows_not_sym: list[dict],
    rows_sym: list[dict] | None,
    out_dir: Path,
    *,
    eps_star_sym: float,
    eps_star_not_sym: float,
    eps_fit_kl_sym: float | None,
    eps_fit_kl_not_sym: float | None,
) -> None:
    _plot_eta_raw(
        rows_not_sym, rows_sym, out_dir,
        eps_star_sym=eps_star_sym, eps_star_not_sym=eps_star_not_sym,
        eps_fit_kl_sym=eps_fit_kl_sym, eps_fit_kl_not_sym=eps_fit_kl_not_sym,
        metric_key="peak_eta",
        ylabel=r"peak $\Delta D / (N \langle C \rangle)$",
        title_metric=r"peak $\eta$",
        outfile="peak_eta_vs_eps12.png",
        suptitle=r"Raw peak $\eta$ vs $\varepsilon_{12}$",
    )


def _plot_mean_eta_raw(
    rows_not_sym: list[dict],
    rows_sym: list[dict] | None,
    out_dir: Path,
    *,
    eps_star_sym: float,
    eps_star_not_sym: float,
    eps_fit_kl_sym: float | None,
    eps_fit_kl_not_sym: float | None,
) -> None:
    _plot_eta_raw(
        rows_not_sym, rows_sym, out_dir,
        eps_star_sym=eps_star_sym, eps_star_not_sym=eps_star_not_sym,
        eps_fit_kl_sym=eps_fit_kl_sym, eps_fit_kl_not_sym=eps_fit_kl_not_sym,
        metric_key="path_efficiency",
        ylabel=r"$\overline{\Delta D / (N \langle C \rangle)}$",
        title_metric=r"mean $\eta$",
        outfile="mean_eta_vs_eps12.png",
        suptitle=r"Raw mean $\eta$ (path efficiency) vs $\varepsilon_{12}$",
        extremum={"symmetric": "max", "coupled": "min"},
    )


def _plot_normalized(
    rows_not_sym: list[dict],
    rows_sym: list[dict] | None,
    out_dir: Path,
    *,
    N: int,
    s1: float,
    s2: float,
    u: list[float],
    v: list[float],
    ux: list[float],
    vx: list[float],
    r: float,
    eps_star_sym: float,
    eps_star_not_sym: float,
    eps_fit_kl_sym: float | None,
    eps_fit_kl_not_sym: float | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _one_panel(
        ax,
        rows: list[dict],
        title: str,
        *,
        eps_star: float,
        eps_fit_kl: float | None,
    ) -> None:
        eps = np.array([r["eps12"] for r in rows])
        o = np.argsort(eps)
        eps = eps[o]
        kl = np.array([rows[i]["kl_eq_bits"] for i in o])
        pe = np.array([rows[i]["path_efficiency"] for i in o])
        pk = np.array([rows[i]["peak_eta"] for i in o])
        logz = np.array([rows[i]["log_Z_over_N"] for i in o])
        mc = np.array([rows[i]["mean_c_cost"] for i in o])

        _dip_vlines(ax, eps, eps_star_pred=eps_star, eps_fit_kl=eps_fit_kl)

        ax.plot(eps, _norm(kl), "o-", lw=1.6, ms=3, label=r"$KL_{\mathrm{eq}}$ (norm.)")
        ax.plot(eps, _norm(pe), "s-", lw=1.6, ms=3, label=r"$\overline{\Delta D/(N\langle C\rangle)}$ (norm.)")
        ax.plot(eps, _norm(pk), "^-", lw=1.6, ms=3, label=r"$\max \Delta D/(N\langle C\rangle)$ (norm.)")
        ax.plot(eps, _norm(logz), "D-", lw=1.6, ms=3, label=r"$\log Z/N$ (norm.)")
        if np.isfinite(mc).any():
            ax.plot(eps, _norm(mc), "v-", lw=1.6, ms=3, color="C5", label=r"$\langle C\rangle$ (norm.)")

        _format_eps_axis(ax, eps)
        ax.set_xlabel(r"$\varepsilon_{12}$")
        ax.set_ylabel("each curve → [0,1] (argmin→0, argmax→1; same x of extrema)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.grid(True, alpha=0.3)

    if rows_sym is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
        _one_panel(
            axes[0], rows_sym, "Symmetric independent mutation",
            eps_star=eps_star_sym, eps_fit_kl=eps_fit_kl_sym,
        )
        _one_panel(
            axes[1], rows_not_sym, r"Coupled mutation ($u_x$, $v_x$)",
            eps_star=eps_star_not_sym, eps_fit_kl=eps_fit_kl_not_sym,
        )
        fig.suptitle(
            rf"Shape comparison ($N={N}$, $s_1={s1:g}$, $s_2={s2:g}$, $r={r:g}$)\n"
            rf"sym: $u=v=({u[0]:g},{u[1]:g})$  |  "
            rf"not_sym: $u_x=({ux[0]:g},{ux[1]:g})$, $v_x=({vx[0]:g},{vx[1]:g})$  |  "
            rf"each metric normalized separately",
            fontsize=9,
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        _one_panel(
            ax, rows_not_sym, r"Coupled mutation ($u_x$, $v_x$)",
            eps_star=eps_star_not_sym, eps_fit_kl=eps_fit_kl_not_sym,
        )
        fig.suptitle(
            rf"Shape comparison ($N={N}$, $s_1={s1:g}$, $s_2={s2:g}$)  "
            rf"$u=({u[0]:g},{u[1]:g})$, $v=({v[0]:g},{v[1]:g})$, "
            rf"$u_x=({ux[0]:g},{ux[1]:g})$, $v_x=({vx[0]:g},{vx[1]:g})$, $r={r:g}$",
            fontsize=9,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "log_Z_metrics_vs_eps12_normalized.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_raw_stacked(
    rows_not_sym: list[dict],
    rows_sym: list[dict] | None,
    out_dir: Path,
    *,
    N: int,
    s1: float,
    s2: float,
    u: list[float],
    v: list[float],
    ux: list[float],
    vx: list[float],
    r: float,
    eps_star_sym: float,
    eps_star_not_sym: float,
    eps_fit_kl_sym: float | None,
    eps_fit_kl_not_sym: float | None,
) -> None:
    """All metrics vs ε₁₂ in original units (one subplot per metric)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dip = {
        "symmetric": (eps_star_sym, eps_fit_kl_sym),
        "coupled": (eps_star_not_sym, eps_fit_kl_not_sym),
    }

    metric_specs = [
        ("kl_eq_bits", r"$KL_{\mathrm{eq}}$ [bits]", "C0"),
        ("path_efficiency", r"$\overline{\Delta D/(N\langle C\rangle)}$", "C2"),
        ("peak_eta", r"$\max \Delta D/(N\langle C\rangle)$", "C3"),
        ("log_Z_over_N", r"$\log Z/N$", "C4"),
        ("mean_c_cost", r"$\langle C\rangle$", "C5"),
    ]

    n_rows = len(metric_specs)
    n_cols = 2 if rows_sym is not None else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 2.4 * n_rows + 1), sharex=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for i, (key, ylab, col) in enumerate(metric_specs):
        col_pairs: list[tuple[list[dict], object, str]] = []
        if rows_sym is not None:
            col_pairs.append((rows_sym, axes[i, 0], "symmetric"))
        col_pairs.append((rows_not_sym, axes[i, 1 if rows_sym is not None else 0], "coupled"))

        for rows, ax, lab in col_pairs:
            eps = np.array([r["eps12"] for r in rows])
            o = np.argsort(eps)
            eps = eps[o]
            y = np.array([rows[j][key] for j in o], dtype=float)
            if not np.isfinite(y).any():
                ax.set_visible(False)
                continue
            est_star, fit_kl = dip[lab]
            _dip_vlines(ax, eps, eps_star_pred=est_star, eps_fit_kl=fit_kl)
            ax.plot(eps, y, "o-", color=col, lw=1.6, ms=4, label=ylab)
            if key == "log_Z_over_N":
                logz0 = float(rows[o[0]].get("log_Z0_over_N", np.nan))
                if np.isfinite(logz0):
                    ax.axhline(logz0, color=col, ls="--", lw=1.0, alpha=0.65,
                               label=rf"neutral $\log Z_0/N={logz0:.4g}$")
            _format_eps_axis(ax, eps)
            ax.set_ylabel(ylab, fontsize=9)
            ax.set_title(lab, fontsize=9)
            ax.legend(fontsize=7, loc="best")
            ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$\varepsilon_{12}$")

    fig.suptitle(
        rf"Raw metrics vs $\varepsilon_{{12}}$ ($N={N}$, $s_1={s1:g}$, $s_2={s2:g}$, $r={r:g}$)\n"
        rf"original units — not normalized",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "log_Z_metrics_vs_eps12_raw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_raw_overlay(
    rows_not_sym: list[dict],
    rows_sym: list[dict] | None,
    out_dir: Path,
    *,
    N: int,
    s1: float,
    s2: float,
    eps_star_sym: float,
    eps_star_not_sym: float,
    eps_fit_kl_sym: float | None,
    eps_fit_kl_not_sym: float | None,
) -> None:
    """All metrics on one axes per model (raw; scales differ — for shape only)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def _one_panel(
        ax,
        rows: list[dict],
        title: str,
        *,
        eps_star: float,
        eps_fit_kl: float | None,
    ) -> None:
        eps = np.array([r["eps12"] for r in rows])
        o = np.argsort(eps)
        eps = eps[o]
        _dip_vlines(ax, eps, eps_star_pred=eps_star, eps_fit_kl=eps_fit_kl)
        ax.plot(eps, [rows[i]["kl_eq_bits"] for i in o], "o-", lw=1.6, ms=3,
                label=r"$KL_{\mathrm{eq}}$ [bits]")
        ax.plot(eps, [rows[i]["path_efficiency"] for i in o], "s-", lw=1.6, ms=3,
                label=r"$\overline{\Delta D/(N\langle C\rangle)}$")
        ax.plot(eps, [rows[i]["peak_eta"] for i in o], "^-", lw=1.6, ms=3,
                label=r"$\max \Delta D/(N\langle C\rangle)$")
        ax.plot(eps, [rows[i]["log_Z_over_N"] for i in o], "D-", lw=1.6, ms=3,
                label=r"$\log Z/N$")
        mc = np.array([rows[i]["mean_c_cost"] for i in o])
        if np.isfinite(mc).any():
            ax.plot(eps, mc, "v-", lw=1.6, ms=3, color="C5", label=r"$\langle C\rangle$")
        _format_eps_axis(ax, eps)
        ax.set_xlabel(r"$\varepsilon_{12}$")
        ax.set_ylabel("raw value (mixed units)")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7, ncol=2, loc="best")
        ax.grid(True, alpha=0.3)

    if rows_sym is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        _one_panel(
            axes[0], rows_sym, "Symmetric — all metrics (raw)",
            eps_star=eps_star_sym, eps_fit_kl=eps_fit_kl_sym,
        )
        _one_panel(
            axes[1], rows_not_sym, "Coupled — all metrics (raw)",
            eps_star=eps_star_not_sym, eps_fit_kl=eps_fit_kl_not_sym,
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        _one_panel(
            ax, rows_not_sym, "Coupled — all metrics (raw)",
            eps_star=eps_star_not_sym, eps_fit_kl=eps_fit_kl_not_sym,
        )

    fig.suptitle(
        rf"All metrics overlay, original scale ($N={N}$, $s_1={s1:g}$, $s_2={s2:g}$)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "log_Z_metrics_vs_eps12_raw_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_all_plots(
    *,
    rows_not_sym: list[dict],
    rows_sym: list[dict] | None,
    out_dir: Path,
    N: int,
    s1: float,
    s2: float,
    u: list[float],
    v: list[float],
    ux: list[float],
    vx: list[float],
    r: float,
    eps_star_sym: float,
    eps_star_not_sym: float,
    eps_fit_kl_sym: float | None,
    eps_fit_kl_not_sym: float | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _plot_normalized(
        rows_not_sym, rows_sym, out_dir,
        N=N, s1=s1, s2=s2, u=u, v=v, ux=ux, vx=vx, r=r,
        eps_star_sym=eps_star_sym, eps_star_not_sym=eps_star_not_sym,
        eps_fit_kl_sym=eps_fit_kl_sym, eps_fit_kl_not_sym=eps_fit_kl_not_sym,
    )
    _plot_peak_eta_raw(
        rows_not_sym, rows_sym, out_dir,
        eps_star_sym=eps_star_sym, eps_star_not_sym=eps_star_not_sym,
        eps_fit_kl_sym=eps_fit_kl_sym, eps_fit_kl_not_sym=eps_fit_kl_not_sym,
    )
    _plot_mean_eta_raw(
        rows_not_sym, rows_sym, out_dir,
        eps_star_sym=eps_star_sym, eps_star_not_sym=eps_star_not_sym,
        eps_fit_kl_sym=eps_fit_kl_sym, eps_fit_kl_not_sym=eps_fit_kl_not_sym,
    )
    _plot_raw_stacked(
        rows_not_sym, rows_sym, out_dir,
        N=N, s1=s1, s2=s2, u=u, v=v, ux=ux, vx=vx, r=r,
        eps_star_sym=eps_star_sym, eps_star_not_sym=eps_star_not_sym,
        eps_fit_kl_sym=eps_fit_kl_sym, eps_fit_kl_not_sym=eps_fit_kl_not_sym,
    )
    _plot_raw_overlay(
        rows_not_sym, rows_sym, out_dir,
        N=N, s1=s1, s2=s2,
        eps_star_sym=eps_star_sym, eps_star_not_sym=eps_star_not_sym,
        eps_fit_kl_sym=eps_fit_kl_sym, eps_fit_kl_not_sym=eps_fit_kl_not_sym,
    )


def _replot_from_metrics(metrics_path: Path, out_dir: Path | None = None) -> None:
    payload = json.loads(metrics_path.read_text())
    out_dir = out_dir or metrics_path.parent
    rows_not_sym = payload["not_sym"]
    rows_sym = payload.get("sym")
    _write_all_plots(
        rows_not_sym=rows_not_sym,
        rows_sym=rows_sym,
        out_dir=out_dir,
        N=int(payload["N"]),
        s1=float(payload["s1"]),
        s2=float(payload["s2"]),
        u=[float(x) for x in payload["u"]],
        v=[float(x) for x in payload["v"]],
        ux=[float(x) for x in payload["ux"]],
        vx=[float(x) for x in payload["vx"]],
        r=float(payload["r"]),
        eps_star_sym=float(payload.get("eps_star_sym", payload.get("eps_star_pred", 0.0))),
        eps_star_not_sym=float(payload.get("eps_star_not_sym", payload.get("eps_star_pred", 0.0))),
        eps_fit_kl_sym=(
            float(payload["eps_fit_kl_sym"])
            if payload.get("eps_fit_kl_sym") is not None
            else float(payload.get("eps_fit_kl", float("nan")))
        ),
        eps_fit_kl_not_sym=(
            float(payload["eps_fit_kl_not_sym"])
            if payload.get("eps_fit_kl_not_sym") is not None
            else None
        ),
    )
    print(f"Replotted from {metrics_path} → {out_dir}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=SCRIPT_DIR / "eps12_not_sym_config.yaml")
    p.add_argument("--out-dir", type=Path, default=SCRIPT_DIR / "plots" / "eps12_not_sym")
    p.add_argument("--quick", action="store_true", help="coarser eps grid, shorter trajectories")
    p.add_argument(
        "--replot",
        type=Path,
        default=None,
        metavar="METRICS_JSON",
        help="regenerate plots from saved metrics.json (no chain recompute)",
    )
    args = p.parse_args()

    if args.replot is not None:
        _replot_from_metrics(args.replot.resolve(), args.out_dir.resolve())
        return

    cfg = load_config(args.config.resolve())
    N = int(cfg.get("N", 30))
    r = float(cfg.get("r", 0.0))
    u = [float(x) for x in np.atleast_1d(cfg.get("u", [0.001, 0.001]))]
    v = [float(x) for x in np.atleast_1d(cfg.get("v", u))]
    if len(u) == 1:
        u = [u[0], u[0]]
    if len(v) == 1:
        v = [v[0], v[0]]
    s1 = float(cfg.get("s1", -0.001))
    s2 = float(cfg.get("s2", 0.0005))
    coupled_factor = float(cfg.get("coupled_factor", 10.0))
    pergen_max_t = int(cfg.get("pergen_max_t", 500))
    compare_sym = bool(cfg.get("compare_symmetric", True))

    if args.quick:
        cfg = dict(cfg)
        cfg["eps12_step"] = float(cfg.get("eps12_step", 0.0004)) * 2
        cfg["pergen_max_t"] = min(pergen_max_t, 200)
        pergen_max_t = int(cfg["pergen_max_t"])

    eps_star_simple = float(-2.0 / 3.0 * (s1 + s2))

    ux_arr = np.atleast_1d(np.asarray(cfg.get("ux", coupled_factor * np.mean(u)), dtype=float))
    vx_arr = np.atleast_1d(np.asarray(cfg.get("vx", coupled_factor * np.mean(v)), dtype=float))
    if ux_arr.size == 1:
        ux = [float(ux_arr[0]), float(ux_arr[0])]
    else:
        ux = [float(ux_arr[0]), float(ux_arr[1])]
    if vx_arr.size == 1:
        vx = [float(vx_arr[0]), float(vx_arr[0])]
    else:
        vx = [float(vx_arr[0]), float(vx_arr[1])]

    M_not_sym = build_mutation_matrix(
        np.array(u), np.array(v), np.array(ux), np.array(vx),
    )
    params_not_sym0 = params_from_config_not_sym(
        _make_cfg(N=N, r=r, u=u, v=v, s1=s1, s2=s2, eps12=0.0, ux=ux, vx=vx,
                  coupled_factor=coupled_factor),
        SCRIPT_DIR,
    )
    phi_not_sym, _, fracs, _, _ = compute_phi(params_not_sym0, M_not_sym)
    eps_star_not_sym = _predict_eps_star(phi_not_sym, fracs, s1, s2)

    eps_star_sym = float("nan")
    M_sym = None
    if compare_sym:
        M_sym = mutation_matrix_asymmetric(np.array(u), np.array(v))
        params_sym0 = params_from_config_sym(
            _make_cfg(N=N, r=r, u=u, v=v, s1=s1, s2=s2, eps12=0.0),
            SCRIPT_DIR,
        )
        phi_sym, _, fracs_sym, _, _ = compute_phi(params_sym0, M_sym)
        eps_star_sym = _predict_eps_star(phi_sym, fracs_sym, s1, s2)

    eps_stars = [e for e in (eps_star_sym, eps_star_not_sym) if np.isfinite(e)]
    eps_grid = _build_eps_grid(cfg, eps_stars=eps_stars)
    print(
        f"  ε grid: {len(eps_grid)} points  "
        f"[{_fmt_eps(float(eps_grid[0]))}, {_fmt_eps(float(eps_grid[-1]))}]",
        flush=True,
    )
    print(
        f"  pred ε* (Cov/Var): sym={_fmt_eps(eps_star_sym)}  "
        f"coupled={_fmt_eps(eps_star_not_sym)}  "
        f"(ref −2/3(s1+s2)={_fmt_eps(eps_star_simple)})",
        flush=True,
    )

    print(
        f"Coupled-mutation ε₁₂ investigation: N={N}, s1={s1:g}, s2={s2:g}, "
        f"{len(eps_grid)} ε points, t=0..{pergen_max_t}",
        flush=True,
    )
    print_mutation_matrix_coupled(M_not_sym, u=u, v=v, ux=ux, vx=vx)

    rows_not_sym, _, _ = _run_sweep(
        label="not_sym",
        M_mut=M_not_sym,
        N=N, r=r, u=u, v=v, s1=s1, s2=s2,
        eps_grid=eps_grid,
        pergen_max_t=pergen_max_t,
        params_fn=params_from_config_not_sym,
    )

    rows_sym = None
    if compare_sym and M_sym is not None:
        print("\nSymmetric baseline (independent mutation):", flush=True)
        rows_sym, _, _ = _run_sweep(
            label="sym",
            M_mut=M_sym,
            N=N, r=r, u=u, v=v, s1=s1, s2=s2,
            eps_grid=eps_grid,
            pergen_max_t=pergen_max_t,
            params_fn=params_from_config_sym,
        )

    eps_fit_kl_sym = _fit_kl_min(rows_sym) if rows_sym else float("nan")
    eps_fit_kl_not_sym = _fit_kl_min(rows_not_sym)
    if rows_sym:
        print(
            f"\nDip markers: fit KL min sym={eps_fit_kl_sym:+.6g}  "
            f"coupled={eps_fit_kl_not_sym:+.6g}",
            flush=True,
        )

    def _json_default(obj):
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        raise TypeError(type(obj))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "N": N, "r": r, "u": u, "v": v, "ux": ux, "vx": vx,
        "s1": s1, "s2": s2, "coupled_factor": coupled_factor,
        "pergen_max_t": pergen_max_t,
        "eps_grid": eps_grid.tolist(),
        "eps_star_sym": float(eps_star_sym) if np.isfinite(eps_star_sym) else None,
        "eps_star_not_sym": float(eps_star_not_sym),
        "eps_star_pred": float(eps_star_sym) if np.isfinite(eps_star_sym) else float(eps_star_not_sym),
        "eps_fit_kl_sym": float(eps_fit_kl_sym) if rows_sym and np.isfinite(eps_fit_kl_sym) else None,
        "eps_fit_kl_not_sym": float(eps_fit_kl_not_sym) if np.isfinite(eps_fit_kl_not_sym) else None,
        "eps_fit_kl": float(eps_fit_kl_sym) if rows_sym and np.isfinite(eps_fit_kl_sym) else None,
        "eps_star_simple": eps_star_simple,
        "not_sym": rows_not_sym,
        "sym": rows_sym,
        "peak_eta_eps_not_sym": _peak_eps(
            np.array([r["eps12"] for r in rows_not_sym]),
            np.array([r["peak_eta"] for r in rows_not_sym]),
        ),
        "peak_eta_eps_sym": (
            _peak_eps(
                np.array([r["eps12"] for r in rows_sym]),
                np.array([r["peak_eta"] for r in rows_sym]),
            )
            if rows_sym else None
        ),
        "mean_eta_eps_not_sym": _peak_eps(
            np.array([r["eps12"] for r in rows_not_sym]),
            np.array([r["path_efficiency"] for r in rows_not_sym]),
        ),
        "mean_eta_min_eps_not_sym": _min_eps(
            np.array([r["eps12"] for r in rows_not_sym]),
            np.array([r["path_efficiency"] for r in rows_not_sym]),
        ),
        "mean_eta_eps_sym": (
            _peak_eps(
                np.array([r["eps12"] for r in rows_sym]),
                np.array([r["path_efficiency"] for r in rows_sym]),
            )
            if rows_sym else None
        ),
        "mean_eta_min_eps_sym": (
            _min_eps(
                np.array([r["eps12"] for r in rows_sym]),
                np.array([r["path_efficiency"] for r in rows_sym]),
            )
            if rows_sym else None
        ),
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, default=_json_default),
    )

    _write_all_plots(
        rows_not_sym=rows_not_sym,
        rows_sym=rows_sym,
        out_dir=args.out_dir,
        N=N, s1=s1, s2=s2, u=u, v=v, ux=ux, vx=vx, r=r,
        eps_star_sym=float(eps_star_sym) if np.isfinite(eps_star_sym) else float(eps_star_not_sym),
        eps_star_not_sym=float(eps_star_not_sym),
        eps_fit_kl_sym=float(eps_fit_kl_sym) if rows_sym and np.isfinite(eps_fit_kl_sym) else None,
        eps_fit_kl_not_sym=float(eps_fit_kl_not_sym) if np.isfinite(eps_fit_kl_not_sym) else None,
    )

    pk_ns = payload["peak_eta_eps_not_sym"]
    pk_s = payload["peak_eta_eps_sym"]
    mk_ns = payload["mean_eta_eps_not_sym"]
    mk_min_ns = payload["mean_eta_min_eps_not_sym"]
    mk_s = payload["mean_eta_eps_sym"]
    print(f"\nPeak η at ε = {pk_ns:+.6g} (coupled)", flush=True)
    if rows_sym:
        print(f"Peak η at ε = {pk_s:+.6g} (symmetric)", flush=True)
    print(f"Mean η argmax at ε = {mk_ns:+.6g} (coupled)", flush=True)
    print(f"Mean η argmin at ε = {mk_min_ns:+.6g} (coupled)", flush=True)
    if rows_sym:
        print(f"Mean η argmax at ε = {mk_s:+.6g} (symmetric)", flush=True)
    print(f"wrote {args.out_dir / 'log_Z_metrics_vs_eps12_normalized.png'}", flush=True)
    print(f"wrote {args.out_dir / 'peak_eta_vs_eps12.png'}", flush=True)
    print(f"wrote {args.out_dir / 'mean_eta_vs_eps12.png'}", flush=True)
    print(f"wrote {args.out_dir / 'log_Z_metrics_vs_eps12_raw.png'}", flush=True)
    print(f"wrote {args.out_dir / 'log_Z_metrics_vs_eps12_raw_overlay.png'}", flush=True)
    print(f"wrote {args.out_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
