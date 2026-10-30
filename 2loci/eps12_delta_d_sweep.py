#!/usr/bin/env python3
r"""
Sweep eps12 (epistasis) with fixed N, u, v, r, s1, s2.

For each eps12, track per-generation
  D(t)   = KL(psi(t) || phi*)
  C(t)   = <C(x)>_psi
  ΔD     = D(t+1) - D(t)
  ΔD/(N C(t))
  F1, F2, F3 and F2/F3 decomposition:
    F1 = <W>_psi + H(psi)/N
    F2 = <ln(x·ω)>_psi - (1/N) KL(psi||phi*)     [KL in nats]
    F3 = <ln(x·ω)>_psi - (1/N) KL(psi||q_cond)   [q_cond = cond-α ref]
    F4 = -(1/N) D_KL(psi||psi*)                    [nats]

Curves are colored by the equilibrium **information score**
  KL_eq = KL(psi* || phi*)  [bits]
at the selection equilibrium psi*.

Run:
  cd 2loci
  python eps12_delta_d_sweep.py --config eps12_delta_d_config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_sigma_dynamics import (  # noqa: E402
    PERGEN_MAX_T,
    cond_alpha_reference_pmf,
    cost_per_state,
    expectation,
    free_f2,
    free_f3,
    free_f4,
    kl_divergence_nats,
)
from simplex import (  # noqa: E402
    build_sigma,
    build_transition_matrix_row_stochastic,
    column_stochastic,
    compute_phi,
    fitness_exp,
    free_fitness_1,
    kl_divergence_bits,
    load_config,
    log_mean_fitness_genotype_simplex,
    mutation_matrix_asymmetric,
    params_from_config,
    try_selection_equilibrium,
    w_potential_per_state,
)

OUT_DIR = SCRIPT_DIR / "plots" / "eps12_delta_d"
SCORE_LABEL = r"$\mathrm{KL}_{\mathrm{eq}}(\psi^*\,\|\,\varphi)$ [bits]"


def _format_tick_labels(vals: np.ndarray, *, signed: bool = False) -> list[str]:
    """Adaptive precision so closely spaced colorbar ticks stay distinct."""
    vals = np.asarray(vals, dtype=float)
    if vals.size == 0:
        return []
    span = float(np.ptp(vals)) if vals.size > 1 else max(abs(float(vals[0])), 1e-15)
    mag = float(np.max(np.abs(vals))) if vals.size else span

    def _one(v: float) -> str:
        if span < 1e-5 or mag < 5e-4:
            return f"{v:+.2e}" if signed else f"{v:.2e}"
        if span < 1e-3:
            return f"{v:+.7f}" if signed else f"{v:.7f}"
        if span < 0.01:
            return f"{v:+.5f}" if signed else f"{v:.5f}"
        if span < 0.1:
            return f"{v:+.4f}" if signed else f"{v:.4f}"
        return f"{v:+.3f}" if signed else f"{v:.3f}"

    return [_one(v) for v in vals]


def _format_scalar(v: float, *, signed: bool = False, ref_span: float | None = None) -> str:
    span = ref_span if ref_span is not None else abs(v)
    return _format_tick_labels(np.array([0.0, span if span > 0 else abs(v) or 1e-6]), signed=signed)[-1]


@dataclass
class Eps12Run:
    eps12: float
    kl_eq: float
    f1_eq: float
    f2_eq: float
    f3_eq: float
    wstar_eq: float
    pergen_times: np.ndarray
    pergen_kl_phi_bits: np.ndarray
    pergen_c_cost: np.ndarray
    pergen_f1: np.ndarray
    pergen_f2: np.ndarray
    pergen_f3: np.ndarray
    pergen_f4: np.ndarray
    pergen_mean_wstar: np.ndarray
    pergen_info_f2: np.ndarray   # KL(psi||phi*)/N [nats]
    pergen_info_f3: np.ndarray   # KL(psi||q_cond)/N [nats]


@dataclass
class Eps12Metrics:
    eps12: float
    kl_eq: float
    f1_eq: float
    f2_eq: float
    f3_eq: float
    wstar_eq: float
    path_efficiency: float
    peak_eta: float
    t_peak_eta: float
    t_half: float
    t_90: float
    t_95: float
    t_wstar_90: float
    d_final: float
    frac_equilibrium: float
    sum_delta_d: float
    sum_nc: float
    frac_info_driven_steps: float  # frac steps with ΔF2>0 and Δw* > Δ(KL/N)
    frac_info_driven_after_wstar_90: float  # same, after t90(w*)
    frac_f2_down_wstar_flat: float  # ΔF2<0 while ⟨w*⟩ flat (KL still rising)


def _eps12_grid(cfg: dict) -> np.ndarray:
    if "eps12_values" in cfg:
        return np.sort(np.unique(np.round(np.asarray(cfg["eps12_values"], dtype=float), 12)))
    lo = float(cfg.get("eps12_min", -0.5))
    hi = float(cfg.get("eps12_max", 0.5))
    step = float(cfg.get("eps12_step", 0.05))
    n = int(round((hi - lo) / step)) + 1
    grid = lo + step * np.arange(n)
    return np.unique(np.round(np.clip(grid, lo, hi), 10))


def _make_cfg(
    *,
    N: int,
    r: float,
    u: float,
    v: float,
    s1: float,
    s2: float,
    eps12: float,
) -> dict:
    return {
        "_config_dir": str(SCRIPT_DIR),
        "N": N,
        "r": float(r),
        "u": [float(u), float(u)],
        "v": [float(v), float(v)],
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
    }


def _score_colors(runs: list[Eps12Run]) -> tuple[list, object, object, np.ndarray | None]:
    """Turbo colors by rank of equilibrium KL_eq (matches eps12_score_vs_speed)."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    sc = np.array([r.kl_eq for r in runs if np.isfinite(r.kl_eq)], dtype=float)
    runs_f = [r for r in runs if np.isfinite(r.kl_eq)]
    cmap = plt.get_cmap("turbo")
    n = len(sc)
    if n == 0:
        return [], cmap, mpl.colors.Normalize(vmin=0, vmax=1), None
    if n == 1:
        return [cmap(0.5)], cmap, mpl.colors.Normalize(vmin=sc[0], vmax=sc[0]), np.array([sc[0]])

    order = np.argsort(sc)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(n)
    colors = [cmap(ranks[i] / (n - 1)) for i in range(n)]
    sorted_sc = sc[order]
    mids = (sorted_sc[:-1] + sorted_sc[1:]) / 2.0
    lo = sorted_sc[0] - (mids[0] - sorted_sc[0] if mids.size else 0.5)
    hi = sorted_sc[-1] + (sorted_sc[-1] - mids[-1] if mids.size else 0.5)
    bounds = np.concatenate([[lo], mids, [hi]])
    cbar_cmap = mpl.colors.ListedColormap([cmap(i / (n - 1)) for i in range(n)])
    cbar_norm = mpl.colors.BoundaryNorm(bounds, n)
    step = max(1, n // 12)
    return colors, cbar_cmap, cbar_norm, sorted_sc[::step]


def _eps12_sign_linestyle(eps12: float, tol: float = 1e-12) -> str:
    """Line style encodes sign of eps12 (color still = KL_eq)."""
    if eps12 > tol:
        return "-"
    if eps12 < -tol:
        return "--"
    return ":"


def _add_eps12_sign_legend(ax) -> None:
    """Second legend: curve linestyle = sign(eps12)."""
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color="0.25", ls="-", lw=1.6, label=r"$\varepsilon_{12} > 0$"),
        Line2D([0], [0], color="0.25", ls="--", lw=1.6, label=r"$\varepsilon_{12} < 0$"),
        Line2D([0], [0], color="0.25", ls=":", lw=1.6, label=r"$\varepsilon_{12} = 0$"),
    ]
    leg = ax.legend(
        handles=handles,
        title=r"sign of $\varepsilon_{12}$",
        loc="upper right",
        fontsize=8,
        title_fontsize=8,
        framealpha=0.92,
    )
    ax.add_artist(leg)


def _add_score_colorbar(fig, ax, cbar_cmap, cbar_norm, tick_vals: np.ndarray | None) -> None:
    import matplotlib as mpl

    sm = mpl.cm.ScalarMappable(cmap=cbar_cmap, norm=cbar_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    if tick_vals is not None:
        tick_vals = np.asarray(tick_vals, dtype=float)
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels(_format_tick_labels(tick_vals))
    cbar.set_label(SCORE_LABEL)
    cbar.ax.tick_params(labelsize=7)


def _add_eps12_colorbar(fig, ax, cmap, norm) -> None:
    """Continuous colorbar for eps12 with enough digits near zero."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    span = float(norm.vmax - norm.vmin)
    cbar.locator = MaxNLocator(nbins=10, min_n_ticks=6)
    cbar.update_ticks()

    def _fmt(x: float, _pos: int) -> str:
        return _format_scalar(x, signed=True, ref_span=span)

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
    cbar.set_label(r"$\varepsilon_{12}$")
    cbar.ax.tick_params(labelsize=7)


def track_d_c_f_per_generation(
    psi0: np.ndarray,
    P_col: np.ndarray,
    phi: np.ndarray,
    c_state: np.ndarray,
    fracs: np.ndarray,
    w_ps: np.ndarray,
    wstar_ln: np.ndarray,
    q_cond: np.ndarray,
    psi_star: np.ndarray,
    N: int,
    max_t: int,
) -> tuple[np.ndarray, ...]:
    """Per-generation D, C, F1–F4 and F2/F3 decomposition (KL terms in nats)."""
    n = max_t + 1
    d = np.empty(n)
    c = np.empty(n)
    f1 = np.empty(n)
    f2 = np.empty(n)
    f3 = np.empty(n)
    f4 = np.empty(n)
    mean_wstar = np.empty(n)
    info_f2 = np.empty(n)
    info_f3 = np.empty(n)

    psi = psi0.copy()
    inv_n = 1.0 / float(N)

    def _record() -> None:
        d[idx] = kl_divergence_bits(psi, phi)
        c[idx] = expectation(psi, c_state)
        mean_wstar[idx] = expectation(psi, wstar_ln)
        kl_phi_nats = kl_divergence_nats(psi, phi)
        kl_q_nats = kl_divergence_nats(psi, q_cond)
        info_f2[idx] = kl_phi_nats * inv_n
        info_f3[idx] = kl_q_nats * inv_n
        f1[idx] = free_fitness_1(psi, w_ps, N)
        f2[idx] = mean_wstar[idx] - info_f2[idx]
        f3[idx] = mean_wstar[idx] - info_f3[idx]
        f4[idx] = free_f4(psi, psi_star, N)

    idx = 0
    _record()
    for _ in range(max_t):
        psi = P_col @ psi
        s = float(psi.sum())
        if s > 0:
            psi /= s
        idx += 1
        _record()

    return (
        np.arange(n, dtype=float), d, c, f1, f2, f3, f4,
        mean_wstar, info_f2, info_f3,
    )


def run_one_eps12(
    eps12: float,
    *,
    N: int,
    r: float,
    u: float,
    v: float,
    s1: float,
    s2: float,
    phi: np.ndarray,
    fracs: np.ndarray,
    M_mut: np.ndarray,
    q_cond: np.ndarray,
    pergen_max_t: int,
    verbose: bool,
) -> Eps12Run:
    params = params_from_config(
        _make_cfg(N=N, r=r, u=u, v=v, s1=s1, s2=s2, eps12=eps12),
        SCRIPT_DIR,
    )
    omega = fitness_exp(params.Sigma)
    c_state = cost_per_state(fracs, omega)
    w_ps = w_potential_per_state(fracs, omega, M_mut, N)
    wstar_ln = log_mean_fitness_genotype_simplex(fracs, omega)

    T_row, _, _ = build_transition_matrix_row_stochastic(
        N, omega, M_mut, r, n_jobs=-1,
        label=f"eps{eps12:.3g}"[:20], verbose=False,
    )
    P_col = column_stochastic(T_row)
    psi_star, resid, ok = try_selection_equilibrium(P_col)
    if not ok or psi_star is None:
        raise RuntimeError(f"eps12={eps12}: psi* not found (residual={resid})")
    kl_eq = kl_divergence_bits(psi_star, phi)
    wstar_eq = expectation(psi_star, wstar_ln)
    f1_eq = free_fitness_1(psi_star, w_ps, N)
    f2_eq = free_f2(psi_star, wstar_ln, phi, N)
    f3_eq = free_f3(psi_star, wstar_ln, q_cond, N)

    pg = track_d_c_f_per_generation(
        phi.copy(), P_col, phi, c_state, fracs, w_ps, wstar_ln, q_cond, psi_star, N, pergen_max_t,
    )
    pg_t, pg_d, pg_c, pg_f1, pg_f2, pg_f3, pg_f4, pg_w, pg_i2, pg_i3 = pg
    if verbose:
        print(
            f"  eps12={eps12:+.3f}  KL_eq={kl_eq:.4f} bits  "
            f"F2_eq={f2_eq:.5f}  w*_eq={wstar_eq:.5f}",
            flush=True,
        )
    return Eps12Run(
        eps12=float(eps12),
        kl_eq=float(kl_eq),
        f1_eq=float(f1_eq),
        f2_eq=float(f2_eq),
        f3_eq=float(f3_eq),
        wstar_eq=float(wstar_eq),
        pergen_times=pg_t,
        pergen_kl_phi_bits=pg_d,
        pergen_c_cost=pg_c,
        pergen_f1=pg_f1,
        pergen_f2=pg_f2,
        pergen_f3=pg_f3,
        pergen_f4=pg_f4,
        pergen_mean_wstar=pg_w,
        pergen_info_f2=pg_i2,
        pergen_info_f3=pg_i3,
    )


def analyze_run(run: Eps12Run, N: int) -> Eps12Metrics:
    """Path-normalized efficiency and convergence-time metrics for one eps12."""
    d = run.pergen_kl_phi_bits
    c = run.pergen_c_cost
    f2 = run.pergen_f2
    wstar = run.pergen_mean_wstar
    info_f2 = run.pergen_info_f2
    t = run.pergen_times
    delta_d = np.diff(d)
    delta_f2 = np.diff(f2)
    delta_wstar = np.diff(wstar)
    delta_info = np.diff(info_f2)  # KL/N change; ≥0 along φ→ψ* path
    nc = N * c[1:]
    sum_nc = float(np.sum(nc))
    sum_dd = float(d[-1] - d[0])
    path_eff = sum_dd / sum_nc if sum_nc > 1e-300 else float("nan")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(nc > 1e-300, delta_d / nc, np.nan)
    i_peak = int(np.nanargmax(ratio))
    peak_eta = float(ratio[i_peak])
    t_peak = float(t[i_peak + 1])

    kl_eq = run.kl_eq
    wstar_eq = run.wstar_eq

    def _time_to_fraction(frac: float) -> float:
        if kl_eq < 1e-10:
            return 0.0 if d[-1] < 1e-10 else float("nan")
        target = frac * kl_eq
        hit = np.where(d >= target - 1e-12)[0]
        return float(t[hit[0]]) if hit.size else float("nan")

    def _time_wstar_fraction(frac: float) -> float:
        if abs(wstar_eq) < 1e-14:
            return 0.0
        target = frac * wstar_eq
        hit = np.where(wstar >= target - 1e-12)[0]
        return float(t[hit[0]]) if hit.size else float("nan")

    frac_eq = float(d[-1] / kl_eq) if kl_eq > 1e-10 else (1.0 if d[-1] < 1e-10 else float("nan"))
    t_w90 = _time_wstar_fraction(0.9)

    # F2 = ⟨w*⟩ − KL/N. Along φ→ψ*, KL/N rises; ΔF2>0 only when Δw* > Δ(KL/N).
    wstar_scale = max(abs(wstar_eq), 1e-12)
    wstar_flat = np.abs(delta_wstar) < 1e-3 * wstar_scale
    f2_up = delta_f2 > 1e-12
    wstar_wins = delta_wstar > delta_info + 1e-12
    wstar_driven = f2_up & wstar_wins
    f2_down_flat = (delta_f2 < -1e-12) & wstar_flat
    frac_wstar_driven = float(np.mean(wstar_driven)) if wstar_driven.size else float("nan")
    frac_f2_down_flat = float(np.mean(f2_down_flat)) if f2_down_flat.size else float("nan")
    if np.isfinite(t_w90):
        after = t[1:] >= t_w90
        frac_wstar_after = (
            float(np.mean(wstar_driven[after])) if np.any(after) else float("nan")
        )
        frac_down_after = (
            float(np.mean(f2_down_flat[after])) if np.any(after) else float("nan")
        )
    else:
        frac_wstar_after = float("nan")
        frac_down_after = float("nan")

    return Eps12Metrics(
        eps12=run.eps12,
        kl_eq=kl_eq,
        f1_eq=run.f1_eq,
        f2_eq=run.f2_eq,
        f3_eq=run.f3_eq,
        wstar_eq=wstar_eq,
        path_efficiency=path_eff,
        peak_eta=peak_eta,
        t_peak_eta=t_peak,
        t_half=_time_to_fraction(0.5),
        t_90=_time_to_fraction(0.9),
        t_95=_time_to_fraction(0.95),
        t_wstar_90=t_w90,
        d_final=float(d[-1]),
        frac_equilibrium=frac_eq,
        sum_delta_d=sum_dd,
        sum_nc=sum_nc,
        frac_info_driven_steps=frac_wstar_driven,
        frac_info_driven_after_wstar_90=frac_wstar_after,
        frac_f2_down_wstar_flat=frac_f2_down_flat,
    )


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x[mask], y[mask]).correlation)
    except ImportError:
        rx = np.argsort(np.argsort(x[mask])).astype(float)
        ry = np.argsort(np.argsort(y[mask])).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])


def _filter_branch(items: list, branch: str) -> list:
    if branch == "pos":
        return [x for x in items if x.eps12 > 1e-12]
    if branch == "neg":
        return [x for x in items if x.eps12 < -1e-12]
    raise ValueError(f"branch must be 'pos' or 'neg', got {branch!r}")


BRANCH_TITLES = {
    "pos": r"$\varepsilon_{12} > 0$ (positive epistasis)",
    "neg": r"$\varepsilon_{12} < 0$ (negative epistasis)",
}


def _write_metrics_csv(path: Path, metrics: list[Eps12Metrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "eps12", "kl_eq", "f1_eq", "f2_eq", "f3_eq", "wstar_eq",
        "path_efficiency", "peak_eta", "t_peak_eta",
        "t_half", "t_90", "t_95", "t_wstar_90",
        "d_final", "frac_equilibrium",
        "sum_delta_d", "sum_nc",
        "frac_info_driven_steps", "frac_info_driven_after_wstar_90",
        "frac_f2_down_wstar_flat",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in metrics:
            w.writerow({k: getattr(m, k) for k in fields})
    print(f"  wrote {path}", flush=True)


def plot_efficiency_convergence_investigation(
    metrics: list[Eps12Metrics],
    out_dir: Path,
    *,
    N: int,
    u: float,
    s1: float,
    s2: float,
    r: float,
    pergen_max_t: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    kl = np.array([m.kl_eq for m in metrics])
    path_eff = np.array([m.path_efficiency for m in metrics])
    t90 = np.array([m.t_90 for m in metrics])
    t50 = np.array([m.t_half for m in metrics])
    peak = np.array([m.peak_eta for m in metrics])
    frac = np.array([m.frac_equilibrium for m in metrics])

    rho_pe = _spearman(kl, path_eff)
    rho_t90 = _spearman(kl, t90)
    rho_t50 = _spearman(kl, t50)
    rho_peak = _spearman(kl, peak)

    summary = {
        "spearman_kl_eq_vs_path_efficiency": rho_pe,
        "spearman_kl_eq_vs_t_90": rho_t90,
        "spearman_kl_eq_vs_t_half": rho_t50,
        "spearman_kl_eq_vs_peak_eta": rho_peak,
        "pergen_max_t": pergen_max_t,
    }
    (out_dir / "efficiency_convergence_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"  Spearman KL_eq vs path_eff={rho_pe:.3f}  vs t_90={rho_t90:.3f}  "
        f"vs t_50={rho_t50:.3f}  vs peak_eta={rho_peak:.3f}",
        flush=True,
    )

    def _color_scatter(ax, xvals, yvals, xlabel, ylabel, title, rho: float):
        for m in metrics:
            if m.eps12 > 1e-12:
                c, mk = "C3", "o"
            elif m.eps12 < -1e-12:
                c, mk = "C0", "s"
            else:
                c, mk = "0.45", "^"
            xv = getattr(m, xvals) if isinstance(xvals, str) else xvals(m)
            yv = getattr(m, yvals) if isinstance(yvals, str) else yvals(m)
            ax.scatter(xv, yv, c=c, s=58, alpha=0.9, marker=mk)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nSpearman $\\rho={rho:.2f}$", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    _color_scatter(
        axes[0, 0], "kl_eq", "path_efficiency",
        r"$\mathrm{KL}_{\mathrm{eq}}(\psi^*\|\varphi)$ [bits]",
        r"$\bar{\eta} = \sum_t \Delta D \;/\; \sum_t N C(t)$",
        "Path-normalized efficiency", rho_pe,
    )
    _color_scatter(
        axes[0, 1], "kl_eq", "t_90",
        r"$\mathrm{KL}_{\mathrm{eq}}$ [bits]",
        r"$t_{90}$: first $t$ with $D(t)\geq 0.9\,\mathrm{KL}_{\mathrm{eq}}$",
        "Convergence time vs equilibrium information", rho_t90,
    )
    _color_scatter(
        axes[1, 0], "kl_eq", "peak_eta",
        r"$\mathrm{KL}_{\mathrm{eq}}$ [bits]",
        r"$\max_t \Delta D/(N C(t))$",
        "Peak instantaneous efficiency", rho_peak,
    )
    _color_scatter(
        axes[1, 1], "kl_eq", "frac_equilibrium",
        r"$\mathrm{KL}_{\mathrm{eq}}$ [bits]",
        rf"$D(t={pergen_max_t})/\mathrm{{KL}}_{{\mathrm{{eq}}}}$",
        rf"Fraction of equilibrium reached by $t={pergen_max_t}$", float("nan"),
    )
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C3", markersize=8, label=r"$\varepsilon_{12}>0$"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="C0", markersize=8, label=r"$\varepsilon_{12}<0$"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="0.45", markersize=8, label=r"$\varepsilon_{12}=0$"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        rf"Efficiency \& convergence vs $\mathrm{{KL}}_{{\mathrm{{eq}}}}$ "
        rf"($N={N}$, $u=v={u:g}$, $r={r:g}$, $s_1={s1:g}$, $s_2={s2:g}$)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = out_dir / "efficiency_convergence_vs_kl_eq.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}", flush=True)

    # eps12 as x-axis (dual metrics)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    eps = np.array([m.eps12 for m in metrics])
    order = np.argsort(eps)
    ax1.plot(eps[order], path_eff[order], "o-", color="C2", lw=1.5, ms=5, label=r"$\bar{\eta}$ (path efficiency)")
    ax2 = ax1.twinx()
    ax2.plot(eps[order], t90[order], "s--", color="C4", lw=1.5, ms=5, label=r"$t_{90}$")
    ax1.set_xlabel(r"$\varepsilon_{12}$")
    ax1.set_ylabel(r"$\bar{\eta} = \sum \Delta D / \sum N C$", color="C2")
    ax2.set_ylabel(r"$t_{90}$ [generations]", color="C4")
    ax1.tick_params(axis="y", labelcolor="C2")
    ax2.tick_params(axis="y", labelcolor="C4")
    ax1.grid(True, alpha=0.3)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="upper center", fontsize=9)
    ax1.set_title(
        rf"Path efficiency and convergence time across $\varepsilon_{{12}}$ "
        rf"($t \leq {pergen_max_t}$; NaN = not reached)",
        fontsize=10,
    )
    fig.tight_layout()
    p2 = out_dir / "efficiency_and_t90_vs_eps12.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p2}", flush=True)


def plot_free_fitness_vs_kl_eq(
    metrics: list[Eps12Metrics],
    out_dir: Path,
    *,
    N: int,
    u: float,
    s1: float,
    s2: float,
    r: float,
) -> None:
    """Equilibrium F1,F2,F3 at psi* vs KL_eq — test dip vs free fitness."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    kl = np.array([m.kl_eq for m in metrics])
    f1 = np.array([m.f1_eq for m in metrics])
    f2 = np.array([m.f2_eq for m in metrics])
    f3 = np.array([m.f3_eq for m in metrics])
    eps = np.array([m.eps12 for m in metrics])

    rho_f1 = _spearman(kl, f1)
    rho_f2 = _spearman(kl, f2)
    rho_f3 = _spearman(kl, f3)

    panels = [
        (f1, r"$F_1(\psi^*)$", rho_f1),
        (f2, r"$F_2(\psi^*)$", rho_f2),
        (f3, r"$F_3(\psi^*)$", rho_f3),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (y, ylab, rho) in zip(axes, panels):
        for m in metrics:
            if m.eps12 > 1e-12:
                c, mk = "C3", "o"
            elif m.eps12 < -1e-12:
                c, mk = "C0", "s"
            else:
                c, mk = "C2", "^"
            yv = m.f1_eq if ylab.startswith("$F_1") else (m.f2_eq if ylab.startswith("$F_2") else m.f3_eq)
            ax.scatter(m.kl_eq, yv, c=c, s=55, marker=mk, edgecolors="white", linewidths=0.4)
        ax.set_xlabel(r"$\mathrm{KL}_{\mathrm{eq}}(\psi^*\|\varphi)$ [bits]")
        ax.set_ylabel(ylab)
        ax.set_title(f"{ylab} vs $\\mathrm{{KL}}_{{\\mathrm{{eq}}}}$ ($\\rho={rho:.2f}$)", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        rf"Equilibrium free fitness at $\psi^*$ ($N={N}$, $s_1={s1:g}$, $s_2={s2:g}$, $r={r:g}$)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = out_dir / "free_fitness_vs_kl_eq.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}", flush=True)

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    o = np.argsort(eps)
    ax1.plot(eps[o], f1[o], "o-", label=r"$F_1(\psi^*)$", color="C0")
    ax1.plot(eps[o], f2[o], "s-", label=r"$F_2(\psi^*)$", color="C1")
    ax1.plot(eps[o], f3[o], "^-", label=r"$F_3(\psi^*)$", color="C2")
    ax2 = ax1.twinx()
    ax2.plot(eps[o], kl[o], "d--", color="C4", alpha=0.8, label=r"$\mathrm{KL}_{\mathrm{eq}}$")
    ax1.set_xlabel(r"$\varepsilon_{12}$")
    ax1.set_ylabel(r"free fitness at $\psi^*$ [nats]")
    ax2.set_ylabel(r"$\mathrm{KL}_{\mathrm{eq}}$ [bits]", color="C4")
    ax2.tick_params(axis="y", labelcolor="C4")
    ax1.grid(True, alpha=0.3)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="best", fontsize=8)
    ax1.set_title(r"$F_1,F_2,F_3$ and $\mathrm{KL}_{\mathrm{eq}}$ across $\varepsilon_{12}$", fontsize=10)
    fig.tight_layout()
    p2 = out_dir / "free_fitness_vs_eps12.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p2}", flush=True)

    summary = {
        "spearman_kl_eq_vs_f1_eq": float(rho_f1),
        "spearman_kl_eq_vs_f2_eq": float(rho_f2),
        "spearman_kl_eq_vs_f3_eq": float(rho_f3),
    }
    (out_dir / "free_fitness_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"  Spearman KL_eq vs F1_eq={rho_f1:.3f}  "
        f"F2_eq={rho_f2:.3f}  F3_eq={rho_f3:.3f}",
        flush=True,
    )


def _pick_eps(runs: list[Eps12Run], target: float) -> float | None:
    if not runs:
        return None
    eps_vals = [r.eps12 for r in runs]
    if target in eps_vals:
        return target
    return min(eps_vals, key=lambda e: abs(e - target))


def plot_f2_decomposition(
    runs: list[Eps12Run],
    metrics: list[Eps12Metrics],
    out_dir: Path,
    *,
    N: int,
    s1: float,
    s2: float,
    r: float,
    t_plot_max: int = 150,
) -> None:
    r"""F2 = ⟨ln W*⟩ − (1/N)KL(ψ||φ*): when KL term drives F2 while W* is flat."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    eps_pick: list[float] = []
    z = _pick_eps(runs, 0.0)
    if z is not None:
        eps_pick.append(z)
    for target in (-0.2, 0.2):
        e = _pick_eps(runs, target)
        if e is not None and e not in eps_pick:
            eps_pick.append(e)
    run_by_eps = {r.eps12: r for r in runs}

    n = len(eps_pick)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.2 * n), squeeze=False)
    for row, eps in enumerate(eps_pick):
        run = run_by_eps[eps]
        t = run.pergen_times
        tmax = min(t_plot_max, len(t) - 1)
        sl = slice(0, tmax + 1)
        w = run.pergen_mean_wstar[sl]
        neg_info = -run.pergen_info_f2[sl]
        f2 = run.pergen_f2[sl]

        ax = axes[row, 0]
        ax.plot(t[sl], w, "-", color="C0", lw=1.6)
        ax.set_ylabel("[nats]")
        ax.set_title(rf"$\varepsilon_{{12}}={eps:+.2f}$  $\langle W^*\rangle$", fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[row, 1]
        ax.plot(t[sl], neg_info, "-", color="C3", lw=1.6)
        ax.set_ylabel("[nats]")
        ax.set_title(r"information term ($-$KL$/N$)", fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[row, 2]
        ax.plot(t[sl], f2, "-", color="C2", lw=1.6, label=r"$F_2$")
        ax.plot(t[sl], w, "--", color="C0", alpha=0.45, lw=1.0, label=r"$\langle W^*\rangle$")
        ax.plot(t[sl], neg_info, "--", color="C3", alpha=0.45, lw=1.0, label=r"$-\mathrm{KL}/N$")
        ax.set_ylabel("[nats]")
        ax.set_title(r"$F_2$ and components", fontsize=9)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("generation $t$")
    fig.suptitle(
        rf"$F_2 = \langle\ln(x\!\cdot\!\omega)\rangle_{{\psi}} - \frac{{1}}{{N}}KL(\psi\|\varphi^*)$ "
        rf"($N={N}$, $s_1={s1:g}$, $s_2={s2:g}$, $r={r:g}$)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out_dir / "f2_decomposition_selected.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}", flush=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in metrics:
        yv = (
            m.frac_info_driven_after_wstar_90
            if np.isfinite(m.frac_info_driven_after_wstar_90)
            else m.frac_info_driven_steps
        )
        if not np.isfinite(yv):
            continue
        if m.eps12 > 1e-12:
            c, mk = "C3", "o"
        elif m.eps12 < -1e-12:
            c, mk = "C0", "s"
        else:
            c, mk = "C2", "^"
        ax.scatter(m.kl_eq, yv, c=c, s=58, marker=mk)
    ax.set_xlabel(r"$\mathrm{KL}_{\mathrm{eq}}$ [bits]")
    ax.set_ylabel(r"frac.\ $\Delta F_2>0$ with $\Delta w^* > \Delta(\mathrm{KL}/N)$")
    ax.set_title(
        r"Fitness-driven $F_2$ increases (KL$/N$ rises monotonically on $\varphi\to\psi^*$ path)",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p2 = out_dir / "f2_info_driven_fraction.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p2}", flush=True)


def plot_f123_trajectories(
    runs: list[Eps12Run],
    out_path: Path,
    *,
    N: int,
    u: float,
    s1: float,
    s2: float,
    r: float,
    branch_title: str | None = None,
    show_sign_legend: bool = True,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors, cbar_cmap, cbar_norm, cbar_ticks = _score_colors(runs)
    runs_f = [r for r in runs if np.isfinite(r.kl_eq)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for run, color in zip(runs_f, colors):
        ls = "-" if not show_sign_legend else _eps12_sign_linestyle(run.eps12)
        axes[0, 0].plot(run.pergen_times, run.pergen_f1, ls, lw=1.2, color=color, alpha=0.9)
        axes[0, 1].plot(run.pergen_times, run.pergen_f2, ls, lw=1.2, color=color, alpha=0.9)
        axes[1, 0].plot(run.pergen_times, run.pergen_f3, ls, lw=1.2, color=color, alpha=0.9)
        axes[1, 1].plot(run.pergen_times, run.pergen_f4, ls, lw=1.2, color=color, alpha=0.9)
    for ax, ylab in zip(
        axes.flat,
        [r"$F_1(t)$", r"$F_2(t)$", r"$F_3(t)$",
         r"$F_4(t)=-\frac{1}{N}D_{\mathrm{KL}}(\psi\|\psi^*)$ [nats]"],
    ):
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    for ax in axes[1, :]:
        ax.set_xlabel("generation $t$")
    title_lines = []
    if branch_title:
        title_lines.append(branch_title)
    title_lines.append(r"$F_1$–$F_4$ dynamics colored by $\mathrm{KL}_{\mathrm{eq}}$")
    title_lines.append(rf"$N={N}$, $u=v={u:g}$, $r={r:g}$, $s_1={s1:g}$, $s_2={s2:g}$")
    fig.suptitle("\n".join(title_lines), fontsize=10)
    if show_sign_legend:
        _add_eps12_sign_legend(axes[0, 1])
    _add_score_colorbar(fig, axes[1, 1], cbar_cmap, cbar_norm, cbar_ticks)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def _branch_panel_scatter_fixed(ax, sub, y_attr, ylabel, title, rho, branch_color, order):
    y = np.array([getattr(m, y_attr) for m in sub])
    kl = np.array([m.kl_eq for m in sub])
    ax.plot(kl[order], y[order], "-", color=branch_color, lw=1.2, alpha=0.5, zorder=1)
    ax.scatter(kl, y, c=branch_color, s=70, alpha=0.95, zorder=2, edgecolors="white", linewidths=0.4)
    for m in sub:
        ax.annotate(
            f"{m.eps12:+.2f}", (m.kl_eq, getattr(m, y_attr)),
            textcoords="offset points", xytext=(4, 3), fontsize=6.5, color="0.25",
        )
    ax.set_xlabel(r"$\mathrm{KL}_{\mathrm{eq}}(\psi^*\|\varphi)$ [bits]")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nSpearman $\\rho={rho:.2f}$", fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_efficiency_convergence_branch(
    metrics: list[Eps12Metrics],
    out_dir: Path,
    branch: str,
    *,
    N: int,
    u: float,
    s1: float,
    s2: float,
    r: float,
    pergen_max_t: int,
) -> None:
    """Efficiency/convergence vs KL_eq for one epistasis branch only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = _filter_branch(metrics, branch)
    if len(sub) < 2:
        print(f"  skip branch {branch}: fewer than 2 points", flush=True)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    kl = np.array([m.kl_eq for m in sub])
    path_eff = np.array([m.path_efficiency for m in sub])
    t90 = np.array([m.t_90 for m in sub])
    peak = np.array([m.peak_eta for m in sub])
    eps = np.array([m.eps12 for m in sub])
    order = np.argsort(kl)

    rho_pe = _spearman(kl, path_eff)
    rho_t90 = _spearman(kl, t90)
    rho_peak = _spearman(kl, peak)

    summary = {
        "branch": branch,
        "branch_title": BRANCH_TITLES[branch],
        "n": len(sub),
        "spearman_kl_eq_vs_path_efficiency": rho_pe,
        "spearman_kl_eq_vs_t_90": rho_t90,
        "spearman_kl_eq_vs_peak_eta": rho_peak,
        "pergen_max_t": pergen_max_t,
    }
    (out_dir / f"efficiency_convergence_summary_{branch}.json").write_text(
        json.dumps(summary, indent=2),
    )
    print(
        f"  [{branch}] Spearman KL_eq vs path_eff={rho_pe:.3f}  "
        f"vs t_90={rho_t90:.3f}  vs peak_eta={rho_peak:.3f}",
        flush=True,
    )

    branch_color = "C3" if branch == "pos" else "C0"

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    _branch_panel_scatter_fixed(
        axes[0, 0], sub, "path_efficiency",
        r"$\bar{\eta} = \sum_t \Delta D / \sum_t N C(t)$",
        "Path-normalized efficiency", rho_pe, branch_color, order,
    )
    _branch_panel_scatter_fixed(
        axes[0, 1], sub, "t_90",
        r"$t_{90}$ [generations]",
        r"Convergence: first $t$ with $D(t)\geq 0.9\,\mathrm{KL}_{\mathrm{eq}}$",
        rho_t90, branch_color, order,
    )
    _branch_panel_scatter_fixed(
        axes[1, 0], sub, "peak_eta",
        r"$\max_t \Delta D/(N C(t))$",
        "Peak instantaneous efficiency", rho_peak, branch_color, order,
    )
    ax = axes[1, 1]
    o2 = np.argsort(eps)
    ax.plot(eps[o2], path_eff[o2], "o-", color="C2", lw=1.5, ms=5, label=r"$\bar{\eta}$")
    ax2 = ax.twinx()
    ax2.plot(eps[o2], t90[o2], "s--", color="C4", lw=1.5, ms=5, label=r"$t_{90}$")
    ax.set_xlabel(r"$\varepsilon_{12}$")
    ax.set_ylabel(r"$\bar{\eta}$", color="C2")
    ax2.set_ylabel(r"$t_{90}$", color="C4")
    ax.tick_params(axis="y", labelcolor="C2")
    ax2.tick_params(axis="y", labelcolor="C4")
    ax.grid(True, alpha=0.3)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="best", fontsize=8)
    ax.set_title(r"$\bar{\eta}$ and $t_{90}$ vs $\varepsilon_{12}$", fontsize=9)

    fig.suptitle(
        rf"{BRANCH_TITLES[branch]}" + "\n"
        rf"Efficiency \& convergence ($N={N}$, $u=v={u:g}$, $r={r:g}$, "
        rf"$s_1={s1:g}$, $s_2={s2:g}$)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out_dir / f"efficiency_convergence_vs_kl_eq_{branch}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}", flush=True)


def plot_delta_d_over_nc(
    runs: list[Eps12Run],
    out_path: Path,
    *,
    N: int,
    u: float,
    s1: float,
    s2: float,
    r: float,
    branch_title: str | None = None,
    show_sign_legend: bool = True,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors, cbar_cmap, cbar_norm, cbar_ticks = _score_colors(runs)
    runs_f = [r for r in runs if np.isfinite(r.kl_eq)]

    fig, ax = plt.subplots(figsize=(9, 5))
    for run, color in zip(runs_f, colors):
        t = run.pergen_times
        delta_d = np.diff(run.pergen_kl_phi_bits)
        nc = N * run.pergen_c_cost[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(nc > 1e-300, delta_d / nc, np.nan)
        ax.plot(
            t[1:], ratio, "-" if not show_sign_legend else _eps12_sign_linestyle(run.eps12),
            lw=1.3, color=color, alpha=0.92,
        )
    ax.axhline(1.0, color="0.5", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("generation $t$")
    ax.set_ylabel(r"$\Delta D / (N\,C(t))$")
    title_lines = []
    if branch_title:
        title_lines.append(branch_title)
    title_lines.append(
        rf"$\Delta D / (N\,C(t))$ colored by $\mathrm{{KL}}_{{\mathrm{{eq}}}}$"
    )
    title_lines.append(
        rf"$N={N}$, $u=v={u:g}$, $r={r:g}$, $s_1={s1:g}$, $s_2={s2:g}$, "
        rf"$t=1\ldots{int(runs_f[0].pergen_times[-1])}$",
    )
    ax.set_title("\n".join(title_lines), fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(runs_f[0].pergen_times[0], runs_f[0].pergen_times[-1])
    if show_sign_legend:
        _add_eps12_sign_legend(ax)
    _add_score_colorbar(fig, ax, cbar_cmap, cbar_norm, cbar_ticks)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def plot_delta_d_and_nc(
    runs: list[Eps12Run],
    out_path: Path,
    *,
    N: int,
    u: float,
    s1: float,
    s2: float,
    r: float,
    branch_title: str | None = None,
    show_sign_legend: bool = True,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors, cbar_cmap, cbar_norm, cbar_ticks = _score_colors(runs)
    runs_f = [r for r in runs if np.isfinite(r.kl_eq)]

    fig, ax = plt.subplots(figsize=(9, 5))
    for run, color in zip(runs_f, colors):
        t = run.pergen_times
        delta_d = np.diff(run.pergen_kl_phi_bits)
        nc = N * run.pergen_c_cost
        ax.plot(
            t[1:], delta_d, "-" if not show_sign_legend else _eps12_sign_linestyle(run.eps12),
            lw=1.2, color=color, alpha=0.9,
        )
        ax.plot(t, nc, "-.", lw=0.9, color=color, alpha=0.55)
    ax.axhline(0.0, color="0.5", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("generation $t$")
    ax.set_ylabel(r"$\Delta D$ [bits/gen]  and  $N\,C(t)$")
    title_lines = []
    if branch_title:
        title_lines.append(branch_title)
    title_lines.append(
        rf"$\Delta D$ (solid) and $N\,C(t)$ (dash-dot); color = $\mathrm{{KL}}_{{\mathrm{{eq}}}}$"
    )
    title_lines.append(
        rf"$N={N}$, $u=v={u:g}$, $r={r:g}$, $s_1={s1:g}$, $s_2={s2:g}$, "
        rf"$t=0\ldots{int(runs_f[0].pergen_times[-1])}$",
    )
    ax.set_title("\n".join(title_lines), fontsize=10)
    ax.grid(True, alpha=0.3)
    ylo = min(
        float(np.min(np.diff(r.pergen_kl_phi_bits))) for r in runs_f
    )
    yhi = max(
        float(np.max(np.diff(r.pergen_kl_phi_bits))) for r in runs_f
    )
    yhi = max(yhi, max(float(np.max(N * r.pergen_c_cost)) for r in runs_f))
    pad = 0.05 * (yhi - ylo) if yhi > ylo else 1e-6
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.set_xlim(runs_f[0].pergen_times[0], runs_f[0].pergen_times[-1])
    if show_sign_legend:
        _add_eps12_sign_legend(ax)
    _add_score_colorbar(fig, ax, cbar_cmap, cbar_norm, cbar_ticks)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def plot_delta_d_only(
    runs: list[Eps12Run],
    out_path: Path,
    *,
    N: int,
    u: float,
    s1: float,
    s2: float,
    r: float,
    branch_title: str | None = None,
    show_sign_legend: bool = True,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors, cbar_cmap, cbar_norm, cbar_ticks = _score_colors(runs)
    runs_f = [r for r in runs if np.isfinite(r.kl_eq)]

    fig, ax = plt.subplots(figsize=(9, 5))
    for run, color in zip(runs_f, colors):
        t = run.pergen_times
        ax.plot(
            t[1:], np.diff(run.pergen_kl_phi_bits),
            "-" if not show_sign_legend else _eps12_sign_linestyle(run.eps12),
            lw=1.3, color=color, alpha=0.92,
        )
    ax.axhline(0.0, color="0.5", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("generation $t$")
    ax.set_ylabel(r"$\Delta D$ [bits/gen]")
    title_lines = []
    if branch_title:
        title_lines.append(branch_title)
    title_lines.append(rf"$\Delta D$ colored by $\mathrm{{KL}}_{{\mathrm{{eq}}}}$")
    title_lines.append(rf"$N={N}$, $u=v={u:g}$, $r={r:g}$, $s_1={s1:g}$, $s_2={s2:g}$")
    ax.set_title("\n".join(title_lines), fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(runs_f[0].pergen_times[0], runs_f[0].pergen_times[-1])
    if show_sign_legend:
        _add_eps12_sign_legend(ax)
    _add_score_colorbar(fig, ax, cbar_cmap, cbar_norm, cbar_ticks)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def plot_nc_only(
    runs: list[Eps12Run],
    out_path: Path,
    *,
    N: int,
    u: float,
    s1: float,
    s2: float,
    r: float,
    branch_title: str | None = None,
    show_sign_legend: bool = True,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors, cbar_cmap, cbar_norm, cbar_ticks = _score_colors(runs)
    runs_f = [r for r in runs if np.isfinite(r.kl_eq)]

    fig, ax = plt.subplots(figsize=(9, 5))
    for run, color in zip(runs_f, colors):
        ax.plot(
            run.pergen_times, N * run.pergen_c_cost,
            "-" if not show_sign_legend else _eps12_sign_linestyle(run.eps12),
            lw=1.3, color=color, alpha=0.92,
        )
    ax.set_xlabel("generation $t$")
    ax.set_ylabel(r"$N\,C(t)$")
    title_lines = []
    if branch_title:
        title_lines.append(branch_title)
    title_lines.append(rf"$N\,C(t)$ colored by $\mathrm{{KL}}_{{\mathrm{{eq}}}}$")
    title_lines.append(rf"$N={N}$, $u=v={u:g}$, $r={r:g}$, $s_1={s1:g}$, $s_2={s2:g}$")
    ax.set_title("\n".join(title_lines), fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(runs_f[0].pergen_times[0], runs_f[0].pergen_times[-1])
    if show_sign_legend:
        _add_eps12_sign_legend(ax)
    _add_score_colorbar(fig, ax, cbar_cmap, cbar_norm, cbar_ticks)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def plot_branch_suite(
    runs: list[Eps12Run],
    metrics: list[Eps12Metrics],
    out_dir: Path,
    branch: str,
    *,
    N: int,
    u: float,
    s1: float,
    s2: float,
    r: float,
    pergen_max_t: int,
) -> None:
    br = _filter_branch(runs, branch)
    if not br:
        print(f"  skip branch {branch}: no runs", flush=True)
        return
    bt = BRANCH_TITLES[branch]
    kw = dict(N=N, u=u, s1=s1, s2=s2, r=r, branch_title=bt, show_sign_legend=False)
    plot_delta_d_over_nc(br, out_dir / f"eps12_delta_d_over_nc_{branch}.png", **kw)
    plot_delta_d_and_nc(br, out_dir / f"eps12_delta_d_and_nc_{branch}.png", **kw)
    plot_delta_d_only(br, out_dir / f"eps12_delta_d_{branch}.png", **kw)
    plot_nc_only(br, out_dir / f"eps12_nc_{branch}.png", **kw)
    plot_f123_trajectories(br, out_dir / f"f123_trajectories_{branch}.png", **kw)
    plot_efficiency_convergence_branch(
        metrics, out_dir, branch,
        N=N, u=u, s1=s1, s2=s2, r=r, pergen_max_t=pergen_max_t,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="eps12 sweep: ΔD/(N C) colored by KL_eq")
    ap.add_argument("--config", type=Path, default=SCRIPT_DIR / "eps12_delta_d_config.yaml")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    cfg = load_config(args.config.resolve())
    N = int(cfg.get("N", 30))
    r = float(cfg.get("r", 0.0))
    u = float(cfg["u"][0]) if isinstance(cfg.get("u"), list) else float(cfg.get("u", 0.01))
    v = float(cfg["v"][0]) if isinstance(cfg.get("v"), list) else float(cfg.get("v", u))
    s1 = float(cfg.get("s1", 0.001))
    s2 = float(cfg.get("s2", -0.001))
    pergen_max_t = int(cfg.get("pergen_max_t", PERGEN_MAX_T))
    eps_grid = _eps12_grid(cfg)

    print(
        f"eps12 sweep: N={N}, u=v={u:g}, r={r:g}, s1={s1:g}, s2={s2:g}, "
        f"{len(eps_grid)} eps12 values, t=0..{pergen_max_t}",
        flush=True,
    )

    M_mut = mutation_matrix_asymmetric(np.array([u, u]), np.array([v, v]))
    params0 = params_from_config(_make_cfg(N=N, r=r, u=u, v=v, s1=s1, s2=s2, eps12=0.0), SCRIPT_DIR)
    phi, _, fracs, _, src = compute_phi(params0, M_mut)
    print(f"  phi: {src}", flush=True)
    q_cond = cond_alpha_reference_pmf(fracs, M_mut, N)
    print("  q_cond: cond-α reference built", flush=True)

    runs: list[Eps12Run] = []
    for i, eps in enumerate(eps_grid, start=1):
        print(f"[{i}/{len(eps_grid)}]", end=" ", flush=True)
        runs.append(run_one_eps12(
            float(eps), N=N, r=r, u=u, v=v, s1=s1, s2=s2,
            phi=phi, fracs=fracs, M_mut=M_mut, q_cond=q_cond,
            pergen_max_t=pergen_max_t, verbose=True,
        ))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "settings.json").write_text(json.dumps({
        "N": N, "r": r, "u": u, "v": v, "s1": s1, "s2": s2,
        "pergen_max_t": pergen_max_t,
        "eps12": [r.eps12 for r in runs],
        "kl_eq_bits": [r.kl_eq for r in runs],
        "f1_eq": [r.f1_eq for r in runs],
        "f2_eq": [r.f2_eq for r in runs],
        "f3_eq": [r.f3_eq for r in runs],
        "wstar_eq": [r.wstar_eq for r in runs],
    }, indent=2))

    kw = dict(N=N, u=u, s1=s1, s2=s2, r=r)
    plot_delta_d_over_nc(runs, args.out_dir / "eps12_delta_d_over_nc.png", **kw)
    plot_delta_d_and_nc(runs, args.out_dir / "eps12_delta_d_and_nc.png", **kw)
    plot_delta_d_only(runs, args.out_dir / "eps12_delta_d.png", **kw)
    plot_nc_only(runs, args.out_dir / "eps12_nc.png", **kw)

    metrics = [analyze_run(r, N) for r in runs]
    _write_metrics_csv(args.out_dir / "eps12_metrics.csv", metrics)
    plot_free_fitness_vs_kl_eq(metrics, args.out_dir, N=N, u=u, s1=s1, s2=s2, r=r)
    plot_f2_decomposition(runs, metrics, args.out_dir, N=N, s1=s1, s2=s2, r=r)
    plot_f123_trajectories(runs, args.out_dir / "f123_trajectories.png", **kw)
    plot_efficiency_convergence_investigation(
        metrics, args.out_dir, N=N, u=u, s1=s1, s2=s2, r=r, pergen_max_t=pergen_max_t,
    )
    for branch in ("pos", "neg"):
        plot_branch_suite(
            runs, metrics, args.out_dir, branch,
            N=N, u=u, s1=s1, s2=s2, r=r, pergen_max_t=pergen_max_t,
        )

    print(f"\nDone. Outputs in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
