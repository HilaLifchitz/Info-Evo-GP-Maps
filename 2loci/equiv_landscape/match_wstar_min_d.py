#!/usr/bin/env python3
r"""
Iso-⟨ln W⟩ scan: match expected log mean fitness at ψ* while varying ε₁₂.

For each reference (s₁, s₂, ε₁₂):
  1. Build Moran chain, compute ψ* and neutral φ*
  2. Record  W_ref = ⟨ln W⟩_ψ*  and  D_ref = KL(ψ* || φ*) [bits]
  3. For each trial ε₁₂, search (s₁, s₂) so that  ⟨ln W⟩_ψ* ≈ W_ref
  4. Report D at the match — goal: find ε where D is lower at same ⟨ln W⟩

Uses the same chain setup as ``log_Z_param_sweep`` (φ* = neutral stationary, ψ* from
selection equilibrium).  ⟨ln W⟩ is ``E_ψ[log_mean_fitness_genotype_simplex(x, ω)]``.

Usage:
  cd 2loci
  # Iso-⟨ln W⟩ scan in ε₁₂
  python -m equiv_landscape.match_wstar_min_d --mode scan --plot

  # D(ψ*||φ*) heatmap over (s₁, s₂) at fixed ε₁₂=10⁻², fast mutation u=v=0.005
  python -m equiv_landscape.match_wstar_min_d --mode heatmap --plot

  # Batch iso-⟨ln W⟩ scans (several reference landscapes; ε scan from 0)
  python -m equiv_landscape.match_wstar_min_d --mode batch-scan --plot
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from compare_sigma_dynamics import expectation  # noqa: E402
from simplex import (  # noqa: E402
    build_sigma,
    build_transition_matrix_row_stochastic,
    column_stochastic,
    compute_phi,
    fitness_exp,
    kl_divergence_bits,
    log_mean_fitness_genotype_simplex,
    make_simplex_grid,
    mutation_matrix_asymmetric,
    params_from_config,
    try_selection_equilibrium,
)


@dataclass
class Metrics:
    s1: float
    s2: float
    eps12: float
    wstar_mean: float
    d_kl_bits: float
    psi_resid: float
    ok: bool


@dataclass
class ChainContext:
    N: int
    r: float
    u: np.ndarray
    v: np.ndarray
    fracs: np.ndarray
    M_mut: np.ndarray
    phi: np.ndarray


def _make_cfg_vec(
    *,
    N: int,
    r: float,
    u: np.ndarray,
    v: np.ndarray,
    s1: float,
    s2: float,
    eps12: float,
) -> dict:
    return {
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
    }


def build_context(*, N: int, u: tuple[float, float], v: tuple[float, float], r: float) -> ChainContext:
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    M_mut = mutation_matrix_asymmetric(u_arr, v_arr)
    params0 = params_from_config(
        _make_cfg_vec(N=N, r=r, u=u_arr, v=v_arr, s1=0.0, s2=0.0, eps12=0.0),
        SCRIPT_DIR,
    )
    phi, _, fracs, _, _ = compute_phi(params0, M_mut)
    return ChainContext(N=N, r=r, u=u_arr, v=v_arr, fracs=fracs, M_mut=M_mut, phi=phi)


def evaluate_metrics(
    ctx: ChainContext,
    s1: float,
    s2: float,
    eps12: float,
    *,
    verbose: bool = False,
) -> Metrics:
    params = params_from_config(
        _make_cfg_vec(
            N=ctx.N, r=ctx.r, u=ctx.u, v=ctx.v,
            s1=float(s1), s2=float(s2), eps12=float(eps12),
        ),
        SCRIPT_DIR,
    )
    omega = fitness_exp(params.Sigma)
    wstar_ln = log_mean_fitness_genotype_simplex(ctx.fracs, omega)
    T_row, _, _ = build_transition_matrix_row_stochastic(
        ctx.N, omega, ctx.M_mut, ctx.r, n_jobs=-1,
        label=f"eq_e{eps12:.2g}"[:16], verbose=verbose,
    )
    P_col = column_stochastic(T_row)
    psi_star, resid, ok = try_selection_equilibrium(P_col)
    if not ok or psi_star is None:
        return Metrics(float(s1), float(s2), float(eps12), float("nan"), float("nan"), float(resid), False)
    wstar_mean = float(expectation(psi_star, wstar_ln))
    d_kl = float(kl_divergence_bits(psi_star, ctx.phi))
    return Metrics(float(s1), float(s2), float(eps12), wstar_mean, d_kl, float(resid), True)


def find_s1_s2_for_wstar(
    ctx: ChainContext,
    eps12: float,
    target_wstar: float,
    *,
    x0: tuple[float, float],
    s1_bounds: tuple[float, float],
    s2_bounds: tuple[float, float],
) -> tuple[float, float, Metrics, float]:
    from scipy.optimize import minimize

    def obj(x: np.ndarray) -> float:
        m = evaluate_metrics(ctx, float(x[0]), float(x[1]), eps12)
        if not m.ok:
            return 1e6 + m.psi_resid
        return (m.wstar_mean - target_wstar) ** 2

    res = minimize(
        obj,
        np.array(x0, dtype=float),
        method="L-BFGS-B",
        bounds=[s1_bounds, s2_bounds],
    )
    s1, s2 = float(res.x[0]), float(res.x[1])
    m = evaluate_metrics(ctx, s1, s2, eps12)
    return s1, s2, m, float(res.fun)


def scan_eps_isobar(
    ctx: ChainContext,
    ref: Metrics,
    *,
    eps_values: np.ndarray,
    x0: tuple[float, float],
    s1_bounds: tuple[float, float],
    s2_bounds: tuple[float, float],
    wstar_tol: float,
) -> list[dict]:
    rows: list[dict] = []
    for eps in eps_values:
        s1, s2, m, sse = find_s1_s2_for_wstar(
            ctx, float(eps), ref.wstar_mean,
            x0=(s1, s2) if rows else x0,
            s1_bounds=s1_bounds,
            s2_bounds=s2_bounds,
        )
        row = {
            "eps12": float(eps),
            "eps_delta": float(eps - ref.eps12),
            "s1": s1,
            "s2": s2,
            "wstar_mean": m.wstar_mean,
            "wstar_err": m.wstar_mean - ref.wstar_mean,
            "d_kl_bits": m.d_kl_bits,
            "d_delta": m.d_kl_bits - ref.d_kl_bits,
            "sse_wstar": sse,
            "ok": m.ok,
        }
        rows.append(row)
    return rows


def find_min_d_on_wstar_isobar(
    ctx: ChainContext,
    eps12: float,
    target_wstar: float,
    *,
    x0: tuple[float, float],
    s1_bounds: tuple[float, float],
    s2_bounds: tuple[float, float],
    wstar_weight: float = 1e6,
) -> tuple[float, float, Metrics, float]:
    """Minimize D on the iso-⟨ln W⟩ curve at fixed ε₁₂ (penalty on ⟨ln W⟩ mismatch)."""
    from scipy.optimize import minimize

    def obj(x: np.ndarray) -> float:
        m = evaluate_metrics(ctx, float(x[0]), float(x[1]), eps12)
        if not m.ok:
            return 1e6 + m.psi_resid
        dw = m.wstar_mean - target_wstar
        return m.d_kl_bits + wstar_weight * dw * dw

    res = minimize(
        obj,
        np.array(x0, dtype=float),
        method="L-BFGS-B",
        bounds=[s1_bounds, s2_bounds],
    )
    s1, s2 = float(res.x[0]), float(res.x[1])
    m = evaluate_metrics(ctx, s1, s2, eps12)
    dw = m.wstar_mean - target_wstar if m.ok else float("nan")
    return s1, s2, m, float(dw)


def iso_wstar_slice_from_grid(
    s1_grid: np.ndarray,
    s2_grid: np.ndarray,
    w_map: np.ndarray,
    d_map: np.ndarray,
    w_target: float,
    *,
    w_tol_frac: float = 0.02,
) -> list[dict]:
    """Grid cells near iso-⟨ln W⟩ with their D (coarse slice for plotting)."""
    if not np.isfinite(w_target) or w_target == 0:
        w_tol = max(abs(w_target) * w_tol_frac, 1e-8)
    else:
        w_tol = abs(w_target) * w_tol_frac
    rows: list[dict] = []
    for j, s2 in enumerate(s2_grid):
        for i, s1 in enumerate(s1_grid):
            w, d = w_map[j, i], d_map[j, i]
            if not (np.isfinite(w) and np.isfinite(d)):
                continue
            if abs(w - w_target) <= w_tol:
                rows.append({"s1": float(s1), "s2": float(s2), "wstar_mean": float(w), "d_kl_bits": float(d)})
    return rows


BATCH_SCAN_CASES: list[dict] = [
    {
        "label": "weak_uv_x",
        "s1": 1e-3, "s2": -5e-4, "eps12": 3e-4,
        "u0": 5e-3, "u1": 1e-3, "v0": 1e-3, "v1": 5e-3,
        "s_span": 0.003, "eps_hi": 0.003,
    },
    {
        "label": "moderate_sym_eps01",
        "s1": 1e-3, "s2": -5e-4, "eps12": 0.01,
        "u0": 5e-3, "u1": 5e-3, "v0": 5e-3, "v1": 5e-3,
        "s_span": 0.004, "eps_hi": 0.015,
    },
    {
        "label": "high_s_eps01",
        "s1": 3e-3, "s2": -1.5e-3, "eps12": 0.01,
        "u0": 5e-3, "u1": 5e-3, "v0": 5e-3, "v1": 5e-3,
        "s_span": 0.006, "eps_hi": 0.015,
    },
    {
        "label": "high_s_eps02",
        "s1": 5e-3, "s2": -2.5e-3, "eps12": 0.02,
        "u0": 5e-3, "u1": 5e-3, "v0": 5e-3, "v1": 5e-3,
        "s_span": 0.008, "eps_hi": 0.025,
    },
    {
        "label": "very_high_s_eps01",
        "s1": 8e-3, "s2": -4e-3, "eps12": 0.01,
        "u0": 5e-3, "u1": 5e-3, "v0": 5e-3, "v1": 5e-3,
        "s_span": 0.010, "eps_hi": 0.015,
    },
]


def grid_s1_s2_d(
    ctx: ChainContext,
    eps12: float,
    *,
    s1_grid: np.ndarray,
    s2_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return S1, S2 mesh, D [bits], ⟨ln W⟩ [nats] on grid."""
    n1, n2 = len(s1_grid), len(s2_grid)
    d_map = np.full((n2, n1), np.nan)
    w_map = np.full((n2, n1), np.nan)
    total = n1 * n2
    for j, s2 in enumerate(s2_grid):
        for i, s1 in enumerate(s1_grid):
            k = j * n1 + i + 1
            print(f"    [{k}/{total}] s1={s1:+.4g} s2={s2:+.4g}", flush=True)
            m = evaluate_metrics(ctx, float(s1), float(s2), eps12)
            if m.ok:
                d_map[j, i] = m.d_kl_bits
                w_map[j, i] = m.wstar_mean
    S1, S2 = np.meshgrid(s1_grid, s2_grid)
    return S1, S2, d_map, w_map


def _plot_heatmap(
    ref: Metrics,
    s1_grid: np.ndarray,
    s2_grid: np.ndarray,
    d_map: np.ndarray,
    w_map: np.ndarray,
    ctx: ChainContext,
    out_path: Path,
    *,
    iso_min: Metrics | None = None,
    iso_slice: list[dict] | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, z, title, cbar_label in (
        (axes[0], d_map, r"$D_{KL}(\psi^*\|\varphi^*)$", "bits"),
        (axes[1], w_map, r"$\langle\ln W\rangle_{\psi^*}$", "nats"),
    ):
        im = ax.pcolormesh(
            s1_grid, s2_grid, z,
            shading="auto",
            cmap="viridis" if ax is axes[0] else "plasma",
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cbar_label, fontsize=9)
        ax.scatter([ref.s1], [ref.s2], s=120, c="white", edgecolors="k", lw=1.5, zorder=5,
                   marker="*", label="reference")
        ax.set_xlabel(r"$s_1$")
        ax.set_ylabel(r"$s_2$")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, loc="upper right")

    for ax in axes:
        if np.isfinite(w_map).any():
            ax.contour(
                s1_grid, s2_grid, w_map,
                levels=[ref.wstar_mean],
                colors="white",
                linewidths=1.2,
                linestyles="--",
            )

    if iso_slice:
        s_iso = [p["s1"] for p in iso_slice]
        d_iso = [p["d_kl_bits"] for p in iso_slice]
        axes[0].scatter(s_iso, [p["s2"] for p in iso_slice], s=18, c=d_iso, cmap="coolwarm",
                        edgecolors="white", linewidths=0.4, zorder=4, label="iso-⟨ln W⟩ grid")

    if iso_min is not None and iso_min.ok:
        axes[0].scatter(
            [iso_min.s1], [iso_min.s2], s=100, c="C3", edgecolors="k", lw=1.0, zorder=6,
            marker="D", label=rf"min $D$ on iso ($D={iso_min.d_kl_bits:.3g}$)",
        )
        axes[0].legend(fontsize=7, loc="upper right")

    fig.suptitle(
        rf"$D$ and $\langle\ln W\rangle$ on $(s_1,s_2)$ grid  "
        rf"($N={ctx.N}$, $\varepsilon_{{12}}={ref.eps12:g}$, "
        rf"$u=({ctx.u[0]:g},{ctx.u[1]:g})$, $v=({ctx.v[0]:g},{ctx.v[1]:g})$, $r={ctx.r:g}$)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_iso_slice(ref: Metrics, iso_min: Metrics, out_path: Path) -> None:
    """D at reference vs optimized min on iso-⟨ln W⟩ at fixed ε."""
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["reference", "iso-min D"]
    vals = [ref.d_kl_bits, iso_min.d_kl_bits]
    colors = ["C3", "C2"]
    ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.set_ylabel(r"$D_{KL}(\psi^*\|\varphi^*)$ [bits]")
    ax.set_title(
        rf"Same $\langle\ln W\rangle$ at $\varepsilon_{{12}}={ref.eps12:g}$\n"
        rf"ref: $(s_1,s_2)=({ref.s1:g},{ref.s2:g})$  "
        rf"min: $({iso_min.s1:g},{iso_min.s2:g})$  "
        rf"$\Delta D={iso_min.d_kl_bits - ref.d_kl_bits:+.4g}$",
        fontsize=9,
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_batch_summary(cases: list[dict], out_path: Path) -> None:
    labels = [c["label"] for c in cases]
    d_ref = [c["d_ref"] for c in cases]
    d_min = [c["d_min"] for c in cases]
    eps_min = [c["eps_at_min"] for c in cases]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax = axes[0]
    w = 0.35
    ax.bar(x - w / 2, d_ref, w, label=r"$D_{\rm ref}$", color="C3", alpha=0.85)
    ax.bar(x + w / 2, d_min, w, label=r"$D_{\min}$ (iso-$\langle\ln W\rangle$)", color="C2", alpha=0.85)
    ax.set_ylabel(r"$D_{KL}$ [bits]")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(r"Can small $\varepsilon_{12}$ lower $D$ at matched $\langle\ln W\rangle$?", fontsize=10)

    ax = axes[1]
    ax.bar(x, eps_min, color="C0", alpha=0.85)
    ax.axhline(1e-3, color="C3", ls="--", lw=1, label=r"$\varepsilon=10^{-3}$")
    for i, s in enumerate(cases):
        if s.get("small_eps_wins"):
            ax.annotate("*", (i, s["eps_at_min"]), ha="center", va="bottom", fontsize=14, color="C3")
    ax.set_ylabel(r"$\varepsilon_{12}$ at $D_{\min}$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scan(ref: Metrics, rows: list[dict], out_path: Path) -> None:
    ok = [r for r in rows if r["ok"]]
    if not ok:
        return
    eps = np.array([r["eps12"] for r in ok])
    d = np.array([r["d_kl_bits"] for r in ok])
    d0 = ref.d_kl_bits

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax = axes[0]
    ax.plot(eps, d, "o-", color="C0", lw=1.5, ms=4, label=r"$D_{KL}(\psi^*\|\varphi^*)$")
    ax.axhline(d0, color="C3", ls="--", lw=1.2, label=rf"ref $D$={d0:.4g} bits")
    ax.axvline(ref.eps12, color="0.4", ls=":", lw=1.0)
    j = int(np.nanargmin(d))
    ax.scatter([eps[j]], [d[j]], s=80, c="C2", zorder=5, label=rf"min $D$ at eps={eps[j]:+.4g}")
    ax.set_ylabel(r"$D_{KL}(\psi^*\|\varphi^*)$ [bits]")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        rf"Iso-$\langle\ln W\rangle_{{\psi^*}}$ scan  "
        rf"(ref: $s_1={ref.s1:g}$, $s_2={ref.s2:g}$, $\varepsilon_{{12}}={ref.eps12:g}$)",
        fontsize=9,
    )

    ax = axes[1]
    ax.plot(eps, [r["s1"] for r in ok], "s-", label=r"$s_1$", ms=4)
    ax.plot(eps, [r["s2"] for r in ok], "^-", label=r"$s_2$", ms=4)
    ax.axhline(ref.s1, color="C0", ls=":", alpha=0.5)
    ax.axhline(ref.s2, color="C1", ls=":", alpha=0.5)
    ax.set_xlabel(r"$\varepsilon_{12}$")
    ax.set_ylabel("compensating selection")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_heatmap(args: argparse.Namespace) -> None:
    ctx = build_context(
        N=args.N,
        u=(args.u0, args.u1),
        v=(args.v0, args.v1),
        r=args.r,
    )
    print(
        f"Reference (N={args.N}, u=({args.u0:g},{args.u1:g}), "
        f"v=({args.v0:g},{args.v1:g}), r={args.r:g}, eps12={args.eps12:g})",
        flush=True,
    )
    ref = evaluate_metrics(ctx, args.s1, args.s2, args.eps12, verbose=False)
    if not ref.ok:
        raise SystemExit(f"reference psi* failed resid={ref.psi_resid}")
    print(f"  ref (s1,s2)=({ref.s1:g}, {ref.s2:g})")
    print(f"  ⟨ln W⟩_ψ* = {ref.wstar_mean:.8f} nats")
    print(f"  D_KL(ψ*||φ*) = {ref.d_kl_bits:.6f} bits\n")

    s1_grid = np.linspace(args.s1 - args.s1_span, args.s1 + args.s1_span, args.grid_n)
    s2_grid = np.linspace(args.s2 - args.s2_span, args.s2 + args.s2_span, args.grid_n)
    print(f"Grid {args.grid_n}×{args.grid_n} on s1∈[{s1_grid[0]:g},{s1_grid[-1]:g}], "
          f"s2∈[{s2_grid[0]:g},{s2_grid[-1]:g}]", flush=True)

    S1, S2, d_map, w_map = grid_s1_s2_d(ctx, args.eps12, s1_grid=s1_grid, s2_grid=s2_grid)

    ok = np.isfinite(d_map)
    if not ok.any():
        raise SystemExit("heatmap: all grid points failed")
    j_min = np.unravel_index(int(np.nanargmin(d_map)), d_map.shape)
    print(
        f"\nMin D on grid: {d_map[j_min]:.6f} bits at "
        f"s1={s1_grid[j_min[1]]:+.6g} s2={s2_grid[j_min[0]]:+.6g} "
        f"(ΔD={d_map[j_min]-ref.d_kl_bits:+.6f})",
        flush=True,
    )

    s1_bounds = (float(s1_grid[0]), float(s1_grid[-1]))
    s2_bounds = (float(s2_grid[0]), float(s2_grid[-1]))
    print("Optimizing min D on iso-⟨ln W⟩ at fixed ε …", flush=True)
    _, _, iso_min, dw_iso = find_min_d_on_wstar_isobar(
        ctx, args.eps12, ref.wstar_mean,
        x0=(args.s1, args.s2),
        s1_bounds=s1_bounds,
        s2_bounds=s2_bounds,
    )
    iso_slice = iso_wstar_slice_from_grid(s1_grid, s2_grid, w_map, d_map, ref.wstar_mean)
    if iso_min.ok:
        print(
            f"Iso-min D: {iso_min.d_kl_bits:.6f} bits at "
            f"(s1,s2)=({iso_min.s1:+.6g},{iso_min.s2:+.6g})  "
            f"ΔD={iso_min.d_kl_bits - ref.d_kl_bits:+.6f}  Δ⟨ln W⟩={dw_iso:+.2e}",
            flush=True,
        )
    else:
        print("Iso-min optimization failed", flush=True)

    payload = {
        "mode": "heatmap",
        "reference": asdict(ref),
        "iso_min": asdict(iso_min) if iso_min.ok else None,
        "iso_slice_n": len(iso_slice),
        "context": {
            "N": args.N, "r": args.r,
            "u": [args.u0, args.u1], "v": [args.v0, args.v1],
        },
        "s1_grid": s1_grid.tolist(),
        "s2_grid": s2_grid.tolist(),
        "d_kl_bits": d_map.tolist(),
        "wstar_mean": w_map.tolist(),
    }
    out_dir = SCRIPT_DIR / "equiv_landscape" / "results"
    out_json = args.out or out_dir / f"D_heatmap_N{args.N}_eps{args.eps12:g}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_json}", flush=True)

    if args.plot:
        plot_path = out_json.with_suffix(".png")
        _plot_heatmap(
            ref, s1_grid, s2_grid, d_map, w_map, ctx, plot_path,
            iso_min=iso_min if iso_min.ok else None,
            iso_slice=iso_slice,
        )
        print(f"wrote {plot_path}", flush=True)
        if iso_min.ok:
            iso_plot = out_json.with_name(out_json.stem + "_iso_min.png")
            _plot_iso_slice(ref, iso_min, iso_plot)
            print(f"wrote {iso_plot}", flush=True)


def _make_eps_grid(
    args: argparse.Namespace,
    *,
    eps_ref: float | None = None,
    eps_hi: float | None = None,
) -> np.ndarray:
    ref_eps = eps_ref if eps_ref is not None else args.eps12
    if args.eps_from_zero:
        lo = 0.0
        hi = eps_hi if eps_hi is not None else (ref_eps + args.eps_delta)
    else:
        lo = ref_eps - args.eps_delta
        hi = ref_eps + args.eps_delta
    return np.linspace(lo, hi, max(3, args.n_eps))


def _run_scan_core(
    ctx: ChainContext,
    ref: Metrics,
    *,
    eps_grid: np.ndarray,
    s1_bounds: tuple[float, float],
    s2_bounds: tuple[float, float],
    x0: tuple[float, float],
) -> tuple[list[dict], dict]:
    rows = scan_eps_isobar(
        ctx, ref,
        eps_values=eps_grid,
        x0=x0,
        s1_bounds=s1_bounds,
        s2_bounds=s2_bounds,
        wstar_tol=1e-8,
    )
    ok_rows = [r for r in rows if r["ok"]]
    if not ok_rows:
        raise SystemExit("no successful matches")
    best = min(ok_rows, key=lambda r: r["d_kl_bits"])
    return rows, best


def _run_scan(args: argparse.Namespace) -> None:
    ctx = build_context(
        N=args.N,
        u=(args.u0, args.u1),
        v=(args.v0, args.v1),
        r=args.r,
    )

    print(f"Reference (N={args.N}, u=({args.u0:g},{args.u1:g}), v=({args.v0:g},{args.v1:g}), r={args.r:g})")
    ref = evaluate_metrics(ctx, args.s1, args.s2, args.eps12, verbose=False)
    if not ref.ok:
        raise SystemExit(f"reference psi* failed resid={ref.psi_resid}")
    print(f"  s1={ref.s1:g}  s2={ref.s2:g}  eps12={ref.eps12:g}")
    print(f"  ⟨ln W⟩_ψ* = {ref.wstar_mean:.8f} nats")
    print(f"  D_KL(ψ*||φ*) = {ref.d_kl_bits:.6f} bits")
    print()

    eps_grid = _make_eps_grid(args)
    s1_bounds = (args.s1 - args.s_span, args.s1 + args.s_span)
    s2_bounds = (args.s2 - args.s_span, args.s2 + args.s_span)

    print(f"Scanning {len(eps_grid)} eps12 values in [{eps_grid[0]:g}, {eps_grid[-1]:g}] …")
    rows, best = _run_scan_core(
        ctx, ref,
        eps_grid=eps_grid,
        s1_bounds=s1_bounds,
        s2_bounds=s2_bounds,
        x0=(args.s1, args.s2),
    )

    ok_rows = [r for r in rows if r["ok"]]
    print(f"\nBest (lowest D at matched ⟨ln W⟩):")
    print(
        f"  eps12={best['eps12']:+.6g}  s1={best['s1']:+.6g}  s2={best['s2']:+.6g}\n"
        f"  D={best['d_kl_bits']:.6f} bits  (ΔD={best['d_delta']:+.6f})  "
        f"⟨ln W⟩ err={best['wstar_err']:.2e} nats"
    )

    print(f"\n{'eps12':>12} {'s1':>12} {'s2':>12} {'D [bits]':>12} {'ΔD':>12} {'Δ⟨lnW⟩':>12}")
    for r in ok_rows:
        mark = " *" if r is best else ""
        print(
            f"{r['eps12']:+.6g} {r['s1']:+.6g} {r['s2']:+.6g} "
            f"{r['d_kl_bits']:12.6f} {r['d_delta']:+.6f} {r['wstar_err']:+.2e}{mark}"
        )

    payload = {
        "mode": "scan",
        "reference": asdict(ref),
        "context": {
            "N": args.N, "r": args.r,
            "u": [args.u0, args.u1], "v": [args.v0, args.v1],
        },
        "scan": rows,
        "best": best,
    }

    out_dir = SCRIPT_DIR / "equiv_landscape" / "results"
    out_json = args.out or out_dir / f"wstar_isobar_N{args.N}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_json}")

    if args.plot:
        plot_path = out_json.with_suffix(".png")
        _plot_scan(ref, rows, plot_path)
        print(f"wrote {plot_path}")


def _run_batch_scan(args: argparse.Namespace) -> None:
    out_dir = SCRIPT_DIR / "equiv_landscape" / "results"
    cases = BATCH_SCAN_CASES
    if args.case:
        cases = [c for c in cases if c["label"] == args.case]
        if not cases:
            raise SystemExit(f"unknown --case {args.case!r}; choices: "
                             + ", ".join(c["label"] for c in BATCH_SCAN_CASES))

    summaries: list[dict] = []
    for case in cases:
        label = case["label"]
        print(f"\n{'=' * 60}\nCASE {label}\n{'=' * 60}", flush=True)
        ctx = build_context(
            N=args.N,
            u=(case["u0"], case["u1"]),
            v=(case["v0"], case["v1"]),
            r=args.r,
        )
        ref = evaluate_metrics(ctx, case["s1"], case["s2"], case["eps12"], verbose=False)
        if not ref.ok:
            print(f"  SKIP: reference failed resid={ref.psi_resid}", flush=True)
            continue
        print(
            f"  ref eps={ref.eps12:g}  (s1,s2)=({ref.s1:g},{ref.s2:g})  "
            f"⟨ln W⟩={ref.wstar_mean:.6g}  D={ref.d_kl_bits:.6f} bits",
            flush=True,
        )

        eps_grid = _make_eps_grid(args, eps_ref=case["eps12"], eps_hi=case.get("eps_hi"))
        s_span = case.get("s_span", args.s_span)
        s1_bounds = (case["s1"] - s_span, case["s1"] + s_span)
        s2_bounds = (case["s2"] - s_span, case["s2"] + s_span)
        print(f"  ε scan [{eps_grid[0]:g}, {eps_grid[-1]:g}]  n={len(eps_grid)}", flush=True)

        rows, best = _run_scan_core(
            ctx, ref,
            eps_grid=eps_grid,
            s1_bounds=s1_bounds,
            s2_bounds=s2_bounds,
            x0=(case["s1"], case["s2"]),
        )
        _, _, iso_at_ref, _ = find_min_d_on_wstar_isobar(
            ctx, ref.eps12, ref.wstar_mean,
            x0=(case["s1"], case["s2"]),
            s1_bounds=s1_bounds,
            s2_bounds=s2_bounds,
        )
        small_eps = best["eps12"] <= 1e-3
        print(
            f"  BEST: eps={best['eps12']:+.6g}  D={best['d_kl_bits']:.6f}  "
            f"ΔD={best['d_delta']:+.6f}  small_ε={small_eps}",
            flush=True,
        )
        if iso_at_ref.ok:
            print(
                f"  ISO@refε: D={iso_at_ref.d_kl_bits:.6f}  "
                f"ΔD={iso_at_ref.d_kl_bits - ref.d_kl_bits:+.6f}  "
                f"(s1,s2)=({iso_at_ref.s1:+.4g},{iso_at_ref.s2:+.4g})",
                flush=True,
            )

        case_payload = {
            "label": label,
            "reference": asdict(ref),
            "context": {"N": args.N, "r": args.r, "u": [case["u0"], case["u1"]], "v": [case["v0"], case["v1"]]},
            "scan": rows,
            "best": best,
        }
        case_json = out_dir / f"isobar_scan_{label}_N{args.N}.json"
        case_json.write_text(json.dumps(case_payload, indent=2))
        print(f"  wrote {case_json}", flush=True)

        if args.plot:
            _plot_scan(ref, rows, case_json.with_suffix(".png"))

        summaries.append({
            "label": label,
            "eps_ref": ref.eps12,
            "wstar_ref": ref.wstar_mean,
            "d_ref": ref.d_kl_bits,
            "d_min": best["d_kl_bits"],
            "d_delta": best["d_delta"],
            "eps_at_min": best["eps12"],
            "s1_at_min": best["s1"],
            "s2_at_min": best["s2"],
            "small_eps_wins": small_eps and best["d_delta"] < 0,
            "rel_drop": best["d_delta"] / ref.d_kl_bits if ref.d_kl_bits else float("nan"),
            "d_iso_at_ref_eps": iso_at_ref.d_kl_bits if iso_at_ref.ok else None,
            "d_iso_delta_at_ref_eps": (iso_at_ref.d_kl_bits - ref.d_kl_bits) if iso_at_ref.ok else None,
        })

    summary_path = out_dir / f"isobar_batch_summary_N{args.N}.json"
    summary_path.write_text(json.dumps({"cases": summaries}, indent=2))
    print(f"\nwrote {summary_path}", flush=True)

    if args.plot and summaries:
        _plot_batch_summary(summaries, out_dir / f"isobar_batch_summary_N{args.N}.png")
        print(f"wrote {out_dir / f'isobar_batch_summary_N{args.N}.png'}", flush=True)

    print(f"\n{'label':>22} {'D_ref':>10} {'D_min':>10} {'ΔD':>10} {'ε@min':>10} {'small ε?':>8}")
    for s in summaries:
        print(
            f"{s['label']:>22} {s['d_ref']:10.6f} {s['d_min']:10.6f} "
            f"{s['d_delta']:+10.6f} {s['eps_at_min']:10.6g} "
            f"{'yes' if s['small_eps_wins'] else 'no':>8}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("scan", "heatmap", "batch-scan"), default="heatmap")
    p.add_argument("--N", type=int, default=40)
    p.add_argument("--r", type=float, default=0.0)
    p.add_argument("--u0", type=float, default=5e-3)
    p.add_argument("--u1", type=float, default=5e-3)
    p.add_argument("--v0", type=float, default=5e-3)
    p.add_argument("--v1", type=float, default=5e-3)
    p.add_argument("--s1", type=float, default=1e-3)
    p.add_argument("--s2", type=float, default=-5e-4)
    p.add_argument("--eps12", type=float, default=1e-2)
    p.add_argument("--eps-delta", type=float, default=0.002)
    p.add_argument("--eps-from-zero", action=argparse.BooleanOptionalAction, default=True,
                   help="scan ε from 0 upward (default: on)")
    p.add_argument("--n-eps", type=int, default=17)
    p.add_argument("--case", type=str, default=None, help="batch-scan: run one case label only")
    p.add_argument("--s-span", type=float, default=0.003, help="half-width for s1,s2 search (scan mode)")
    p.add_argument("--s1-span", type=float, default=0.002, help="half-width for s1 grid (heatmap)")
    p.add_argument("--s2-span", type=float, default=0.001, help="half-width for s2 grid (heatmap)")
    p.add_argument("--grid-n", type=int, default=9, help="grid points per axis (heatmap)")
    p.add_argument("--wstar-tol", type=float, default=1e-8, help="|Δ⟨ln W⟩| tolerance [nats]")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.mode == "heatmap":
        _run_heatmap(args)
    elif args.mode == "batch-scan":
        _run_batch_scan(args)
    else:
        _run_scan(args)


if __name__ == "__main__":
    main()
