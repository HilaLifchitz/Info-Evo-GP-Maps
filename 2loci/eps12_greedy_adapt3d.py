#!/usr/bin/env python3
r"""
Greedy adaptation: ONE of (s1, s2, eps12) may change per step.

Alg 1: pick neighbor maximizing next eta(t+1) = Delta D / (N C).
Alg 2: pick neighbor maximizing next Delta D(t+1).

Baseline: fixed (s1, s2, eps12) at start. Includes 2×2 genotype fitness heatmaps
along the adaptive journey.

Run:
  cd 2loci
  python eps12_greedy_adapt3d.py --eps0 -0.2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_sigma_dynamics import cost_per_state, expectation  # noqa: E402
from eps12_delta_d_sweep import _make_cfg, track_d_c_per_generation  # noqa: E402
from simplex import (  # noqa: E402
    build_transition_matrix_row_stochastic,
    column_stochastic,
    compute_phi,
    fitness_exp,
    kl_divergence_bits,
    log_mean_fitness_genotype_simplex,
    mutation_matrix_asymmetric,
    omega_to_fitness_grid,
    params_from_config,
)

OUT_DIR = SCRIPT_DIR / "plots" / "eps12_greedy_adapt3d"
Mode = Literal["eta", "delta_d"]
ChangeAxis = Literal["s1", "s2", "eps12", "none"]

GENOTYPE_LABELS = np.array([["00", "01"], ["10", "11"]])


@dataclass(frozen=True)
class ParamState:
    s1: float
    s2: float
    eps12: float

    def key(self, *, s1_step: float, s2_step: float, eps_step: float) -> tuple[float, float, float]:
        return (
            round(self.s1 / s1_step) * s1_step,
            round(self.s2 / s2_step) * s2_step,
            round(self.eps12 / eps_step) * eps_step,
        )


@dataclass
class ModelParams:
    N: int = 30
    r: float = 0.0
    u: float = 0.01
    v: float = 0.01


@dataclass
class Landscape:
    state: ParamState
    P_col: np.ndarray
    c_state: np.ndarray
    omega: np.ndarray
    omega_grid: np.ndarray


@dataclass
class GreedyTrajectory3:
    mode: Mode
    start: ParamState
    t: np.ndarray
    s1_path: np.ndarray
    s2_path: np.ndarray
    eps_path: np.ndarray
    change_axis: list[str]
    d: np.ndarray
    c: np.ndarray
    delta_d: np.ndarray
    eta: np.ndarray
    chosen_score: np.ndarray
    mean_log_fit: np.ndarray   # E_psi[log(x·omega)] at each t [nats]
    mean_fit: np.ndarray       # E_psi[x·omega] at each t
    omega_grids: list[np.ndarray] = field(default_factory=list)


@dataclass
class GreedyComparison3:
    mode: Mode
    start: ParamState
    adaptive: GreedyTrajectory3
    fixed: GreedyTrajectory3


@dataclass
class SearchBounds:
    s1_min: float = 0.0005
    s1_max: float = 0.005
    s2_min: float = -0.005
    s2_max: float = 0.0005
    eps_min: float = -0.4
    eps_max: float = 0.4


def _clip_state(st: ParamState, b: SearchBounds) -> ParamState:
    return ParamState(
        s1=float(np.clip(st.s1, b.s1_min, b.s1_max)),
        s2=float(np.clip(st.s2, b.s2_min, b.s2_max)),
        eps12=float(np.clip(st.eps12, b.eps_min, b.eps_max)),
    )


def _snap_state(
    st: ParamState,
    *,
    s1_step: float,
    s2_step: float,
    eps_step: float,
    bounds: SearchBounds,
) -> ParamState:
    return _clip_state(ParamState(
        s1=round(st.s1 / s1_step) * s1_step,
        s2=round(st.s2 / s2_step) * s2_step,
        eps12=round(st.eps12 / eps_step) * eps_step,
    ), bounds)


def _one_param_neighbors(
    st: ParamState,
    *,
    s1_step: float,
    s2_step: float,
    eps_step: float,
    bounds: SearchBounds,
) -> list[tuple[ParamState, ChangeAxis]]:
    raw: list[tuple[ParamState, ChangeAxis]] = [(st, "none")]
    for ns1 in (st.s1 - s1_step, st.s1 + s1_step):
        raw.append((_clip_state(ParamState(ns1, st.s2, st.eps12), bounds), "s1"))
    for ns2 in (st.s2 - s2_step, st.s2 + s2_step):
        raw.append((_clip_state(ParamState(st.s1, ns2, st.eps12), bounds), "s2"))
    for ne in (st.eps12 - eps_step, st.eps12 + eps_step):
        raw.append((_clip_state(ParamState(st.s1, st.s2, ne), bounds), "eps12"))

    seen: set[tuple[float, float, float]] = set()
    uniq: list[tuple[ParamState, ChangeAxis]] = []
    for cand, axis in raw:
        snap = _snap_state(cand, s1_step=s1_step, s2_step=s2_step, eps_step=eps_step, bounds=bounds)
        k = snap.key(s1_step=s1_step, s2_step=s2_step, eps_step=eps_step)
        if k in seen:
            continue
        seen.add(k)
        uniq.append((ParamState(*k), axis))
    return uniq


class LandscapeCache:
    def __init__(
        self,
        params: ModelParams,
        fracs: np.ndarray,
        M_mut: np.ndarray,
        *,
        s1_step: float,
        s2_step: float,
        eps_step: float,
    ) -> None:
        self.params = params
        self.fracs = fracs
        self.M_mut = M_mut
        self.s1_step = s1_step
        self.s2_step = s2_step
        self.eps_step = eps_step
        self._cache: dict[tuple[float, float, float], Landscape] = {}
        self.n_builds = 0

    def get(self, st: ParamState, *, verbose: bool = False) -> Landscape:
        k = st.key(s1_step=self.s1_step, s2_step=self.s2_step, eps_step=self.eps_step)
        if k in self._cache:
            return self._cache[k]
        if verbose:
            print(f"    landscape s1={k[0]:g} s2={k[1]:g} eps12={k[2]:+.3f}", flush=True)
        cfg = params_from_config(
            _make_cfg(
                N=self.params.N, r=self.params.r, u=self.params.u, v=self.params.v,
                s1=k[0], s2=k[1], eps12=k[2],
            ),
            SCRIPT_DIR,
        )
        omega = fitness_exp(cfg.Sigma)
        c_state = cost_per_state(self.fracs, omega)
        T_row, _, _ = build_transition_matrix_row_stochastic(
            self.params.N, omega, self.M_mut, self.params.r,
            n_jobs=-1, label="g3d"[:20], verbose=False,
        )
        land = Landscape(
            state=ParamState(k[0], k[1], k[2]),
            P_col=column_stochastic(T_row),
            c_state=c_state,
            omega=omega,
            omega_grid=omega_to_fitness_grid(omega),
        )
        self._cache[k] = land
        self.n_builds += 1
        return land


def _mean_fitness_at_psi(
    psi: np.ndarray,
    fracs: np.ndarray,
    omega: np.ndarray,
) -> tuple[float, float]:
    """Return (E[log(x·ω)], E[x·ω]) under psi on the genotype lattice."""
    log_fit = log_mean_fitness_genotype_simplex(fracs, omega)
    wbar = fracs @ omega
    return float(np.sum(psi * log_fit)), float(np.sum(psi * wbar))


def _step_metrics(
    psi: np.ndarray,
    d_now: float,
    phi: np.ndarray,
    land: Landscape,
    N: int,
) -> tuple[np.ndarray, float, float, float, float]:
    psi_n = land.P_col @ psi
    s = float(psi_n.sum())
    if s > 0:
        psi_n = psi_n / s
    d_next = kl_divergence_bits(psi_n, phi)
    c_next = expectation(psi_n, land.c_state)
    dd = d_next - d_now
    nc = N * c_next
    eta = dd / nc if nc > 1e-300 else float("nan")
    return psi_n, d_next, c_next, dd, eta


def run_fixed_3(
    start: ParamState,
    phi: np.ndarray,
    cache: LandscapeCache,
    mode: Mode,
    *,
    max_t: int,
    N: int,
) -> GreedyTrajectory3:
    land = cache.get(start)
    fracs = cache.fracs
    t_arr, d, c = track_d_c_per_generation(phi.copy(), land.P_col, phi, land.c_state, max_t)
    dd = np.diff(d)
    nc = N * c[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(nc > 1e-300, dd / nc, np.nan)
    mean_log_fit = np.empty(max_t + 1)
    mean_fit = np.empty(max_t + 1)
    psi = phi.copy()
    mean_log_fit[0], mean_fit[0] = _mean_fitness_at_psi(psi, fracs, land.omega)
    for t in range(max_t):
        psi = land.P_col @ psi
        s = float(psi.sum())
        if s > 0:
            psi = psi / s
        mean_log_fit[t + 1], mean_fit[t + 1] = _mean_fitness_at_psi(psi, fracs, land.omega)
    return GreedyTrajectory3(
        mode=mode,
        start=start,
        t=t_arr,
        s1_path=np.full(max_t, start.s1),
        s2_path=np.full(max_t, start.s2),
        eps_path=np.full(max_t, start.eps12),
        change_axis=["none"] * max_t,
        d=d, c=c, delta_d=dd, eta=eta, chosen_score=eta.copy(),
        mean_log_fit=mean_log_fit,
        mean_fit=mean_fit,
        omega_grids=[land.omega_grid.copy()],
    )


def run_greedy_3(
    mode: Mode,
    start: ParamState,
    phi: np.ndarray,
    cache: LandscapeCache,
    bounds: SearchBounds,
    *,
    max_t: int,
    N: int,
    s1_step: float,
    s2_step: float,
    eps_step: float,
) -> GreedyTrajectory3:
    cur = start
    psi = phi.copy()
    d = np.empty(max_t + 1)
    c = np.empty(max_t + 1)
    s1_path = np.empty(max_t)
    s2_path = np.empty(max_t)
    eps_path = np.empty(max_t)
    dd_arr = np.empty(max_t)
    eta_arr = np.empty(max_t)
    score_arr = np.empty(max_t)
    change_axis: list[str] = []
    omega_grids: list[np.ndarray] = [cache.get(cur).omega_grid.copy()]
    mean_log_fit = np.empty(max_t + 1)
    mean_fit = np.empty(max_t + 1)
    fracs = cache.fracs

    land0 = cache.get(cur)
    d[0] = kl_divergence_bits(psi, phi)
    c[0] = expectation(psi, land0.c_state)
    mean_log_fit[0], mean_fit[0] = _mean_fitness_at_psi(psi, fracs, land0.omega)

    for _t in range(max_t):
        cands = _one_param_neighbors(
            cur, s1_step=s1_step, s2_step=s2_step, eps_step=eps_step, bounds=bounds,
        )
        best_st = cur
        best_axis: ChangeAxis = "none"
        best_score = -np.inf
        best_psi = psi
        best_d = d[_t]
        best_c = c[_t]
        best_dd = 0.0
        best_eta = float("nan")

        for cand_st, axis in cands:
            land = cache.get(cand_st)
            psi_n, d_n, c_n, dd, eta = _step_metrics(psi, d[_t], phi, land, N)
            score = eta if mode == "eta" else dd
            if not np.isfinite(score):
                continue
            if score > best_score + 1e-15 or (
                abs(score - best_score) <= 1e-15 and cand_st == cur
            ):
                best_score = score
                best_st = cand_st
                best_axis = axis
                best_psi = psi_n
                best_d = d_n
                best_c = c_n
                best_dd = dd
                best_eta = eta

        psi = best_psi
        d[_t + 1] = best_d
        c[_t + 1] = best_c
        s1_path[_t] = best_st.s1
        s2_path[_t] = best_st.s2
        eps_path[_t] = best_st.eps12
        dd_arr[_t] = best_dd
        eta_arr[_t] = best_eta
        score_arr[_t] = best_score
        change_axis.append(best_axis)
        cur = best_st
        land_cur = cache.get(cur)
        omega_grids.append(land_cur.omega_grid.copy())
        mean_log_fit[_t + 1], mean_fit[_t + 1] = _mean_fitness_at_psi(psi, fracs, land_cur.omega)

    return GreedyTrajectory3(
        mode=mode,
        start=start,
        t=np.arange(max_t + 1, dtype=float),
        s1_path=s1_path,
        s2_path=s2_path,
        eps_path=eps_path,
        change_axis=change_axis,
        d=d, c=c,
        delta_d=dd_arr,
        eta=eta_arr,
        chosen_score=score_arr,
        mean_log_fit=mean_log_fit,
        mean_fit=mean_fit,
        omega_grids=omega_grids,
    )


def _write_csv(path: Path, comp: GreedyComparison3) -> None:
    ad = comp.adaptive
    fx = comp.fixed
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t", "changed", "s1_ad", "s2_ad", "eps_ad",
            "s1_fix", "s2_fix", "eps_fix",
            "D_ad", "D_fix", "DeltaD_ad", "DeltaD_fix", "eta_ad", "eta_fix",
            "mean_log_fit_ad", "mean_log_fit_fix",
            "mean_fit_ad", "mean_fit_fix",
        ])
        for i in range(len(ad.delta_d)):
            w.writerow([
                i + 1, ad.change_axis[i],
                ad.s1_path[i], ad.s2_path[i], ad.eps_path[i],
                fx.s1_path[i], fx.s2_path[i], fx.eps_path[i],
                ad.d[i + 1], fx.d[i + 1],
                ad.delta_d[i], fx.delta_d[i],
                ad.eta[i], fx.eta[i],
                ad.mean_log_fit[i + 1], fx.mean_log_fit[i + 1],
                ad.mean_fit[i + 1], fx.mean_fit[i + 1],
            ])


def _heatmap_panel(ax, grid: np.ndarray, title: str) -> None:
    log_w = np.log10(np.maximum(grid, 1e-300))
    im = ax.imshow(log_w, cmap="viridis", aspect="equal", origin="upper")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{GENOTYPE_LABELS[row, col]}\n{grid[row, col]:.4f}",
                    ha="center", va="center", color="white", fontsize=7, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["loc2=0", "loc2=1"], fontsize=7)
    ax.set_yticklabels(["loc1=0", "loc1=1"], fontsize=7)
    ax.set_title(title, fontsize=8)
    return im


def plot_all(
    comp_eta: GreedyComparison3,
    comp_dd: GreedyComparison3,
    params: ModelParams,
    out_dir: Path,
    *,
    t_plot_max: int = 120,
    heatmap_stride: int = 15,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out_dir.mkdir(parents=True, exist_ok=True)
    tmax = min(t_plot_max, len(comp_eta.adaptive.delta_d))
    st = comp_eta.start

    # --- main 2×2 comparison ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    t1 = comp_eta.adaptive.t[1:tmax + 1]

    ax = axes[0, 0]
    ax.plot(t1, comp_eta.adaptive.eta[:tmax], "-", lw=1.8, color="C2", label="adaptive")
    ax.plot(t1, comp_eta.fixed.eta[:tmax], "--", lw=1.8, color="C7", label="fixed")
    ax.axhline(1.0, color="0.5", ls=":", lw=0.8)
    ax.set_ylabel(r"$\eta(t)$")
    ax.set_title("Alg 1: greedy eta (one param / step)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t1, comp_eta.adaptive.eps_path[:tmax], "-", lw=1.5, label=r"$\varepsilon_{12}$")
    ax.plot(t1, comp_eta.adaptive.s1_path[:tmax] * 100, "-", lw=1.5, label=r"$s_1\times100$")
    ax.plot(t1, comp_eta.adaptive.s2_path[:tmax] * 100, "-", lw=1.5, label=r"$s_2\times100$")
    ax.set_title("Alg 1: parameter journey", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t1, comp_dd.adaptive.delta_d[:tmax], "-", lw=1.8, color="C3", label="adaptive")
    ax.plot(t1, comp_dd.fixed.delta_d[:tmax], "--", lw=1.8, color="C7", label="fixed")
    ax.set_ylabel(r"$\Delta D(t)$")
    ax.set_title("Alg 2: greedy Delta D (one param / step)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    colors = {"s1": "C0", "s2": "C1", "eps12": "C4", "none": "0.7"}
    for i in range(tmax):
        ax.scatter(i + 1, 0, c=colors.get(comp_eta.adaptive.change_axis[i], "0.5"),
                   s=28, marker="|", linewidths=2)
    ax.set_yticks([])
    ax.set_xlabel("generation $t$")
    ax.set_title("Alg 1: which param changed", fontsize=10)
    ax.set_xlim(0.5, tmax + 0.5)
    leg = [Line2D([0], [0], color=colors[k], lw=3, label=k) for k in ("s1", "s2", "eps12", "none")]
    ax.legend(handles=leg, fontsize=7, loc="upper right")

    fig.suptitle(
        rf"3-param greedy ($s_1={st.s1:g}$, $s_2={st.s2:g}$, $\varepsilon_{{12}}={st.eps12:+.2f}$, "
        rf"$N={params.N}$, $r={params.r:g}$)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out_dir / "greedy_adapt3d_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}", flush=True)

    # --- fitness landscape heatmap journey ---
    for comp, tag, alg_name in [
        (comp_eta, "eta", "Alg 1"),
        (comp_dd, "delta_d", "Alg 2"),
    ]:
        traj = comp.adaptive
        snap_times = list(range(0, min(len(traj.omega_grids) - 1, tmax + 1), heatmap_stride))
        if snap_times[-1] != min(tmax, len(traj.omega_grids) - 1):
            snap_times.append(min(tmax, len(traj.omega_grids) - 1))
        n = len(snap_times)
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
        axes_flat = np.atleast_1d(axes).ravel()
        im_last = None
        for j, ti in enumerate(snap_times):
            ax = axes_flat[j]
            grid = traj.omega_grids[ti]
            if ti > 0:
                st_t = ParamState(traj.s1_path[ti - 1], traj.s2_path[ti - 1], traj.eps_path[ti - 1])
                ch = traj.change_axis[ti - 1]
            else:
                st_t = traj.start
                ch = "start"
            title = (
                f"t={ti}\n"
                f"s1={st_t.s1:g} s2={st_t.s2:g}\n"
                f"eps={st_t.eps12:+.2f} ({ch})"
            )
            im_last = _heatmap_panel(ax, grid, title)
        for j in range(n, len(axes_flat)):
            axes_flat[j].axis("off")
        if im_last is not None:
            fig.colorbar(im_last, ax=axes_flat.tolist(), shrink=0.7, label=r"$\log_{10}\omega(g)$")
        fig.suptitle(f"{alg_name}: genotype fitness $\\omega(g)=e^{{g^T\\Sigma g}}$ journey", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        p2 = out_dir / f"fitness_landscape_journey_{tag}.png"
        fig.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p2}", flush=True)

    # --- mean fitness trajectories (adaptive vs fixed start params) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    t0 = comp_eta.adaptive.t[: tmax + 1]

    for ax, comp, ylab, title in [
        (axes[0, 0], comp_eta, r"$\langle\log(x\!\cdot\!\omega)\rangle_\psi$ [nats]",
         "Alg 1: mean log fitness"),
        (axes[0, 1], comp_eta, r"$\langle x\!\cdot\!\omega\rangle_\psi$",
         "Alg 1: mean fitness"),
        (axes[1, 0], comp_dd, r"$\langle\log(x\!\cdot\!\omega)\rangle_\psi$ [nats]",
         "Alg 2: mean log fitness"),
        (axes[1, 1], comp_dd, r"$\langle x\!\cdot\!\omega\rangle_\psi$",
         "Alg 2: mean fitness"),
    ]:
        ad, fx = comp.adaptive, comp.fixed
        use_log = "log" in ylab
        ya = ad.mean_log_fit if use_log else ad.mean_fit
        yf = fx.mean_log_fit if use_log else fx.mean_fit
        ax.plot(t0, ya[: tmax + 1], "-", lw=1.8, color="C0", label="adaptive params")
        ax.plot(t0, yf[: tmax + 1], "--", lw=1.8, color="C7",
                label=rf"fixed ($s_1={st.s1:g}$, $s_2={st.s2:g}$, $\varepsilon_{{12}}={st.eps12:+.2f}$)")
        ax.set_xlabel("generation $t$")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        rf"Mean fitness under moving vs fixed $\Sigma$ "
        rf"($N={params.N}$, $r={params.r:g}$)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p3 = out_dir / "mean_fitness_trajectory.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p3}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Greedy s1/s2/eps12 adaptation")
    ap.add_argument("--s1", type=float, default=0.001)
    ap.add_argument("--s2", type=float, default=-0.001)
    ap.add_argument("--eps0", type=float, default=-0.2)
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--u", type=float, default=0.01)
    ap.add_argument("--v", type=float, default=0.01)
    ap.add_argument("--max-t", type=int, default=120)
    ap.add_argument("--s1-step", type=float, default=0.0005)
    ap.add_argument("--s2-step", type=float, default=0.0005)
    ap.add_argument("--eps-step", type=float, default=0.05)
    ap.add_argument("--t-plot-max", type=int, default=120)
    ap.add_argument("--heatmap-stride", type=int, default=15)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    params = ModelParams(N=args.N, r=args.r, u=args.u, v=args.v)
    bounds = SearchBounds()
    start = _snap_state(
        ParamState(args.s1, args.s2, args.eps0),
        s1_step=args.s1_step, s2_step=args.s2_step, eps_step=args.eps_step,
        bounds=bounds,
    )

    M_mut = mutation_matrix_asymmetric(np.array([params.u, params.u]), np.array([params.v, params.v]))
    params0 = params_from_config(
        _make_cfg(N=params.N, r=params.r, u=params.u, v=params.v,
                  s1=start.s1, s2=start.s2, eps12=start.eps12),
        SCRIPT_DIR,
    )
    phi, _, fracs, _, src = compute_phi(params0, M_mut)
    print(f"phi: {src}  start={start}", flush=True)

    cache = LandscapeCache(
        params, fracs, M_mut,
        s1_step=args.s1_step, s2_step=args.s2_step, eps_step=args.eps_step,
    )

    print("Alg 1 (eta)...", flush=True)
    ad_e = run_greedy_3(
        "eta", start, phi, cache, bounds,
        max_t=args.max_t, N=params.N,
        s1_step=args.s1_step, s2_step=args.s2_step, eps_step=args.eps_step,
    )
    fx_e = run_fixed_3(start, phi, cache, "eta", max_t=args.max_t, N=params.N)
    comp_eta = GreedyComparison3("eta", start, ad_e, fx_e)

    print("Alg 2 (Delta D)...", flush=True)
    ad_d = run_greedy_3(
        "delta_d", start, phi, cache, bounds,
        max_t=args.max_t, N=params.N,
        s1_step=args.s1_step, s2_step=args.s2_step, eps_step=args.eps_step,
    )
    fx_d = run_fixed_3(start, phi, cache, "delta_d", max_t=args.max_t, N=params.N)
    comp_dd = GreedyComparison3("delta_d", start, ad_d, fx_d)

    summary = {
        "start": {"s1": start.s1, "s2": start.s2, "eps12": start.eps12},
        "steps": {"s1": args.s1_step, "s2": args.s2_step, "eps12": args.eps_step},
        "landscapes_built": cache.n_builds,
        "alg1_eta": {
            "mean_eta_ad": float(np.nanmean(ad_e.eta)),
            "mean_eta_fix": float(np.nanmean(fx_e.eta)),
            "D_final_ad": float(ad_e.d[-1]),
            "D_final_fix": float(fx_e.d[-1]),
            "mean_log_fit_final_ad": float(ad_e.mean_log_fit[-1]),
            "mean_log_fit_final_fix": float(fx_e.mean_log_fit[-1]),
            "mean_fit_final_ad": float(ad_e.mean_fit[-1]),
            "mean_fit_final_fix": float(fx_e.mean_fit[-1]),
            "final_params": {
                "s1": float(ad_e.s1_path[-1]), "s2": float(ad_e.s2_path[-1]),
                "eps12": float(ad_e.eps_path[-1]),
            },
            "n_changes": int(sum(1 for c in ad_e.change_axis if c != "none")),
            "changes_by_axis": {
                k: int(sum(1 for c in ad_e.change_axis if c == k))
                for k in ("s1", "s2", "eps12", "none")
            },
        },
        "alg2_delta_d": {
            "sum_dd_ad": float(np.sum(ad_d.delta_d)),
            "sum_dd_fix": float(np.sum(fx_d.delta_d)),
            "D_final_ad": float(ad_d.d[-1]),
            "D_final_fix": float(fx_d.d[-1]),
            "mean_log_fit_final_ad": float(ad_d.mean_log_fit[-1]),
            "mean_log_fit_final_fix": float(fx_d.mean_log_fit[-1]),
            "mean_fit_final_ad": float(ad_d.mean_fit[-1]),
            "mean_fit_final_fix": float(fx_d.mean_fit[-1]),
            "final_params": {
                "s1": float(ad_d.s1_path[-1]), "s2": float(ad_d.s2_path[-1]),
                "eps12": float(ad_d.eps_path[-1]),
            },
            "changes_by_axis": {
                k: int(sum(1 for c in ad_d.change_axis if c == k))
                for k in ("s1", "s2", "eps12", "none")
            },
        },
    }
    print(json.dumps(summary, indent=2), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_csv(args.out_dir / "trajectory_alg1_eta.csv", comp_eta)
    _write_csv(args.out_dir / "trajectory_alg2_delta_d.csv", comp_dd)
    plot_all(comp_eta, comp_dd, params, args.out_dir,
             t_plot_max=args.t_plot_max, heatmap_stride=args.heatmap_stride)
    print(f"\nDone. {cache.n_builds} landscapes built. Outputs in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
