#!/usr/bin/env python3
r"""
Greedy adaptive eps12 path (not global optimization).

At each generation t, from the current distribution psi(t) and current eps12,
try eps12 values in a local neighborhood. Commit one step with the neighbor
that maximizes the *next* generation's score:

  alg1 (eta)      pick eps maximizing eta(t+1) = Delta D / (N C)
  alg2 (delta_d)  pick eps maximizing Delta D(t+1)

Compare each adaptive run to a baseline that keeps eps12 fixed at eps0.

Run:
  cd 2loci
  python eps12_greedy_adapt.py --eps0 -0.2
  python eps12_greedy_adapt.py --eps0 0.1 --max-t 150 --nhbr 2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
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
    mutation_matrix_asymmetric,
    params_from_config,
)

OUT_DIR = SCRIPT_DIR / "plots" / "eps12_greedy_adapt"
Mode = Literal["eta", "delta_d"]


@dataclass
class ModelParams:
    N: int = 30
    r: float = 0.0
    u: float = 0.01
    v: float = 0.01
    s1: float = 0.001
    s2: float = -0.001


@dataclass
class EpsLandscape:
    eps12: float
    P_col: np.ndarray
    c_state: np.ndarray


@dataclass
class GreedyTrajectory:
    mode: Mode
    eps0: float
    t: np.ndarray
    eps_path: np.ndarray       # eps used for step t -> t+1 (length max_t)
    d: np.ndarray              # D(t), length max_t+1
    c: np.ndarray
    delta_d: np.ndarray        # length max_t
    eta: np.ndarray            # length max_t
    chosen_score: np.ndarray   # score that was maximized each step


@dataclass
class GreedyComparison:
    mode: Mode
    eps0: float
    adaptive: GreedyTrajectory
    fixed: GreedyTrajectory


def _eps_grid(lo: float, hi: float, step: float) -> np.ndarray:
    n = int(round((hi - lo) / step)) + 1
    g = lo + step * np.arange(n)
    return np.unique(np.round(np.clip(g, lo, hi), 10))


def _snap_to_grid(eps: float, grid: np.ndarray) -> float:
    i = int(np.argmin(np.abs(grid - eps)))
    return float(grid[i])


def _neighbors(eps: float, grid: np.ndarray, nhbr: int) -> np.ndarray:
    i = int(np.argmin(np.abs(grid - eps)))
    lo = max(0, i - nhbr)
    hi = min(len(grid) - 1, i + nhbr)
    return grid[lo : hi + 1].copy()


def precompute_landscapes(
    params: ModelParams,
    eps_grid: np.ndarray,
    *,
    phi: np.ndarray,
    fracs: np.ndarray,
    M_mut: np.ndarray,
    verbose: bool = True,
) -> dict[float, EpsLandscape]:
    cache: dict[float, EpsLandscape] = {}
    for i, eps in enumerate(eps_grid):
        if verbose:
            print(f"  precompute [{i + 1}/{len(eps_grid)}] eps12={eps:+.3f}", flush=True)
        cfg = params_from_config(
            _make_cfg(N=params.N, r=params.r, u=params.u, v=params.v,
                      s1=params.s1, s2=params.s2, eps12=float(eps)),
            SCRIPT_DIR,
        )
        omega = fitness_exp(cfg.Sigma)
        c_state = cost_per_state(fracs, omega)
        T_row, _, _ = build_transition_matrix_row_stochastic(
            params.N, omega, M_mut, params.r, n_jobs=-1,
            label=f"g{eps:.3g}"[:20], verbose=False,
        )
        cache[float(eps)] = EpsLandscape(
            eps12=float(eps),
            P_col=column_stochastic(T_row),
            c_state=c_state,
        )
    return cache


def _step_metrics(
    psi: np.ndarray,
    d_now: float,
    phi: np.ndarray,
    land: EpsLandscape,
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


def run_fixed(
    eps0: float,
    phi: np.ndarray,
    land: EpsLandscape,
    *,
    max_t: int,
    N: int,
) -> GreedyTrajectory:
    t_arr, d, c = track_d_c_per_generation(phi.copy(), land.P_col, phi, land.c_state, max_t)
    dd = np.diff(d)
    nc = N * c[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(nc > 1e-300, dd / nc, np.nan)
    return GreedyTrajectory(
        mode="eta",
        eps0=eps0,
        t=t_arr,
        eps_path=np.full(max_t, eps0),
        d=d,
        c=c,
        delta_d=dd,
        eta=eta,
        chosen_score=eta.copy(),
    )


def run_greedy(
    mode: Mode,
    eps0: float,
    phi: np.ndarray,
    landscapes: dict[float, EpsLandscape],
    eps_grid: np.ndarray,
    *,
    max_t: int,
    N: int,
    nhbr: int,
) -> GreedyTrajectory:
    eps_cur = _snap_to_grid(eps0, eps_grid)
    psi = phi.copy()
    d = np.empty(max_t + 1)
    c = np.empty(max_t + 1)
    eps_path = np.empty(max_t)
    dd_arr = np.empty(max_t)
    eta_arr = np.empty(max_t)
    score_arr = np.empty(max_t)

    d[0] = kl_divergence_bits(psi, phi)
    c[0] = expectation(psi, landscapes[eps_cur].c_state)

    for t in range(max_t):
        nbrs = _neighbors(eps_cur, eps_grid, nhbr)
        best_eps = eps_cur
        best_score = -np.inf
        best_psi = psi
        best_d = d[t]
        best_c = c[t]
        best_dd = 0.0
        best_eta = float("nan")

        for eps_cand in nbrs:
            land = landscapes[float(eps_cand)]
            psi_n, d_n, c_n, dd, eta = _step_metrics(psi, d[t], phi, land, N)
            score = eta if mode == "eta" else dd
            if not np.isfinite(score):
                continue
            if score > best_score + 1e-15 or (
                score > best_score - 1e-15 and eps_cand == eps_cur
            ):
                best_score = score
                best_eps = float(eps_cand)
                best_psi = psi_n
                best_d = d_n
                best_c = c_n
                best_dd = dd
                best_eta = eta

        psi = best_psi
        d[t + 1] = best_d
        c[t + 1] = best_c
        eps_path[t] = best_eps
        dd_arr[t] = best_dd
        eta_arr[t] = best_eta
        score_arr[t] = best_score
        eps_cur = best_eps

    return GreedyTrajectory(
        mode=mode,
        eps0=eps0,
        t=np.arange(max_t + 1, dtype=float),
        eps_path=eps_path,
        d=d,
        c=c,
        delta_d=dd_arr,
        eta=eta_arr,
        chosen_score=score_arr,
    )


def compare_greedy(
    mode: Mode,
    eps0: float,
    params: ModelParams,
    landscapes: dict[float, EpsLandscape],
    eps_grid: np.ndarray,
    *,
    max_t: int,
    nhbr: int,
    phi: np.ndarray,
) -> GreedyComparison:
    eps_snap = _snap_to_grid(eps0, eps_grid)
    adaptive = run_greedy(
        mode, eps_snap, phi, landscapes, eps_grid,
        max_t=max_t, N=params.N, nhbr=nhbr,
    )
    fixed = run_fixed(
        eps_snap, phi, landscapes[float(eps_snap)],
        max_t=max_t, N=params.N,
    )
    fixed.mode = mode
    return GreedyComparison(mode=mode, eps0=eps_snap, adaptive=adaptive, fixed=fixed)


def _write_csv(path: Path, comp: GreedyComparison) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ad, fx = comp.adaptive, comp.fixed
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t", "eps_adaptive", "eps_fixed",
            "D_adaptive", "D_fixed",
            "DeltaD_adaptive", "DeltaD_fixed",
            "eta_adaptive", "eta_fixed",
            "chosen_score",
        ])
        for i in range(len(ad.delta_d)):
            w.writerow([
                i + 1,
                ad.eps_path[i], fx.eps_path[i],
                ad.d[i + 1], fx.d[i + 1],
                ad.delta_d[i], fx.delta_d[i],
                ad.eta[i], fx.eta[i],
                ad.chosen_score[i],
            ])


def plot_comparison(
    comp_eta: GreedyComparison,
    comp_dd: GreedyComparison,
    params: ModelParams,
    out_dir: Path,
    *,
    t_plot_max: int | None = 120,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    tmax = t_plot_max or len(comp_eta.adaptive.t) - 1

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Alg1: eta
    ax = axes[0, 0]
    t1 = comp_eta.adaptive.t[1 : tmax + 1]
    ax.plot(t1, comp_eta.adaptive.eta[:tmax], "-", lw=1.8, color="C2",
            label=rf"adaptive (start $\varepsilon_{{12}}={comp_eta.eps0:+.2f}$)")
    ax.plot(t1, comp_eta.fixed.eta[:tmax], "--", lw=1.8, color="C7",
            label=rf"fixed $\varepsilon_{{12}}={comp_eta.eps0:+.2f}$")
    ax.axhline(1.0, color="0.5", ls=":", lw=0.8, alpha=0.6)
    ax.set_ylabel(r"$\eta(t)=\Delta D/(N C)$")
    ax.set_title(r"Alg 1: greedy $\eta$ path vs fixed $\varepsilon_{12}$", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t1, comp_eta.adaptive.eps_path[:tmax], "-", lw=1.8, color="C0")
    ax.axhline(comp_eta.eps0, color="0.5", ls="--", lw=1.0, alpha=0.7)
    ax.set_ylabel(r"$\varepsilon_{12}(t)$")
    ax.set_title(r"Alg 1: $\varepsilon_{12}$ journey", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Alg2: Delta D
    ax = axes[1, 0]
    t2 = comp_dd.adaptive.t[1 : tmax + 1]
    ax.plot(t2, comp_dd.adaptive.delta_d[:tmax], "-", lw=1.8, color="C3",
            label="adaptive")
    ax.plot(t2, comp_dd.fixed.delta_d[:tmax], "--", lw=1.8, color="C7",
            label=rf"fixed $\varepsilon_{{12}}={comp_dd.eps0:+.2f}$")
    ax.set_ylabel(r"$\Delta D(t)$ [bits/gen]")
    ax.set_title(r"Alg 2: greedy $\Delta D$ path vs fixed $\varepsilon_{12}$", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t2, comp_dd.adaptive.eps_path[:tmax], "-", lw=1.8, color="C0")
    ax.axhline(comp_dd.eps0, color="0.5", ls="--", lw=1.0, alpha=0.7)
    ax.set_ylabel(r"$\varepsilon_{12}(t)$")
    ax.set_xlabel("generation $t$")
    ax.set_title(r"Alg 2: $\varepsilon_{12}$ journey", fontsize=10)
    ax.grid(True, alpha=0.3)

    for ax in axes[0, :]:
        ax.set_xlim(1, tmax)
    for ax in axes[1, :]:
        ax.set_xlim(1, tmax)

    fig.suptitle(
        rf"Greedy $\varepsilon_{{12}}$ adaptation ($N={params.N}$, $u=v={params.u:g}$, "
        rf"$r={params.r:g}$, $s_1={params.s1:g}$, $s_2={params.s2:g}$)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out_dir / "greedy_adapt_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}", flush=True)

    # Separate journey plots (step path as stairs)
    for comp, tag in [(comp_eta, "eta"), (comp_dd, "delta_d")]:
        fig, ax = plt.subplots(figsize=(9, 3.5))
        t = comp.adaptive.t[1 : tmax + 1]
        ax.step(t, comp.adaptive.eps_path[:tmax], where="post", lw=1.8, color="C0")
        ax.axhline(comp.eps0, color="C3", ls="--", lw=1.2,
                   label=rf"fixed start $\varepsilon_{{12}}={comp.eps0:+.2f}$")
        ax.set_xlabel("generation $t$")
        ax.set_ylabel(r"$\varepsilon_{12}$ chosen for next step")
        alg_label = "Alg 1 (eta)" if tag == "eta" else "Alg 2 (Delta D)"
        ax.set_title(f"{alg_label}: epsilon_12 journey", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p2 = out_dir / f"eps12_journey_{tag}.png"
        fig.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p2}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Greedy eps12 adaptation (eta or Delta D)")
    ap.add_argument("--eps0", type=float, default=-0.2, help="starting eps12 (snapped to grid)")
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--u", type=float, default=0.01)
    ap.add_argument("--v", type=float, default=0.01)
    ap.add_argument("--s1", type=float, default=0.001)
    ap.add_argument("--s2", type=float, default=-0.001)
    ap.add_argument("--max-t", type=int, default=120)
    ap.add_argument("--nhbr", type=int, default=2,
                    help="neighbor radius in grid steps (2 => ±2*step on eps12 grid)")
    ap.add_argument("--eps-min", type=float, default=-0.4)
    ap.add_argument("--eps-max", type=float, default=0.4)
    ap.add_argument("--eps-step", type=float, default=0.05)
    ap.add_argument("--t-plot-max", type=int, default=120)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    params = ModelParams(N=args.N, r=args.r, u=args.u, v=args.v, s1=args.s1, s2=args.s2)
    eps_grid = _eps_grid(args.eps_min, args.eps_max, args.eps_step)
    eps0 = _snap_to_grid(args.eps0, eps_grid)

    M_mut = mutation_matrix_asymmetric(np.array([params.u, params.u]), np.array([params.v, params.v]))
    params0 = params_from_config(
        _make_cfg(N=params.N, r=params.r, u=params.u, v=params.v,
                  s1=params.s1, s2=params.s2, eps12=0.0),
        SCRIPT_DIR,
    )
    phi, _, fracs, _, src = compute_phi(params0, M_mut)
    print(f"phi: {src}", flush=True)
    print(f"Precomputing {len(eps_grid)} landscapes...", flush=True)
    landscapes = precompute_landscapes(params, eps_grid, phi=phi, fracs=fracs, M_mut=M_mut)

    print(f"\nAlg 1: greedy eta (eps0={eps0:+.3f}, nhbr={args.nhbr})", flush=True)
    comp_eta = compare_greedy(
        "eta", eps0, params, landscapes, eps_grid,
        max_t=args.max_t, nhbr=args.nhbr, phi=phi,
    )
    print(f"Alg 2: greedy Delta D (eps0={eps0:+.3f})", flush=True)
    comp_dd = compare_greedy(
        "delta_d", eps0, params, landscapes, eps_grid,
        max_t=args.max_t, nhbr=args.nhbr, phi=phi,
    )

    ad_e, fx_e = comp_eta.adaptive, comp_eta.fixed
    ad_d, fx_d = comp_dd.adaptive, comp_dd.fixed
    summary = {
        "eps0": eps0,
        "max_t": args.max_t,
        "nhbr": args.nhbr,
        "eps_grid_step": args.eps_step,
        "alg1_eta": {
            "mean_eta_adaptive": float(np.nanmean(ad_e.eta)),
            "mean_eta_fixed": float(np.nanmean(fx_e.eta)),
            "D_final_adaptive": float(ad_e.d[-1]),
            "D_final_fixed": float(fx_e.d[-1]),
            "eps_final_adaptive": float(ad_e.eps_path[-1]),
            "n_eps_switches": int(np.sum(np.diff(ad_e.eps_path) != 0)),
        },
        "alg2_delta_d": {
            "sum_delta_d_adaptive": float(np.sum(ad_d.delta_d)),
            "sum_delta_d_fixed": float(np.sum(fx_d.delta_d)),
            "mean_delta_d_adaptive": float(np.nanmean(ad_d.delta_d)),
            "mean_delta_d_fixed": float(np.nanmean(fx_d.delta_d)),
            "D_final_adaptive": float(ad_d.d[-1]),
            "D_final_fixed": float(fx_d.d[-1]),
            "eps_final_adaptive": float(ad_d.eps_path[-1]),
            "n_eps_switches": int(np.sum(np.diff(ad_d.eps_path) != 0)),
        },
    }
    print(json.dumps(summary, indent=2), flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_csv(args.out_dir / "trajectory_alg1_eta.csv", comp_eta)
    _write_csv(args.out_dir / "trajectory_alg2_delta_d.csv", comp_dd)
    plot_comparison(comp_eta, comp_dd, params, args.out_dir, t_plot_max=args.t_plot_max)
    print(f"\nDone. Outputs in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
