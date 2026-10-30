#!/usr/bin/env python3
r"""
Optimize eps12 for high eta(t) through time, or jointly with large Delta D.

Fixed background (s1, s2, N, u, v, r). Scans / refines eps12 to maximize:

  sustained_eta   mean eta(t) while D(t) < frac * KL_eq  (eta high *through* time)
  joint_throughput  sum_t Delta D(t) * eta(t) = sum_t Delta D(t)^2 / (N C(t))

Run:
  cd 2loci
  python eps12_optimize_eta.py
  python eps12_optimize_eta.py --r 0.1 --out-dir plots/eps12_optimize_eta
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eps12_delta_d_sweep import Eps12Run, analyze_run, run_one_eps12, _make_cfg  # noqa: E402
from simplex import compute_phi, mutation_matrix_asymmetric, params_from_config  # noqa: E402

ObjectiveName = Literal["sustained_eta", "joint_throughput"]

OUT_DIR = SCRIPT_DIR / "plots" / "eps12_optimize_eta"


@dataclass
class TrajectoryScores:
    eps12: float
    kl_eq: float
    t: np.ndarray
    d: np.ndarray
    c: np.ndarray
    delta_d: np.ndarray
    eta: np.ndarray
    t_eta: np.ndarray
    path_efficiency: float
    peak_eta: float
    t_peak_eta: float
    t_90: float
    sum_delta_d: float
    sum_nc: float
    sustained_eta: float
    joint_throughput: float
    mean_delta_d: float


@dataclass
class OptimizeResult:
    objective: str
    eps12_opt: float
    score_opt: float
    scores: TrajectoryScores
    eps12_grid: np.ndarray
    objective_grid: np.ndarray


@dataclass
class ModelParams:
    N: int = 30
    r: float = 0.0
    u: float = 0.01
    v: float = 0.01
    s1: float = 0.001
    s2: float = -0.001
    pergen_max_t: int = 500
    active_frac: float = 0.9  # mean eta while D < active_frac * KL_eq


def _trajectory_scores(
    run: Eps12Run,
    N: int,
    *,
    active_frac: float,
) -> TrajectoryScores:
    m = analyze_run(run, N)
    t = run.pergen_times
    d = run.pergen_kl_phi_bits
    c = run.pergen_c_cost
    delta_d = np.diff(d)
    t_eta = t[1:]
    nc = N * c[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(nc > 1e-300, delta_d / nc, np.nan)

    kl_eq = run.kl_eq
    if kl_eq > 1e-10:
        target = active_frac * kl_eq
        active = d[1:] < target - 1e-12
        if np.any(active):
            sustained = float(np.nanmean(eta[active]))
        else:
            sustained = float(np.nanmean(eta))
    else:
        sustained = float(np.nanmean(eta))

    dd = delta_d
    with np.errstate(divide="ignore", invalid="ignore"):
        joint = float(np.nansum(dd * eta))

    return TrajectoryScores(
        eps12=run.eps12,
        kl_eq=kl_eq,
        t=t,
        d=d,
        c=c,
        delta_d=delta_d,
        eta=eta,
        t_eta=t_eta,
        path_efficiency=m.path_efficiency,
        peak_eta=m.peak_eta,
        t_peak_eta=m.t_peak_eta,
        t_90=m.t_90,
        sum_delta_d=m.sum_delta_d,
        sum_nc=m.sum_nc,
        sustained_eta=sustained,
        joint_throughput=joint,
        mean_delta_d=float(np.nanmean(delta_d)),
    )


def evaluate_eps12(
    eps12: float,
    params: ModelParams,
    *,
    phi: np.ndarray,
    fracs: np.ndarray,
    M_mut: np.ndarray,
) -> TrajectoryScores:
    run = run_one_eps12(
        float(eps12),
        N=params.N,
        r=params.r,
        u=params.u,
        v=params.v,
        s1=params.s1,
        s2=params.s2,
        phi=phi,
        fracs=fracs,
        M_mut=M_mut,
        pergen_max_t=params.pergen_max_t,
        verbose=False,
    )
    return _trajectory_scores(run, params.N, active_frac=params.active_frac)


def objective_value(scores: TrajectoryScores, objective: ObjectiveName) -> float:
    if objective == "sustained_eta":
        return scores.sustained_eta
    if objective == "joint_throughput":
        return scores.joint_throughput
    raise ValueError(f"unknown objective {objective!r}")


def _prepare_neutral(params: ModelParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    M_mut = mutation_matrix_asymmetric(np.array([params.u, params.u]), np.array([params.v, params.v]))
    params0 = params_from_config(
        _make_cfg(N=params.N, r=params.r, u=params.u, v=params.v, s1=params.s1, s2=params.s2, eps12=0.0),
        SCRIPT_DIR,
    )
    phi, _, fracs, _, _ = compute_phi(params0, M_mut)
    return phi, fracs, M_mut


def scan_eps12(
    params: ModelParams,
    eps12_grid: np.ndarray,
    objective: ObjectiveName,
    *,
    phi: np.ndarray | None = None,
    fracs: np.ndarray | None = None,
    M_mut: np.ndarray | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[TrajectoryScores]]:
    if phi is None or fracs is None or M_mut is None:
        phi, fracs, M_mut = _prepare_neutral(params)

    objs: list[float] = []
    all_scores: list[TrajectoryScores] = []
    for i, eps in enumerate(eps12_grid):
        if verbose:
            print(f"  [{i + 1}/{len(eps12_grid)}] eps12={eps:+.3f}", flush=True)
        sc = evaluate_eps12(eps, params, phi=phi, fracs=fracs, M_mut=M_mut)
        all_scores.append(sc)
        objs.append(objective_value(sc, objective))
    return eps12_grid.copy(), np.asarray(objs, dtype=float), all_scores


def _refine_eps12(
    params: ModelParams,
    objective: ObjectiveName,
    x0: float,
    width: float,
    *,
    phi: np.ndarray,
    fracs: np.ndarray,
    M_mut: np.ndarray,
    n_eval: int = 25,
) -> tuple[float, float]:
    """Local golden-section search on [x0-width, x0+width]."""
    lo = x0 - width
    hi = x0 + width

    def f(x: float) -> float:
        return objective_value(
            evaluate_eps12(x, params, phi=phi, fracs=fracs, M_mut=M_mut),
            objective,
        )

    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(n_eval):
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    if fc >= fd:
        x_best, s_best = c, fc
    else:
        x_best, s_best = d, fd
    return float(x_best), float(s_best)


def optimize_eps12(
    params: ModelParams,
    objective: ObjectiveName,
    *,
    eps12_min: float = -0.4,
    eps12_max: float = 0.4,
    eps12_step: float = 0.05,
    refine: bool = True,
    verbose: bool = True,
) -> OptimizeResult:
    """Find eps12 maximizing the chosen objective (coarse grid + optional local refine)."""
    n = int(round((eps12_max - eps12_min) / eps12_step)) + 1
    grid = eps12_min + eps12_step * np.arange(n)
    grid = np.unique(np.round(np.clip(grid, eps12_min, eps12_max), 10))

    phi, fracs, M_mut = _prepare_neutral(params)
    if verbose:
        print(f"Scanning {len(grid)} eps12 values for objective={objective!r}", flush=True)

    eps_g, obj_g, all_sc = scan_eps12(
        params, grid, objective, phi=phi, fracs=fracs, M_mut=M_mut, verbose=verbose,
    )
    i_best = int(np.nanargmax(obj_g))
    eps_best = float(eps_g[i_best])
    score_best = float(obj_g[i_best])
    best_sc = all_sc[i_best]

    if refine and len(grid) >= 2:
        width = float(eps12_step)
        eps_ref, score_ref = _refine_eps12(
            params, objective, eps_best, width,
            phi=phi, fracs=fracs, M_mut=M_mut,
        )
        if score_ref > score_best:
            eps_best, score_best = eps_ref, score_ref
            best_sc = evaluate_eps12(eps_best, params, phi=phi, fracs=fracs, M_mut=M_mut)
        if verbose:
            print(f"  refined eps12={eps_best:+.5f}  score={score_best:.6g}", flush=True)

    return OptimizeResult(
        objective=objective,
        eps12_opt=eps_best,
        score_opt=score_best,
        scores=best_sc,
        eps12_grid=eps_g,
        objective_grid=obj_g,
    )


def optimize_both(
    params: ModelParams,
    **kwargs,
) -> dict[str, OptimizeResult]:
    return {
        "sustained_eta": optimize_eps12(params, "sustained_eta", **kwargs),
        "joint_throughput": optimize_eps12(params, "joint_throughput", **kwargs),
    }


def _write_scan_csv(path: Path, results: dict[str, OptimizeResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["objective", "eps12", "score"])
        for name, res in results.items():
            for eps, sc in zip(res.eps12_grid, res.objective_grid):
                w.writerow([name, eps, sc])
            w.writerow([name, res.eps12_opt, res.score_opt])
            w.writerow([])


def plot_optimization(
    params: ModelParams,
    results: dict[str, OptimizeResult],
    out_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=False)
    titles = {
        "sustained_eta": r"mean $\eta(t)$ while $D<0.9\,\mathrm{KL}_{\mathrm{eq}}$",
        "joint_throughput": r"$\sum_t \Delta D(t)\,\eta(t)$",
    }
    for ax, (name, res) in zip(axes, results.items()):
        ax.plot(res.eps12_grid, res.objective_grid, "o-", lw=1.5, ms=5, color="C0")
        ax.axvline(res.eps12_opt, color="C3", ls="--", lw=1.2,
                   label=rf"opt $\varepsilon_{{12}}={res.eps12_opt:+.3f}$")
        ax.scatter([res.eps12_opt], [res.score_opt], c="C3", s=80, zorder=5, edgecolors="k", lw=0.5)
        ax.set_xlabel(r"$\varepsilon_{12}$")
        ax.set_ylabel(titles[name])
        ax.set_title(titles[name], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        rf"$\varepsilon_{{12}}$ optimization ($N={params.N}$, $u=v={params.u:g}$, "
        rf"$r={params.r:g}$, $s_1={params.s1:g}$, $s_2={params.s2:g}$)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = out_dir / "objective_vs_eps12.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}", flush=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    colors = {"sustained_eta": "C2", "joint_throughput": "C4"}
    labels = {
        "sustained_eta": r"sustained-$\eta$ opt",
        "joint_throughput": r"joint throughput opt",
    }
    for name, res in results.items():
        sc = res.scores
        axes[0].plot(
            sc.t_eta, sc.eta, lw=1.8, color=colors[name],
            label=rf"{labels[name]} ($\varepsilon_{{12}}={res.eps12_opt:+.3f}$)",
        )
        axes[1].plot(
            sc.t_eta, sc.delta_d, lw=1.8, color=colors[name],
            label=rf"{labels[name]} ($\varepsilon_{{12}}={res.eps12_opt:+.3f}$)",
        )
    for ax in axes:
        ax.axhline(1.0 if ax is axes[0] else 0.0, color="0.5", ls=":", lw=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(0, min(120, params.pergen_max_t))
    axes[0].set_ylabel(r"$\eta(t)=\Delta D/(N C)$")
    axes[1].set_ylabel(r"$\Delta D(t)$ [bits/gen]")
    axes[1].set_xlabel("generation $t$")
    axes[0].set_title(r"Optimal trajectories compared", fontsize=10)
    fig.tight_layout()
    p2 = out_dir / "optimal_eta_and_delta_d.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p2}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Optimize eps12 for sustained eta or joint throughput")
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--u", type=float, default=0.01)
    ap.add_argument("--v", type=float, default=0.01)
    ap.add_argument("--s1", type=float, default=0.001)
    ap.add_argument("--s2", type=float, default=-0.001)
    ap.add_argument("--pergen-max-t", type=int, default=500)
    ap.add_argument("--active-frac", type=float, default=0.9)
    ap.add_argument("--eps12-min", type=float, default=-0.4)
    ap.add_argument("--eps12-max", type=float, default=0.4)
    ap.add_argument("--eps12-step", type=float, default=0.05)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    params = ModelParams(
        N=args.N, r=args.r, u=args.u, v=args.v,
        s1=args.s1, s2=args.s2,
        pergen_max_t=args.pergen_max_t,
        active_frac=args.active_frac,
    )
    kw = dict(
        eps12_min=args.eps12_min,
        eps12_max=args.eps12_max,
        eps12_step=args.eps12_step,
        refine=not args.no_refine,
    )

    print(
        f"Optimizing eps12: N={params.N}, u=v={params.u:g}, r={params.r:g}, "
        f"s1={params.s1:g}, s2={params.s2:g}",
        flush=True,
    )
    results = optimize_both(params, **kw)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, res in results.items():
        sc = res.scores
        summary[name] = {
            "eps12_opt": res.eps12_opt,
            "score_opt": res.score_opt,
            "kl_eq": sc.kl_eq,
            "path_efficiency": sc.path_efficiency,
            "peak_eta": sc.peak_eta,
            "sustained_eta": sc.sustained_eta,
            "joint_throughput": sc.joint_throughput,
            "sum_delta_d": sc.sum_delta_d,
            "t_90": sc.t_90,
        }
        print(
            f"\n[{name}]\n"
            f"  eps12* = {res.eps12_opt:+.5f}\n"
            f"  score  = {res.score_opt:.6g}\n"
            f"  KL_eq  = {sc.kl_eq:.4f} bits\n"
            f"  bar_eta = {sc.path_efficiency:.4f},  peak_eta = {sc.peak_eta:.4f}\n"
            f"  sustained_eta = {sc.sustained_eta:.4f},  joint = {sc.joint_throughput:.6g}\n"
            f"  sum Delta D = {sc.sum_delta_d:.4f},  t_90 = {sc.t_90:.0f}",
            flush=True,
        )

    (args.out_dir / "optimize_summary.json").write_text(json.dumps(summary, indent=2))
    _write_scan_csv(args.out_dir / "objective_scan.csv", results)
    plot_optimization(params, results, args.out_dir)
    print(f"\nDone. Outputs in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
