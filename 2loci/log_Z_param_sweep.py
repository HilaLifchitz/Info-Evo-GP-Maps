#!/usr/bin/env python3
r"""
Multi-parameter sweep: metrics vs $\varepsilon_{12}$ for several $(s_1,s_2,u,v,r)$.

For each scenario, computes per eps12:
  KL_eq, bar eta, peak eta, log Z/N (cond-alpha)

Outputs per scenario (same as log_Z_vs_dynamics):
  plots/f2b/param_sweeps/<label>/log_Z_metrics_vs_eps12.png
  plots/f2b/param_sweeps/<label>/data.csv

Summary grid across scenarios:
  plots/f2b/param_sweeps_grid.png
  plots/f2b/param_sweeps_normalized_grid.png
  plots/f2b/all_scenarios.csv

Run:
  cd 2loci
  python log_Z_param_sweep.py
  python log_Z_param_sweep.py --fine          # eps12 step 0.05
  python log_Z_param_sweep.py --quick         # fewer scenarios, coarser eps12
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from compare_sigma_dynamics import (  # noqa: E402
    cond_alpha_reference_pmf,
    cost_per_state,
    expectation,
    free_f2,
    free_f3,
)
from eps12_delta_d_sweep import (  # noqa: E402
    Eps12Run,
    analyze_run,
    track_d_c_f_per_generation,
    _eps12_grid,
)
from f2b_reference import log_Z_cond_alpha  # noqa: E402
from log_Z_vs_dynamics import _plot_vs_eps12, _slope  # noqa: E402
from simplex import (  # noqa: E402
    build_sigma,
    build_transition_matrix_row_stochastic,
    column_stochastic,
    compute_phi,
    fitness_exp,
    free_fitness_1,
    kl_divergence_bits,
    log_mean_fitness_genotype_simplex,
    make_simplex_grid,
    mutation_matrix_asymmetric,
    params_from_config,
    try_selection_equilibrium,
    w_potential_per_state,
)

OUT = SCRIPT_DIR / "plots" / "f2b" / "param_sweeps"


@dataclass(frozen=True)
class Scenario:
    label: str
    title: str
    N: int
    s1: float
    s2: float
    u: tuple[float, float]
    v: tuple[float, float]
    r: float


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("delta_d", r"$\delta d$ default", 30, 0.001, -0.02, (0.01, 0.01), (0.01, 0.01), 0.0),
    Scenario("weak_mut", r"weak $u\!=\!v\!=\!10^{-3}$", 30, 0.001, -0.02, (1e-3, 1e-3), (1e-3, 1e-3), 0.0),
    Scenario("dip_regime", r"KL-dip $(s_1,s_2,u,r)$", 30, -1e-3, 5e-4, (1e-3, 1e-3), (1e-3, 1e-3), 1e-3),
    Scenario("high_r", r"high $r=0.01$", 30, 0.001, -0.02, (0.01, 0.01), (0.01, 0.01), 0.01),
    Scenario("asym_uv", r"asymmetric $u\!\neq\!v$", 30, 0.001, -0.02, (0.02, 0.01), (0.01, 0.02), 0.0),
    Scenario("pos_sum", r"$s_1+s_2>0$", 30, 0.001, 5e-4, (0.01, 0.01), (0.01, 0.01), 0.0),
    Scenario("strong_u", r"strong $u=0.05$", 30, 0.001, -0.02, (0.05, 0.05), (0.05, 0.05), 0.0),
    Scenario("N40", r"$N=40$", 40, 0.001, -0.02, (0.01, 0.01), (0.01, 0.01), 0.0),
)

QUICK_SCENARIOS: tuple[Scenario, ...] = (
    SCENARIOS[0], SCENARIOS[1], SCENARIOS[3], SCENARIOS[5], SCENARIOS[6],
)


def _cfg_from_scenario(sc: Scenario, *, eps12_min: float, eps12_max: float, eps12_step: float, pergen_max_t: int) -> dict:
    return {
        "N": sc.N,
        "r": sc.r,
        "u": list(sc.u),
        "v": list(sc.v),
        "s1": sc.s1,
        "s2": sc.s2,
        "eps12_min": eps12_min,
        "eps12_max": eps12_max,
        "eps12_step": eps12_step,
        "pergen_max_t": pergen_max_t,
    }


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
        "u": np.asarray(u, dtype=float).tolist(),
        "v": np.asarray(v, dtype=float).tolist(),
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


def _run_one_eps12_vec(
    eps12: float,
    *,
    sc: Scenario,
    phi: np.ndarray,
    fracs: np.ndarray,
    M_mut: np.ndarray,
    q_cond: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    pergen_max_t: int,
) -> Eps12Run:
    N = sc.N
    params = params_from_config(
        _make_cfg_vec(N=N, r=sc.r, u=u, v=v, s1=sc.s1, s2=sc.s2, eps12=float(eps12)),
        SCRIPT_DIR,
    )
    omega = fitness_exp(params.Sigma)
    c_state = cost_per_state(fracs, omega)
    w_ps = w_potential_per_state(fracs, omega, M_mut, N)
    wstar_ln = log_mean_fitness_genotype_simplex(fracs, omega)
    T_row, _, _ = build_transition_matrix_row_stochastic(
        N, omega, M_mut, sc.r, n_jobs=-1,
        label=f"{sc.label}_e{eps12:.2g}"[:20], verbose=False,
    )
    P_col = column_stochastic(T_row)
    psi_star, resid, ok = try_selection_equilibrium(P_col)
    if not ok or psi_star is None:
        raise RuntimeError(f"{sc.label} eps12={eps12}: psi* failed resid={resid}")
    kl_eq = kl_divergence_bits(psi_star, phi)
    wstar_eq = expectation(psi_star, wstar_ln)
    pg = track_d_c_f_per_generation(
        phi.copy(), P_col, phi, c_state, fracs, w_ps, wstar_ln,
        q_cond, psi_star, N, pergen_max_t,
    )
    pg_t, pg_d, pg_c, pg_f1, pg_f2, pg_f3, pg_f4, pg_w, pg_i2, pg_i3 = pg
    return Eps12Run(
        eps12=float(eps12),
        kl_eq=float(kl_eq),
        f1_eq=float(free_fitness_1(psi_star, w_ps, N)),
        f2_eq=float(free_f2(psi_star, wstar_ln, phi, N)),
        f3_eq=float(free_f3(psi_star, wstar_ln, q_cond, N)),
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


def _run_scenario(sc: Scenario, cfg: dict, *, return_runs: bool = False) -> list[dict] | tuple[list[dict], list[Eps12Run]]:
    u = np.asarray(sc.u, dtype=float)
    v = np.asarray(sc.v, dtype=float)
    N = sc.N
    eps_grid = _eps12_grid(cfg)
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

    rows: list[dict] = []
    runs: list[Eps12Run] = []
    max_t = int(cfg["pergen_max_t"])
    for i, eps12 in enumerate(eps_grid, 1):
        print(f"    [{i}/{len(eps_grid)}] eps12={eps12:+.3f}", flush=True)
        run = _run_one_eps12_vec(
            float(eps12), sc=sc, phi=phi, fracs=fracs, M_mut=M_mut,
            q_cond=q_cond, u=u, v=v, pergen_max_t=max_t,
        )
        m = analyze_run(run, N)
        omega = fitness_exp(build_sigma(sc.s1, sc.s2, float(eps12)))
        log_z_n_f = float(log_Z_cond_alpha(fracs, omega, M_mut, N, mode="interior")["log_Z_over_N"])
        t = run.pergen_times
        f4 = run.pergen_f4
        mean_c = float(np.mean(run.pergen_c_cost))
        rows.append({
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
            "log_Z_over_N": log_z_n_f,
            "log_Z0_over_N": float(log_z0),
            "kl_eq_bits": m.kl_eq,
            "path_efficiency": m.path_efficiency,
            "peak_eta": m.peak_eta,
            "mean_c_cost": mean_c,
            "d_final_bits": m.d_final,
            "t_90": m.t_90,
            "f4_slope_early": _slope(t, f4, 0, 50),
        })
        if return_runs:
            runs.append(run)
    if return_runs:
        return rows, runs
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _scenario_subtitle(sc: Scenario) -> str:
    return (
        rf"$N={sc.N}$, $s_1={sc.s1:g}$, $s_2={sc.s2:g}$, "
        rf"$u=({sc.u[0]:g},{sc.u[1]:g})$, $v=({sc.v[0]:g},{sc.v[1]:g})$, $r={sc.r:g}$"
    )


def _plot_scenario_panels(rows: list[dict], sc: Scenario, out_dir: Path) -> None:
    """Four stacked panels vs eps12 for one scenario."""
    cfg_rows = [{k: r[k] for k in r if k not in ("scenario", "N", "s1", "s2", "u0", "u1", "v0", "v1", "r")}
                for r in rows]
    # map keys for _plot_vs_eps12
    mapped = []
    for r in cfg_rows:
        mapped.append({
            "eps12": r["eps12"],
            "kl_eq_bits": r["kl_eq_bits"],
            "path_efficiency": r["path_efficiency"],
            "peak_eta": r["peak_eta"],
            "log_Z_over_N": r["log_Z_over_N"],
            "log_Z0_over_N": r["log_Z0_over_N"],
        })
    _plot_vs_eps12(mapped, out_dir, N=sc.N, s1=sc.s1, s2=sc.s2)
    # rename default output to keep per-scenario dir clean
    default = out_dir / "log_Z_metrics_vs_eps12.png"
    target = out_dir / "metrics_vs_eps12.png"
    if default.exists() and default != target:
        default.rename(target)


def _plot_summary_grid(
    results: dict[str, list[dict]],
    scenarios: tuple[Scenario, ...],
    out_path: Path,
    *,
    normalized: bool = False,
) -> None:
    n_sc = len(scenarios)
    metrics = [
        ("kl_eq_bits", r"$KL_{\mathrm{eq}}$ [bits]", "C0"),
        ("path_efficiency", r"$\bar\eta$", "C2"),
        ("peak_eta", r"peak $\eta$", "C3"),
        ("log_Z_over_N", r"$\log Z/N$", "C4"),
    ]
    fig, axes = plt.subplots(len(metrics), n_sc, figsize=(3.4 * n_sc, 9), sharex=True, squeeze=False)

    for j, sc in enumerate(scenarios):
        rows = results[sc.label]
        eps = np.array([r["eps12"] for r in rows])
        o = np.argsort(eps)
        eps = eps[o]
        for i, (key, ylab, col) in enumerate(metrics):
            ax = axes[i, j]
            y = np.array([rows[k][key] for k in o])
            if normalized:
                lo, hi = float(y.min()), float(y.max())
                y = (y - lo) / (hi - lo) if hi - lo > 1e-15 else np.zeros_like(y)
            ax.plot(eps, y, "o-", color=col, lw=1.4, ms=3)
            ax.axvline(0, color="0.35", ls=":", lw=0.8)
            if i == 0:
                ax.set_title(sc.title + "\n" + _scenario_subtitle(sc), fontsize=7)
            if j == 0:
                ax.set_ylabel(ylab if not normalized else ylab + " (norm.)", fontsize=8)
            if i == len(metrics) - 1:
                ax.set_xlabel(r"$\varepsilon_{12}$", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

    ylabel = "normalized" if normalized else "value"
    fig.suptitle(
        rf"Metrics vs $\varepsilon_{{12}}$ across parameter scenarios ({ylabel})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="5 scenarios, eps12 step 0.1")
    p.add_argument("--fine", action="store_true", help="eps12 step 0.05 (default 0.1)")
    p.add_argument("--pergen-max-t", type=int, default=500)
    p.add_argument("--eps12-min", type=float, default=-0.4)
    p.add_argument("--eps12-max", type=float, default=0.4)
    args = p.parse_args()

    scenarios = QUICK_SCENARIOS if args.quick else SCENARIOS
    eps_step = 0.05 if args.fine else (0.2 if args.quick else 0.1)
    grid_cfg = {
        "eps12_min": args.eps12_min,
        "eps12_max": args.eps12_max,
        "eps12_step": eps_step,
        "pergen_max_t": args.pergen_max_t,
    }

    all_rows: list[dict] = []
    results: dict[str, list[dict]] = {}

    print(f"Running {len(scenarios)} scenarios, eps12 step={eps_step}, t=0..{args.pergen_max_t}\n", flush=True)
    for si, sc in enumerate(scenarios, 1):
        print(f"[{si}/{len(scenarios)}] {sc.label}: {sc.title}", flush=True)
        cfg = _cfg_from_scenario(sc, **grid_cfg)
        rows = _run_scenario(sc, cfg)
        results[sc.label] = rows
        all_rows.extend(rows)
        sc_dir = OUT / sc.label
        _write_csv(rows, sc_dir / "data.csv")
        _plot_scenario_panels(rows, sc, sc_dir)
        print(f"  wrote {sc_dir / 'metrics_vs_eps12.png'}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    _write_csv(all_rows, OUT.parent / "all_scenarios.csv")
    meta = {
        "scenarios": [s.label for s in scenarios],
        "eps12_step": eps_step,
        "pergen_max_t": args.pergen_max_t,
    }
    (OUT / "index.json").write_text(json.dumps(meta, indent=2))

    _plot_summary_grid(
        results, scenarios,
        OUT.parent / "param_sweeps_grid.png",
        normalized=False,
    )
    _plot_summary_grid(
        results, scenarios,
        OUT.parent / "param_sweeps_normalized_grid.png",
        normalized=True,
    )

    print(f"\nwrote {OUT.parent / 'all_scenarios.csv'}")
    print(f"wrote {OUT.parent / 'param_sweeps_grid.png'}")
    print(f"wrote {OUT.parent / 'param_sweeps_normalized_grid.png'}")
    print(f"per-scenario plots in {OUT}/")


if __name__ == "__main__":
    main()
