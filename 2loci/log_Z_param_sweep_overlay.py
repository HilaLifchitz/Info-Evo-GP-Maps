#!/usr/bin/env python3
r"""
Extended parameter sweep + ALL scenarios on two overlay figures.

Metrics vs $\varepsilon_{12}$ (x-axis):
  - $KL_{\mathrm{eq}}$ (final information)
  - $\sum\Delta D / \sum N\langle C\rangle$  (path metric, was $\bar\eta$)
  - $\max \Delta D/(N\langle C\rangle)$       (peak, was peak $\eta$)
  - $\log Z/N$  (= free fitness at equilibrium, continuous diffusion)

Vertical dashed line: predicted information dip
  $\varepsilon^*$ from Cov/Var formula (see ``f2b_reference.eps_star_dip_prediction``).

Outputs:
  plots/f2b/log_Z_all_scenarios_normalized.png
  plots/f2b/log_Z_all_scenarios_raw.png
  plots/f2b/all_scenarios.csv
  plots/f2b/param_sweeps/<label>/data.csv  (cached per scenario)

Run:
  cd 2loci
  python log_Z_param_sweep_overlay.py
  python log_Z_param_sweep_overlay.py --reuse   # skip finished scenarios
  python log_Z_param_sweep_overlay.py --quick   # smaller set
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

from f2b_reference import eps_star_dip_prediction  # noqa: E402
from log_Z_param_sweep import (  # noqa: E402
    OUT,
    Scenario,
    _cfg_from_scenario,
    _run_scenario,
    _write_csv,
)

_STR_FIELDS = frozenset({"scenario", "scenario_title"})

PLOT_OUT = SCRIPT_DIR / "plots" / "f2b"

METRICS = [
    ("kl_eq_bits", r"$KL_{\mathrm{eq}}$ [bits]", True),
    ("path_efficiency", r"$\overline{\Delta D/(N\langle C\rangle)}$", False),
    ("peak_eta", r"$\max \Delta D / (N\langle C\rangle)$", False),
    ("log_Z_over_N", r"$\log Z/N$ [nats]", True),
]

NOTE = (
    r"$\log Z/N$ = free fitness at equilibrium (continuous diffusion, cond-$\alpha$ $Z$).  "
    r"Dashed vertical: predicted $KL_{\mathrm{eq}}$ dip $\varepsilon^*$."
)


def _eps12_grid_for(sc: Scenario, *, step: float) -> tuple[float, float, float]:
    """Adaptive eps12 range from N and selection scale."""
    ns = max(sc.N * abs(sc.s1), sc.N * abs(sc.s2), 1.0)
    if ns >= 2.5:
        half = min(0.15, max(0.08, 4.0 / sc.N))
    elif ns >= 1.5:
        half = min(0.25, max(0.1, 5.0 / sc.N))
    else:
        half = min(0.4, max(0.12, 8.0 / sc.N))
    return -half, half, step


def build_scenarios(*, quick: bool) -> tuple[Scenario, ...]:
    """Cartesian-ish coverage: N, u/v/r, (s1,s2), and high N·s_i cases."""
    out: list[Scenario] = []

    def add(label: str, title: str, N: int, s1: float, s2: float,
            u: tuple[float, float], v: tuple[float, float], r: float) -> None:
        out.append(Scenario(label, title, N, s1, s2, u, v, r))

    # --- baseline family (delta_d-like) ---
    for N, tag in ((20, "N20"), (30, "N30"), (40, "N40"), (50, "N50")):
        add(f"{tag}_dd", rf"{tag} $\delta d$", N, 0.001, -0.02, (0.01, 0.01), (0.01, 0.01), 0.0)

    if not quick:
        add("N30_weak_u", r"weak $u\!=\!10^{-3}$", 30, 0.001, -0.02, (1e-3, 1e-3), (1e-3, 1e-3), 0.0)
        add("N30_strong_u", r"strong $u=0.05$", 30, 0.001, -0.02, (0.05, 0.05), (0.05, 0.05), 0.0)
        add("N30_high_r", r"$r=0.01$", 30, 0.001, -0.02, (0.01, 0.01), (0.01, 0.01), 0.01)
        add("N30_asym_uv", r"asym $u\!\neq\!v$", 30, 0.001, -0.02, (0.02, 0.01), (0.01, 0.02), 0.0)
        add("N30_pos_s", r"$s_1\!+\!s_2\!>0$", 30, 0.001, 5e-4, (0.01, 0.01), (0.01, 0.01), 0.0)
        add("N30_dip_sr", r"KL-dip $s,u,r$", 30, -1e-3, 5e-4, (1e-3, 1e-3), (1e-3, 1e-3), 1e-3)

    # --- high N·s_i ("crazy") at several N (max N=50; N=100 simplex is impractical) ---
    ns_pairs = [
        (1, -2), (2, -2), (3, -3), (4, -4), (5, -5),
        (2, -1), (3, -2), (1, -1), (2, -3), (3, -1),
    ]
    Ns_list = [10, 15, 20, 30, 50] if not quick else [10, 30, 50]
    for N in Ns_list:
        for ns1, ns2 in ns_pairs:
            if max(abs(ns1), abs(ns2)) > min(8, N // 2):
                continue
            s1, s2 = ns1 / N, ns2 / N
            label = f"N{N}_Ns{ns1}m{abs(ns2)}"
            title = rf"$N={N}$, $N s_1={ns1}$, $N s_2={ns2}$"
            add(label, title, N, s1, s2, (0.01, 0.01), (0.01, 0.01), 0.0)

    # --- extra u/r on N=30 Ns cases ---
    if not quick:
        for ns1, ns2 in [(1, -2), (3, -3), (2, -1)]:
            s1, s2 = ns1 / 30.0, ns2 / 30.0
            add(f"N30_Ns{ns1}m{abs(ns2)}_wu", rf"Ns + weak $u$", 30, s1, s2,
                (1e-3, 1e-3), (1e-3, 1e-3), 0.0)
            add(f"N30_Ns{ns1}m{abs(ns2)}_hr", rf"Ns + $r$", 30, s1, s2,
                (0.01, 0.01), (0.01, 0.01), 0.01)

    # dedupe labels
    seen: set[str] = set()
    uniq: list[Scenario] = []
    for sc in out:
        if sc.label in seen:
            continue
        seen.add(sc.label)
        uniq.append(sc)
    return tuple(uniq)


def _load_scenario_csv(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    try:
        rows: list[dict] = []
        with path.open() as f:
            for row in csv.DictReader(f):
                rec: dict = {}
                for k, v in row.items():
                    if k in _STR_FIELDS or not v:
                        rec[k] = v if v else float("nan")
                    else:
                        rec[k] = float(v)
                rows.append(rec)
        return rows if len(rows) >= 3 else None
    except (OSError, ValueError, KeyError):
        return None


def _compute_eps_star(rows: list[dict]) -> float:
    """Use first row's scenario params + stored prediction if present."""
    if rows and "eps_star_pred" in rows[0] and np.isfinite(rows[0]["eps_star_pred"]):
        return float(rows[0]["eps_star_pred"])
    return float("nan")


def _run_all(
    scenarios: tuple[Scenario, ...],
    *,
    step: float,
    pergen_max_t: int,
    reuse: bool,
) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {}
    for si, sc in enumerate(scenarios, 1):
        cache = OUT / sc.label / "data.csv"
        if reuse and cache.exists():
            cached = _load_scenario_csv(cache)
            if cached and len(cached) >= 3:
                need_eps = not np.isfinite(cached[0].get("eps_star_pred", np.nan))
                if need_eps:
                    from log_Z_param_sweep import _make_cfg_vec  # noqa: E402
                    from simplex import compute_phi, make_simplex_grid, mutation_matrix_asymmetric, params_from_config  # noqa: E402

                    u = np.asarray(sc.u, dtype=float)
                    v = np.asarray(sc.v, dtype=float)
                    M_mut = mutation_matrix_asymmetric(u, v)
                    fracs = make_simplex_grid(sc.N)
                    params0 = params_from_config(
                        _make_cfg_vec(N=sc.N, r=sc.r, u=u, v=v, s1=sc.s1, s2=sc.s2, eps12=0.0),
                        SCRIPT_DIR,
                    )
                    phi, _, _, _, _ = compute_phi(params0, M_mut)
                    eps_star = eps_star_dip_prediction(phi, fracs, sc.s1, sc.s2, eps_lin=0.0)
                    for row in cached:
                        row["eps_star_pred"] = float(eps_star)
                        row.setdefault("scenario_title", sc.title)
                    _write_csv(cached, cache)
                print(f"[{si}/{len(scenarios)}] {sc.label}  (cached)", flush=True)
                results[sc.label] = cached
                continue

        lo, hi, st = _eps12_grid_for(sc, step=step)
        cfg = _cfg_from_scenario(
            sc, eps12_min=lo, eps12_max=hi, eps12_step=st, pergen_max_t=pergen_max_t,
        )
        print(f"[{si}/{len(scenarios)}] {sc.label}  eps12∈[{lo:g},{hi:g}] step={st:g}", flush=True)

        from log_Z_param_sweep import _make_cfg_vec  # noqa: E402
        from simplex import compute_phi, make_simplex_grid, mutation_matrix_asymmetric, params_from_config  # noqa: E402

        u = np.asarray(sc.u, dtype=float)
        v = np.asarray(sc.v, dtype=float)
        M_mut = mutation_matrix_asymmetric(u, v)
        fracs = make_simplex_grid(sc.N)
        params0 = params_from_config(
            _make_cfg_vec(N=sc.N, r=sc.r, u=u, v=v, s1=sc.s1, s2=sc.s2, eps12=0.0),
            SCRIPT_DIR,
        )
        phi, _, _, _, _ = compute_phi(params0, M_mut)
        eps_star = eps_star_dip_prediction(phi, fracs, sc.s1, sc.s2, eps_lin=0.0)

        rows = _run_scenario(sc, cfg)
        for row in rows:
            row["eps_star_pred"] = float(eps_star)
            row["scenario_title"] = sc.title
        results[sc.label] = rows
        _write_csv(rows, cache)
    return results


def _norm(y: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(y)), float(np.max(y))
    if hi - lo < 1e-15:
        return np.zeros_like(y)
    return (y - lo) / (hi - lo)


def _plot_all_overlay(
    results: dict[str, list[dict]],
    scenarios: tuple[Scenario, ...],
    out_path: Path,
    *,
    normalized: bool,
) -> None:
    n_sc = len(scenarios)
    cmap = plt.cm.turbo(np.linspace(0.05, 0.95, max(n_sc, 1)))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=False)
    axes = axes.flat

    for j, (key, ylab, show_dip) in enumerate(METRICS):
        ax = axes[j]
        for i, sc in enumerate(scenarios):
            rows = results.get(sc.label)
            if not rows:
                continue
            eps = np.array([r["eps12"] for r in rows])
            o = np.argsort(eps)
            eps = eps[o]
            y = np.array([rows[k][key] for k in o])
            if normalized:
                y = _norm(y)
            ax.plot(eps, y, "-", color=cmap[i], lw=1.2, alpha=0.85)
            ax.scatter(eps, y, s=12, color=cmap[i], alpha=0.9, edgecolors="none", zorder=3)
            if show_dip and np.isfinite(rows[0].get("eps_star_pred", np.nan)):
                est = float(rows[0]["eps_star_pred"])
                if eps.min() <= est <= eps.max():
                    ax.axvline(est, color=cmap[i], ls="--", lw=0.8, alpha=0.55)

        ax.axvline(0, color="0.35", ls=":", lw=0.9, zorder=0)
        ax.set_xlabel(r"$\varepsilon_{12}$")
        ax.set_ylabel(ylab if not normalized else ylab.replace("[bits]", "").replace("[nats]", "") + " (norm.)")
        ax.grid(True, alpha=0.3)
        ax.set_title(ylab, fontsize=10)

    fig.suptitle(
        ("Normalized shape comparison — all parameter sweeps\n" if normalized else
         "Raw values — all parameter sweeps\n") + NOTE,
        fontsize=10,
    )

    # colorbar for scenario index
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(0, n_sc - 1))
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("scenario index", fontsize=8)
    labels_short = [f"{i}:{sc.label[:18]}" for i, sc in enumerate(scenarios)]
    cbar.ax.yaxis.set_tick_params(labelsize=5)
    # tick at a few indices
    ticks = np.linspace(0, n_sc - 1, min(n_sc, 8)).astype(int)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([labels_short[t] for t in ticks])

    fig.tight_layout(rect=(0, 0, 0.9, 0.88))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--reuse", action="store_true", help="load cached per-scenario CSVs")
    p.add_argument("--step", type=float, default=None, help="eps12 step (default 0.05 or 0.1 quick)")
    p.add_argument("--plot-only", action="store_true", help="plot from cached CSVs only (skip simulation)")
    p.add_argument("--pergen-max-t", type=int, default=500)
    args = p.parse_args()

    scenarios = build_scenarios(quick=args.quick)
    step = args.step if args.step is not None else (0.1 if args.quick else 0.05)

    if args.plot_only:
        results: dict[str, list[dict]] = {}
        done: list[Scenario] = []
        for sc in scenarios:
            cached = _load_scenario_csv(OUT / sc.label / "data.csv")
            if cached:
                results[sc.label] = cached
                done.append(sc)
        if not done:
            print("No cached scenario data found.", flush=True)
            return
        scenarios = tuple(done)
        print(f"Plotting {len(scenarios)} cached scenarios\n", flush=True)
    else:
        print(f"{len(scenarios)} scenarios, eps12 step={step}, t=0..{args.pergen_max_t}\n", flush=True)
        results = _run_all(scenarios, step=step, pergen_max_t=args.pergen_max_t, reuse=args.reuse)

    all_rows: list[dict] = []
    for sc in scenarios:
        all_rows.extend(results[sc.label])

    PLOT_OUT.mkdir(parents=True, exist_ok=True)
    _write_csv(all_rows, PLOT_OUT / "all_scenarios.csv")
    (PLOT_OUT / "param_sweeps_index.json").write_text(json.dumps({
        "n_scenarios": len(scenarios),
        "labels": [s.label for s in scenarios],
        "step": step,
    }, indent=2))

    _plot_all_overlay(
        results, scenarios,
        PLOT_OUT / "log_Z_all_scenarios_normalized.png",
        normalized=True,
    )
    _plot_all_overlay(
        results, scenarios,
        PLOT_OUT / "log_Z_all_scenarios_raw.png",
        normalized=False,
    )

    print(f"\nwrote {PLOT_OUT / 'log_Z_all_scenarios_normalized.png'}")
    print(f"wrote {PLOT_OUT / 'log_Z_all_scenarios_raw.png'}")
    print(f"wrote {PLOT_OUT / 'all_scenarios.csv'}")


if __name__ == "__main__":
    main()
