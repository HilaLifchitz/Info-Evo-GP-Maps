#!/usr/bin/env python3
r"""
Fine $\varepsilon_{12}$ scan around $0$ and $\varepsilon^*$ to compare $C(t)$ and $\eta(t)$.

For each parameter case:
  - dense grid spanning $[\min(0,\varepsilon^*),\max(0,\varepsilon^*)]$ (plus margin)
  - anchors at $\varepsilon_{12}=0$ and predicted $\varepsilon^*$
  - compares time-mean $\langle C\rangle_t$ at $\varepsilon=0$ vs $\varepsilon=\varepsilon^*$

Outputs per case in ``plots/f2b/eps_star_fine/<label>/``:
  C_mean_vs_eps12_fine.png
  C_compare_zero_vs_star.png
  C_vs_time_by_kl_eq.png / C_vs_time_by_eps12.png
  eta_vs_time_by_kl_eq.png / eta_vs_time_by_eps12.png
  fine_summary.json

Run:
  cd 2loci
  python log_Z_eps_star_fine_scan.py --case N30_mix_pp_sm_u1e3_s11e-3_s2m-5e-4
  python log_Z_eps_star_fine_scan.py --only-n 30 40 --reuse
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

from eps12_delta_d_sweep import (  # noqa: E402
    Eps12Run,
    _add_eps12_colorbar,
    _add_score_colorbar,
    _format_scalar,
    _score_colors,
)
from log_Z_eps_star_gallery import (  # noqa: E402
    GalleryCase,
    build_cases,
    predict_eps_star,
)
from log_Z_param_sweep import (  # noqa: E402
    Scenario,
    _make_cfg_vec,
    _run_one_eps12_vec,
    _slope,
    _write_csv,
)
from compare_sigma_dynamics import cond_alpha_reference_pmf  # noqa: E402
from eps12_delta_d_sweep import analyze_run  # noqa: E402
from f2b_reference import log_Z_cond_alpha  # noqa: E402
from simplex import (  # noqa: E402
    build_sigma,
    compute_phi,
    fitness_exp,
    make_simplex_grid,
    mutation_matrix_asymmetric,
    params_from_config,
)

OUT = SCRIPT_DIR / "plots" / "f2b" / "eps_star_fine"
GALLERY = SCRIPT_DIR / "plots" / "f2b" / "eps_star_gallery_N30_N40"


def _fine_eps_grid(eps_star: float, *, half: float, step: float) -> np.ndarray:
    """Grid from min(0,eps*)-margin to max(0,eps*)+margin; always includes 0 and eps*."""
    lo = min(0.0, eps_star) - half
    hi = max(0.0, eps_star) + half
    grid = np.arange(lo, hi + 0.5 * step, step)
    anchors = np.array([0.0, float(eps_star)], dtype=float)
    return np.sort(np.unique(np.round(np.concatenate([grid, anchors]), 12)))


def _case_from_gallery_dir(d: Path) -> GalleryCase | None:
    meta_path = d / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    sc = Scenario(
        meta["label"],
        meta["label"],
        int(meta["N"]),
        float(meta["s1"]),
        float(meta["s2"]),
        tuple(meta["u"]),
        tuple(meta["v"]),
        float(meta["r"]),
    )
    eps_star = float(meta.get("eps_star_pred", predict_eps_star(sc.N, sc.s1, sc.s2, sc.u, sc.v, sc.r)))
    return GalleryCase(sc.label, sc.title, sc, eps_star)


def _discover_cases(*, only_n: tuple[int, ...] | None, case_label: str | None) -> list[GalleryCase]:
    if case_label:
        for root in (GALLERY, SCRIPT_DIR / "plots" / "f2b" / "eps_star_gallery"):
            d = root / case_label
            if c := _case_from_gallery_dir(d):
                return [c]
        sc = None
        for c in build_cases(quick=False, include_zero=False, min_abs_eps=0.0, only_n=only_n):
            if c.label == case_label:
                return [c]
        raise SystemExit(f"case not found: {case_label}")

    cases: list[GalleryCase] = []
    for d in sorted(GALLERY.iterdir()) if GALLERY.exists() else []:
        if d.is_dir() and (c := _case_from_gallery_dir(d)):
            if only_n is None or c.scenario.N in only_n:
                cases.append(c)
    if cases:
        return cases
    return list(build_cases(quick=False, include_zero=False, min_abs_eps=2e-4, only_n=only_n))


def _eta_trajectory(run: Eps12Run, N: int) -> tuple[np.ndarray, np.ndarray]:
    t = run.pergen_times
    d = run.pergen_kl_phi_bits
    c = run.pergen_c_cost
    delta_d = np.diff(d)
    nc = N * c[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(nc > 1e-300, delta_d / nc, np.nan)
    return t[1:], eta


def _c_variation(run: Eps12Run) -> float:
    c = run.pergen_c_cost
    mu = float(np.mean(c))
    if mu < 1e-300:
        return float("nan")
    return float(np.std(c) / mu)


def _run_fine(case: GalleryCase, *, half: float, step: float, pergen_max_t: int) -> tuple[list[dict], list[Eps12Run]]:
    sc = case.scenario
    u = np.asarray(sc.u, dtype=float)
    v = np.asarray(sc.v, dtype=float)
    N = sc.N
    eps_grid = _fine_eps_grid(case.eps_star, half=half, step=step)
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
    for i, eps12 in enumerate(eps_grid, 1):
        print(f"    [{i}/{len(eps_grid)}] eps12={eps12:+.2e}", flush=True)
        run = _run_one_eps12_vec(
            float(eps12), sc=sc, phi=phi, fracs=fracs, M_mut=M_mut,
            q_cond=q_cond, u=u, v=v, pergen_max_t=pergen_max_t,
        )
        m = analyze_run(run, N)
        t = run.pergen_times
        f4 = run.pergen_f4
        mean_c = float(np.mean(run.pergen_c_cost))
        c_cv = _c_variation(run)
        rows.append({
            "scenario": sc.label,
            "N": N,
            "eps12": float(eps12),
            "kl_eq_bits": m.kl_eq,
            "mean_c_cost": mean_c,
            "c_cv": c_cv,
            "path_efficiency": m.path_efficiency,
            "peak_eta": m.peak_eta,
            "log_Z_over_N": float(log_Z_cond_alpha(
                fracs, fitness_exp(build_sigma(sc.s1, sc.s2, float(eps12))), M_mut, N,
                mode="interior",
            )["log_Z_over_N"]),
            "log_Z0_over_N": float(log_z0),
            "f4_slope_early": _slope(t, f4, 0, 50),
        })
        runs.append(run)
    return rows, runs


def _nearest_run(runs: list[Eps12Run], target: float) -> Eps12Run:
    return min(runs, key=lambda r: abs(r.eps12 - target))


def _plot_c_mean_vs_eps(rows: list[dict], case: GalleryCase, out_dir: Path) -> None:
    sc = case.scenario
    eps = np.array([r["eps12"] for r in rows])
    o = np.argsort(eps)
    eps = eps[o]
    mc = np.array([rows[i]["mean_c_cost"] for i in o])
    kl = np.array([rows[i]["kl_eq_bits"] for i in o])

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(eps, mc, "o-", color="C5", lw=1.6, ms=4, label=r"$\langle C\rangle_t$")
    ax1.axvline(0.0, color="0.35", ls=":", lw=1.0)
    ax1.axvline(case.eps_star, color="C3", ls="--", lw=1.2,
                label=rf"$\varepsilon^*={_format_scalar(case.eps_star, signed=True, ref_span=float(eps.max()-eps.min()))}$")
    ax1.set_xlabel(r"$\varepsilon_{12}$")
    ax1.set_ylabel(r"$\langle C\rangle_t$")
    ax2 = ax1.twinx()
    ax2.plot(eps, kl, "s--", color="C0", lw=1.2, ms=3, alpha=0.7, label=r"$KL_{\mathrm{eq}}$")
    ax2.set_ylabel(r"$KL_{\mathrm{eq}}$ [bits]", color="C0")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper left", fontsize=8)
    ax1.set_title(
        rf"Fine scan: $\langle C\rangle_t$ vs $\varepsilon_{{12}}$ ($N={sc.N}$, "
        rf"$s_1={sc.s1:g}$, $s_2={sc.s2:g}$)",
        fontsize=9,
    )
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "C_mean_vs_eps12_fine.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_c_compare(summary: dict, case: GalleryCase, out_dir: Path) -> None:
    sc = case.scenario
    labels = [r"$\varepsilon=0$", r"$\varepsilon=\varepsilon^*$", r"nearest grid pt"]
    keys = ["C_at_zero", "C_at_eps_star", "C_at_nearest_star"]
    vals = [summary[k] for k in keys]
    kl_vals = [summary["KL_at_zero"], summary["KL_at_eps_star"], summary["KL_at_nearest_star"]]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(labels, vals, color=["C0", "C3", "C5"], alpha=0.85)
    axes[0].set_ylabel(r"$\langle C\rangle_t$")
    axes[0].set_title(r"Mean cost: $\varepsilon=0$ vs $\varepsilon^*$")
    for i, v in enumerate(vals):
        axes[0].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    pct = summary.get("C_ratio_zero_over_star")
    if np.isfinite(pct):
        axes[0].annotate(
            rf"$\langle C\rangle_0 / \langle C\rangle_{{*}} = {pct:.4f}$"
            f"\n({summary['C_pct_lower_at_star']:+.2f}% at $\varepsilon^*$ vs 0)",
            xy=(0.5, 0.95), xycoords="axes fraction", ha="center", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    axes[1].bar(labels, kl_vals, color=["C0", "C3", "C5"], alpha=0.85)
    axes[1].set_ylabel(r"$KL_{\mathrm{eq}}$ [bits]")
    axes[1].set_title("Information at equilibrium")
    fig.suptitle(
        rf"$N={sc.N}$, $s_1={sc.s1:g}$, $s_2={sc.s2:g}$, $\varepsilon^*={case.eps_star:+.2e}$",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "C_compare_zero_vs_star.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_traj_panel(
    runs: list[Eps12Run],
    case: GalleryCase,
    out_dir: Path,
    *,
    quantity: str,
    color_by: str,
) -> None:
    import matplotlib as mpl

    sc = case.scenario
    N = sc.N
    runs_f = [r for r in runs if np.isfinite(r.kl_eq)]

    if color_by == "kl":
        colors, cbar_cmap, cbar_norm, cbar_ticks = _score_colors(runs_f)
        cbar_label = r"$KL_{\mathrm{eq}}$ [bits]"
        fname = f"{quantity}_vs_time_by_kl_eq.png"
        title_color = r"$KL_{\mathrm{eq}}$"
    else:
        runs_f = sorted(runs_f, key=lambda r: r.eps12)
        eps_vals = np.array([r.eps12 for r in runs_f])
        cmap = plt.get_cmap("coolwarm")
        cbar_norm = mpl.colors.Normalize(vmin=float(eps_vals.min()), vmax=float(eps_vals.max()))
        colors = [cmap(cbar_norm(r.eps12)) for r in runs_f]
        cbar_cmap, cbar_ticks = cmap, None
        cbar_label = r"$\varepsilon_{12}$"
        fname = f"{quantity}_vs_time_by_eps12.png"
        title_color = r"$\varepsilon_{12}$"

    fig, ax = plt.subplots(figsize=(9, 5))
    for run, color in zip(runs_f, colors):
        if quantity == "C":
            ax.plot(run.pergen_times, run.pergen_c_cost, "-", lw=1.3, color=color, alpha=0.92)
            ylab = r"$C(t)=\langle C(x)\rangle_\psi$"
        else:
            t_eta, eta = _eta_trajectory(run, N)
            ax.plot(t_eta, eta, "-", lw=1.2, color=color, alpha=0.92)
            ylab = r"$\eta(t)=\Delta D/(N\,C(t))$"

    ax.axvline(0, color="none")
    ax.set_xlabel("generation $t$")
    ax.set_ylabel(ylab)
    ax.set_title(
        rf"{ylab} colored by {title_color}  ($N={sc.N}$, $s_1={sc.s1:g}$, $s_2={sc.s2:g}$)",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    if color_by == "kl":
        _add_score_colorbar(fig, ax, cbar_cmap, cbar_norm, cbar_ticks)
    else:
        _add_eps12_colorbar(fig, ax, cbar_cmap, cbar_norm)
    fig.tight_layout()
    fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _summarize(rows: list[dict], runs: list[Eps12Run], case: GalleryCase) -> dict:
    def row_at(eps_target: float) -> dict:
        i = int(np.argmin([abs(r["eps12"] - eps_target) for r in rows]))
        return rows[i]

    rw0 = row_at(0.0)
    rw_star = row_at(case.eps_star)
    c0 = float(rw0["mean_c_cost"])
    c_star = float(rw_star["mean_c_cost"])
    ratio = c0 / c_star if c_star > 1e-300 else float("nan")
    pct = 100.0 * (c_star - c0) / c0 if c0 > 1e-300 else float("nan")

    c_cvs = [float(r["c_cv"]) for r in rows if np.isfinite(r["c_cv"])]
    kl = np.array([r["kl_eq_bits"] for r in rows])
    mc = np.array([r["mean_c_cost"] for r in rows])
    ok = np.isfinite(kl) & np.isfinite(mc)
    rho = float(np.corrcoef(kl[ok], mc[ok])[0, 1]) if ok.sum() >= 3 else float("nan")

    return {
        "label": case.label,
        "eps_star": case.eps_star,
        "eps_at_zero": float(rw0["eps12"]),
        "eps_at_star_grid": float(rw_star["eps12"]),
        "C_at_zero": c0,
        "C_at_eps_star": c_star,
        "C_at_nearest_star": c_star,
        "C_ratio_zero_over_star": ratio,
        "C_pct_lower_at_star": pct,
        "C_lower_at_star": c_star < c0,
        "KL_at_zero": float(rw0["kl_eq_bits"]),
        "KL_at_eps_star": float(rw_star["kl_eq_bits"]),
        "KL_at_nearest_star": float(rw_star["kl_eq_bits"]),
        "mean_c_cv_over_eps": float(np.mean(c_cvs)) if c_cvs else float("nan"),
        "spearman_kl_vs_mean_c": rho,
        "n_grid": len(rows),
    }


def _run_one(case: GalleryCase, out_root: Path, *, half: float, step: float,
             pergen_max_t: int, reuse: bool) -> dict:
    out_dir = out_root / case.label
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "fine_data.csv"
    traj = out_dir / "fine_trajectories.npz"

    if reuse and cache.exists() and traj.exists():
        print(f"  {case.label}  (cached)", flush=True)
        with cache.open() as f:
            for row in csv.DictReader(f):
                rec = {}
                for k, v in row.items():
                    if k == "scenario" or not v:
                        rec[k] = v
                    else:
                        rec[k] = float(v)
                rows.append(rec)
        d = np.load(traj)
        runs = []
        for i in range(len(d["eps12"])):
            runs.append(Eps12Run(
                eps12=float(d["eps12"][i]),
                kl_eq=float(d["kl_eq"][i]),
                f1_eq=float("nan"), f2_eq=float("nan"), f3_eq=float("nan"), wstar_eq=float("nan"),
                pergen_times=d["times"],
                pergen_kl_phi_bits=d["d_kl"][i],
                pergen_c_cost=d["c_cost"][i],
                pergen_f1=np.full_like(d["times"], np.nan),
                pergen_f2=np.full_like(d["times"], np.nan),
                pergen_f3=np.full_like(d["times"], np.nan),
                pergen_f4=np.full_like(d["times"], np.nan),
                pergen_mean_wstar=np.full_like(d["times"], np.nan),
                pergen_info_f2=np.full_like(d["times"], np.nan),
                pergen_info_f3=np.full_like(d["times"], np.nan),
            ))
    else:
        print(f"  {case.label}  eps*={case.eps_star:+.2e}  half={half:g} step={step:g}", flush=True)
        rows, runs = _run_fine(case, half=half, step=step, pergen_max_t=pergen_max_t)
        _write_csv(rows, cache)
        np.savez(
            traj,
            eps12=np.array([r.eps12 for r in runs]),
            kl_eq=np.array([r.kl_eq for r in runs]),
            times=runs[0].pergen_times,
            c_cost=np.stack([r.pergen_c_cost for r in runs]),
            d_kl=np.stack([r.pergen_kl_phi_bits for r in runs]),
        )

    summary = _summarize(rows, runs, case)
    (out_dir / "fine_summary.json").write_text(json.dumps(summary, indent=2))

    _plot_c_mean_vs_eps(rows, case, out_dir)
    _plot_c_compare(summary, case, out_dir)
    _plot_traj_panel(runs, case, out_dir, quantity="C", color_by="kl")
    _plot_traj_panel(runs, case, out_dir, quantity="C", color_by="eps")
    _plot_traj_panel(runs, case, out_dir, quantity="eta", color_by="kl")
    _plot_traj_panel(runs, case, out_dir, quantity="eta", color_by="eps")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=str, default=None, help="single case label")
    p.add_argument("--only-n", type=int, nargs="+", default=None)
    p.add_argument("--half", type=float, default=0.0015, help="half-width beyond [0, eps*]")
    p.add_argument("--step", type=float, default=2e-5, help="eps12 step (default 2e-5)")
    p.add_argument("--pergen-max-t", type=int, default=500)
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--max-cases", type=int, default=None)
    args = p.parse_args()

    only_n = tuple(args.only_n) if args.only_n else None
    cases = _discover_cases(only_n=only_n, case_label=args.case)
    if args.max_cases:
        cases = cases[: args.max_cases]
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"{len(cases)} fine-scan cases, step={args.step:g}, half={args.half:g}\n", flush=True)
    all_summ: list[dict] = []
    for ci, case in enumerate(cases, 1):
        print(f"[{ci}/{len(cases)}]", flush=True)
        summ = _run_one(case, OUT, half=args.half, step=args.step,
                        pergen_max_t=args.pergen_max_t, reuse=args.reuse)
        all_summ.append(summ)
        star_lower = summ["C_lower_at_star"]
        print(
            f"  C(0)={summ['C_at_zero']:.5f}  C(* )={summ['C_at_eps_star']:.5f}  "
            f"ratio={summ['C_ratio_zero_over_star']:.4f}  "
            f"lower_at_star={star_lower}  mean_cv={summ['mean_c_cv_over_eps']:.4f}",
            flush=True,
        )

    (OUT / "batch_summary.json").write_text(json.dumps(all_summ, indent=2))
    n_lower = sum(1 for s in all_summ if s["C_lower_at_star"])
    print(f"\nwrote {OUT}/batch_summary.json  ({n_lower}/{len(all_summ)} cases: C lower at eps*)")


if __name__ == "__main__":
    main()
