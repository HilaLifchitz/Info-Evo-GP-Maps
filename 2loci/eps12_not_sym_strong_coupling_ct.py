#!/usr/bin/env python3
r"""
Strong coupled mutation: verify Cov/Var ε* breakdown + C(t) sym vs coupled.

Outputs under ``plots/eps12_not_sym_s1p1_s2m2/strong_coupling_ct/``.

  cd 2loci
  python eps12_not_sym_strong_coupling_ct.py --ux 0.5
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
from eps12_delta_d_sweep import Eps12Run, run_one_eps12, _make_cfg  # noqa: E402
from eps12_not_sym_investigation import _fit_kl_min, _fmt_eps, _predict_eps_star  # noqa: E402
from simplex import compute_phi, mutation_matrix_asymmetric, params_from_config as params_from_config_sym  # noqa: E402
from simplex_not_sym import build_mutation_matrix, params_from_config as params_from_config_not_sym  # noqa: E402


def _base_cfg(*, N, r, u, v, s1, s2, eps12, ux, vx) -> dict:
    from simplex import build_sigma

    return {
        "_config_dir": str(SCRIPT_DIR),
        "N": N, "r": float(r), "u": u, "v": v, "ux": ux, "vx": vx,
        "sigma": build_sigma(s1, s2, eps12).tolist(),
        "times": [0], "n_jobs": -1, "neutral_cache_dir": "cache",
        "neutral": False, "top_k": 3, "plot_dynamics": False,
        "compute_f2": False, "ld_top_k_times": "first_last", "debug_kl": False,
    }


def _anchor_grid(base: np.ndarray, anchors: list[float]) -> np.ndarray:
    parts = [base] + [[float(a)] for a in anchors if np.isfinite(a)]
    return np.sort(np.unique(np.round(np.concatenate(parts), 12)))


def _default_base_grid(eps_star_sym: float, eps_star_cpl: float) -> np.ndarray:
    """Span wide enough to show 0 and both predicted ε* clearly."""
    anchors = [0.0]
    for e in (eps_star_sym, eps_star_cpl):
        if np.isfinite(e):
            anchors.append(float(e))
    span = max(0.012, 1.8 * max(abs(a) for a in anchors))
    step = max(span / 12.0, 0.001)
    base = np.arange(-span, span + 0.5 * step, step)
    return np.clip(base, -span, span)


def _run_eps(
    eps12: float,
    *,
    M_mut: np.ndarray,
    phi: np.ndarray,
    fracs: np.ndarray,
    N: int,
    r: float,
    s1: float,
    s2: float,
    pergen_max_t: int,
) -> Eps12Run:
    q_cond = cond_alpha_reference_pmf(fracs, M_mut, N)
    return run_one_eps12(
        float(eps12), N=N, r=r, u=0.001, v=0.001, s1=s1, s2=s2,
        phi=phi, fracs=fracs, M_mut=M_mut, q_cond=q_cond,
        pergen_max_t=pergen_max_t, verbose=False,
    )


def _sweep(
    eps_grid: np.ndarray,
    *,
    M_mut: np.ndarray,
    phi: np.ndarray,
    fracs: np.ndarray,
    N: int,
    r: float,
    s1: float,
    s2: float,
    pergen_max_t: int,
) -> list[dict]:
    rows: list[dict] = []
    for eps in eps_grid:
        run = _run_eps(
            float(eps), M_mut=M_mut, phi=phi, fracs=fracs,
            N=N, r=r, s1=s1, s2=s2, pergen_max_t=pergen_max_t,
        )
        rows.append({"eps12": float(eps), "kl_eq_bits": float(run.kl_eq), "run": run})
    return rows


def _plot_kl_verification(
    rows_sym: list[dict],
    rows_cpl: list[dict],
    out_path: Path,
    *,
    N: int,
    s1: float,
    s2: float,
    r: float,
    ux: float,
    eps_star_sym: float,
    eps_star_cpl: float,
    eps_fit_sym: float,
    eps_fit_cpl: float,
    ct_eps: list[tuple[str, float]],
) -> None:
    """KL_eq curves with vlines; mark ε values used for C(t) panels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False, sharey=False)

    specs = (
        (axes[0], rows_sym, "Symmetric mutation", eps_star_sym, eps_fit_sym, False),
        (axes[1], rows_cpl, rf"Coupled ($u_x=v_x={ux:g}$)", eps_star_cpl, eps_fit_cpl, True),
    )
    ct_vals = {e for _, e in ct_eps}

    for ax, rows, title, pred, fit, show_sym_pred in specs:
        eps = np.array([r["eps12"] for r in rows])
        kl = np.array([r["kl_eq_bits"] for r in rows])
        ax.plot(eps, kl, "o-", lw=1.8, ms=5, color="C0", zorder=3)

        # highlight C(t) anchor points on the curve
        for lab, e in ct_eps:
            j = int(np.argmin(np.abs(eps - e)))
            ax.scatter([eps[j]], [kl[j]], s=80, c="C1", zorder=5, edgecolors="k", linewidths=0.5)

        ax.axvline(0.0, color="0.35", ls=":", lw=1.0, zorder=0)
        ax.axvline(
            pred, color="C3", ls="--", lw=1.4, zorder=1,
            label=rf"pred $\varepsilon^*$ = {_fmt_eps(pred)}",
        )
        ax.axvline(
            fit, color="C2", ls="-.", lw=1.4, zorder=1,
            label=rf"fit $KL_{{\mathrm{{eq}}}}$ min = {_fmt_eps(fit)}",
        )
        if show_sym_pred:
            ax.axvline(
                eps_star_sym, color="C4", ls=":", lw=1.2, zorder=0,
                label=rf"sym pred (wrong $\varphi$) = {_fmt_eps(eps_star_sym)}",
            )

        xlo = min(float(eps.min()), pred, fit, 0.0, eps_star_sym if show_sym_pred else pred) - 0.15 * max(abs(pred), abs(fit), 0.005)
        xhi = max(float(eps.max()), pred, fit, 0.0) + 0.15 * max(abs(pred), abs(fit), 0.005)
        ax.set_xlim(xlo, xhi)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: _fmt_eps(float(x))))
        ax.set_xlabel(r"$\varepsilon_{12}$")
        ax.set_ylabel(r"$KL_{\mathrm{eq}}$ [bits]")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

        err = abs(pred - fit)
        ax.text(
            0.03, 0.97, rf"$|\mathrm{{pred}}-\mathrm{{fit}}|={err:.2e}$",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )

    # legend for C(t) markers
    axes[1].scatter([], [], s=80, c="C1", edgecolors="k", label="ε used in C(t) panels")
    axes[1].legend(fontsize=7, loc="lower right")

    fig.suptitle(
        rf"KL$_{{\mathrm{{eq}}}}$ dip vs Cov/Var $\varepsilon^*$  "
        rf"($N={N}$, $s_1={s1:g}$, $s_2={s2:g}$, $r={r:g}$)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_c_verification(
    cases: list[tuple[str, float, Eps12Run, Eps12Run]],
    out_path: Path,
    *,
    ux: float,
    N: int,
    s1: float,
    s2: float,
    t_max_plot: int,
) -> None:
    """One row per reference ε; sym vs coupled C(t) with explicit role label."""
    n = len(cases)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.0 * n + 0.8), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (role, eps, run_sym, run_cpl) in zip(axes, cases):
        for run, lab, col, ls in (
            (run_sym, "symmetric", "C0", "-"),
            (run_cpl, "coupled", "C3", "--"),
        ):
            t = run.pergen_times
            m = t <= t_max_plot
            ax.plot(t[m], run.pergen_c_cost[m], ls=ls, lw=1.8, color=col, label=lab)
        ax.set_ylabel(r"$C(t)$")
        ax.set_title(
            rf"{role}  —  $\varepsilon_{{12}}={_fmt_eps(eps)}$  "
            rf"($KL_{{\mathrm{{eq}}}}$ sym={run_sym.kl_eq:.4g}, cpl={run_cpl.kl_eq:.4g} bits)",
            fontsize=9,
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("generation $t$")
    fig.suptitle(
        rf"$C(t)$ at reference $\varepsilon_{{12}}$ values ($u_x=v_x={ux:g}$, $N={N}$)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ux", type=float, default=0.5, help="coupled rate u_x=v_x")
    p.add_argument("--N", type=int, default=30)
    p.add_argument("--r", type=float, default=0.0)
    p.add_argument("--s1", type=float, default=0.02)
    p.add_argument("--s2", type=float, default=-0.005)
    p.add_argument("--eps-min", type=float, default=None, help="override auto grid min")
    p.add_argument("--eps-max", type=float, default=None, help="override auto grid max")
    p.add_argument("--eps-step", type=float, default=None, help="override auto grid step")
    p.add_argument("--pergen-max-t", type=int, default=200)
    p.add_argument("--t-plot-max", type=int, default=150)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR / "plots" / "eps12_not_sym_s1p1_s2m2" / "strong_coupling_ct" / "s1p02_s2m005",
    )
    args = p.parse_args()

    N, r, s1, s2 = int(args.N), float(args.r), float(args.s1), float(args.s2)
    u = v = [0.001, 0.001]
    ux = vx = float(args.ux)
    pergen_max_t = int(args.pergen_max_t)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    M_sym = mutation_matrix_asymmetric(np.array(u), np.array(v))
    phi_sym, _, fracs_sym, _, _ = compute_phi(
        params_from_config_sym(_make_cfg(N=N, r=r, u=0.001, v=0.001, s1=s1, s2=s2, eps12=0.0), SCRIPT_DIR),
        M_sym,
    )
    eps_star_sym = _predict_eps_star(phi_sym, fracs_sym, s1, s2)

    cfg0 = _base_cfg(N=N, r=r, u=u, v=v, s1=s1, s2=s2, eps12=0.0, ux=[ux, ux], vx=[vx, vx])
    M_cpl = build_mutation_matrix(np.array(u), np.array(v), np.array([ux, ux]), np.array([vx, vx]))
    phi_cpl, _, fracs_cpl, _, _ = compute_phi(params_from_config_not_sym(cfg0, SCRIPT_DIR), M_cpl)
    eps_star_cpl = _predict_eps_star(phi_cpl, fracs_cpl, s1, s2)

    if args.eps_min is not None and args.eps_max is not None and args.eps_step is not None:
        base_grid = np.arange(float(args.eps_min), float(args.eps_max) + 0.5 * float(args.eps_step), float(args.eps_step))
    else:
        base_grid = _default_base_grid(eps_star_sym, eps_star_cpl)

    # Grid must include both predicted ε* so KL curves cover them
    eps_grid_sym = _anchor_grid(base_grid, [0.0, eps_star_sym])
    eps_grid_cpl = _anchor_grid(base_grid, [0.0, eps_star_sym, eps_star_cpl])

    print(f"ux=vx={ux:g}", flush=True)
    print(f"  sym pred ε*     = {eps_star_sym:+.6g}", flush=True)
    print(f"  coupled pred ε* = {eps_star_cpl:+.6g}", flush=True)

    print("KL sweep (sym)...", flush=True)
    rows_sym = _sweep(
        eps_grid_sym, M_mut=M_sym, phi=phi_sym, fracs=fracs_sym,
        N=N, r=r, s1=s1, s2=s2, pergen_max_t=pergen_max_t,
    )
    eps_fit_sym = _fit_kl_min(rows_sym)

    print("KL sweep (coupled)...", flush=True)
    rows_cpl = _sweep(
        eps_grid_cpl, M_mut=M_cpl, phi=phi_cpl, fracs=fracs_cpl,
        N=N, r=r, s1=s1, s2=s2, pergen_max_t=pergen_max_t,
    )
    eps_fit_cpl = _fit_kl_min(rows_cpl)

    print(f"  fit KL min sym={eps_fit_sym:+.6g}  coupled={eps_fit_cpl:+.6g}", flush=True)
    print(f"  |pred-fit| sym={abs(eps_star_sym-eps_fit_sym):.2e}  "
          f"coupled={abs(eps_star_cpl-eps_fit_cpl):.2e}", flush=True)

    # C(t) at clearly named reference ε values (always include pred ε* for each model)
    ct_specs: list[tuple[str, float]] = [
        (r"$\varepsilon_{12}=0$", 0.0),
        (rf"sym pred $\varepsilon^*$", float(eps_star_sym)),
        (rf"coupled pred $\varepsilon^*$", float(eps_star_cpl)),
        (rf"sym fit $KL_{{\mathrm{{eq}}}}$ min", float(eps_fit_sym)),
        (rf"coupled fit $KL_{{\mathrm{{eq}}}}$ min", float(eps_fit_cpl)),
    ]
    # dedupe by ε value but keep first label
    seen: set[float] = set()
    ct_unique: list[tuple[str, float]] = []
    for lab, e in ct_specs:
        key = round(e, 12)
        if key in seen:
            continue
        seen.add(key)
        ct_unique.append((lab, e))

    print(f"C(t) panels at: {[(lab, _fmt_eps(e)) for lab, e in ct_unique]}", flush=True)

    ct_cases: list[tuple[str, float, Eps12Run, Eps12Run]] = []
    for lab, eps in ct_unique:
        rs = _run_eps(eps, M_mut=M_sym, phi=phi_sym, fracs=fracs_sym, N=N, r=r, s1=s1, s2=s2, pergen_max_t=pergen_max_t)
        rc = _run_eps(eps, M_mut=M_cpl, phi=phi_cpl, fracs=fracs_cpl, N=N, r=r, s1=s1, s2=s2, pergen_max_t=pergen_max_t)
        ct_cases.append((lab, eps, rs, rc))

    _plot_kl_verification(
        rows_sym, rows_cpl, out_dir / "kl_eq_eps_star_breakdown.png",
        N=N, s1=s1, s2=s2, r=r, ux=ux,
        eps_star_sym=eps_star_sym, eps_star_cpl=eps_star_cpl,
        eps_fit_sym=eps_fit_sym, eps_fit_cpl=eps_fit_cpl,
        ct_eps=ct_unique,
    )
    _plot_c_verification(
        ct_cases, out_dir / "C_vs_time_sym_vs_coupled_by_eps.png",
        ux=ux, N=N, s1=s1, s2=s2, t_max_plot=int(args.t_plot_max),
    )

    summary = {
        "N": N, "r": r, "s1": s1, "s2": s2, "u": u, "v": v, "ux": ux,
        "eps_star_sym": eps_star_sym,
        "eps_star_coupled": eps_star_cpl,
        "eps_fit_kl_sym": eps_fit_sym,
        "eps_fit_kl_coupled": eps_fit_cpl,
        "abs_pred_minus_fit_sym": abs(eps_star_sym - eps_fit_sym),
        "abs_pred_minus_fit_coupled": abs(eps_star_cpl - eps_fit_cpl),
        "ct_panels": [{"label": lab, "eps12": e} for lab, e in ct_unique],
        "eps_grid_sym": eps_grid_sym.tolist(),
        "eps_grid_coupled": eps_grid_cpl.tolist(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
