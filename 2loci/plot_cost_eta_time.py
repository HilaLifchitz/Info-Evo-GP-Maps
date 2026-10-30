#!/usr/bin/env python3
"""Plot C(t) and eta(t) for a single eps12 value (default r=0 sweep settings)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eps12_delta_d_sweep import run_one_eps12, analyze_run  # noqa: E402
from simplex import (  # noqa: E402
    compute_phi,
    mutation_matrix_asymmetric,
    params_from_config,
)
from eps12_delta_d_sweep import _make_cfg  # noqa: E402

OUT_DIR = SCRIPT_DIR / "plots" / "eps12_delta_d"


def plot_cost_eta_time(
    *,
    eps12: float,
    N: int,
    r: float,
    u: float,
    v: float,
    s1: float,
    s2: float,
    pergen_max_t: int,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M_mut = mutation_matrix_asymmetric(np.array([u, u]), np.array([v, v]))
    params0 = params_from_config(_make_cfg(N=N, r=r, u=u, v=v, s1=s1, s2=s2, eps12=0.0), SCRIPT_DIR)
    phi, _, fracs, _, _ = compute_phi(params0, M_mut)

    run = run_one_eps12(
        eps12, N=N, r=r, u=u, v=v, s1=s1, s2=s2,
        phi=phi, fracs=fracs, M_mut=M_mut,
        pergen_max_t=pergen_max_t, verbose=True,
    )
    m = analyze_run(run, N)

    t = run.pergen_times
    d = run.pergen_kl_phi_bits
    c = run.pergen_c_cost
    nc = N * c
    delta_d = np.diff(d)
    t_eta = t[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(nc[1:] > 1e-300, delta_d / nc[1:], np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    ax = axes[0]
    ax.plot(t, d, "C0", lw=1.8)
    ax.axhline(run.kl_eq, color="C0", ls=":", lw=1.2, alpha=0.7,
               label=rf"$\mathrm{{KL}}_{{\mathrm{{eq}}}}={run.kl_eq:.3f}$ bits")
    ax.set_ylabel(r"$D(t)=\mathrm{KL}(\psi\|\varphi)$ [bits]")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_title(
        rf"$\varepsilon_{{12}}={eps12:+.2f}$, $N={N}$, $u=v={u:g}$, $r={r:g}$, "
        rf"$s_1={s1:g}$, $s_2={s2:g}$",
        fontsize=10,
    )

    ax = axes[1]
    ax.plot(t, c, "C3", lw=1.8, label=r"$C(t)=\langle C(x)\rangle_\psi$")
    ax.plot(t, nc, "C4", ls="--", lw=1.5, alpha=0.85, label=rf"$N\,C(t)$")
    ax.set_ylabel("cost / bound")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(t_eta, eta, "C2", lw=1.8)
    ax.axhline(1.0, color="0.5", ls=":", lw=0.9, alpha=0.7, label=r"bound ($\eta=1$)")
    ax.scatter([m.t_peak_eta], [m.peak_eta], c="C2", s=60, zorder=5, edgecolors="k", linewidths=0.6)
    ax.annotate(
        rf"peak $\eta={m.peak_eta:.3f}$ at $t={int(m.t_peak_eta)}$",
        xy=(m.t_peak_eta, m.peak_eta),
        xytext=(m.t_peak_eta + 18, m.peak_eta - 0.08),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="0.35", lw=0.8),
    )
    ax.set_xlabel("generation $t$")
    ax.set_ylabel(r"$\eta(t)=\Delta D/(N\,C(t))$")
    ax.set_xlim(t[0], min(120, t[-1]))  # focus on early dynamics
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.text(
        0.99, 0.01,
        rf"$\bar{{\eta}}={m.path_efficiency:.4f}$,  $t_{{90}}={int(m.t_90)}$,  "
        rf"$D(T)={m.d_final:.3f}$ bits",
        ha="right", va="bottom", fontsize=8, color="0.35",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot C(t) and eta(t) for one eps12")
    ap.add_argument("--eps12", type=float, default=-0.2)
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--r", type=float, default=0.0)
    ap.add_argument("--u", type=float, default=0.01)
    ap.add_argument("--v", type=float, default=0.01)
    ap.add_argument("--s1", type=float, default=0.001)
    ap.add_argument("--s2", type=float, default=-0.001)
    ap.add_argument("--pergen-max-t", type=int, default=500)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or OUT_DIR / f"cost_eta_vs_time_eps12_{args.eps12:+.2f}.png".replace("+", "p").replace("-", "m")
    plot_cost_eta_time(
        eps12=args.eps12, N=args.N, r=args.r, u=args.u, v=args.v,
        s1=args.s1, s2=args.s2, pergen_max_t=args.pergen_max_t, out_path=out,
    )


if __name__ == "__main__":
    main()
