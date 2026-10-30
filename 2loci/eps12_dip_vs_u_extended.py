#!/usr/bin/env python3
r"""
Extended KL-dip-vs-mutation experiment: how does the dip location track the
prediction eps12* = -2/3 (s1+s2) as u=v grows all the way to 10^0 ?

N=30, r fixed. For two (s1,s2) pairs and each u=v in a log grid up to 1.0, sweep
eps12 on a (wide) grid, fit a local parabola to KL(psi*||phi*), record the vertex.

Run:
  python3 eps12_dip_vs_u_extended.py --out plots/eps12_dip_robustness
  python3 eps12_dip_vs_u_extended.py --out plots/eps12_dip_robustness --replot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eps12_equilibrium_kl_sweep import run_one_N

SCRIPT_DIR = Path(__file__).resolve().parent

N = 30
R_FIXED = 1.0e-3
HALF_WIDTH = 2.0e-3        # wider than before: dip drifts at large u
N_GRID = 17
N_LOCAL = 7

PAIRS = [(1.0e-3, -5.0e-4), (-5.0e-4, -5.0e-4)]
U_VALUES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1e0]

EPS_SCALE = 1.0e-3


def predicted_eps(s1, s2):
    return -2.0 / 3.0 * (s1 + s2)


def local_vertex(eps, kl):
    ok = np.isfinite(kl)
    eps, kl = eps[ok], kl[ok]
    if len(eps) < 5:
        return np.nan
    j = int(np.argmin(kl))
    lo = max(0, j - N_LOCAL // 2)
    hi = min(len(eps), lo + N_LOCAL)
    lo = max(0, hi - N_LOCAL)
    if hi - lo < 3:
        return np.nan
    a, b, c = np.polyfit(eps[lo:hi], kl[lo:hi], 2)
    return (-b / (2.0 * a)) if a != 0 else np.nan


def _fit_dip(s1, s2, u):
    eps_pred = predicted_eps(s1, s2)
    eps_grid = eps_pred + np.linspace(-HALF_WIDTH, HALF_WIDTH, N_GRID)
    uv = np.array([u, u])
    res = run_one_N(N=N, eps_grid=eps_grid, r=R_FIXED, u=uv, v=uv, s1=s1, s2=s2,
                    n_jobs=-1, neutral_cache_dir="cache", verbose=False)
    res.sort(key=lambda x: x.eps12)
    eps = np.array([x.eps12 for x in res])
    kl = np.array([x.kl_bits for x in res])
    return local_vertex(eps, kl)


def _compute(out_dir: Path):
    dip = np.full((len(PAIRS), len(U_VALUES)), np.nan)
    for pi, (s1, s2) in enumerate(PAIRS):
        for ui, u in enumerate(U_VALUES):
            print(f"[u-ext] pair=({s1:g},{s2:g}) u=v={u:g}", flush=True)
            dip[pi, ui] = _fit_dip(s1, s2, u)
    np.savez(out_dir / "data_u_ext.npz", pairs=np.array(PAIRS),
             u=np.array(U_VALUES), dip=dip)
    return dip


def _load(out_dir: Path):
    d = np.load(out_dir / "data_u_ext.npz")
    return d["dip"]


def run(out_dir: Path, replot: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    except ImportError:
        pass
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "figure.facecolor": "white"})
    out_dir.mkdir(parents=True, exist_ok=True)

    dip = _load(out_dir) if replot else _compute(out_dir)
    colors = ["#3b528b", "#b5367a"]
    log = [f"N={N}, r={R_FIXED:g}, eps window +/-{HALF_WIDTH:g}",
           "prediction: eps12* = -2/3 (s1+s2)", ""]

    fig, ax = plt.subplots(figsize=(9, 5.6))
    for pi, (s1, s2) in enumerate(PAIRS):
        pred = predicted_eps(s1, s2) / EPS_SCALE
        ax.plot(U_VALUES, dip[pi] / EPS_SCALE, "o-", color=colors[pi], lw=1.8,
                label=rf"$s_1$={s1:g}, $s_2$={s2:g} (fit)")
        ax.axhline(pred, color=colors[pi], ls="--", lw=1.2,
                   label=rf"pred $-\frac{{2}}{{3}}(s_1+s_2)$={pred:+.2f}")
        log.append(f"s1={s1:g} s2={s2:g} pred={pred*EPS_SCALE:+.5g}: " +
                   "  ".join(f"u={u:g}->{d:+.4g}" for u, d in zip(U_VALUES, dip[pi])))
    ax.set_xscale("log")
    ax.set_xlabel(r"mutation $u=v$")
    ax.set_ylabel(r"fitted dip $\varepsilon_{12}^*\;(\times 10^{-3})$")
    ax.set_title(rf"KL-dip location vs mutation, extended to $u=v=10^0$ "
                 rf"($N$={N}, $r$={R_FIXED:g})")
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    p = out_dir / "dip_vs_u_extended.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {p}")

    (out_dir / "dip_vs_u_extended.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=SCRIPT_DIR / "plots" / "eps12_dip_robustness")
    p.add_argument("--replot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run(args.out.resolve(), replot=args.replot)


if __name__ == "__main__":
    main()
