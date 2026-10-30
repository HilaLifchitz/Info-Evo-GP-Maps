#!/usr/bin/env python3
r"""
ROBUSTNESS of the KL-dip prediction  eps12* = -2/3 (s1 + s2)  (symmetric u=v),
as we vary recombination r and the (symmetric) mutation rate u=v, at N=30.

For a couple of fixed (s1, s2) pairs we, for each r (resp. each u=v), sweep eps12
on a fine grid centred on the predicted dip, fit a local parabola to
KL(psi*||phi*), read the empirical vertex, and check it stays at -2/3 (s1+s2).

Outputs:
  dip_vs_r.png    fitted dip vs recombination r (u=v fixed)
  dip_vs_u.png    fitted dip vs mutation u=v   (r fixed)
  data_r.npz, data_u.npz, robustness.log

Run:
  python3 eps12_dip_robustness.py --out plots/eps12_dip_robustness
  python3 eps12_dip_robustness.py --out plots/eps12_dip_robustness --replot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eps12_equilibrium_kl_sweep import run_one_N

SCRIPT_DIR = Path(__file__).resolve().parent

N = 30
HALF_WIDTH = 1.0e-3
N_GRID = 11
N_LOCAL = 7

# fixed values when not being swept
R_FIXED = 1.0e-3
U_FIXED = 1.0e-3

# (s1, s2) pairs: one with s1+s2>0, one with s1+s2<0
PAIRS = [(1.0e-3, -5.0e-4), (-5.0e-4, -5.0e-4)]

R_VALUES = [0.0, 5.0e-3, 2.0e-2, 1.0e-1, 5.0e-1]
U_VALUES = [1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2]

EPS_SCALE = 1.0e-3
KL_LABEL = r"$D_{\mathrm{KL}}(\psi^*\,\|\,\varphi^*)$ [bits]"


def predicted_eps(s1, s2):
    return -2.0 / 3.0 * (s1 + s2)


def local_vertex(eps, kl):
    j = int(np.argmin(kl))
    lo = max(0, j - N_LOCAL // 2)
    hi = min(len(eps), lo + N_LOCAL)
    lo = max(0, hi - N_LOCAL)
    a, b, c = np.polyfit(eps[lo:hi], kl[lo:hi], 2)
    return (-b / (2.0 * a)) if a != 0 else np.nan


def _fit_dip(s1, s2, r, u):
    eps_pred = predicted_eps(s1, s2)
    eps_grid = eps_pred + np.linspace(-HALF_WIDTH, HALF_WIDTH, N_GRID)
    uv = np.array([u, u])
    res = run_one_N(N=N, eps_grid=eps_grid, r=r, u=uv, v=uv, s1=s1, s2=s2,
                    n_jobs=-1, neutral_cache_dir="cache", verbose=False)
    res.sort(key=lambda x: x.eps12)
    eps = np.array([x.eps12 for x in res])
    kl = np.array([x.kl_bits for x in res])
    ok = np.isfinite(kl)
    return local_vertex(eps[ok], kl[ok])


def _compute(out_dir: Path):
    # r-sweep
    dip_r = np.full((len(PAIRS), len(R_VALUES)), np.nan)
    for pi, (s1, s2) in enumerate(PAIRS):
        for ri, r in enumerate(R_VALUES):
            print(f"[r-sweep] pair=({s1:g},{s2:g}) r={r:g}", flush=True)
            dip_r[pi, ri] = _fit_dip(s1, s2, r, U_FIXED)
    # u-sweep
    dip_u = np.full((len(PAIRS), len(U_VALUES)), np.nan)
    for pi, (s1, s2) in enumerate(PAIRS):
        for ui, u in enumerate(U_VALUES):
            print(f"[u-sweep] pair=({s1:g},{s2:g}) u=v={u:g}", flush=True)
            dip_u[pi, ui] = _fit_dip(s1, s2, R_FIXED, u)
    np.savez(out_dir / "data_r.npz", pairs=np.array(PAIRS), r=np.array(R_VALUES), dip=dip_r)
    np.savez(out_dir / "data_u.npz", pairs=np.array(PAIRS), u=np.array(U_VALUES), dip=dip_u)
    return dip_r, dip_u


def _load(out_dir: Path):
    dr = np.load(out_dir / "data_r.npz")
    du = np.load(out_dir / "data_u.npz")
    return dr["dip"], du["dip"]


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

    dip_r, dip_u = _load(out_dir) if replot else _compute(out_dir)

    colors = ["#3b528b", "#b5367a"]
    log = [f"N={N}, fixed r={R_FIXED:g} (u-sweep), fixed u=v={U_FIXED:g} (r-sweep)",
           "prediction: eps12* = -2/3 (s1+s2)", ""]

    # ---- dip vs r ----
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for pi, (s1, s2) in enumerate(PAIRS):
        pred = predicted_eps(s1, s2) / EPS_SCALE
        ax.plot(R_VALUES, dip_r[pi] / EPS_SCALE, "o-", color=colors[pi], lw=1.8,
                label=rf"$s_1$={s1:g}, $s_2$={s2:g} (fit)")
        ax.axhline(pred, color=colors[pi], ls="--", lw=1.2,
                   label=rf"pred $-\frac{{2}}{{3}}(s_1+s_2)$={pred:+.2f}")
        log.append(f"[r] s1={s1:g} s2={s2:g} pred={pred*EPS_SCALE:+.5g}: " +
                   "  ".join(f"r={r:g}->{d:+.4g}" for r, d in zip(R_VALUES, dip_r[pi])))
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel(r"recombination $r$")
    ax.set_ylabel(r"fitted dip $\varepsilon_{12}^*\;(\times 10^{-3})$")
    ax.set_title(rf"KL-dip location vs recombination ($N$={N}, $u=v$={U_FIXED:g})")
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    fig.tight_layout(); p = out_dir / "dip_vs_r.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {p}")

    # ---- dip vs u ----
    fig2, ax2 = plt.subplots(figsize=(8, 5.2))
    for pi, (s1, s2) in enumerate(PAIRS):
        pred = predicted_eps(s1, s2) / EPS_SCALE
        ax2.plot(U_VALUES, dip_u[pi] / EPS_SCALE, "o-", color=colors[pi], lw=1.8,
                 label=rf"$s_1$={s1:g}, $s_2$={s2:g} (fit)")
        ax2.axhline(pred, color=colors[pi], ls="--", lw=1.2,
                    label=rf"pred $-\frac{{2}}{{3}}(s_1+s_2)$={pred:+.2f}")
        log.append(f"[u] s1={s1:g} s2={s2:g} pred={pred*EPS_SCALE:+.5g}: " +
                   "  ".join(f"u={u:g}->{d:+.4g}" for u, d in zip(U_VALUES, dip_u[pi])))
    ax2.set_xscale("log")
    ax2.set_xlabel(r"mutation $u=v$")
    ax2.set_ylabel(r"fitted dip $\varepsilon_{12}^*\;(\times 10^{-3})$")
    ax2.set_title(rf"KL-dip location vs mutation ($N$={N}, $r$={R_FIXED:g})")
    ax2.legend(fontsize=8, ncol=2); ax2.grid(True, alpha=0.3)
    fig2.tight_layout(); p2 = out_dir / "dip_vs_u.png"
    fig2.savefig(p2, dpi=170, bbox_inches="tight"); plt.close(fig2)
    print(f"Wrote {p2}")

    (out_dir / "robustness.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'robustness.log'}")
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
