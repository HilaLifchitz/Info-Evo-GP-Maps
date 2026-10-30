#!/usr/bin/env python3
"""Plot KL-dip vs u=v from saved data_u.npz (+ optional incremental extension)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "plots" / "eps12_dip_robustness"
EPS_SCALE = 1.0e-3

PAIRS = [(1.0e-3, -5.0e-4), (-5.0e-4, -5.0e-4)]
PRED = [-2.0 / 3.0 * (s1 + s2) for s1, s2 in PAIRS]

# extra u to compute if missing (skip u=1.0 — period-2 chain, no stationary)
U_EXTRA = [1e-5, 3e-2, 1e-1, 3e-1]
N, R = 30, 1e-3
HALF_WIDTH, N_GRID, N_LOCAL = 2e-3, 17, 7


def predicted_eps(s1, s2):
    return -2.0 / 3.0 * (s1 + s2)


def local_vertex(eps, kl):
    ok = np.isfinite(kl)
    eps, kl = eps[ok], kl[ok]
    j = int(np.argmin(kl))
    lo = max(0, j - N_LOCAL // 2)
    hi = min(len(eps), lo + N_LOCAL)
    lo = max(0, hi - N_LOCAL)
    a, b, c = np.polyfit(eps[lo:hi], kl[lo:hi], 2)
    return (-b / (2.0 * a)) if a != 0 else np.nan


def fit_dip(s1, s2, u):
    from eps12_equilibrium_kl_sweep import run_one_N
    eps_pred = predicted_eps(s1, s2)
    eps_grid = eps_pred + np.linspace(-HALF_WIDTH, HALF_WIDTH, N_GRID)
    uv = np.array([u, u])
    res = run_one_N(N=N, eps_grid=eps_grid, r=R, u=uv, v=uv, s1=s1, s2=s2,
                    n_jobs=-1, neutral_cache_dir="cache", verbose=False)
    res.sort(key=lambda x: x.eps12)
    eps = np.array([x.eps12 for x in res])
    kl = np.array([x.kl_bits for x in res])
    return local_vertex(eps, kl)


def load_saved():
    merged = OUT_DIR / "data_u_merged.npz"
    src = merged if merged.exists() else OUT_DIR / "data_u.npz"
    d = np.load(src)
    return list(d["u"]), d["dip"].copy()


def extend(u_list, dip, extra=None):
    from eps12_equilibrium_kl_sweep import run_one_N  # noqa: F401
    for u in (extra if extra is not None else U_EXTRA):
        if u in u_list:
            continue
        print(f"computing u=v={u:g} ...", flush=True)
        row = []
        for s1, s2 in PAIRS:
            row.append(fit_dip(s1, s2, u))
        u_list.append(u)
        dip = np.column_stack([dip, row])
        np.savez(OUT_DIR / "data_u_merged.npz", pairs=np.array(PAIRS),
                 u=np.array(u_list), dip=dip)
    order = np.argsort(u_list)
    return np.array(u_list)[order], dip[:, order]


def plot(u, dip, out_path: Path):
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
    colors = ["#3b528b", "#b5367a"]
    fig, ax = plt.subplots(figsize=(9, 5.6))
    for pi, (s1, s2) in enumerate(PAIRS):
        pred = PRED[pi] / EPS_SCALE
        ax.plot(u, dip[pi] / EPS_SCALE, "o-", color=colors[pi], lw=1.8,
                label=rf"$s_1$={s1:g}, $s_2$={s2:g} (fit)")
        ax.axhline(pred, color=colors[pi], ls="--", lw=1.2,
                   label=rf"pred $-\frac{{2}}{{3}}(s_1+s_2)$={pred:+.2f}")
    ax.set_xscale("log")
    ax.set_xlabel(r"mutation $u=v$")
    ax.set_ylabel(r"fitted dip $\varepsilon_{12}^*\;(\times 10^{-3})$")
    ax.set_title(rf"KL-dip location vs mutation ($N$={N}, $r$={R:g})")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extend", action="store_true",
                   help="compute any missing u in U_EXTRA and merge")
    p.add_argument("--add-u", type=float, nargs="*", default=None,
                   help="compute specific u=v values and merge (e.g. --add-u 1e-5)")
    p.add_argument("--replot-only", action="store_true",
                   help="plot from data_u_merged.npz or data_u.npz only")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    merged = OUT_DIR / "data_u_merged.npz"
    if args.replot_only:
        src = merged if merged.exists() else OUT_DIR / "data_u.npz"
        d = np.load(src)
        plot(d["u"], d["dip"], OUT_DIR / "dip_vs_u_extended.png")
        return

    u, dip = load_saved()
    extra = args.add_u if args.add_u is not None else (U_EXTRA if args.extend else None)
    if extra is not None:
        u, dip = extend(list(u), dip, extra=extra)
        np.savez(merged, pairs=np.array(PAIRS), u=u, dip=dip)
    plot(u, dip, OUT_DIR / "dip_vs_u_extended.png")


if __name__ == "__main__":
    main()
