#!/usr/bin/env python3
r"""
KL-dip vs u=v when s1+s2 = 0  (prediction eps* = 0 for all such pairs).

Tests whether the fitted dip stays at zero or drifts with u / |s| when the
linear term cancels but per-locus selection remains.

  ( 5e-4, -5e-4)
  ( 1e-3, -1e-3)
  ( 2e-3, -2e-3)

Run:
  python3 eps12_dip_sum0.py
  python3 eps12_dip_sum0.py --replot
"""

from __future__ import annotations

import argparse

import numpy as np

from eps12_dip_vs_u_plot import (
    EPS_SCALE, N, R, OUT_DIR, fit_dip, predicted_eps,
)

PAIRS = [
    (5.0e-4, -5.0e-4, r"$|s|=5\times10^{-4}$"),
    (1.0e-3, -1.0e-3, r"$|s|=10^{-3}$"),
    (2.0e-3, -2.0e-3, r"$|s|=2\times10^{-3}$"),
]
COLORS = ["#5ec962", "#3b528b", "#e07b39"]
DATA_PATH = OUT_DIR / "data_u_sum0.npz"
PLOT_PATH = OUT_DIR / "dip_vs_u_sum0.png"


def compute_all(u_arr: np.ndarray) -> np.ndarray:
    dip = np.full((len(PAIRS), len(u_arr)), np.nan)
    if DATA_PATH.exists():
        d = np.load(DATA_PATH)
        if len(d["u"]) == len(u_arr) and np.allclose(d["u"], u_arr):
            dip = d["dip"].copy()

    for ui, u in enumerate(u_arr):
        for pi, (s1, s2, tag) in enumerate(PAIRS):
            if np.isfinite(dip[pi, ui]):
                continue
            print(f"  u=v={u:g}  {tag}  s1={s1:g} s2={s2:g}", flush=True)
            dip[pi, ui] = fit_dip(s1, s2, float(u))
            np.savez(DATA_PATH, u=u_arr, dip=dip,
                     pairs=np.array([(p[0], p[1]) for p in PAIRS]))
    return dip


def plot(u, dip):
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

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    for pi, (s1, s2, tag) in enumerate(PAIRS):
        ax.plot(u, dip[pi] / EPS_SCALE, "o-", color=COLORS[pi], lw=1.8, ms=5,
                label=rf"$(s_1,s_2)=({s1:g},{s2:g})$  {tag}")

    ax.axhline(0, color="0.25", ls="--", lw=1.2,
               label=r"pred $\varepsilon_{12}^*=0$")
    ax.set_xscale("log")
    ax.set_xlabel(r"mutation $u=v$")
    ax.set_ylabel(r"fitted dip $\varepsilon_{12}^*\;(\times 10^{-3})$")
    ax.set_title(rf"$s_1+s_2=0$: dip vs mutation ($N$={N}, $r$={R:g})")
    ax.legend(fontsize=8.5, loc="best")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {PLOT_PATH}")

    print("\n  fitted dip (x1e-3) at each u; pred = 0")
    hdr = "  u".ljust(10) + "".join(f"  {tag:>12}" for _, _, tag in PAIRS)
    print(hdr)
    for ui, uval in enumerate(u):
        row = f"  {uval:g}".ljust(10)
        for pi in range(len(PAIRS)):
            row += f"  {dip[pi, ui]/EPS_SCALE:+.4f}".rjust(14)
        print(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replot", action="store_true")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    u_ref = np.load(OUT_DIR / "data_u_merged.npz")["u"]

    if args.replot and DATA_PATH.exists():
        d = np.load(DATA_PATH)
        plot(d["u"], d["dip"])
        return

    dip = compute_all(u_ref)
    plot(u_ref, dip)


if __name__ == "__main__":
    main()
