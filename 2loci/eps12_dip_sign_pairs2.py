#!/usr/bin/env python3
r"""
Six-pair KL-dip vs u=v panel: 3 with s1+s2>0, 3 with s1+s2<0 (all distinct sums).

  (+)  ( 2e-3, -5e-4)  sum +1.5e-3
  (+)  ( 1.2e-3,  4e-4) sum +1.6e-3
  (+)  ( 1e-3,  -5e-4)  sum +5e-4   [from sign4 cache]
  (−)  (-4e-4, -1.2e-3) sum -1.6e-3
  (−)  (-5e-4, -5e-4)  sum -1e-3   [from sign4 cache]
  (−)  (-1e-3,  5e-4)  sum -5e-4   [from sign4 cache]
  (0)  ( 1e-3, -1e-3)  sum 0       [from sum0 cache]

Run:
  python3 eps12_dip_sign_pairs2.py
  python3 eps12_dip_sign_pairs2.py --replot
"""

from __future__ import annotations

import argparse

import numpy as np

from eps12_dip_vs_u_plot import (
    EPS_SCALE, N, R, OUT_DIR, fit_dip, predicted_eps,
)

PAIRS = [
    (2.0e-3,  -5.0e-4, "mixed (+,-)"),
    (1.2e-3,   4.0e-4, "same (+,+) unequal"),
    (1.0e-3,  -5.0e-4, "mixed (+,-) small"),
    (-4.0e-4, -1.2e-3, "same (−,−) unequal"),
    (-5.0e-4, -5.0e-4, "same (−,−) equal"),
    (-1.0e-3,  5.0e-4, "mixed (−,+)"),
    (1.0e-3,  -1.0e-3, r"$s_1+s_2=0$"),
]
COLORS = ["#3b528b", "#b5367a", "#5ec962", "#e07b39", "#9c4f96", "#21908d", "#1a1a1a"]
DATA_PATH = OUT_DIR / "data_u_sign6.npz"
PLOT_PATH = OUT_DIR / "dip_vs_u_sign4b.png"


def _seed(dip: np.ndarray, u_arr: np.ndarray) -> None:
    """Fill from earlier runs where (s1,s2) and u grid match."""
    seeds = []
    for path in (OUT_DIR / "data_u_sign4b.npz", OUT_DIR / "data_u_sign4.npz",
                 OUT_DIR / "data_u_merged.npz", OUT_DIR / "data_u_sum0.npz"):
        if not path.exists():
            continue
        d = np.load(path)
        if len(d["u"]) != len(u_arr) or not np.allclose(d["u"], u_arr):
            continue
        old_pairs = d["pairs"]
        for pi, (s1, s2, _) in enumerate(PAIRS):
            for oi, op in enumerate(old_pairs):
                if np.isclose(op[0], s1) and np.isclose(op[1], s2):
                    dip[pi] = d["dip"][oi]
                    break
    n = int(np.isfinite(dip).sum())
    if n:
        print(f"Seeded {n}/{dip.size} values from cache", flush=True)


def compute_all(u_arr: np.ndarray) -> np.ndarray:
    dip = np.full((len(PAIRS), len(u_arr)), np.nan)
    if DATA_PATH.exists():
        d = np.load(DATA_PATH)
        if (len(d["u"]) == len(u_arr) and np.allclose(d["u"], u_arr)
                and len(d["pairs"]) == len(PAIRS)):
            dip = d["dip"].copy()
    _seed(dip, u_arr)

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

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    for pi, (s1, s2, tag) in enumerate(PAIRS):
        pred = predicted_eps(s1, s2) / EPS_SCALE
        ssum = s1 + s2
        ax.plot(u, dip[pi] / EPS_SCALE, "o-", color=COLORS[pi], lw=1.8, ms=4,
                label=rf"{tag}: $s_1+s_2$={ssum:.1e}")
        ax.axhline(pred, color=COLORS[pi], ls="--", lw=1.0, alpha=0.8,
                   label=rf"pred $={pred:+.2f}$")

    ax.set_xscale("log")
    ax.set_xlabel(r"mutation $u=v$")
    ax.set_ylabel(r"fitted dip $\varepsilon_{12}^*\;(\times 10^{-3})$")
    ax.set_title(rf"KL-dip vs mutation ($N$={N}, $r$={R:g})")
    ax.legend(fontsize=6.5, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    ax.axhline(0, color="0.3", lw=0.8)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {PLOT_PATH}")


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
