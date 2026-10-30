#!/usr/bin/env python3
r"""
Four (s1,s2) pairs demonstrating sign/cancellation in the KL-dip vs u=v plot:

  1. mixed  (+,-)  s1+s2 > 0   (1e-3, -5e-4)     pred negative
  2. mixed  (-,+)  s1+s2 < 0   (-1e-3, +5e-4)    pred positive  [flip of 1]
  3. same   (-,-)  s1+s2 < 0   (-5e-4, -5e-4)   pred positive
  4. same   (+,+)  s1+s2 > 0   (+5e-4, +5e-4)   pred negative  [flip of 3]

Reuses u grid from data_u_merged.npz. Computes missing pairs incrementally.

Run:
  python3 eps12_dip_sign_pairs.py
  python3 eps12_dip_sign_pairs.py --replot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eps12_dip_vs_u_plot import (
    EPS_SCALE, N, R, HALF_WIDTH, N_GRID, N_LOCAL, OUT_DIR, fit_dip, predicted_eps,
)

PAIRS = [
    (1.0e-3,  -5.0e-4, "mixed (+,-)"),
    (-1.0e-3,  5.0e-4, "mixed (-,+)"),
    (-5.0e-4, -5.0e-4, "same (−,−)"),
    (5.0e-4,   5.0e-4, "same (+,+)"),
]
COLORS = ["#3b528b", "#21908d", "#b5367a", "#e07b39"]
DATA_PATH = OUT_DIR / "data_u_sign4.npz"


def _load_or_init(u_ref: np.ndarray):
    if DATA_PATH.exists():
        d = np.load(DATA_PATH)
        return d["u"], d["dip"], d["pairs"]
    dip = np.full((len(PAIRS), len(u_ref)), np.nan)
    # seed pairs 0,2 from earlier 2-pair sweep if available
    old = OUT_DIR / "data_u_merged.npz"
    if old.exists():
        d = np.load(old)
        if len(d["u"]) == len(u_ref) and np.allclose(d["u"], u_ref):
            dip[0] = d["dip"][0]   # mixed (+,-)
            dip[2] = d["dip"][1]   # same (−,−)
            print("Seeded pairs 1 & 3 from data_u_merged.npz", flush=True)
    return u_ref.copy(), dip, np.array([(p[0], p[1]) for p in PAIRS])


def compute_missing(u_arr, dip):
    for ui, u in enumerate(u_arr):
        for pi, (s1, s2, tag) in enumerate(PAIRS):
            if np.isfinite(dip[pi, ui]):
                continue
            print(f"  u=v={u:g}  pair {pi+1}/4 {tag}  s1={s1:g} s2={s2:g}", flush=True)
            dip[pi, ui] = fit_dip(s1, s2, float(u))
            np.savez(DATA_PATH, u=u_arr, dip=dip,
                     pairs=np.array([(p[0], p[1]) for p in PAIRS]))
    return dip


def plot(u, dip, out: Path):
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

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for pi, (s1, s2, tag) in enumerate(PAIRS):
        pred = predicted_eps(s1, s2) / EPS_SCALE
        ax.plot(u, dip[pi] / EPS_SCALE, "o-", color=COLORS[pi], lw=1.8, ms=5,
                label=rf"{tag}: $s_1$={s1:g}, $s_2$={s2:g}")
        ax.axhline(pred, color=COLORS[pi], ls="--", lw=1.1, alpha=0.85,
                   label=rf"pred $={pred:+.2f}$")

    ax.set_xscale("log")
    ax.set_xlabel(r"mutation $u=v$")
    ax.set_ylabel(r"fitted dip $\varepsilon_{12}^*\;(\times 10^{-3})$")
    ax.set_title(rf"KL-dip vs mutation: sign/cancellation test ($N$={N}, $r$={R:g})")
    ax.legend(fontsize=7.5, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    ax.axhline(0, color="0.3", lw=0.8)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    # summary table: c_eff vs 2/3 at a few u
    print("\n  c_eff = -eps_fit/(s1+s2)  (2/3 = 0.6667)")
    for ui, u in enumerate(u):
        print(f"  u={u:g}:")
        for pi, (s1, s2, tag) in enumerate(PAIRS):
            ssum = s1 + s2
            c = -dip[pi, ui] / ssum if ssum != 0 else np.nan
            print(f"    {tag:14s}  c_eff={c:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replot", action="store_true")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    u_ref = np.load(OUT_DIR / "data_u_merged.npz")["u"]
    if args.replot and DATA_PATH.exists():
        d = np.load(DATA_PATH)
        plot(d["u"], d["dip"], OUT_DIR / "dip_vs_u_sign4.png")
        return

    u_arr, dip, _ = _load_or_init(u_ref)
    dip = compute_missing(u_arr, dip)
    plot(u_arr, dip, OUT_DIR / "dip_vs_u_sign4.png")


if __name__ == "__main__":
    main()
