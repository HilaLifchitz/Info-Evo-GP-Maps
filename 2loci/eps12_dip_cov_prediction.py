#!/usr/bin/env python3
r"""
Sanity check: compare fitted KL-dips to the full weak-selection prediction

    eps12* = -Cov_phi(B, x11) / Var_phi(x11)

with B = s1 p1 + s2 p2  (additive background fitness at each lattice point),
x11 = haplotype-11 frequency, and all moments under neutral phi* at the same u,v.

Run:
  python3 eps12_dip_cov_prediction.py
  python3 eps12_dip_cov_prediction.py --replot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eps12_dip_vs_u_plot import EPS_SCALE, N, R, OUT_DIR
from neutral_cache import cache_path_from_params, load_neutral_phi
from simplex import allele_freqs_from_genotype_fracs, make_simplex_grid

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "cache"
DATA_PATH = OUT_DIR / "data_cov_check.npz"
SCATTER_PATH = OUT_DIR / "dip_cov_vs_fit.png"
PANEL_PATH = OUT_DIR / "dip_vs_u_cov_check.png"

# Pairs from sign6 panel + sum0 magnitudes (tag, s1, s2)
PAIRS = [
    ("mixed (+,-)",        2.0e-3,  -5.0e-4),
    ("same (+,+) unequal", 1.2e-3,   4.0e-4),
    ("mixed (+,-) small",  1.0e-3,  -5.0e-4),
    ("same (−,−) unequal", -4.0e-4, -1.2e-3),
    ("same (−,−) equal",   -5.0e-4, -5.0e-4),
    ("mixed (−,+)",       -1.0e-3,   5.0e-4),
    (r"$s_1+s_2=0$",       1.0e-3,  -1.0e-3),
]
COLORS = ["#3b528b", "#b5367a", "#5ec962", "#e07b39", "#9c4f96", "#21908d", "#1a1a1a"]


def cov_prediction(s1: float, s2: float, u: float, *, N_pop: int = N, r: float = R) -> float:
    """eps12* = -Cov_phi(B, x11) / Var_phi(x11) at neutral phi for u=v."""
    uv = np.array([u, u], dtype=float)
    phi = load_neutral_phi(
        cache_path_from_params(uv, uv, r, CACHE_DIR), N_pop, uv, uv, r,
    )
    if phi is None:
        raise FileNotFoundError(f"no neutral cache for u=v={u:g}, r={r:g}")

    fracs = make_simplex_grid(N_pop)
    p1, p2 = allele_freqs_from_genotype_fracs(fracs)
    x11 = fracs[:, 3]
    B = s1 * p1 + s2 * p2

    Eb = float(np.sum(phi * B))
    Ex = float(np.sum(phi * x11))
    cov = float(np.sum(phi * (B - Eb) * (x11 - Ex)))
    var = float(np.sum(phi * (x11 - Ex) ** 2))
    if var <= 0.0:
        return float("nan")
    return -cov / var


def _load_fit_dips(u_arr: np.ndarray) -> np.ndarray:
    """Assemble fitted dips for PAIRS from any cached u-sweep."""
    dip = np.full((len(PAIRS), len(u_arr)), np.nan)
    for path in (
        OUT_DIR / "data_u_sign6.npz",
        OUT_DIR / "data_u_sign4b.npz",
        OUT_DIR / "data_u_sign4.npz",
        OUT_DIR / "data_u_merged.npz",
        OUT_DIR / "data_u_sum0.npz",
    ):
        if not path.exists():
            continue
        d = np.load(path)
        if len(d["u"]) != len(u_arr) or not np.allclose(d["u"], u_arr):
            continue
        for pi, (_, s1, s2) in enumerate(PAIRS):
            if np.all(np.isfinite(dip[pi])):
                continue
            for oi, op in enumerate(d["pairs"]):
                if np.isclose(op[0], s1) and np.isclose(op[1], s2):
                    dip[pi] = d["dip"][oi]
                    break
    return dip


def compute_all(u_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.full((len(PAIRS), len(u_arr)), np.nan)
    if DATA_PATH.exists():
        d = np.load(DATA_PATH)
        if len(d["u"]) == len(u_arr) and np.allclose(d["u"], u_arr):
            pred = d["pred_cov"].copy()

    for ui, u in enumerate(u_arr):
        for pi, (_, s1, s2) in enumerate(PAIRS):
            if np.isfinite(pred[pi, ui]):
                continue
            print(f"  u=v={u:g}  s1={s1:g} s2={s2:g}", flush=True)
            pred[pi, ui] = cov_prediction(s1, s2, float(u))
            np.savez(DATA_PATH, u=u_arr, pred_cov=pred,
                     pairs=np.array([(p[1], p[2]) for p in PAIRS]))

    fit = _load_fit_dips(u_arr)
    return pred, fit


def plot(u_arr: np.ndarray, pred: np.ndarray, fit: np.ndarray) -> None:
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

    # --- scatter: cov pred vs fit ---
    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    xs, ys = [], []
    for pi, (tag, s1, s2) in enumerate(PAIRS):
        ok = np.isfinite(pred[pi]) & np.isfinite(fit[pi])
        x = pred[pi, ok] / EPS_SCALE
        y = fit[pi, ok] / EPS_SCALE
        xs.append(x); ys.append(y)
        ax.scatter(x, y, s=42, color=COLORS[pi], alpha=0.85, edgecolors="white", lw=0.4,
                   label=tag)
    all_x = np.concatenate(xs)
    all_y = np.concatenate(ys)
    lim = max(np.max(np.abs(all_x)), np.max(np.abs(all_y)), 0.05) * 1.12
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.1, alpha=0.6, label=r"$y=x$")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Cov/Var prediction $\varepsilon_{12}^*\;(\times 10^{-3})$")
    ax.set_ylabel(r"parabola-fit dip $\varepsilon_{12}^*\;(\times 10^{-3})$")
    ax.set_title(rf"Sanity check: $-\mathrm{{Cov}}_\varphi(B,x_{{11}})/\mathrm{{Var}}_\varphi(x_{{11}})$ "
                 rf"($N$={N}, $r$={R:g})")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(SCATTER_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {SCATTER_PATH}")

    # --- panel: fit (solid) vs cov pred (dashed) vs linear -2/3 (dotted) ---
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    for pi, (tag, s1, s2) in enumerate(PAIRS):
        lin = -2.0 / 3.0 * (s1 + s2) / EPS_SCALE
        ax.plot(u_arr, fit[pi] / EPS_SCALE, "o-", color=COLORS[pi], lw=1.7, ms=4,
                label=rf"fit: {tag}")
        ax.plot(u_arr, pred[pi] / EPS_SCALE, "s--", color=COLORS[pi], lw=1.2, ms=3,
                alpha=0.9, label=rf"Cov/Var: {tag}")
        if abs(s1 + s2) > 1e-15:
            ax.axhline(lin, color=COLORS[pi], ls=":", lw=0.9, alpha=0.55)

    ax.set_xscale("log")
    ax.set_xlabel(r"mutation $u=v$")
    ax.set_ylabel(r"$\varepsilon_{12}^*\;(\times 10^{-3})$")
    ax.set_title(rf"Fitted dip vs $-\mathrm{{Cov}}_\varphi(B,x_{{11}})/\mathrm{{Var}}_\varphi(x_{{11}})$ "
                 rf"($N$={N}, $r$={R:g})")
    ax.legend(fontsize=5.8, ncol=2, loc="upper left")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(PANEL_PATH, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {PANEL_PATH}")

    print("\n  rel_err vs Cov/Var pred;  abs_err for s1+s2=0 (pred ~ 0)")
    print(f"  {'u':>8}", end="")
    for tag, _, _ in PAIRS:
        print(f"  {tag[:12]:>12}", end="")
    print()
    for ui, u in enumerate(u_arr):
        print(f"  {u:8g}", end="")
        for pi, (_, s1, s2) in enumerate(PAIRS):
            p, f = pred[pi, ui], fit[pi, ui]
            if not (np.isfinite(p) and np.isfinite(f)):
                print(f"  {'—':>12}", end="")
                continue
            if abs(s1 + s2) < 1e-15:
                print(f"  {f - p:+.3e}".rjust(12), end="")
            else:
                rel = (f - p) / max(abs(p), 1e-12)
                print(f"  {rel:+.3f}".rjust(12), end="")
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replot", action="store_true")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    u_ref = np.load(OUT_DIR / "data_u_merged.npz")["u"]

    if args.replot and DATA_PATH.exists():
        d = np.load(DATA_PATH)
        fit = _load_fit_dips(d["u"])
        plot(d["u"], d["pred_cov"], fit)
        return

    pred, fit = compute_all(u_ref)
    plot(u_ref, pred, fit)


if __name__ == "__main__":
    main()
