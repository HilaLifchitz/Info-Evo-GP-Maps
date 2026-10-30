#!/usr/bin/env python3
r"""
TEST the prediction for the location of the KL(psi*||phi*) dip under an eps12 sweep:

    eps12*  =  -2/3 (s1 + s2)      (weak selection, symmetric mutation u = v)

For several (s1, s2) pairs at N=30, we sweep eps12 on a fine grid centred on the
predicted dip, fit a parabola to KL(eps12), read off the empirical vertex
(argmin), and compare it to the prediction. A clean confirmation looks like:
the fitted vertices lying on the y = x line vs the prediction, and (since the
prediction depends only on s1+s2) collapsing onto -2/3 (s1+s2).

Run:
  python3 eps12_dip_prediction_test.py --out plots/eps12_dip_test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eps12_equilibrium_kl_sweep import run_one_N

SCRIPT_DIR = Path(__file__).resolve().parent

N = 30
R = 0.001
U = np.array([1.0e-3, 1.0e-3])
V = np.array([1.0e-3, 1.0e-3])

# (s1, s2) pairs. Kept small so N*max|effect| stays well inside the diffusion
# regime (<~0.05 at N=30). Spread of s1+s2 (hence of predicted eps*), plus two
# pairs sharing the same sum to check the prediction depends only on s1+s2.
PAIRS = [
    (0.0,     0.0),       # sum 0     -> eps* = 0
    (-5.0e-4, 2.5e-4),    # sum -2.5e-4 -> eps* = +1.67e-4
    (1.0e-3, -5.0e-4),    # sum +5e-4  -> eps* = -3.33e-4
    (5.0e-4,  5.0e-4),    # sum +1e-3  -> eps* = -6.67e-4
    (2.0e-3, -1.0e-3),    # sum +1e-3  -> eps* = -6.67e-4 (same sum as above)
    (-5.0e-4, -5.0e-4),   # sum -1e-3  -> eps* = +6.67e-4
]

HALF_WIDTH = 1.0e-3       # eps12 window = eps* +/- HALF_WIDTH
N_GRID = 15               # points across the window (odd -> includes centre)
N_LOCAL = 7               # central points used for the refined local parabola fit


def predicted_eps(s1: float, s2: float) -> float:
    return -2.0 / 3.0 * (s1 + s2)


def parabola_vertex(eps: np.ndarray, kl: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Fit kl ~ a eps^2 + b eps + c; return (vertex, curvature a, coeffs)."""
    a, b, c = np.polyfit(eps, kl, 2)
    vertex = -b / (2.0 * a) if a != 0 else np.nan
    return float(vertex), float(a), np.array([a, b, c])


EPS_SCALE = 1.0e-3          # x-axis plotted in these units for readability
EPS_SCALE_LABEL = r"$\epsilon_{12}\;(\times 10^{-3})$"
KL_LABEL = r"$D_{\mathrm{KL}}(\psi^*\,\|\,\varphi^*)$ [bits]"


def _compute_raw(out_dir: Path):
    """Run the sweep for every pair, return [(eps, kl), ...] and cache to npz."""
    raw = []
    for idx, (s1, s2) in enumerate(PAIRS):
        eps_pred = predicted_eps(s1, s2)
        eps_grid = eps_pred + np.linspace(-HALF_WIDTH, HALF_WIDTH, N_GRID)
        print(f"\n=== pair {idx+1}/{len(PAIRS)}: s1={s1:g}, s2={s2:g}  "
              f"(s1+s2={s1+s2:g}, eps_pred={eps_pred:+.4g}) ===", flush=True)
        res = run_one_N(
            N=N, eps_grid=eps_grid, r=R, u=U, v=V, s1=s1, s2=s2,
            n_jobs=-1, neutral_cache_dir="cache", verbose=False,
        )
        res.sort(key=lambda x: x.eps12)
        eps = np.array([x.eps12 for x in res])
        kl = np.array([x.kl_bits for x in res])
        ok = np.isfinite(kl)
        raw.append((eps[ok], kl[ok]))
    data = {"pairs": np.array(PAIRS, dtype=float)}
    for i, (eps, kl) in enumerate(raw):
        data[f"eps_{i}"] = eps
        data[f"kl_{i}"] = kl
    np.savez(out_dir / "data.npz", **data)
    return raw


def _load_raw(out_dir: Path):
    d = np.load(out_dir / "data.npz")
    n = len(d["pairs"])
    return [(d[f"eps_{i}"], d[f"kl_{i}"]) for i in range(n)]


def run(out_dir: Path, replot: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    except ImportError:
        pass
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "figure.facecolor": "white", "axes.titlesize": 9,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 7})
    out_dir.mkdir(parents=True, exist_ok=True)

    from matplotlib.ticker import MaxNLocator

    raw = _load_raw(out_dir) if replot else _compute_raw(out_dir)

    records = []
    log = [f"N={N}, r={R}, u=v={U[0]:g}",
           "prediction: eps12* = -2/3 (s1+s2)", ""]

    ncol = 3
    nrow = int(np.ceil(len(PAIRS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.2 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for idx, (s1, s2) in enumerate(PAIRS):
        eps_pred = predicted_eps(s1, s2)
        eps, kl = raw[idx]

        vtx_full, a_full, coef = parabola_vertex(eps, kl)
        j = int(np.argmin(kl))
        lo = max(0, j - N_LOCAL // 2)
        hi = min(len(eps), lo + N_LOCAL)
        lo = max(0, hi - N_LOCAL)
        vtx_loc, a_loc, _ = parabola_vertex(eps[lo:hi], kl[lo:hi])

        err = vtx_loc - eps_pred
        records.append((s1, s2, s1 + s2, eps_pred, vtx_full, vtx_loc, a_loc, err))
        log.append(f"s1={s1:+.4g} s2={s2:+.4g}  eps_pred={eps_pred:+.5g}  "
                   f"fit_full={vtx_full:+.5g}  fit_local={vtx_loc:+.5g}  "
                   f"err(local-pred)={err:+.3g}  curvature={a_loc:.3g}")
        print(log[-1], flush=True)

        ax = axes[idx]
        es = EPS_SCALE
        ax.plot(eps / es, kl, "o", ms=4, color="#3b528b", label="KL (chain)")
        xx = np.linspace(eps.min(), eps.max(), 200)
        ax.plot(xx / es, coef[0] * xx**2 + coef[1] * xx + coef[2], "-",
                color="#5ec962", lw=1.4, alpha=0.8, label="parabola fit")
        ax.axvline(eps_pred / es, color="0.4", ls="--", lw=1.3,
                   label=rf"pred $={eps_pred/es:+.2f}$")
        ax.axvline(vtx_loc / es, color="#b5367a", ls=":", lw=1.7,
                   label=rf"fit $={vtx_loc/es:+.2f}$")
        ax.set_title(rf"$s_1$={s1:g}, $s_2$={s2:g} ($s_1+s_2$={s1+s2:g})")
        ax.set_xlabel(EPS_SCALE_LABEL)
        ax.set_ylabel(KL_LABEL)
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10], prune="both"))
        ax.tick_params(axis="x", labelsize=8)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.legend(loc="upper center", ncol=2, columnspacing=0.8,
                  handlelength=1.4, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.02)

    for k in range(len(PAIRS), len(axes)):
        axes[k].set_visible(False)
    fig.suptitle(rf"KL dip location vs prediction $\varepsilon_{{12}}^*=-\frac{{2}}{{3}}(s_1+s_2)$ "
                 rf"($N$={N}, $u=v$={U[0]:g}, $r$={R})", y=1.0, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98), w_pad=2.0, h_pad=2.0)
    p1 = out_dir / "dip_curves.png"
    fig.savefig(p1, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"\nWrote {p1}")

    # summary scatter: fitted vertex vs predicted (plotted in 1e-3 units)
    rec = np.array([(r[3], r[5]) for r in records]) / EPS_SCALE
    fig2, ax2 = plt.subplots(figsize=(6.2, 6.0))
    lo = min(rec.min(), 0) * 1.2
    hi = max(rec.max(), 0) * 1.2
    ax2.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax2.scatter(rec[:, 0], rec[:, 1], s=60, color="#b5367a", zorder=3)
    for (s1, s2, ssum, ep, vf, vl, a, e) in records:
        ax2.annotate(f"({s1:g},{s2:g})", (ep / EPS_SCALE, vl / EPS_SCALE), fontsize=7,
                     textcoords="offset points", xytext=(5, 4))
    ax2.set_xlabel(r"predicted $-\frac{2}{3}(s_1+s_2)\;(\times 10^{-3})$")
    ax2.set_ylabel(r"fitted dip $\varepsilon_{12}^*$ (local parabola, $\times 10^{-3}$)")
    ax2.set_title(f"Prediction vs measured KL-dip location (N={N})")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    p2 = out_dir / "dip_pred_vs_fit.png"
    fig2.savefig(p2, dpi=170, bbox_inches="tight"); plt.close(fig2)
    print(f"Wrote {p2}")

    (out_dir / "dip_test.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'dip_test.log'}")

    print("\n  s1       s2       s1+s2     pred       fit_local   err")
    for (s1, s2, ssum, ep, vf, vl, a, e) in records:
        print(f"  {s1:+.4g}  {s2:+.4g}  {ssum:+.4g}  {ep:+.5g}  {vl:+.5g}  {e:+.3g}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=SCRIPT_DIR / "plots" / "eps12_dip_test")
    p.add_argument("--replot", action="store_true",
                   help="skip recompute; reload cached data.npz and redraw")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.out.resolve(), replot=args.replot)


if __name__ == "__main__":
    main()
