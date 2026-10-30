"""Strategy-resolved views for the info-vs-speed dispute.

FOUR strategies (by how many of the two 00->11 routes are locally 'alluring',
i.e. monotone uphill in single steps), exactly as the user framed them:

  S2  '2 alluring (additive)'          n_mono == 2   both routes uphill
  S1  '1 alluring (sign)'              n_mono == 1   one route uphill, one not
  S0v '0 alluring - valley (recip)'    n_mono == 0, margin>0   11 still the peak,
                                                    but intermediates neutral/down
  S0x '0 routes (11 not peak)'         n_mono == 0, margin<=0  11 is not the winner

Figures (existing CSV = info_speed_20260727; new eta CSV = info_speed_20260728):
  fig_within_strategy.png  : within each strategy, is D_eq indicative of tau?
  fig_route_positioning.png: route means -- info/free-fit  vs  speed/efficiency
"""
from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from info_speed_analyze import partial_spearman

SCRIPT_DIR = Path(__file__).resolve().parent
RES = SCRIPT_DIR / "results"
OLD = RES / "info_speed_20260727"          # has D_eq, tau, t95, Fstar, scale
NEW = RES / "info_speed_20260728"          # has eta/cost (in progress)

S2, S1, S0V, S0X = ("2 alluring (additive)", "1 alluring (sign)",
                    "0 alluring - valley (recip)", "0 routes (11 not peak)")
ORDER = [S2, S1, S0V, S0X]
COL = {S2: "#1b9e77", S1: "#d95f02", S0V: "#7570b3", S0X: "#999999"}


def strategy(row) -> str:
    n = int(row.n_mono)
    if n == 2:
        return S2
    if n == 1:
        return S1
    return S0V if row.margin > 0 else S0X


def load(path: Path) -> pd.DataFrame | None:
    p = path / "info_speed.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["strat"] = df.apply(strategy, axis=1)
    return df


def _rho(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return float("nan")
    return float(spearmanr(x[m], y[m])[0])


# --------------------------------------------------------------------------
# FIG 1: within a single strategy, does info order relaxation time?
# --------------------------------------------------------------------------
def fig_within(df: pd.DataFrame, clock: str):
    d0 = df[df.regime == clock]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    for ax, st in zip(axes.ravel(), ORDER):
        g = d0[d0.strat == st]
        if len(g) < 3:
            ax.set_title(f"{st}\n(n={len(g)} too few)"); continue
        sc = ax.scatter(g.D_eq, g.tau_spec, c=g.scale, cmap="viridis",
                        s=40, edgecolor="k", linewidth=.3)
        ax.set_yscale("log")
        raw = _rho(g.D_eq.values, g.tau_spec.values)
        par = partial_spearman(g.D_eq.values, g.tau_spec.values, [g.scale.values])
        ax.set_title(f"{st}   (n={len(g)})\n"
                     rf"$\rho(D_{{eq}},\tau)$ raw={raw:+.2f}   "
                     rf"$|\,$scale={par:+.2f}", fontsize=10)
        ax.set_xlabel(r"$D_{eq}$ information (bits)")
        ax.set_ylabel(r"$\tau_{spec}$ relaxation time (gen, log)")
        cb = fig.colorbar(sc, ax=ax); cb.set_label("selection scale", fontsize=8)
    fig.suptitle("Within each strategy: is equilibrium information indicative of "
                 f"relaxation speed?\nclock: {clock}   (colour = selection scale)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OLD / "fig_within_strategy.png"
    fig.savefig(out, dpi=130); print("wrote", out)


# --------------------------------------------------------------------------
# FIG 2: route means -- info & free fitness  vs  speed & efficiency
# --------------------------------------------------------------------------
def fig_positioning(df: pd.DataFrame, dfe: pd.DataFrame | None):
    """Each panel: 4 route-mean points. x in {D_eq, F*}; y in {tau, eta}."""
    def mean_sem(frame, col):
        g = frame.groupby("strat")[col]
        return g.mean(), g.sem()

    have_eta = dfe is not None and "eta_100" in (dfe.columns if dfe is not None else [])
    ncol = 2 if not have_eta else 2
    nrow = 1 if not have_eta else 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 5.0 * nrow),
                             squeeze=False)

    def scatter_panel(ax, frame, xcol, ycol, xlab, ylab, ylog=False):
        mx, sx = mean_sem(frame, xcol)
        my, sy = mean_sem(frame, ycol)
        for st in ORDER:
            if st not in mx.index:
                continue
            ax.errorbar(mx[st], my[st], xerr=sx[st], yerr=sy[st], fmt="o",
                        ms=13, c=COL[st], capsize=4, label=st)
        if ylog:
            ax.set_yscale("log")
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)

    # top row: speed (tau) vs info and free fitness (existing data)
    scatter_panel(axes[0][0], df, "D_eq", "tau_spec",
                  r"mean $D_{eq}$ (info, bits)", r"mean $\tau_{spec}$ (slow$\to$)")
    axes[0][0].set_title("(i) info  vs  relaxation time")
    scatter_panel(axes[0][1], df, "Fstar", "tau_spec",
                  r"mean $F^*$ (free fitness)", r"mean $\tau_{spec}$ (slow$\to$)")
    axes[0][1].set_title("(ii) free fitness  vs  relaxation time")
    axes[0][0].legend(fontsize=8, loc="best")

    # bottom row: efficiency (eta) vs info and free fitness (needs new data)
    if have_eta:
        dfe2 = dfe.copy(); dfe2["strat"] = dfe2.apply(strategy, axis=1)
        # merge F* etc for eta frame already present in dfe (it has Fstar,D_eq)
        scatter_panel(axes[1][0], dfe2, "D_eq", "eta_100",
                      r"mean $D_{eq}$ (info, bits)",
                      r"mean $\eta_{100}$ (efficiency, fast$\uparrow$)")
        axes[1][0].set_title(r"(iii) info  vs  efficiency $\eta$")
        scatter_panel(axes[1][1], dfe2, "Fstar", "eta_100",
                      r"mean $F^*$ (free fitness)",
                      r"mean $\eta_{100}$ (efficiency, fast$\uparrow$)")
        axes[1][1].set_title(r"(iv) free fitness  vs  efficiency $\eta$")

    tag = "info/free-fitness  vs  relaxation-time" + (" & efficiency" if have_eta
                                                      else " (efficiency pending)")
    fig.suptitle(f"Strategy positioning: {tag}\n"
                 "points = route means +/- SEM, pooled over clocks", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OLD / "fig_route_positioning.png"
    fig.savefig(out, dpi=130); print("wrote", out)


# --------------------------------------------------------------------------
# FIG 3: multi-metric bar comparison across strategies (info, F*, speed[, eta])
# --------------------------------------------------------------------------
def fig_bars(df: pd.DataFrame, dfe: pd.DataFrame | None):
    have_eta = dfe is not None and "eta_100" in (dfe.columns if dfe is not None else [])
    metrics = [("D_eq", "info $D_{eq}$ (bits)", df, False),
               ("Fstar", "free fitness $F^*$", df, False),
               ("tau_spec", r"relaxation $\tau$ (gen)", df, False),
               ("t95_info", r"arrival $t_{95}$ (gen)", df, False)]
    if have_eta:
        de = dfe.copy(); de["strat"] = de.apply(strategy, axis=1)
        metrics += [("eta_100", r"efficiency $\eta_{100}$", de, False),
                    ("eta_peak", r"peak efficiency $\eta$", de, False)]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 4.2), squeeze=False)
    for ax, (col, lab, frame, _) in zip(axes[0], metrics):
        m = frame.groupby("strat")[col].mean()
        e = frame.groupby("strat")[col].sem()
        xs = [s for s in ORDER if s in m.index]
        ax.bar(range(len(xs)), [m[s] for s in xs], yerr=[e[s] for s in xs],
               color=[COL[s] for s in xs], capsize=3)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([s.split(" ")[0] + "\n" + s.split("(")[-1].rstrip(")")
                            for s in xs], fontsize=7)
        ax.set_title(lab, fontsize=9)
    fig.suptitle("Strategy comparison across metrics (mean +/- SEM, pooled clocks)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OLD / "fig_strategy_bars.png"
    fig.savefig(out, dpi=130); print("wrote", out)


def main():
    df = load(OLD)
    dfe = load(NEW)
    clock = df.regime.unique()[0]
    fig_within(df, clock)
    fig_positioning(df, dfe)
    fig_bars(df, dfe)
    # quick text: within-strategy raw & partial rho for tau and t95
    print("\nwithin-strategy rho(D_eq, TARGET)  [pooled clocks]  raw | partial|scale")
    for st in ORDER:
        g = df[df.strat == st]
        for tgt in ("tau_spec", "t95_info"):
            raw = _rho(g.D_eq.values, g[tgt].values)
            par = partial_spearman(g.D_eq.values, g[tgt].values, [g.scale.values])
            print(f"  {st:<28} {tgt:<9} raw={raw:+.3f}  |scale={par:+.3f}  n={len(g)}")


if __name__ == "__main__":
    main()
