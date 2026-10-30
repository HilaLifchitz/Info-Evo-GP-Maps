#!/usr/bin/env python3
r"""
Analyse info_speed_experiment.py output: is the EQUILIBRIUM information score
predictive of the SPEED of information accumulation, or is it free fitness, or
the (iso-free-fitness) epistasis decomposition?

Produces, per (u,v,r) regime (each a separate "clock"):
  * Spearman correlations of speed metrics vs predictors
  * PARTIAL Spearman correlations (speed ~ D_eq | Fstar, and speed ~ Fstar | D_eq,
    and speed ~ valley | D_eq, Fstar)   -> which survives controlling the others
  * iso-information level-set analysis (bin by D_eq, residual speed vs Fstar/valley)
  * iso-free-fitness level-set analysis (bin by Fstar, residual speed vs valley)

Part 1 (A-E family): do the epistatic routes' arrival times track their info?

Figures + ordering_report.txt written next to the CSV (in <indir>/analysis/).

Run:
  python info_speed_analyze.py --indir results/info_speed_YYYYMMDD
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr

SPEED = ["t_half_info", "t95_info", "tau_spec"]     # larger = slower
SPEED_LABEL = {"t_half_info": r"$t_{1/2}$ (info)",
               "t95_info": r"$t_{95}$ (arrival)",
               "tau_spec": r"$\tau_{spec}=-1/\ln\lambda_2$"}
# rate targets: LARGER = FASTER (opposite sign convention to the time targets)
RATE = ["init_rate", "avg_rate_50", "avg_rate_100", "kl_at_100"]
RATE_LABEL = {"init_rate": "init rate dKL/dt",
              "avg_rate_50": "avg rate t<=50", "avg_rate_100": "avg rate t<=100",
              "kl_at_100": "KL(t=100)"}
# targets used for the info-vs-free-fitness head-to-head (sign = "faster")
#   value is +1 if larger-target means faster, -1 if larger means slower
TARGETS = {"init_rate": +1, "avg_rate_100": +1, "kl_at_100": +1,
           "t95_info": -1, "tau_spec": -1}


def route_type(row) -> str:
    """Coarse epistasis route class from #uphill paths & winner."""
    n = int(row.n_mono)
    if n == 2:
        return "2-path (additive-like)"
    if n == 1:
        return "1-path (sign)"
    # n == 0 : no monotone uphill route to 11
    return "0-path (reciprocal/valley)" if row.margin > 0 else "0-path (11 not peak)"
PRED = ["D_eq", "Fstar", "mlf", "valley", "barrier", "margin", "n_mono", "scale"]
PRED_LABEL = {"D_eq": r"$D_{eq}$ info", "Fstar": r"$F^*$ free-fit",
              "mlf": r"$\langle\log W\rangle$", "valley": "valley depth",
              "barrier": "path barrier", "margin": "winner margin",
              "n_mono": "# uphill paths", "scale": "sel. scale"}
CASE_NAME = {"A": "sign", "B": "additive", "C": "reciprocal",
             "D": "neutral-int", "E": "multi-peak"}
CASE_COLOR = {"A": "#c44e52", "B": "#4c72b0", "C": "#55a868",
              "D": "#8172b3", "E": "#dd8452"}


# --------------------------------------------------------------------------
def load(indir: Path):
    import pandas as pd
    df = pd.read_csv(indir / "info_speed.csv")
    curves = dict(np.load(indir / "curves.npz"))
    return df, curves


def _resid_rank(y_rank, Z_rank):
    """Residual of y_rank after linear regression on columns of Z_rank (+const)."""
    X = np.column_stack([np.ones(len(y_rank))] + [Z_rank[:, j]
                                                  for j in range(Z_rank.shape[1])])
    beta, *_ = np.linalg.lstsq(X, y_rank, rcond=None)
    return y_rank - X @ beta


def partial_spearman(x, y, Zcols):
    """Partial Spearman corr of x,y controlling for columns in Zcols (list of arrays)."""
    m = np.isfinite(x) & np.isfinite(y)
    for z in Zcols:
        m &= np.isfinite(z)
    if m.sum() < 5:
        return float("nan")
    xr = rankdata(x[m]); yr = rankdata(y[m])
    if not Zcols:
        r, _ = spearmanr(x[m], y[m]); return float(r)
    Z = np.column_stack([rankdata(z[m]) for z in Zcols])
    rx = _resid_rank(xr, Z); ry = _resid_rank(yr, Z)
    if rx.std() < 1e-12 or ry.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def sp(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return float("nan")
    r, _ = spearmanr(x[m], y[m])
    return float(r)


# --------------------------------------------------------------------------
def part1_AE(df, curves, regime, outdir):
    sub = df[(df.regime == regime) & (df.family == "AE")]
    avals = sorted(sub.a.unique())
    fig, axes = plt.subplots(2, len(avals), figsize=(4.6 * len(avals), 8.4),
                             squeeze=False)
    lines = [f"REGIME {regime}  --- Part 1: A-E arrival-time vs info", ""]
    for j, a in enumerate(avals):
        s = sub[sub.a == a]
        ax = axes[0, j]
        for _, row in s.iterrows():
            key = f"{regime}__AE_{row.case}_a{a:g}"
            if key + "__t" in curves:
                t = curves[key + "__t"]; D = curves[key + "__D"]
                ax.plot(t, D, "-", lw=1.8, color=CASE_COLOR[row.case],
                        label=f"{row.case}:{CASE_NAME[row.case]} "
                              f"(D={row.D_eq:.2f})")
        ax.set_xscale("log"); ax.grid(True, alpha=0.3)
        ax.set_title(f"a={a:g}: info gain $D(t)$"); ax.legend(fontsize=7)
        ax.set_xlabel("generation $t$"); ax.set_ylabel("$D(t)$ [bits]")

        ax2 = axes[1, j]
        for _, row in s.iterrows():
            ax2.scatter(row.D_eq, row.t95_info, s=90, color=CASE_COLOR[row.case],
                        zorder=3)
            ax2.annotate(row.case, (row.D_eq, row.t95_info), fontsize=9,
                         ha="center", va="center")
        ax2.set_xlabel(r"$D_{eq}$ (info at equilibrium) [bits]")
        ax2.set_ylabel(r"$t_{95}$ arrival time [gen]")
        ax2.grid(True, alpha=0.3)
        # rank agreement between info and arrival-time
        rr = sp(s.D_eq.values, s.t95_info.values)
        rh = sp(s.D_eq.values, s.t_half_info.values)
        rt = sp(s.D_eq.values, s.tau_spec.values)
        ax2.set_title(f"a={a:g}: Spearman(D,t95)={rr:+.2f}")
        lines.append(f"a={a:g}:  Spearman(D_eq, t95)={rr:+.2f}   "
                     f"(D,t_half)={rh:+.2f}   (D,tau_spec)={rt:+.2f}")
        # per-case detail sorted by info
        for _, row in s.sort_values("D_eq").iterrows():
            lines.append(f"    {row.case}:{CASE_NAME[row.case]:11s} "
                         f"D_eq={row.D_eq:5.2f}  F*={row.Fstar:+.3f}  "
                         f"mlf={row.mlf:+.3f}  t_half={row.t_half_info:6.1f}  "
                         f"t95={row.t95_info:7.1f}  tau={row.tau_spec:6.1f}  "
                         f"valley={row.valley:+.2f} margin={row.margin:+.2f}")
        lines.append("")
    fig.suptitle(f"Part 1  {regime}: do epistatic routes' arrival times track "
                 f"their equilibrium information?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(outdir / f"fig_partAE_{regime}.png", dpi=140)
    plt.close(fig)
    return lines


def corr_report(df, regime, outdir):
    g = df[(df.regime == regime) & (df.family == "grid")].copy()
    g = g[np.isfinite(g.D_eq) & (g.D_eq > 1e-6)]
    lines = [f"REGIME {regime}  --- Part 2: grid correlations (n={len(g)})", ""]

    # Spearman table SPEED x PRED
    lines.append("Spearman(speed, predictor):")
    header = "  " + " ".join(f"{p:>10s}" for p in PRED)
    lines.append("            " + header)
    M = np.full((len(SPEED), len(PRED)), np.nan)
    for i, sm in enumerate(SPEED):
        cells = []
        for j, p in enumerate(PRED):
            r = sp(g[sm].values, g[p].values); M[i, j] = r
            cells.append(f"{r:+.2f}".rjust(10))
        lines.append(f"  {sm:>10s}  " + " ".join(cells))
    lines.append("")

    # partial correlations (control the OTHER main candidate)
    lines.append("PARTIAL Spearman (does the association SURVIVE controls?):")
    for sm in SPEED:
        x = g[sm].values
        r_D = sp(x, g.D_eq.values)
        r_F = sp(x, g.Fstar.values)
        pD_givenF = partial_spearman(x, g.D_eq.values, [g.Fstar.values])
        pF_givenD = partial_spearman(x, g.Fstar.values, [g.D_eq.values])
        pValley = partial_spearman(x, g.valley.values,
                                   [g.D_eq.values, g.Fstar.values])
        pBarrier = partial_spearman(x, g.barrier.values,
                                    [g.D_eq.values, g.Fstar.values])
        lines.append(
            f"  {sm:>10s}:  D_eq raw={r_D:+.2f}->|F*={pD_givenF:+.2f}   "
            f"F* raw={r_F:+.2f}->|D={pF_givenD:+.2f}   "
            f"valley|D,F*={pValley:+.2f}   barrier|D,F*={pBarrier:+.2f}")
    lines.append("")
    return lines, M, g


def rate_route_report(df, regime):
    g = df[(df.regime == regime) & (df.family == "grid")].copy()
    g = g[np.isfinite(g.D_eq) & (g.D_eq > 1e-6)]
    lines = [f"REGIME {regime} --- rate targets & route-conditional (n={len(g)})", ""]
    lines.append("Head-to-head INFO (D_eq) vs FREE FITNESS (F*): "
                 "rho, and PARTIAL controlling the other:")
    lines.append(f"  {'target':>14s} {'rho(D)':>7s} {'D|F*':>7s}  "
                 f"{'rho(F*)':>7s} {'F*|D':>7s}   winner")
    for tgt in TARGETS:
        x = g[tgt].values
        rD = sp(x, g.D_eq.values); rF = sp(x, g.Fstar.values)
        pD = partial_spearman(x, g.D_eq.values, [g.Fstar.values])
        pF = partial_spearman(x, g.Fstar.values, [g.D_eq.values])
        win = "D_eq" if abs(pD) >= abs(pF) else "F*"
        lines.append(f"  {tgt:>14s} {rD:+7.2f} {pD:+7.2f}  {rF:+7.2f} {pF:+7.2f}   {win}")
    lines.append("")
    g["route"] = g.apply(route_type, axis=1)
    lines.append("Route-conditional predictiveness (within each epistasis route):")
    for rt, grt in g.groupby("route"):
        if len(grt) < 5:
            lines.append(f"  [{rt}] n={len(grt)} (too few)")
            continue
        lines.append(f"  [{rt}] n={len(grt)}:")
        for tgt in ("avg_rate_100", "t95_info"):
            rD = sp(grt[tgt].values, grt.D_eq.values)
            rF = sp(grt[tgt].values, grt.Fstar.values)
            rV = sp(grt[tgt].values, grt.valley.values)
            lines.append(f"      {tgt:>13s}: rho(D_eq)={rD:+.2f}  rho(F*)={rF:+.2f}  "
                         f"rho(valley)={rV:+.2f}")
    lines.append("")
    return lines


def fig_pred_compare(df, regimes, outdir):
    tgts = list(TARGETS)
    fig, axes = plt.subplots(1, len(regimes), figsize=(4.2 * len(regimes), 4.4),
                             squeeze=False)
    for k, reg in enumerate(regimes):
        g = df[(df.regime == reg) & (df.family == "grid")].copy()
        g = g[np.isfinite(g.D_eq) & (g.D_eq > 1e-6)]
        pDs, pFs = [], []
        for tgt in tgts:
            x = g[tgt].values
            pDs.append(partial_spearman(x, g.D_eq.values, [g.Fstar.values]))
            pFs.append(partial_spearman(x, g.Fstar.values, [g.D_eq.values]))
        ax = axes[0, k]
        xpos = np.arange(len(tgts)); w = 0.38
        ax.bar(xpos - w / 2, np.abs(pDs), w, label=r"$D_{eq}\,|\,F^*$",
               color="#4c72b0")
        ax.bar(xpos + w / 2, np.abs(pFs), w, label=r"$F^*\,|\,D_{eq}$",
               color="#c44e52")
        ax.set_xticks(xpos)
        ax.set_xticklabels(tgts, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1); ax.set_ylabel("|partial Spearman|")
        ax.set_title(reg, fontsize=7); ax.grid(True, alpha=0.3, axis="y")
        if k == 0:
            ax.legend(fontsize=9)
    fig.suptitle("Head-to-head: does equilibrium INFO or FREE FITNESS predict speed "
                 "after controlling for the other?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outdir / "fig_pred_compare.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_scatter(df, regime, outdir):
    g = df[(df.regime == regime) & (df.family == "grid")].copy()
    g = g[np.isfinite(g.D_eq) & (g.D_eq > 1e-6)]
    keypred = ["D_eq", "Fstar", "valley", "barrier"]
    fig, axes = plt.subplots(len(SPEED), len(keypred),
                             figsize=(4.2 * len(keypred), 3.4 * len(SPEED)),
                             squeeze=False)
    for i, sm in enumerate(SPEED):
        for j, p in enumerate(keypred):
            ax = axes[i, j]
            sc = ax.scatter(g[p], g[sm], c=g["valley"], cmap="viridis",
                            s=26, alpha=0.8)
            ax.set_xlabel(PRED_LABEL[p]); ax.set_ylabel(SPEED_LABEL[sm])
            ax.grid(True, alpha=0.3)
            ax.set_title(f"$\\rho$={sp(g[p].values, g[sm].values):+.2f}",
                         fontsize=9)
    fig.colorbar(sc, ax=axes, shrink=0.6, label="valley depth")
    fig.suptitle(f"{regime}: speed vs predictors (color = valley depth)",
                 fontsize=12)
    fig.savefig(outdir / f"fig_scatter_{regime}.png", dpi=140,
                bbox_inches="tight")
    plt.close(fig)


def fig_isolevels(df, regime, outdir):
    """Within iso-info bins, is leftover speed spread explained by F*/valley?
    Within iso-F* bins, by valley?  Uses t95_info as the arrival-speed metric."""
    g = df[(df.regime == regime) & (df.family == "grid")].copy()
    g = g[np.isfinite(g.D_eq) & (g.D_eq > 1e-6) & np.isfinite(g.t95_info)]
    sm = "t95_info"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # iso-information: bin D_eq into terciles, look at speed vs F* within each
    qs = np.quantile(g.D_eq, [0, 1 / 3, 2 / 3, 1.0])
    lines = [f"REGIME {regime}  --- iso-level sets (speed metric={sm})", ""]
    ax = axes[0]
    for b in range(3):
        lo, hi = qs[b], qs[b + 1]
        m = (g.D_eq >= lo) & (g.D_eq <= hi if b == 2 else g.D_eq < hi)
        gb = g[m]
        if len(gb) < 4:
            continue
        r = sp(gb.Fstar.values, gb[sm].values)
        rv = sp(gb.valley.values, gb[sm].values)
        ax.scatter(gb.Fstar, gb[sm], s=30, alpha=0.8,
                   label=f"D_eq∈[{lo:.2f},{hi:.2f}]  ρ(F*,t95)={r:+.2f}")
        lines.append(f"  iso-info bin D_eq∈[{lo:.2f},{hi:.2f}] (n={len(gb)}): "
                     f"ρ(F*,t95)={r:+.2f}  ρ(valley,t95)={rv:+.2f}")
    ax.set_xlabel(r"$F^*$ free fitness"); ax.set_ylabel(SPEED_LABEL[sm])
    ax.set_title("iso-INFORMATION bins: speed vs $F^*$")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # iso-free-fitness: bin Fstar into terciles, speed vs valley within each
    qs = np.quantile(g.Fstar, [0, 1 / 3, 2 / 3, 1.0])
    ax = axes[1]
    lines.append("")
    for b in range(3):
        lo, hi = qs[b], qs[b + 1]
        m = (g.Fstar >= lo) & (g.Fstar <= hi if b == 2 else g.Fstar < hi)
        gb = g[m]
        if len(gb) < 4:
            continue
        r = sp(gb.valley.values, gb[sm].values)
        rd = sp(gb.D_eq.values, gb[sm].values)
        ax.scatter(gb.valley, gb[sm], s=30, alpha=0.8,
                   label=f"F*∈[{lo:.2f},{hi:.2f}]  ρ(valley,t95)={r:+.2f}")
        lines.append(f"  iso-F* bin F*∈[{lo:.2f},{hi:.2f}] (n={len(gb)}): "
                     f"ρ(valley,t95)={r:+.2f}  ρ(D_eq,t95)={rd:+.2f}")
    ax.set_xlabel("valley depth"); ax.set_ylabel(SPEED_LABEL[sm])
    ax.set_title("iso-FREE-FITNESS bins: speed vs valley depth")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle(f"{regime}: level-set decoupling", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outdir / f"fig_isolevels_{regime}.png", dpi=140)
    plt.close(fig)
    return lines


def fig_corr_heatmap(Ms, regimes, outdir):
    n = len(regimes)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.6), squeeze=False)
    for k, (reg, M) in enumerate(zip(regimes, Ms)):
        ax = axes[0, k]
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(PRED)))
        ax.set_xticklabels([PRED_LABEL[p] for p in PRED], rotation=60, ha="right",
                           fontsize=7)
        ax.set_yticks(range(len(SPEED)))
        ax.set_yticklabels([SPEED_LABEL[s] for s in SPEED], fontsize=7)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i,j]:+.2f}", ha="center", va="center",
                        fontsize=6.5)
        ax.set_title(reg, fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.7, label="Spearman ρ")
    fig.suptitle("Speed (rows) vs predictors (cols): Spearman per regime",
                 fontsize=11)
    fig.savefig(outdir / "fig_corr_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, required=True)
    args = ap.parse_args()
    outdir = args.indir / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    df, curves = load(args.indir)
    regimes = list(dict.fromkeys(df.regime.tolist()))
    report = [f"info-speed analysis  ({args.indir})",
              f"regimes: {regimes}", "=" * 70, ""]

    Ms = []
    for reg in regimes:
        report += ["#" * 70, ""]
        report += part1_AE(df, curves, reg, outdir)
        cl, M, _ = corr_report(df, reg, outdir)
        report += cl
        Ms.append(M)
        report += rate_route_report(df, reg)
        fig_scatter(df, reg, outdir)
        report += fig_isolevels(df, reg, outdir)
        report += [""]

    fig_corr_heatmap(Ms, regimes, outdir)
    fig_pred_compare(df, regimes, outdir)

    (outdir / "ordering_report.txt").write_text("\n".join(report) + "\n")
    print("wrote", outdir / "ordering_report.txt")
    print("figures in", outdir)


if __name__ == "__main__":
    main()
