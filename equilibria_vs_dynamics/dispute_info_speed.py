"""Dispute the claim  "smaller equilibrium information  =>  faster to reach it".

The claim, stated as a monotone rule, predicts a POSITIVE rank-correlation between
D_eq (equilibrium information) and the time-to-equilibrium targets (t95_info,
tau_spec): small info -> small time.  We test this head-on on the N=30 sweep
(760 landscapes x 4 mutation/recombination clocks) and try to break it three ways:

  1. sign of the RAW correlation                         (is the rule even true?)
  2. remove the selection-scale confound (partial rho | scale)
     -- larger info usually rides on stronger selection, which mixes faster; that
        is NOT "information helping", it is just a hotter clock.
  3. within a fixed epistasis ROUTE (additive / sign / reciprocal-valley), and
     across routes at MATCHED selection scale -- look for explicit counterexamples
     (low info yet slow;  high info yet fast).

Outputs:  results/<src>/dispute_report.txt , fig_dispute.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr

from info_speed_analyze import route_type, partial_spearman

SCRIPT_DIR = Path(__file__).resolve().parent
SRC = SCRIPT_DIR / "results" / (sys.argv[1] if len(sys.argv) > 1
                                else "info_speed_20260727")
TIME = ["t95_info", "tau_spec"]        # larger = slower to equilibrium
TIME_LABEL = {"t95_info": r"$t_{95}$ (arrival)",
              "tau_spec": r"$\tau_{spec}$"}


def _rho(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return float("nan"), 0
    r, _ = spearmanr(x[m], y[m])
    return float(r), int(m.sum())


def main() -> None:
    df = pd.read_csv(SRC / "info_speed.csv")
    df["route"] = df.apply(route_type, axis=1)
    clocks = list(df.regime.unique())
    L = []
    p = L.append

    p("=" * 78)
    p("DISPUTE:  \"smaller equilibrium information  =>  faster to equilibrium\"")
    p("=" * 78)
    p("Claim as a monotone rule  =>  corr(D_eq, TIME) should be POSITIVE")
    p("  (small info -> small arrival time).   TIME = t95_info, tau_spec.")
    p("A negative or ~zero or route-dependent sign REFUTES the rule.\n")

    # ---- (1)+(2) raw vs scale-controlled vs route-controlled, per clock ------
    p("-" * 78)
    p("(1)/(2)  rho(D_eq, TIME):  raw  |  partial|scale  |  partial|scale+route")
    p("-" * 78)
    hdr = f"{'clock':<26}{'target':<11}{'raw':>8}{'|scale':>9}{'|scl+rte':>10}   n"
    p(hdr)
    routef = ["scale"]
    for clk in clocks:
        d = df[df.regime == clk]
        for tgt in TIME:
            raw, n = _rho(d.D_eq.values, d[tgt].values)
            ps = partial_spearman(d.D_eq.values, d[tgt].values,
                                  [d.scale.values])
            psr = partial_spearman(d.D_eq.values, d[tgt].values,
                                   [d.scale.values, d.valley.values,
                                    d.n_mono.values.astype(float)])
            p(f"{clk:<26}{tgt:<11}{raw:>8.3f}{ps:>9.3f}{psr:>10.3f}   {n}")
        p("")

    # ---- (3a) route reversal: reciprocal valleys = high info + slow ----------
    p("-" * 78)
    p("(3a)  ROUTE means (pooled over clocks): does more info track faster?")
    p("-" * 78)
    p(f"{'route':<30}{'n':>5}{'D_eq':>8}{'scale':>8}{'t95':>9}{'tau':>9}")
    order = ["2-path (additive-like)", "1-path (sign)",
             "0-path (reciprocal/valley)", "0-path (11 not peak)"]
    for rt in order:
        d = df[df.route == rt]
        if len(d) == 0:
            continue
        p(f"{rt:<30}{len(d):>5}{d.D_eq.mean():>8.3f}{d.scale.mean():>8.3f}"
          f"{d.t95_info.mean():>9.1f}{d.tau_spec.mean():>9.1f}")
    p("\n  If the rule held, the row with the SMALLEST D_eq would be FASTEST.")

    # ---- (3b) sign of rho(D_eq,TIME) WITHIN each route (scale-controlled) -----
    p("\n" + "-" * 78)
    p("(3b)  partial rho(D_eq, TIME | scale) WITHIN each route  (pooled clocks)")
    p("-" * 78)
    p(f"{'route':<30}{'n':>5}{'t95|scale':>11}{'tau|scale':>11}")
    for rt in order:
        d = df[df.route == rt]
        if len(d) < 8:
            continue
        r95 = partial_spearman(d.D_eq.values, d.t95_info.values, [d.scale.values])
        rtau = partial_spearman(d.D_eq.values, d.tau_spec.values, [d.scale.values])
        p(f"{rt:<30}{len(d):>5}{r95:>11.3f}{rtau:>11.3f}")

    # ---- (3c) explicit counterexamples at MATCHED selection scale ------------
    p("\n" + "-" * 78)
    p("(3c)  COUNTEREXAMPLES at matched selection scale (r=0 weak-mut clock)")
    p("-" * 78)
    d = df[df.regime == clocks[0]].copy()
    d = d[np.isfinite(d.t95_info)]
    # scale quintiles; within each, tercile flags on info & arrival time
    d["sbin"] = pd.qcut(d.scale, 5, labels=False, duplicates="drop")
    lo_info_slow = []   # small D_eq yet SLOW  -> contradicts "small info fast"
    hi_info_fast = []   # large D_eq yet FAST  -> contradicts "small info fast"
    for b, g in d.groupby("sbin"):
        if len(g) < 6:
            continue
        di_lo = g.D_eq.quantile(1 / 3.)
        di_hi = g.D_eq.quantile(2 / 3.)
        t_lo = g.t95_info.quantile(1 / 3.)
        t_hi = g.t95_info.quantile(2 / 3.)
        lo_info_slow.append(g[(g.D_eq <= di_lo) & (g.t95_info >= t_hi)])
        hi_info_fast.append(g[(g.D_eq >= di_hi) & (g.t95_info <= t_lo)])
    lo_info_slow = pd.concat(lo_info_slow) if lo_info_slow else d.iloc[:0]
    hi_info_fast = pd.concat(hi_info_fast) if hi_info_fast else d.iloc[:0]
    p(f"  within-scale-bin 'low-info yet SLOW'  landscapes: {len(lo_info_slow)}")
    p(f"  within-scale-bin 'high-info yet FAST' landscapes: {len(hi_info_fast)}")
    p("  (both populations non-empty => info does not order arrival time"
      " within a fixed selection scale)\n")

    def _show(g, tag):
        p(f"  [{tag}]  (D_eq, scale, t95, tau, route)")
        for _, r in g.sort_values("scale").head(6).iterrows():
            p(f"    D={r.D_eq:5.2f}  s={r.scale:4.2f}  t95={r.t95_info:7.1f}"
              f"  tau={r.tau_spec:7.1f}  {r.route}")
    _show(lo_info_slow, "low-info yet slow")
    _show(hi_info_fast, "high-info yet fast")

    report = "\n".join(L)
    (SRC / "dispute_report.txt").write_text(report + "\n")
    print(report)

    _figure(df, clocks, lo_info_slow, hi_info_fast, order)


def _figure(df, clocks, lo_slow, hi_fast, order):
    ROUTE_C = {"2-path (additive-like)": "#1b9e77", "1-path (sign)": "#d95f02",
               "0-path (reciprocal/valley)": "#7570b3",
               "0-path (11 not peak)": "#999999"}
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (a) D_eq vs t95 on the reversible clock, coloured by route
    d0 = df[df.regime == clocks[0]]
    a = ax[0, 0]
    for rt in order:
        g = d0[d0.route == rt]
        a.scatter(g.D_eq, g.t95_info, s=22, alpha=.75, c=ROUTE_C[rt], label=rt)
    a.scatter(lo_slow.D_eq, lo_slow.t95_info, s=140, facecolors="none",
              edgecolors="red", linewidths=1.6, label="low-info yet SLOW")
    a.scatter(hi_fast.D_eq, hi_fast.t95_info, s=140, facecolors="none",
              edgecolors="k", linewidths=1.6, label="high-info yet FAST")
    a.set_yscale("log")
    a.set_xlabel(r"$D_{eq}$  (equilibrium information, bits)")
    a.set_ylabel(r"$t_{95}$ arrival time  (gen, log)")
    a.set_title("(a) info vs arrival, weak-mut r=0 clock\n"
                "'small info=>fast' needs UP-slope; raw trend is DOWN & routes split")
    a.legend(fontsize=7, loc="upper right")

    # (b) raw vs scale-controlled partial corr, per clock (t95 and tau)
    a = ax[0, 1]
    labels, raw95, pa95, rawtau, patau = [], [], [], [], []
    for clk in clocks:
        d = df[df.regime == clk]
        labels.append(clk.replace("N30_", "").replace("_", " "))
        raw95.append(_rho(d.D_eq.values, d.t95_info.values)[0])
        pa95.append(partial_spearman(d.D_eq.values, d.t95_info.values,
                                     [d.scale.values]))
        rawtau.append(_rho(d.D_eq.values, d.tau_spec.values)[0])
        patau.append(partial_spearman(d.D_eq.values, d.tau_spec.values,
                                      [d.scale.values]))
    x = np.arange(len(labels)); w = 0.2
    a.bar(x - 1.5 * w, raw95, w, label="t95 raw", color="#d95f02")
    a.bar(x - 0.5 * w, pa95, w, label="t95 | scale", color="#fdae61")
    a.bar(x + 0.5 * w, rawtau, w, label="tau raw", color="#1b9e77")
    a.bar(x + 1.5 * w, patau, w, label="tau | scale", color="#a6d96a")
    a.axhline(0, color="k", lw=.8)
    a.set_xticks(x); a.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
    a.set_ylabel(r"Spearman $\rho(D_{eq},\,$time$)$")
    a.set_title("(b) rule predicts POSITIVE bars.\n"
                "raw is negative (opposite), and shrinks once scale removed")
    a.legend(fontsize=7)

    # (c) route means: D_eq vs tau (the reciprocal reversal)
    a = ax[1, 0]
    for rt in order:
        g = df[df.route == rt]
        if len(g) == 0:
            continue
        a.errorbar(g.D_eq.mean(), g.tau_spec.mean(),
                   xerr=g.D_eq.sem(), yerr=g.tau_spec.sem(),
                   fmt="o", ms=11, c=ROUTE_C[rt], capsize=4, label=rt)
    a.set_xlabel(r"mean $D_{eq}$"); a.set_ylabel(r"mean $\tau_{spec}$")
    a.set_title("(c) across routes info & speed both ride selection scale:\n"
                "additive = highest-info & FASTEST; recip-valley = lowest-info & SLOWEST")
    a.legend(fontsize=7)

    # (d) within-route partial corr(D_eq, tau | scale)
    a = ax[1, 1]
    rts, vals = [], []
    for rt in order:
        d = df[df.route == rt]
        if len(d) < 8:
            continue
        rts.append(rt.split(" ")[0])
        vals.append(partial_spearman(d.D_eq.values, d.tau_spec.values,
                                     [d.scale.values]))
    a.bar(range(len(rts)), vals, color=[ROUTE_C[o] for o in order[:len(rts)]])
    a.axhline(0, color="k", lw=.8)
    a.set_xticks(range(len(rts))); a.set_xticklabels(rts, fontsize=8)
    a.set_ylabel(r"$\rho(D_{eq},\tau\,|\,$scale$)$ within route")
    a.set_title("(d) within a fixed route, info->speed sign is\n"
                "inconsistent / near-zero, not a reliable positive rule")

    fig.suptitle("Refuting  'smaller info => faster to equilibrium'   "
                 "(N=30, 760 landscapes x 4 clocks)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = SRC / "fig_dispute.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
