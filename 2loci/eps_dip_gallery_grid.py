#!/usr/bin/env python3
r"""
Combine the saved eps12-sweep gallery CSVs into a single 3x3 grid figure in the
dip_curves.py style: per panel, the KL points for each N, a parabola fit to each,
and the predicted dip line eps12* = -2/3 (s1+s2).

Reads plots/eps_sweep_prediction_dip/gallery_index.txt + the per-config CSVs.
No chain recompute -- pure replot.

Run:
  python3 eps_dip_gallery_grid.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
GAL_DIR = SCRIPT_DIR / "plots" / "eps_sweep_prediction_dip"

EPS_SCALE = 1.0e-3
EPS_LABEL = r"$\epsilon_{12}\;(\times 10^{-3})$"
KL_LABEL = r"$D_{\mathrm{KL}}(\psi^*\,\|\,\varphi^*)$ [bits]"

DROP_IDX = {10}   # leave one out so 9 remain -> 3x3

LINE = re.compile(
    r"\[(?P<idx>\d+)\]\s+N=\((?P<N>[^)]+)\)\s+r=(?P<r>\S+)\s+u=v=(?P<u>\S+)\s+"
    r"s1=(?P<s1>\S+)\s+s2=(?P<s2>\S+)\s+eps\*=(?P<eps>\S+)\s+->\s+(?P<png>\S+)"
)


def parse_index():
    entries = []
    for line in (GAL_DIR / "gallery_index.txt").read_text().splitlines():
        m = LINE.match(line.strip())
        if not m:
            continue
        d = m.groupdict()
        entries.append({
            "idx": int(d["idx"]),
            "N": [int(x) for x in d["N"].split(",")],
            "r": float(d["r"]), "u": float(d["u"]),
            "s1": float(d["s1"]), "s2": float(d["s2"]),
            "eps_star": float(d["eps"]),
            "csv": GAL_DIR / d["png"].replace(".png", ".csv"),
        })
    return entries


def load_csv(path: Path):
    arr = np.genfromtxt(path, delimiter=",", names=True)
    return arr["N"].astype(int), arr["eps12"], arr["KL_psi_star_vs_phi_star_bits"]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9)
    except ImportError:
        pass
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "figure.facecolor": "white", "axes.titlesize": 9,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "legend.fontsize": 6.5})

    entries = [e for e in parse_index() if e["idx"] not in DROP_IDX]
    entries.sort(key=lambda e: e["idx"])

    colors = {0: "#3b528b", 1: "#e07b39", 2: "#3a9e54"}  # up to 3 N per panel

    fig, axes = plt.subplots(3, 3, figsize=(15.5, 11.5))
    axes = axes.ravel()

    for ax, e in zip(axes, entries):
        Ns, eps_all, kl_all = load_csv(e["csv"])
        es = EPS_SCALE
        for k, Nval in enumerate(sorted(set(Ns))):
            mask = Ns == Nval
            eps = eps_all[mask]
            kl = kl_all[mask]
            order = np.argsort(eps)
            eps, kl = eps[order], kl[order]
            col = colors.get(k, "0.3")
            ax.plot(eps / es, kl, "o", ms=4, color=col, label=rf"$N={Nval}$")
            # parabola fit + vertex
            a, b, c = np.polyfit(eps, kl, 2)
            xx = np.linspace(eps.min(), eps.max(), 200)
            ax.plot(xx / es, a * xx**2 + b * xx + c, "-", color=col, lw=1.3, alpha=0.7)
            if a != 0:
                vtx = -b / (2.0 * a)
                ax.axvline(vtx / es, color=col, ls=":", lw=1.2, alpha=0.7)

        ax.axvline(e["eps_star"] / es, color="0.15", ls="--", lw=1.5,
                   label=rf"pred $-\frac{{2}}{{3}}(s_1+s_2)$={e['eps_star']/es:+.2f}")
        ax.set_title(rf"$s_1$={e['s1']:g}, $s_2$={e['s2']:g}, $r$={e['r']:g}, "
                     rf"$u=v$={e['u']:g}")
        ax.set_xlabel(EPS_LABEL)
        ax.set_ylabel(KL_LABEL)
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10], prune="both"))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.legend(loc="upper center", ncol=2, columnspacing=0.8,
                  handlelength=1.3, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.margins(x=0.02)

    for ax in axes[len(entries):]:
        ax.set_visible(False)

    fig.suptitle(r"KL$(\psi^*\|\varphi^*)$ epistasis sweeps: dip vs prediction "
                 r"$\varepsilon_{12}^*=-\frac{2}{3}(s_1+s_2)$  "
                 r"(dotted = per-$N$ parabola minima)", y=0.998, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985), w_pad=2.0, h_pad=2.5)
    out = GAL_DIR / "dip_grid.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
