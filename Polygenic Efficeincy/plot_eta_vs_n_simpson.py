#!/usr/bin/env python3
"""Figures for η_maint vs n at fixed expected Simpson c."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.ticker import LogLocator, NullFormatter

HERE = Path(__file__).resolve().parent
DATA = HERE / "eta_vs_n_simpson_large.json"
OUT = HERE / "figures"

CURVE_STYLE = {
    "canvas default": dict(color="#1d4ed8", lw=2.4, zorder=3),
    "poster 16/4": dict(color="#b91c1c", lw=2.4, zorder=3),
    "high c, modest N": dict(color="#ea580c", lw=1.5, zorder=2),
    "weak sel": dict(color="#6b7280", lw=1.5, ls="--", zorder=2),
    "strong sel": dict(color="#7c3aed", lw=1.5, zorder=2),
    "small N": dict(color="#0f766e", lw=1.5, ls=":", zorder=2),
    "mid c": dict(color="#db2777", lw=1.5, zorder=2),
    "low c": dict(color="#365314", lw=1.5, zorder=2),
}

LABEL = {
    "canvas default": r"default  $c{=}0.3$, $N{=}50$, $V_s{=}0.0625$",
    "poster 16/4": r"poster  $c{=}0.75$, $N{=}100$, $V_s{=}0.0625$",
    "high c, modest N": r"$c{=}0.75$, $N{=}20$",
    "weak sel": r"weak sel.  $c{=}0.3$, $V_s{=}1$",
    "strong sel": r"strong sel.  $c{=}0.3$, $V_s{=}0.02$",
    "small N": r"$c{=}0.3$, $N{=}10$",
    "mid c": r"$c{=}0.5$, $N{=}50$",
    "low c": r"$c{=}0.26$, $N{=}50$, $V_s{=}0.25$",
}


def _setup() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def _xy(curve: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    n = np.array([p["n"] for p in curve["points"]], dtype=float)
    y = np.array([p[key] for p in curve["points"]], dtype=float)
    return n, y


def _log_n_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"loci $n$")
    ticks = [4, 8, 16, 32, 64, 128, 256, 512, 1000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_locator(LogLocator(base=2, subs=np.arange(1.0, 2.0, 0.25)))
    ax.xaxis.set_minor_formatter(NullFormatter())


def fig_eta_curves(payload: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    ax, axr = axes

    for curve in payload["curves"]:
        sty = CURVE_STYLE[curve["name"]]
        n, eta = _xy(curve, "eta")
        ax.plot(n, eta, marker="o", ms=3.5, label=LABEL[curve["name"]], **sty)
        eta16 = eta[n == 16][0]
        axr.plot(n, eta / eta16, marker="o", ms=3.5, label=LABEL[curve["name"]], **sty)

    ax.axvline(16, color="0.7", lw=0.8, zorder=0)
    axr.axvline(16, color="0.7", lw=0.8, zorder=0)
    axr.axhline(1.0, color="0.55", lw=0.8, zorder=0)
    _log_n_axis(ax)
    _log_n_axis(axr)
    ax.set_ylabel(r"$\eta_{\mathrm{maint}}=D/(2N\langle C\rangle)$")
    axr.set_ylabel(r"$\eta(n)\,/\,\eta(16)$")
    ax.set_title(r"Maintenance efficiency at fixed $\mathbb{E}[S]=c$")
    axr.set_title("Gain relative to $n=16$")
    ax.legend(loc="upper left", framealpha=0.92, borderpad=0.4)
    fig.suptitle(
        r"Boltzmann $\eta_{\mathrm{maint}}$ vs polygenicity, Hamming-layer Dirichlet, $z^*=0.5$",
        fontsize=12,
        y=1.03,
    )
    path = OUT / "eta_vs_n_simpson_curves.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def fig_D_C(payload: dict) -> Path:
    names = ("canvas default", "poster 16/4", "strong sel", "weak sel")
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    for curve in payload["curves"]:
        if curve["name"] not in names:
            continue
        sty = CURVE_STYLE[curve["name"]]
        n, D = _xy(curve, "D")
        _, C = _xy(curve, "C")
        axes[0].plot(n, D, marker="o", ms=3.5, label=LABEL[curve["name"]], **sty)
        axes[1].plot(n, C, marker="o", ms=3.5, label=LABEL[curve["name"]], **sty)

    for ax, ylab, title in zip(
        axes,
        (r"$D_{\mathrm{KL}}(\rho^*\|\rho_\varphi)$  (bits)", r"$\langle C\rangle_*$  (bits)"),
        ("Information maintained", "Maintenance cost"),
    ):
        _log_n_axis(ax)
        ax.set_yscale("log")
        ax.set_ylabel(ylab)
        ax.set_title(title)
    axes[0].legend(loc="lower left", framealpha=0.92)
    fig.suptitle(
        r"Both $D$ and $\langle C\rangle$ fall with $n$; $\langle C\rangle$ falls faster",
        fontsize=12,
        y=1.03,
    )
    path = OUT / "eta_vs_n_simpson_D_C.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def fig_grid(payload: dict) -> Path:
    cells = payload["grid"]
    cs = sorted({g["c"] for g in cells})
    Ns = sorted({g["N"] for g in cells})
    cmap = plt.get_cmap("YlOrRd")

    def mat(key: str) -> np.ndarray:
        M = np.full((len(Ns), len(cs)), np.nan)
        for g in cells:
            i = Ns.index(g["N"])
            j = cs.index(g["c"])
            M[i, j] = g[key]
        return M

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), constrained_layout=True)
    specs = (
        (axes[0], "ratio_1000_16", r"$\eta(1000)\,/\,\eta(16)$", (1.0, 6.0)),
        (axes[1], "ratio_1000_256", r"$\eta(1000)\,/\,\eta(256)$", (0.9, 1.7)),
    )
    ims = []
    for ax, key, title, vlim in specs:
        M = mat(key)
        im = ax.imshow(
            M,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=Normalize(vmin=vlim[0], vmax=vlim[1]),
        )
        ims.append(im)
        ax.set_xticks(range(len(cs)))
        ax.set_xticklabels([f"{c:g}" for c in cs])
        ax.set_yticks(range(len(Ns)))
        ax.set_yticklabels([f"{N:g}" for N in Ns])
        ax.set_xlabel(r"Simpson $c$")
        ax.set_ylabel(r"population $N$")
        ax.set_title(title)
        ax.grid(False)
        for i in range(len(Ns)):
            for j in range(len(cs)):
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="0.1" if M[i, j] < 0.55 * vlim[1] + 0.45 * vlim[0] else "1",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(
        r"Does the $n$-gain survive to $n=1000$?  ($V_s=0.0625$, $z^*=0.5$, ESS$\geq 8$)",
        fontsize=12,
        y=1.04,
    )
    path = OUT / "eta_vs_n_simpson_grid.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def fig_grid_lines(payload: dict) -> Path:
    cs = sorted({g["c"] for g in payload["grid"]})
    Ns = sorted({g["N"] for g in payload["grid"]})
    cmap = plt.get_cmap("viridis")
    n_colors = {N: cmap(i / max(len(Ns) - 1, 1)) for i, N in enumerate(Ns)}

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4), sharex=True, sharey=False)
    axes = axes.ravel()
    for ax, c in zip(axes, cs):
        for g in payload["grid"]:
            if g["c"] != c:
                continue
            n = np.array([p["n"] for p in g["points"]], dtype=float)
            eta = np.array([p["eta"] for p in g["points"]], dtype=float)
            ax.plot(
                n,
                eta,
                marker="o",
                ms=4,
                color=n_colors[g["N"]],
                lw=1.7,
                label=rf"$N={g['N']:g}$",
            )
        ax.set_title(rf"$c={c:g}$")
        _log_n_axis(ax)
        ax.set_yscale("log")
    for ax in axes[len(cs) :]:
        ax.set_visible(False)
    axes[0].set_ylabel(r"$\eta_{\mathrm{maint}}$")
    axes[3].set_ylabel(r"$\eta_{\mathrm{maint}}$")
    axes[0].legend(loc="upper left", framealpha=0.92, fontsize=8)
    fig.suptitle(
        r"$\eta$ vs $n$ at fixed $c$, $V_s=0.0625$, $z^*=0.5$  (log–log)",
        fontsize=12,
    )
    fig.tight_layout()
    path = OUT / "eta_vs_n_simpson_grid_lines.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def main() -> None:
    _setup()
    OUT.mkdir(exist_ok=True)
    payload = json.loads(DATA.read_text())
    paths = [
        fig_eta_curves(payload),
        fig_D_C(payload),
        fig_grid(payload),
        fig_grid_lines(payload),
    ]
    for p in paths:
        print(p)
        print(p.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
