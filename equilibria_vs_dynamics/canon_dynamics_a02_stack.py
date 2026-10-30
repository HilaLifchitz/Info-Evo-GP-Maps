"""Stacked two-row dynamics figure at a=0.2 for the iso-peak strategies A-F.

Top row    (the "Part 1" trio): information gain D(t), free fitness F(t),
                                mean log-fitness log W-bar(t).
Bottom row: cost C(t), efficiency eta(t), and -- in place of a third curve --
            the two "clocks" of equilibration, stacked in one cell:
              (i)  relaxation to equilibrium: D_KL(psi(t) || psi*) vs t
                   (a running-down curve: how far the population still is from
                   its own Boltzmann equilibrium), and
              (ii) mixing time tau = -1/ln(lambda_2): a single CONSTANT per
                   strategy (set by the spectral gap of the transition matrix),
                   shown as a bar chart -- it is not a time course.

Most curves are read from canon_dynamics.py's output
    results/canon_dynamics_relabel/curves.npz
(run canon_dynamics.py first if that file is missing).  The two clocks need the
full distribution psi(t), psi* and the transition matrix, which are not stored
there, so they are recomputed once from the chain and cached in
    results/canon_dynamics_relabel/clocks_a02.npz
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from adaptation_per_a import CASE_DEFS, CASE_COLORS, CASE_ORDER
from node_jump_analysis import build_chain, Landscape, evolve_many, selection_P
from simplex import stationary_fast
from compare_two_cases import kl_series
from info_speed_experiment import spectral_gap

SRC = Path(__file__).resolve().parent / "results" / "canon_dynamics_relabel"
# display relabel: swap E<->F letters (F = reciprocal sign, E = neutral).
# Data keys / colours stay internal (internal E = reciprocal, internal F = neutral).
DISP = {"A": "A", "B": "B", "C": "C", "D": "D", "E": "F", "F": "E"}
# iterate so the legend reads in display order A,B,C,D,E,F
CASES = ["A", "B", "C", "D", "F", "E"]
LABEL = {s: f"{DISP[s]}: {CASE_DEFS[s][0]}" for s in CASES}
A = 0.2
TMAX = 1500
N, U, V, R = 30, 5e-3, 5e-3, 0.0

# (metric key, y-axis label) for each of the five ordinary panels
TOP = [
    ("info", "information gain $D(t)$\n[bits]"),
    ("F",    "information–fitness score $F(t)$\n[nats]"),
    ("mlf",  "mean log-fitness $\\log\\bar W(t)$\n[nats]"),
]
BOTTOM = [
    ("cost",  "cost $C(t)$\n[bits/gen]"),
    ("eta",   r"efficiency $\eta(t)$"),
]


def compute_clocks(times):
    """Relaxation curve D_KL(psi(t)||psi*) [bits] and mixing time tau [gen].

    tau = -1/ln(lambda_2), where lambda_2 is the second-largest |eigenvalue| of
    the transition matrix -- the (constant) asymptotic relaxation timescale.
    Cached to clocks_a02.npz keyed by strategy.
    """
    cache = SRC / "clocks_a02.npz"
    if cache.exists():
        d = np.load(cache)
        if int(d["times"].max()) >= int(times.max()) and len(d["times"]) == len(times):
            relax = {s: d[f"relax_{s}"] for s in CASES}
            tau = {s: float(d[f"tau_{s}"]) for s in CASES}
            return relax, tau

    chain = build_chain(N, U, V, R, 1)
    print(f"chain S={len(chain.phi)}", flush=True)
    relax, tau = {}, {}
    Psi0 = chain.phi.reshape(-1, 1)
    for s in CASES:
        s1, s2, eps = CASE_DEFS[s][1](A)
        L = Landscape(f"{s}: {CASE_DEFS[s][0]}", s, s1=s1, s2=s2, eps12=eps, blurb="")
        P = selection_P(chain, L, 1)
        psi_star = stationary_fast(P)
        snaps = evolve_many(P, Psi0, times)
        relax[s] = kl_series(snaps, psi_star, 0)          # D_KL(psi(t)||psi*) [bits]
        lam2 = spectral_gap(P)
        tau[s] = float(-1.0 / np.log(lam2)) if 0 < lam2 < 1 else np.inf
        print(f"  {s}: tau={tau[s]:.1f} gen (lambda2={lam2:.5f})", flush=True)

    np.savez_compressed(cache, times=times,
                        **{f"relax_{s}": relax[s] for s in CASES},
                        **{f"tau_{s}": tau[s] for s in CASES})
    return relax, tau


def main():
    d = np.load(SRC / "curves.npz")
    t = d["times"].astype(float)
    times = d["times"].astype(int)

    def curve(metric, s):
        return d[f"{A}_{s}_{metric}"]

    relax, tau = compute_clocks(times)

    FIGW, FIGH = 16, 5.6
    fig = plt.figure(figsize=(FIGW, FIGH))
    outer = fig.add_gridspec(2, 3, left=0.055, right=0.995, top=0.73,
                             bottom=0.105, hspace=0.55, wspace=0.30)

    # ---- five ordinary panels ------------------------------------------------
    axes = {}
    for col, (key, ylab) in enumerate(TOP):
        axes[(0, col)] = fig.add_subplot(outer[0, col])
    for col, (key, ylab) in enumerate(BOTTOM):
        axes[(1, col)] = fig.add_subplot(outer[1, col])

    for (row, col), a in axes.items():
        key, ylab = (TOP if row == 0 else BOTTOM)[col]
        for s in CASES:
            a.plot(t, curve(key, s), "-", lw=1.8, color=CASE_COLORS[s], label=LABEL[s])
        if key == "eta":
            a.axhline(1.0, color="0.5", ls=":", lw=0.9)
        a.set_xscale("log")
        a.set_xlim(1, TMAX)
        a.set_xlabel("generation $t$")
        a.set_ylabel(ylab)
        a.grid(True, alpha=0.3)
    # figure-level legend as a horizontal strip above the panels
    handles, labels = axes[(0, 0)].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=14,
                     frameon=False, bbox_to_anchor=(0.5, 0.885),
                     columnspacing=1.4, handlelength=1.8)
    for ln in leg.get_lines():
        ln.set_linewidth(4.5)

    # ---- circular magnifier on the free-fitness panel -----------------------
    # zoom the converged plateau to show the F-ordering is locked (no crossings)
    import math
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D

    ax_F = axes[(0, 1)]
    ax_F.set_ylim(-0.34, 0.35)                     # fix so the lens region lands right
    posF = ax_F.get_position()
    px0, py0 = posF.x0 * FIGW, posF.y0 * FIGH      # panel corners in inches
    px1, py1 = posF.x1 * FIGW, posF.y1 * FIGH

    sq = 0.95                                       # lens diameter [inches] -> round
    lx0, ly0 = px0 + 0.04, py1 - sq - 0.02          # lens box: top-left (empty region)
    axins = fig.add_axes([lx0 / FIGW, ly0 / FIGH, sq / FIGW, sq / FIGH])
    for s in CASES:
        axins.plot(t, curve("F", s), "-", lw=1.6, color=CASE_COLORS[s])
    axins.set_xscale("log")
    axins.set_xlim(700, TMAX)
    axins.set_ylim(0.298, 0.327)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.minorticks_off()
    for sp in axins.spines.values():
        sp.set_visible(False)
    # clip the curves to a circle -> a round lens with a white face
    bg = Circle((0.5, 0.5), 0.5, transform=axins.transAxes, facecolor="white",
                edgecolor="none", zorder=0)
    axins.add_patch(bg)
    for ln in axins.get_lines():
        ln.set_clip_path(bg)
    axins.patch.set_visible(False)
    axins.add_patch(Circle((0.5, 0.5), 0.5, transform=axins.transAxes,
                           fill=False, ec="0.25", lw=1.6, zorder=10))
    axins.set_title("ordering locked", fontsize=8, pad=2)

    # source circle over the converged bundle + connector (round: inch coords)
    lens_cx, lens_cy, lens_r = lx0 + sq / 2, ly0 + sq / 2, sq / 2
    src_cx = px0 + 0.90 * (px1 - px0)
    src_cy = py0 + 0.945 * (py1 - py0)
    src_r = 0.15
    fig.add_artist(Circle((src_cx, src_cy), src_r, transform=fig.dpi_scale_trans,
                          fill=False, ec="0.25", lw=1.3, zorder=9))
    dx, dy = lens_cx - src_cx, lens_cy - src_cy
    dist = math.hypot(dx, dy)
    ux, uy = dx / dist, dy / dist
    fig.add_artist(Line2D([src_cx + ux * src_r, lens_cx - ux * lens_r],
                          [src_cy + uy * src_r, lens_cy - uy * lens_r],
                          transform=fig.dpi_scale_trans, color="0.55", lw=1.0,
                          zorder=8))

    # ---- bottom-right cell: relaxation to equilibrium -----------------------
    # D_KL(psi(t)||psi*) [bits]: how far the population still is from its own
    # steady state; falls to ~0 once it has settled.
    ax_eq = fig.add_subplot(outer[1, 2])
    for s in CASES:
        ax_eq.plot(t, np.maximum(relax[s], 1e-12), "-", lw=1.8,
                   color=CASE_COLORS[s], label=LABEL[s])
    ax_eq.set_xscale("log")
    ax_eq.set_yscale("log")
    ax_eq.set_xlim(1, TMAX)
    ax_eq.set_ylim(1e-4, 40)
    ax_eq.set_xlabel("generation $t$")
    ax_eq.set_ylabel("$D_{\\mathrm{KL}}(\\psi(t)\\,\\|\\,\\psi^\\ast)$\n[bits]")
    ax_eq.set_title("relaxation to equilibrium", fontsize=10)
    ax_eq.grid(True, which="both", alpha=0.3)

    # ---- panel letters a)-f) in reading order -------------------------------
    panel_axes = [axes[(0, 0)], axes[(0, 1)], axes[(0, 2)],
                  axes[(1, 0)], axes[(1, 1)], ax_eq]
    for ltr, a in zip("abcdef", panel_axes):
        a.text(-0.12, 1.12, f"{ltr})", transform=a.transAxes,
               fontsize=16, fontweight="bold", va="top", ha="left")

    # nudge panel a)'s y-axis title left so the bold "a)" doesn't sit on it
    axes[(0, 0)].yaxis.labelpad = 12

    out = SRC / "fig_canon_dyn_a02_stacked.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
