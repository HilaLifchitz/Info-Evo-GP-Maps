#!/usr/bin/env python3
"""Iso-peak (ε, α) surface: equilibrium info, F*, cost, maintenance efficiency.

Parameterization (lf[00]=0, lf[11]=2a fixed):
  asymmetry  α = s1 - s2 = lf[10] - lf[01]   ∈ [α_min, 2a]
  epistasis  ε = eps12                        ∈ [-4a, 4a]
  s1 = (2a - ε + α)/2
  s2 = (2a - ε - α)/2

Maintenance efficiency:
  η_maint = D_eq / C* ,   C* = <C(x)>_ψ*

Uses diffusion / cond-alpha Boltzmann equilibrium (r-independent).
"""
from __future__ import annotations

from math import log
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd

from node_jump_analysis import build_chain
from adaptation_per_a import CASE_DEFS, CASE_COLORS, CASE_ORDER, cost_per_state
from compare_two_cases import boltz_reference
from simplex import (
    build_sigma, fitness_exp, log_mean_fitness_genotype_simplex,
    mutation_matrix_asymmetric,
)
from f2b_reference import kl_divergence_nats
from route_partition import draw_route_partition

OUT = Path(__file__).resolve().parent / "results" / "eps_asym_surface"
LN2 = log(2.0)

N = 30
N_EPS, N_ALPHA = 81, 45
# tiny bit of negative asymmetry (a-units), then up to 2
ALPHA_OA_MIN, ALPHA_OA_MAX = -0.15, 2.0
EPS_OA_MIN, EPS_OA_MAX = -4.0, 4.0

# parameter powers / decade-ish grids
A_VALS = [0.02, 0.05, 0.10, 0.20, 0.40]  # a=0.02 → Na=0.6 < 1
U_VALS = [1e-3, 5e-3, 2e-2]
R_VALS = [0.0, 0.1, 0.5]  # diffusion ψ* ignores r; kept for bookkeeping / later chain


def s1_s2(a: float, eps: float, alpha: float) -> tuple[float, float]:
    return (2.0 * a - eps + alpha) / 2.0, (2.0 * a - eps - alpha) / 2.0


def landmarks(a: float) -> list[dict]:
    out = []
    for slug in CASE_ORDER:
        name, fn = CASE_DEFS[slug]
        s1, s2, eps = fn(a)
        alpha = abs(s1 - s2)
        out.append(dict(
            case=slug, name=name,
            eps=eps, alpha=alpha,
            eps_over_a=eps / a, alpha_over_a=alpha / a,
        ))
    return out


def sweep_one(a: float, u: float, r: float,
              chain=None, fracs=None, M=None, phi_b=None) -> pd.DataFrame:
    """One (ε,α) surface at fixed (a,u,r). Reuse chain/phi if provided."""
    if chain is None:
        chain = build_chain(N, u, u, r, 2)
        fracs = chain.fracs
        M = mutation_matrix_asymmetric(np.array([u, u]), np.array([u, u]))
        phi_b, _ = boltz_reference(fracs, fitness_exp(np.zeros((2, 2))), M, N)

    eps_oa = np.linspace(EPS_OA_MIN, EPS_OA_MAX, N_EPS)
    alpha_oa = np.linspace(ALPHA_OA_MIN, ALPHA_OA_MAX, N_ALPHA)
    rows = []
    for e_oa in eps_oa:
        for al_oa in alpha_oa:
            eps = e_oa * a
            alpha = al_oa * a
            s1, s2 = s1_s2(a, eps, alpha)
            omega = fitness_exp(build_sigma(s1, s2, eps))
            logmf = log_mean_fitness_genotype_simplex(fracs, omega)
            psi, logZ = boltz_reference(fracs, omega, M, N)
            dkl_nats = kl_divergence_nats(psi, phi_b)
            D = dkl_nats / LN2
            lwb = float(psi @ logmf)
            Fstar = lwb - dkl_nats / N
            Cstar = float(psi @ cost_per_state(fracs, omega))
            eta_m = (D / Cstar) if Cstar > 1e-15 else float("nan")
            logfit = np.array([0.0, s2, s1, s1 + s2 + eps], dtype=float)
            mlf = float(psi @ (fracs @ logfit))  # ⟨ℓf⟩_ψ*
            rows.append(dict(
                N=N, u=u, v=u, r=r, a=a,
                eps=eps, alpha=alpha,
                eps_over_a=e_oa, alpha_over_a=al_oa,
                s1=s1, s2=s2,
                D_eq=D, Fstar=Fstar, Cstar=Cstar, eta_maint=eta_m, mlf=mlf,
            ))
    return pd.DataFrame(rows)


def sweep_grid(a_vals=A_VALS, u_vals=U_VALS, r_vals=R_VALS) -> pd.DataFrame:
    """Sweep (a,u,r). Diffusion Boltzmann is independent of r — compute r=0
    only once per (a,u) and replicate rows for other r (flagged)."""
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    parts = []
    # unique (u,) chains: fracs/phi independent of a and (for diffusion) of r
    cache = {}
    n_jobs = len(a_vals) * len(u_vals)
    k = 0
    for u in u_vals:
        if u not in cache:
            print(f"building chain u=v={u:g} r=0 ...", flush=True)
            chain = build_chain(N, u, u, 0.0, 2)
            fracs = chain.fracs
            M = mutation_matrix_asymmetric(np.array([u, u]), np.array([u, u]))
            phi_b, _ = boltz_reference(fracs, fitness_exp(np.zeros((2, 2))), M, N)
            cache[u] = (chain, fracs, M, phi_b)
        chain, fracs, M, phi_b = cache[u]
        for a in a_vals:
            k += 1
            ta = time.perf_counter()
            df0 = sweep_one(a, u, 0.0, chain=chain, fracs=fracs, M=M, phi_b=phi_b)
            print(f"  [{k}/{n_jobs}] a={a:g} u={u:g}  "
                  f"({time.perf_counter()-ta:.1f}s, total {time.perf_counter()-t0:.1f}s)",
                  flush=True)
            for r in r_vals:
                dfr = df0.copy()
                dfr["r"] = r
                dfr["r_note"] = ("diffusion_r_independent_copy" if r != 0.0
                                 else "computed")
                parts.append(dfr)

    df = pd.concat(parts, ignore_index=True)
    csv = OUT / "surface_grid_auv.csv"
    df.to_csv(csv, index=False)
    print("wrote", csv, f"rows={len(df)}", flush=True)
    return df


def _eta_panel(ax, df_one, a, u, r, vmin, vmax, show_ylab, show_xlab, marks):
    piv = df_one.pivot_table(index="alpha_over_a", columns="eps_over_a",
                             values="eta_maint")
    alp_ax = piv.index.values
    eps_ax = piv.columns.values
    Z = piv.values
    extent = [eps_ax.min(), eps_ax.max(), alp_ax.min(), alp_ax.max()]
    im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent,
                   cmap="plasma", interpolation="bilinear",
                   vmin=vmin, vmax=vmax, zorder=2)
    draw_route_partition(ax, show_labels=True,
                         alpha_min=float(alp_ax.min()),
                         alpha_max=float(alp_ax.max()))
    EE, AA = np.meshgrid(eps_ax, alp_ax)
    zmin, zmax = np.nanmin(Z), np.nanmax(Z)
    if np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin:
        levels = np.linspace(zmin, zmax, 7)[1:-1]
        cs = ax.contour(EE, AA, Z, levels=levels, colors="k",
                        linewidths=0.45, alpha=0.5)
        ax.clabel(cs, fmt="%.3g", fontsize=6, inline=True, inline_spacing=2)
    for m in marks:
        if ALPHA_OA_MIN <= m["alpha_over_a"] <= ALPHA_OA_MAX:
            ax.scatter(m["eps_over_a"], m["alpha_over_a"], s=28,
                       c=CASE_COLORS[m["case"]], edgecolors="white",
                       linewidths=0.5, zorder=5)
            ax.annotate(m["case"], (m["eps_over_a"], m["alpha_over_a"]),
                        textcoords="offset points", xytext=(3, 2),
                        fontsize=7, fontweight="bold",
                        color=CASE_COLORS[m["case"]])
    ax.axhline(0, color="w", lw=0.5, alpha=0.6)
    ax.axvline(0, color="w", lw=0.5, alpha=0.6)
    ax.set_xlim(EPS_OA_MIN, EPS_OA_MAX)
    ax.set_ylim(ALPHA_OA_MIN, ALPHA_OA_MAX)
    ax.set_title(rf"$a={a:g}$, $u={u:g}$" + (rf", $r={r:g}$" if r else ""),
                 fontsize=9)
    if show_xlab:
        ax.set_xlabel(r"$\varepsilon/a$", fontsize=9)
    if show_ylab:
        ax.set_ylabel(r"$\alpha/a$", fontsize=9)
    return im


def plot_eta_gallery(df: pd.DataFrame) -> Path:
    """Efficiency-only gallery: rows = a, cols = u, one figure per r."""
    # shared color scale over all computed (use r=0 block)
    sub0 = df[np.isclose(df.r, 0.0)]
    vmin = float(sub0.eta_maint.quantile(0.02))
    vmax = float(sub0.eta_maint.quantile(0.98))

    outs = []
    for r in R_VALS:
        fig, axs = plt.subplots(
            len(A_VALS), len(U_VALS),
            figsize=(3.6 * len(U_VALS), 2.8 * len(A_VALS)),
            sharex=True, sharey=True, constrained_layout=True,
        )
        if len(A_VALS) == 1:
            axs = np.array([axs])
        im = None
        for i, a in enumerate(A_VALS):
            marks = landmarks(a)
            for j, u in enumerate(U_VALS):
                ax = axs[i, j]
                d1 = df[np.isclose(df.a, a) & np.isclose(df.u, u)
                        & np.isclose(df.r, r)]
                im = _eta_panel(ax, d1, a, u, r, vmin, vmax,
                                show_ylab=(j == 0), show_xlab=(i == len(A_VALS) - 1),
                                marks=marks)
        fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02,
                     label=r"maintenance efficiency  $D_{\mathrm{eq}}/C^*$")
        note = ""
        if r != 0.0:
            note = ("\n[diffusion Boltzmann $\\psi^*$ is independent of $r$ — "
                    "panel matches $r=0$; kept for layout / later chain work]")
        fig.suptitle(
            rf"Maintenance efficiency $D_{{\mathrm{{eq}}}}/C^*$  on "
            rf"$(\varepsilon/a,\alpha/a)$  ·  $N={N}$, $r={r:g}$"
            + note,
            fontsize=12, fontweight="bold",
        )
        out = OUT / f"fig_eta_maint_gallery_r{r:g}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("wrote", out)
        outs.append(out)
        plt.close(fig)

    # also a compact single figure at r=0 (main deliverable)
    return outs[0]


def plot_single_full(df_one: pd.DataFrame, a: float, u: float, r: float) -> Path:
    """4-metric figure for one clock (optional / debug)."""
    marks = landmarks(a)
    metrics = [
        ("D_eq", r"$D_{\mathrm{eq}}$ [bits]", "viridis"),
        ("Fstar", r"$F^*$ [nats]", "magma"),
        ("mlf", r"mean log-fitness  $\langle \ell f\rangle_{\psi^*}$", "YlOrBr"),
        ("Cstar", r"$C^*$ [bits/gen]", "cividis"),
        ("eta_maint", r"$D_{\mathrm{eq}}/C^*$", "plasma"),
    ]
    metrics = [(c, t, m) for c, t, m in metrics if c in df_one.columns]
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)
    for ax, (col, title, cmap) in zip(list(axs.ravel())[:len(metrics)], metrics):
        piv = df_one.pivot_table(index="alpha_over_a", columns="eps_over_a",
                                 values=col)
        alp_ax, eps_ax, Z = piv.index.values, piv.columns.values, piv.values
        extent = [eps_ax.min(), eps_ax.max(), alp_ax.min(), alp_ax.max()]
        im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent,
                       cmap=cmap, interpolation="bilinear", zorder=2)
        draw_route_partition(ax, show_labels=(ax is axs.ravel()[0]),
                             alpha_min=float(alp_ax.min()),
                             alpha_max=float(alp_ax.max()))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        EE, AA = np.meshgrid(eps_ax, alp_ax)
        levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 9)[1:-1]
        cs = ax.contour(EE, AA, Z, levels=levels, colors="k",
                        linewidths=0.55, alpha=0.55)
        ax.clabel(cs, fmt="%.3g", fontsize=7)
        for m in marks:
            ax.scatter(m["eps_over_a"], m["alpha_over_a"], s=70,
                       c=CASE_COLORS[m["case"]], edgecolors="white",
                       linewidths=0.9, zorder=5)
            ax.annotate(m["case"], (m["eps_over_a"], m["alpha_over_a"]),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=9, fontweight="bold",
                        color=CASE_COLORS[m["case"]])
        ax.axhline(0, color="w", lw=0.5, alpha=0.6)
        ax.set_xlabel(r"epistasis  $\varepsilon/a$")
        ax.set_ylabel(r"asymmetry  $\alpha/a$")
        ax.set_title(title)
        ax.set_xlim(EPS_OA_MIN, EPS_OA_MAX)
        ax.set_ylim(ALPHA_OA_MIN, ALPHA_OA_MAX)
    for ax in axs.ravel()[len(metrics):]:
        ax.axis("off")
    fig.suptitle(
        rf"Iso-peak surface · $N={N}$, $u=v={u:g}$, $r={r:g}$, $a={a:g}$",
        fontsize=12, fontweight="bold",
    )
    out = OUT / f"fig_eps_asym_surface_a{a:g}_u{u:g}_r{r:g}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    plt.close(fig)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true",
                    help="replot from existing CSV with route overlays")
    args = ap.parse_args()
    if args.plot_only:
        df = pd.read_csv(OUT / "surface_grid_auv.csv")
    else:
        df = sweep_grid()
    plot_eta_gallery(df)
    if "mlf" in df.columns and df["mlf"].notna().any():
        # reuse gallery helper pattern via temporary column loop
        sub0 = df[np.isclose(df.r, 0.0)]
        vmin = float(sub0.mlf.quantile(0.02))
        vmax = float(sub0.mlf.quantile(0.98))
        for r in R_VALS:
            fig, axs = plt.subplots(
                len(A_VALS), len(U_VALS),
                figsize=(3.6 * len(U_VALS), 2.8 * len(A_VALS)),
                sharex=True, sharey=True, constrained_layout=True,
            )
            if len(A_VALS) == 1:
                axs = np.array([axs])
            im = None
            for i, a in enumerate(A_VALS):
                marks = landmarks(a)
                for j, u in enumerate(U_VALS):
                    ax = axs[i, j]
                    d1 = df[np.isclose(df.a, a) & np.isclose(df.u, u)
                            & np.isclose(df.r, r)]
                    piv = d1.pivot_table(index="alpha_over_a",
                                         columns="eps_over_a", values="mlf")
                    alp_ax, eps_ax, Z = (piv.index.values, piv.columns.values,
                                        piv.values)
                    extent = [eps_ax.min(), eps_ax.max(),
                              alp_ax.min(), alp_ax.max()]
                    im = ax.imshow(Z, origin="lower", aspect="auto",
                                   extent=extent, cmap="YlOrBr",
                                   interpolation="bilinear",
                                   vmin=vmin, vmax=vmax, zorder=2)
                    draw_route_partition(ax, show_labels=True,
                                         alpha_min=float(alp_ax.min()),
                                         alpha_max=float(alp_ax.max()))
                    for m in marks:
                        ax.scatter(m["eps_over_a"], m["alpha_over_a"], s=28,
                                   c=CASE_COLORS[m["case"]], edgecolors="white",
                                   linewidths=0.5, zorder=5)
                        ax.annotate(m["case"],
                                    (m["eps_over_a"], m["alpha_over_a"]),
                                    textcoords="offset points", xytext=(3, 2),
                                    fontsize=7, fontweight="bold",
                                    color=CASE_COLORS[m["case"]])
                    ax.set_title(rf"$a={a:g}$, $u={u:g}$", fontsize=9)
                    if j == 0:
                        ax.set_ylabel(r"$\alpha/a$", fontsize=9)
                    if i == len(A_VALS) - 1:
                        ax.set_xlabel(r"$\varepsilon/a$", fontsize=9)
            fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02,
                         label=r"$\langle \ell f\rangle_{\psi^*}$")
            fig.suptitle(
                rf"Mean log-fitness at eq.  ·  diffusion  ·  $N={N}$, $r={r:g}$",
                fontsize=12, fontweight="bold",
            )
            out = OUT / f"fig_mlf_gallery_r{r:g}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print("wrote", out)
            plt.close(fig)
    for a, u in [(0.20, 5e-3), (0.05, 5e-3), (0.10, 5e-3), (0.40, 5e-3)]:
        d1 = df[np.isclose(df.a, a) & np.isclose(df.u, u) & np.isclose(df.r, 0.0)]
        if len(d1):
            plot_single_full(d1, a, u, 0.0)
    # also refresh classic single-a surfaces if present
    for a in (0.05, 0.18):
        p = OUT / f"surface_a{a:g}_r0.csv"
        if p.exists():
            d1 = pd.read_csv(p)
            # u column may be absent on early CSVs
            if "u" not in d1.columns:
                d1 = d1.assign(u=5e-3, r=0.0)
            plot_single_full(d1, a, float(d1.u.iloc[0]), 0.0)


if __name__ == "__main__":
    main()
