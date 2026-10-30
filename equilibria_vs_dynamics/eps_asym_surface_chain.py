#!/usr/bin/env python3
"""Iso-peak (ε, α) surface using the TRUE 2-loci Wright–Fisher Markov chain.

ψ* = stationary_fast(selection_P(...))  — recombination r DOES enter P.
Also τ_spec = -1/ln|λ₂| from the same P (spectral relaxation / mixing time).

Parameterization (same as diffusion surfaces):
  α = s1 - s2 ∈ [α_min, 2a],  ε = eps12 ∈ [-4a, 4a]
  s1,s2 = (2a - ε ± α)/2
  η_maint = D_eq / C*

Outputs under results/eps_asym_surface_chain/

Per grid point is ANALYTICAL / spectral (NO trajectory dynamics):
  build P → ψ*=stat(P), τ_spec from λ₂, D_eq, F*, C*, η_maint, x̄₁₁, …
Sized for ~12h overnight on N=30 (≈2–2.5 s/point).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, BoundaryNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from node_jump_analysis import build_chain, Landscape, selection_P
from adaptation_per_a import CASE_DEFS, CASE_COLORS, CASE_ORDER, cost_per_state
from compare_two_cases import free_fitness5_asymptote
from info_speed_experiment import spectral_gap
from route_partition import draw_route_partition, n_mono_routes
from simplex import (
    build_sigma, fitness_exp, stationary_fast, kl_divergence_bits,
    mutation_matrix_asymmetric,
)

OUT = Path(__file__).resolve().parent / "results" / "eps_asym_surface_chain"
N = 30
N_JOBS = 2

# Larger (ε,α) grid + richer (a,u,r) — aim ~12h (no dynamics trajectories)
N_EPS, N_ALPHA = 31, 15          # 465 pts/clock
ALPHA_OA_MIN, ALPHA_OA_MAX = -0.15, 2.0
EPS_OA_MIN, EPS_OA_MAX = -4.0, 4.0

A_VALS = [0.02, 0.05, 0.10, 0.20, 0.40]  # a=0.02 → Na=0.6 < 1
U_VALS = [5e-4, 1e-3, 5e-3, 2e-2]   # extra low mutation
R_VALS = [0.0, 0.1, 0.5]            # r matters on the chain

REQUIRED_COLS = ("D_eq", "Fstar", "Cstar", "eta_maint", "tau_spec", "x11")


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
            eps_over_a=eps / a, alpha_over_a=alpha / a,
        ))
    return out


def sweep_one(a: float, u: float, r: float) -> pd.DataFrame:
    t0 = time.perf_counter()
    chain = build_chain(N, u, u, r, N_JOBS)
    M = mutation_matrix_asymmetric(np.array([u, u]), np.array([u, u]))
    fracs = chain.fracs
    phi = chain.phi
    eps_oa = np.linspace(EPS_OA_MIN, EPS_OA_MAX, N_EPS)
    alpha_oa = np.linspace(ALPHA_OA_MIN, ALPHA_OA_MAX, N_ALPHA)
    rows = []
    n_tot = N_EPS * N_ALPHA
    n = 0
    for e_oa in eps_oa:
        for al_oa in alpha_oa:
            eps = e_oa * a
            alpha = al_oa * a
            s1, s2 = s1_s2(a, eps, alpha)
            L = Landscape("grid", "grid", s1=s1, s2=s2, eps12=eps, blurb="")
            P = selection_P(chain, L, N_JOBS)
            psi = stationary_fast(P)
            lam2 = spectral_gap(P)
            tau = (float(-1.0 / np.log(lam2)) if 0.0 < lam2 < 1.0
                   else float("nan"))
            gap = float(1.0 - lam2) if np.isfinite(lam2) else float("nan")
            omega = fitness_exp(build_sigma(s1, s2, eps))
            D = float(kl_divergence_bits(psi, phi))
            Cstar = float(psi @ cost_per_state(fracs, omega))
            Fstar, _ = free_fitness5_asymptote(psi, phi, fracs, omega, M, N)
            mlf = float(psi @ (fracs @ L.logfit))
            xbar = fracs.T @ psi
            eta_m = (D / Cstar) if Cstar > 1e-15 else float("nan")
            rows.append(dict(
                N=N, u=u, v=u, r=r, a=a,
                eps=eps, alpha=alpha,
                eps_over_a=e_oa, alpha_over_a=al_oa,
                s1=s1, s2=s2,
                D_eq=D, Fstar=Fstar, Cstar=Cstar, eta_maint=eta_m,
                tau_spec=tau, lam2=float(lam2), gap=gap,
                mlf=mlf,
                x00=float(xbar[0]), x01=float(xbar[1]),
                x10=float(xbar[2]), x11=float(xbar[3]),
                method="markov_chain_no_traj",
            ))
            n += 1
        if (n // N_ALPHA) % 5 == 0 or e_oa == eps_oa[-1]:
            print(f"    eps/a={e_oa:+.2f}  {n}/{n_tot}  "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def _clock_complete(g: pd.DataFrame) -> bool:
    if len(g) < N_EPS * N_ALPHA:
        return False
    for c in REQUIRED_COLS:
        if c not in g.columns or g[c].isna().all():
            return False
    # require tau present for (almost) all points
    if "tau_spec" not in g.columns or g["tau_spec"].isna().mean() > 0.05:
        return False
    return True


def sweep_grid(a_vals=A_VALS, u_vals=U_VALS, r_vals=R_VALS) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / "surface_grid_auv_chain.csv"
    t0 = time.perf_counter()
    parts = []
    done = set()
    if csv.exists():
        old = pd.read_csv(csv)
        keep = []
        for (a, u, r), g in old.groupby(["a", "u", "r"], sort=False):
            if _clock_complete(g):
                keep.append(g)
                done.add((float(a), float(u), float(r)))
            else:
                print(f"  dropping incomplete/no-tau clock "
                      f"a={a:g} u={u:g} r={r:g} (rows={len(g)})", flush=True)
        if keep:
            parts.append(pd.concat(keep, ignore_index=True))
        print(f"resuming — {len(done)} complete clocks with tau_spec", flush=True)

    jobs = [(a, u, r) for r in r_vals for u in u_vals for a in a_vals]
    for i, (a, u, r) in enumerate(jobs, 1):
        key = (float(a), float(u), float(r))
        if any(abs(key[0] - d[0]) < 1e-12 and abs(key[1] - d[1]) < 1e-15
               and abs(key[2] - d[2]) < 1e-15 for d in done):
            print(f"[{i}/{len(jobs)}] skip a={a:g} u={u:g} r={r:g}", flush=True)
            continue
        print(f"[{i}/{len(jobs)}] CHAIN a={a:g} u={u:g} r={r:g}  "
              f"grid={N_EPS}×{N_ALPHA}  (+tau_spec)", flush=True)
        df1 = sweep_one(a, u, r)
        parts.append(df1)
        df = pd.concat(parts, ignore_index=True)
        df.to_csv(csv, index=False)
        print(f"  checkpoint → {csv}  rows={len(df)}  "
              f"elapsed {time.perf_counter()-t0:.0f}s", flush=True)
    return pd.read_csv(csv)


def _metric_panel(ax, df_one, col, a, u, vmin, vmax, show_ylab, show_xlab,
                  marks, log_color=False):
    piv = df_one.pivot_table(index="alpha_over_a", columns="eps_over_a",
                             values=col)
    alp_ax = piv.index.values
    eps_ax = piv.columns.values
    Z = piv.values
    extent = [eps_ax.min(), eps_ax.max(), alp_ax.min(), alp_ax.max()]
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_color else None
    im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent,
                   cmap="magma_r" if col == "tau_spec" else "plasma",
                   interpolation="bilinear",
                   vmin=None if log_color else vmin,
                   vmax=None if log_color else vmax,
                   norm=norm, zorder=2)
    draw_route_partition(ax, show_labels=True)
    EE, AA = np.meshgrid(eps_ax, alp_ax)
    zmin, zmax = np.nanmin(Z), np.nanmax(Z)
    if np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin:
        if log_color and zmin > 0:
            levels = np.geomspace(zmin, zmax, 7)[1:-1]
        else:
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
    ax.set_title(rf"$a={a:g}$, $u={u:g}$", fontsize=9)
    if show_xlab:
        ax.set_xlabel(r"$\varepsilon/a$", fontsize=9)
    if show_ylab:
        ax.set_ylabel(r"$\alpha/a$", fontsize=9)
    return im


def plot_metric_gallery(df: pd.DataFrame, col: str, label: str,
                        fname_prefix: str, log_color: bool = False) -> list[Path]:
    outs = []
    for r in R_VALS:
        sub = df[np.isclose(df.r, r)]
        if sub.empty or col not in sub.columns:
            continue
        vals = sub[col].replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            continue
        if log_color:
            vmin = float(max(vals.quantile(0.02), vals[vals > 0].min()))
            vmax = float(vals.quantile(0.98))
        else:
            vmin = float(vals.quantile(0.02))
            vmax = float(vals.quantile(0.98))
        a_present = [a for a in A_VALS if np.any(np.isclose(sub.a, a))]
        u_present = [u for u in U_VALS if np.any(np.isclose(sub.u, u))]
        if not a_present or not u_present:
            continue
        fig, axs = plt.subplots(
            len(a_present), len(u_present),
            figsize=(3.6 * len(u_present), 2.8 * len(a_present)),
            sharex=True, sharey=True, constrained_layout=True,
        )
        if len(a_present) == 1 and len(u_present) == 1:
            axs = np.array([[axs]])
        elif len(a_present) == 1:
            axs = np.array([axs])
        elif len(u_present) == 1:
            axs = axs.reshape(-1, 1)
        im = None
        for i, a in enumerate(a_present):
            marks = landmarks(a)
            for j, u in enumerate(u_present):
                ax = axs[i, j]
                d1 = sub[np.isclose(sub.a, a) & np.isclose(sub.u, u)]
                if d1.empty:
                    ax.set_visible(False)
                    continue
                im = _metric_panel(ax, d1, col, a, u, vmin, vmax,
                                   show_ylab=(j == 0),
                                   show_xlab=(i == len(a_present) - 1),
                                   marks=marks, log_color=log_color)
        if im is not None:
            fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02, label=label)
        fig.suptitle(
            rf"{label}  ·  2-loci MARKOV chain  ·  $N={N}$, $r={r:g}$"
            "\n"
            rf"grid {N_EPS}$\times${N_ALPHA} in $(\varepsilon/a,\alpha/a)$"
            + (r";  color log-scaled" if log_color else ""),
            fontsize=12, fontweight="bold",
        )
        out = OUT / f"{fname_prefix}_r{r:g}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("wrote", out)
        plt.close(fig)
        outs.append(out)
    return outs


def plot_single_full(df_one: pd.DataFrame, a: float, u: float, r: float) -> Path:
    marks = landmarks(a)
    metrics = [
        ("D_eq", r"$D_{\mathrm{eq}}$ [bits]", "viridis", False),
        ("Fstar", r"$F^*$ [nats]", "magma", False),
        ("mlf", r"mean log-fitness  $\langle \ell f\rangle_{\psi^*}$", "YlOrBr", False),
        ("Cstar", r"$C^*$ [bits/gen]", "cividis", False),
        ("eta_maint", r"$D_{\mathrm{eq}}/C^*$", "plasma", False),
        ("tau_spec", r"$\tau_{\mathrm{spec}}=-1/\ln|\lambda_2|$ [gen]",
         "magma_r", True),
    ]
    fig, axs = plt.subplots(2, 3, figsize=(15.5, 9.2), constrained_layout=True)
    for ax, (col, title, cmap, logc) in zip(axs.ravel(), metrics):
        piv = df_one.pivot_table(index="alpha_over_a", columns="eps_over_a",
                                 values=col)
        alp_ax, eps_ax, Z = piv.index.values, piv.columns.values, piv.values
        extent = [eps_ax.min(), eps_ax.max(), alp_ax.min(), alp_ax.max()]
        if logc:
            zpos = Z[np.isfinite(Z) & (Z > 0)]
            norm = LogNorm(vmin=zpos.min(), vmax=zpos.max()) if len(zpos) else None
            im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent,
                           cmap=cmap, interpolation="bilinear", norm=norm,
                           zorder=2)
        else:
            im = ax.imshow(Z, origin="lower", aspect="auto", extent=extent,
                           cmap=cmap, interpolation="bilinear", zorder=2)
        draw_route_partition(ax, show_labels=(ax is axs.ravel()[0]))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        EE, AA = np.meshgrid(eps_ax, alp_ax)
        zmin, zmax = np.nanmin(Z), np.nanmax(Z)
        if np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin:
            levels = (np.geomspace(zmin, zmax, 9)[1:-1]
                      if logc and zmin > 0
                      else np.linspace(zmin, zmax, 9)[1:-1])
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
    fig.suptitle(
        rf"Iso-peak surface · 2-loci MARKOV chain · "
        rf"$N={N}$, $u=v={u:g}$, $r={r:g}$, $a={a:g}$",
        fontsize=12, fontweight="bold",
    )
    out = OUT / f"fig_eps_asym_chain_a{a:g}_u{u:g}_r{r:g}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    plt.close(fig)
    return out


def _epistasis_label(eps_oa: float, alpha_oa: float) -> str:
    """Rough type tag for a peak location in a-units."""
    if eps_oa < -0.5:
        return "antagonistic/multi-peak-ish (ε<0)"
    if abs(eps_oa) < 0.35 and alpha_oa < 0.35:
        return "near-additive"
    if abs(eps_oa - 1.0) < 0.4 and alpha_oa < 0.35:
        return "near-synergy"
    if abs(eps_oa - 2.0) < 0.45 and alpha_oa < 0.35:
        return "near-neutral-intermediates"
    if abs(eps_oa - 4.0) < 0.6 and alpha_oa < 0.35:
        return "near-reciprocal-sign"
    if abs(eps_oa - 2.0) < 0.6 and abs(alpha_oa - 2.0) < 0.5:
        return "near-sign-epistasis"
    if abs(eps_oa) < 0.35 and abs(alpha_oa - 1.0) < 0.4:
        return "near-asymmetric-additive"
    if eps_oa > 0.5 and alpha_oa < 0.5:
        return "positive-epistasis ridge (α≈0)"
    if alpha_oa > 1.0:
        return "high-asymmetry"
    return "other"


def write_peak_report(df: pd.DataFrame) -> Path:
    """Global max η_maint and min τ_spec per clock — hunt multiple good regimes."""
    lines = [
        "PEAK HUNT — maintenance efficiency & spectral speed",
        "==================================================",
        "Per (a,u,r): argmax η_maint=D_eq/C* and argmin τ_spec.",
        "Expectation to test: additivity (ε≈0) often best on η;",
        "look for OTHER good epistatic regimes too.",
        "",
    ]
    rows_out = []
    for (a, u, r), g in df.groupby(["a", "u", "r"]):
        if "eta_maint" not in g or "tau_spec" not in g:
            continue
        imax = g["eta_maint"].idxmax()
        imin = g["tau_spec"].idxmin()
        rmax, rmin = g.loc[imax], g.loc[imin]
        for kind, row in (("eta_max", rmax), ("tau_min", rmin)):
            rows_out.append(dict(
                kind=kind, a=a, u=u, r=r,
                eps_over_a=row.eps_over_a, alpha_over_a=row.alpha_over_a,
                eta_maint=row.eta_maint, tau_spec=row.tau_spec,
                D_eq=row.D_eq, Cstar=row.Cstar, x11=row.get("x11", np.nan),
                label=_epistasis_label(row.eps_over_a, row.alpha_over_a),
            ))
        lines.append(
            f"a={a:g} u={u:g} r={r:g}\n"
            f"  η_max @ (ε/a,α/a)=({rmax.eps_over_a:+.2f},{rmax.alpha_over_a:.2f})  "
            f"η={rmax.eta_maint:.4g}  τ={rmax.tau_spec:.4g}  "
            f"[{_epistasis_label(rmax.eps_over_a, rmax.alpha_over_a)}]\n"
            f"  τ_min @ (ε/a,α/a)=({rmin.eps_over_a:+.2f},{rmin.alpha_over_a:.2f})  "
            f"τ={rmin.tau_spec:.4g}  η={rmin.eta_maint:.4g}  "
            f"[{_epistasis_label(rmin.eps_over_a, rmin.alpha_over_a)}]"
        )
    rep = pd.DataFrame(rows_out)
    csv = OUT / "peak_hunt_eta_tau.csv"
    rep.to_csv(csv, index=False)
    # summary: how often is best η near-additive?
    if len(rep):
        em = rep[rep.kind == "eta_max"]
        frac_add = float(em.label.str.contains("additive").mean())
        lines += [
            "",
            f"Fraction of clocks with η_max labeled *additive*: {frac_add:.2f}",
            "Label counts (η_max):",
            em.label.value_counts().to_string(),
            "",
            "Label counts (τ_min / fastest):",
            rep[rep.kind == "tau_min"].label.value_counts().to_string(),
        ]
    txt = OUT / "peak_hunt_report.txt"
    txt.write_text("\n".join(lines) + "\n")
    print("wrote", txt)
    print("wrote", csv)
    return txt


def plot_route_legend_card() -> Path:
    """Standalone schematic of 0/1/2-route provinces on the iso-peak plane."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    e = np.linspace(EPS_OA_MIN, EPS_OA_MAX, 500)
    al = np.linspace(ALPHA_OA_MIN, ALPHA_OA_MAX, 300)
    EE, AA = np.meshgrid(e, al)
    Nmono = n_mono_routes(EE, AA)
    ax.contourf(EE, AA, Nmono, levels=[-0.5, 0.5, 1.5, 2.5],
                colors=["#6b3a7a", "#3d7ab5", "#3aa66a"], alpha=0.85)
    ax.contour(EE, AA, Nmono, levels=[0.5, 1.5], colors="k", linewidths=1.5)
    for slug_m in landmarks(0.18):
        ax.scatter(slug_m["eps_over_a"], slug_m["alpha_over_a"], s=80,
                   c=CASE_COLORS[slug_m["case"]], edgecolors="white",
                   linewidths=0.9, zorder=5)
        ax.annotate(slug_m["case"],
                    (slug_m["eps_over_a"], slug_m["alpha_over_a"]),
                    textcoords="offset points", xytext=(5, 4),
                    fontsize=10, fontweight="bold",
                    color=CASE_COLORS[slug_m["case"]])
    for (x, y, txt) in [(0.0, 0.4, "2 accessible routes"),
                        (2.2, 1.5, "1 accessible route"),
                        (3.3, 0.3, "0 accessible routes\n(valley)")]:
        ax.text(x, y, txt, ha="center", va="center", fontsize=9,
                fontweight="bold", color="white")
    ax.set_xlabel(r"epistasis  $\varepsilon/a$")
    ax.set_ylabel(r"asymmetry  $\alpha/a$")
    ax.set_xlim(EPS_OA_MIN, EPS_OA_MAX)
    ax.set_ylim(ALPHA_OA_MIN, ALPHA_OA_MAX)
    ax.set_title(
        "Topology partition of the iso-peak plane\n"
        r"by # of fitness-increasing paths $00\to11$ "
        r"(steps $>0$ on both edges)",
        fontsize=11,
    )
    handles = [
        Line2D([0], [0], color="#3aa66a", lw=8, label="2 routes (e.g. B,C,F,A)"),
        Line2D([0], [0], color="#3d7ab5", lw=8, label="1 route (e.g. D sign)"),
        Line2D([0], [0], color="#6b3a7a", lw=8, label="0 routes (e.g. E reciprocal)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.9)
    out = OUT / "fig_route_partition_schematic.png"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    plt.close(fig)
    return out


def make_all_plots(df: pd.DataFrame) -> None:
    plot_route_legend_card()
    plot_metric_gallery(
        df, "eta_maint",
        r"maintenance efficiency  $D_{\mathrm{eq}}/C^*$",
        "fig_eta_maint_gallery_chain",
    )
    plot_metric_gallery(
        df, "tau_spec",
        r"spectral relaxation  $\tau_{\mathrm{spec}}=-1/\ln|\lambda_2|$ [gen]"
        "\n(smaller = faster mixing)",
        "fig_tau_spec_gallery_chain",
        log_color=True,
    )
    if "mlf" in df.columns:
        plot_metric_gallery(
            df, "mlf",
            r"mean log-fitness at equilibrium  $\langle \ell f\rangle_{\psi^*}$",
            "fig_mlf_gallery_chain",
        )
    if "x11" in df.columns:
        plot_metric_gallery(
            df, "x11",
            r"mean double-mutant frequency  $\bar x_{11}$",
            "fig_x11_gallery_chain",
        )
    write_peak_report(df)
    for a, u, r in [(0.05, 5e-3, 0.0), (0.20, 5e-3, 0.0), (0.20, 5e-3, 0.5),
                    (0.10, 5e-4, 0.0)]:
        d1 = df[np.isclose(df.a, a) & np.isclose(df.u, u) & np.isclose(df.r, r)]
        if len(d1) and "tau_spec" in d1.columns:
            plot_single_full(d1, a, u, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true",
                    help="only replot from existing CSV (adds route overlays)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        csv = OUT / "surface_grid_auv_chain.csv"
        df = pd.read_csv(csv)
        print(f"plot-only from {csv}  rows={len(df)}", flush=True)
        make_all_plots(df)
        return

    n_clocks = len(A_VALS) * len(U_VALS) * len(R_VALS)
    n_pts = N_EPS * N_ALPHA
    print("=" * 60)
    print("2-loci MARKOV chain (ε,α) — ANALYTICAL (no trajectories)")
    print(f"  per point: P → ψ*, τ_spec(λ2), D, F*, C*, η, x̄")
    print(f"grid {N_EPS}×{N_ALPHA} = {n_pts} pts/clock")
    print(f"clocks: a={A_VALS}")
    print(f"        u={U_VALS}")
    print(f"        r={R_VALS}")
    print(f"total points ~{n_clocks * n_pts}  (~{n_clocks} clocks)")
    print(f"out: {OUT}")
    print("=" * 60, flush=True)
    # schematic available immediately (does not need sweep)
    plot_route_legend_card()
    df = sweep_grid()
    make_all_plots(df)


if __name__ == "__main__":
    main()
