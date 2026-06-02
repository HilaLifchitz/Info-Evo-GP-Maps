#!/usr/bin/env python3
r"""
How does the chain-vs-diffusion fit scale with SELECTION STRENGTH?

We fix the selection "direction" (s1:s2:eps12 ratios) and scale its overall
magnitude across orders of magnitude, tracking the scaled selection parameter
    Ns_max = N * max(|s1|, |s2|, |eps12|).

For each magnitude we compute the interior TV / KL between the chain psi* and:
  * genotype-simplex diffusion  W^N * prod x_i^{alpha_i-1}  (alpha="cond")  at r=0
  * allele-frequency LE diffusion W(x1,x2)^N * prod x_i^{2Nu-1}(1-x_i)^{2Nv-1} at r=1

Prediction: both diffusions are near-exact at small Ns_max (weak selection) and
degrade as Ns_max grows, because the leading correction scales with the
per-generation selection s itself (not with finite-N discreteness).

Run:
  python3 selection_magnitude_sweep.py
  python3 selection_magnitude_sweep.py --out results/sel_magnitude
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

from selection_diffusion_vs_chain import (
    Setting,
    diffusion_logdensity,
    discrete_psi_star,
    _normalize_log,
    _tv_kl,
)
from allele_diffusion_vs_chain import (
    chain_allele_joint,
    allele_diffusion,
    _norm_interior_2d,
    _norm_logdensity_2d,
)

SCRIPT_DIR = Path(__file__).resolve().parent

# selection DIRECTION (ratios). Overall magnitude is scaled below.
DIR_S1, DIR_S2, DIR_EPS = 0.08, -0.04, 0.15
S_MAX_DIR = max(abs(DIR_S1), abs(DIR_S2), abs(DIR_EPS))   # = 0.15

N_POP = 40
U = 0.05
V = 0.05

# target Ns_max values (log-spaced), "scaling from ~1" upward and a bit below
NS_TARGETS = [0.3, 0.6, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 25.0]


def genotype_fit_r0(scale: float):
    st = Setting("g", N=N_POP, s1=DIR_S1 * scale, s2=DIR_S2 * scale,
                 eps12=DIR_EPS * scale, u=U, v=V, r=0.0)
    psi, fracs, counts, omega, M_mut = discrete_psi_star(st)
    log_p, interior = diffusion_logdensity(st, fracs, omega, M_mut, "cond")
    psi_int = np.zeros_like(psi)
    psi_int[interior] = psi[interior]
    s = psi_int.sum()
    if s > 0:
        psi_int /= s
    pdiff = _normalize_log(log_p, interior)
    tv, kl = _tv_kl(psi_int, pdiff)
    boundary = 1.0 - float(psi[interior].sum())
    return tv, kl, boundary


def allele_fit_r1(scale: float):
    st = Setting("a", N=N_POP, s1=DIR_S1 * scale, s2=DIR_S2 * scale,
                 eps12=DIR_EPS * scale, u=U, v=V, r=1.0)
    psi, fracs, counts, omega, M_mut = discrete_psi_star(st)
    H = chain_allele_joint(psi, counts, N_POP)
    log_rho, interior2d = allele_diffusion(N_POP, omega, U, V)
    H_int = _norm_interior_2d(H, interior2d)
    rho_int = _norm_logdensity_2d(log_rho, interior2d)
    tv, kl = _tv_kl(H_int.ravel(), rho_int.ravel())
    boundary = 1.0 - float(H[interior2d].sum())
    return tv, kl, boundary


def run(out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    except ImportError:
        pass
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "figure.facecolor": "white"})
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"selection magnitude sweep   N={N_POP}, u=v={U}")
    print(f"  direction s1:s2:eps = {DIR_S1}:{DIR_S2}:{DIR_EPS}  (s_max_dir={S_MAX_DIR})")
    print(f"  Ns_max targets: {NS_TARGETS}")
    print("=" * 72, flush=True)

    rows = []
    for ns in NS_TARGETS:
        scale = ns / (N_POP * S_MAX_DIR)            # so N*s_max = ns
        s_max = S_MAX_DIR * scale
        t0 = time.perf_counter()
        tv_g, kl_g, bd_g = genotype_fit_r0(scale)
        tv_a, kl_a, bd_a = allele_fit_r1(scale)
        rows.append({"ns": ns, "scale": scale, "s_max": s_max,
                     "tv_g": tv_g, "kl_g": kl_g, "bd_g": bd_g,
                     "tv_a": tv_a, "kl_a": kl_a, "bd_a": bd_a})
        print(f"  Ns_max={ns:<5g} (s_max={s_max:.4f}) | "
              f"GENO@r0 TV={tv_g:.4f} KL={kl_g:.4f} bd={bd_g:.3f} | "
              f"ALLELE@r1 TV={tv_a:.4f} KL={kl_a:.4f} bd={bd_a:.3f} "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)

    ns = np.array([x["ns"] for x in rows])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].plot(ns, [x["tv_g"] for x in rows], "o-", color="#1f77b4", lw=1.9, ms=6,
                 label=r"genotype/$\alpha$ @ $r=0$")
    axes[0].plot(ns, [x["tv_a"] for x in rows], "s-", color="#d62728", lw=1.9, ms=6,
                 label=r"allele/LE @ $r=1$")
    axes[1].plot(ns, [x["kl_g"] for x in rows], "o-", color="#1f77b4", lw=1.9, ms=6,
                 label=r"genotype/$\alpha$ @ $r=0$")
    axes[1].plot(ns, [x["kl_a"] for x in rows], "s-", color="#d62728", lw=1.9, ms=6,
                 label=r"allele/LE @ $r=1$")
    for ax, ylab, title in ((axes[0], "interior TV", "Total-variation distance"),
                            (axes[1], "interior KL [bits]", "KL divergence")):
        ax.set_xscale("log")
        ax.set_xlabel(r"$N\, s_{\max}$  (scaled selection)")
        ax.set_ylabel(ylab); ax.set_title(title)
        ax.axvline(1.0, color="gray", ls=":", lw=1)
        ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(rf"Chain vs diffusion fit vs selection strength  ($N$={N_POP}, $u$={U}, "
                 rf"direction {DIR_S1}:{DIR_S2}:{DIR_EPS})", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out_dir / "fit_vs_selection.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"Wrote {p}")

    csv = out_dir / "sel_magnitude.csv"
    with csv.open("w", encoding="utf-8") as f:
        f.write("Ns_max,scale,s_max,tv_geno_r0,kl_geno_r0,bdry_geno,tv_allele_r1,kl_allele_r1,bdry_allele\n")
        for x in rows:
            f.write(f"{x['ns']},{x['scale']:.6g},{x['s_max']:.6g},"
                    f"{x['tv_g']:.6g},{x['kl_g']:.6g},{x['bd_g']:.6g},"
                    f"{x['tv_a']:.6g},{x['kl_a']:.6g},{x['bd_a']:.6g}\n")
    print(f"Wrote {csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or (SCRIPT_DIR / "results" /
                       f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_sel_magnitude")
    run(out)
    print(f"\nAll outputs in: {out}")


if __name__ == "__main__":
    main()
