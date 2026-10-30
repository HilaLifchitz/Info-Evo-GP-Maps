#!/usr/bin/env python3
r"""
EXPERIMENT: how does the diffusion approximation of the discrete 2-locus selection
equilibrium degrade as we push N*s_eff from << 1 up past 1?

Fixed N, fixed (very weak) mutation u=v, r=0. We pick a Sigma "shape" (direction
in (s1, s2, eps12) space) and scale it by a factor c so that the binding selection
strength  N * max_g |s_eff(g)|  sweeps a requested grid of TARGETS that starts well
below 1 and surpasses 1.

For each target we build the discrete Wright-Fisher stationary psi* and the
diffusion Boltzmann density p(x) (alpha="cond"), restrict both to the interior of
the simplex, renormalize, and report interior TV and KL(psi*||diff) in bits, plus
psi*'s boundary mass.

Effective per-genotype selection (log-fitness relative to baseline 00):
    s_eff(01) = s2,  s_eff(10) = s1,  s_eff(11) = s1 + s2 + eps12.

Run:
  python3 sigma_sweep_surpass_Ns.py --out results/sigma_sweep_surpass
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

import simplex as sp
from selection_diffusion_vs_chain import (
    Setting,
    diffusion_logdensity,
    discrete_psi_star,
    _normalize_log,
    _tv_kl,
    genotype_marginals,
)

SCRIPT_DIR = Path(__file__).resolve().parent
GENO_LABELS = ["00", "01", "10", "11"]

# ----------------------------- experiment config -----------------------------
N = 40
U = V = 0.1
R = 0.0
ALPHA_MODE = "cond"

# Sigma direction (positive epistasis shape). Only the RELATIVE sizes matter;
# the script rescales it to hit each N*s_eff target.
DIR_S1, DIR_S2, DIR_EPS = 0.08, -0.04, 0.15

# N*s_eff targets: start well below 1, then surpass it.
TARGETS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0]

# which targets get a marginals overlay plot
MARGINAL_TARGETS = [0.1, 1.0, 3.0]


def _seff(s1: float, s2: float, eps12: float) -> dict[str, float]:
    return {"01": s2, "10": s1, "11": s1 + s2 + eps12}


def _scaled_setting(target: float) -> tuple[Setting, dict[str, float]]:
    """Scale the Sigma direction so that N*max|s_eff| == target."""
    base = _seff(DIR_S1, DIR_S2, DIR_EPS)
    base_max = max(abs(v) for v in base.values())
    c = target / (N * base_max)
    s1, s2, eps12 = c * DIR_S1, c * DIR_S2, c * DIR_EPS
    st = Setting(label=f"Nseff={target:g}", N=N, s1=s1, s2=s2, eps12=eps12, u=U, v=V, r=R)
    return st, _seff(s1, s2, eps12)


def run(out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="notebook", font_scale=0.9)
    except ImportError:
        pass
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm",
                         "figure.facecolor": "white"})
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    marg_cache: dict[float, tuple] = {}
    log_lines = [f"N={N}, u=v={U}, r={R}, alpha={ALPHA_MODE}",
                 f"Sigma direction (s1,s2,eps12) = ({DIR_S1},{DIR_S2},{DIR_EPS})", ""]

    for target in TARGETS:
        st, seff = _scaled_setting(target)
        Ns = {g: N * v for g, v in seff.items()}
        print(f"\n### target N*s_eff={target:g}: s1={st.s1:.5f}, s2={st.s2:.5f}, "
              f"eps12={st.eps12:.5f}  (N*s_eff: 01={Ns['01']:.3f}, 10={Ns['10']:.3f}, "
              f"11={Ns['11']:.3f})", flush=True)

        psi, fracs, counts, omega, M_mut = discrete_psi_star(st)
        interior = np.all(fracs > 0.0, axis=1)
        boundary_mass = float(psi[~interior].sum())

        log_p, _ = diffusion_logdensity(st, fracs, omega, M_mut, ALPHA_MODE)
        psi_int = np.zeros_like(psi)
        psi_int[interior] = psi[interior]
        s = psi_int.sum()
        psi_int = psi_int / s if s > 0 else psi_int
        pdiff_int = _normalize_log(log_p, interior)
        tv, kl = _tv_kl(psi_int, pdiff_int)

        rows.append((target, Ns["11"], tv, kl, boundary_mass))
        log_lines.append(f"target={target:g}  Ns_eff(11)={Ns['11']:.3f}  "
                         f"TV={tv:.4e}  KL={kl:.4e} bits  boundary={boundary_mass:.4e}")
        print(f"  interior TV={tv:.4e}  KL={kl:.4e} bits  boundary mass={boundary_mass:.4e}",
              flush=True)

        if target in MARGINAL_TARGETS:
            marg_cache[target] = (
                genotype_marginals(psi_int, counts, N),
                genotype_marginals(pdiff_int, counts, N),
                omega,
            )

    rows = np.array([(t, n, tv, kl, b) for (t, n, tv, kl, b) in rows], dtype=float)
    targets = rows[:, 0]
    tvs = rows[:, 2]
    kls = rows[:, 3]
    bmass = rows[:, 4]

    # ---- (1) TV / KL / boundary mass vs N*s_eff target ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    axL.plot(targets, tvs, "o-", color="#3b528b", label="interior TV")
    axL.plot(targets, kls, "s-", color="#5ec962", label="interior KL (bits)")
    axL.axvline(1.0, color="k", ls="--", lw=1, alpha=0.7, label=r"$N s_{eff}=1$")
    axL.set_xlabel(r"$N\,\max_g|s_{eff}(g)|$ (binding selection strength)")
    axL.set_ylabel("distance (chain vs diffusion)")
    axL.set_title(rf"Diffusion accuracy vs selection, $N$={N}, $u=v$={U}")
    axL.legend(); axL.grid(True, alpha=0.3)

    axR.plot(targets, bmass, "^-", color="#b5367a")
    axR.axvline(1.0, color="k", ls="--", lw=1, alpha=0.7)
    axR.set_xlabel(r"$N\,\max_g|s_{eff}(g)|$")
    axR.set_ylabel(r"$\psi^*$ boundary mass (>=1 genotype absent)")
    axR.set_title("Boundary mass of the chain equilibrium")
    axR.set_yscale("log")
    axR.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    p = out_dir / "tv_kl_vs_Nseff.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"\nWrote {p}")

    # ---- (2) representative marginals overlays ----
    for target in MARGINAL_TARGETS:
        if target not in marg_cache:
            continue
        mp, md, omega = marg_cache[target]
        ks = np.arange(N + 1) / N
        fig2, axes = plt.subplots(2, 2, figsize=(11, 8))
        for g in range(4):
            ax = axes[g // 2][g % 2]
            ax.plot(ks, mp[g], "o", ms=3.5, color="C0", label=r"chain $\psi^*$")
            ax.plot(ks, md[g], "-", lw=1.6, color="C3", alpha=0.85, label="diffusion")
            ax.set_title(rf"genotype {GENO_LABELS[g]}  ($\omega$={omega[g]:.3f})", fontsize=9)
            ax.set_xlabel("genotype frequency"); ax.set_ylabel("marginal prob.")
            ax.grid(True, alpha=0.3)
            if g == 0:
                ax.legend(fontsize=8)
        fig2.suptitle(rf"marginals at $N s_{{eff}}$={target:g}  ($N$={N}, $u=v$={U}, "
                      rf"$\alpha$={ALPHA_MODE})", y=0.99, fontsize=12)
        fig2.tight_layout(rect=(0, 0, 1, 0.97))
        p2 = out_dir / f"marginals_Nseff_{target:g}.png"
        fig2.savefig(p2, dpi=170, bbox_inches="tight"); plt.close(fig2)
        print(f"Wrote {p2}")

    (out_dir / "sweep.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir / 'sweep.log'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or (SCRIPT_DIR / "results" /
                       f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_sigma_sweep_surpass")
    run(out)
    print(f"\nAll outputs in: {out}")


if __name__ == "__main__":
    main()
