#!/usr/bin/env python3
r"""
Sweep recombination r and population size N: does the discrete SELECTION
equilibrium psi* get closer to the genotype-simplex DIFFUSION equilibrium
p(x) ∝ W(x)^N * prod_i x_i^{alpha_i - 1} ?

NOTE: the diffusion formula has NO recombination term (it is the 4-allele /
r=0 limit). Raising r only changes the discrete chain, so this measures whether
the chain happens to land closer to the (r-independent) diffusion.

Fixed selection/mutation (moderate setting); we vary r and N, and for each
(N, r) report interior TV and KL(psi* || diffusion) using the "cond" alpha.

Run:
  python3 selection_diffusion_r_N_sweep.py
  python3 selection_diffusion_r_N_sweep.py --out results/sel_diff_rN
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
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
)

SCRIPT_DIR = Path(__file__).resolve().parent

# fixed selection + mutation; N and r are swept
BASE = Setting("moderate sel + mut", N=40, s1=0.08, s2=-0.04, eps12=0.15, u=0.08, v=0.08)
N_VALUES = [40, 45, 50]
R_VALUES = [0.0, 0.1, 0.25, 0.5, 1.0]
ALPHA_MODE = "cond"


def one_point(N: int, r: float) -> dict:
    st = replace(BASE, N=N, r=r)
    psi, fracs, counts, omega, M_mut = discrete_psi_star(st)
    interior = np.all(fracs > 0.0, axis=1)
    boundary_mass = float(psi[~interior].sum())

    log_p, _ = diffusion_logdensity(st, fracs, omega, M_mut, ALPHA_MODE)
    psi_int = np.zeros_like(psi)
    psi_int[interior] = psi[interior]
    s = psi_int.sum()
    if s > 0:
        psi_int /= s
    pdiff_int = _normalize_log(log_p, interior)
    tv, kl = _tv_kl(psi_int, pdiff_int)
    return {"N": N, "r": r, "tv": tv, "kl": kl, "boundary": boundary_mass}


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
    print(f"selection diffusion r/N sweep  (alpha={ALPHA_MODE})")
    print(f"  selection s1={BASE.s1}, s2={BASE.s2}, eps12={BASE.eps12}, u=v={BASE.u}")
    print(f"  N in {N_VALUES}, r in {R_VALUES}")
    print("=" * 72, flush=True)

    rows: list[dict] = []
    for N in N_VALUES:
        for r in R_VALUES:
            t0 = time.perf_counter()
            res = one_point(N, r)
            rows.append(res)
            print(f"  N={N:2d} r={r:<4g} | TV={res['tv']:.4f}  KL={res['kl']:.4f} bits  "
                  f"boundary={res['boundary']:.3f}  ({time.perf_counter()-t0:.1f}s)", flush=True)

    # CSV
    csv = out_dir / "rN_sweep.csv"
    with csv.open("w", encoding="utf-8") as f:
        f.write("N,r,tv,kl_bits,boundary_mass\n")
        for x in rows:
            f.write(f"{x['N']},{x['r']},{x['tv']:.6g},{x['kl']:.6g},{x['boundary']:.6g}\n")
    print(f"Wrote {csv}")

    # plots: TV vs r and KL vs r, one line per N
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    markers = ["o", "s", "^", "D", "v"]
    for i, N in enumerate(N_VALUES):
        sub = [x for x in rows if x["N"] == N]
        rs = [x["r"] for x in sub]
        tvs = [x["tv"] for x in sub]
        kls = [x["kl"] for x in sub]
        axes[0].plot(rs, tvs, marker=markers[i % len(markers)], lw=1.8, ms=6, label=f"N={N}")
        axes[1].plot(rs, kls, marker=markers[i % len(markers)], lw=1.8, ms=6, label=f"N={N}")
    axes[0].set_xlabel(r"recombination rate $r$")
    axes[0].set_ylabel(r"interior TV$(\psi^*,\,\mathrm{diffusion})$")
    axes[0].set_title("Total-variation distance")
    axes[1].set_xlabel(r"recombination rate $r$")
    axes[1].set_ylabel(r"interior KL$(\psi^*\,\|\,\mathrm{diffusion})$ [bits]")
    axes[1].set_title("KL divergence")
    for ax in axes:
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle(
        rf"Discrete vs diffusion selection equilibrium ($W^N$, $\alpha$={ALPHA_MODE}):  "
        rf"$s_1$={BASE.s1}, $s_2$={BASE.s2}, $\epsilon_{{12}}$={BASE.eps12}, $u$={BASE.u}",
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = out_dir / "rN_sweep.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out or (SCRIPT_DIR / "results" /
                       f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_sel_diff_rN")
    run(out)
    print(f"\nAll outputs in: {out}")


if __name__ == "__main__":
    main()
