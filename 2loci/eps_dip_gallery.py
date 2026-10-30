#!/usr/bin/env python3
r"""
Gallery driver: produce ~10 eps12-sweep KL plots (each with the predicted dip
line eps12* = -2/3 (s1+s2)) like eps12_equilibrium_kl_sweep.py, varying
u=v, r, s1, s2 and N (each plot shows 3 different N). All kept small so that
N*s_eff stays small. Outputs to plots/eps_sweep_prediction_dip/.

Reuses run_sweep / plot_results / save_csv from eps12_equilibrium_kl_sweep so the
plots are visually identical in style; just renames each output to a unique file.

Run:
  python3 eps_dip_gallery.py
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from eps12_equilibrium_kl_sweep import plot_results, run_sweep, save_csv

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "plots" / "eps_sweep_prediction_dip"

# Each config: N tuple, r, u (=v), s1, s2, eps window (min,max,step).
# Predicted dip eps* = -2/3 (s1+s2) is kept inside every window.
CONFIGS = [
    # 1. reference-like, + N=20
    dict(N=(20, 30, 40), r=1e-3, u=1e-3, s1=-2e-4, s2=5e-4,  eps=(-2e-3, 2e-3, 3.3e-4)),
    # 2. smaller mutation, positive sum
    dict(N=(20, 30, 40), r=1e-3, u=5e-4, s1=5e-4,  s2=5e-4,  eps=(-2e-3, 2e-3, 3.3e-4)),
    # 3. bigger mutation, negative sum
    dict(N=(20, 30, 40), r=1e-3, u=3e-3, s1=-5e-4, s2=-5e-4, eps=(-2e-3, 2e-3, 3.3e-4)),
    # 4. no recombination
    dict(N=(20, 30, 40), r=0.0,  u=1e-3, s1=-1e-3, s2=5e-4,  eps=(-2e-3, 2e-3, 3.3e-4)),
    # 5. higher r, larger |sum|
    dict(N=(20, 30, 40), r=1e-2, u=1e-3, s1=1e-3,  s2=1e-3,  eps=(-3e-3, 3e-3, 5e-4)),
    # 6. high r
    dict(N=(20, 30, 40), r=1e-1, u=1e-3, s1=-5e-4, s2=-1e-3, eps=(-3e-3, 3e-3, 5e-4)),
    # 7. very small mutation
    dict(N=(20, 30, 40), r=1e-3, u=1e-4, s1=5e-4,  s2=-2e-4, eps=(-2e-3, 2e-3, 3.3e-4)),
    # 8. large mutation (expect small offset from -2/3 law)
    dict(N=(20, 30, 40), r=1e-3, u=1e-2, s1=-5e-4, s2=2.5e-4, eps=(-2e-3, 2e-3, 3.3e-4)),
    # 9. includes N=50
    dict(N=(30, 40, 50), r=1e-3, u=1e-3, s1=1e-3,  s2=-5e-4, eps=(-1.5e-3, 1.5e-3, 3e-4)),
    # 10. includes N=50
    dict(N=(30, 40, 50), r=1e-3, u=5e-4, s1=-5e-4, s2=-5e-4, eps=(-1.5e-3, 1.5e-3, 3e-4)),
]


def make_args(cfg: dict) -> SimpleNamespace:
    uv = np.array([cfg["u"], cfg["u"]], dtype=float)
    emin, emax, estep = cfg["eps"]
    return SimpleNamespace(
        N=tuple(cfg["N"]),
        r=float(cfg["r"]),
        u=uv, v=uv.copy(),
        s1=float(cfg["s1"]), s2=float(cfg["s2"]),
        eps_min=float(emin), eps_max=float(emax), eps_step=float(estep),
        n_jobs=-1,
        neutral_cache_dir="cache",
        output_dir=OUT_DIR,
        no_plot=False, show=False, quiet=True,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for i, cfg in enumerate(CONFIGS, start=1):
        args = make_args(cfg)
        eps_star = -2.0 / 3.0 * (args.s1 + args.s2)
        t0 = time.perf_counter()
        print(f"\n############ [{i}/{len(CONFIGS)}] N={args.N} r={args.r:g} "
              f"u=v={cfg['u']:g} s1={args.s1:g} s2={args.s2:g} "
              f"eps*={eps_star:+.3g} ############", flush=True)
        results = run_sweep(args)
        csv_path = save_csv(results, OUT_DIR, args)
        png_path = plot_results(results, OUT_DIR, args)

        tag = f"{i:02d}_u{cfg['u']:g}_v{cfg['u']:g}"
        new_png = png_path.with_name(f"{tag}_{png_path.name}")
        new_csv = csv_path.with_name(f"{tag}_{csv_path.name}")
        png_path.replace(new_png)
        csv_path.replace(new_csv)
        dt = time.perf_counter() - t0
        print(f"  wrote {new_png.name}  ({dt:.1f}s)", flush=True)
        summary.append(f"[{i:02d}] N={args.N} r={args.r:g} u=v={cfg['u']:g} "
                       f"s1={args.s1:g} s2={args.s2:g} eps*={eps_star:+.3g}  -> {new_png.name}")

    (OUT_DIR / "gallery_index.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))
    print(f"\nAll plots in: {OUT_DIR}")


if __name__ == "__main__":
    main()
