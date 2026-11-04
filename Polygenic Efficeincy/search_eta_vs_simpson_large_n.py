#!/usr/bin/env python3
"""η_maint vs n at fixed E[S]=c, out to n=1000 Hamming layers."""

from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path

from eta_maint_boltzmann import compute

NS_CURVE = (4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1000)
NS_GRID = (16, 64, 256, 1000)
OUT = Path(__file__).with_name("eta_vs_n_simpson_large.json")

CURVES = [
    dict(name="canvas default", c=0.3, N=50.0, vs=0.0625, z_star=0.5),
    dict(name="poster 16/4", c=0.75, N=100.0, vs=0.0625, z_star=0.5),
    dict(name="high c, modest N", c=0.75, N=20.0, vs=0.0625, z_star=0.5),
    dict(name="weak sel", c=0.3, N=50.0, vs=1.0, z_star=0.5),
    dict(name="strong sel", c=0.3, N=50.0, vs=0.02, z_star=0.5),
    dict(name="small N", c=0.3, N=10.0, vs=0.0625, z_star=0.5),
    dict(name="mid c", c=0.5, N=50.0, vs=0.0625, z_star=0.5),
    dict(name="low c", c=0.26, N=50.0, vs=0.25, z_star=0.5),
]


def n_samples(n: int, ess_retry: bool = False) -> int:
    if ess_retry:
        return 12000 if n <= 256 else 8000
    if n <= 32:
        return 4000
    if n <= 128:
        return 3000
    if n <= 512:
        return 2000
    return 1500


def one_point(n: int, c: float, N: float, vs: float, z_star: float) -> dict:
    r = compute(
        n=n,
        N=N,
        vs=vs,
        z_star=z_star,
        n_samples=n_samples(n),
        seed=n * 19 + int(c * 1e4) + int(N) + int(vs * 1e6),
        alpha=2.0,
        alpha_mode="direct",
        simpson_c=c,
    )
    if r["ess"] < 20:
        r2 = compute(
            n=n,
            N=N,
            vs=vs,
            z_star=z_star,
            n_samples=n_samples(n, ess_retry=True),
            seed=n * 19 + int(c * 1e4) + int(N) + int(vs * 1e6) + 97,
            alpha=2.0,
            alpha_mode="direct",
            simpson_c=c,
        )
        if r2["ess"] > r["ess"]:
            r = r2
    return {
        "n": n,
        "eta": r["eta_maint"],
        "D": r["D_bits"],
        "C": r["mean_C_bits"],
        "ess": r["ess"],
        "F": r["F_star"],
        "alpha0": r["alpha0"],
        "log_alpha": r["log_alpha"],
        "samples": r["n_samples"],
    }


def fmt_curve(points: list[dict]) -> str:
    bits = []
    for p in points:
        flag = "" if p["ess"] >= 20 else ("~" if p["ess"] >= 8 else "!")
        bits.append(f"n={p['n']}:{flag}{p['eta']:.3f}")
    return "  ".join(bits)


def main() -> None:
    payload: dict = {"curves": [], "grid": []}

    print("=== dense even-n curves, E[S]=c fixed ===", flush=True)
    for spec in CURVES:
        points = []
        for n in NS_CURVE:
            p = one_point(n, spec["c"], spec["N"], spec["vs"], spec["z_star"])
            points.append(p)
            flag = "ok" if p["ess"] >= 20 else f"ESS={p['ess']:.1f}"
            print(
                f"  {spec['name']:22s} n={n:4d}  η={p['eta']:8.3f}  "
                f"D={p['D']:.3f}  C={p['C']:.4f}  α0={p['alpha0']:.4g}  {flag}",
                flush=True,
            )
        finite = [p for p in points if math.isfinite(p["eta"])]
        e16 = next((p["eta"] for p in finite if p["n"] == 16), float("nan"))
        e1000 = next((p["eta"] for p in finite if p["n"] == 1000), float("nan"))
        e256 = next((p["eta"] for p in finite if p["n"] == 256), float("nan"))
        print(
            f"  → {spec['name']}: {fmt_curve(points)}\n"
            f"     256/16={e256 / e16:.3f}  1000/16={e1000 / e16:.3f}  "
            f"1000/256={e1000 / e256:.3f}\n",
            flush=True,
        )
        payload["curves"].append({**spec, "points": points})

    print("=== coarse grid  n=16,64,256,1000  (Vs=0.0625, z*=0.5) ===", flush=True)
    for c, N in product((0.26, 0.3, 0.4, 0.5, 0.6, 0.75), (10.0, 20.0, 50.0, 100.0)):
        pts = [one_point(n, c, N, 0.0625, 0.5) for n in NS_GRID]
        etas = {p["n"]: p["eta"] for p in pts}
        esss = {p["n"]: p["ess"] for p in pts}
        r64 = etas[64] / etas[16] if etas[16] > 1e-9 else float("nan")
        r256 = etas[256] / etas[16] if etas[16] > 1e-9 else float("nan")
        r1000 = etas[1000] / etas[16] if etas[16] > 1e-9 else float("nan")
        r_tail = etas[1000] / etas[256] if etas[256] > 1e-9 else float("nan")
        ess_min = min(esss.values())
        print(
            f"  c={c:g} N={N:g}  η[16,64,256,1000]="
            f"{etas[16]:.3f} {etas[64]:.3f} {etas[256]:.3f} {etas[1000]:.3f}  "
            f"64/16={r64:.3f} 256/16={r256:.3f} 1000/16={r1000:.3f}  "
            f"1000/256={r_tail:.3f}  ess≥{ess_min:.1f}",
            flush=True,
        )
        payload["grid"].append(
            {
                "c": c,
                "N": N,
                "vs": 0.0625,
                "z_star": 0.5,
                "points": pts,
                "ratio_64_16": r64,
                "ratio_256_16": r256,
                "ratio_1000_16": r1000,
                "ratio_1000_256": r_tail,
                "ess_min": ess_min,
            }
        )

    tails = [g for g in payload["grid"] if g["ess_min"] >= 8]
    still_up = [g for g in tails if g["ratio_1000_16"] > 1.05]
    still_up_tail = [g for g in tails if g["ratio_1000_256"] > 1.02]
    print(
        f"\n{len(still_up)}/{len(tails)} grid cells with η(1000)/η(16)>1.05 "
        f"(ESS≥8 on all four n)",
        flush=True,
    )
    print(
        f"{len(still_up_tail)}/{len(tails)} with η(1000)/η(256)>1.02 "
        f"(still rising past n=256)",
        flush=True,
    )
    if tails:
        mean_tail = sum(g["ratio_1000_256"] for g in tails) / len(tails)
        mean_far = sum(g["ratio_1000_16"] for g in tails) / len(tails)
        print(f"mean 1000/16={mean_far:.3f}  mean 1000/256={mean_tail:.3f}", flush=True)

    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
