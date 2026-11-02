#!/usr/bin/env python3
"""Search for regimes where η_maint increases with polygenicity n."""

from __future__ import annotations

import math
from itertools import product

import numpy as np

from eta_maint_boltzmann import compute

NS = (2, 3, 4, 6, 8, 12, 16)
SAMPLES = {2: 5000, 3: 4000, 4: 4000, 6: 3000, 8: 3000, 12: 2500, 16: 2000}


def eta_curve(
    N: float,
    vs: float,
    z_star: float,
    alpha0: float | None = None,
    alpha: float = 2.0,
) -> list[dict]:
    rows = []
    for n in NS:
        r = compute(
            n=n,
            N=N,
            vs=vs,
            z_star=z_star,
            n_samples=SAMPLES[n],
            seed=n * 17 + 3,
            alpha=alpha,
            alpha_mode="direct",
            alpha0=alpha0,
        )
        rows.append(r)
    return rows


def summarize(rows: list[dict]) -> dict | None:
    etas = []
    for r in rows:
        e = r["eta_maint"]
        if not math.isfinite(e) or r["ess"] < 4:
            return None
        etas.append(e)
    e2, e4, e8, e16 = etas[0], etas[2], etas[4], etas[6]
    even = [etas[i] for i, n in enumerate(NS) if n % 2 == 0]
    # skip n=2 vs n=3 odd/even jump; look at even n only
    ratios = [even[i + 1] / even[i] for i in range(len(even) - 1) if even[i] > 1e-9]
    n_up = sum(1 for x in ratios if x > 1.02)
    return {
        "eta": etas,
        "ratio_16_4": e16 / e4 if e4 > 1e-9 else float("nan"),
        "ratio_8_2": e8 / e2 if e2 > 1e-9 else float("nan"),
        "n_up_even": n_up,
        "even_mono": all(x > 1.0 for x in ratios),
        "ess_min": min(r["ess"] for r in rows),
    }


def main() -> None:
    hits: list[tuple[float, str, dict, list[dict]]] = []
    grid = list(
        product(
            ("alpha0",),
            (4.0, 16.0, 32.0, 128.0, 512.0, 2048.0),
            (8.0, 20.0, 50.0, 120.0),
            (0.004, 0.015, 0.0625, 0.25, 1.0),
            (0.5, 0.25),
        )
    )
    print(f"grid {len(grid)}  n={NS}", flush=True)
    for i, (mode, a0, N, vs, zstar) in enumerate(grid):
        rows = eta_curve(N=N, vs=vs, z_star=zstar, alpha0=a0)
        s = summarize(rows)
        tag = f"α0={a0:g} N={N:g} Vs={vs:g} z*={zstar:g}"
        if s is None:
            print(f"  skip  {tag}  (ESS/nan)", flush=True)
            continue
        etas = " ".join(f"{e:.3f}" for e in s["eta"])
        print(
            f"  {tag}  η=[{etas}]  16/4={s['ratio_16_4']:.3f}  "
            f"even↑={s['n_up_even']}  ess≥{s['ess_min']:.1f}",
            flush=True,
        )
        hits.append((s["ratio_16_4"], tag, s, rows))

    hits.sort(key=lambda t: t[0], reverse=True)
    print("\n=== top 12 by η(16)/η(4), α₀-fixed, ESS ok ===")
    for ratio, tag, s, _ in hits[:12]:
        etas = " ".join(f"{e:.3f}" for e in s["eta"])
        print(f"  {ratio:6.3f}  {tag}  η[{','.join(str(n) for n in NS)}]= {etas}")

    rising = [h for h in hits if h[2]["ratio_16_4"] > 1.05 and h[2]["n_up_even"] >= 2]
    print(f"\n{len(rising)} regimes with η(16)/η(4)>1.05 and ≥2 even-n increases")
    for ratio, tag, s, _ in rising[:20]:
        print(f"  {ratio:6.3f}  {tag}")

    # A few extra probes: very strong selection, small α0, large N
    print("\n=== extra probes ===")
    extras = [
        dict(alpha0=8.0, N=200.0, vs=0.002, z_star=0.5),
        dict(alpha0=8.0, N=200.0, vs=0.008, z_star=0.5),
        dict(alpha0=32.0, N=200.0, vs=0.008, z_star=0.5),
        dict(alpha0=4.0, N=80.0, vs=0.01, z_star=0.5),
        dict(alpha0=64.0, N=15.0, vs=0.02, z_star=0.5),
        dict(alpha0=16.0, N=50.0, vs=0.005, z_star=0.75),
        dict(alpha0=None, alpha=0.4, N=50.0, vs=0.015, z_star=0.5),
        dict(alpha0=None, alpha=0.05, N=50.0, vs=0.015, z_star=0.5),
    ]
    for kw in extras:
        rows = eta_curve(
            N=kw["N"],
            vs=kw["vs"],
            z_star=kw["z_star"],
            alpha0=kw.get("alpha0"),
            alpha=kw["alpha"] if kw.get("alpha0") is None else 2.0,
        )
        s = summarize(rows)
        tag = (
            f"α0={kw.get('alpha0')} α={kw.get('alpha')} N={kw['N']} "
            f"Vs={kw['vs']} z*={kw['z_star']}"
        )
        if s is None:
            print(f"  skip {tag}")
            continue
        etas = " ".join(f"{e:.3f}" for e in s["eta"])
        print(f"  {tag}  16/4={s['ratio_16_4']:.3f}  η={etas}  ess≥{s['ess_min']:.1f}")


if __name__ == "__main__":
    main()
