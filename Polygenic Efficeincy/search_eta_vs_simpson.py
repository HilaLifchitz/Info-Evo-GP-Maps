#!/usr/bin/env python3
"""Grid search: hold expected Simpson c fixed, find η_maint rising with n."""

from __future__ import annotations

import math
from itertools import product

from eta_maint_boltzmann import compute

NS = (4, 6, 8, 12, 16)  # even n; c=0.26 already needs n≥2, n=4 is always valid
SAMPLES = {4: 4000, 6: 3000, 8: 3000, 12: 2500, 16: 2000}


def eta_curve(N: float, vs: float, z_star: float, c: float) -> list[dict] | None:
    rows = []
    for n in NS:
        try:
            r = compute(
                n=n,
                N=N,
                vs=vs,
                z_star=z_star,
                n_samples=SAMPLES[n],
                seed=n * 19 + int(c * 1e4) + int(N),
                alpha=2.0,
                alpha_mode="direct",
                simpson_c=c,
            )
        except ValueError:
            return None
        rows.append(r)
    return rows


def summarize(rows: list[dict]) -> dict | None:
    etas = []
    for r in rows:
        e = r["eta_maint"]
        if not math.isfinite(e) or r["ess"] < 8:
            return None
        etas.append(e)
    e4, e8, e16 = etas[0], etas[2], etas[4]
    ratios = [
        etas[i + 1] / etas[i] for i in range(len(etas) - 1) if etas[i] > 1e-9
    ]
    n_up = sum(1 for x in ratios if x > 1.02)
    return {
        "eta": etas,
        "ratio_16_4": e16 / e4 if e4 > 1e-9 else float("nan"),
        "ratio_8_4": e8 / e4 if e4 > 1e-9 else float("nan"),
        "n_up": n_up,
        "ess_min": min(r["ess"] for r in rows),
        "alpha4": rows[0]["alpha"],
        "alpha16": rows[-1]["alpha"],
    }


def main() -> None:
    grid = list(
        product(
            (0.26, 0.3, 0.35, 0.4, 0.5, 0.6, 0.75),
            (10.0, 20.0, 50.0, 100.0),
            (0.008, 0.02, 0.0625, 0.25, 1.0),
            (0.5, 0.25),
        )
    )
    print(f"grid {len(grid)}  n={NS}  (E[S]=c fixed)", flush=True)
    hits: list[tuple[float, str, dict]] = []
    for c, N, vs, zstar in grid:
        rows = eta_curve(N=N, vs=vs, z_star=zstar, c=c)
        tag = f"c={c:g} N={N:g} Vs={vs:g} z*={zstar:g}"
        if rows is None:
            print(f"  skip  {tag}  (invalid α)", flush=True)
            continue
        s = summarize(rows)
        if s is None:
            print(f"  skip  {tag}  (ESS/nan)", flush=True)
            continue
        etas = " ".join(f"{e:.3f}" for e in s["eta"])
        print(
            f"  {tag}  η=[{etas}]  16/4={s['ratio_16_4']:.3f}  "
            f"↑steps={s['n_up']}  ess≥{s['ess_min']:.1f}",
            flush=True,
        )
        hits.append((s["ratio_16_4"], tag, s))

    hits.sort(key=lambda t: t[0], reverse=True)
    print("\n=== top 15 by η(16)/η(4), E[S]=c fixed, ESS≥8 ===")
    for ratio, tag, s in hits[:15]:
        etas = " ".join(f"{e:.3f}" for e in s["eta"])
        print(
            f"  {ratio:6.3f}  {tag}  η[4,6,8,12,16]={etas}  "
            f"α(4)={s['alpha4']:.4g} α(16)={s['alpha16']:.4g}"
        )

    rising = [h for h in hits if h[0] > 1.05 and h[2]["n_up"] >= 2]
    print(f"\n{len(rising)} / {len(hits)} valid cells with η(16)/η(4)>1.05 and ≥2 even-n increases")
    by_c: dict[float, list[float]] = {}
    for ratio, tag, s in rising:
        c = float(tag.split()[0].split("=")[1])
        by_c.setdefault(c, []).append(ratio)
    print("mean 16/4 among rising cells, by c:")
    for c in sorted(by_c):
        xs = by_c[c]
        print(f"  c={c:g}  n={len(xs)}  mean={sum(xs)/len(xs):.3f}  max={max(xs):.3f}")


if __name__ == "__main__":
    main()
