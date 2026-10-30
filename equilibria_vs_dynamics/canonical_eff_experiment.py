"""Canonical-family efficiency experiment (A,B,C,D,F,G  --  E excluded).

Measures, for each Sigma-defined strategy over a fine selection-strength grid and
across the same 4 mutation/recombination clocks as the big sweep, the full set of
equilibrium + speed + COST/EFFICIENCY metrics from info_speed_experiment.measure:

  strategies (Sigma in units of a):
    A sign            (a,-a,2a)      lf [0,-a, a, 2a]   1 alluring route
    B additive        (a, a, 0)      lf [0, a, a, 2a]   2 alluring routes
    C reciprocal      (-a,-a,4a)     lf [0,-a,-a, 2a]   0 alluring (valley)
    D neutral interm. (0, 0, 2a)     lf [0, 0, 0, 2a]   0 alluring (flat)
    F add.+synergy    (a, a, a)      lf [0, a, a, 3a]   2 routes + positive epi
    G add. asymmetric (a/2,3a/2,0)   lf [0,1.5a,.5a,2a] 2 routes, one front-loaded

Outputs incrementally to results/canonical_eff_<date>/canonical_eff.csv
"""
from __future__ import annotations
import argparse, csv, time
from datetime import datetime
from pathlib import Path
import numpy as np

from node_jump_analysis import build_chain, Landscape
from adaptation_per_a import CASE_DEFS
from info_speed_experiment import measure

SCRIPT_DIR = Path(__file__).resolve().parent
CANON = ["A", "B", "C", "D", "F", "G"]            # E excluded on purpose


def _write_csv(path: Path, rows: list[dict], appended: bool) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    mode = "a" if appended else "w"
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if not appended:
            w.writeheader()
        w.writerows(rows if not appended else rows[-1:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=30)
    ap.add_argument("--a", type=float, nargs="+",
                    default=[0.05, 0.08, 0.11, 0.14, 0.18, 0.23, 0.29, 0.36])
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", type=Path,
                    default=SCRIPT_DIR / "results" /
                    f"canonical_eff_{datetime.now():%Y%m%d}")
    args = ap.parse_args()

    if args.quick:
        args.a = [0.1, 0.2]
        regimes = [dict(u=5e-3, v=5e-3, r=0.0)]
    else:
        regimes = [
            dict(u=5e-3, v=5e-3, r=0.0),     # weak-mut reversible (SSWM-ish)
            dict(u=2e-2, v=2e-2, r=0.0),     # stronger-mut reversible
            dict(u=5e-3, v=5e-3, r=0.5),     # weak-mut + recombination
            dict(u=2e-2, v=5e-3, r=0.0),     # asymmetric mutation
        ]

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "canonical_eff.csv"
    t0 = time.perf_counter()
    rows: list[dict] = []
    appended = False

    for reg in regimes:
        u, v, r = reg["u"], reg["v"], reg["r"]
        reversible = (r == 0.0 and u == v)
        tag = f"N{args.N}_u{u:g}_v{v:g}_r{r:g}"
        chain = build_chain(args.N, u, v, r, args.n_jobs)
        print(f"\n=== {tag}  S={len(chain.phi)}  (rev={reversible}) "
              f"({time.perf_counter()-t0:.1f}s) ===", flush=True)
        for slug in CANON:
            name, fn = CASE_DEFS[slug]
            for a in args.a:
                s1, s2, eps = fn(a)
                L = Landscape(f"{slug}: {name}", slug, s1=s1, s2=s2,
                              eps12=eps, blurb="")
                m, *_ = measure(chain, L, args.n_jobs, reversible)
                m.update(dict(regime=tag, u=u, v=v, r=r, case=slug, a=float(a)))
                rows.append(m)
                _write_csv(csv_path, rows, appended)
                appended = True
            print(f"    [{slug} {name}] done  ({time.perf_counter()-t0:.1f}s)",
                  flush=True)

    print(f"\nDone in {time.perf_counter()-t0:.1f}s -> {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
