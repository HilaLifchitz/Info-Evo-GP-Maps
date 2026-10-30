#!/usr/bin/env python3
"""
Fitness ratio for a symmetric Sigma matrix (any n × n).

For every haploid genotype g in {0,1}^n:

    log ω(g) = gᵀ Σ g
    ω(g)     = exp(gᵀ Σ g)

Reports the highest- and lowest-fitness genotypes and the scalar

    (ω_max / ω_min) · log₂(ω_max / ω_min)

Edit SIGMA and N below, or pass on the command line.

Examples:
  python C.py
  python C.py --N 30
  python C.py --sigma '[[0.001, 0.002], [0.002, 0.005]]'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# --- edit this matrix (symmetric n × n) ------------------------------------
SIGMA = np.array([
    [1.0e-2, 2.0e-7],
    [2.0e-7, -1.0e-2]
], dtype=float)

N = 30  # population size (checks ln(ω_max/ω_min) ≤ 1/N)

def all_genotypes(n: int) -> np.ndarray:
    """All g in {0,1}^n as (2^n, n), bit order matches project convention."""
    idx = np.arange(2**n, dtype=np.uint64)[:, None]
    bits = np.arange(n - 1, -1, -1, dtype=np.uint64)
    return ((idx >> bits) & 1).astype(np.int8)


def log_fitness(G: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """log ω(g) = gᵀ Σ g for each row of G."""
    Gf = G.astype(float)
    return np.einsum("bi,ij,bj->b", Gf, sigma, Gf)


def genotype_label(g: np.ndarray) -> str:
    return "".join(str(int(b)) for b in g)


def fitness_ratio(sigma: np.ndarray, N: float) -> dict:
    sigma = np.asarray(sigma, dtype=float)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"Sigma must be square, got shape {sigma.shape}")
    if not np.allclose(sigma, sigma.T, atol=0, rtol=0):
        raise ValueError("Sigma must be symmetric")

    n = sigma.shape[0]
    G = all_genotypes(n)
    log_w = log_fitness(G, sigma)
    i_max = int(np.argmax(log_w))
    i_min = int(np.argmin(log_w))

    log_max = float(log_w[i_max])
    log_min = float(log_w[i_min])
    w_max = float(np.exp(log_max))
    w_min = float(np.exp(log_min))
    ratio = w_max / w_min
    log2_ratio = float(np.log2(ratio))
    ln_ratio = float(np.log(ratio))
    one_over_N = 1.0 / float(N)

    return {
        "n_loci": n,
        "n_genotypes": len(G),
        "best": {
            "index": i_max,
            "genotype": genotype_label(G[i_max]),
            "log_fitness": log_max,
            "fitness": w_max,
        },
        "worst": {
            "index": i_min,
            "genotype": genotype_label(G[i_min]),
            "log_fitness": log_min,
            "fitness": w_min,
        },
        "ratio": ratio,
        "ln_ratio": ln_ratio,
        "log2_ratio": log2_ratio,
        "score": ratio * log2_ratio,
        "N": float(N),
        "one_over_N": one_over_N,
        "ln_ratio_le_one_over_N": ln_ratio <= one_over_N,
    }


def _load_sigma(path_or_json: str) -> np.ndarray:
    p = Path(path_or_json)
    if p.suffix == ".npy":
        return np.load(p)
    if p.is_file():
        return np.asarray(json.loads(p.read_text()), dtype=float)
    return np.asarray(json.loads(path_or_json), dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fitness ratio ω_max / ω_min from Sigma")
    ap.add_argument(
        "--sigma", default=None,
        help="Path to .npy / .json, or JSON list literal; default: SIGMA in file",
    )
    ap.add_argument("--N", type=float, default=N, help="population size N")
    ap.add_argument("--json", action="store_true", help="Print full result as JSON")
    args = ap.parse_args()

    sigma = _load_sigma(args.sigma) if args.sigma is not None else SIGMA
    out = fitness_ratio(sigma, args.N)

    if args.json:
        print(json.dumps(out, indent=2))
        return

    b, w = out["best"], out["worst"]
    print(f"Sigma shape: {out['n_loci']} × {out['n_loci']}  ({out['n_genotypes']} genotypes)")
    print(f"N = {out['N']}")
    print(f"Sigma = \n{sigma}")
    print(f"  best  g={b['genotype']}  ω={b['fitness']:.6g}")
    print(f"  worst g={w['genotype']}  ω={w['fitness']:.6g}")
    #print(f"  ω_max / ω_min = {out['ratio']:.6g}")
    #print(f"  log₂(ω_max / ω_min) = {out['log2_ratio']:.6g}")
    print(f"bound on C(x)=  (ω_max/ω_min) · log₂(ω_max/ω_min) = {out['score']:.6g}")
    ok = out["ln_ratio_le_one_over_N"]
    print(
        f"  ln(ω_max/ω_min) ≤ 1/N : {out['ln_ratio']:.6g} ≤ {out['one_over_N']:.6g}"
        f"  → {'True' if ok else 'False'}"
    )


if __name__ == "__main__":
    main()
