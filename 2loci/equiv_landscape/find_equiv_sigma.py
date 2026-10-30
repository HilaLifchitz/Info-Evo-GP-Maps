#!/usr/bin/env python3
r"""
Find (s1, s2, eps12) triples with the same *relative* fitness landscape on four genotypes.

Fitness model (same as ``simplex.py``):
  Sigma = [[s1, eps12/2], [eps12/2, s2]]
  omega_g = exp(g^T Sigma g)     genotype order: 00, 01, 10, 11

Relative fitness (default: normalize to genotype 00):
  w_rel(g) = omega_g / omega_00

Log-relative scores (reference 00):
  ell_00 = 0
  ell_01 = s2
  ell_10 = s1
  ell_11 = s1 + s2 + eps12

So three ratios (vs 00) fix the shape; eps12 is the epistasis in log space.
If eps12 is perturbed, exact match of all three ratios is impossible with only
(s1, s2) free — we search for the best-fit (s1', s2') by least squares.

Usage:
  cd 2loci
  python -m equiv_landscape.find_equiv_sigma
  python -m equiv_landscape.find_equiv_sigma --s1 1e-3 --s2 -5e-4 --eps12 3e-4 \\
      --eps-new 1e-3 --scan
  python -m equiv_landscape.find_equiv_sigma --eps-delta 5e-4 --scan --scan-step 2e-4
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from simplex import GENOTYPE_LABELS, build_sigma, fitness_exp  # noqa: E402


@dataclass(frozen=True)
class Landscape:
    s1: float
    s2: float
    eps12: float
    omega: np.ndarray
    rel: np.ndarray
    log_rel: np.ndarray

    @classmethod
    def from_params(cls, s1: float, s2: float, eps12: float, *, ref: int = 0) -> Landscape:
        omega = fitness_exp(build_sigma(s1, s2, eps12))
        rel = omega / omega[ref]
        log_rel = np.log(omega) - np.log(omega[ref])
        return cls(float(s1), float(s2), float(eps12), omega, rel, log_rel)

    def signature(self) -> np.ndarray:
        """Log-relative scores for 01, 10, 11 (00 is always 0)."""
        return self.log_rel[[1, 2, 3]].copy()

    def as_dict(self) -> dict:
        d = asdict(self)
        d["omega"] = self.omega.tolist()
        d["rel"] = self.rel.tolist()
        d["log_rel"] = self.log_rel.tolist()
        d["signature"] = self.signature().tolist()
        return d


def _format_landscape(l: Landscape) -> str:
    lines = [
        f"  (s1, s2, eps12) = ({l.s1:g}, {l.s2:g}, {l.eps12:g})",
        "  genotype   omega          rel(·/00)    log-rel",
    ]
    for i, lab in enumerate(GENOTYPE_LABELS):
        lines.append(
            f"  {lab:>2}   {l.omega[i]:12.6g}   {l.rel[i]:12.6g}   {l.log_rel[i]:+.6g}"
        )
    sig = l.signature()
    lines.append(
        f"  signature (ell_01, ell_10, ell_11) = ({sig[0]:+.6g}, {sig[1]:+.6g}, {sig[2]:+.6g})"
    )
    return "\n".join(lines)


def _match_error(s1: float, s2: float, eps12: float, target_sig: np.ndarray) -> float:
    sig = Landscape.from_params(s1, s2, eps12).signature()
    return float(np.sum((sig - target_sig) ** 2))


def find_s1_s2_for_eps(
    target: Landscape,
    eps12: float,
    *,
    s1_bounds: tuple[float, float] = (-0.05, 0.05),
    s2_bounds: tuple[float, float] = (-0.05, 0.05),
) -> tuple[float, float, float, Landscape]:
    """Best (s1, s2) at fixed eps12 matching target log-relative signature."""
    from scipy.optimize import minimize

    tgt = target.signature()
    x0 = np.array([target.s1, target.s2], dtype=float)

    def obj(x: np.ndarray) -> float:
        return _match_error(float(x[0]), float(x[1]), eps12, tgt)

    res = minimize(
        obj,
        x0,
        method="L-BFGS-B",
        bounds=[s1_bounds, s2_bounds],
    )
    s1, s2 = float(res.x[0]), float(res.x[1])
    err = float(res.fun)
    return s1, s2, err, Landscape.from_params(s1, s2, eps12)


def scan_equivalent(
    target: Landscape,
    *,
    eps_values: np.ndarray,
    s1_span: float,
    s2_span: float,
    step: float,
    tol: float,
) -> list[dict]:
    """Grid search: fixed eps, scan (s1,s2) for near-exact matches."""
    tgt = target.signature()
    hits: list[dict] = []
    s1_c, s2_c = target.s1, target.s2
    s1_grid = np.arange(s1_c - s1_span, s1_c + s1_span + 0.5 * step, step)
    s2_grid = np.arange(s2_c - s2_span, s2_c + s2_span + 0.5 * step, step)

    for eps in eps_values:
        for s1 in s1_grid:
            for s2 in s2_grid:
                sig = Landscape.from_params(float(s1), float(s2), float(eps)).signature()
                err = float(np.max(np.abs(sig - tgt)))
                if err <= tol:
                    hits.append({
                        "s1": float(s1),
                        "s2": float(s2),
                        "eps12": float(eps),
                        "max_abs_log_err": err,
                        "signature": sig.tolist(),
                    })
    hits.sort(key=lambda h: (h["max_abs_log_err"], abs(h["eps12"] - target.eps12)))
    return hits


def explain_degrees_of_freedom() -> str:
    return (
        "Analytic note: with w_rel referenced to 00,\n"
        "  log w(01)/w(00) = s2,   log w(10)/w(00) = s1,   log w(11)/w(00) = s1+s2+eps12.\n"
        "Exact same shape ⇔ same (s1, s2, eps12). If you change eps12 alone, you cannot\n"
        "keep all three ratios unless s1 or s2 move; with only (s1,s2) free at fixed eps12\n"
        "you have 2 parameters for 3 constraints → best-fit (least squares).\n"
        "Special case: holding ell_01 and ell_10 fixes s2 and s1; then ell_11 fixes eps12 uniquely."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--s1", type=float, default=1.0e-3)
    p.add_argument("--s2", type=float, default=-5.0e-4)
    p.add_argument("--eps12", type=float, default=3.0e-4)
    p.add_argument("--eps-new", type=float, default=None, help="single perturbed eps12")
    p.add_argument("--eps-delta", type=float, default=5.0e-4, help="±delta for eps scan")
    p.add_argument("--n-eps", type=int, default=9, help="eps values in [eps-delta, eps+delta]")
    p.add_argument("--scan", action="store_true", help="grid scan for near matches")
    p.add_argument("--scan-step", type=float, default=2.0e-4)
    p.add_argument("--scan-span", type=float, default=2.0e-3, help="half-width for s1,s2 scan")
    p.add_argument("--tol", type=float, default=1e-6, help="max |Δlog-rel| for scan hits")
    p.add_argument("--out", type=Path, default=None, help="write JSON results")
    args = p.parse_args()

    ref = Landscape.from_params(args.s1, args.s2, args.eps12)
    print("Reference landscape")
    print(_format_landscape(ref))
    print()
    print(explain_degrees_of_freedom())
    print()

    results: dict = {"reference": ref.as_dict(), "matches": []}

    if args.eps_new is not None:
        eps_list = [float(args.eps_new)]
    else:
        lo = args.eps12 - args.eps_delta
        hi = args.eps12 + args.eps_delta
        eps_list = np.linspace(lo, hi, max(2, args.n_eps))

    print(f"Searching (s1, s2) for each eps12 in {len(eps_list)} value(s)…\n")
    for eps in eps_list:
        s1, s2, err, cand = find_s1_s2_for_eps(ref, float(eps))
        print(f"--- eps12 = {eps:+.6g}  (Δ from ref = {eps - ref.eps12:+.6g}) ---")
        print(f"  best-fit: s1={s1:+.6g}, s2={s2:+.6g}  SSE={err:.3e}")
        print(f"  max |Δlog-rel| vs ref = {np.max(np.abs(cand.signature() - ref.signature())):.3e}")
        print(_format_landscape(cand))
        print()
        results["matches"].append({
            "eps12": float(eps),
            "eps_delta": float(eps - ref.eps12),
            "s1": s1,
            "s2": s2,
            "sse": err,
            "landscape": cand.as_dict(),
        })

    if args.scan:
        print(
            f"Grid scan: eps in [{ref.eps12 - args.eps_delta:g}, {ref.eps12 + args.eps_delta:g}], "
            f"s1,s2 ±{args.scan_span:g} step {args.scan_step:g}, tol {args.tol:g}"
        )
        hits = scan_equivalent(
            ref,
            eps_values=np.linspace(
                ref.eps12 - args.eps_delta,
                ref.eps12 + args.eps_delta,
                max(2, args.n_eps),
            ),
            s1_span=args.scan_span,
            s2_span=args.scan_span,
            step=args.scan_step,
            tol=args.tol,
        )
        print(f"  {len(hits)} grid hit(s) within tolerance")
        for h in hits[:15]:
            print(
                f"    s1={h['s1']:+.4g} s2={h['s2']:+.4g} eps={h['eps12']:+.4g} "
                f"max_err={h['max_abs_log_err']:.2e}"
            )
        if len(hits) > 15:
            print(f"    … and {len(hits) - 15} more")
        results["grid_hits"] = hits

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
