#!/usr/bin/env python3
r"""
Is the EQUILIBRIUM information score predictive of the SPEED of information
accumulation?  (decoupling info / free-fitness / epistasis-decomposition)
======================================================================

For a fixed "clock" (N, u, v, r) we build the neutral chain once, then for a
large family of selection landscapes (s1, s2, eps12) we measure, on the SAME
clock so timescales are comparable:

  EQUILIBRIUM (magnitude) quantities
    D_eq   = KL(psi* || phi)                 information at equilibrium [bits]
    mlf    = <sum_g x_g log w_g>_{psi*}       mean log-fitness at equilibrium
    Fstar  = <log Wbar>_{psi*} - D_eq_nats/N  free fitness (nats)  [logB excluded]

  SPEED (magnitude-normalised) quantities -- pure "how fast", not "how far"
    t_half_info  : gens for D(t)=KL(psi(t)||phi) to reach 0.5 * D_eq   (start phi)
    t95_info     : ... to reach 0.95 * D_eq
    t_half_relax : gens for R(t)=KL(psi(t)||psi*) to reach 0.5 * R(0)
    gap          : spectral gap 1 - |lambda_2(P)|   (intrinsic relaxation rate)
    tau_spec     : -1/ln|lambda_2|

  EPISTASIS descriptors (from the log-fitness lf=[0, s2, s1, s1+s2+eps])
    n_mono   : # monotone-uphill routes 00->11 (0,1,2)
    barrier  : best route's worst single uphill step (higher = more accessible)
    valley   : -min(lf[01], lf[10])  (depth intermediates dip below 00)
    margin   : lf[11] - max(other lf)  (how dominant the 11 corner is)
    peak     : argmax lf   (which genotype wins)

At r=0 the selected chain is reversible, so psi(t) is reconstructed exactly from
one eigendecomposition of P (no long iteration); for r>0 (non-reversible) we fall
back to direct iteration.  All raw curves + scalars are saved for re-plotting.

Run:
  conda activate miso
  cd equilibria_vs_dynamics
  python info_speed_experiment.py --quick          # fast smoke test
  python info_speed_experiment.py                  # full sweep (background)
"""
from __future__ import annotations

import argparse
import itertools
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from node_jump_analysis import (
    Chain, Landscape, build_chain, evolve_many, selection_P,
)
from adaptation_per_a import CASE_DEFS, CASE_ORDER, all_landscapes, cost_per_state
from compare_two_cases import free_fitness5_asymptote
from simplex import (
    build_sigma, fitness_exp, kl_divergence_bits,
    log_mean_fitness_genotype_simplex, mutation_matrix_asymmetric, stationary_fast,
)

SCRIPT_DIR = Path(__file__).resolve().parent
GENO = ("00", "01", "10", "11")


# --------------------------------------------------------------------------
# trajectory: exact psi(t) via eigendecomposition (reversible r=0) with
# iterative fallback
# --------------------------------------------------------------------------
def eig_trajectory(P: np.ndarray, phi: np.ndarray):
    """Return (psi_at, lam2, recon_err).  psi_at(t) -> pmf via spectral sum."""
    w, V = np.linalg.eig(P)
    c = np.linalg.solve(V, phi.astype(complex))
    order = np.argsort(-np.abs(w))
    lam2 = float(np.abs(w[order[1]]))

    def psi_at(t: float) -> np.ndarray:
        vals = (w ** t) * c
        x = np.real(V @ vals)
        x = np.clip(x, 0.0, None)
        s = x.sum()
        return x / s if s > 0 else x

    recon_err = float(np.abs(psi_at(1.0) - (P @ phi)).max())
    return psi_at, lam2, recon_err


def iter_trajectory(P: np.ndarray, phi: np.ndarray, times: np.ndarray):
    """Fallback: exact iteration, snapshot at requested (integer) times."""
    ti = np.unique(np.round(times).astype(int))
    snaps = evolve_many(P, phi.reshape(-1, 1), ti)[:, :, 0]   # (len(ti), S)
    lookup = {int(t): k for k, t in enumerate(ti)}

    def psi_at(t: float) -> np.ndarray:
        return snaps[lookup[int(round(t))]]

    return psi_at, ti


def spectral_gap(P: np.ndarray) -> float:
    """Second-largest |eigenvalue| of P via Arnoldi (cheap, N=30-friendly)."""
    try:
        from scipy.sparse.linalg import eigs
        w = eigs(P, k=2, which="LM", return_eigenvectors=False,
                 maxiter=5000, tol=1e-9)
        return float(np.sort(np.abs(w))[-2])
    except Exception:
        w = np.linalg.eigvals(P)
        return float(np.sort(np.abs(w))[-2])


def first_crossing(times: np.ndarray, y: np.ndarray, level: float) -> float:
    """First t where y crosses `level` (linear interp in t).  NaN if never."""
    above = y >= level
    if not above.any():
        return float("nan")
    k = int(np.argmax(above))
    if k == 0:
        return float(times[0])
    t0, t1, y0, y1 = times[k - 1], times[k], y[k - 1], y[k]
    if y1 == y0:
        return float(t1)
    return float(t0 + (level - y0) * (t1 - t0) / (y1 - y0))


# --------------------------------------------------------------------------
# epistasis descriptors
# --------------------------------------------------------------------------
def descriptors(s1: float, s2: float, eps: float) -> dict:
    lf = np.array([0.0, s2, s1, s1 + s2 + eps])
    step_A = (s1, s2 + eps)      # 00->10->11
    step_B = (s2, s1 + eps)      # 00->01->11
    mono_A = step_A[0] > 0 and step_A[1] > 0
    mono_B = step_B[0] > 0 and step_B[1] > 0
    barrier = max(min(step_A), min(step_B))     # best route's worst step
    valley = -min(lf[1], lf[2])
    margin = lf[3] - max(lf[0], lf[1], lf[2])
    return dict(
        lf00=lf[0], lf01=lf[1], lf10=lf[2], lf11=lf[3],
        n_mono=int(mono_A) + int(mono_B),
        barrier=barrier, valley=valley, margin=margin,
        peak=int(np.argmax(lf)), scale=float(np.sqrt((lf ** 2).sum())),
    )


# --------------------------------------------------------------------------
# per-landscape metrics
# --------------------------------------------------------------------------
def measure(chain: Chain, L: Landscape, n_jobs: int, reversible: bool) -> dict:
    P = selection_P(chain, L, n_jobs)
    psi_star = stationary_fast(P)
    phi = chain.phi
    N = chain.N

    omega = fitness_exp(build_sigma(L.s1, L.s2, L.eps12))
    logmeanfit = log_mean_fitness_genotype_simplex(chain.fracs, omega)
    mlf = float(psi_star @ logmeanfit)
    D_eq = kl_divergence_bits(psi_star, phi)               # bits
    R0 = kl_divergence_bits(phi, psi_star)                 # bits
    M_mut = mutation_matrix_asymmetric(np.array([chain.u, chain.u]),
                                       np.array([chain.v, chain.v]))
    Fstar, _ = free_fitness5_asymptote(psi_star, phi, chain.fracs, omega, M_mut, N)

    # Trajectory psi(t).  For small state spaces the exact eigendecomposition
    # reconstruction is fastest; for large ones (N>=~26) O(S^3) eig is too slow,
    # so we get the spectral gap via Arnoldi (eigs) and iterate for the curve.
    HORIZONS = (20, 50, 100, 200)
    S = P.shape[0]
    if S <= 2500:
        psi_at, lam2, recon_err = eig_trajectory(P, phi)
        used_iter = not (recon_err < 1e-3)
    else:
        lam2 = spectral_gap(P)
        recon_err = float("nan")
        used_iter = True

    tau = (-1.0 / np.log(lam2)) if (lam2 == lam2 and 0 < lam2 < 1) else 200.0

    if not used_iter:
        tmax = int(np.clip(np.ceil(10 * tau), 300, 2_000_000))
        tgrid = np.unique(np.concatenate([
            np.arange(0, 201, dtype=float),
            np.geomspace(1.0, float(tmax), 400),
        ]))
        D = np.array([kl_divergence_bits(psi_at(t), phi) for t in tgrid])
        R = np.array([kl_divergence_bits(psi_at(t), psi_star) for t in tgrid])
    else:
        tmax = int(np.clip(np.ceil(8 * tau), 300, 60000))
        tgrid_req = np.unique(np.concatenate([
            np.arange(0, 201, dtype=float),
            np.geomspace(1.0, float(tmax), 300),
        ]))
        psi_at, ti = iter_trajectory(P, phi, tgrid_req)
        tgrid = ti.astype(float)
        D = np.array([kl_divergence_bits(psi_at(t), phi) for t in tgrid])
        R = np.array([kl_divergence_bits(psi_at(t), psi_star) for t in tgrid])

    # per-generation selective COST C(t)=<C(x)>_{psi(t)} [bits] and its cumulative
    # sum, for the efficiency eta = info gained / (N * cost paid).
    c_state = cost_per_state(chain.fracs, omega)           # bits per state
    C = np.array([float(psi_at(t) @ c_state) for t in tgrid])

    t_half_info = first_crossing(tgrid, D, 0.5 * D_eq) if D_eq > 1e-9 else float("nan")
    t95_info = first_crossing(tgrid, D, 0.95 * D_eq) if D_eq > 1e-9 else float("nan")
    t_half_relax = first_crossing(tgrid, -(R - R0), 0.5 * R0) if R0 > 1e-9 else float("nan")

    # horizon-based accumulation rates + efficiency (robust; no equilibration).
    # Integer times 0..200 are all present in tgrid, so cumulative sums are exact.
    def _idx(t: int) -> int:
        return int(np.argmin(np.abs(tgrid - t)))
    D_int = np.array([D[_idx(t)] for t in range(0, 201)])  # D at t=0..200
    C_int = np.array([C[_idx(t)] for t in range(0, 201)])  # C at t=0..200
    init_rate = float(D_int[1] - D_int[0])                 # bits/gen
    init_eff = (init_rate / (float(N) * C_int[1])) if C_int[1] > 1e-12 else float("nan")
    horizon = {}
    for H in HORIZONS:
        dH = float(D_int[H])
        cumC = float(C_int[1:H + 1].sum())                 # total cost paid t=1..H
        horizon[f"kl_at_{H}"] = dH
        horizon[f"avg_rate_{H}"] = dH / float(H)
        horizon[f"cost_{H}"] = cumC                        # cumulative cost [bits]
        horizon[f"Cbar_{H}"] = cumC / float(H)             # mean per-gen cost
        horizon[f"eta_{H}"] = (dH / (float(N) * cumC)) if cumC > 1e-12 else float("nan")
    # peak instantaneous efficiency over the dense early window (t=1..200)
    with np.errstate(divide="ignore", invalid="ignore"):
        dDdt = np.diff(D_int)                              # bits/gen at t=1..200
        eta_t = np.where(C_int[1:] > 1e-12, dDdt / (float(N) * C_int[1:]), np.nan)
    horizon["eta_peak"] = float(np.nanmax(eta_t)) if np.isfinite(eta_t).any() else float("nan")

    gap = (1.0 - lam2) if lam2 == lam2 else float("nan")
    tau_spec = (-1.0 / np.log(lam2)) if (lam2 == lam2 and 0 < lam2 < 1) else float("nan")

    star_basin = [float(psi_star[chain.basin == g].sum()) for g in range(4)]
    out = dict(
        s1=L.s1, s2=L.s2, eps=L.eps12,
        D_eq=D_eq, R0=R0, mlf=mlf, Fstar=Fstar,
        t_half_info=t_half_info, t95_info=t95_info, t_half_relax=t_half_relax,
        init_rate=init_rate, init_eff=init_eff, gap=gap, tau_spec=tau_spec,
        recon_err=recon_err, used_iter=int(used_iter), star_11=star_basin[3],
    )
    out.update(horizon)
    out.update(descriptors(L.s1, L.s2, L.eps12))
    return out, tgrid, D, R, D_eq


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def selection_grid(quick: bool):
    if quick:
        s1s = np.linspace(-0.12, 0.12, 3)
        s2s = np.linspace(-0.12, 0.12, 3)
        epss = np.linspace(-0.24, 0.24, 3)
    else:
        s1s = np.linspace(-0.15, 0.15, 5)
        s2s = np.linspace(-0.15, 0.15, 5)
        epss = np.linspace(-0.30, 0.30, 7)
    return s1s, s2s, epss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--a-list", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out-dir", type=Path,
                    default=SCRIPT_DIR / "results" /
                    f"info_speed_{datetime.now():%Y%m%d}")
    args = ap.parse_args()

    # (u, v, r) regimes -- each is a separate "clock"
    if args.quick:
        regimes = [dict(u=5e-3, v=5e-3, r=0.0)]
    else:
        regimes = [
            dict(u=5e-3, v=5e-3, r=0.0),     # weak-mut reversible (SSWM-ish)
            dict(u=2e-2, v=2e-2, r=0.0),     # stronger-mut reversible
            dict(u=5e-3, v=5e-3, r=0.5),     # weak-mut + recombination (non-rev)
            dict(u=2e-2, v=5e-3, r=0.0),     # asymmetric mutation (reversible? no)
        ]

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    s1s, s2s, epss = selection_grid(args.quick)
    rows: list[dict] = []
    curves: dict[str, np.ndarray] = {}

    csv_path = out / "info_speed.csv"
    header_written = False

    for ri, reg in enumerate(regimes):
        u, v, r = reg["u"], reg["v"], reg["r"]
        reversible = (r == 0.0 and u == v)
        tag = f"N{args.N}_u{u:g}_v{v:g}_r{r:g}"
        print(f"\n=== regime {ri}: {tag}  (reversible={reversible}) ===", flush=True)
        chain = build_chain(args.N, u, v, r, args.n_jobs)
        S = len(chain.phi)
        print(f"    S={S} states, phi corner mass="
              f"{chain.phi[chain.cls==1].sum():.4f}  ({time.perf_counter()-t0:.1f}s)",
              flush=True)

        # ---- Part 1: the A-E family at a few a ----
        for a in args.a_list:
            for L in all_landscapes(a):
                m, tg, D, R, D_eq = measure(chain, L, args.n_jobs, reversible)
                m.update(dict(regime=tag, u=u, v=v, r=r, family="AE",
                              case=L.slug, a=a))
                rows.append(m)
                key = f"{tag}__AE_{L.slug}_a{a:g}"
                curves[key + "__t"] = tg
                curves[key + "__D"] = D
                curves[key + "__R"] = R
            print(f"    [AE a={a:g}] done  ({time.perf_counter()-t0:.1f}s)", flush=True)

        # ---- Part 2: dense (s1,s2,eps) grid ----
        combos = list(itertools.product(s1s, s2s, epss))
        for ci, (s1, s2, eps) in enumerate(combos):
            L = Landscape(f"grid s1={s1:.3f} s2={s2:.3f} eps={eps:.3f}",
                          f"g{ci}", s1=float(s1), s2=float(s2), eps12=float(eps),
                          blurb="")
            m, tg, D, R, D_eq = measure(chain, L, args.n_jobs, reversible)
            m.update(dict(regime=tag, u=u, v=v, r=r, family="grid",
                          case=f"g{ci}", a=float("nan")))
            rows.append(m)
            if ci % 25 == 0:
                key = f"{tag}__grid_{ci}"
                curves[key + "__t"] = tg
                curves[key + "__D"] = D
                curves[key + "__R"] = R
            if (ci + 1) % 20 == 0:
                print(f"    [grid {ci+1}/{len(combos)}]  "
                      f"({time.perf_counter()-t0:.1f}s)", flush=True)

        # incremental CSV write
        _write_csv(csv_path, rows, header_written)
        header_written = True
        print(f"    wrote {csv_path} ({len(rows)} rows)", flush=True)

    np.savez_compressed(out / "curves.npz", **curves)
    print(f"\nDone in {time.perf_counter()-t0:.1f}s. rows={len(rows)}")
    print(f"  CSV:    {csv_path}")
    print(f"  curves: {out / 'curves.npz'}")


def _write_csv(path: Path, rows: list[dict], _appended: bool) -> None:
    import csv
    if not rows:
        return
    keys = list(rows[0].keys())
    # unify keys across rows
    for rr in rows:
        for k in rr:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for rr in rows:
            w.writerow(rr)


if __name__ == "__main__":
    main()
