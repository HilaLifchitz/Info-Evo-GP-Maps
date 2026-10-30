#!/usr/bin/env python3
"""Sweep (N, u, v, r): is cond-α a good substitute for discrete chain φ*?

φ* is the neutral stationary — it does NOT depend on s1, s2, or ε12.
This script sweeps mutation/recombination regimes used elsewhere in the repo.

References compared (see f2b_reference.py):
  f2b_b_int     — F2b-B: state-dependent 2N cond-α, interior only
  f2b_b_smeared — F2b-B interior + 1-step mutation smear (mass-matched boundary)
  f2b_b_eps     — F2b-B cond-α with x←max(x,1/N) on full lattice
  f2b_a_int     — F2b-A: fixed α at E_φ[x]
  f3_n          — F3: state-dependent N cond-α (not 2N), interior only

Run:
  cd 2loci && python f2b_phi_substitute_sweep.py
  python f2b_phi_substitute_sweep.py --quick   # fewer cases
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from compare_sigma_dynamics import _make_cfg, cond_alpha_reference_pmf  # noqa: E402
from f2b_reference import (  # noqa: E402
    build_phi_boundary_smeared,
    build_phi_f2b_a,
    build_phi_f2b_b,
    build_phi_f2b_b_epsilon,
    compare_phi_refs,
)
from simplex import (  # noqa: E402
    build_transition_matrix_row_stochastic,
    column_stochastic,
    compute_phi,
    fitness_exp,
    mutation_matrix_asymmetric,
    params_from_config,
)

OUT = SCRIPT_DIR / "plots" / "f2b"


@dataclass(frozen=True)
class Case:
    label: str
    N: int
    u: tuple[float, float]
    v: tuple[float, float]
    r: float
    note: str = ""


# Regimes mirroring eps12-dip, delta_d, compare_sigma, and boundary work
FULL_CASES: tuple[Case, ...] = (
    Case("weak_u", 30, (1e-4, 1e-4), (1e-4, 1e-4), 0.0, "KL-dip small-u regime"),
    Case("dip_u_r", 30, (1e-3, 1e-3), (1e-3, 1e-3), 1e-3, "KL-dip default u,v,r"),
    Case("dip_u_r0", 30, (1e-3, 1e-3), (1e-3, 1e-3), 0.0, "KL-dip u, no recomb"),
    Case("delta_d", 30, (0.01, 0.01), (0.01, 0.01), 0.0, "eps12_delta_d config"),
    Case("delta_d_r", 30, (0.01, 0.01), (0.01, 0.01), 1e-3, "delta_d + r=1e-3"),
    Case("moderate_u", 30, (0.05, 0.05), (0.05, 0.05), 0.0, "stronger mutation"),
    Case("large_u", 30, (0.1, 0.1), (0.1, 0.1), 0.0, "u=0.1, interior-heavy φ*"),
    Case("asym_uv", 30, (0.02, 0.01), (0.01, 0.02), 0.0, "u≠v per locus"),
    Case("high_r", 30, (1e-3, 1e-3), (1e-3, 1e-3), 0.01, "r=0.01"),
    Case("N20_weak", 20, (1e-3, 1e-3), (1e-3, 1e-3), 0.0, "smaller N"),
    Case("N20_mod", 20, (0.01, 0.01), (0.01, 0.01), 0.0, "N=20, u=0.01"),
    Case("N40_weak", 40, (1e-3, 1e-3), (1e-3, 1e-3), 0.0, "larger N"),
    Case("N40_mod", 40, (0.01, 0.01), (0.01, 0.01), 0.0, "N=40, u=0.01"),
)

QUICK_CASES: tuple[Case, ...] = (
    Case("weak_u", 30, (1e-4, 1e-4), (1e-4, 1e-4), 0.0),
    Case("dip_u_r", 30, (1e-3, 1e-3), (1e-3, 1e-3), 1e-3),
    Case("delta_d", 30, (0.01, 0.01), (0.01, 0.01), 0.0),
    Case("large_u", 30, (0.1, 0.1), (0.1, 0.1), 0.0),
    Case("asym_uv", 30, (0.02, 0.01), (0.01, 0.02), 0.0),
    Case("N40_mod", 40, (0.01, 0.01), (0.01, 0.01), 0.0),
)

REF_KEYS = (
    "f2b_b_int",
    "f2b_b_smeared",
    "f2b_b_eps",
    "f2b_a_int",
    "f3_n",
)
REF_LABELS = {
    "f2b_b_int": "F2b-B interior",
    "f2b_b_smeared": "F2b-B + smear",
    "f2b_b_eps": "F2b-B ε=1/N",
    "f2b_a_int": "F2b-A interior",
    "f3_n": "F3 (N) interior",
}


def run_case(case: Case) -> list[dict]:
    u = np.array(case.u, dtype=float)
    v = np.array(case.v, dtype=float)
    M_mut = mutation_matrix_asymmetric(u, v)
    params0 = params_from_config(
        _make_cfg(np.zeros((2, 2)), case.N, float(u[0]), float(v[0]), case.r, [0]),
        SCRIPT_DIR,
    )
    params0.u = u
    params0.v = v
    params0.r = case.r
    phi_chain, _, fracs, _, src = compute_phi(params0, M_mut)

    omega0 = fitness_exp(np.zeros((2, 2)))
    T_row, _, _ = build_transition_matrix_row_stochastic(
        case.N, omega0, M_mut, case.r, n_jobs=-1, label="neutral", verbose=False,
    )
    P_col = column_stochastic(T_row)

    q_b = build_phi_f2b_b(fracs, M_mut, case.N)
    q_a = build_phi_f2b_a(fracs, M_mut, case.N, phi_chain)
    refs_named = {
        "F2b-B interior": q_b,
        "F2b-B + smear": build_phi_boundary_smeared(fracs, phi_chain, q_b, P_col),
        "F2b-B ε=1/N": build_phi_f2b_b_epsilon(fracs, M_mut, case.N),
        "F2b-A interior": q_a,
        "F3 (N) interior": cond_alpha_reference_pmf(fracs, M_mut, case.N),
    }
    key_map = {
        "F2b-B interior": "f2b_b_int",
        "F2b-B + smear": "f2b_b_smeared",
        "F2b-B ε=1/N": "f2b_b_eps",
        "F2b-A interior": "f2b_a_int",
        "F3 (N) interior": "f3_n",
    }
    rows = compare_phi_refs(phi_chain, refs_named, fracs)
    interior = np.all(fracs > 0.0, axis=1)
    bnd_chain = float(1.0 - phi_chain[interior].sum())
    out = []
    for row in rows:
        out.append({
            "case": case.label,
            "N": case.N,
            "u0": case.u[0],
            "u1": case.u[1],
            "v0": case.v[0],
            "v1": case.v[1],
            "r": case.r,
            "note": case.note,
            "phi_source": src,
            "bnd_mass_chain": bnd_chain,
            "ref": key_map[row["name"]],
            "ref_label": row["name"],
            "tv_full": row["tv_full"],
            "kl_full_bits": row["kl_full_bits"],
            "bnd_mass_ref": row["bnd_mass_ref"],
            "tv_int": row["tv_int"],
            "kl_int_bits": row["kl_int_bits"],
            "corr_int": row["corr_int"],
        })
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_summary(rows: list[dict], path: Path) -> None:
    cases = []
    seen = set()
    for r in rows:
        if r["case"] not in seen:
            cases.append(r["case"])
            seen.add(r["case"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, len(REF_KEYS)))

    # TV interior vs case (grouped bars per ref)
    ax = axes[0, 0]
    x = np.arange(len(cases))
    w = 0.15
    for i, ref in enumerate(REF_KEYS):
        ys = []
        for c in cases:
            m = [r for r in rows if r["case"] == c and r["ref"] == ref]
            ys.append(m[0]["tv_int"] if m else np.nan)
        ax.bar(x + (i - 2) * w, ys, width=w, label=REF_LABELS[ref], color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(r"TV interior (renorm)")
    ax.set_title("Interior agreement with chain $\\varphi^*$")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # TV full — smear vs interior F2b-B
    ax = axes[0, 1]
    for ref, ls, lbl in (
        ("f2b_b_int", "-", "F2b-B int"),
        ("f2b_b_smeared", "-", "F2b-B smear"),
        ("f2b_b_eps", "--", "F2b-B ε"),
    ):
        ys = []
        for c in cases:
            m = [r for r in rows if r["case"] == c and r["ref"] == ref]
            ys.append(m[0]["tv_full"] if m else np.nan)
        ax.plot(x, ys, ls + "o", ms=4, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(r"TV full lattice")
    ax.set_title("Full-lattice TV (boundary mass matters)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # corr interior
    ax = axes[1, 0]
    for i, ref in enumerate(REF_KEYS):
        ys = []
        for c in cases:
            m = [r for r in rows if r["case"] == c and r["ref"] == ref]
            ys.append(m[0]["corr_int"] if m else np.nan)
        ax.plot(x, ys, "o-", ms=3, lw=1.2, color=colors[i], label=REF_LABELS[ref])
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("corr (interior)")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Interior correlation")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # boundary mass chain vs best smeared ref
    ax = axes[1, 1]
    bnd_c = []
    bnd_s = []
    for c in cases:
        rc = [r for r in rows if r["case"] == c]
        bnd_c.append(rc[0]["bnd_mass_chain"])
        rs = [r for r in rc if r["ref"] == "f2b_b_smeared"][0]
        bnd_s.append(rs["bnd_mass_ref"])
    ax.plot(x, bnd_c, "k-o", label="chain $\\varphi^*$")
    ax.plot(x, bnd_s, "C1--s", label="F2b-B smear ref")
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("boundary mass fraction")
    ax.set_title("Boundary mass: chain vs smeared cond-α")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Cond-α vs discrete chain $\\varphi^*$ across $(N,u,v,r)$ "
        "(selection s1,s2,ε12 do not affect $\\varphi^*$)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(rows: list[dict]) -> None:
    print("\n=== Cond-α substitute quality (lower TV / KL is better) ===\n")
    print(f"{'case':<12} {'N':>3} {'u≈':>6} {'r':>6} {'bnd':>5}  "
          f"{'ref':<14} {'TV_int':>7} {'KL_int':>8} {'corr':>6} {'TV_full':>8}")
    print("-" * 95)
    for case in dict.fromkeys(r["case"] for r in rows):
        sub = [r for r in rows if r["case"] == case]
        sub.sort(key=lambda r: r["tv_int"])
        head = sub[0]
        u_mean = 0.5 * (head["u0"] + head["u1"])
        print(f"\n{case:<12} N={head['N']} u~{u_mean:g} r={head['r']:g} "
              f"bnd={head['bnd_mass_chain']:.2f}  ({head['note']})")
        for r in sub:
            print(
                f"{'':12} {'':3} {'':6} {'':6} {'':5}  "
                f"{r['ref']:<14} {r['tv_int']:7.4f} {r['kl_int_bits']:8.4f} "
                f"{r['corr_int']:6.3f} {r['tv_full']:8.4f}",
            )

    # Best ref per case
    print("\n=== Best substitute per case (by TV_int) ===")
    for case in dict.fromkeys(r["case"] for r in rows):
        sub = [r for r in rows if r["case"] == case]
        best = min(sub, key=lambda r: r["tv_int"])
        best_full = min(sub, key=lambda r: r["tv_full"])
        print(
            f"  {case:<12} interior: {best['ref']:<14} TV_int={best['tv_int']:.4f}  "
            f"full: {best_full['ref']:<14} TV_full={best_full['tv_full']:.4f}",
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="fewer parameter cases")
    args = p.parse_args()
    cases = QUICK_CASES if args.quick else FULL_CASES

    all_rows: list[dict] = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case.label}  N={case.N} u={case.u} v={case.v} r={case.r}", flush=True)
        all_rows.extend(run_case(case))

    csv_path = OUT / "phi_substitute_sweep.csv"
    write_csv(all_rows, csv_path)
    plot_summary(all_rows, OUT / "phi_substitute_sweep.png")
    print_summary(all_rows)
    print(f"\nwrote {csv_path}")
    print(f"wrote {OUT / 'phi_substitute_sweep.png'}")


if __name__ == "__main__":
    main()
