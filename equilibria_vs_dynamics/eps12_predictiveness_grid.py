#!/usr/bin/env python3
r"""
Is equilibrium information predictive of the RATE of information accumulation?

We test this across a GRID of selection backgrounds. For each fixed background
(s1, s2) we sweep the epistatic coefficient eps12 and, for every eps12, measure:

  EQUILIBRIUM INFORMATION (how "good"/informative the selected equilibrium is)
    * KL_eq  = KL(psi* || phi)   [bits]        (information content of psi*)
    * F_eq   = F_N(psi*)         (free fitness; secondary)
        psi* = the chain's own selection stationary distribution (ARPACK + fallback)

  RATE OF INFORMATION ACCUMULATION (replay the chain from psi(0)=phi)
    * init_slope = d/dt KL(psi(t)||phi)  over the first generation   [PRIMARY]
    * early_avg  = average dKL/dt over the first EARLY_WINDOW generations
    * t_half     = generations to reach 50% of this background's own KL_eq
                   (censored = NaN if not reached within max(TIMES))

Predictiveness = does the ORDERING of KL_eq across eps12 match the ordering of
the rate?  We report Spearman rank correlation per background, plus per-background
scatter (KL_eq vs rate) and eps12-aligned dip plots.

Stage A (default): N>=30, r=0, weak symmetric mutation, N*s_max << 1, all sign
combos of (s1, s2).  Stage B (SENSITIVITY=True): repeat one background while
varying u,v and turning on recombination, to see if the verdict changes.

Run:
  conda activate miso
  cd equilibria_vs_dynamics
  python eps12_predictiveness_grid.py
  python eps12_predictiveness_grid.py --out-dir results/pred_grid_myrun
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

# --- make the 2loci codebase importable ----------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TWOLOCI_DIR = REPO_ROOT / "2loci"
if str(TWOLOCI_DIR) not in sys.path:
    sys.path.insert(0, str(TWOLOCI_DIR))

from simplex import (  # noqa: E402
    build_sigma,
    build_transition_matrix_row_stochastic,
    column_stochastic,
    compute_phi,
    dynamics_series,
    fitness_exp,
    free_fitness_1,
    kl_divergence_bits,
    linkage_disequilibrium_per_state,
    mutation_matrix_asymmetric,
    params_from_config,
    try_selection_equilibrium,
    w_potential_per_state,
)


# =============================================================================
# SETTINGS
# =============================================================================
N_MARKOV: int = 30
N_JOBS: int = -1

# Baseline mutation + recombination for Stage A (weak, symmetric, no linkage relief).
U_BASE: float = 1.0e-4
V_BASE: float = 1.0e-4
R_BASE: float = 0.0

# Epistasis grid (eps12 = 2*Sigma_12). Kept small so N*s_max stays << 1.
EPS12_MIN: float = -0.02
EPS12_MAX: float = 0.02
EPS12_STEP: float = 0.004

# Dynamics sample times. TIMES[0] sets the initial slope (= 1 generation).
TIMES: list[int] = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 170, 300, 500, 800]
EARLY_WINDOW: int = 20          # generations for the "early average rate"
HALF_FRACTION: float = 0.5      # t_half target = this fraction of KL_eq


@dataclass(frozen=True)
class Background:
    label: str
    s1: float
    s2: float


# Selection backgrounds: |s| magnitudes x all sign combinations + one asymmetric.
def _make_backgrounds() -> list[Background]:
    bgs: list[Background] = []
    for m in (0.005, 0.010):
        for sg1, t1 in ((+1, "p"), (-1, "m")):
            for sg2, t2 in ((+1, "p"), (-1, "m")):
                bgs.append(Background(f"s{int(m*1000)}_{t1}{t2}", sg1 * m, sg2 * m))
    bgs.append(Background("asym_pm", 0.010, -0.003))
    return bgs


BACKGROUNDS: list[Background] = _make_backgrounds()
_BG_BY_LABEL: dict[str, Background] = {b.label: b for b in BACKGROUNDS}

# Stage B (sensitivity). Each entry overrides (u, v, r) for ONE chosen background.
SENSITIVITY: bool = False
SENS_BACKGROUND: Background = Background("s10_pm", 0.010, -0.010)
SENS_VARIANTS: list[dict] = [
    {"label": "base u=v=1e-4, r=0", "u": 1e-4, "v": 1e-4, "r": 0.0},
    {"label": "stronger mut u=v=1e-2, r=0", "u": 1e-2, "v": 1e-2, "r": 0.0},
    {"label": "asym mut u=1e-2 v=1e-4, r=0", "u": 1e-2, "v": 1e-4, "r": 0.0},
    {"label": "recomb r=0.5", "u": 1e-4, "v": 1e-4, "r": 0.5},
    {"label": "recomb r=1.0 (LE)", "u": 1e-4, "v": 1e-4, "r": 1.0},
]

# --- Coupled mutation (simplex_not_sym) ----------------------------------
# When COUPLED, the 4x4 mutation matrix adds coupled flips on top of the
# per-locus singles: ux (00<->11, both loci flip together) and vx (01<->10,
# reciprocal swap). Defaults ux=vx=COUPLED_FACTOR * u,v (per locus). Setting
# UX/VX to a float overrides those rates. The independent-mutation neutral
# cache is invalid under coupling, so phi is always recomputed here.
COUPLED: bool = False
COUPLED_FACTOR: float = 10.0
UX: float | None = None
VX: float | None = None

OUT_DIR: Path = SCRIPT_DIR / "results" / "pred_grid"
# =============================================================================


def _build_M(u: float, v: float) -> np.ndarray:
    """Independent (default) or coupled 4x4 mutation matrix for scalar u, v."""
    uu = np.array([float(u), float(u)])
    vv = np.array([float(v), float(v)])
    if COUPLED:
        from simplex_not_sym import build_mutation_matrix
        return build_mutation_matrix(uu, vv, ux=UX, vx=VX, coupled_factor=COUPLED_FACTOR)
    return mutation_matrix_asymmetric(uu, vv)


@dataclass
class EpsPoint:
    eps12: float
    kl_eq: float
    ff_eq: float
    psi_ok: bool
    init_slope_kl: float
    early_avg_kl: float
    t_half_kl: float
    init_slope_f1: float
    kl_final: float
    kl_at_20: float


def _eps12_grid() -> np.ndarray:
    n = int(round((EPS12_MAX - EPS12_MIN) / EPS12_STEP)) + 1
    grid = EPS12_MIN + EPS12_STEP * np.arange(n)
    return np.unique(np.round(grid, 10))


def _make_params(N: int, u: float, v: float, r: float, s1: float, s2: float, eps12: float):
    cfg = {
        "_config_dir": str(TWOLOCI_DIR),
        "N": N,
        "r": float(r),
        "u": [float(u), float(u)],
        "v": [float(v), float(v)],
        "sigma": build_sigma(s1, s2, float(eps12)).tolist(),
        "times": TIMES,
        "n_jobs": N_JOBS,
        "neutral_cache_dir": "cache",
        "neutral": False,
        "top_k": 5,
        "plot_dynamics": False,
        "compute_f2": False,
        "ld_top_k_times": "first_last",
        "debug_kl": False,
    }
    params = params_from_config(cfg, config_dir=TWOLOCI_DIR)
    if COUPLED:
        # The independent-mutation neutral cache does not match the coupled M.
        params.neutral_cache_file = None
    return params


def _time_to_fraction(times: np.ndarray, kl: np.ndarray, target: float) -> float:
    """First time KL(t) crosses `target`, by linear interpolation. NaN if never."""
    if not np.isfinite(target) or target <= 0:
        return float("nan")
    t = np.concatenate(([0.0], times))
    y = np.concatenate(([0.0], kl))
    for i in range(1, len(y)):
        if y[i] >= target:
            if y[i] == y[i - 1]:
                return float(t[i])
            frac = (target - y[i - 1]) / (y[i] - y[i - 1])
            return float(t[i - 1] + frac * (t[i] - t[i - 1]))
    return float("nan")  # censored: not reached within the sampled horizon


def evaluate_background(
    bg: Background,
    N: int,
    u: float,
    v: float,
    r: float,
    eps_grid: np.ndarray,
    phi: np.ndarray,
    fracs: np.ndarray,
    counts: np.ndarray,
    d_per_state: np.ndarray,
    M_mut: np.ndarray,
    times: list[int],
) -> list[EpsPoint]:
    pts: list[EpsPoint] = []
    for j, eps in enumerate(eps_grid, start=1):
        eps = float(eps)
        params = _make_params(N, u, v, r, bg.s1, bg.s2, eps)
        omega = fitness_exp(params.Sigma)
        w_per_state = w_potential_per_state(fracs, omega, M_mut, params.N)

        T_row, _, _ = build_transition_matrix_row_stochastic(
            params.N, omega, M_mut, params.r,
            n_jobs=params.n_jobs, label=f"{bg.label}_eps{eps:+.3g}", verbose=False,
        )
        P_col = column_stochastic(T_row)

        dyn = dynamics_series(
            P_col, phi, times, w_per_state, d_per_state,
            counts, fracs, params.N, params.top_k, params.ld_top_k_times, psi_star=None,
        )
        kl_t = np.asarray(dyn.kl_phi_bits, dtype=float)
        t_arr = np.asarray(dyn.times, dtype=float)
        rate_t = np.asarray(dyn.rate_times, dtype=float)
        rate_kl = np.asarray(dyn.rate_kl_bits, dtype=float)

        init_slope_kl = float(rate_kl[0])
        init_slope_f1 = float(dyn.rate_f1[0])
        early_mask = rate_t <= EARLY_WINDOW
        early_avg_kl = float(np.mean(rate_kl[early_mask])) if early_mask.any() else init_slope_kl
        idx20 = int(np.where(t_arr == EARLY_WINDOW)[0][0])
        kl_at_20 = float(kl_t[idx20])

        psi_star, _resid, ok = try_selection_equilibrium(P_col)
        if psi_star is not None:
            kl_eq = kl_divergence_bits(psi_star, phi)
            ff_eq = free_fitness_1(psi_star, w_per_state, params.N)
        else:
            kl_eq = float("nan")
            ff_eq = float("nan")

        t_half = _time_to_fraction(t_arr, kl_t, HALF_FRACTION * kl_eq)

        pts.append(EpsPoint(
            eps12=eps, kl_eq=kl_eq, ff_eq=ff_eq, psi_ok=bool(ok),
            init_slope_kl=init_slope_kl, early_avg_kl=early_avg_kl, t_half_kl=t_half,
            init_slope_f1=init_slope_f1, kl_final=float(kl_t[-1]), kl_at_20=kl_at_20,
        ))
        print(
            f"    [{j:2d}/{len(eps_grid)}] eps={eps:+.3f} | "
            f"KL_eq={kl_eq:.4f} | slope={init_slope_kl:.3e} "
            f"early={early_avg_kl:.3e} t_half={t_half if np.isfinite(t_half) else float('inf'):.1f}",
            flush=True,
        )
    return pts


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation; NaN if <3 finite pairs or no variance."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(x[m], y[m])
        return float(rho)
    except Exception:
        rx = np.argsort(np.argsort(x[m])).astype(float)
        ry = np.argsort(np.argsort(y[m])).astype(float)
        if rx.std() == 0 or ry.std() == 0:
            return float("nan")
        return float(np.corrcoef(rx, ry)[0, 1])


# =============================================================================
# Stage A: predictiveness grid over backgrounds
# =============================================================================
def run_stage_a(out_dir: Path) -> dict:
    eps_grid = _eps12_grid()
    times = [int(t) for t in TIMES]
    M_mut = _build_M(U_BASE, V_BASE)

    print("=" * 72)
    print("eps12_predictiveness_grid.py  —  Stage A")
    print(f"  mutation model: {'COUPLED (ux/vx)' if COUPLED else 'independent per-locus'}")
    if COUPLED:
        print(f"  coupled_factor={COUPLED_FACTOR}, ux={UX}, vx={VX}")
    print(f"  N={N_MARKOV}, u=v={U_BASE}, r={R_BASE}")
    print(f"  eps12 grid: {len(eps_grid)} values [{eps_grid[0]:+g}, {eps_grid[-1]:+g}] step {EPS12_STEP}")
    print(f"  backgrounds ({len(BACKGROUNDS)}): {[b.label for b in BACKGROUNDS]}")
    print("=" * 72, flush=True)

    # phi depends only on (N, u, v, r): compute once.
    params0 = _make_params(N_MARKOV, U_BASE, V_BASE, R_BASE, 0.0, 0.0, 0.0)
    phi, _, fracs, counts, src = compute_phi(params0, M_mut)
    d_per_state = linkage_disequilibrium_per_state(fracs)
    print(f"  phi source: {src}\n", flush=True)

    results: dict[str, list[EpsPoint]] = {}
    for bg in BACKGROUNDS:
        print(f"--- background {bg.label}: s1={bg.s1:+g}, s2={bg.s2:+g} ---", flush=True)
        results[bg.label] = evaluate_background(
            bg, N_MARKOV, U_BASE, V_BASE, R_BASE,
            eps_grid, phi, fracs, counts, d_per_state, M_mut, times,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "stage_a.csv", results)
    summary = _summary_table(results)
    _write_summary(out_dir / "stage_a_spearman.csv", summary)
    plot_scatter_grid(results, summary, out_dir / "stage_a_scatter.png")
    plot_dip_grid(results, out_dir / "stage_a_dip.png")
    plot_spearman_bars(summary, out_dir / "stage_a_spearman.png")
    return {"results": results, "summary": summary}


def _summary_table(results: dict[str, list[EpsPoint]]) -> list[dict]:
    rows = []
    for label, pts in results.items():
        eps = np.array([p.eps12 for p in pts])
        kl_eq = np.array([p.kl_eq for p in pts])
        ff_eq = np.array([p.ff_eq for p in pts])
        slope = np.array([p.init_slope_kl for p in pts])
        early = np.array([p.early_avg_kl for p in pts])
        kl20 = np.array([p.kl_at_20 for p in pts])
        slope020 = kl20 / float(EARLY_WINDOW)
        thalf = np.array([p.t_half_kl for p in pts])
        rows.append({
            "background": label,
            "rho_KLeq_slope": _spearman(kl_eq, slope),
            "rho_KLeq_early": _spearman(kl_eq, early),
            "rho_KLeq_kl20": _spearman(kl_eq, kl20),
            "rho_KLeq_slope020": _spearman(kl_eq, slope020),
            # faster accumulation = SMALLER t_half, so negate for "info predicts speed"
            "rho_KLeq_invThalf": _spearman(kl_eq, -thalf),
            "rho_FFeq_slope": _spearman(ff_eq, slope),
            "eps_at_min_KLeq": float(eps[np.nanargmin(kl_eq)]) if np.isfinite(kl_eq).any() else float("nan"),
            "eps_at_min_slope": float(eps[np.nanargmin(slope)]) if np.isfinite(slope).any() else float("nan"),
        })
    return rows


def _write_csv(path: Path, results: dict[str, list[EpsPoint]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["background", "eps12", "kl_eq", "ff_eq", "psi_ok",
                    "init_slope_kl", "early_avg_kl", "t_half_kl",
                    "init_slope_f1", "kl_final", "kl_at_20"])
        for label, pts in results.items():
            for p in pts:
                w.writerow([label, p.eps12, p.kl_eq, p.ff_eq, int(p.psi_ok),
                            p.init_slope_kl, p.early_avg_kl, p.t_half_kl,
                            p.init_slope_f1, p.kl_final, p.kl_at_20])
    print(f"  wrote {path}", flush=True)


def _load_results_csv(path: Path) -> dict[str, list[EpsPoint]]:
    by_bg: dict[str, list[EpsPoint]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            label = row["background"]
            by_bg.setdefault(label, []).append(EpsPoint(
                eps12=float(row["eps12"]),
                kl_eq=float(row["kl_eq"]),
                ff_eq=float(row["ff_eq"]),
                psi_ok=bool(int(row["psi_ok"])),
                init_slope_kl=float(row["init_slope_kl"]),
                early_avg_kl=float(row["early_avg_kl"]),
                t_half_kl=float(row["t_half_kl"]) if row["t_half_kl"] not in ("", "nan") else float("nan"),
                init_slope_f1=float(row["init_slope_f1"]),
                kl_final=float(row["kl_final"]),
                kl_at_20=float(row.get("kl_at_20") or 0.0),
            ))
    return by_bg


def _write_summary(path: Path, summary: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        for row in summary:
            w.writerow(row)
    print(f"  wrote {path}", flush=True)
    print("\n  Spearman(KL_eq, rate metrics) per background:")
    for row in summary:
        print(f"    {row['background']:10s}  rho(slope)={row['rho_KLeq_slope']:+.3f}  "
              f"rho(early)={row['rho_KLeq_early']:+.3f}  "
              f"rho(KL20)={row['rho_KLeq_kl20']:+.3f}  "
              f"rho(slope020)={row['rho_KLeq_slope020']:+.3f}")


# =============================================================================
# Plots
# =============================================================================
def _grid_shape(n: int) -> tuple[int, int]:
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    return nrow, ncol


def _dip_panel_title(label: str) -> str:
    """Readable dip-panel title, e.g. s5_pm -> 's1=5, s2=5 (+,-)'."""
    bg = _BG_BY_LABEL.get(label)
    if bg is not None:
        s1_mag = int(round(abs(bg.s1) * 1000))
        s2_mag = int(round(abs(bg.s2) * 1000))
        s1_sign = "+" if bg.s1 >= 0 else "-"
        s2_sign = "+" if bg.s2 >= 0 else "-"
        return f"s1={s1_mag}, s2={s2_mag} ({s1_sign},{s2_sign})"
    return label


def _is_redundant_dip_background(label: str) -> bool:
    """Drop redundant sign mirrors when |s1|=|s2| (pm/mp and pp/mm locus swaps)."""
    bg = _BG_BY_LABEL.get(label)
    if bg is None or abs(bg.s1) != abs(bg.s2):
        return False
    if label.endswith("_mp"):
        return f"{label[:-3]}_pm" in _BG_BY_LABEL
    if label.endswith("_mm"):
        return f"{label[:-3]}_pp" in _BG_BY_LABEL
    return False


def _dip_plot_labels(results: dict[str, list[EpsPoint]], *, filter_symmetric: bool) -> list[str]:
    if any(label in _BG_BY_LABEL for label in results):
        labels = [b.label for b in BACKGROUNDS if b.label in results]
    else:
        labels = list(results.keys())
    if filter_symmetric:
        labels = [label for label in labels if not _is_redundant_dip_background(label)]
    return labels


def plot_scatter_grid(results, summary, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    nrow, ncol = _grid_shape(len(labels))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.2 * nrow), squeeze=False)
    rho_by_label = {row["background"]: row for row in summary}
    for k, label in enumerate(labels):
        ax = axes[k // ncol][k % ncol]
        pts = results[label]
        kl_eq = np.array([p.kl_eq for p in pts])
        slope = np.array([p.init_slope_kl for p in pts])
        eps = np.array([p.eps12 for p in pts])
        sc = ax.scatter(kl_eq, slope, c=eps, cmap="coolwarm", s=42, edgecolor="k", linewidth=0.4)
        rho = rho_by_label[label]["rho_KLeq_slope"]
        ax.set_title(f"{label}  (rho={rho:+.2f})", fontsize=9)
        ax.set_xlabel(r"$KL_{eq}(\psi^*\|\varphi)$ [bits]", fontsize=8)
        ax.set_ylabel(r"initial $d\,KL/dt$", fontsize=8)
        ax.tick_params(labelsize=7)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r"$\varepsilon_{12}$", fontsize=7)
        cb.ax.tick_params(labelsize=6)
    for k in range(len(labels), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle("Equilibrium information vs initial accumulation rate (per background)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def plot_dip_grid(results, path: Path, *, filter_symmetric: bool = True) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = _dip_plot_labels(results, filter_symmetric=filter_symmetric)
    nrow, ncol = _grid_shape(len(labels))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.0 * nrow), squeeze=False)
    for k, label in enumerate(labels):
        ax = axes[k // ncol][k % ncol]
        pts = results[label]
        eps = np.array([p.eps12 for p in pts])
        kl_eq = np.array([p.kl_eq for p in pts])
        slope = np.array([p.init_slope_kl for p in pts])
        ax.plot(eps, kl_eq, "o-", color="C0", lw=1.6, ms=4, label=r"$KL_{eq}$")
        ax.set_xlabel(r"$\varepsilon_{12}$", fontsize=8)
        ax.set_ylabel(r"$KL_{eq}$ [bits]", color="C0", fontsize=8)
        ax.tick_params(axis="y", labelcolor="C0", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax2 = ax.twinx()
        ax2.plot(eps, slope, "s--", color="C3", lw=1.4, ms=3.5, label="init rate")
        ax2.set_ylabel(r"init $d\,KL/dt$", color="C3", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="C3", labelsize=7)
        ax.set_title(_dip_panel_title(label), fontsize=9)
    for k in range(len(labels), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(r"$KL_{eq}$ and initial rate vs $\varepsilon_{12}$ (do the dips align?)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def plot_spearman_bars(summary, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [row["background"] for row in summary]
    rho_slope = [row["rho_KLeq_slope"] for row in summary]
    rho_early = [row["rho_KLeq_early"] for row in summary]
    rho_kl20 = [row["rho_KLeq_kl20"] for row in summary]
    rho_slope020 = [row["rho_KLeq_slope020"] for row in summary]
    x = np.arange(len(labels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(1.2 * len(labels) + 3, 4.5))
    ax.bar(x - 1.5 * w, rho_slope, w, label="vs init slope", color="C0")
    ax.bar(x - 0.5 * w, rho_early, w, label="vs early-avg rate", color="C2")
    ax.bar(x + 0.5 * w, rho_kl20, w, label=rf"vs $KL({EARLY_WINDOW})$", color="C1")
    ax.bar(x + 1.5 * w, rho_slope020, w, label=rf"vs $KL({EARLY_WINDOW})/{EARLY_WINDOW}$", color="C4")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(r"Spearman $\rho$ ($KL_{eq}$ vs rate)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Is equilibrium information predictive of accumulation rate?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


# =============================================================================
# Stage B: sensitivity to u,v and recombination (one background)
# =============================================================================
def run_stage_b(out_dir: Path) -> dict:
    eps_grid = _eps12_grid()
    times = [int(t) for t in TIMES]
    bg = SENS_BACKGROUND
    print("\n" + "=" * 72)
    print(f"Stage B — sensitivity, background {bg.label} (s1={bg.s1:+g}, s2={bg.s2:+g})")
    print("=" * 72, flush=True)

    variant_results: dict[str, list[EpsPoint]] = {}
    for var in SENS_VARIANTS:
        u, v, r = float(var["u"]), float(var["v"]), float(var["r"])
        print(f"--- variant: {var['label']} ---", flush=True)
        M_mut = _build_M(u, v)
        params0 = _make_params(N_MARKOV, u, v, r, 0.0, 0.0, 0.0)
        phi, _, fracs, counts, _src = compute_phi(params0, M_mut)
        d_per_state = linkage_disequilibrium_per_state(fracs)
        variant_results[var["label"]] = evaluate_background(
            bg, N_MARKOV, u, v, r, eps_grid, phi, fracs, counts, d_per_state, M_mut, times,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "stage_b.csv", variant_results)
    summary = _summary_table(variant_results)
    _write_summary(out_dir / "stage_b_spearman.csv", summary)
    plot_dip_grid(variant_results, out_dir / "stage_b_dip.png", filter_symmetric=False)
    plot_spearman_bars(summary, out_dir / "stage_b_spearman.png")
    return {"results": variant_results, "summary": summary}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--coupled", action="store_true",
                    help="use coupled mutation matrix (simplex_not_sym: ux 00<->11, vx 01<->10)")
    ap.add_argument("--coupled-factor", type=float, default=10.0,
                    help="default ux,vx = factor * u,v when --ux/--vx not given")
    ap.add_argument("--ux", type=float, default=None, help="coupled 00<->11 rate (overrides factor)")
    ap.add_argument("--vx", type=float, default=None, help="coupled 01<->10 rate (overrides factor)")
    ap.add_argument("--replot-dip", action="store_true",
                    help="replot stage_a_dip.png from existing stage_a.csv (no recompute)")
    args = ap.parse_args()

    global COUPLED, COUPLED_FACTOR, UX, VX
    COUPLED = bool(args.coupled)
    COUPLED_FACTOR = float(args.coupled_factor)
    UX = args.ux
    VX = args.vx

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {out_dir}")
    print(f"started: {datetime.now().isoformat(timespec='seconds')}\n", flush=True)

    if args.replot_dip:
        csv_path = out_dir / "stage_a.csv"
        results = _load_results_csv(csv_path)
        plot_dip_grid(results, out_dir / "stage_a_dip.png")
        print(f"\ndone: {datetime.now().isoformat(timespec='seconds')}", flush=True)
        return

    run_stage_a(out_dir)
    if SENSITIVITY or args.sensitivity:
        run_stage_b(out_dir / "sensitivity")

    print(f"\ndone: {datetime.now().isoformat(timespec='seconds')}", flush=True)


if __name__ == "__main__":
    main()
