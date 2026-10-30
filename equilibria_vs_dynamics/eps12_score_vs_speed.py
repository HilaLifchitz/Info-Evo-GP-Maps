#!/usr/bin/env python3
r"""
Does the equilibrium "score" of an epistatic coefficient predict adaptation speed?

For fixed s1, s2 we sweep the epistatic coefficient eps12 (= 2*Sigma_12) and, for
each value, measure two things:

  (A) EQUILIBRIUM SCORES  — how "good" the equilibrium is, from two sources:
      * diffusion : haploid allele-frequency diffusion equilibrium (epistasis_equilibria/)
                    -> KL(rho_eq || rho_neutral) [bits],  free fitness F_N
      * discrete  : the N=30 genotype-simplex Markov chain's own selection
                    equilibrium psi*  (2loci/)
                    -> KL(psi* || phi) [bits],  free fitness F_N(psi*)

  (B) ADAPTATION SPEED    — initial rate of accumulation when replaying the
      discrete Markov-chain dynamics from the neutral start psi(0)=phi:
        init_rate_kl = d/dt KL(psi(t)||phi)   over the first generation
        init_rate_f1 = d/dt F_N(psi(t))       over the first generation

We then test, separately for each recombination rate r, whether the *ordering*
of equilibrium scores across eps12 matches the ordering of adaptation speed
(Spearman rank correlation).

Reaching both codebases
-----------------------
This script lives in its own folder and puts 2loci/ and epistasis_equilibria/
on sys.path, so nothing has to be copied between folders.

Run:
  python eps12_score_vs_speed.py
  python eps12_score_vs_speed.py --out-dir results/my_run
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np

# --- make both sibling codebases importable -------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
TWOLOCI_DIR = REPO_ROOT / "2loci"
EPI_DIR = REPO_ROOT / "epistasis_equilibria"
for _d in (TWOLOCI_DIR, EPI_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

# from 2loci/simplex.py  (discrete Markov chain + dynamics)
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

# from epistasis_equilibria/  (continuous haploid diffusion equilibrium)
from equilibria_allele_freq import (  # noqa: E402
    _as_float_vec,
    all_genotypes_haploid,
    log_fitness_haploid,
    log_mean_fitness_batch,
)
from haploid_allele_grid_metrics import tensor_grid_centers_haploid  # noqa: E402


# =============================================================================
# SETTINGS — edit these to define a comparison run
# =============================================================================
# Fixed diagonal selection (keep SMALL: N is small, weak-selection regime).
S1: float = -0.02
S2: float = -0.01

# Recombination rates to test (one curve / Spearman score per r).
# Parked at r=0 for now; treating BOTH models as haploid (see notes in module docstring).
R_VALUES: list[float] = [0.0]

# Which equilibrium scores to compute.
#   discrete : the N=30 Markov chain's own selection equilibrium psi*  (fast solver)
#   diffusion: the continuous HAPLOID allele-frequency equilibrium (matched to the chain)
COMPUTE_DISCRETE: bool = True
COMPUTE_DIFFUSION: bool = True

# Epistatic coefficient grid (eps12 = 2 * Sigma_12).
EPS12_MIN: float = -1.0
EPS12_MAX: float = 1.0
EPS12_STEP: float = 0.1

# --- Discrete Markov chain (2loci) ---
N_MARKOV: int = 30
U_MARKOV: float = 1.0e-4
V_MARKOV: float = 1.0e-4
N_JOBS: int = -1

# Sample times for the dynamics. The first interval [0, TIMES[0]] sets the
# "initial rate". Keep TIMES[0] small (= 1 generation) for a clean slope.
TIMES: list[int] = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 170, 300, 500, 800, 1200, 2000]

# --- Continuous diffusion equilibrium (epistasis_equilibria) ---
# HAPLOID diffusion limit of the chain: rho ∝ W^{2N} x^{2Nu-1}(1-x)^{2Nv-1}.
# To compare DIRECTLY against the discrete chain we match gene copies and u:
#   N_POP_DIFF = N_MARKOV (gene copies),  U/V_DIFF = U/V_MARKOV.
# NOTE: this allele-frequency diffusion assumes LINKAGE EQUILIBRIUM across loci
#       (no LD). That is the high-r limit, so a direct distributional match with
#       the r=0 chain (which is maximally LD) is NOT expected — see module docstring.
N_POP_DIFF: float = float(N_MARKOV)
U_DIFF: float = U_MARKOV
V_DIFF: float = V_MARKOV
DIFF_RESOLUTION: int = 401
DIFF_X_EPS: float = 1.0e-8

OUT_DIR: Path = SCRIPT_DIR / "results"
# =============================================================================


@dataclass
class Row:
    r: float
    eps12: float
    # adaptation speed (discrete dynamics, initial rate)
    init_rate_kl: float
    init_rate_f1: float
    # equilibrium scores — discrete chain psi*
    kl_eq_disc: float
    ff_eq_disc: float
    psi_star_ok: bool
    # equilibrium scores — continuous diffusion
    kl_eq_diff: float
    ff_eq_diff: float
    meanW_diff: float
    # context
    kl_dyn_final: float
    ff_dyn_final: float


def _eps12_grid() -> np.ndarray:
    n = int(round((EPS12_MAX - EPS12_MIN) / EPS12_STEP)) + 1
    grid = EPS12_MIN + EPS12_STEP * np.arange(n)
    grid = np.clip(grid, EPS12_MIN, EPS12_MAX)
    return np.unique(np.round(grid, 10))


def _markov_params(r: float, eps12: float):
    """Build a 2loci SimplexParams; config_dir points at 2loci/ to reuse cache."""
    cfg = {
        "_config_dir": str(TWOLOCI_DIR),
        "N": N_MARKOV,
        "r": float(r),
        "u": [U_MARKOV, U_MARKOV],
        "v": [V_MARKOV, V_MARKOV],
        "sigma": build_sigma(S1, S2, float(eps12)).tolist(),
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
    return params_from_config(cfg, config_dir=TWOLOCI_DIR)


def _diffusion_scores(eps12: float, u_vec: np.ndarray, v_vec: np.ndarray) -> tuple[float, float, float]:
    """
    HAPLOID allele-frequency diffusion equilibrium matching the chain:

        rho(x) ∝ W(x)^{2N} · Π_i x_i^{2N u_i - 1} (1 - x_i)^{2N v_i - 1},

    on a uniform tensor grid over [x_eps, 1-x_eps]^2 (loci independent = LE).
    Returns (free fitness = logZ/(2N),  KL(eq||neutral) bits,  E_{p_eq}[W]).
    Independent of r (no linkage in this diffusion).
    """
    N = float(N_POP_DIFF)
    G = all_genotypes_haploid(2)
    Sigma = build_sigma(S1, S2, float(eps12))  # log omega(g) = g^T Sigma g
    log_omega_eq = log_fitness_haploid(G, Sigma)

    X, cell_vol = tensor_grid_centers_haploid(2, DIFF_RESOLUTION, DIFF_X_EPS)
    Xs = np.clip(X, DIFF_X_EPS, 1.0 - DIFF_X_EPS)

    log_w = log_mean_fitness_batch(G, log_omega_eq, X, "haploid", DIFF_X_EPS)
    mut = (((2.0 * N * u_vec - 1.0)[None, :]) * np.log(Xs)).sum(axis=1) \
        + (((2.0 * N * v_vec - 1.0)[None, :]) * np.log(1.0 - Xs)).sum(axis=1)

    ell_eq = 2.0 * N * log_w + mut       # haploid selection exponent = 2N
    ell_neutral = mut                    # neutral: log W = 0

    def _density(logw: np.ndarray) -> np.ndarray:
        m = float(np.max(logw))
        w = np.exp(logw - m)
        return w / float(np.sum(w) * cell_vol)

    rho_eq = _density(ell_eq)
    rho_neutral = _density(ell_neutral)
    p_eq = rho_eq * cell_vol
    p_eq /= float(np.sum(p_eq))

    kl_bits = float(np.sum(p_eq * (np.log(rho_eq + 1e-300) - np.log(rho_neutral + 1e-300))) / np.log(2.0))

    m_eq = float(np.max(ell_eq))
    log_z = float(m_eq + np.log(float(np.sum(np.exp(ell_eq - m_eq)) * cell_vol)))
    free_fitness = log_z / (2.0 * N)

    mwx = float(np.max(log_w))
    mean_w_eq = float(np.exp(mwx) * np.sum(p_eq * np.exp(log_w - mwx)))
    return free_fitness, kl_bits, mean_w_eq


def run() -> list[Row]:
    eps_grid = _eps12_grid()
    times = [int(t) for t in TIMES]
    u_diff = _as_float_vec(U_DIFF, 2, "u")
    v_diff = _as_float_vec(V_DIFF, 2, "v")

    print("=" * 72)
    print("eps12_score_vs_speed.py")
    print(f"  s1={S1}, s2={S2}")
    print(f"  r values: {R_VALUES}")
    print(f"  eps12 grid: {len(eps_grid)} values [{eps_grid[0]:g}, {eps_grid[-1]:g}] step {EPS12_STEP}")
    print(f"  Markov: N={N_MARKOV}, u=v={U_MARKOV}")
    print(f"  diffusion: N_pop={N_POP_DIFF}, u=v={U_DIFF}, res={DIFF_RESOLUTION}")
    print("=" * 72, flush=True)

    # Diffusion scores depend only on eps12 (compute once, reuse across r).
    diff_cache: dict[float, tuple[float, float, float]] = {}
    if COMPUTE_DIFFUSION:
        print("Computing diffusion equilibrium scores (r-independent) ...", flush=True)
        for eps in eps_grid:
            diff_cache[float(eps)] = _diffusion_scores(float(eps), u_diff, v_diff)
    else:
        nan3 = (float("nan"), float("nan"), float("nan"))
        for eps in eps_grid:
            diff_cache[float(eps)] = nan3

    M_mut = mutation_matrix_asymmetric(
        np.array([U_MARKOV, U_MARKOV]), np.array([V_MARKOV, V_MARKOV])
    )

    rows: list[Row] = []
    for r in R_VALUES:
        print(f"\n--- r = {r} ---", flush=True)
        # phi (neutral equilibrium) depends only on (N, u, v, r): compute once per r.
        params0 = _markov_params(r, 0.0)
        phi, _, fracs, counts, src = compute_phi(params0, M_mut)
        print(f"  phi source: {src}", flush=True)
        d_per_state = linkage_disequilibrium_per_state(fracs)

        for i, eps in enumerate(eps_grid, start=1):
            eps = float(eps)
            params = _markov_params(r, eps)
            omega = fitness_exp(params.Sigma)
            w_per_state = w_potential_per_state(fracs, omega, M_mut, params.N)

            T_row, _, _ = build_transition_matrix_row_stochastic(
                params.N, omega, M_mut, params.r,
                n_jobs=params.n_jobs, label=f"r{r}_eps{eps:.2g}", verbose=False,
            )
            P_col = column_stochastic(T_row)

            # (B) adaptation speed: replay dynamics from phi, take initial rate
            dyn = dynamics_series(
                P_col, phi, times, w_per_state, d_per_state,
                counts, fracs, params.N, params.top_k, params.ld_top_k_times,
                psi_star=None,
            )
            init_rate_kl = float(dyn.rate_kl_bits[0])
            init_rate_f1 = float(dyn.rate_f1[0])

            # (A) discrete equilibrium psi*  (fast ARPACK solver, power-iteration fallback)
            if COMPUTE_DISCRETE:
                psi_star, _resid, ok = try_selection_equilibrium(P_col)
                if psi_star is not None:
                    kl_eq_disc = kl_divergence_bits(psi_star, phi)
                    ff_eq_disc = free_fitness_1(psi_star, w_per_state, params.N)
                else:
                    kl_eq_disc = float("nan")
                    ff_eq_disc = float("nan")
            else:
                kl_eq_disc = float("nan")
                ff_eq_disc = float("nan")
                ok = False

            # (A) diffusion equilibrium
            ff_eq_diff, kl_eq_diff, mw_diff = diff_cache[eps]

            rows.append(Row(
                r=r, eps12=eps,
                init_rate_kl=init_rate_kl, init_rate_f1=init_rate_f1,
                kl_eq_disc=kl_eq_disc, ff_eq_disc=ff_eq_disc, psi_star_ok=bool(ok),
                kl_eq_diff=kl_eq_diff, ff_eq_diff=ff_eq_diff, meanW_diff=mw_diff,
                kl_dyn_final=float(dyn.kl_phi_bits[-1]), ff_dyn_final=float(dyn.f1[-1]),
            ))
            print(
                f"  [{i:2d}/{len(eps_grid)}] eps={eps:+.2f} | "
                f"rate_kl={init_rate_kl:.3e} rate_F={init_rate_f1:.3e} | "
                f"KL_eq(disc)={kl_eq_disc:.4f} KL_eq(diff)={kl_eq_diff:.4f}",
                flush=True,
            )

    return rows


# =============================================================================
# Trajectories (full KL(t) / F_N(t) curves, colored by equilibrium score)
# =============================================================================


@dataclass
class TrajResult:
    eps12: float
    kl_eq_disc: float
    ff_eq_disc: float
    times: np.ndarray
    kl_bits: np.ndarray
    f1: np.ndarray
    rate_times: np.ndarray
    rate_kl: np.ndarray
    rate_f1: np.ndarray


def run_trajectories(times: list[int], r: float = 0.0, verbose: bool = True) -> list[TrajResult]:
    """Per eps12 at a single r: full KL(t)/F_N(t) trajectory + discrete psi* score."""
    eps_grid = _eps12_grid()
    times = [int(t) for t in times]
    M_mut = mutation_matrix_asymmetric(
        np.array([U_MARKOV, U_MARKOV]), np.array([V_MARKOV, V_MARKOV])
    )
    if verbose:
        print("=" * 72)
        print(f"run_trajectories: N={N_MARKOV}, r={r}, s1={S1}, s2={S2}")
        print(f"  eps12: {len(eps_grid)} values, times up to t={max(times)}")
        print("=" * 72, flush=True)

    params0 = _markov_params(r, 0.0)
    phi, _, fracs, counts, src = compute_phi(params0, M_mut)
    d_per_state = linkage_disequilibrium_per_state(fracs)
    if verbose:
        print(f"  phi source: {src}", flush=True)

    out: list[TrajResult] = []
    for i, eps in enumerate(eps_grid, start=1):
        eps = float(eps)
        params = _markov_params(r, eps)
        omega = fitness_exp(params.Sigma)
        w_per_state = w_potential_per_state(fracs, omega, M_mut, params.N)
        T_row, _, _ = build_transition_matrix_row_stochastic(
            params.N, omega, M_mut, params.r,
            n_jobs=params.n_jobs, label=f"eps{eps:.2g}", verbose=False,
        )
        P_col = column_stochastic(T_row)
        dyn = dynamics_series(
            P_col, phi, times, w_per_state, d_per_state,
            counts, fracs, params.N, params.top_k, params.ld_top_k_times, psi_star=None,
        )
        psi_star, _resid, ok = try_selection_equilibrium(P_col)
        kl_eq = kl_divergence_bits(psi_star, phi) if psi_star is not None else float("nan")
        ff_eq = free_fitness_1(psi_star, w_per_state, params.N) if psi_star is not None else float("nan")
        out.append(TrajResult(
            eps12=eps, kl_eq_disc=kl_eq, ff_eq_disc=ff_eq,
            times=np.asarray(dyn.times, dtype=float),
            kl_bits=np.asarray(dyn.kl_phi_bits, dtype=float),
            f1=np.asarray(dyn.f1, dtype=float),
            rate_times=np.asarray(dyn.rate_times, dtype=float),
            rate_kl=np.asarray(dyn.rate_kl_bits, dtype=float),
            rate_f1=np.asarray(dyn.rate_f1, dtype=float),
        ))
        if verbose:
            print(f"  [{i:2d}/{len(eps_grid)}] eps={eps:+.2f}  KL_eq={kl_eq:.4f}", flush=True)
    return out


# panel registry for trajectory plots: key -> (y_attr, x_attr, ylabel, base filename)
_TRAJ_PANELS = {
    "kl": ("kl_bits", "times", r"$\mathrm{KL}(\psi(t)\,\|\,\varphi)$ [bits]", "info_kl_vs_t"),
    "f1": ("f1", "times", r"free fitness $\mathcal{F}_N(\psi(t))$", "free_fitness_vs_t"),
    "rate_kl": ("rate_kl", "rate_times", r"$\mathrm{d\,KL}/\mathrm{d}t$ [bits/gen]", "info_kl_rate_vs_t"),
    "rate_f1": ("rate_f1", "rate_times", r"$\mathrm{d}\mathcal{F}_N/\mathrm{d}t$ [1/gen]", "free_fitness_rate_vs_t"),
}


def plot_trajectories_by_score(
    trajs: list[TrajResult],
    out_dir: Path,
    *,
    score_attr: str = "kl_eq_disc",
    score_label: str = r"equilibrium information $\mathrm{KL}_{\mathrm{eq}}(\psi^*\,\|\,\varphi)$ [bits]",
    score_symbol: str = r"\mathrm{KL}_{\mathrm{eq}}",
    panels: list[str] | None = None,
    fname_suffix: str = "",
    show: bool = False,
) -> dict[str, Path]:
    """KL(t), F_N(t) and their rates vs t; each curve colored by an equilibrium score.

    `score_attr` selects which equilibrium score colors the curves (e.g.
    "kl_eq_disc" or "ff_eq_disc"). `panels` restricts which of _TRAJ_PANELS to
    draw. Curves are colored by the RANK of the chosen score (turbo cmap) so
    clustered scores stay visually distinct.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    _setup_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    trajs_f = [t for t in trajs if np.isfinite(getattr(t, score_attr))]
    sc = np.array([getattr(t, score_attr) for t in trajs_f], dtype=float)
    cmap = plt.get_cmap("turbo")
    n = len(sc)
    if n > 1:
        order = np.argsort(sc)
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(n)
        curve_colors = [cmap(ranks[i] / (n - 1)) for i in range(n)]
        sorted_sc = sc[order]
        mids = (sorted_sc[:-1] + sorted_sc[1:]) / 2.0
        lo = sorted_sc[0] - (mids[0] - sorted_sc[0] if mids.size else 0.5)
        hi = sorted_sc[-1] + (sorted_sc[-1] - mids[-1] if mids.size else 0.5)
        bounds = np.concatenate([[lo], mids, [hi]])
        cbar_cmap = mpl.colors.ListedColormap([cmap(i / (n - 1)) for i in range(n)])
        cbar_norm = mpl.colors.BoundaryNorm(bounds, n)
        cbar_ticks = sorted_sc[:: max(1, n // 10)]
    else:
        curve_colors = [cmap(0.5)] * n
        cbar_cmap = cmap
        cbar_norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        cbar_ticks = None

    title_fix = (
        rf"Information dynamics colored by equilibrium score" + "\n"
        rf"$N={N_MARKOV},\ r={R_VALUES[0]:g},\ s_1={S1:g},\ s_2={S2:g}$   "
        rf"(curve color $={score_symbol}$ of $\psi^*$)"
    )

    panel_keys = panels if panels is not None else list(_TRAJ_PANELS.keys())
    paths: dict[str, Path] = {}
    for pk in panel_keys:
        ykey, xkey, ylabel, fname = _TRAJ_PANELS[pk]
        fig, ax = plt.subplots(figsize=(11, 7))
        for i, t in enumerate(trajs_f):
            ax.plot(getattr(t, xkey), getattr(t, ykey),
                    color=curve_colors[i], lw=1.7, alpha=0.95,
                    label=rf"$\epsilon_{{12}}={t.eps12:+.2f}$")
        ax.set_xlabel(r"generation $t$")
        ax.set_ylabel(ylabel)
        ax.set_title(title_fix, pad=12)
        ax.grid(True, alpha=0.3)
        sm = mpl.cm.ScalarMappable(cmap=cbar_cmap, norm=cbar_norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels([f"{v:.2f}" for v in cbar_ticks])
        cbar.set_label(score_label)
        # legend below: identify the epistatic coefficient of each curve
        ncol = min(7, max(1, len(trajs_f)))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=ncol,
                  fontsize=7.5, framealpha=0.9, columnspacing=1.1, handlelength=1.6)
        fig.tight_layout()
        p = out_dir / f"{fname}{fname_suffix}.png"
        fig.savefig(p, dpi=170, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        paths[f"{fname}{fname_suffix}"] = p
        print(f"Wrote {p}")
    return paths


def plot_equilibrium_sweep_vs_eps12(trajs: list[TrajResult], out_path: Path, show: bool = False) -> Path:
    """Epistasis-equilibria-style sweep: equilibrium free fitness (left axis) and
    equilibrium information KL_eq (right axis) vs eps12, from the discrete chain psi*.
    """
    import matplotlib.pyplot as plt

    _setup_style()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ts = sorted(trajs, key=lambda t: t.eps12)
    eps = np.array([t.eps12 for t in ts])
    ff = np.array([t.ff_eq_disc for t in ts])
    kl = np.array([t.kl_eq_disc for t in ts])

    fig, ax_ff = plt.subplots(figsize=(9.5, 5.5))
    ax_kl = ax_ff.twinx()
    (ln0,) = ax_ff.plot(eps, ff, "o-", color="C0", lw=1.8, ms=5,
                        label=r"free fitness $\mathcal{F}_N(\psi^*)$")
    (ln1,) = ax_kl.plot(eps, kl, "s-", color="C1", lw=1.8, ms=4, alpha=0.9,
                        label=r"information $\mathrm{KL}_{\mathrm{eq}}(\psi^*\,\|\,\varphi)$ [bits]")
    ax_ff.set_xlabel(r"epistasis $\epsilon_{12}$")
    ax_ff.set_ylabel(r"free fitness $\mathcal{F}_N(\psi^*)$", color="C0")
    ax_kl.set_ylabel(r"information $\mathrm{KL}_{\mathrm{eq}}(\psi^*\,\|\,\varphi)$ [bits]", color="C1")
    ax_ff.tick_params(axis="y", labelcolor="C0")
    ax_kl.tick_params(axis="y", labelcolor="C1")
    ax_ff.set_title(
        rf"Discrete-chain equilibrium $\psi^*$ vs $\epsilon_{{12}}$   "
        rf"($N={N_MARKOV},\ r={R_VALUES[0]:g},\ s_1={S1:g},\ s_2={S2:g}$)"
    )
    ax_ff.grid(True, alpha=0.35)
    ax_ff.legend(handles=[ln0, ln1], loc="best", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Wrote {out_path}")
    return out_path


# =============================================================================
# Output
# =============================================================================


def write_csv(rows: list[Row], path: Path) -> None:
    fields = list(asdict(rows[0]).keys())
    lines = [",".join(fields)]
    for row in rows:
        d = asdict(row)
        lines.append(",".join(f"{d[f]}" for f in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation without a SciPy dependency."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    rx = _rankdata(x[mask])
    ry = _rankdata(y[mask])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt(np.sum(rx**2) * np.sum(ry**2))
    if denom == 0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), 1-based."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg = sums / counts
    return avg[inv]


def _setup_style():
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
    except ImportError:
        pass
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "figure.facecolor": "white",
        "axes.facecolor": "#fafafa",
    })


# predictor (equilibrium score) keys -> axis labels
_SCORES = {
    "kl": ("KL", r"information $\mathrm{KL}(\mathrm{eq}\,\|\,\mathrm{neutral})$ [bits]"),
    "ff": ("F_N", r"free fitness $\mathcal{F}_N$ at equilibrium"),
}
# response (speed) keys -> axis labels
_SPEEDS = {
    "init_rate_kl": r"initial $\mathrm{d}\,\mathrm{KL}/\mathrm{d}t$ [bits/gen]",
    "init_rate_f1": r"initial $\mathrm{d}\mathcal{F}_N/\mathrm{d}t$ [1/gen]",
}

_R_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def _title_lines(source: str) -> str:
    src = "discrete chain $\\psi^*$ ($N=%d$)" % N_MARKOV if source == "disc" else \
          "diffusion equilibrium ($N_{\\mathrm{pop}}=%g$)" % N_POP_DIFF
    return (
        rf"Equilibrium score vs. adaptation speed — {src}" + "\n"
        rf"$s_1={S1:g},\ s_2={S2:g}$,   $\epsilon_{{12}}\in[{EPS12_MIN:g},{EPS12_MAX:g}]$"
    )


def plot_score_vs_speed(rows: list[Row], source: str, out_path: Path) -> None:
    """2x2 scatter: rows = {KL_eq, F_N_eq predictor}, cols = {rate_kl, rate_f1}."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    _setup_style()
    score_suffix = "disc" if source == "disc" else "diff"
    r_values = sorted({row.r for row in rows})
    eps_all = np.array([row.eps12 for row in rows])
    cmap = plt.get_cmap("coolwarm")
    norm = mpl.colors.Normalize(vmin=EPS12_MIN, vmax=EPS12_MAX)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ri, (skey, (sname, slabel)) in enumerate(_SCORES.items()):
        score_attr = f"{skey}_eq_{score_suffix}"
        for ci, (vkey, vlabel) in enumerate(_SPEEDS.items()):
            ax = axes[ri, ci]
            for k, r in enumerate(r_values):
                sub = [row for row in rows if row.r == r]
                xs = np.array([getattr(row, score_attr) for row in sub])
                ys = np.array([getattr(row, vkey) for row in sub])
                es = np.array([row.eps12 for row in sub])
                rho = _spearman(xs, ys)
                ax.scatter(
                    xs, ys, c=es, cmap=cmap, norm=norm,
                    marker=_R_MARKERS[k % len(_R_MARKERS)],
                    s=55, edgecolor="black", linewidth=0.4,
                    label=rf"$r={r:g}$  ($\rho_S={rho:.2f}$)",
                )
            ax.set_xlabel(slabel)
            ax.set_ylabel(vlabel)
            ax.legend(loc="best", fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, pad=0.02, fraction=0.025)
    cbar.set_label(r"epistasis $\epsilon_{12}$")
    fig.suptitle(_title_lines(source), y=0.99)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_vs_eps12(rows: list[Row], out_path: Path) -> None:
    """Dual-axis vs eps12: equilibrium score (left) and adaptation speed (right).

    One row of panels per r; left axis shows both equilibrium KL scores,
    right axis shows the initial KL rate. The intuitive 'do they track' view.
    """
    import matplotlib.pyplot as plt

    _setup_style()
    r_values = sorted({row.r for row in rows})
    nrows = len(r_values)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.4 * nrows), squeeze=False)

    for k, r in enumerate(r_values):
        ax = axes[k, 0]
        sub = sorted([row for row in rows if row.r == r], key=lambda x: x.eps12)
        eps = np.array([row.eps12 for row in sub])
        kl_disc = np.array([row.kl_eq_disc for row in sub])
        kl_diff = np.array([row.kl_eq_diff for row in sub])
        rate_kl = np.array([row.init_rate_kl for row in sub])

        handles = []
        if np.any(np.isfinite(kl_disc)):
            ln1, = ax.plot(eps, kl_disc, "o-", color="#1f77b4", lw=1.8, ms=4,
                           label=r"KL$_{\mathrm{eq}}$ discrete $\psi^*$")
            handles.append(ln1)
        if np.any(np.isfinite(kl_diff)):
            ln2, = ax.plot(eps, kl_diff, "s--", color="#2ca02c", lw=1.6, ms=4,
                           label=r"KL$_{\mathrm{eq}}$ diffusion")
            handles.append(ln2)
        ax.set_ylabel("equilibrium KL [bits]")
        ax.set_xlabel(r"epistasis $\epsilon_{12}$")

        ax_r = ax.twinx()
        ln3, = ax_r.plot(eps, rate_kl, "^-", color="#d62728", lw=1.8, ms=4,
                         label=r"initial $\mathrm{d}\,\mathrm{KL}/\mathrm{d}t$")
        ax_r.set_ylabel("initial KL rate [bits/gen]", color="#d62728")
        ax_r.tick_params(axis="y", labelcolor="#d62728")
        handles.append(ln3)

        rho_disc = _spearman(kl_disc, rate_kl)
        rho_diff = _spearman(kl_diff, rate_kl)
        title_bits = [rf"$r={r:g}$"]
        if np.any(np.isfinite(kl_disc)):
            title_bits.append(rf"$\rho_S$[disc,rate]$={rho_disc:.2f}$")
        if np.any(np.isfinite(kl_diff)):
            title_bits.append(rf"$\rho_S$[diff,rate]$={rho_diff:.2f}$")
        ax.set_title("   ".join(title_bits))
        ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        rf"Epistatic sweep: equilibrium score and adaptation speed vs. $\epsilon_{{12}}$"
        + "\n" + rf"$s_1={S1:g},\ s_2={S2:g}$",
        y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Equilibrium score vs adaptation speed over eps12 / r")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: results/<timestamp>_s1_..._s2_...)")
    p.add_argument("--no-plot", action="store_true", help="Only write CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = run()

    if args.out_dir is not None:
        run_dir = args.out_dir
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = OUT_DIR / f"{ts}_s1_{S1:g}_s2_{S2:g}"
    run_dir.mkdir(parents=True, exist_ok=True)

    write_csv(rows, run_dir / "scores_speed.csv")

    if not args.no_plot:
        if COMPUTE_DISCRETE:
            plot_score_vs_speed(rows, "disc", run_dir / "score_vs_speed_discrete.png")
        if COMPUTE_DIFFUSION:
            plot_score_vs_speed(rows, "diff", run_dir / "score_vs_speed_diffusion.png")
        plot_vs_eps12(rows, run_dir / "score_and_speed_vs_eps12.png")

    print(f"\nAll outputs in: {run_dir}")


if __name__ == "__main__":
    main()
