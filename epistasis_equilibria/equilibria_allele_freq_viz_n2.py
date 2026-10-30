"""
Visualization helper for n=2 haploid equilibrium on allele-frequency space.

Creates 3D surfaces for:
  1) equilibrium density with the configured matrix M,
  2) neutral density (M = 0),
  3) mean fitness landscape W(x),

and comparison panels showing how selection deforms neutrality.

Run from epistasis_equilibria/:
  python equilibria_allele_freq_viz_n2.py --config equilibria_config_allele_freq_example.yaml

Run from repo root (script lives under epistasis_equilibria/):
  python epistasis_equilibria/equilibria_allele_freq_viz_n2.py save --config epistasis_equilibria/equilibria_config_allele_freq_example.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from equilibria_allele_freq import (
    _as_float_vec,
    all_genotypes_haploid,
    ell_batch,
    load_config,
    log_fitness_haploid,
    log_mean_fitness_batch,
    logsumexp1d,
    matrix_M_from_sigma,
    mutation_log_terms_batch,
)


def _load_M_from_cfg(cfg: dict, n: int) -> np.ndarray:
    if "sigma" in cfg and cfg["sigma"] is not None:
        M = matrix_M_from_sigma(np.array(cfg["sigma"], dtype=float))
    elif cfg.get("sigma_file"):
        p = str(cfg["sigma_file"])
        if p.endswith(".npy"):
            M = np.load(p).astype(float)
        else:
            M = np.loadtxt(p).astype(float)
        M = matrix_M_from_sigma(M)
    else:
        raise ValueError("Config must provide either 'sigma:' or 'sigma_file:'.")
    if M.shape != (n, n):
        raise ValueError(f"Expected M shape {(n, n)}, got {M.shape}")
    return M


def _normalize_density_from_logweights(logw: np.ndarray, dxdy: float) -> np.ndarray:
    m = float(np.max(logw))
    w = np.exp(logw - m)
    Z = float(np.sum(w) * dxdy)
    return w / Z


def _corner_dict(Z: np.ndarray) -> dict[str, float]:
    """Corner values for a 2D grid indexed as [row(x2), col(x1)]."""
    return {
        "(x1≈0, x2≈0)": float(Z[0, 0]),
        "(x1≈1, x2≈0)": float(Z[0, -1]),
        "(x1≈0, x2≈1)": float(Z[-1, 0]),
        "(x1≈1, x2≈1)": float(Z[-1, -1]),
    }


def _corner_spread(corners: dict[str, float]) -> tuple[float, float]:
    vals = np.array(list(corners.values()), dtype=float)
    abs_spread = float(vals.max() - vals.min())
    rel_spread = float(abs_spread / max(1e-300, np.mean(vals)))
    return abs_spread, rel_spread


def _corner_text(corners: dict[str, float]) -> str:
    return (
        "Corners:\n"
        f"00={corners['(x1≈0, x2≈0)']:.3e}, 10={corners['(x1≈1, x2≈0)']:.3e}\n"
        f"01={corners['(x1≈0, x2≈1)']:.3e}, 11={corners['(x1≈1, x2≈1)']:.3e}"
    )


def _add_surface(
    fig: plt.Figure,
    position: int,
    X1: np.ndarray,
    X2: np.ndarray,
    Z: np.ndarray,
    *,
    title: str,
    zlabel: str,
    cmap: str,
) -> None:
    ax = fig.add_subplot(2, 2, position, projection="3d")
    ax.plot_surface(
        X1,
        X2,
        Z,
        cmap=cmap,
        linewidth=0,
        edgecolor="none",
        antialiased=False,
        shade=True,
        rcount=Z.shape[0],
        ccount=Z.shape[1],
    )
    ax.view_init(elev=28, azim=-135)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel(zlabel)


# -----------------------------------------------------------------------------
# Example invocations
#
# From repo root (e.g. ~/Projects/miso): the script is NOT alongside equilibria_allele_freq.py
# at top level — use the epistasis_equilibria/ prefix for script and config.
#   python3 epistasis_equilibria/equilibria_allele_freq_viz_n2.py save --config epistasis_equilibria/equilibria_config_allele_freq_example.yaml
#   (with save: outputs go to ./results/<timestamp>/ relative to cwd — here, repo root)
#
# From epistasis_equilibria/ (short paths):
#   python3 equilibria_allele_freq_viz_n2.py --config equilibria_config_allele_freq_example.yaml
#   python3 equilibria_allele_freq_viz_n2.py save --config equilibria_config_allele_freq_example.yaml
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="n=2 haploid equilibrium/neutral visualization.")
    parser.add_argument(
        "action",
        nargs="?",
        choices=["save"],
        help="Use 'save' to write figures and a run log under results/<date-time>/.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="equilibria_config_allele_freq_example.yaml",
        help="YAML/JSON config used by equilibria_allele_freq.py",
    )
    parser.add_argument("--resolution", type=int, default=81, help="Grid points per axis.")
    parser.add_argument(
        "--smooth-resolution",
        type=int,
        default=181,
        help="Grid points per axis for the extra static smooth-view figure.",
    )
    parser.add_argument("--x-eps", type=float, default=None, help="Override x_eps from config.")
    parser.add_argument("--save-prefix", type=str, default="", help="Optional prefix to save figures.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory used by the 'save' action.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    n = int(cfg.get("n_loci", 2))
    ploidy = str(cfg.get("ploidy", "haploid")).lower()
    if n != 2:
        raise ValueError(f"This helper is for n_loci=2 only (got n_loci={n}).")
    if ploidy != "haploid":
        raise ValueError(f"This helper is for ploidy=haploid only (got ploidy={ploidy}).")

    N_pop = float(cfg.get("N", 50.0))
    u_vec = _as_float_vec(cfg.get("u", 1e-4), 2, "u")
    v_vec = _as_float_vec(cfg.get("v", 1e-4), 2, "v")
    x_eps = float(
        args.x_eps
        if args.x_eps is not None
        else cfg.get("explore", {}).get("x_eps", 1e-8)
    )
    M = _load_M_from_cfg(cfg, 2)

    G = all_genotypes_haploid(2)
    log_omega_eq = log_fitness_haploid(G, M)
    log_omega_neutral = np.zeros_like(log_omega_eq)

    # 2D allele-frequency grid
    r = int(args.resolution)
    x1_edges = np.linspace(x_eps, 1.0 - x_eps, r + 1)
    x2_edges = np.linspace(x_eps, 1.0 - x_eps, r + 1)
    x1 = 0.5 * (x1_edges[:-1] + x1_edges[1:])
    x2 = 0.5 * (x2_edges[:-1] + x2_edges[1:])
    X1, X2 = np.meshgrid(x1, x2, indexing="xy")
    X = np.column_stack([X1.ravel(), X2.ravel()])

    dx = x1_edges[1] - x1_edges[0]
    dy = x2_edges[1] - x2_edges[0]
    dxdy = dx * dy

    ell_eq = ell_batch(X, G, log_omega_eq, N_pop, u_vec, v_vec, "haploid", x_eps)
    ell_neutral = ell_batch(X, G, log_omega_neutral, N_pop, u_vec, v_vec, "haploid", x_eps)

    rho_eq = _normalize_density_from_logweights(ell_eq, dxdy).reshape(r, r)
    rho_neutral = _normalize_density_from_logweights(ell_neutral, dxdy).reshape(r, r)

    logW_eq = log_mean_fitness_batch(G, log_omega_eq, X, "haploid", x_eps).reshape(r, r)
    W_eq = np.exp(logW_eq)

    # Selection effect on equilibrium relative to neutral
    log_ratio = np.log(rho_eq + 1e-300) - np.log(rho_neutral + 1e-300)

    p_eq_grid = (rho_eq.ravel() * dxdy).astype(float)
    p_eq_grid /= float(np.sum(p_eq_grid))
    kl_eq_to_neutral_bits = float(np.sum(p_eq_grid * log_ratio.ravel()) / np.log(2.0))
    entropy_disc = -float(np.sum(p_eq_grid * np.log(p_eq_grid + 1e-300)))
    mut_support = mutation_log_terms_batch(X, N_pop, u_vec, v_vec, x_eps)
    local_free = logW_eq.ravel() + mut_support / N_pop
    mean_logW = float(np.sum(p_eq_grid * logW_eq.ravel()))
    mean_local_free = float(np.sum(p_eq_grid * local_free))
    entropy_cont = entropy_disc + np.log(dxdy)
    free_fitness = float(mean_local_free + entropy_cont / N_pop)
    free_fitness_logZ = float((logsumexp1d(ell_eq) + np.log(dxdy)) / N_pop)

    # Fast debugging diagnostics at corners
    c_eq = _corner_dict(rho_eq)
    c_neu = _corner_dict(rho_neutral)
    c_w = _corner_dict(W_eq)
    spread_eq_abs, spread_eq_rel = _corner_spread(c_eq)
    spread_neu_abs, spread_neu_rel = _corner_spread(c_neu)
    spread_w_abs, spread_w_rel = _corner_spread(c_w)

    print("\n--- Fast debugging: corner symmetry diagnostics ---")
    print(f"x_eps = {x_eps:g}, grid resolution = {r}")
    print("rho_neutral corners:")
    for k, v in c_neu.items():
        print(f"  {k}: {v:.8e}")
    print(
        f"  neutral corner spread = {spread_neu_abs:.3e} "
        f"({100*spread_neu_rel:.3f}% of corner mean)"
    )
    print("rho_eq corners:")
    for k, v in c_eq.items():
        print(f"  {k}: {v:.8e}")
    print(
        f"  eq corner spread = {spread_eq_abs:.3e} "
        f"({100*spread_eq_rel:.3f}% of corner mean)"
    )
    print("W(x) corners:")
    for k, v in c_w.items():
        print(f"  {k}: {v:.8e}")
    print(
        f"  W corner spread = {spread_w_abs:.3e} "
        f"({100*spread_w_rel:.3f}% of corner mean)"
    )

    print("\n--- Grid diagnostics ---")
    print(f"KL(ρ_eq || ρ_neutral[M=0]) = {kl_eq_to_neutral_bits:.6g} bits")
    print(f"E_ρ[log W(x)] = {mean_logW:.6g}")
    print(f"E_ρ[logW + mutation/N] = {mean_local_free:.6g}")
    print(f"Free fitness F_N(ρ_eq) = {free_fitness:.6g}")
    print(f"Log-partition check log∫exp(ℓ)dx/N = {free_fitness_logZ:.6g}")

    # Higher-resolution log-density surfaces for the lower row of Figure 1.
    rs = int(args.smooth_resolution)
    xs1_edges = np.linspace(x_eps, 1.0 - x_eps, rs + 1)
    xs2_edges = np.linspace(x_eps, 1.0 - x_eps, rs + 1)
    xs1 = 0.5 * (xs1_edges[:-1] + xs1_edges[1:])
    xs2 = 0.5 * (xs2_edges[:-1] + xs2_edges[1:])
    XS1, XS2 = np.meshgrid(xs1, xs2, indexing="xy")
    XS = np.column_stack([XS1.ravel(), XS2.ravel()])
    dxs = xs1_edges[1] - xs1_edges[0]
    dys = xs2_edges[1] - xs2_edges[0]
    dxsdys = dxs * dys

    ell_eq_s = ell_batch(XS, G, log_omega_eq, N_pop, u_vec, v_vec, "haploid", x_eps)
    ell_neutral_s = ell_batch(XS, G, log_omega_neutral, N_pop, u_vec, v_vec, "haploid", x_eps)
    rho_eq_s = _normalize_density_from_logweights(ell_eq_s, dxsdys).reshape(rs, rs)
    rho_neutral_s = _normalize_density_from_logweights(ell_neutral_s, dxsdys).reshape(rs, rs)

    # ---- Figure 1: raw density surfaces with log10 sketches beneath ----
    fig = plt.figure(figsize=(13, 10))
    _add_surface(
        fig,
        1,
        X1,
        X2,
        rho_eq,
        title="Equilibrium density ρ_eq (with M)",
        zlabel="density",
        cmap="viridis_r",
    )
    _add_surface(
        fig,
        2,
        X1,
        X2,
        rho_neutral,
        title="Neutral density ρ_neutral (M=0)",
        zlabel="density",
        cmap="cividis_r",
    )
    _add_surface(
        fig,
        3,
        XS1,
        XS2,
        np.log10(rho_eq_s + 1e-300),
        title="Smooth sketch: log10 ρ_eq",
        zlabel="log10 density",
        cmap="viridis_r",
    )
    _add_surface(
        fig,
        4,
        XS1,
        XS2,
        np.log10(rho_neutral_s + 1e-300),
        title="Smooth sketch: log10 ρ_neutral",
        zlabel="log10 density",
        cmap="cividis_r",
    )

    fig.suptitle(
        f"n=2 haploid | N={N_pop:g} | u={u_vec} | v={v_vec}\n"
        f"M=\n{np.array2string(M, precision=4, suppress_small=True)}",
        fontsize=10,
    )
    fig.tight_layout()

    # ---- Figure 2: log(ρ_eq/ρ_neutral) ≈ N_e log W (contour + 3D) and diagnostics ----
    fig2 = plt.figure(figsize=(16, 5.2))
    gs2 = fig2.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 0.85], wspace=0.28)

    ax2_0 = fig2.add_subplot(gs2[0, 0])
    im0 = ax2_0.contourf(X1, X2, log_ratio, levels=40, cmap="magma")
    ax2_0.set_title(r"$N_e \log W(\mathbf{x})$" + "\n(≈ log(ρ_eq / ρ_neutral))")
    ax2_0.set_xlabel("x1")
    ax2_0.set_ylabel("x2")
    plt.colorbar(im0, ax=ax2_0, fraction=0.046, pad=0.04)

    ax2_1 = fig2.add_subplot(gs2[0, 1], projection="3d")
    ax2_1.plot_surface(
        X1,
        X2,
        log_ratio,
        cmap="magma",
        linewidth=0,
        edgecolor="none",
        antialiased=False,
        shade=True,
        rcount=log_ratio.shape[0],
        ccount=log_ratio.shape[1],
    )
    ax2_1.view_init(elev=28, azim=-135)
    ax2_1.set_title(r"$N_e \log W$" + "\n(3D: same field as left)")
    ax2_1.set_xlabel("x1")
    ax2_1.set_ylabel("x2")
    ax2_1.set_zlabel(r"$\log(\rho_{eq}/\rho_{neutral})$")

    ax2_t = fig2.add_subplot(gs2[0, 2])
    ax2_t.axis("off")
    diagnostics_text = (
        "Diagnostics\n\n"
        f"population size N\n(config) = {N_pop:g}\n\n"
        f"KL(ρ_eq || ρ_neutral)\n= {kl_eq_to_neutral_bits:.6g} bits\n\n"
        f"E_ρ[log W]\n= {mean_logW:.6g}\n\n"
        f"Free fitness F_N(ρ_eq)\n= {free_fitness:.6g}\n\n"
        f"log∫exp(ℓ)dx / N\n= {free_fitness_logZ:.6g}"
    )
    ax2_t.text(
        0.02,
        0.98,
        diagnostics_text,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="0.6", alpha=0.95),
    )

    fig2.suptitle(
        r"Allele-frequency plane: $N_e\log W(\mathbf{x}) \approx \log(\rho_{eq}/\rho_{neutral})$",
        fontsize=11,
    )
    fig2.tight_layout()

    if args.action == "save" or args.save_prefix:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.action == "save":
            save_dir = Path(args.results_dir) / timestamp
            save_dir.mkdir(parents=True, exist_ok=True)
            prefix = save_dir / f"allele_freq_{timestamp}"
        else:
            prefix = Path(args.save_prefix)
            save_dir = prefix.parent if str(prefix.parent) else Path(".")
            save_dir.mkdir(parents=True, exist_ok=True)

        fig1_path = Path(f"{prefix}_figure1_surfaces.png")
        fig2_path = Path(f"{prefix}_figure2_diagnostics.png")
        fig.savefig(fig1_path, dpi=220)
        fig2.savefig(fig2_path, dpi=220)

        log_path = save_dir / f"allele_freq_{timestamp}_run_log.txt"
        run_log = {
            "timestamp": timestamp,
            "config_path": str(Path(args.config).resolve()),
            "config_values": cfg,
            "effective_values": {
                "n_loci": n,
                "ploidy": ploidy,
                "N": N_pop,
                "u": u_vec.tolist(),
                "v": v_vec.tolist(),
                "x_eps": x_eps,
                "resolution": r,
                "smooth_resolution": rs,
                "M": M.tolist(),
                "haploid_scaling": "ell = N log W + sum_i(2N u_i - 1) log x_i + sum_i(2N v_i - 1) log(1 - x_i)",
            },
            "diagnostics": {
                "kl_eq_to_neutral_bits": kl_eq_to_neutral_bits,
                "mean_logW": mean_logW,
                "mean_logW_plus_mutation_over_N": mean_local_free,
                "free_fitness": free_fitness,
                "log_partition_over_N": free_fitness_logZ,
            },
            "outputs": {
                "figure1": str(fig1_path),
                "figure2": str(fig2_path),
            },
        }
        log_path.write_text(json.dumps(run_log, indent=2), encoding="utf-8")
        print(f"Saved figures and run log under: {save_dir}")

    print("Done.")
    print(f"Integral approx of rho_eq: {(rho_eq.sum() * dxdy):.6f}")
    print(f"Integral approx of rho_neutral: {(rho_neutral.sum() * dxdy):.6f}")
    plt.show()


if __name__ == "__main__":
    main()
