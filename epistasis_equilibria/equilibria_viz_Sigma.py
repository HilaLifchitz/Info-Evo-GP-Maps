"""
equilibria_viz.py — Visualize equilibrium distribution for n=2 (3-simplex / tetrahedron)

For n=2 loci, the state space is the 3-simplex with 4 vertices (genotypes 00, 01, 10, 11).
Plots the tetrahedron with points colored by probability mass. By default adds a second panel (mass spread)
and a third panel (fitness landscape): same lattice points, size & color ∝ ρ in panel 2, ∝ W(x) in panel 3.

Usage:
  python equilibria_viz.py --config equilibria_config_viz_example.yaml
  python equilibria_viz.py --config equilibria_config_sigma_direct.yaml --out plot.png
"""

import argparse
from pathlib import Path

import numpy as np

# Import from equilibria_explored (must be in same directory or on path)
from equilibria_explored import (
    apply_config_to_args,
    build_sigma_omega_alpha,
    equilibrium_distribution_grid,
    load_config,
    mcmc_equilibrium_sampler,
)


def simplex_3_to_3d(points: np.ndarray) -> np.ndarray:
    """
    Map 3-simplex points (4 coords, sum=1) to 3D tetrahedron.
    Vertices of a regular tetrahedron:
      00=(0,0,0), 01=(1,0,0), 10=(0.5,√3/2,0), 11=(0.5,√3/6,√6/3)
    """
    sqrt3 = np.sqrt(3)
    sqrt6 = np.sqrt(6)
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, sqrt3 / 2, 0],
        [0.5, sqrt3 / 6, sqrt6 / 3],
    ])
    return points @ vertices  # (n_pts, 4) @ (4, 3) -> (n_pts, 3)


def plot_equilibrium(
    points: np.ndarray,
    probs: np.ndarray,
    out_path: str | None = None,
    title: str = "Equilibrium distribution (n=2)",
    show: bool = True,
    omega: np.ndarray | None = None,
    G: np.ndarray | None = None,
    Sigma: np.ndarray | None = None,
    N: int | None = None,
    u: float | None = None,
    v: float | None = None,
    mass_view: str = "projection",
    alpha: np.ndarray | None = None,
    mcmc_samples: int = 5000,
    mcmc_seed: int = 42,
    show_fitness: bool = True,
) -> None:
    """
    3D scatter plot of simplex points (tetrahedron) colored by probability mass.
    Optionally add a second panel showing mass spread:
    - 'sampled': sample grid points with probability ∝ ρ(x) — more points where mass is higher
    - 'mcmc': MCMC samples from ρ — same effect, uses MCMC chain
    - 'projection': plot each grid point once, colored by ρ (no density effect)
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        raise ImportError("matplotlib required: pip install matplotlib")

    xyz = simplex_3_to_3d(points)
    verts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3) / 2, 0],
        [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3],
    ])
    verts_2d = verts[:, :2]

    show_2d = mass_view in ("projection", "mcmc", "sampled")
    show_fit = show_fitness and omega is not None

    # Figure 1: 3D mass | 3D fitness (or just mass if no fitness)
    n_3d = 2 if show_fit else 1
    fig_3d = plt.figure(figsize=(6 * n_3d, 6))
    fig_3d.patch.set_facecolor("white")
    ax = fig_3d.add_subplot(1, n_3d, 1, projection="3d")
    ax.set_facecolor("white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("lightgray")
    ax.yaxis.pane.set_edgecolor("lightgray")
    ax.zaxis.pane.set_edgecolor("lightgray")

    sizes = np.maximum(probs / (probs.max() + 1e-300) * 150, 6)  # size & color ∝ ρ
    sc = ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=probs,
        s=sizes,
        cmap="viridis",
        alpha=0.7,
        edgecolors="none",
    )

    # Draw tetrahedron edges
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for i, j in edges:
        ax.plot(
            [verts[i, 0], verts[j, 0]],
            [verts[i, 1], verts[j, 1]],
            [verts[i, 2], verts[j, 2]],
            "k-",
            linewidth=1,
            alpha=0.5,
        )

    labels = ["00", "01", "10", "11"]
    for i, v in enumerate(verts):
        lbl = f"{labels[i]}: ω={omega[i]:.3g}" if omega is not None else labels[i]
        ax.text(v[0], v[1], v[2], lbl, fontsize=9, fontweight="bold")

    plt.colorbar(sc, ax=ax, shrink=0.6, label="Probability mass ρ(x)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Mass (3D)" if (show_fit or show_2d) else title)
    ax.set_box_aspect([1, 1, 1])

    # Legend: Σ, N, u, v, and ω(g)
    legend_parts = []
    if Sigma is not None:
        sigma_str = np.array2string(Sigma, precision=2, suppress_small=True)
        legend_parts.append(f"Σ =\n{sigma_str}")
    if N is not None:
        legend_parts.append(f"N = {N}")
    if u is not None and v is not None:
        u_val = float(np.asarray(u).flat[0])
        v_val = float(np.asarray(v).flat[0])
        legend_parts.append(f"u = {u_val:.2e}, v = {v_val:.2e}")
    if omega is not None and G is not None:
        omega_lines = [f"{''.join(str(int(b)) for b in G[i])}: ω={omega[i]:.4g}" for i in range(len(omega))]
        legend_parts.append("ω(g):\n" + "\n".join(omega_lines))
    if legend_parts:
        legend_text = "\n".join(legend_parts)
        ax.text2D(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=8,
                  verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.95),
                  family="monospace")

    # 3D fitness
    if show_fit:
        ax_fit3d = fig_3d.add_subplot(1, n_3d, 2, projection="3d")
        ax_fit3d.set_facecolor("white")
        ax_fit3d.xaxis.pane.fill = False
        ax_fit3d.yaxis.pane.fill = False
        ax_fit3d.zaxis.pane.fill = False
        ax_fit3d.xaxis.pane.set_edgecolor("lightgray")
        ax_fit3d.yaxis.pane.set_edgecolor("lightgray")
        ax_fit3d.zaxis.pane.set_edgecolor("lightgray")
        W_vals = points @ omega
        W_max = W_vals.max()
        W_min = W_vals.min()
        W_range = W_max - W_min + 1e-300
        sizes = 3 + 5 * (W_vals - W_min) / W_range
        sc_fit = ax_fit3d.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=W_vals, s=sizes, cmap="viridis", alpha=0.8, edgecolors="none")
        plt.colorbar(sc_fit, ax=ax_fit3d, shrink=0.6, label="W(x)")
        for i, j in edges:
            ax_fit3d.plot([verts[i, 0], verts[j, 0]], [verts[i, 1], verts[j, 1]], [verts[i, 2], verts[j, 2]],
                          "k-", linewidth=1, alpha=0.5)
        for i, v in enumerate(verts):
            ax_fit3d.text(v[0], v[1], v[2], labels[i], fontsize=9, fontweight="bold")
        ax_fit3d.set_xlabel("x")
        ax_fit3d.set_ylabel("y")
        ax_fit3d.set_zlabel("z")
        ax_fit3d.set_title("Fitness (3D)")
        ax_fit3d.set_box_aspect([1, 1, 1])

    plt.figure(fig_3d.number)
    plt.tight_layout()
    if out_path:
        p = Path(out_path)
        out_3d = str(p.parent / (p.stem + "_3d" + p.suffix))
        fig_3d.savefig(out_3d, dpi=150, bbox_inches="tight")
        print(f"Saved 3D to {out_3d}")
    if show and not show_2d:
        plt.figure(fig_3d.number)
        plt.show()

    # Figure 2: 2D mass projection | 2D fitness projection
    if show_2d:
        n_2d = 2 if show_fit else 1
        fig_2d = plt.figure(figsize=(6 * n_2d, 6))
        fig_2d.patch.set_facecolor("white")
        ax2 = fig_2d.add_subplot(1, n_2d, 1)
        ax2.set_facecolor("white")
        labels = ["00", "01", "10", "11"]

        # Draw simplex outline first (so scatter appears on top)
        for i, j in [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]:
            ax2.plot([verts_2d[i, 0], verts_2d[j, 0]], [verts_2d[i, 1], verts_2d[j, 1]],
                     "k-", linewidth=1.5, alpha=0.7)
        for i, v2 in enumerate(verts_2d):
            ax2.annotate(labels[i], v2, fontsize=10, fontweight="bold", xytext=(4, 4), textcoords="offset points")

        if mass_view == "mcmc" and alpha is not None and N is not None:
            # MCMC samples from ρ — chain spends more time where ρ is high, so we get more points there
            print("Running MCMC for mass spread view...")
            k = len(omega)
            thin = 10
            samples_x = []
            for counts, x, _W, _ell in mcmc_equilibrium_sampler(
                k, N, omega, alpha,
                n_steps=mcmc_samples * thin + 5000,
                burn_in=5000,
                thin=thin,
                seed=mcmc_seed,
            ):
                samples_x.append(x)
                if len(samples_x) >= mcmc_samples:
                    break
            X = np.array(samples_x)
            n_unique = len(set(tuple(x) for x in X))
            print(f"  MCMC: {n_unique} unique states visited out of {len(X)} samples")
            if n_unique < 20:
                print("  (Chain may be stuck — try --mass-view sampled for full distribution)")
            pts_2d = simplex_3_to_3d(X)[:, :2]
            ax2.scatter(pts_2d[:, 0], pts_2d[:, 1], c="steelblue", s=12, alpha=0.5, edgecolors="none")
            ax2.set_title(f"Mass spread (MCMC: {n_unique} unique states)")
        elif mass_view == "sampled":
            # Sample grid points with probability ∝ ρ(x) — same visual: more points where mass is higher
            rng = np.random.default_rng(mcmc_seed)
            n_draw = min(mcmc_samples, len(points))
            idx = rng.choice(len(points), size=n_draw, replace=True, p=probs)
            sampled_pts = points[idx]
            pts_2d = simplex_3_to_3d(sampled_pts)[:, :2]
            ax2.scatter(pts_2d[:, 0], pts_2d[:, 1], c="steelblue", s=12, alpha=0.5, edgecolors="none")
            ax2.set_title("Mass spread (sampled ∝ ρ — more points where mass is higher)")
        else:
            # All lattice points in simplex; size and color ∝ probability (higher ρ = bigger + brighter)
            xy = xyz[:, :2]
            p_max = probs.max() + 1e-300
            sizes = np.maximum(probs / p_max * 180, 6)  # high-prob points much bigger
            sc2 = ax2.scatter(xy[:, 0], xy[:, 1], c=probs, s=sizes, cmap="viridis", alpha=0.85, edgecolors="none")
            plt.colorbar(sc2, ax=ax2, shrink=0.6, label="ρ(x)")
            ax2.set_title("Mass (2D projection)")

        # Explicit axis limits so points are visible (tetrahedron base: x in [0,1], y in [0,√3/2])
        pad = 0.08
        ax2.set_xlim(-pad, 1 + pad)
        ax2.set_ylim(-pad, np.sqrt(3) / 2 + pad)
        ax2.set_aspect("equal", adjustable="box")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")

        # 2D fitness projection
        if show_fit:
            ax3 = fig_2d.add_subplot(1, n_2d, 2)
            ax3.set_facecolor("white")
            W_vals = points @ omega
            W_max = W_vals.max()
            W_min = W_vals.min()
            W_range = W_max - W_min + 1e-300
            sizes = np.maximum((W_vals - W_min) / W_range * 80, 6)
            sc3 = ax3.scatter(xyz[:, 0], xyz[:, 1], c=W_vals, s=sizes, cmap="viridis", alpha=0.85, edgecolors="none")
            plt.colorbar(sc3, ax=ax3, shrink=0.6, label="W(x)")
            for i, j in [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]:
                ax3.plot([verts_2d[i, 0], verts_2d[j, 0]], [verts_2d[i, 1], verts_2d[j, 1]],
                         "k-", linewidth=1.5, alpha=0.7)
            for i, v2 in enumerate(verts_2d):
                ax3.annotate(labels[i], v2, fontsize=10, fontweight="bold", xytext=(4, 4), textcoords="offset points")
            ax3.set_xlim(-0.08, 1.08)
            ax3.set_ylim(-0.08, np.sqrt(3) / 2 + 0.08)
            ax3.set_aspect("equal", adjustable="box")
            ax3.set_xlabel("x")
            ax3.set_ylabel("y")
            ax3.set_title("Fitness (2D projection)")

        plt.figure(fig_2d.number)
        plt.tight_layout()
        if out_path:
            p = Path(out_path)
            out_2d = str(p.parent / (p.stem + "_2d" + p.suffix))
            fig_2d.savefig(out_2d, dpi=150, bbox_inches="tight")
            print(f"Saved 2D to {out_2d}")
        if show:
            plt.figure(fig_2d.number)
            plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize equilibrium distribution for n=2 (3-simplex)."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--N-vis",
        type=int,
        default=None,
        help="Grid size for visualization (default: use N from config). Use smaller value (e.g. 10) for coarser grid.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=5000,
        help="Max points to plot when grid is large (default: 5000).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Save figures: plot.png -> plot_3d.png (3D) and plot_2d.png (2D projections).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plot (only save if --out given).",
    )
    parser.add_argument(
        "--mass-view",
        type=str,
        default="projection",
        choices=["projection", "sampled", "mcmc", "none"],
        help="Second panel: 'projection' = all lattice points, size & color ∝ ρ (default), "
             "'sampled' = sample ∝ ρ, 'mcmc' = MCMC samples, 'none' = single panel.",
    )
    parser.add_argument(
        "--mcmc-samples",
        type=int,
        default=5000,
        help="Number of MCMC samples for mass view (default: 5000).",
    )
    parser.add_argument(
        "--no-fitness",
        action="store_true",
        help="Do not show the fitness landscape panel.",
    )
    args = parser.parse_args()

    # Load config and build args
    config = load_config(args.config)
    parser_eq = argparse.ArgumentParser()
    parser_eq.add_argument("--config", default="")
    parser_eq.add_argument("--n-loci", type=int, default=2)
    parser_eq.add_argument("--N", type=int, default=20)
    parser_eq.add_argument("--u", type=float, default=1e-4)
    parser_eq.add_argument("--v", type=float, default=1e-4)
    parser_eq.add_argument("--seed", type=int, default=42)
    parser_eq.add_argument("--s-dist", default="fixed")
    parser_eq.add_argument("--s-fixed", type=float, default=0.1)
    parser_eq.add_argument("--s-mean", type=float, default=0.1)
    parser_eq.add_argument("--s-std", type=float, default=0.05)
    parser_eq.add_argument("--s-var", type=float, default=None)
    parser_eq.add_argument("--s-low", type=float, default=-0.1)
    parser_eq.add_argument("--s-high", type=float, default=0.2)
    parser_eq.add_argument("--eps-dist", default="fixed")
    parser_eq.add_argument("--eps-fixed", type=float, default=0.0)
    parser_eq.add_argument("--eps-mean", type=float, default=0.0)
    parser_eq.add_argument("--eps-std", type=float, default=0.05)
    parser_eq.add_argument("--eps-var", type=float, default=None)
    parser_eq.add_argument("--eps-low", type=float, default=-0.1)
    parser_eq.add_argument("--eps-high", type=float, default=0.1)
    parser_eq.add_argument("--sigma-file", default="")
    args_eq = parser_eq.parse_args([])
    apply_config_to_args(args_eq, config)

    if args_eq.n_loci != 2:
        raise ValueError(
            f"Visualization only supports n=2 loci (tetrahedron). Your config has n_loci={args_eq.n_loci}. "
            "Use a config with n_loci: 2, e.g. equilibria_config_viz_example.yaml"
        )

    N_vis = args.N_vis if args.N_vis is not None else args_eq.N
    args_eq.N = N_vis

    n, k, N_pop, Sigma, omega, alpha, G = build_sigma_omega_alpha(args_eq)

    # Print Σ and genotype fitnesses (like equilibria_explored)
    print("\nΣ (epistasis matrix):")
    np.set_printoptions(precision=4, suppress=True)
    print(Sigma)
    print("\nGenotypes and fitness ω(g) = 1 + gᵀ Σ g:")
    for i in range(k):
        g_str = "".join(str(int(b)) for b in G[i])
        print(f"  {g_str}  →  ω = {omega[i]:.6g}")
    print()

    print(f"Computing equilibrium grid: n={n}, k={k}, N={N_vis}")
    points, probs = equilibrium_distribution_grid(
        k, N_pop, omega, alpha, max_points=args.max_points
    )
    print(f"Plotting {len(points)} points")

    title = f"Equilibrium distribution (n=2, N={N_vis})"
    if args.no_show:
        import matplotlib
        matplotlib.use("Agg")
    mass_view = args.mass_view if args.mass_view != "none" else ""
    plot_equilibrium(
        points,
        probs,
        out_path=args.out if args.out else None,
        title=title,
        show=not args.no_show,
        omega=omega,
        G=G,
        Sigma=Sigma,
        N=N_vis,
        u=args_eq.u,
        v=args_eq.v,
        mass_view=mass_view,
        alpha=alpha,
        mcmc_samples=args.mcmc_samples,
        mcmc_seed=args_eq.seed,
        show_fitness=not args.no_fitness,
    )


if __name__ == "__main__":
    main()
# Example: python equilibria_viz.py --config equilibria_config_viz_example.yaml