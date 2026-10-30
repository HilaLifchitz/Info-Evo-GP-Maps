"""0/1/2 accessible-route partition of the iso-peak (ε/a, α/a) plane.

A route 00→11 is accessible if both mutational steps increase log-fitness.
"""
from __future__ import annotations

import numpy as np

# default axis window (a-units)
ALPHA_OA_MIN, ALPHA_OA_MAX = -0.15, 2.0
EPS_OA_MIN, EPS_OA_MAX = -4.0, 4.0


def n_mono_routes(eps_oa, alpha_oa) -> np.ndarray:
    """Number of fitness-increasing paths 00→11 (0, 1, or 2)."""
    e = np.asarray(eps_oa, dtype=float)
    al = np.asarray(alpha_oa, dtype=float)
    s1 = (2.0 - e + al) / 2.0
    s2 = (2.0 - e - al) / 2.0
    path10 = (s1 > 0) & ((s2 + e) > 0)
    path01 = (s2 > 0) & ((s1 + e) > 0)
    return path10.astype(int) + path01.astype(int)


def draw_route_partition(ax, show_labels: bool = True,
                         alpha_min: float = ALPHA_OA_MIN,
                         alpha_max: float = ALPHA_OA_MAX,
                         eps_min: float = EPS_OA_MIN,
                         eps_max: float = EPS_OA_MAX) -> None:
    """Overlay 0/1/2-route provinces (tint, dashed borders, labels)."""
    e = np.linspace(eps_min, eps_max, 500)
    al = np.linspace(alpha_min, alpha_max, 300)
    EE, AA = np.meshgrid(e, al)
    Nmono = n_mono_routes(EE, AA)
    ax.contourf(
        EE, AA, Nmono, levels=[-0.5, 0.5, 1.5, 2.5],
        colors=["#3b0a45", "#1f6aa5", "#2a9d5c"],
        alpha=0.12, zorder=1,
    )
    ax.contour(
        EE, AA, Nmono, levels=[0.5, 1.5],
        colors="white", linewidths=1.35, linestyles="--",
        zorder=4,
    )
    if show_labels:
        for (x, y, txt) in [
            (0.0, 0.40, "2 routes"),
            (-2.4, 1.35, "1 route"),
            (2.4, 1.35, "1 route"),
            (-3.35, 0.35, "0 routes"),
            (3.35, 0.35, "0 routes"),
        ]:
            want = {"2 routes": 2, "1 route": 1, "0 routes": 0}[txt]
            if n_mono_routes(x, y) != want:
                continue
            if not (eps_min <= x <= eps_max and alpha_min <= y <= alpha_max):
                continue
            ax.text(
                x, y, txt, color="white", fontsize=7, fontweight="bold",
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="black",
                          ec="none", alpha=0.40),
            )
