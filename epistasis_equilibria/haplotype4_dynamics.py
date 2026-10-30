"""
Four haplotype frequencies [00],[01],[11],[10] as x1,x2,x3,x4 with x4 = 1 - x1 - x2 - x3.
Mutation: u is [0]->[1], v is [1]->[0].
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# Update rules (x4 = 1 - x1 - x2 - x3 at each step before applying the map):
#
#   x1 <- x1 - u*x1 + v*(x2/2 + x4/2)
#   x2 <- x2 - u*x2/2 - v*x2/2 + u*x1/2 + v*x3/2
#   x3 <- x3 - v*x3 + u*x4/2 + u*x2/2
#   x4 <- x4 - u*x4/2 - v*x4/2 + u*x1/2 + v*x3/2


def step(
    x1: float,
    x2: float,
    x3: float,
    x4: float,
    u: float,
    v: float,
) -> tuple[float, float, float, float]:
    """One discrete update; x4 must be 1 - x1 - x2 - x3."""
    x1n = x1 - u * x1 + v * (x2 / 2 + x4 / 2)
    x2n = x2 - u * x2 / 2 - v * x2 / 2 + u * x1 / 2 + v * x3 / 2
    x3n = x3 - v * x3 + u * x4 / 2 + u * x2 / 2
    x4n = x4 - u * x4 / 2 - v * x4 / 2 + u * x1 / 2 + v * x3 / 2
    return x1n, x2n, x3n, x4n


def simulate(
    x1_0: float,
    x2_0: float,
    x3_0: float,
    u: float,
    v: float,
    n_steps: int,
) -> tuple[list[int], list[float], list[float], list[float], list[float]]:
    """
    Initial simplex point (x1_0, x2_0, x3_0); x4_0 = 1 - x1_0 - x2_0 - x3_0.
    Returns time and x1,x2,x3,x4 trajectories (length n_steps + 1).
    """
    x1, x2, x3 = x1_0, x2_0, x3_0
    x4 = 1.0 - x1 - x2 - x3

    times = [0]
    s1, s2, s3, s4 = [x1], [x2], [x3], [x4]

    for t in range(1, n_steps + 1):
        x1, x2, x3, x4 = step(x1, x2, x3, x4, u, v)
        times.append(t)
        s1.append(x1)
        s2.append(x2)
        s3.append(x3)
        s4.append(x4)

    return times, s1, s2, s3, s4


def print_final_frequencies(s1: list[float], s2: list[float], s3: list[float], s4: list[float]) -> None:
    """Print the last simulated values of x1, x2, x3, x4."""
    x1, x2, x3, x4 = s1[-1], s2[-1], s3[-1], s4[-1]
    print(f"Final frequencies: x1 = {x1}, x2 = {x2}, x3 = {x3}, x4 = {x4}")


def plot_trajectories(
    times: list[int],
    s1: list[float],
    s2: list[float],
    s3: list[float],
    s4: list[float],
    u: float,
    v: float,
) -> None:
    # Explicit colors/linestyles so curves do not read as a single trace when values are close.
    series = [
        (s1, r"$x_1$ [00]", "#1f77b4", "-"),
        (s2, r"$x_2$ [01]", "#ff7f0e", "-"),
        (s3, r"$x_3$ [11]", "#2ca02c", "--"),
        (s4, r"$x_4$ [10]", "#d62728", "-."),
    ]
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    for ys, lab, color, ls in series:
        plt.plot(
            times,
            ys,
            label=lab,
            color=color,
            linestyle=ls,
            linewidth=2.0,
            marker="o",
            markersize=3,
            markevery=max(1, len(times) // 40),
        )
    plt.xlabel("time")
    plt.ylabel("frequency")
    plt.title("Haplotype frequencies")
    param_lines = f"u [0] --> [1] = {u:g}\nv [1] --> [0] = {v:g}"
    ax.text(
        0.02,
        0.98,
        param_lines,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7", alpha=0.92),
    )
    plt.legend(loc="best", framealpha=0.95)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.show()


def main() -> None:
    # --- parameters (edit these) ---
    # Use non-symmetric initials: with x1=x2=x3=x4=0.25 and u=v, all four stay equal and overlap.
    x1_0 = 0.05
    x2_0 = 0.2
    x3_0 = 0.35  # x4_0 = 1 - x1 - x2 - x3
    u = 0.006  # [0]->[1]
    v = 0.01  # [1]->[0]
    n_steps = 500

    times, s1, s2, s3, s4 = simulate(x1_0, x2_0, x3_0, u, v, n_steps)
    print_final_frequencies(s1, s2, s3, s4)
    plot_trajectories(times, s1, s2, s3, s4, u, v)


if __name__ == "__main__":
    main()
