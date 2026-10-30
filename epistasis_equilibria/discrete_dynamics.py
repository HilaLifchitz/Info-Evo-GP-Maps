"""
Discrete dynamical system: x_{t+1} = x_t + u(1 - x_t) - v x_t, with y = 1 - x.
"""

import matplotlib.pyplot as plt


def simulate(x0: float, u: float, v: float, n_steps: int) -> tuple[list[int], list[float]]:
    """Return time indices and x values for t = 0, 1, ..., n_steps."""
    x = x0
    times = [0]
    xs = [x]
    for t in range(1, n_steps + 1):
        x = x + u * (1.0 - x) - v * x
        times.append(t)
        xs.append(x)
    return times, xs

def plot_trajectory(times: list[int], xs: list[float]) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(times, xs, marker="o", markersize=2, linewidth=1, color="blue")
    plt.xlabel("time")
    plt.ylabel("x")
    plt.title(r"$x_{t+1} = x_t + u(1-x_t) - v x_t$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    # --- parameters (edit these) ---
    x0 = 0.2
    u = 0.05
    v = 0.05
    n_steps = 100

    times, xs = simulate(x0, u, v, n_steps)

    plot_trajectory(times, xs)


if __name__ == "__main__":
    main()
