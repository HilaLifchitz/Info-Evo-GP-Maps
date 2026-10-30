"""
Plot  f(ψ) = ψ log2(N ψ) + (1 − ψ) log2( N(1 − ψ) / (N − 1) )  for fixed N, ψ ∈ [0, 1].

Requires N > 1. Endpoints: ψ=0 and ψ=1 handled without NaNs.
"""

import numpy as np
import matplotlib.pyplot as plt


def f_psi(psi: np.ndarray, N: float) -> np.ndarray:
    """
    ψ log2(Nψ) + (1−ψ) log2( N(1−ψ)/(N−1) ).
    """
    if N <= 1:
        raise ValueError("N must be > 1 (denominator N−1).")
    psi = np.asarray(psi, dtype=float)
    tiny = np.finfo(float).tiny
    p = psi
    t1 = np.where(p > 0, p * np.log2(np.maximum(N * p, tiny)), 0.0)
    inner = N * (1.0 - p) / (N - 1.0)
    t2 = np.where((1.0 - p) > 0, (1.0 - p) * np.log2(np.maximum(inner, tiny)), 0.0)
    return t1 + t2


def main() -> None:
    N = 100.0
    n_pts = 2001
    psi = np.linspace(0.0, 1.0, n_pts)
    y = f_psi(psi, N)

    plt.figure(figsize=(8, 4.5))
    plt.plot(psi, y, color="C0", linewidth=1.5)
    plt.xlabel(r"$\psi_{\mathrm{fix}}$")
    plt.ylabel(
        r"$\psi_{\mathrm{fix}}\log_2(N\psi_{\mathrm{fix}})"
        r" + (1-\psi_{\mathrm{fix}})\log_2\left(\frac{N(1-\psi_{\mathrm{fix}})}{N-1}\right)$"
    )
    plt.title(rf"$N = {int(N)}$")
    plt.grid(True, alpha=0.35)
    plt.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
