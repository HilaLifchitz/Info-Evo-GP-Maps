#!/usr/bin/env python3
"""Compare F2 with KL/N vs KL/(2N) on the compare_sigma config."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from compare_sigma_dynamics import (  # noqa: E402
    _make_cfg,
    expectation,
    kl_divergence_nats,
    log_mean_fitness_nats,
    run_one_sigma,
)
from simplex import (  # noqa: E402
    mutation_matrix_asymmetric,
    params_from_config,
    compute_phi,
    fitness_exp,
    build_transition_matrix_row_stochastic,
    column_stochastic,
)


def mono(values: np.ndarray, tol: float = 1e-12) -> tuple[bool, int, float]:
    d = np.diff(values)
    n_bad = int(np.sum(d < -tol))
    return n_bad == 0, n_bad, float(values[-1] - values[0])


def f2_series(phi, wstar, P, times, N: int, kl_div: float) -> np.ndarray:
    psi = phi.copy()
    prev = 0
    out = []
    for t in times:
        if t > prev:
            for _ in range(t - prev):
                psi = P @ psi
                s = float(psi.sum())
                if s > 0:
                    psi /= s
            prev = t
        ws = expectation(psi, wstar)
        kl = kl_divergence_nats(psi, phi)
        out.append(ws - kl / kl_div)
    return np.asarray(out)


def main() -> None:
    cfg_path = SCRIPT_DIR / "compare_sigma_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    N = int(cfg["N"])
    r = float(cfg["r"])
    u = float(cfg["u"][0])
    v = float(cfg["v"][0])
    times = [int(t) for t in cfg["times"]]

    out_dir = SCRIPT_DIR / "plots" / "compare_sigma"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(cfg["compare"]), 2, figsize=(11, 4 * len(cfg["compare"])), sharex=True)
    if len(cfg["compare"]) == 1:
        axes = np.array([axes])

    print(f"Population size N={N}  (formula uses divisor N vs 2N, not changing N itself)\n")
    print(f"{'run':8s}  {'/N mono?':10s}  {'drops':>5s}  {'net Δ':>12s}  |  {'/2N mono?':10s}  {'drops':>5s}  {'net Δ':>12s}")
    print("-" * 72)

    SD = SCRIPT_DIR
    M = mutation_matrix_asymmetric(np.array([u, u]), np.array([v, v]))
    params0 = params_from_config(_make_cfg(np.zeros((2, 2)), N, u, v, r, [0]), SD)
    phi, _, fracs, _, _ = compute_phi(params0, M)

    for row, run in enumerate(cfg["compare"]):
        label = run["label"]
        sigma = np.asarray(run["sigma"], dtype=float)
        omega = fitness_exp(sigma)
        wstar = log_mean_fitness_nats(fracs, omega)
        P = column_stochastic(
            build_transition_matrix_row_stochastic(
                N, omega, M, r, n_jobs=-1, label=label, verbose=False
            )[0]
        )

        f2_n = f2_series(phi, wstar, P, times, N, float(N))
        f2_2n = f2_series(phi, wstar, P, times, N, 2.0 * float(N))

        ok_n, bad_n, net_n = mono(f2_n)
        ok_2, bad_2, net_2 = mono(f2_2n)
        print(
            f"{label:8s}  {'yes' if ok_n else 'NO':10s}  {bad_n:5d}  {net_n:12.6f}  |  "
            f"{'yes' if ok_2 else 'NO':10s}  {bad_2:5d}  {net_2:12.6f}"
        )

        ax0, ax1 = axes[row, 0], axes[row, 1]
        ax0.plot(times, f2_n, "o-", ms=4, lw=1.5, label=rf"$F_2=\langle W^*\rangle - KL/N$")
        ax0.set_ylabel(rf"{label}  $F_2$ [nats]", fontsize=9)
        ax0.grid(True, alpha=0.3)
        ax0.legend(fontsize=8)
        status = "monotone ↑" if ok_n else f"{bad_n} drops"
        ax0.set_title(rf"KL divisor $N={N}$  ({status})", fontsize=9)

        ax1.plot(times, f2_2n, "s-", ms=4, lw=1.5, color="C1",
                 label=rf"$F_2=\langle W^*\rangle - KL/(2N)$")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        status2 = "monotone ↑" if ok_2 else f"{bad_2} drops"
        ax1.set_title(rf"KL divisor $2N={2*N}$  ({status2})", fontsize=9)

    for ax in axes[-1, :]:
        ax.set_xlabel("generation $t$")

    fig.suptitle(
        rf"F2: same $\langle W^*\rangle$, only KL divisor changes ($N={N}$, $u=v={u:g}$)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / "compare_f2_N_vs_2N.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
