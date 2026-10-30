#!/usr/bin/env python3
r"""
F2b-A / F2b-B monotonicity and decomposition check.

  F2b = ⟨ln W⟩_ψ − (1/N) D_KL(ψ || φ_ref)   [nats]

  F2b-A: φ_ref = fixed cond-α exponents at x̄ = E_{φ_chain}[x],  α = 2N·influx/(1−x_g)
  F2b-B: φ_ref = normalized state-dependent 2N cond-α neutral density on lattice

Run:
  cd 2loci
  python f2b_check.py
  python f2b_check.py --config compare_sigma_config.yaml
  python f2b_check.py --eps12-sweep --config eps12_delta_d_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_sigma_dynamics import (  # noqa: E402
    _make_cfg,
    expectation,
    kl_divergence_nats,
    log_mean_fitness_nats,
)
from f2b_reference import build_phi_f2b_a, build_phi_f2b_b, free_f2b, kl_to_ref_nats  # noqa: E402
from simplex import (  # noqa: E402
    build_transition_matrix_row_stochastic,
    column_stochastic,
    compute_phi,
    fitness_exp,
    mutation_matrix_asymmetric,
    params_from_config,
    try_selection_equilibrium,
)

OUT_DIR = SCRIPT_DIR / "plots" / "f2b"


def _mono(values: np.ndarray, tol: float = 1e-12) -> tuple[bool, int]:
    d = np.diff(values)
    n_bad = int(np.sum(d < -tol))
    return n_bad == 0, n_bad


def _track(
    phi0: np.ndarray,
    P_col: np.ndarray,
    wstar_ln: np.ndarray,
    phi_chain: np.ndarray,
    phi_f2b_a: np.ndarray,
    phi_f2b_b: np.ndarray,
    psi_star: np.ndarray,
    N: int,
    max_t: int,
) -> dict[str, np.ndarray]:
    n = max_t + 1
    t = np.arange(n, dtype=float)
    w = np.empty(n)
    f2 = np.empty(n)
    f2a = np.empty(n)
    f2b = np.empty(n)
    f4 = np.empty(n)
    kl_chain = np.empty(n)
    kl_a = np.empty(n)
    kl_b = np.empty(n)
    inv_n = 1.0 / float(N)

    psi = phi0.copy()
    for i in range(n):
        w[i] = expectation(psi, wstar_ln)
        kl_chain[i] = kl_to_ref_nats(psi, phi_chain)
        kl_a[i] = kl_to_ref_nats(psi, phi_f2b_a)
        kl_b[i] = kl_to_ref_nats(psi, phi_f2b_b)
        f2[i] = w[i] - kl_chain[i] * inv_n
        f2a[i] = free_f2b(psi, wstar_ln, phi_f2b_a, N)
        f2b[i] = free_f2b(psi, wstar_ln, phi_f2b_b, N)
        f4[i] = -kl_to_ref_nats(psi, psi_star) * inv_n
        if i < max_t:
            psi = P_col @ psi
            s = float(psi.sum())
            if s > 0:
                psi /= s
    return dict(
        t=t, w=w, f2=f2, f2a=f2a, f2b=f2b, f4=f4,
        kl_chain=kl_chain, kl_a=kl_a, kl_b=kl_b,
    )


def _report(label: str, tr: dict[str, np.ndarray]) -> None:
    print(f"\n--- {label} ---", flush=True)
    for name, key in [
        ("F2 (chain φ*)", "f2"),
        ("F2b-A", "f2a"),
        ("F2b-B", "f2b"),
        ("F4", "f4"),
    ]:
        ok, n_bad = _mono(tr[key])
        print(f"  {name}: {'monotone ↑' if ok else f'NOT monotone ({n_bad} drops)'}", flush=True)
    dw = np.diff(tr["w"])
    for name, kl_key in [
        ("KL(ψ||φ_chain)", "kl_chain"),
        ("KL(ψ||φ_F2b-A)", "kl_a"),
        ("KL(ψ||φ_F2b-B)", "kl_b"),
    ]:
        dk = np.diff(tr[kl_key])
        up = int(np.sum(dk > 1e-14))
        dn = int(np.sum(dk < -1e-14))
        print(f"  Δ{name}: {up} up / {dn} down steps (of {len(dk)})", flush=True)
    w_scale = max(float(np.max(np.abs(tr["w"]))), 1e-12)
    flat = np.abs(dw) < 1e-4 * w_scale
    print(
        f"  late stage: flat-W* & rising KL_A: "
        f"{int(np.sum(flat & (np.diff(tr['kl_a']) > 1e-14)))}/{len(dw)} steps",
        flush=True,
    )


def _plot_decomposition(
    tr: dict[str, np.ndarray],
    out_path: Path,
    *,
    title: str,
    ref_label: str,
    f_key: str,
    kl_key: str,
    N: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = tr["t"]
    w = tr["w"]
    neg_kl = -tr[kl_key] / float(N)
    f = tr[f_key]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].plot(t, w, "C0", lw=1.5)
    axes[0].set_title(r"$\langle \ln W\rangle_\psi$")
    axes[0].set_ylabel("[nats]")

    axes[1].plot(t, neg_kl, "C3", lw=1.5)
    axes[1].set_title(rf"$-\frac{{1}}{{N}}D_{{\mathrm{{KL}}}}(\psi\|{ref_label})$")

    axes[2].plot(t, f, "C2", lw=1.5, label=f_key)
    axes[2].plot(t, w, "--", color="C0", alpha=0.45, lw=1)
    axes[2].plot(t, neg_kl, "--", color="C3", alpha=0.45, lw=1)
    axes[2].set_title(f"{f_key} and components")
    axes[2].legend(fontsize=7)

    for ax in axes:
        ax.set_xlabel("generation $t$")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)


def run_sigma_compare(cfg_path: Path, out_dir: Path, pergen_max_t: int) -> None:
    cfg = yaml.safe_load(cfg_path.read_text())
    N = int(cfg["N"])
    r = float(cfg["r"])
    u = float(cfg["u"][0])
    v = float(cfg["v"][0])

    M_mut = mutation_matrix_asymmetric(np.array([u, u]), np.array([v, v]))
    params0 = params_from_config(_make_cfg(np.zeros((2, 2)), N, u, v, r, [0]), SCRIPT_DIR)
    phi_chain, _, fracs, _, _ = compute_phi(params0, M_mut)
    phi_f2b_a = build_phi_f2b_a(fracs, M_mut, N, phi_chain)
    phi_f2b_b = build_phi_f2b_b(fracs, M_mut, N)

    print(f"compare_sigma F2b check  N={N}  u=v={u:g}  T={pergen_max_t}", flush=True)
    print(f"  TV(φ_chain, φ_F2b-A) = {0.5*np.sum(np.abs(phi_chain-phi_f2b_a)):.4f}", flush=True)
    print(f"  TV(φ_chain, φ_F2b-B) = {0.5*np.sum(np.abs(phi_chain-phi_f2b_b)):.4f}", flush=True)
    print(f"  TV(φ_F2b-A, φ_F2b-B) = {0.5*np.sum(np.abs(phi_f2b_a-phi_f2b_b)):.4f}", flush=True)

    fig_data = []
    for run in cfg["compare"]:
        label = run["label"]
        sigma = np.asarray(run["sigma"], dtype=float)
        omega = fitness_exp(sigma)
        wstar = log_mean_fitness_nats(fracs, omega)
        T_row, _, _ = build_transition_matrix_row_stochastic(
            N, omega, M_mut, r, n_jobs=-1, label=label, verbose=False,
        )
        P_col = column_stochastic(T_row)
        psi_star, _, ok = try_selection_equilibrium(P_col)
        if not ok or psi_star is None:
            raise RuntimeError(f"{label}: psi* not found")

        tr = _track(phi_chain.copy(), P_col, wstar, phi_chain, phi_f2b_a, phi_f2b_b, psi_star, N, pergen_max_t)
        _report(label, tr)
        fig_data.append((label, tr))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(fig_data), 4, figsize=(14, 3.2 * len(fig_data)), squeeze=False)
    for row, (label, tr) in enumerate(fig_data):
        for col, (key, ylab) in enumerate([
            ("f2", r"$F_2$"),
            ("f2a", r"$F_{2b-A}$"),
            ("f2b", r"$F_{2b-B}$"),
            ("f4", r"$F_4$"),
        ]):
            ax = axes[row, col]
            ax.plot(tr["t"], tr[key], lw=1.4)
            ok, n_bad = _mono(tr[key])
            ax.set_title(f"{label} {ylab}\n{'↑ mono' if ok else f'{n_bad} drops'}", fontsize=8)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel("[nats]")
            if row == len(fig_data) - 1:
                ax.set_xlabel("$t$")
    fig.suptitle(rf"F2 / F2b-A / F2b-B / F4 ($N={N}$, KL$/N$)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = out_dir / "compare_sigma_f2b_monotone.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {p}", flush=True)

    for label, tr in fig_data:
        tag = label.strip("()").replace(" ", "")
        _plot_decomposition(
            tr, out_dir / f"decomp_f2bA_{tag}.png",
            title=f"{label}: F2b-A decomposition",
            ref_label=r"\varphi^*_{\mathrm{F2b-A}}",
            f_key="f2a", kl_key="kl_a", N=N,
        )
        _plot_decomposition(
            tr, out_dir / f"decomp_f2bB_{tag}.png",
            title=f"{label}: F2b-B decomposition",
            ref_label=r"\varphi^*_{\mathrm{F2b-B}}",
            f_key="f2b", kl_key="kl_b", N=N,
        )


def run_eps12_sweep(cfg_path: Path, out_dir: Path) -> None:
    from eps12_delta_d_sweep import _eps12_grid, _make_cfg as make_eps_cfg

    cfg = yaml.safe_load(cfg_path.read_text())
    N = int(cfg["N"])
    r = float(cfg["r"])
    u = float(cfg["u"][0])
    v = float(cfg["v"][0])
    s1 = float(cfg["s1"])
    s2 = float(cfg["s2"])
    T = int(cfg.get("pergen_max_t", 200))
    eps_grid = _eps12_grid(cfg)

    M_mut = mutation_matrix_asymmetric(np.array([u, u]), np.array([v, v]))
    params0 = params_from_config(make_eps_cfg(N=N, r=r, u=u, v=v, s1=s1, s2=s2, eps12=0.0), SCRIPT_DIR)
    phi_chain, _, fracs, _, _ = compute_phi(params0, M_mut)
    phi_f2b_a = build_phi_f2b_a(fracs, M_mut, N, phi_chain)
    phi_f2b_b = build_phi_f2b_b(fracs, M_mut, N)

    print(f"\neps12 sweep F2b  n={len(eps_grid)}  T={T}", flush=True)
    print(f"{'eps12':>7} {'F2':>8} {'F2b-A':>8} {'F2b-B':>8} {'F4':>8}  {'ΔKL_A':>12}", flush=True)
    results = []
    for eps in eps_grid:
        params = params_from_config(make_eps_cfg(N=N, r=r, u=u, v=v, s1=s1, s2=s2, eps12=float(eps)), SCRIPT_DIR)
        omega = fitness_exp(params.Sigma)
        wstar = log_mean_fitness_nats(fracs, omega)
        T_row, _, _ = build_transition_matrix_row_stochastic(
            N, omega, M_mut, r, n_jobs=-1, label=f"e{eps:.2g}", verbose=False,
        )
        P_col = column_stochastic(T_row)
        psi_star, _, ok = try_selection_equilibrium(P_col)
        if not ok:
            continue
        tr = _track(phi_chain.copy(), P_col, wstar, phi_chain, phi_f2b_a, phi_f2b_b, psi_star, N, T)
        flags = []
        for key in ("f2", "f2a", "f2b", "f4"):
            ok_m, _ = _mono(tr[key])
            flags.append("Y" if ok_m else "N")
        dk = np.diff(tr["kl_a"])
        results.append((eps, flags, int(np.sum(dk > 1e-14)), int(np.sum(dk < 1e-14))))
        print(f"{eps:+7.2f} {flags[0]:>8} {flags[1]:>8} {flags[2]:>8} {flags[3]:>8}  {results[-1][2]:>4}↑ {results[-1][3]:>4}↓", flush=True)

    n = len(results)
    for i, name in enumerate(["F2", "F2b-A", "F2b-B", "F4"]):
        print(f"  {name} monotone: {sum(r[1][i]=='Y' for r in results)}/{n}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="F2b-A/B monotonicity check")
    ap.add_argument("--config", type=Path, default=SCRIPT_DIR / "compare_sigma_config.yaml")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--pergen-max-t", type=int, default=200)
    ap.add_argument("--eps12-sweep", action="store_true")
    ap.add_argument("--eps12-config", type=Path, default=SCRIPT_DIR / "eps12_delta_d_config.yaml")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_sigma_compare(args.config, args.out_dir, args.pergen_max_t)
    if args.eps12_sweep:
        run_eps12_sweep(args.eps12_config, args.out_dir)
    print(f"\nDone. Outputs in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
