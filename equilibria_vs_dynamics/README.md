# Equilibria vs dynamics

Does an **equilibrium score** of a two-locus landscape predict **how fast** the Wright–Fisher chain adapts from the neutral start \(\psi(0)=\varphi\)?

This folder is the glue between:

- discrete chain: [`../2loci/`](../2loci/) (`simplex.py`)
- continuous allele-frequency equilibrium: [`../epistasis_equilibria/`](../epistasis_equilibria/)

Scripts put both sibling folders on `sys.path`. Nothing is copied between directories.

```bash
conda activate miso
cd equilibria_vs_dynamics
```

PNG/CSV/NPZ under `results/` are gitignored (see repo `.gitignore`).

---

## Scientific questions (three waves)

1. **Score vs speed (June 2026).** For fixed \((s_1,s_2)\), sweep \(\varepsilon_{12}\). Does \(\mathrm{KL}(\psi^*\|\varphi)\) rank-order the initial \(\mathrm{d\,KL}/\mathrm{d}t\)? Answer: **yes at weak selection** (Spearman \(\rho\approx 0.85\)–\(0.99\)); **no as a clean monotone map** at coarser/stronger selection — high \(\mathrm{KL}_{\mathrm{eq}}\) is fast, the information dip is slow, but many \(\varepsilon_{12}<0\) curves share ~the same \(\mathrm{KL}_{\mathrm{eq}}\) and still differ in transients.
2. **Four (then six) canonical landscapes (July 2026).** Same endpoints \(\log\omega(00)=0\), \(\log\omega(11)=2a\); different single-mutant intermediates. Additive (B) has highest free fitness and fastest MFPT; reciprocal-sign valley (C) has most information and **slows down** as \(a\) grows.
3. **Node-hopping / SSWM.** At small \(Nu\) the simplex is a 4-corner graph. Path *accessibility* (not the equilibrium) sets speed: same \(\psi^*\) peak, different first-step supply.

---

## 1. Score vs speed

| File | Purpose |
|------|---------|
| `eps12_score_vs_speed.py` | Coarse \(\varepsilon_{12}\in[-1,1]\) at fixed \(s_1,s_2\); discrete vs diffusion scores vs initial rate; trajectory overlays |
| `eps12_predictiveness_grid.py` | 9 \((s_1,s_2)\) backgrounds, fine \(\varepsilon_{12}\); Spearman of \(\mathrm{KL}_{\mathrm{eq}}\) vs init slope / early-20 avg |
| `pred_trajectory_horizons.py` | User-designed backgrounds; dense \(t=1\ldots100\); Spearman vs horizons 20/50/100/200 |
| `pred_first20_analysis.py` | Replot / extend `pred_grid` CSV with \(\mathrm{KL}(20)\) |
| `info_speed_experiment.py` / `info_speed_analyze.py` | Larger sweep of the same claim |
| `dispute_info_speed.py` | Stress-test “lower \(D_{\mathrm{eq}}\) ⇒ faster equilibration” |
| `strategy_views.py` | Same data, split by number of alluring 00→11 routes |

**Typical run:**

```bash
python eps12_score_vs_speed.py --out-dir results/my_run
python eps12_predictiveness_grid.py --out-dir results/pred_grid
python pred_trajectory_horizons.py
```

Trajectory colorings live under `results/traj_by_score_*`:

- `info_kl_rate_vs_t.png` — color = \(\mathrm{KL}_{\mathrm{eq}}\) (the overlay from June 2, \(N=20\), \(T=800\))
- `*_byFF.png` — same curves colored by equilibrium **free fitness** \(\mathcal{F}_N(\psi^*)\), which is monotone in \(\varepsilon_{12}\) and makes the U-shaped speed relation obvious

---

## 2. Canonical landscapes (A–G)

Shared fitness convention: \(\Sigma=[[s_1,\varepsilon_{12}/2],[\varepsilon_{12}/2,s_2]]\), \(\log\omega = [0,\,s_2,\,s_1,\,s_1+s_2+\varepsilon_{12}]\).

| case | \((s_1,s_2,\varepsilon_{12})\) | intermediates | role |
|------|-------------------------------|---------------|------|
| **A** | \((a,-a,2a)\) | one up, one down | sign epistasis |
| **B** | \((a,a,0)\) | both up | additive |
| **C** | \((-a,-a,4a)\) | both down | reciprocal-sign valley |
| **D** | \((0,0,2a)\) | both flat | neutral intermediates |
| **F, G** | later | synergy / asymmetric additive | iso-peak extensions |

Headline free fitness (the monotone Lyapunov one, against the *true* chain \(\psi^*\)):

\[
F(\psi)=F^*-\frac{1}{N}D_{\mathrm{KL}}(\psi\|\psi^*),\qquad
F^*=\langle\log W\rangle_{\psi^*}-\frac{1}{N}D_{\mathrm{KL}}(\psi^*\|\varphi)
\]

| File | Purpose |
|------|---------|
| `compare_two_cases.py` | Original A vs B + F5 machinery, MFPT, cycle affinity |
| `adaptation_per_a.py` / `adaptation_per_u.py` | 3-panel \(D(t)\), \(\overline{\log W}\), \(F(t)\); one row per \(a\) or \(u\) |
| `summary_4cases.py` | Equilibrium \(I\) and \(F^*\) vs \(a\) and vs \(u\) |
| `mfpt_4cases.py` | Mean first passage to the 11 basin |
| `info_compare.py` | A vs B information vs \(a\), \(u\) |
| `compare_sweep.py` | Robustness of A/B over \(a\) and \(u\) |
| `canon_dynamics.py` | Iso-peak A–F dynamics: \(D\), cost, \(\eta\), \(F\) |
| `canonical_eff_experiment.py` / `canonical_eff_views.py` | Efficiency + speed sweep for A,B,C,D,F,G |
| `epistasis_landscapes_fig.py` | Sketch panels of the six landscapes |
| `ordering_*.py`, `bump_*.py`, `surprise_analysis*.py` | Rank stability, bump charts, \(\eta\) views |
| `eps_asym_surface.py` / `eps_asym_surface_chain.py` | Iso-peak \((\varepsilon,\alpha)\) surface (diffusion vs chain) |
| `route_partition.py` | Partition that surface by 0/1/2 accessible routes |

**Findings (N=30, see `SESSION_HANDOFF_2026-07-15.md`):** free-fitness order \(B>A>D>C\); information order \(C>D>A>B\) at modest \(u\); increasing \(u\) *widens* gaps, increasing \(a\) *closes* them; C’s MFPT **grows** with \(a\).

---

## 3. Node-jump / SSWM

| File | Purpose |
|------|---------|
| `node_jump_analysis.py` | Corner concentration, basin flow, lumped hop matrix \(Q\) vs analytic SSWM rates, detailed-balance residual |
| `basin_flow_u_r.py` | Corner mass vs \(u=v\) (controlled by \(\theta=2Nu\)) then vs \(r\) |

Default regime in those notes: \(N=24\), \(u=v=2\times10^{-4}\), \(r=0\). At that \(\theta\), \(\varphi\) is ~93% monomorphic; recombination is invisible until both loci segregate.

---

## Outputs

Example result trees (all under `results/`, not in git):

- `traj_by_score_N20_t800/` — June 2 trajectory overlays
- `pred_grid/`, `pred_traj_design/` — predictiveness tables and Spearman bars
- `compare_cases_*`, `adaptation_per_*`, `summary_4cases_*`, `mfpt_4cases_*`
- `node_jumps_sswm_*`, `basin_flow_u_r_*`

Session notes: `SESSION_HANDOFF_2026-06-12.md`, `SESSION_HANDOFF_2026-07-13.md`, `SESSION_HANDOFF_2026-07-15.md`.
