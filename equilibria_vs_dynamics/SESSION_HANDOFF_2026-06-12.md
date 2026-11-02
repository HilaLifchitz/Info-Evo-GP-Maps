# Session handoff — 2026-06-12 (equilibrium information vs adaptation rate)

## Where we paused

User had to leave mid-investigation. We were **returning to the question:**
does equilibrium information KL(psi*||phi) predict how fast the chain adapts
from the neutral start psi(0)=phi — especially over the **first ~20 generations**?

A background re-run was started but is likely redundant (see below).

# What already exists in this folder

## Scripts

| File | Purpose |
|------|---------|
| `eps12_score_vs_speed.py` | Coarse eps12 sweep [-1,1] step 0.1; s1=-0.02, s2=-0.01; N=30; compares discrete psi* vs LE diffusion equilibrium scores vs **initial** dKL/dt; trajectory plots |
| `eps12_predictiveness_grid.py` | **Better tool for current question.** Grid of 9 (s1,s2) backgrounds, fine eps12 in [-0.02,0.02]; measures KL_eq, init_slope, **early_avg over first 20 gens**, t_half; Spearman per background |
| `README.md` | Summary of score-vs-speed conclusions (June 1–2 runs) |

## Saved results (under `results/` — PNGs not in git)

### Predictiveness grid (June 5, 2026) — **USE THIS FIRST**
`results/pred_grid/`
- `stage_a.csv` — full table (9 backgrounds × 11 eps12)
- `stage_a_spearman.csv` — rank correlations
- `stage_a_scatter.png` — KL_eq vs initial slope per background
- `stage_a_dip.png` — KL_eq and rate vs eps12 per background
- `stage_a_spearman.png` — bar chart of rho values
- Log: `results/pred_grid_run.log`

Settings: N=30, u=v=1e-4, r=0, |s| in {0.005, 0.01}, EARLY_WINDOW=20.

**Spearman rho(KL_eq, rate) — already computed:**

| background | rho(slope, gen 1) | rho(early avg, ≤20 gen) |
|------------|-------------------|-------------------------|
| s5_pp | +0.96 | +0.96 |
| s5_pm | +0.99 | +0.93 |
| s5_mp | +0.99 | +0.93 |
| s5_mm | +0.97 | +0.97 |
| s10_pp | +0.92 | +0.92 |
| s10_pm | +0.96 | +0.89 |
| s10_mp | +0.97 | +0.89 |
| s10_mm | +0.85 | +0.85 |
| asym_pm | +0.98 | +0.98 |

**Verdict from June 5 run (weak selection, fine eps12 grid):**
KL_eq is a **strong rank predictor** of early adaptation rate (rho ≈ 0.85–0.99).
The dip in KL_eq vs eps12 aligns with the dip in rate (min slope at eps12 ≈ 0
or slightly offset). t_half was always inf (adaptation too slow to reach 50%
of KL_eq within TIMES max=800).

### Trajectory plots (June 2) — coarse regime, N=20
`results/traj_by_score_N20_t800/`
- `info_kl_vs_t.png`, `info_kl_rate_vs_t.png` — KL(t) and dKL/dt, colored by KL_eq
- `*_byFF.png` — same colored by free fitness (shows U-shaped speed)
- `equilibrium_vs_eps12.png` — F_N monotone, KL_eq U-shaped

### Other
- `results/dip_scan_N20/` — earlier dip-related scan
- `results/traj_by_score_N6/`, `traj_by_score_N20_t500/`, etc.

## Prior conclusions (README + SESSION_HANDOFF 2026-06-02)

From **`eps12_score_vs_speed.py`** (coarse eps12, stronger effective selection
Ns ~ 0.3–0.6 with s1=-0.02):

1. **Information dip:** KL_eq is U-shaped in eps12; minimum near eps12 ≈ +0.02–0.03
   (not exactly 0). At the dip, adaptation is slowest (flattest KL(t), lowest dKL/dt).

2. **Not a perfect monotone map:** extremes track (high KL_eq → fast, dip → slow),
   but many eps12<0 curves share ~same KL_eq yet different transient speeds.

3. **Free fitness F_N is monotone in eps12** while KL_eq is U-shaped; coloring
   rates by F_N shows U-shaped speed (fast at both F extremes, slow at dip).

4. **Discrete vs diffusion equilibrium scores:** trend agrees, distributional match
   poor at r=0 (diffusion assumes LE). For **scalar** KL_eq comparison, diffusion
   is secondary; dynamics are always simulated on the **discrete chain**.

5. **Markov chain KL_eq ≈ diffusion KL_eq** only in weak selection / high-r LE
   limit; at r=0 use discrete psi* as ground truth.

# Reinterpretation for today's question

## What is "equilibrium information"?
**KL_eq = KL(psi* || phi)** [bits], where psi* is the selected stationary
distribution (ARPACK on transition matrix) and phi is the neutral stationary
distribution. This is the information content of the equilibrium relative to neutrality.

## What is "adaptation speed"?
Replay chain dynamics from psi(0)=phi. Two metrics in code:

1. **init_slope_kl** = dKL/dt over the first generation (interval 0→1)
2. **early_avg_kl** = mean dKL/dt over all consecutive intervals with t_end ≤ 20
   (`EARLY_WINDOW=20` in `eps12_predictiveness_grid.py`)

Rates are finite-difference slopes between sampled times in `dynamics_series()`
(`simplex.py` → `consecutive_rates`).

## Is KL_eq predictive of the first 20 steps?

**In the weak-selection fine-grid regime (pred_grid, June 5): YES, strongly.**
Spearman rho ≈ 0.85–0.99 between KL_eq and both gen-1 slope and 20-gen average.

Mechanism: within a fixed (s1,s2) background, varying eps12 moves both the
equilibrium information AND the initial flux of probability toward psi* in tandem.
The **dip** (min KL_eq at eps12 near 0) is where both equilibrium information
and early accumulation rate are smallest.

**Caveats:**
- Predictiveness is tested as **rank correlation across eps12**, not absolute
  calibration (KL_eq magnitude vs KL(20)).
- Mixed-sign backgrounds (s10_pm, s10_mp) show slightly lower rho for early_avg
  (0.89) than gen-1 slope (0.96–0.97) — worth checking if first-gen spike differs
  from 20-gen average.
- Coarse strong-selection trajectories (N=20, s=-0.02) showed messier behavior
  (crossing curves, identical KL_eq different speeds) — different parameter regime.
- t_half not reached in weak regime within 800 gens — early slope is the right metric.

# Background job started today (may be redundant)

```bash
cd equilibria_vs_dynamics
python eps12_predictiveness_grid.py --out-dir results/pred_grid_early20
```

Started ~2026-06-12 16:02. Same script/settings as June 5 `pred_grid` run.
Output dir `results/pred_grid_early20/` was empty when user left; process may
still be running (matrix builds are slow, conda buffering may delay log output).

**On return:** check if finished; if duplicate of `pred_grid/`, safe to kill.
```bash
ls equilibria_vs_dynamics/results/pred_grid_early20/
# or compare to existing pred_grid/
```

# Suggested next steps when you return

1. **Open existing plots** (no recompute needed):
   ```bash
   # scatter: KL_eq vs rate per background
   xdg-open equilibria_vs_dynamics/results/pred_grid/stage_a_scatter.png
   xdg-open equilibria_vs_dynamics/results/pred_grid/stage_a_dip.png
   xdg-open equilibria_vs_dynamics/results/traj_by_score_N20_t800/info_kl_rate_vs_t.png
   ```

2. **Quantify first-20 specifically** — add to analysis (small code change):
   - KL(20) vs KL_eq scatter
   - slope_0_20 = KL(20)/20 vs KL_eq
   - compare rho(KL_eq, init_slope) vs rho(KL_eq, early_avg) vs rho(KL_eq, KL(20))

3. **Bridge to eps12 dip work** (`2loci/`): the predictiveness dip at eps12≈0
   may relate to the KL(psi*||phi) dip location eps12* = -Cov/Var. Test whether
   eps12 at min(KL_eq) ≈ eps12 at min slope ≈ Cov/Var prediction.

4. **Optional:** rerun `eps12_score_vs_speed.py` trajectories with TIMES
   including 1..20 densely for the coarse regime comparison.

5. **Stage B** in predictiveness_grid (`SENSITIVITY=True`): vary u,v,r for one
   background — does predictiveness survive stronger mutation / recombination?

# How to run

```bash
conda activate miso   # or: source ~/miniconda3/etc/profile.d/conda.sh

cd equilibria_vs_dynamics

# predictiveness grid (9 backgrounds, fine eps12, early window=20)
python eps12_predictiveness_grid.py --out-dir results/pred_grid

# coarse eps12 sweep + diffusion comparison
python eps12_score_vs_speed.py --out-dir results/my_run

# trajectories colored by equilibrium score (edit script to call run_trajectories)
```

# Related handoffs

- `../SESSION_HANDOFF_2026-06-02.md` — diffusion validation + first score-vs-speed work
- `../2loci/SESSION_HANDOFF_2026-06-12.md` — eps12 dip panels, Cov/Var check, conditional alpha
