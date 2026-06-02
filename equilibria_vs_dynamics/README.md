# Equilibrium score vs. adaptation speed

Does the **equilibrium "score"** of an epistatic coefficient predict the **speed of adaptation**? This folder links the two sibling codebases — the discrete two-locus Markov chain (`../2loci/`) and the continuous allele-frequency diffusion equilibrium (`../epistasis_equilibria/`) — to test that question over a sweep of the epistasis coefficient \(\varepsilon_{12}\) (and recombination \(r\)).

The script puts both sibling folders on `sys.path`, so nothing is copied between directories.

## What it computes

For fixed \(s_1, s_2\) and each \(\varepsilon_{12}\) (and each \(r\)):

- **(A) Equilibrium scores** — how "good" the equilibrium is:
  - *discrete*: the chain's own selection equilibrium \(\psi^*\) → \(\mathrm{KL}(\psi^*\|\varphi)\) [bits], free fitness \(\mathcal{F}_N(\psi^*)\). (\(\psi^*\) solved via ARPACK, power-iteration fallback.)
  - *diffusion*: the haploid allele-frequency equilibrium → \(\mathrm{KL}(\rho_{\mathrm{eq}}\|\rho_{\mathrm{neutral}})\), \(\mathcal{F}_N\). (Haploid scaling, matched to the chain's \(N\), \(u\), \(v\).)
- **(B) Adaptation speed** — replay the chain dynamics from the neutral start \(\psi(0)=\varphi\) and take the **initial rate** \(\tfrac{d}{dt}\mathrm{KL}\), \(\tfrac{d}{dt}\mathcal{F}_N\) over the first generation.

Spearman rank correlation (per \(r\)) tests whether the *ordering* of equilibrium scores across \(\varepsilon_{12}\) matches the ordering of adaptation speed.

## File

| File | Purpose |
|------|---------|
| `eps12_score_vs_speed.py` | Orchestrator: scores + speeds + all plots. All settings in the `SETTINGS` block at top. |

Key functions: `run()` (scores+speeds table), `run_trajectories()` (full \(\mathrm{KL}(t)/\mathcal{F}_N(t)\) curves per \(\varepsilon_{12}\)), `plot_score_vs_speed()`, `plot_vs_eps12()`, `plot_trajectories_by_score()`, `plot_equilibrium_sweep_vs_eps12()`.

## Run

```bash
conda activate miso
cd equilibria_vs_dynamics
python eps12_score_vs_speed.py                 # full scores-vs-speed run + plots
python eps12_score_vs_speed.py --out-dir results/my_run
```

Trajectory / equilibrium-sweep plots are produced via `run_trajectories()` + the plot helpers (see the module for the small driver pattern; example outputs live under `results/traj_by_score_*`).

## Outputs (under `results/`)

- `score_vs_speed_discrete.png`, `score_vs_speed_diffusion.png` — 2×2 scatter (equilibrium score vs adaptation speed), colored by \(\varepsilon_{12}\), Spearman \(\rho\) annotated per \(r\).
- `score_and_speed_vs_eps12.png` — dual-axis: equilibrium KL (discrete & diffusion) and initial KL rate vs \(\varepsilon_{12}\).
- `traj_by_score_*/` — \(\mathrm{KL}(t)\), \(\mathcal{F}_N(t)\) and their rates, each curve **colored by its equilibrium score** (rank-based `turbo` coloring; legend by \(\varepsilon_{12}\)):
  - `info_kl_vs_t.png`, `free_fitness_vs_t.png`, `info_kl_rate_vs_t.png`, `free_fitness_rate_vs_t.png` (colored by \(\mathrm{KL}_{\mathrm{eq}}\))
  - `*_byFF.png` — rate panels colored by equilibrium **free fitness** \(\mathcal{F}_N(\psi^*)\)
  - `equilibrium_vs_eps12.png` — epistasis-style sweep: equilibrium \(\mathcal{F}_N\) (left axis) and \(\mathrm{KL}_{\mathrm{eq}}\) (right axis) vs \(\varepsilon_{12}\)

## Conclusions

- **An information "dip" exists and predicts slow adaptation.** Equilibrium information \(\mathrm{KL}_{\mathrm{eq}}\) is **U-shaped** in \(\varepsilon_{12}\): it bottoms out near \(\varepsilon_{12}\approx+0.02\)–\(0.03\) (not exactly 0; the minimum is set by the full nonlinear \(W^{2N}\) weighting, near where epistasis cancels the marginal selection). At that dip the chain barely adapts — \(\mathrm{KL}(t)\) is flattest and \(\tfrac{d}{dt}\mathrm{KL}\) lowest throughout (confirmed to \(T=800\), \(N=20\)).
- **Equilibrium score broadly tracks speed at the extremes, but is *not* a clean monotone predictor.** Highest-\(\mathrm{KL}_{\mathrm{eq}}\) curves adapt fastest; lowest adapt slowest. But (i) the many \(\varepsilon_{12}<0\) curves have **near-identical** \(\mathrm{KL}_{\mathrm{eq}}\) (\(\approx0.49\)) yet different transient speeds and cross over in time; and (ii) coloring by equilibrium **free fitness** \(\mathcal{F}_N\) (monotone in \(\varepsilon_{12}\)) reveals a **U-shaped** speed relation — both the lowest- and highest-\(\mathcal{F}_N\) curves adapt fast while mid-\(\mathcal{F}_N\) (the dip) is slowest. So equilibrium score captures the trend/extremes, not a one-to-one rank.
- **Free fitness is monotone in \(\varepsilon_{12}\)**, while equilibrium information is U-shaped (see `equilibrium_vs_eps12.png`).
- **Discrete vs diffusion scores agree on the trend but not distributionally.** The allele-frequency diffusion assumes **linkage equilibrium** (no LD), which clashes with the maximally-LD \(r=0\) chain; scalar scores (KL) are still comparable, but a distributional match is not expected. (See `../2loci/` for a detailed validation: the diffusion ↔ chain match is near-exact only for weak selection \(N s_{\max}=O(1)\), and each diffusion form matches its own recombination regime.)

## Settings (top of `eps12_score_vs_speed.py`)

`S1`, `S2`; `R_VALUES`; `EPS12_MIN/MAX/STEP`; `N_MARKOV`, `U_MARKOV`, `V_MARKOV`, `TIMES`; diffusion `N_POP_DIFF`, `U_DIFF`, `V_DIFF`, `DIFF_RESOLUTION`; toggles `COMPUTE_DISCRETE`, `COMPUTE_DIFFUSION`.
