# Miso — information, epistasis, and Wright–Fisher dynamics

Python research code for **finite-population** evolutionary dynamics, with an emphasis on:

- exact two-locus Wright–Fisher on the genotype simplex
- information-theoretic scores ($\mathrm{KL}$, free fitness, cost $C$, efficiency $\eta$)
- how **epistasis** $\varepsilon_{12}$ shapes both the *equilibrium* and the *speed* of adaptation
- diffusion / Boltzmann approximations of the same objects

GitHub: [HilaLifchitz/project-M](https://github.com/HilaLifchitz/project-M).

```bash
conda env create -f environment.yml   # Python 3.11, numpy, scipy, matplotlib, seaborn, pandas, networkx
conda activate miso
```

The environment lives at `~/miniconda3/envs/miso` if it was created earlier.

---

## Map of the repo

| Folder | What it is |
|--------|------------|
| [`2loci/`](2loci/README.md) | **Main simulator.** Exact haploid WF chain on 4 genotypes; KL dips, $\eta$, coupled mutation, log $Z$ |
| [`equilibria_vs_dynamics/`](equilibria_vs_dynamics/README.md) | Does equilibrium score predict adaptation speed? Canonical landscapes A–G, SSWM node-hops |
| [`epistasis_equilibria/`](epistasis_equilibria/README.md) | Continuous haploid equilibria (allele-frequency cube and genotype simplex) |
| [`initials_thoughts&codes/`](initials_thoughts%26codes/README.md) | First-generation 1-locus / phenotype-$Z$ scripts (moved off the repo root) |
| [`ProjectGradientFLows/`](ProjectGradientFLows/README.md) | JKO / Wasserstein gradient flows (1D–2D heat; Shahshahani simplex) |
| [`types of mutations/`](types%20of%20mutations/README.md) | Small-world sparse epistasis on $\{0,1\}^n$ |
| [`regulatory networks/`](regulatory%20networks/README.md) | GP-map notes and figures (no code) |
| [`docs/`](docs/README.md) | Papers and slides |
| [`Polygenic Efficeincy/`](Polygenic%20Efficeincy/README.md) | Additive polygenic trait; maintenance efficiency $\eta_{\mathrm{maint}}(n)$ vs number of loci |
| [`web/`](web/README.md) | GitHub Pages app for the polygenic sketch (sliders + lattice; JSON from Python later) |
| `C.py` | Fitness-ratio diagnostic for a symmetric $\Sigma$ (any $n$) |

Each folder has its own README with scripts, run commands, and findings.

---

## Mental model (two-locus chain)

State = haplotype counts $(c_{00},c_{01},c_{10},c_{11})$ summing to $N$. One generation:

**selection → recombination ($r$) → mutation ($M$) → multinomial drift.**

Fitness $\omega(g)=\exp(g^\top\Sigma g)$ with $\Sigma_{ii}=s_i$, $\Sigma_{12}=\varepsilon_{12}/2$.

Start from the **neutral** stationary $\varphi$ ($\Sigma=0$). Track

$$
D(t)=\mathrm{KL}\bigl(\psi(t)\,\|\,\varphi\bigr)\quad\text{[bits]},
\qquad
\eta(t)=\frac{\Delta D}{N\,C(t)}.
$$

A recurring fact: $\mathrm{KL}(\psi^*\|\varphi)$ is **U-shaped** in $\varepsilon_{12}$. The dip sits near

$$
\varepsilon_{12}^*\approx -\frac{\mathrm{Cov}_\varphi(B,x_{11})}{\mathrm{Var}_\varphi(x_{11})}
\approx -\tfrac23(s_1+s_2)
$$

at weak selection and symmetric mutation. High equilibrium information is *often* fast adaptation, but not a one-to-one rank (see `equilibria_vs_dynamics`).

---

## Quick start (current main path)

```bash
cd 2loci
python build_neutral_cache.py --config neutral_cache_config.yaml   # once per (u,v,r)
python simplex.py --config simplex_config.yaml
python eps12_sweep.py
```

Then, depending on the question:

```bash
# KL dip location
python eps12_dip_cov_prediction.py

# cost / efficiency vs ε12
python eps12_delta_d_sweep.py --config eps12_delta_d_config.yaml

# does KL_eq predict speed?
cd ../equilibria_vs_dynamics
python eps12_predictiveness_grid.py
```

---

## What is *not* in git

`.gitignore` drops caches, plots, and logs (`*.npz`, `*.png`, `*.csv`, `*.log`, `2loci/cache/`, `*/results/`, …). Re-run scripts to regenerate figures. Session handoff markdown files **are** tracked.

---

## Layout vs older READMEs

Root-level files named `QE.py`, `Markov_1_locus.py`, `diffusion_vs_markov.py`, etc. now live in [`initials_thoughts&codes/`](initials_thoughts%26codes/README.md). The June 2026 `2loci/` chain is the production two-locus model.
