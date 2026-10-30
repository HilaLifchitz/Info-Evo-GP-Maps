# Session handoff — allele-frequency sweeps & epistasis

## Where things live

- **`haploid_allele_grid_metrics.py`** — Shared haploid equilibrium on a uniform tensor grid in $(x_\epsilon,1-x_\epsilon)^n$: builds `X`, cell volume $\Delta$, `ell_batch` weights, $\rho \propto e^\ell$, **$F_N = \frac{1}{N}\log Z$**, **KL$(\rho_{\mathrm{eq}}\|\rho_{\mathrm{neutral}})$** in bits, **`mean_W_eq` = $\mathbb{E}_{p_{\mathrm{eq}}}[W]$** on the same discrete $p_k$. Works for **any $n$**; cost $\sim$ `resolution^n` per $M$.
- **`allele_freq_eps12_sweep.py`** — $n{=}2$, sweeps $\varepsilon_{12}$ with fixed $s_1,s_2$, **$M_{12}=\varepsilon_{12}/2$**. Defaults in **`SWEEP_DEFAULTS`**. Auto output: **`results/eps12_sweep_<timestamp>_s1_…_s2_…/`** with **`sweep.csv`**, **`run.log`** (full ASCII table + summary for **$\varepsilon_{12}<0$** free fitness vs at 0), **`sweep.png`**.
- **`allele_freq_eps3_sweep.py`** — $n{=}3$, Cartesian grid over $(\varepsilon_{12},\varepsilon_{13},\varepsilon_{23})$; diagonals **`--s1 --s2 --s3`**. Defaults **`SWEEP_DEFAULTS_N3`** (includes lower `resolution` default). Output pattern **`results/eps3_sweep_…/`**, **`sweep.csv`**, **`run.log`** (stats + “all three $\varepsilon$ negative” subset).
- **`equilibria_allele_freq.py`** — Source of **`ell_batch`**, **`log_mean_fitness_batch`**, genotypes, etc.

## Conventions already agreed

- $\ell(x) = N\log W(x) +$ mutation log terms; neutral = $M=0$.
- $Z = \sum_k e^{\ell_k}\Delta$ on the grid; **free fitness** computed as **$\frac{1}{N}\log Z$** (with internal consistency check vs entropy decomposition).

## TODO (next session)

1. **Mean fitness under equilibrium** — Implemented: **`mean_W_eq`** in `sweep.csv` / logs = **$\mathbb{E}_{p_{\mathrm{eq}}}[W(x)]$** (same $p_k$ as KL / $F_N$); see **`haploid_allele_grid_metrics.free_fitness_and_kl_bits_haploid`**.

2. **Expected allele frequencies** — **$\mathbb{E}_{\rho_{\mathrm{eq}}}[x_i]$** for each locus $i$ (vector of length $n$), same $p_k$; add to CSV or a sidecar; useful for linking to “state of the population” under the Gibbs approximation.

3. **Epistatic information** — You’ll connect the above observables to your notion of epistatic information; no code in repo for that definition yet—placeholder for the next design pass.

## Minor notes

- **`--out-prefix`** still overrides auto-named run directories.
- Very coarse **`resolution`** biases integrated quantities; KL and $Z$ both feel the grid.
