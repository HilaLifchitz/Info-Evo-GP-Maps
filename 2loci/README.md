# Two-locus Wright–Fisher on the genotype simplex

Exact finite-population Markov chain for two linked loci, discretized on the 4-genotype simplex (00, 01, 10, 11). Tracks information-theoretic dynamics from a **neutral start** \(\psi(0)=\varphi\), where \(\varphi\) is the neutral (\(\Sigma=0\)) stationary distribution.

## Model

**State space.** For population size \(N\), haplotype counts \((c_{00}, c_{01}, c_{10}, c_{11})\) are non-negative integers summing to \(N\). There are \(S = \binom{N+3}{3}\) lattice states.

**One generation** at genotype frequency vector \(x\):

1. **Selection:** \(p_{\mathrm{sel}}(g) \propto x_g\,\omega(g)\)
2. **Recombination:** LD shift with rate \(r\) (same convention as `Markov_2_loci.py`)
3. **Mutation:** \(p_{\mathrm{mut}} = p_{\mathrm{rec}} M\) (per-locus asymmetric rates \(u_i\), \(v_i\))
4. **Drift:** offspring counts \(\sim \mathrm{Multinomial}(N, p_{\mathrm{mut}})\)

**Fitness:**

\[
\omega(g) = \exp(g^\top \Sigma g)
\]

with symmetric \(2\times2\) matrix \(\Sigma\) (\(\Sigma_{ii}=s_i\), \(\Sigma_{ij}=\varepsilon_{12}/2\) for \(i\neq j\)).

## Files

| File | Purpose |
|------|---------|
| `simplex.py` | Main driver: build transition matrix, evolve \(\psi(t)\), diagnostics, plots |
| `simplex_config.yaml` | Runtime parameters for single-run `simplex.py` |
| `eps12_sweep.py` | Sweep \(\varepsilon_{12}\); all settings in the script (no YAML) |
| `build_neutral_cache.py` | Precompute neutral equilibria \(\varphi\) for several \(N\) |
| `neutral_cache.py` | Cache I/O (`.npz` keyed by \(u\), \(v\), \(r\)) |
| `neutral_cache_config.yaml` | Which \(N\) values to cache |
| `explore_neutral_phi.py` | Characterize the neutral \(\varphi\) across \(N\), \(u/v\), \(r\); marginal heatmaps + LD + diffusion-Beta comparison |
| `selection_diffusion_vs_chain.py` | Discrete selection equilibrium \(\psi^*\) vs **genotype-simplex** diffusion \(p\propto W^N\prod_i x_i^{\alpha_i-1}\) |
| `selection_diffusion_r_N_sweep.py` | Sweep \(r\) and \(N\) for the genotype-simplex fit |
| `allele_diffusion_vs_chain.py` | Chain allele-frequency marginals vs **allele/LE** diffusion \(\rho\propto W^N\prod_i x_i^{2Nu_i-1}(1-x_i)^{2Nv_i-1}\) |
| `selection_magnitude_sweep.py` | Chain-vs-diffusion fit as a function of scaled selection \(N s_{\max}\) |

Generated at runtime (not committed):

- `cache/neutral_u0_..._r....npz` — neutral stationary distributions
- `plots/information_and_f1_vs_t_*.png` — single-run dynamics figures
- `plots/eps12_sweep_*_vs_t_*.png` — epistatic sweep overlay figures

## Quick start

From the repo root (after `conda activate miso`):

```bash
# 1. Build neutral cache (once per u, v, r combination; slow for large N)
cd 2loci
python build_neutral_cache.py --config neutral_cache_config.yaml

# 2. Run with selection (loads phi from cache when available)
python simplex.py --config simplex_config.yaml

# 3. Epistatic sweep (many ε₁₂ curves on one figure)
python eps12_sweep.py
# All sweep parameters live in SWEEP SETTINGS at top of eps12_sweep.py
```

Edit `simplex_config.yaml` for single runs with `simplex.py`. For sweeps, edit **`eps12_sweep.py`** only (`N`, `r`, `u`, `v`, `s₁`, `s₂`, `TIMES`, ε₁₂ range).

## Epistatic sweep (`eps12_sweep.py`)

Self-contained sweep script: fixes \(N\), \(r\), \(u\), \(v\), \(s_1\), \(s_2\) and varies \(\varepsilon_{12}\) from \(-1\) to \(+1\) (default step 0.1). Produces four overlay plots (KL and \(\mathcal{F}_N\), values and rates), one colored curve per \(\varepsilon_{12}\) with a coolwarm colorbar.

**All parameters are in the `SWEEP SETTINGS` block** at the top of `eps12_sweep.py` — no YAML. The only external input is the neutral cache file matching your \((u, v, r)\).

```bash
python eps12_sweep.py
python eps12_sweep.py --output-dir plots/my_sweep
```

Output files: `plots/eps12_sweep_kl_vs_t_r_...png`, `..._kl_rates_...`, `..._f1_vs_t_...`, `..._f1_rates_...`.

## Information dynamics

When `times` is set, the script always computes:

| Quantity | Definition |
|----------|------------|
| **\(D_{\mathrm{KL}}(\psi\|\varphi)\)** | KL divergence in **bits** vs neutral equilibrium \(\varphi\) |
| **\(\mathcal{F}_N\)** | Free fitness: \(\langle W\rangle_\rho + H(\rho)/(2N)\) with \(W(x)=\log\sum_g x_g\omega(g) - \frac{1}{2N}\sum_i (xM)_i\log x_i\) |
| **mean LD** | \(\mathbb{E}_\rho[x_{00}x_{11}-x_{01}x_{10}]\) |

Optional (off by default):

| Quantity | Definition |
|----------|------------|
| **\(F_2\)** | \(D_{\mathrm{KL}}(\psi(t)\|\psi^*)\) where \(\psi^*\) is the selection stationary distribution |

### CLI flags

```bash
python simplex.py --config simplex_config.yaml          # default
python simplex.py --f2                                  # enable F_2
python simplex.py --no-f2                               # disable F_2 from config
python simplex.py --debug-kl                            # sanity checks (omega, P vs neutral, KL after 1 gen)
```

### Plots (`plot_dynamics: true`)

Three PNG files in `plots/`, with filenames encoding \(r\), \(s_1\), \(s_2\), \(\varepsilon_{12}=2\Sigma_{12}\):

1. `information_and_f1_vs_t_r_...png` — KL [bits, left] and \(\mathcal{F}_N\) [right] vs \(t\)
2. `information_and_f1_rates_vs_t_r_...png` — discrete rates \(\Delta/\Delta t\)
3. `mean_ld_vs_t_r_...png` — mean linkage disequilibrium vs \(t\)

### Config reference (`simplex_config.yaml`)

```yaml
N: 30
r: 0.005
u: [1.0e-4, 1.0e-4]    # per-locus 0→1
v: [1.0e-4, 1.0e-4]    # per-locus 1→0
sigma:
  - [-0.1, 0.0]         # s1, eps12/2
  - [0.0, -0.005]       # eps12/2, s2
times: [50, 100, ..., 2000]
neutral_cache_dir: cache
plot_dynamics: true
compute_f2: false
ld_top_k_times: first_last   # top-k LD tables at t=0 and final t
debug_kl: false
```

Alternative: set `s: [s1, s2]` and `eps12: ...` instead of `sigma` (ignored when `sigma` is present).

Set `neutral: true` to run dynamics with \(\Sigma=0\) (selection off).

## Neutral cache

`simplex.py` needs the neutral equilibrium \(\varphi\) as the initial condition \(\psi(0)=\varphi\) and as the KL reference. For \(N\ge 20\), building \(\varphi\) on the fly is slow; use the cache:

```bash
python build_neutral_cache.py --config neutral_cache_config.yaml
```

The cache filename is derived from \((u, v, r)\). **`simplex_config.yaml` must use the same \(u\), \(v\), and \(r\)** as the cache config, or a matching cache file for that triple.

Example output: `cache/neutral_u0_1em4_u1_1em4_v0_1em4_v1_1em4_r0.npz`

## Performance

- Transition matrix size: \(S\times S\) with \(S=\binom{N+3}{3}\) (e.g. \(N=30 \Rightarrow S=5456\))
- Row construction is parallelized (`n_jobs: -1` uses all CPU cores)
- First run for a new \((N, u, v, r, \Sigma)\) builds the selection matrix; subsequent runs with the same parameters reuse nothing yet (matrix is rebuilt each invocation)

## Equilibrium structure & diffusion-approximation validation

A set of analysis scripts (added 2026-06-02) characterize the chain's equilibria and test how well **diffusion approximations** reproduce them. All write to `results/<timestamp>_*/` (PNG + CSV/log).

### A. Neutral equilibrium \(\varphi\) — `explore_neutral_phi.py`

Sweeps scenarios \((N, u, v, r)\); for each, solves the \(\Sigma=0\) stationary \(\varphi\) and reports corner/edge/interior mass, per-locus marginals, **average LD**, and a comparison of each locus marginal to the diffusion prediction \(\mathrm{Beta}(2Nu_i, 2Nv_i)\).

**Conclusions.**

- \(\varphi\) is **not uniform** over the simplex. With weak mutation (\(2Nu\ll1\)) ~97% of the mass sits on the 4 monomorphic vertices (U-shape). It is only U-shaped in that limit:
  - \(2Nu\gg1\) (strong mutation) → unimodal **central hump** (interior mass → 1).
  - asymmetric \(u\neq v\) → mass shifts toward the high-frequency corner; per-locus marginals diverge.
- The **per-locus diffusion Beta is an excellent approximation** for the marginals across all regimes (TV \(\sim10^{-4}\)–\(10^{-2}\)), even far from the U-shape.
- The diffusion product-of-Betas predicts **\(D=0\)** (independent loci). The chain's signed \(\mathbb{E}[D]\approx0\) by symmetry, but \(\mathbb{E}|D|\) is substantial at \(r=0\) (\(\approx0.04\)) and **recombination roughly halves it** (\(r=0.5\Rightarrow\approx0.018\)) without changing the marginals — i.e. recombination only reshapes the *joint*, pushing the chain toward the diffusion's independence assumption.

### B. Selection equilibrium vs diffusion — `selection_diffusion_vs_chain.py`, `*_r_N_sweep.py`, `allele_diffusion_vs_chain.py`, `selection_magnitude_sweep.py`

Two diffusion equilibria are compared to the discrete selection equilibrium \(\psi^*\):

- **genotype-simplex** (treats the 4 genotypes as 4 alleles, no recombination): \(p(x)\propto W(x)^N\prod_i x_i^{\alpha_i-1}\), \(W(x)=\sum_i x_i\omega(g_i)\). The mutation exponent \(\alpha_i=2N\cdot(\text{rate into }i)\) is ambiguous for our non-reversible \(M\); we use the per-capita conditional influx \(\alpha_i(x)=2N\sum_{j\neq i}x_jM_{ji}/\sum_{j\neq i}x_j\) (reduces exactly to the 2-allele Wright exponent).
- **allele-frequency / LE** (per-locus, assumes linkage equilibrium + HWE): \(\rho(x_1,x_2)\propto W(x_1,x_2)^N\prod_i x_i^{2Nu_i-1}(1-x_i)^{2Nv_i-1}\), \(W\) computed from product genotype frequencies. Uses the clean Wright exponents (no \(\alpha\)).

Both use the **\(W^N\)** (haploid) selection scaling. Comparisons are made on the interior support; \(\psi^*\) boundary mass is reported separately.

**Conclusions.**

1. **Selection strength \(N s_{\max}\) governs the fit** (`selection_magnitude_sweep.py`). In the diffusion regime \(N s_{\max}\lesssim1\) (small per-generation \(s\)) the match is **near-exact**: KL \(< 0.05\) bits, TV \(< 0.09\). The error grows **monotonically** with \(N s_{\max}\) (e.g. KL \(\approx0.2\) bits at \(Ns\approx5\), \(\gtrsim0.45\) bits at \(Ns\gtrsim18\)). This is a genuine **strong-selection correction** (the diffusion drops \(O(s^2)\) terms), not a finite-\(N\) artifact.
2. **This is why "bigger \(N\) doesn't help" at fixed \(s\)**: raising \(N\) raises \(N s\), moving you *off* the weak-selection regime. The diffusion limit requires \(s\to0\) with \(Ns\) fixed. Confirmed by `*_r_N_sweep.py`: at fixed \(s\), \(N=40 < 45 < 50\) in fit quality (slightly worse with \(N\)).
3. **The residual is a selection-scaling bias**, not the \(\alpha\) surrogate: with \(W^{2N}\) the diffusion *over*-pushes the favored genotype; with \(W^{N}\) it *under*-pushes — so the chain's effective exponent lies **between \(W^N\) and \(W^{2N}\)**. Shrinking mutation to very small values does **not** close the gap (`allele_diffusion_vs_chain.py`), ruling out \(\alpha\) as the dominant error.
4. **Recombination picks the relevant diffusion.** The genotype-simplex form is the \(r=0\) limit (best at \(r=0\), degrades as \(r\uparrow\)); the allele/LE form assumes LE (improves as \(r\uparrow\)). They **cross over** around \(Ns\approx2\)–3 / intermediate \(r\): the genotype-simplex fit is best at low \(r\) (optimum around \(r\approx0.25\)), the allele/LE fit best at high \(r\).

**One-line takeaway:** the diffusion ↔ exact-chain agreement is excellent whenever selection is weak (\(N s_{\max}=O(1)\), \(s\ll1\)); apparent discrepancies are strong-selection corrections, and each diffusion form matches its own recombination regime.

```bash
python explore_neutral_phi.py
python selection_diffusion_vs_chain.py
python selection_diffusion_r_N_sweep.py
python allele_diffusion_vs_chain.py
python selection_magnitude_sweep.py
```

## Relation to other code in this repo

- **`Markov_2_loci.py`** — earlier two-locus tools (LD model, spectral modes, D/I series on X/P/G spaces). `simplex.py` is a self-contained genotype-simplex implementation with the same recombination convention and a focus on \(\psi(t)\) vs neutral \(\varphi\).
- **`epistasis_equilibria/equilibria_allele_freq.py`** — continuous allele-frequency equilibrium on \([0,1]^n\); complementary diffusion-limit view vs this exact Markov chain.

## Notes

- KL is reported in **bits** (\(\log_2\)).
- Varying \(\varepsilon_{12}\) with fixed diagonal \(s_1,s_2\) only changes \(\omega(11)\); the other three genotypes share the same fitness. Compare curves on a **common y-axis** or normalize by \(D_{\mathrm{KL}}(t_{\mathrm{final}})\) to see shape differences across \(\Sigma\).
- `--debug-kl` prints whether selection enters the dynamics and whether KL responds to \(\Sigma\) (not a substitute for comparing absolute KL values across runs).
