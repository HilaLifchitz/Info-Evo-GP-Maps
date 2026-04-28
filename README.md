# Miso: Population Genetics Simulation and Analysis

A comprehensive Python package for simulating and analyzing population genetics dynamics, with a focus on Wright-Fisher models, selection effects, and information-theoretic measures.

## Overview

This project implements various population genetics simulations and analysis tools, including:
- Wright-Fisher simulations with and without recombination
- Selection modeling (directional and stabilizing)
- Kullback-Leibler divergence analysis between distributions
- Convergence analysis and autocorrelation studies
- Phenotype distribution analysis

## Recent Updates (Apr 2026)

Work in `epistasis_equilibria/` added a focused allele-frequency equilibrium workflow:

- `equilibria_allele_freq.py`
  - Equilibrium on `[0,1]^n` for haploid/diploid settings with config-driven matrix `M` (`sigma` in YAML).
  - Added diagnostics for:
    - `KL(ρ_eq || ρ_neutral[M=0])`
    - Free-fitness terms at equilibrium
    - Weak-selection check: `N * max_g |g^T M g|`
    - ESS quality reporting (`ESS/support_points` with labels: bad/weak/okay/good/very good)
- `equilibria_allele_freq_viz_n2.py`
  - Dedicated `n=2`, haploid visualization helper with:
    - 3D surfaces for `ρ_eq`, `ρ_neutral`, and `W(x)`
    - Comparison panels (`log(ρ_eq/ρ_neutral)`, `Δℓ`, and `logW` correlation view)
    - Fast corner-symmetry debug output (corner values + spread metrics)
- `equilibria_config_allele_freq_example.yaml`
  - Example config for the new allele-frequency equilibrium tools.

## Main Components

### Core Simulation Engine

#### `QE.py`
The main simulation engine containing:
- **`WrightFisherSimulator`**: Basic Wright-Fisher simulation
- **`WrightFisherSimulator_FreeRecombination`**: Wright-Fisher with free recombination
- **`TraitEvolutionSimulator_directional`**: Trait evolution under directional selection
- **`TraitEvolutionSimulator_stabilizing`**: Trait evolution under stabilizing selection
- Analytical functions for neutral distribution `φ(t)` and selection distribution `ψ(t)`
- Variance computation and normal distribution PDFs

#### `Markov_1_locus.py`
Comprehensive Wright-Fisher Markov chain analysis for single-locus population genetics:

**Core Class: `WrightFisherMarkovChain`**
- Implements finite Wright-Fisher process with selection and mutation
- Haploid population of size N with two alleles ("0" wildtype, "1" beneficial)
- Selection coefficient s: fitness of "1" allele relative to "0" 
- Symmetric mutation rate μ: probability of allele flipping per generation
- State space: {0, 1, 2, ..., N} representing number of "1" alleles

**Key Methods:**
- `construct_transition_matrix()`: Builds (N+1)×(N+1) transition matrix P
- `solve_stationary_direct()`: Computes stationary distribution via eigenvalue decomposition
- `solve_stationary_iterative()`: Power method for stationary distribution
- `get_stationary_statistics()`: Mean, variance, mode, boundary probabilities
- `plot_stationary_distribution()`: Visualization of steady-state distribution

**Evolution Analysis: `evolve_from_neutral_to_selection()`**
Implements Iwasa's "free fitness" information-theoretic framework:
- **φ***: Neutral steady state (s=0)
- **ψ***: Selection steady state (s>0) 
- **ψ(t)**: Time-evolving distribution from φ* to ψ*
- **D(t)**: KL divergence D(t) = KL(ψ(t) || φ*) - total information
- **I(t)**: KL divergence I(t) = KL(ψ(t) || ψ*) - "free fitness" information
- **rest(t)**: Cross-term rest = ∫ₓ ψ(t)(x) log(ψ*(x)/φ*(x))
- **Decomposition**: D(t) = I(t) + rest(t) (information conservation)

This analysis tracks how information flows during evolution from neutral to selection equilibrium, providing insights into the information-theoretic cost of adaptation.

### Analysis Scripts

#### `3.6B_maybe.py`
Main analysis script for comparing phenotype distributions:
- **`kl_bits()`**: Numerically stable KL divergence calculation with symmetric smoothing
- **`kl_pheno_bits()`**: KL divergence between phenotype distributions with proper binning
- **`kl_psi_vs_binomial()`**: Compare empirical selection distribution to theoretical binomial
- **`sample_Z_hist()`**: Sample phenotype histograms from allele frequency distributions
- **`run()`**: Main execution function with parameterized simulations
- **`plot_pheno_distributions()`**: Visualization of phenotype distributions

#### `convergence_analysis_nick.py`
Advanced convergence analysis with multiple simulations:
- **`run_multiple_simulations()`**: Run multiple Wright-Fisher simulations with checkpointing
- **`compute_average_dynamics()`**: Average KL and phenotype dynamics across simulations
- **`compute_autocorrelation()`**: Calculate autocorrelation for time series
- **`analyze_autocorrelation_at_points()`**: Autocorrelation analysis at specific time points
- **`plot_average_dynamics()`**: Visualization of average dynamics with confidence bands
- **`plot_autocorrelation_analysis()`**: Autocorrelation decay analysis plots

#### `diffusion_vs_markov.py`
Compares discrete Wright-Fisher Markov chain with continuous diffusion approximation:

**Core Functionality:**
- **π_M(i)**: Stationary distribution from finite Markov chain (states i=0..N)
- **π_D(i)**: Diffusion stationary density discretized to N+1 bins
- **Diffusion Formula**: f(x) ∝ x^{2Nv-1}(1-x)^{2Nu-1}exp[2Nsx] (haploid scaling)
- **Discretization**: Centered integration over bins [(i-0.5)/N, (i+0.5)/N]

**Comparison Metrics:**
- **L1 Distance**: Total variation distance between distributions
- **KL(M||D)**: Kullback-Leibler divergence from Markov to diffusion
- **KL(D||M)**: Kullback-Leibler divergence from diffusion to Markov  
- **KS D**: Kolmogorov-Smirnov statistic (maximum CDF difference)

**Key Functions:**
- `analyze_distances_vs_N(s, mu)`: Batch analysis across N=[50,100,150,200,500,1000,1500,2000]
- `plot_distances_vs_N(df, s, mu)`: Visualization of distance metrics vs population size
- `run_comparison()`: Single comparison with optional joint Ne optimization
- `--optimize-ne`: Jointly searches single Ne used for both Markov and diffusion

**Usage:**
```bash
# Basic comparison
python diffusion_vs_markov.py --N 100 --s 0.005 --mu 0.0005

# Joint Ne optimization  
python diffusion_vs_markov.py --N 100 --s 0.005 --mu 0.0005 --optimize-ne

# Batch analysis (in Python)
from diffusion_vs_markov import analyze_distances_vs_N, plot_distances_vs_N
df = analyze_distances_vs_N(s=0.005, mu=0.0005)
plot_distances_vs_N(df, s=0.005, mu=0.0005)
```

This tool validates the diffusion approximation against exact Markov chain results and investigates how approximation quality depends on population size.

#### `Markov_2_loci.py`
Two-locus Wright-Fisher Markov chain utilities. The file contains:

1) Independent-loci wrapper (`MarkovTwoLoci`) for quick analyses
- Builds two single-locus chains (`WrightFisherMarkovChain`) with (N, μ, s1) and (N, μ, s2)
- Computes per-locus stationary distributions and constructs joint via outer product (independence)
- Convenience plotting (`plot_joint_heatmap`, `plot_joint_3d`) and helpers
- Information measures over time on allele (P), simplex (X), and genotype (G) spaces

2) LD model with recombination (`TwoLocusLD`) and spectral analysis
- Full two-locus Wright–Fisher with the pipeline selection → recombination → mutation → sampling
- Pushforwards from simplex (X) to:
  - allele grid (P): `pushforward_X_to_P(w)` → (N+1)×(N+1)
  - genotype distribution (G): `pushforward_X_to_G(w)` → 4-vector [00,01,10,11]
- KL helpers from X: `kl_P_from_X(w, baseline_P)`, `kl_G_from_X(w, baseline_G)`
- G sanity check: `debug_check_G_pushforward(w)` verifies G via fractions vs counts
- Spectral modes for many timepoints:
  - `leading_modes(k, ...)` and streaming evaluator (no need to store all X(t))
  - `compute_D_I_series_LD(...)` returns tidy DataFrame with columns:
    - D-series (vs φ*): `D_X, D_P, D_G`
    - I-series (vs ψ*): `I_X, I_P, I_G`
- Plotting:
  - `plot_D_I_series_from_df(df, title=None, r=None)` — two panels (D and I)
  - `plot_D_I_series_dual_axis_from_df(df, title=None, r=None)` — same but with a separate y-axis for G
  - `plot_joint_P_3d(P, ...)` — 3D bar plot of joint P measure

**LD usage example (compute and plot D/I series):**
```python
from Markov_2_loci import compute_D_I_series_LD, plot_D_I_series_dual_axis_from_df
import matplotlib.pyplot as plt

df = compute_D_I_series_LD(
    N=40, s1=0.005, s2=0.01, r=0.001, mu=5e-4,
    Ts=[0, 10, 20, 50, 100, 200, 500],
    modes_k=30
)
fig, axes = plot_D_I_series_dual_axis_from_df(
    df, title='N=40, μ=5e-4, s1=0.005, s2=0.01', r=0.001
)
plt.show()
```

**Validate G pushforward for an X-distribution `w`:**
```python
from Markov_2_loci import TwoLocusLD, count_simplex_points
import numpy as np

ld = TwoLocusLD(N=30, s1=0.005, s2=0.10, r=0.01, mu=0.001)
S = count_simplex_points(30, 3)
phi_star = np.ones(S) / S
res = ld.debug_check_G_pushforward(phi_star, verbose=True)
```

This module provides both the independent-loci analysis path and a full LD pipeline with efficient spectral evaluation for multi-timepoint information-theoretic tracking across X, P, and G.

#### `kl_comparison_test.py`
Systematic comparison of different KL divergence calculation methods:
- Compares old vs. new KL calculation methods
- Demonstrates effects of smoothing and reference distributions
- Provides detailed explanations of numerical differences

### Testing and Validation

#### `test_simple_neutral_approach.py`
Previous attempt at equilibrium detection (reference for KL divergence values).

#### `quick_80bit_test.py`
Quick validation tests for KL divergence calculations.

## Key Features

### Numerical Stability
- Robust KL divergence calculation with symmetric additive smoothing
- Proper handling of zero probabilities and sparse distributions
- Renormalization to ensure valid probability distributions

### Simulation Capabilities
- **Population size**: Configurable population sizes (N)
- **Loci**: Variable number of loci (l)
- **Selection**: Directional and stabilizing selection parameters
- **Mutation**: Configurable mutation rates
- **Recombination**: Free recombination model
- **Generations**: Long-term evolution tracking

### Analysis Tools
- **KL Divergence**: Multiple calculation methods for different use cases
- **Phenotype Analysis**: Mean, variance, and distribution tracking
- **Autocorrelation**: Memory effects and decay rate analysis
- **Convergence**: Multi-simulation averaging and statistical analysis

### Data Management
- **Checkpointing**: Automatic saving of simulation progress
- **Data Persistence**: NPZ format for efficient storage
- **Reproducibility**: Seed-based random number generation

## Usage Examples

### Basic Simulation
```python
from QE import WrightFisherSimulator_FreeRecombination

# Run a single simulation
sim = WrightFisherSimulator_FreeRecombination(
    N=40, l=1000, s=0.01, mu=0.0005, T=10000, seed=42
)
results = sim.run()
```

### KL Divergence Analysis
```python
from 3.6B_maybe import run

# Run phenotype distribution comparison
run(N=40, l=1000, N_pop=2000, seed=42)
```

### Convergence Analysis
```python
from convergence_analysis_nick import main

# Run comprehensive convergence analysis
main()  # Uses default parameters
```

## Installation

### Environment Setup

This project includes an `environment.yml` file for easy environment setup with conda, mamba, or micromamba.

**To create a new environment from the file:**
```bash
# Using conda
conda env create -f environment.yml

# Using mamba (faster)
mamba env create -f environment.yml

# Using micromamba
micromamba env create -f environment.yml
```

**To activate the environment:**
```bash
conda activate miso
# or
mamba activate miso
# or
micromamba activate miso
```

**To update an existing environment:**
```bash
conda env update -n miso -f environment.yml --prune
```

The `--prune` flag removes packages not listed in the file, keeping your environment clean and consistent.

## Dependencies

- **NumPy**: Numerical computing
- **SciPy**: Statistical functions and optimization
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Enhanced statistical graphics
- **Pandas**: Data manipulation and analysis

## File Structure

```
Miso/
├── QE.py                          # Core simulation engine
├── Markov_1_locus.py              # Wright-Fisher Markov chain analysis
├── diffusion_vs_markov.py         # Markov vs diffusion comparison
├── 3.6B_maybe.py                  # Main analysis script
├── convergence_analysis_nick.py   # Convergence analysis
├── kl_comparison_test.py          # KL divergence comparison
├── test_simple_neutral_approach.py # Equilibrium detection
├── quick_80bit_test.py            # Quick validation tests
├── images/                        # Generated plots and figures
└── convergence_results/           # Simulation results and checkpoints
```

## Research Context

This project is designed for population genetics research, particularly focusing on:
- Information-theoretic measures in evolutionary dynamics
- Selection effects on phenotype distributions
- Convergence properties of Wright-Fisher models
- Numerical methods for population genetics simulations

## Notes

- All KL divergence calculations are in bits (log base 2)
- Simulations use checkpointing to prevent data loss
- Results are automatically saved in NPZ format for later analysis
- The project includes comprehensive error handling and progress tracking
