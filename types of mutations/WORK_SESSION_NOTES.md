# Small-World Epistasis Work Notes

Last updated: 2026-05-05

## Current Goal

We are investigating whether mutations in a small-world epistatic fitness landscape behave in a House-of-Cards-like way.

Alleles are binary:

- `0` = wild type
- `1` = mutation

For a genotype `g in {0,1}^n`:

```text
log_fitness(g) = g.T @ M @ g
fitness(g) = exp(g.T @ M @ g)
```

The small-world graph determines which off-diagonal entries of `M` are nonzero.

For each graph edge `(i, j)`, the code samples one epistatic effect `epsilon_ij` and stores:

```text
M[i, j] = M[j, i] = epsilon_ij / 2
```

The diagonal stores single-locus selection coefficients:

```text
M[i, i] = s_i
```

## Main Script

The active script is:

```text
small_world_epistasis.py
```

It currently does the following:

1. Creates a connected Watts-Strogatz small-world graph with `networkx`.
2. Samples edge effects `epsilon_ij` from either a uniform or normal distribution.
3. Samples diagonal selection coefficients `s_i` from constant, uniform, or normal distributions.
4. Builds the symmetric epistasis matrix `M`.
5. Enumerates all binary genotypes when `2^n <= --max-genotypes`.
6. Computes genotype log fitness and fitness.
7. Computes per-locus mutation effects for every `0 -> 1` mutation over all genetic backgrounds.
8. Computes House-of-Cards-style summary statistics.
9. Computes information epistasis `D(g_k | g_-k) - D(g_k)`.
10. Writes CSV/NPY/text report outputs.
11. Generates plots.

## Example Command

Run from the `types of mutations` folder:

```bash
python3 small_world_epistasis.py \
  --n 12 \
  --k 4 \
  --rewire-prob 0.05 \
  --edge-distribution normal \
  --edge-mean 0.0 \
  --edge-std 0.1 \
  --selection-distribution normal \
  --selection-mean 0.0 \
  --selection-std 0.05 \
  --allele-frequency 0.5 \
  --max-genotypes 65536 \
  --comparison-graphs 20 \
  --seed 42 \
  --results-dir results \
  --out-prefix sw12
```

Each run creates a timestamped folder:

```text
results/YYYY-MM-DD_HH-MM-SS/
```

All files from that run go inside this folder.

## Important Outputs

Inside each timestamped output directory:

```text
<prefix>_M.csv
<prefix>_M.npy
<prefix>_edges.csv
<prefix>_selection_coefficients.csv
<prefix>_genotype_fitness.csv
<prefix>_mutation_surprise.csv
<prefix>_mutation_surprise_by_locus.csv
<prefix>_information_epistasis_by_locus.csv
<prefix>_metrics.json
<prefix>_run_report.txt
<prefix>_equilibria_config.yaml
plots/
```

Main plots:

```text
plots/<prefix>_epistatic_network.png
plots/<prefix>_all_genotypes_log_fitness.png
plots/<prefix>_locus_000_effects.png
plots/<prefix>_locus_001_effects.png
...
plots/<prefix>_summary_by_locus.png
plots/<prefix>_house_of_cards_diagnostic.png
plots/<prefix>_information_epistasis_by_locus.png
```

## Mutation-Effect Statistics

For locus `i`, we examine every background where `g_i = 0`.

For each background:

```text
effect_i(g) = log_fitness(g with g_i = 1) - log_fitness(g)
```

Since `fitness = exp(log_fitness)`, this is also:

```text
effect_i(g) = log(fitness_mutant / fitness_background)
```

The per-locus summary includes:

- `average_log_fitness_effect`: average mutation effect across backgrounds.
- `effect_sd`: standard deviation of mutation effects.
- `effect_variance`: variance of mutation effects; equals `effect_sd ** 2`.
- `mean_absolute_surprise`: mean absolute deviation from the average effect.
- `rms_surprise`: root mean square deviation from the average effect.
- `parent_mutant_log_fitness_correlation`: Pearson correlation between background log fitness and mutant log fitness.
- `background_log_fitness_effect_correlation`: Pearson correlation between background log fitness and mutation effect.

## House-of-Cards Diagnostic

The plot:

```text
plots/<prefix>_house_of_cards_diagnostic.png
```

has one point per locus.

```text
x-axis = effect_sd
y-axis = parent_mutant_log_fitness_correlation
```

Interpretation:

- high `effect_sd` = mutation effect varies strongly across backgrounds.
- low parent-mutant correlation = parent fitness does not predict mutant fitness well.
- high `effect_sd` + low correlation is the most House-of-Cards-like region.

The plot now colors points by locus and annotates each point with the locus number.

## Per-Locus Effect Plots

Each locus plot has four panels:

1. Distribution of log-fitness effect.
2. Distribution of surprise:

```text
surprise = effect - average_effect_for_this_locus
```

3. Background log fitness vs mutation effect.
4. Parent vs mutant log fitness density heatmap.

Why some histograms have equal-height bars:

The mutation effect at locus `i` is:

```text
s_i + sum_j epsilon_ij g_j
```

Only loci connected to `i` matter. If `i` has few neighbors, many full backgrounds collapse to the same effect value.

## Information Epistasis

The code computes:

```text
information_epistasis(k) = D(g_k | g_-k) - D(g_k)
```

Baseline distribution:

```text
phi(g) = product_i x^g_i (1-x)^(1-g_i)
```

where `x = --allele-frequency`, default `0.5`.

Fitness-weighted distribution:

```text
psi(g) = phi(g) * exp(g.T @ M @ g) / Z
```

where:

```text
Z = sum_g phi(g) * exp(g.T @ M @ g)
```

Then:

```text
D(g_k) = KL(psi(g_k) || phi(g_k))
```

and:

```text
D(g_k | g_-k)
  = sum_background psi(g_-k) * KL(psi(g_k | g_-k) || phi(g_k))
```

The difference measures how much more information about locus `k` is gained when the genetic background is known.

## Epistatic Network Plot

The plot:

```text
plots/<prefix>_epistatic_network.png
```

shows:

- nodes = loci
- edge between `i` and `j` = nonzero off-diagonal `M[i,j]`
- edge label = `epsilon_ij`
- node label/color = diagonal `s_i = M[i,i]`

Literal self-loops are not drawn; the diagonal/self term is represented on the node.

## Environment Notes

The runtime environment needs:

```text
numpy
networkx
matplotlib
```

The Cursor editor/linter currently reports unresolved import warnings for these packages in its own Python environment, but `python3 -m py_compile small_world_epistasis.py` has been passing.

## Likely Next Things To Do

Possible next steps:

1. Improve plot aesthetics if labels overlap for larger `n`.
2. Add a CLI option to choose whether to draw edge labels on the network plot.
3. Add sampling instead of full genotype enumeration for larger `n`.
4. Compare different `k`, `rewire-prob`, edge distributions, and selection distributions across many runs.
5. Add a batch runner that sweeps parameters and aggregates House-of-Cards diagnostics.
6. Decide whether the information-epistasis measure should use uniform baseline `x=0.5`, user-specified `x`, or equilibrium allele frequencies from the other epistasis code.

