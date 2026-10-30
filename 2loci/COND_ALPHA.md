# Conditional alpha ("cond") — summary for the 2-locus genotype chain

## The problem

In the **detailed-balance / diffusion** picture, one writes a stationary density on the
genotype simplex (4 coordinates \(x_{00}, x_{01}, x_{10}, x_{11}\)) as

\[
p(x) \;\propto\; W(x)^{N}\,\prod_{i=0}^{3} x_i^{\alpha_i - 1},
\]

where \(W(x)=\sum_g x_g\,\omega(g)\) is mean fitness and \(\alpha_i\) is related to
mutation **into** genotype \(i\).

That formula assumes **parent-independent mutation** at the genotype level: each
"allele copy" mutates independently with a fixed rate. Our discrete model does **not**
have that. Mutation is **per locus** (asymmetric \(u_i, v_i\)), then combined into a
4×4 matrix \(M\):

```
M[g_from, g_to] = product of one-locus transition factors
```

So there is **no single constant** \(\alpha_i = 2N u\) on the 4-simplex that makes
\(M\) reversible in the Wright–Dirichlet sense. The joint neutral equilibrium \(\varphi^*\)
is computed **numerically** (`P\varphi=\varphi`, \(\Sigma=0\)), not from
\(\prod_i x_i^{\alpha_i-1}\) with constant exponents.

---

## What *does* work: per-locus marginals

For **one locus**, neutral diffusion gives

\[
\rho(x) \propto x^{2Nu-1}(1-x)^{2Nv-1} = \mathrm{Beta}(2Nu,\,2Nv).
\]

The chain's \(\varphi^*\), projected to a single locus marginal, matches this Beta
very well (`explore_neutral_phi.py`). So \(2Nu\) is the right exponent for **allele
frequency**, not for a naive product on the 4-genotype joint.

---

## Conditional alpha: a local substitute on the 4-simplex

To use the Boltzmann/Dirichlet **shape** on the full genotype simplex anyway, we
replace constant \(\alpha_i\) by a **state-dependent** rate derived from \(M\).

Notation (genotypes \(i,j \in \{0,1,2,3\}\), order 00, 01, 10, 11):

- \(x\) = row vector of genotype frequencies on the lattice point
- \((xM)_i = \sum_j x_j M_{ji}\) = post-mutation mass at genotype \(i\)
- **Influx into \(i\)** (excluding self-loop mass on diagonal):

  \[
  \text{influx}_i(x) = (xM)_i - x_i M_{ii} = \sum_{j\neq i} x_j M_{ji}.
  \]

### Mode `"cond"` (recommended)

Per-capita mutation rate **onto** genotype \(i\) from all other types:

\[
\boxed{
\alpha_i^{\mathrm{cond}}(x)
= 2N \cdot \frac{\text{influx}_i(x)}{\sum_{j\neq i} x_j}
= 2N \cdot \frac{\text{influx}_i(x)}{1 - x_i}
}
\]

**Why divide by \(1-x_i\)?** The denominator is the total mass **not** already at
genotype \(i\); dividing gives a rate per "mutatable" copy, analogous to per-allele
mutation in the 1-locus Wright model.

**1-locus check:** if only two haplotypes at one locus, this reduces to
\(\alpha = 2Nu\) (or \(2Nv\) on the other allele) — the standard Wright exponent.

### Mode `"influx"` (alternative)

\[
\alpha_i^{\mathrm{influx}}(x) = 2N \cdot \text{influx}_i(x)
\]

Raw inflow, not per-capita. Usually a worse fit on the interior (`selection_diffusion_vs_chain.py`).

---

## Diffusion density on the lattice (with selection)

Log unnormalized density at lattice point \(x\):

\[
\log p(x) = N\log W(x) + \sum_i \bigl(\alpha_i(x) - 1\bigr)\log x_i,
\]

with \(\alpha_i(x)\) from `"cond"` or `"influx"` above, \(W(x)=x\cdot\omega\).

**Important:** because \(\alpha_i\) depends on \(x\), this is **not** a single
Dirichlet distribution with fixed parameters. It is a **local** diffusion-inspired
surrogate. Compare to discrete \(\psi^*\) only on the **interior** (all four
\(x_i>0\)), where the log is finite.

Implementation: `selection_diffusion_vs_chain.py` → `diffusion_logdensity()`.

---

## How this relates to `simplex.py` mutation term

In `simplex.py`, the potential uses a **different** mutation object:

\[
\sum_i (xM)_i \log x_i
\]

(index \(i\) runs over the four genotypes), and enters \(W\) as

\[
W(x) = \log(x\!\cdot\!\omega) - \frac{1}{N}\sum_i (xM)_i \log x_i
\]

(haploid scaling \(1/N\), not \(1/2N\), after the 2026 fix).

This is **not** the same as cond-\(\alpha\) exponents. It comes from the
replicator–mutator / gradient-flow construction used for \(F_1\). Whether one
*should* replace \((xM)_i\log x_i\) by a cond-\(\alpha\) form is an open modeling
choice; cond-\(\alpha\) is mainly used for **equilibrium shape** comparisons, not
inside the current \(F_1\) dynamics.

---

## Numerical equilibrium vs cond-\(\alpha\) diffusion

| Object | How we get it |
|--------|----------------|
| \(\varphi^*\) (neutral) | Stationary of Markov chain, \(\Sigma=0\), `compute_phi()` |
| \(\psi^*\) (selection) | Stationary of chain with \(\Sigma\neq 0\), `try_selection_equilibrium()` |
| Diffusion surrogate | Normalize \(\exp(\log p(x))\) on the lattice with cond-\(\alpha\) |

Run the comparison script:

```bash
cd 2loci
python cond_alpha_vs_chain.py --config compare_sigma_config.yaml
```

Outputs: `plots/cond_alpha_check/` — TV/KL, marginals vs Beta\((2Nu,2Nv)\), and
\(x\!\cdot\!\omega\) indexing check.

---

## Mean fitness \(x \cdot \omega\)

At each lattice point \(x=(x_{00},x_{01},x_{10},x_{11})\):

\[
x\!\cdot\!\omega = x_{00}\,\omega_{00} + x_{01}\,\omega_{01} + x_{10}\,\omega_{10} + x_{11}\,\omega_{11},
\]

with flat order `[00, 01, 10, 11]` and \(\omega_g = \exp(g^\top \Sigma g)\).
Code: `log_mean_fitness_genotype_simplex()` → `np.log(x @ omega)`.
Same product as `selection_step` / `diffusion_logdensity` (`X @ omega`).

---

## References in this repo

- `selection_diffusion_vs_chain.py` — cond/influx alpha, \(\psi^*\) vs diffusion
- `explore_neutral_phi.py` — \(\varphi^*\) marginals vs Beta\((2Nu,2Nv)\)
- `allele_diffusion_vs_chain.py` — allele-frequency diffusion (clean \(2Nu-1\) exponents, LE+HWE)
- `SESSION_HANDOFF_2026-06-12.md` — longer derivation notes
- `cond_alpha_vs_chain.py` — neutral + selection comparison table/plots
