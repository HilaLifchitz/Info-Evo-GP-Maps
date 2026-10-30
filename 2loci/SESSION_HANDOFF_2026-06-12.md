# Session handoff — 2026-06-12 (eps12 dip panels, Cov/Var check, diffusion alpha)

## Big picture

Two threads:

1. **KL-dip location** under an eps12 sweep: where does D_KL(psi*||phi*) minimize?
   Leading theory: eps12* = -Cov_phi(B, x11) / Var_phi(x11), which for symmetric
   u=v reduces to eps12* ≈ -2/3 (s1+s2) at weak selection.

2. **Discrete vs diffusion equilibria** on the genotype simplex: compare chain
   stationary psi* to Boltzmann diffusion p(x) ∝ W(x)^N prod_i x_i^{alpha_i-1},
   using a tweaked **conditional alpha** because our mutation matrix is not
   parent-independent.


# Part A — KL-dip work (continued from 2026-06-11)

## Sign / cancellation panels

### Original sign4b (4 pairs)
Demonstrated that overshoot/undershoot vs -2/3 is NOT about mixed vs same signs
alone — at u=v only s1+s2 matters. Pairs with identical s1+s2 give identical
curves (blue/teal overlapped when s1+s2 = +1.5e-3).

### Expanded sign4b panel (7 lines) — `eps12_dip_sign_pairs2.py`
`plots/eps12_dip_robustness/dip_vs_u_sign4b.png`

Kept 3 from sign4b cache + 3 from sign4 cache + 1 sum-zero line; no new chain
fits needed (all seeded from existing .npz files).

| color | pair (s1, s2) | s1+s2 | source |
|-------|---------------|-------|--------|
| blue  | (2e-3, -5e-4) | +1.5e-3 | sign4b |
| purple| (1.2e-3, 4e-4) | +1.6e-3 | sign4b |
| green | (1e-3, -5e-4) | +5e-4 | sign4 |
| orange| (-4e-4, -1.2e-3) | -1.6e-3 | sign4b |
| magenta| (-5e-4, -5e-4) | -1e-3 | sign4 |
| teal  | (-1e-3, 5e-4) | -5e-4 | sign4 |
| black | (1e-3, -1e-3) | 0 | sum0 |

**Key findings (non-zero s1+s2):**
- At small u: fitted dips track -2/3(s1+s2) well.
- As u grows: positive s1+s2 undershoots (dip less negative than pred);
  negative s1+s2 overshoots (dip more positive than pred).
- Effective c(u) = -eps_fit/(s1+s2) grows above 2/3 with u, symmetrically in sign.
- At u=v, r1≈r2: only s1+s2 matters, not which locus dominates.

## s1+s2 = 0 — `eps12_dip_sum0.py`
`plots/eps12_dip_robustness/dip_vs_u_sum0.png`
`plots/eps12_dip_robustness/data_u_sum0.npz`

Three canceling pairs (s1, -s1) with |s| = 5e-4, 1e-3, 2e-3; 27 new fits.

**Results:**
- Linear prediction eps* = 0 is exact, but fitted dips are NOT zero at small u.
- All dips stay **positive** (overshoot vs pred=0).
- Magnitude scales roughly with |s|^2 (ratio 1:4:16 at small u).
- Drift **shrinks** as u increases (by u=0.3 all near zero).

**Interpretation:**
- -2/3(s1+s2) is a first-order formula; when s1+s2=0 that term vanishes
  identically and a second-order leftover (~ s1*s2 = -s1^2) appears.
- At large u the second-order bump fades (closer to neutral), so prediction
  "improves" — opposite trend from non-zero s1+s2 where c(u) drifts.

Example fitted dips (1e-3, -1e-3):
  u=1e-5  -> +0.034 (x1e-3)
  u=1e-2  -> +0.020
  u=0.3   -> +0.0015

## Cov/Var sanity check — `eps12_dip_cov_prediction.py`
`plots/eps12_dip_robustness/dip_cov_vs_fit.png`       (scatter)
`plots/eps12_dip_robustness/dip_vs_u_cov_check.png`   (curves)
`plots/eps12_dip_robustness/data_cov_check.npz`

Full prediction (not the -2/3 shortcut), evaluated at neutral phi* for each u:

    B = s1*p1 + s2*p2     (additive background fitness at each lattice point)
    x11 = haplotype-11 frequency
    eps12* = -Cov_phi(B, x11) / Var_phi(x11)

All moments under neutral phi* loaded from cache; B uses marginal allele freqs
p1, p2 from genotype fractions on the simplex grid.

**Agreement with parabola-fit dips:**

| regime | Cov/Var vs fit |
|--------|----------------|
| u ≲ 1e-3 | ~1–6% relative error (s1+s2 ≠ 0) |
| u ≈ 0.01 | still within ~3% |
| u = 0.3 | ~36% gap (fit overshoots Cov/Var); still better than fixed -2/3 (~42%) |
| s1+s2 = 0 | Cov/Var ≈ 0 exactly; fit has O(|s|^2) bump at small u |

**Takeaway:** Cov/Var is the right first-order theory; it tracks u-dependence
through phi* and beats the constant -2/3 shortcut. Remaining gap at large u is
finite-selection / beyond-first-order. For s1+s2=0 the Cov/Var formula
correctly gives zero; the positive fitted offset is second-order.

At u=0.3 example (1e-3, -5e-4):
  fit = -4.74e-4,  Cov/Var = -3.49e-4,  -2/3 sum = -3.33e-4


# Part B — Discrete vs diffusion equilibria; conditional alpha

## What we compare

**Discrete (ground truth):**
Stationary distribution psi* of the full Wright-Fisher Markov chain on genotype
count states, with selection omega(g)=exp(g^T Sigma g), mutation M, recombination r.

**Diffusion (prediction):**
Boltzmann density on genotype simplex x = (x00, x01, x10, x11):

    p(x) ∝ W(x)^N * prod_i x_i^{alpha_i(x) - 1}
    V(x) = -N log W(x) + sum_i (1 - alpha_i(x)) log x_i

where W(x) = sum_i x_i * omega(g_i) is mean fitness at composition x.

Comparison restricted to **interior** lattice points (all four genotypes present);
boundary mass of psi* reported separately (diffusion has singularities when
alpha_i < 1 and x_i -> 0).

Main script: `selection_diffusion_vs_chain.py`
Related: `allele_diffusion_vs_chain.py` (allele-freq diffusion under LE+HWE,
clean 2Nu exponents), `sigma_sweep_surpass_Ns.py`, `selection_diffusion_r_N_sweep.py`.

## Why we need a tweaked alpha

Standard 4-allele Wright diffusion assumes constant mutation rate into each
genotype. Our M is built from independent per-locus rates (u_i forward, v_i
backward) — NOT parent-independent in the sense needed for a global Dirichlet
on 4 genotypes. So alpha_i cannot be a single constant; we use a local,
x-dependent surrogate.

Two modes implemented; default is **"cond"**:

  "cond" (default):
    alpha_i(x) = 2N * [sum_{j!=i} x_j M_ji] / [sum_{j!=i} x_j]
               = 2N * influx_i / (1 - x_i)

  "influx" (alternative, generally worse):
    alpha_i(x) = 2N * sum_{j!=i} x_j M_ji

## Conditional alpha — step by step

At lattice point x on the genotype simplex:

1. Post-mutation mass:
     XM = x @ M          (row vector x times mutation matrix)

2. Mutational inflow into genotype i (from j ≠ i only):
     influx_i = (XM)_i - x_i * M_ii
              = sum_{j!=i} x_j M_ji

3. Per-capita conditional rate (normalize by everyone NOT already genotype i):
     denom_i = 1 - x_i = sum_{j!=i} x_j
     rate_i  = influx_i / denom_i

4. Wright exponent surrogate:
     alpha_i(x) = 2N * rate_i

5. Log unnormalized density (interior only):
     log p(x) = N * log W(x) + sum_i (alpha_i(x) - 1) * log x_i

Then softmax over interior points to normalize.

**Why "cond" is default:**
- Reduces EXACTLY to alpha = 2Nu in the 1-locus limit (correct asymmetric Wright
  exponent).
- "influx" uses raw inflow without per-capita normalization and fits worse.

**Code location** (`selection_diffusion_vs_chain.py`, `diffusion_logdensity`):

    XM = X @ M_mut
    influx = XM - X * diag(M_mut)
    alpha = 2.0 * N * influx / (1.0 - X)    # "cond" mode
    log_p = N * log_W + sum((alpha - 1) * log(x))

## Mutation matrix reminder (genotype order 00, 01, 10, 11)

Independent per-locus: P(0->0)=1-u, P(0->1)=u, P(1->0)=v, P(1->1)=1-v.
Joint M_ij = product of locus factors for source genotype i -> target j.


# Cached data (eps12 dip robustness)

    plots/eps12_dip_robustness/
      data_u_merged.npz      — 2 pairs, 9 u values
      data_u_sign4.npz       — 4 sign-test pairs
      data_u_sign4b.npz      — 4 asymmetric pairs
      data_u_sum0.npz        — 3 sum-zero pairs
      data_cov_check.npz     — Cov/Var predictions for 7 pairs x 9 u
      dip_vs_u_sign4b.png    — 7-line panel
      dip_vs_u_sum0.png
      dip_cov_vs_fit.png
      dip_vs_u_cov_check.png

Note: data_u_sign6.npz is referenced in sign_pairs2 but may not exist on disk
if all values were seeded without entering the fit loop; sign_pairs2 seeds from
sign4b/sign4/merged/sum0 instead.


# How to run

```bash
# 7-line sign panel (seed + replot)
python 2loci/eps12_dip_sign_pairs2.py
python 2loci/eps12_dip_sign_pairs2.py --replot

# s1+s2=0 sweep (~25 min for 27 fits if cache empty)
python 2loci/eps12_dip_sum0.py
python 2loci/eps12_dip_sum0.py --replot

# Cov/Var sanity check (fast — phi from cache only)
python 2loci/eps12_dip_cov_prediction.py
python 2loci/eps12_dip_cov_prediction.py --replot

# Discrete vs diffusion selection equilibrium
python 2loci/selection_diffusion_vs_chain.py
python 2loci/allele_diffusion_vs_chain.py
```


# Open next steps

1. **Equilibrium agreement revisit:** rerun discrete-vs-diffusion comparison at
   the weak-selection / low-mutation regime used in KL-dip sweeps (N=30,
   u=v~1e-3..1e-1, small s1,s2,eps12).

2. **u ≠ v:** break symmetric mutation; Cov/Var prediction will change (not
   just -2/3(s1+s2)).

3. **Second-order dip at s1+s2=0:** test eps* ~ k * s1*s2 * h(u) numerically.

4. **Extract c(u) from neutral phi:** Var(x11), Cov(B,x11) at each u — partly
   done via eps12_dip_cov_prediction.py; could plot c(u) = -eps_cov/(s1+s2).

5. **Ploidy / diploid diffusion** (from 2026-06-10 handoff): chain is haploid
   Beta(2Nu); naive N->2N diploid diffusion fails.


# Prior session context

- `SESSION_HANDOFF_2026-06-11.md` — initial dip prediction, robustness, gallery
- `SESSION_HANDOFF_2026-06-10.md` — diffusion-vs-chain, sigma sweeps, ploidy
