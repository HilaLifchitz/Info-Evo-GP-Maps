# Session handoff — 2026-06-10 (2-locus diffusion vs discrete chain)

## Big picture / what this thread is about
We are validating the **diffusion (Boltzmann) approximation** of the discrete
2-locus Wright–Fisher **selection** equilibrium on the 4-genotype simplex
(`00, 01, 10, 11`), all at recombination `r = 0`.

- **Chain side** (`2loci/simplex.py`): stationary `psi*` of the discrete WF chain.
  Transition = `Multinomial(N, p)` with `p(g) ∝ x_g·ω(g)`, `ω(g)=exp(gᵀΣg)`,
  asymmetric mutation `M`. Confirmed: it samples **N gene copies** (drift var
  `x(1-x)/N`) with standard haploid selection → **the chain is HAPLOID**.
- **Diffusion side**: `p(x) ∝ W(x)^N · ∏ xᵢ^{αᵢ-1}`, `W(x)=Σ xᵢωᵢ`,
  `αᵢ = 2N·(mutation influx into i)`. Two surrogates for the non-reversible `M`:
  `"cond"` (per-capita, reduces to Wright's `2Nu`; the principled one we use) and
  `"influx"` (raw). Comparison is done on the simplex **interior** (all 4
  genotypes present); chain boundary mass reported separately.
- Effective per-genotype selection: `s_eff(01)=s2`, `s_eff(10)=s1`,
  `s_eff(11)=s1+s2+ε₁₂` (= `log ω(11)`).

## Files touched today
- `2loci/sigma_sweep_surpass_Ns.py` — NEW. Fix `N=40`, fix `u=v`, scale a Σ
  shape so `N·max|s_eff|` sweeps from <1 to >1; plots interior TV/KL and chain
  boundary mass vs `N·s_eff`, plus representative marginals. The `U = V` constant
  near the top is what we edit between runs (currently set to `0.1`).
- `2loci/ploidy_test.py` — NEW. Haploid-vs-diploid test (see below).
- `2loci/selection_diffusion_vs_chain.py` — the base script (helpers imported by
  the two new ones: `Setting`, `discrete_psi_star`, `diffusion_logdensity`,
  `_normalize_log`, `_tv_kl`, `genotype_marginals`). Not modified today.

## Experiments run today (results in `2loci/results/`)

### 1. `N·s_eff` surpass-1 sweep at very weak mutation — `results/sigma_sweep_surpass_u1e-4/`
`N=40`, `u=v=1e-4`, `N·s_eff` 0.05→3.0.
- Chain **boundary mass ≈ 0.99999** everywhere (`2Nu=0.008≪1` → almost always
  monomorphic). Interior comparison is a thin-conditional slice → not very
  meaningful at this mutation rate.
- No cliff at `N·s_eff=1`; smooth, convex rise. KL overtakes TV ~`N·s_eff≈1.5`.
  Genotype `11` goes bimodal in the chain by `N·s_eff=3` while diffusion
  under-pulls it. Weak-`Ns` floor ≈ TV 0.17 (higher than at strong mutation).

### 2. Same sweep at realistic mutation — `results/sigma_sweep_surpass_u0.1/`
`N=40`, `u=v=0.1`, `N·s_eff` 0.05→3.0.
- Boundary mass only ≈ **0.004** (`2Nu=8`, strongly polymorphic) → interior
  carries ~99.6% of mass → **this is the clean, meaningful regime**.
- Approximation good and nearly flat: TV `0.101→0.111`, KL `0.091→0.106 bits`
  across `N·s_eff` 0.05→3.0. Gentle convex degradation past 1, no cliff.
  Marginals (incl. `11`) sit on the diffusion curve even at `N·s_eff=3`.
- Residual floor ≈ TV 0.10 / KL 0.09 is selection-independent
  (mutation-surrogate + discreteness).

### 3. Haploid vs diploid — `results/ploidy_test/`
Motivated by: "what if the discrete model is actually diploid? test by `N→2N`
in the diffusion formula (`W^N→W^{2N}`, `α=2Nu→4Nu`)."
- **TEST A (decisive, selection-free):** neutral chain locus-1 allele-freq
  marginal matches **haploid `Beta(2Nu=8)`** (TV 0.027, var 0.0166 vs 0.0147)
  and is far from diploid `Beta(4Nu=16)` (TV 0.195, var 0.0076).
  → **chain is HAPLOID, n_c = N.**
- **TEST B:** chain `psi*` vs 3 diffusion variants at `u=v=0.1`:
  - current `W^N, 2Nu`: TV 0.10→0.11
  - fix-sel `W^{2N}, 2Nu`: TV 0.10→0.13 (slightly worse at high `Ns`)
  - diploid `W^{2N}, 4Nu` (the `N→2N` idea): **TV ≈ 0.37, KL ≈ 1.24 bits**
    (catastrophically worse) → correctly **rejects diploid**.
- Conclusion: `N→2N` is a **fair test of ploidy/copy-number** (ploidy enters the
  diffusion as the gene-copy count `n_c` multiplying both terms). Caveats: it
  tests *drift scale*, not *dominance*; cleanest discriminator is the neutral
  marginal width (TEST A).

## Open / unresolved
- **Selection-exponent factor-2 puzzle.** Standard single-locus derivation for
  this chain gives `ψ ∝ x^{2Nu-1}(1-x)^{2Nv-1} e^{2Nsx}` → selection should be
  `W^{2N}` (same `2N` as the confirmed mutation `2Nu`). But TEST B showed
  `W^{2N}` did **not** fit better than the current `W^N` (slightly worse at high
  `Ns`). Likely the ~0.10 mutation-surrogate floor (non-reversible `M`) masks the
  selection-exponent signal. **NOT resolved.**
- **Proposed clean test (next):** switch to **parent-independent / reversible
  mutation** so the Wright `α` surrogate becomes *exact* and the mutation floor
  vanishes — then `W^N` vs `W^{2N}` is the only remaining knob and the selection
  exponent can be pinned down cleanly.

## Other candidate next steps
- Implement the **boundary-layer integral** `∫_{xᵢ<1/N} p(x)dx ≈ C·(1/N)^{αᵢ}/αᵢ`
  to compare diffusion boundary mass directly to the chain's boundary atoms
  (discussed: diffusion has no boundary atom; `xᵢ^{αᵢ-1}` gives an integrable
  edge singularity for `0<αᵢ<1`, vanishes for `αᵢ>1`, absorbing for `αᵢ≤0`).
- Sweep `u=v` (1e-4 → 0.1) at fixed weak selection to map where interior mass /
  accuracy become usable.
- Push `N·s_eff` higher (5–10) to find the visible break.

## How to run
```
"/home/hlifchit/miniconda3/bin/conda" run -n miso python 2loci/sigma_sweep_surpass_Ns.py --out 2loci/results/<name>
"/home/hlifchit/miniconda3/bin/conda" run -n miso python 2loci/ploidy_test.py --out 2loci/results/ploidy_test
```
`N=40` chain build ≈ 6 s each; memory ~1.2 GB at N=40 (N=60 would be >12 GB — avoid).
```
