# Session handoff — 2026-06-02

Quick orientation for continuing tomorrow. Scope of today: **diffusion-approximation validation** for the 2-locus chain, plus the **equilibrium-score vs adaptation-speed** analysis. Everything below is committed (`eee989d`) and pushed to `origin/master`.

## Where things live

- `2loci/` — exact Wright–Fisher chain on the 4-genotype simplex (`simplex.py`) + new analysis scripts.
- `equilibria_vs_dynamics/` — links `2loci/` and `epistasis_equilibria/` to test if equilibrium score predicts adaptation speed.
- Result figures are gitignored (`*.png`, `*.npz`); regenerate by running the scripts. They land in `*/results/<timestamp>_*/` or the `--out` dir.

### New scripts (all run with `conda activate miso`, from their folder)

| Script | What it does |
|--------|--------------|
| `2loci/explore_neutral_phi.py` | Neutral φ across (N,u,v,r): marginal heatmaps, LD, vs Beta(2Nu,2Nv) |
| `2loci/selection_diffusion_vs_chain.py` | ψ* vs genotype-simplex diffusion `W^N · ∏ x_i^{α_i-1}` (α="cond" local surrogate) |
| `2loci/selection_diffusion_r_N_sweep.py` | Sweep r and N for the genotype-simplex fit |
| `2loci/allele_diffusion_vs_chain.py` | Chain allele marginals vs allele/LE diffusion `W^N · ∏ x_i^{2Nu-1}(1-x_i)^{2Nv-1}` |
| `2loci/selection_magnitude_sweep.py` | Fit (TV/KL) vs scaled selection `N·s_max` |
| `equilibria_vs_dynamics/eps12_score_vs_speed.py` | Equilibrium score vs adaptation speed; trajectory plots colored by score; eps12 sweep |

## Key conclusions reached today (don't re-derive)

1. **Neutral φ is not uniform.** Corner-concentrated (U-shape) when `2Nu≪1`; central hump when `2Nu≫1`; asymmetric u/v shifts it. Per-locus marginals match Beta(2Nu,2Nv) very well (TV 1e-4–1e-2). Diffusion predicts D=0; chain has E|D|≈0.04 at r=0, ~halved at r=0.5 (recombination reshapes the joint, not the marginals).

2. **Diffusion ↔ chain selection-equilibrium fit is governed by `N·s_max`.**
   - Near-exact for weak selection: `N·s_max ≲ 1` ⇒ KL < 0.05 bits, TV < 0.09.
   - Degrades **monotonically** with `N·s_max` (KL ≈ 0.2 bits at Ns≈5; ≳0.45 at Ns≳18). This is a **strong-selection (O(s²)) correction**, NOT finite-N discreteness.
   - ⇒ This explains why **bigger N didn't help** at fixed s: raising N raises Ns. The diffusion limit needs s→0 with Ns fixed.

3. **The residual is a selection-SCALING bias, not the α surrogate.**
   - `W^{2N}` over-pushes the favored genotype; `W^{N}` under-pushes ⇒ chain's effective exponent is **between W^N and W^{2N}**.
   - Very small mutation did NOT close the gap ⇒ α (the ambiguous "rate into genotype i") is not the dominant error. We use the per-capita conditional influx `α_i(x)=2N·Σ_{j≠i}x_jM_{ji}/Σ_{j≠i}x_j` (reduces exactly to 2-allele Wright).
   - NOTE: V uses **−N·logW** (corrected from −2N earlier today); mutation term keeps 2N. Allele/LE formula uses 2N→N and 4N→2N (haploid).

4. **Each diffusion matches its own recombination regime.** Genotype-simplex form = r=0 limit (best at r=0, optimum of the r/N sweep around r≈0.25, worsens as r↑). Allele/LE form assumes LE (improves as r↑). They cross over around Ns≈2–3.

5. **Equilibrium score vs adaptation speed** (`equilibria_vs_dynamics`):
   - Information **dip**: `KL_eq` is U-shaped in ε₁₂, min near ε₁₂≈+0.02–0.03; at the dip the chain barely adapts (lowest dKL/dt to T=800, N=20).
   - Equilibrium score tracks speed **at the extremes** but is **not a clean monotone predictor**: ε₁₂<0 curves have near-identical KL_eq (~0.49) yet different speeds; coloring by free fitness shows a **U-shaped** speed relation.
   - Free fitness is monotone in ε₁₂; equilibrium information is U-shaped.

## Open thread / next step (where we paused)

**The `W^{cN}` effective-exponent scan** — I proposed it and the user said "yes, but first explore selection magnitudes" (done, conclusion #2). So the natural NEXT action tomorrow:

- Scan effective selection exponent `c ∈ [1,2]` in `p ∝ W^{cN} · (mutation term)`, find the `c` minimizing KL(ψ*‖diffusion), for:
  - genotype-simplex form at r=0, and
  - allele/LE form at high r,
  - across a few `N·s_max` values.
- **Expected result:** best-fit `c` ≈ 1 at weak selection, drifting upward (toward/above the N–2N range) as selection strengthens. This would quantify the selection-scaling bias from conclusion #3.
- Easiest implementation: parametrize the `N`-multiplier in `diffusion_logdensity` (genotype) and in `allele_diffusion` (allele) by `c`, add a small scan loop + plot KL vs c per Ns. Reuse `selection_magnitude_sweep.py` machinery.

### Other possible directions (lower priority)
- Refine r grid around 0.1–0.4 to pin the genotype-simplex optimum.
- LD residual heatmap (chain joint − product-of-marginals) as a function of r.
- Tetrahedron (3-simplex) visualization of ψ* vs diffusion.

## Gotchas
- `python` is not on PATH; use `python3` after `conda activate miso` (`source ~/miniconda3/etc/profile.d/conda.sh`).
- Matplotlib prints a harmless `qt.qpa.plugin: ... wayland` warning — ignore.
- N=50 dense transition matrix ≈ 4.4 GB (one matrix; `column_stochastic` is a view, no copy). ~16 GB free, so fine. Builds: ~5s (N=40), ~10s (N=45), ~17s (N=50).
- ψ* solved by ARPACK (`scipy.sparse.linalg.eigs`) with power-iteration fallback — fast even for weak selection.
- Comparisons are on the **interior** support (all genotypes/loci polymorphic); boundary mass is reported separately and balloons under strong selection.
