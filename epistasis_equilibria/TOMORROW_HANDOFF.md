# Handoff notes (2026-04-28)

## Main goal
Work on allele-frequency equilibrium tooling (`n=2` haploid first), diagnostics, and visualization.

## Files added/updated
- `equilibria_allele_freq.py`
  - Added extra diagnostics:
    - `KL(ρ_eq || ρ_neutral[M=0])`
    - free-fitness quantities at equilibrium (discrete support approximation)
    - weak-selection check: `N * max_g |g^T M g|`
  - Added ESS quality output:
    - prints `ESS / support_points` and label (`bad/weak/okay/good/very good`)
    - guide: `~1% bad`, `5-20% usable`, `>20% good`
- `equilibria_allele_freq_viz_n2.py` (new)
  - Dedicated plotting/debug helper for `n=2`, `haploid`.
  - Figure 1:
    - 3D `ρ_eq` (with M)
    - 3D `ρ_neutral` (M=0)
    - 3D `W(x)`
  - Figure 2:
    - `log(ρ_eq/ρ_neutral)` map
    - `Δℓ = ℓ_eq - ℓ_neutral` map
    - scatter `log W` vs `log(ρ_eq/ρ_neutral)` with linear fit
  - Added fast corner diagnostics (console + on-plot text boxes):
    - corners `00,10,01,11`
    - spread = `max-min`
    - relative spread = spread / mean corner value
  - Colormaps for Figure 1 reversed (`*_r`) for clearer peaks.

## Config currently used
- `equilibria_config_allele_freq_example.yaml`
  - currently set to `n_loci: 2`, `ploidy: haploid`
  - has 2x2 `sigma` matrix.

## Important interpretation note
- Neutral symmetry check is based on corner equality for `ρ_neutral` (with equal u/v assumptions).
- “Spread” in diagnostics:
  - absolute spread: `max(corners)-min(corners)`
  - relative spread: `(max-min)/mean(corners)`.

## Run commands
- Main solver:
  - `python equilibria_allele_freq.py --config equilibria_config_allele_freq_example.yaml`
- n=2 viz/debug:
  - `python equilibria_allele_freq_viz_n2.py --config equilibria_config_allele_freq_example.yaml`
- Save images:
  - `python equilibria_allele_freq_viz_n2.py --config equilibria_config_allele_freq_example.yaml --save-prefix n2`

## Next suggested steps
1. Add optional save of new diagnostics (`kl`, `free_fitness`, `ess_ratio`, etc.) to `.npz`.
2. Add sweep utility over `N` and one matrix scale parameter to map:
   - `KL(ρ_eq || ρ_neutral)`
   - free fitness
   - ESS ratio.
3. If needed, add percentile-clipped color scaling for subtle structure in comparison maps.
