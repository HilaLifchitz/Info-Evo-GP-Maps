# Initial thoughts and codes

The **first-generation** Wright–Fisher / information-theory scripts. They predate the genotype-simplex chain in [`../2loci/`](../2loci/). Kept for reference and for one-locus / phenotype-Z experiments; the current two-locus work does **not** import from here.

These files used to live at the repo root. They were moved here when `2loci/` became the main simulator.

```bash
conda activate miso
cd "initials_thoughts&codes"
```

---

## Markov chains

| File | Purpose |
|------|---------|
| `Markov_1_locus.py` | Finite haploid Wright–Fisher on \(\{0,\ldots,N\}\): transition matrix, stationary \(\pi\), Iwasa-style \(D(t)=\mathrm{KL}(\psi(t)\|\varphi^*)\) and \(I(t)=\mathrm{KL}(\psi(t)\|\psi^*)\) |
| `Markov_2_loci.py` | Two layers: (i) independent-loci product of 1-locus chains; (ii) full LD model with recombination, pushforwards \(X\to P\) and \(X\to G\), spectral \(D/I\) series |
| `diffusion_vs_markov.py` | Compare 1-locus chain \(\pi_M\) to haploid diffusion \(\propto x^{2Nv-1}(1-x)^{2Nu-1}e^{2Nsx}\) (L1, KL, KS vs \(N\)) |
| `evolution_analysis.py` | Neutral→selection trajectory; decomposition \(D(t)=I(t)+\mathrm{rest}(t)\) |

`Markov_2_loci.py` is the ancestor of `2loci/simplex.py` (same recombination convention, different packaging).

---

## Phenotype / “QE” simulations

Finite-locus Wright–Fisher with optional free recombination, tracking a phenotype \(Z\) and KL between distributions.

| File | Purpose |
|------|---------|
| `QE.py` | Core simulators: basic WF, free recombination, directional / stabilizing trait evolution; analytic \(\varphi(t)\), \(\psi(t)\) |
| `3.6B_maybe.py` | Phenotype-histogram KL vs binomial; \(\varphi\) vs \(\psi\) on locus and phenotype spaces |
| `convergence_analysis_nick.py` | Multi-run averaging, autocorrelation, checkpointed long trajectories |
| `test_simple_neutral_approach.py` | Neutral-KL heuristic for “has this stochastic run equilibrated?” |
| `kl_comparison_test.py` | Side-by-side KL methods (historical 80-bit vs 47-bit discrepancy) |
| `quick_80bit_test.py` | Parameter hunt around that discrepancy |

Also here: `Email_to_nick_05_25.pdf` (discussion notes).

---

## Relation to the rest of the repo

- Exact two-locus simplex: [`../2loci/simplex.py`](../2loci/simplex.py)
- Diffusion equilibria: [`../epistasis_equilibria/`](../epistasis_equilibria/)
- Score vs speed: [`../equilibria_vs_dynamics/`](../equilibria_vs_dynamics/)
