# Epistasis equilibria (continuous / diffusion)

Haploid **allele-frequency** and **genotype-simplex** equilibria for pairwise epistasis, as the diffusion limit of a Wright–Fisher process. Complementary to the exact discrete chain in [`../2loci/`](../2loci/).

Convention: log-fitness of genotype $g\in\{0,1\}^n$ is $g^\top M g$ with symmetric $M$ ($M_{ii}=s_i$, $M_{ij}=\varepsilon_{ij}/2$).

```bash
conda activate miso
cd epistasis_equilibria
```

---

## Two pictures of equilibrium

### Allele-frequency cube $[0,1]^n$ (linkage equilibrium)

Density on allele frequencies, typically

$$
\rho(x)\;\propto\; W(x)^{N}\prod_i x_i^{2Nu_i-1}(1-x_i)^{2Nv_i-1}
$$

(haploid scaling; product of Betas when $M=0$). This **cannot** represent LD — it is the high-$r$ picture.

| File | Purpose |
|------|---------|
| `equilibria_allele_freq.py` | Core: genotypes, $W(x)$, $\ell$-weights, ESS diagnostics, KL vs neutral |
| `haploid_allele_grid_metrics.py` | Tensor grid: $F_N=\frac1N\log Z$, $\mathrm{KL}(\rho_{\mathrm{eq}}\|\rho_{\mathrm{neutral}})$ [bits], $\mathbb{E}[W]$ |
| `equilibria_allele_freq_viz_n2.py` | $n=2$ 3D surfaces: $\rho_{\mathrm{eq}}$, $\rho_{\mathrm{neutral}}$, $W(x)$ |
| `allele_freq_eps12_sweep.py` | Sweep $\varepsilon_{12}$ at fixed $s_1,s_2$ → `results/eps12_sweep_*/` |
| `allele_freq_eps3_sweep.py` | $n=3$ Cartesian grid over $(\varepsilon_{12},\varepsilon_{13},\varepsilon_{23})$ |

```bash
python allele_freq_eps12_sweep.py
python equilibria_allele_freq_viz_n2.py
```

Example YAML: `equilibria_config_allele_freq_example.yaml`.

### Genotype simplex (4 haplotypes as 4 “alleles”)

Boltzmann / Wright–Dirichlet on the 4-genotype simplex, $\rho\propto W^{2N}$ times Dirichlet mutation weights. This is closer to the $r=0$ chain (no recombination in the diffusion).

| File | Purpose |
|------|---------|
| `equilibria_explored_Sigma.py` | Pairwise $\Sigma$ equilibrium on the simplex |
| `equilibria_explored_low_n_Sigma.py` | Same machinery, small $n$ |
| `equilibria_viz_Sigma.py` | Tetrahedron visualization for $n=2$ |

Example YAML: `equilibria_config_example_Sigma.yaml`.

---

## Small dynamics helpers

| File | Purpose |
|------|---------|
| `discrete_dynamics.py` | One-locus discrete mutation map $x_{t+1}$ |
| `haplotype4_dynamics.py` | Four-haplotype frequencies with $u,v$ |
| `plot_psi_fix_function.py` | Information-style $f(\psi)$ vs allele frequency |

`basics.nb` is a Mathematica notebook of the same equilibria.

---

## Notes

- Neutral = $M=0$. Free fitness on the grid is $F_N=\frac1N\log Z$ with $Z=\sum_k e^{\ell_k}\Delta$.
- Grid `resolution` biases both $Z$ and KL; keep that in mind for coarse sweeps.
- A distributional match to the **$r=0$** discrete chain is **not** expected from the LE allele-frequency form. Scalar KL scores can still be compared; see [`../equilibria_vs_dynamics/`](../equilibria_vs_dynamics/).
- Session notes: `SESSION_HANDOFF.md`, `TOMORROW_HANDOFF.md`.
