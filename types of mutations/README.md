# Types of mutations — small-world epistasis

Ask whether mutations on a **Watts–Strogatz** interaction graph look House-of-Cards-like: is the effect of flipping locus \(k\) essentially unpredictable from the rest of the genotype?

Alleles are binary (\(0\) wild type, \(1\) mutant). Fitness of \(g\in\{0,1\}^n\):

\[
\log\omega(g)=g^\top M g,\qquad
M_{ii}=s_i,\quad
M_{ij}=M_{ji}=\varepsilon_{ij}/2
\]

for each graph edge \((i,j)\). Off-diagonal entries that are not edges are zero — sparse epistasis.

```bash
conda activate miso
cd "types of mutations"
python small_world_epistasis.py --n 12 --k 4 --rewire-prob 0.05 --seed 42
```

Full flag list and output filenames: [`WORK_SESSION_NOTES.md`](WORK_SESSION_NOTES.md).

---

## What the script does

1. Build a connected Watts–Strogatz graph (`networkx`).
2. Sample edge effects \(\varepsilon_{ij}\) (uniform or normal) and diagonal \(s_i\) (constant / uniform / normal).
3. Enumerate all \(2^n\) genotypes (capped by `--max-genotypes`).
4. For every \(0\to 1\) mutation, compute the effect on every genetic background.
5. House-of-Cards-style summaries (mean/variance of mutational effects, background dependence).
6. Information epistasis \(D(g_k\mid g_{-k})-D(g_k)\).
7. Write CSV / NPY / text report and plots into a timestamped `results/YYYY-MM-DD_HH-MM-SS/` folder (gitignored).

---

## Files

| File | Purpose |
|------|---------|
| `small_world_epistasis.py` | The experiment |
| `WORK_SESSION_NOTES.md` | Goal, CLI, output schema, open questions (last updated 2026-05-05) |

This is **not** the two-locus Markov chain. It is a combinatorial landscape on \(\{0,1\}^n\) with a sparse interaction graph. For dynamics on 4 genotypes see [`../2loci/`](../2loci/).
