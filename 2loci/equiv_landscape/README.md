# Equivalent landscapes

Find different \((s_1,s_2,\varepsilon_{12})\) that realize the **same relative fitness shape** on the four haplotypes, and scan slices of constant mean log-fitness \(\langle\ln W\rangle\).

Fitness (same as [`../simplex.py`](../simplex.py)):

\[
\Sigma=\begin{pmatrix}s_1&\varepsilon_{12}/2\\\varepsilon_{12}/2&s_2\end{pmatrix},\qquad
\log\omega=[0,\,s_2,\,s_1,\,s_1+s_2+\varepsilon_{12}]
\]

Three log-relative scores vs genotype 00 fix the shape. If \(\varepsilon_{12}\) is perturbed, an exact match of all three with only \((s_1,s_2)\) free is impossible — `find_equiv_sigma.py` does a least-squares refit.

```bash
cd 2loci
python -m equiv_landscape.find_equiv_sigma
python -m equiv_landscape.find_equiv_sigma --s1 1e-3 --s2 -5e-4 --eps12 3e-4 --eps-new 1e-3 --scan
python -m equiv_landscape.match_wstar_min_d
```

| File | Purpose |
|------|---------|
| `find_equiv_sigma.py` | Match relative \(\omega\) shape; optional \(\varepsilon\) scan |
| `match_wstar_min_d.py` | Iso-\(\langle\ln W\rangle\) grids; heatmaps of \(D\); batch min-\(D\) at matched \(\langle\ln W\rangle\) |
