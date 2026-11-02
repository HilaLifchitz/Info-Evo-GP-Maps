# Polygenic η_maint — handoff 2 Nov 2026

Live site: https://hilalifchitz.github.io/Info-Evo-GP-Maps/index.html

Three copies: Cursor canvas (`canvases/polygenic-maintenance-efficiency.canvas.tsx`) → this folder → `web/` (Pages). Pages only rebuilds from `web/**`.

## What is in place

- Additive \(z=k/n\), Gaussian stabilizing \(W=\exp[-(z-z^*)^2/(2V_s)]\).
- Boltzmann on the **genotype simplex**, reduced to Hamming layers:
  \(\rho\propto W^{2N}\prod x_g^{\alpha-1}\), layer Dirichlet \(\mathrm{Dir}(\alpha\binom{n}{0},\ldots)\).
- Direct \(\alpha\) (parent-independent). Cond-\(\alpha\) not wired.
- \(\eta=D/(2N\langle C\rangle)\), \(F^*=\langle\log W\rangle_*-D_{\mathrm{nats}}/(2N)=\log Z/(2N)\).
- Two scalings:
  - **α fixed:** \(\alpha_0=2^n\alpha\) explodes; \(D\to0\); η falls.
  - **α₀ = Kα fixed:** \(\alpha(n)=\alpha_0/2^n\).

Python: `eta_maint_boltzmann.py` (`--alpha` or `--alpha0`). Sweep: `search_eta_vs_n.py`.

## Finding: η can rise with n

Hold **α₀ small and fixed**, compare **even n** at \(z^*=0.5\).

Poster case: **α₀=4, N=50, Vs=0.0625**
η(n) ≈ 7.1, 9.8, 10.4, 11.7, 12.0, 15.0 at n=4,6,8,12,16,24.
D falls; ⟨C⟩ falls faster.

- Larger N steepens the n-gain until \(W^{2N}\) kills ESS.
- Smaller Vs (stronger selection) steepens the n-gain; large Vs flattens it and raises the level.
- Canvas default **α₀=32** is near-flat. **α₀=4** is the rising band.
- Odd n at \(z^*=0.5\) dips (no genotype at the peak).
- Large N × small Vs → ESS ~1; ignore those η numbers.

## Open

- Cond-α / independent per-locus mutation (Dirichlet is PIM).
- How N, μ scale with n biologically.
- JSON snapshots into `web/public/data/` if we want a precomputed curve.
- Do not commit `2loci/compare_sigma_dynamics.py` or the pptx unless asked.
