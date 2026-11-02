# TODO — 31 Oct 2026 (handoff from 30 Oct)

Git copy of the sketch (this folder):

`polygenic-maintenance-efficiency.canvas.tsx`

Cursor’s live canvas panel still compiles from its internal `canvases/` directory; keep that copy in sync if you edit the panel. Slider state lives in a `.canvas.data.json` sidecar next to the live file.

---

## What we built today (canvas)

Interactive sketch for: same trait, more loci — does equilibrium **maintenance** get cheaper?

- Title: *More genes, more efficient?*
- Definitions: $\eta_{\mathrm{maint}}(n)=D_{\mathrm{KL}}(\psi_n^*\Vert\varphi_n)/\langle C_n\rangle_{\psi_n^*}$, cost $C(x)=D_{\mathrm{KL}}(x'\Vert x)$, distinct from transient $\eta(t)=\Delta D/(N\langle C\rangle)$.
- Fair scaling table: hold $W(z)$ and $z\in[0,1]$; vary encoding / $n$.
- Additive encoding $z_n(g)=k/n$. First fitness $\omega_n(g)=\exp[-s(z_n(g)-z^*)^2]$.
- Hamming-weight Hasse, **left → right** ($000\ldots$ on the left).
- Trait color: white ($z=0$) → black ($z=1$).
- Fitness color: red (low $W$) → green (high $W$), sliders for $z^*$ and $s$.
- $n$ from **2 to 1000** (log slider). Jumps: 2, 4, 6, 7, 8, 10, 12, 16, 20, 50, 80, 100, 200, 500, 800, 1000.
- **Full node graph only for $n\le 7$** (128 genotypes). For $n\ge 8$: one bar per Hamming layer, height $\sqrt{C(n,k)}$. At $n=1000$ you cannot draw $2^{1000}$ vertices.
- Equations are pre-rendered Computer Modern SVG (canvas cannot import KaTeX).

## What a canvas cannot do

It is an isolated React panel (`import` only from `cursor/canvas`). **No Python, no `2loci/` imports, no running the chain.** Landscape coloring is JS. $\psi^*$, $\varphi$, $C$, $\eta_{\mathrm{maint}}(n)$ stay in Python.

## Next (do this, not more canvas plumbing)

Write a script **in this folder** that computes $\eta_{\mathrm{maint}}(n)$ for the additive + stabilizing family.

Reuse, do not rewrite:

- `2loci/compare_sigma_dynamics.py` — `cost_per_state`
- `2loci/simplex.py` — `stationary_from_matrix` / `stationary_fast`, `kl_divergence_bits`

For $n\le 7$ the chain is $2^n\le 128$ states (exact). Larger $n$: diffusion / Boltzmann as in `epistasis_equilibria/`, or collapse onto Hamming layers (additive $W$ is constant on layers).

Still open with Hila: how $N$, $u$, $v$, recombination scale with $n$.

When numbers exist, paste a JSON snapshot into the canvas if we want a curve there.

## Also from today (not the canvas)

Two-locus KL-dip / coupled mutation work; READMEs pushed (`d4c1851` on `origin/master`); conda env `miso` at `~/miniconda3/envs/miso` (`source ~/miniconda3/etc/profile.d/conda.sh` first).
