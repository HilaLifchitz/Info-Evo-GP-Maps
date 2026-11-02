# Polygenic efficiency

Same trait, more loci: does equilibrium **maintenance** of information get cheaper?

Interactive sketch: [`polygenic-maintenance-efficiency.canvas.tsx`](polygenic-maintenance-efficiency.canvas.tsx) (open in Cursor as a canvas). Handoff notes: [`TODO_tomorrow.md`](TODO_tomorrow.md).

No Python compute here yet. Intended to sit next to the two-locus $\eta=\Delta D/(N C)$ work in [`../2loci/eps12_delta_d_sweep.py`](../2loci/eps12_delta_d_sweep.py).

Maintenance efficiency (once $\Delta D=0$) is

$$
\eta_{\mathrm{maint}}=\frac{D_{\mathrm{eq}}}{kN\langle C\rangle_{\mathrm{eq}}},
\qquad
D(n)=D_{\mathrm{KL}}(\psi_n^*\Vert\varphi_n).
$$
