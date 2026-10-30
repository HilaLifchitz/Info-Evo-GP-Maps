# Session handoff — 2026-06-11 (eps12 KL-dip prediction)

## Big picture
We are testing the weak-selection prediction for where the **KL(psi*||phi*) dip**
occurs under an **eps12 sweep**:

    eps12* = -2/3 (s1 + s2)     [symmetric u = v]

This comes from eps12* = -Cov(B, x11)/Var(x11) at leading order. The dip is
the epistasis value that minimizes stationary information content relative to
neutrality.

## What we did today

### 1. Dip-prediction test (`eps12_dip_prediction_test.py`)
- At N=30, u=v=1e-3, r=1e-3: swept eps12 around predicted dip for 6 (s1,s2)
  pairs; fit local parabola; compare vertex to -2/3(s1+s2).
- **Result:** prediction holds very well (errors ~1e-5, << grid step).
  Depends only on s1+s2 (pairs with same sum give same dip).
- Plots: `plots/eps12_dip_test/dip_curves.png`, `dip_pred_vs_fit.png`
- Data cached: `plots/eps12_dip_test/data.npz` (instant `--replot`)

### 2. Robustness vs r and u (`eps12_dip_robustness.py`)
- **r:** dip location flat across r = 0 → 0.5 (essentially r-independent).
- **u=v (original, up to 1e-2):** flat below ~1e-3; drifts ~10% by u=1e-2.
- Data: `plots/eps12_dip_robustness/data_r.npz`, `data_u.npz`
- Plots: `dip_vs_r.png`, `dip_vs_u.png`

### 3. Gallery of full eps12 sweeps (`eps_dip_gallery.py`)
- 10 configs varying u=v, r, s1,s2; each plot has 3 N curves + predicted-dip line.
- Outputs: `plots/eps_sweep_prediction_dip/` (PNGs + CSVs + gallery_index.txt)
- N=60 skipped (memory: ~12.6 GB matrix).

### 4. 3x3 grid figure (`eps_dip_gallery_grid.py`)
- Combined 9 gallery CSVs into dip_curves-style panel figure.
- Per panel: points per N, parabola fit, dotted per-N minima, dashed prediction.
- Output: `plots/eps_sweep_prediction_dip/dip_grid.png`

### 5. Extended KL-dip vs mutation (`eps12_dip_vs_u_plot.py`)
- Merged original data_u.npz with new points.
- **Skipped u=1.0** (period-2 chain, no unique stationary — caused hour-long hang).
- Final merged data (`data_u_merged.npz`), 9 u values:

  | u=v    | pair (1e-3,-5e-4) fit | pred   | pair (-5e-4,-5e-4) fit | pred   |
  |--------|------------------------|--------|-------------------------|--------|
  | 1e-5   | -3.15e-4               | -3.33e-4 | +6.66e-4              | +6.67e-4 |
  | 1e-4   | -3.15e-4               | -3.33e-4 | +6.70e-4              | +6.67e-4 |
  | 1e-3   | -3.26e-4               | -3.33e-4 | +6.89e-4              | +6.67e-4 |
  | 1e-2   | -3.67e-4               | -3.33e-4 | +7.57e-4              | +6.67e-4 |
  | 0.1    | -4.22e-4               | -3.33e-4 | +8.49e-4              | +6.67e-4 |
  | 0.3    | -4.74e-4               | -3.33e-4 | +9.50e-4              | +6.67e-4 |

- Plot: `plots/eps12_dip_robustness/dip_vs_u_extended.png`

## Key observation to understand next (OPEN)

**As u=v grows, the two pairs drift in OPPOSITE directions from prediction:**
- pair with s1+s2 > 0 (pred negative): fitted dip moves **more negative**
  (magnitude grows: -3.33e-4 → -4.74e-4)
- pair with s1+s2 < 0 (pred positive): fitted dip moves **more positive**
  (magnitude grows: +6.67e-4 → +9.50e-4)

So the effective coefficient c(u) in eps* ≈ -c(u)(s1+s2) has c > 2/3 and
grows with u — but symmetrically in sign. Likely tied to how neutral phi*
reshapes (Dirichlet exponent 2Nu) as mutation strengthens: Var(x11) and
Cov(B,x11) in the -2/3 formula are computed at the *neutral* distribution,
which itself changes with u. Need to derive or numerically extract c(u) and
explain the sign/magnitude of the drift.

## Next moves (user-requested)

1. **Understand dip_vs_u_extended plot:** why divergence is in opposite
   direction for the two (s1,s2) pairs (same underlying mechanism?).
   - Extract neutral Var(x11), Cov at each u
   - Fit c(u) where eps* ≈ -c(u)(s1+s2)
   - Compare to theory correction for finite 2Nu

2. **Break u=v symmetry:** sweep u != v (e.g. fix v, vary u; or u/v ratio).
   - Prediction will change: -2/3(s1+s2) was derived for symmetric u=v
   - Need asymmetric-mutation correction to eps12*
   - Reuse dip-fitting machinery; start with small asymmetry (u=2v, etc.)

## Files created today
- `eps12_dip_prediction_test.py` — 6-pair vertex test + caching
- `eps12_dip_robustness.py` — r and u robustness (original)
- `eps_dip_gallery.py` — 10 full sweep plots
- `eps_dip_gallery_grid.py` — 3x3 combined figure from CSVs
- `eps12_dip_vs_u_extended.py` — abandoned (no incremental save; u=1 hang)
- `eps12_dip_vs_u_plot.py` — merge/replot/extend u sweep (USE THIS)

## How to run
```bash
# replot dip test from cache
python 2loci/eps12_dip_prediction_test.py --replot

# replot extended u sweep
python 2loci/eps12_dip_vs_u_plot.py --replot-only

# add another u point
python 2loci/eps12_dip_vs_u_plot.py --add-u 3e-5

# rebuild 3x3 grid from gallery CSVs
python 2loci/eps_dip_gallery_grid.py
```

## Prior session context (2026-06-10)
See `SESSION_HANDOFF_2026-06-10.md` for diffusion-vs-chain validation work
(selection_diffusion_vs_chain, sigma sweeps, ploidy test, boundary mass).
