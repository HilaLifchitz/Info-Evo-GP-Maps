# Gradient Flows via JKO Scheme

This repository implements numerical solvers for gradient flows using the **Jordan–Kinderlehrer–Otto (JKO) scheme**, a time-discretization method for Wasserstein gradient flows. The ultimate goal is to develop robust JKO solvers that work in complex geometries (non-Euclidean, variable diffusion) for applications in machine learning and PDE numerics.

## Overview

### What We're Trying to Do

The JKO scheme is a **minimizing movement** approach to solving gradient flow PDEs. Instead of discretizing the PDE directly, we solve a sequence of optimization problems:

$$
\rho^{k+1} \in \arg\min_{\rho} \left[ F(\rho) + \frac{1}{2\tau} W_2^2(\rho, \rho^k) \right]
$$

where:
- $F(\rho)$ is a functional (e.g., entropy $F(\rho) = \int \rho \log \rho \, dx$)
- $W_2$ is the 2-Wasserstein distance (optimal transport metric)
- $\tau > 0$ is the time step

For the **heat equation** $\partial_t \rho = \Delta \rho$ with Neumann (reflecting) boundary conditions, this JKO scheme is theoretically equivalent to implicit Euler, making it a perfect test case.

### Why This Matters

1. **1D is "easy"**: In 1D, $W_2$ can be computed via quantile functions, making JKO straightforward.
2. **2D+ is hard**: In higher dimensions, computing $W_2$ becomes the bottleneck. We use **dynamic optimal transport** (Benamou–Brenier formulation) to reformulate the JKO step as a convex optimization problem.
3. **Future goal**: Extend to non-Euclidean geometries and variable diffusion (Fokker–Planck with space-dependent coefficients), where JKO is more natural than traditional PDE discretizations.

---

## Files and Implementations

### 1D Heat Equation Solvers

#### `1D_HeatEqation_final_version.py` ⭐ **Recommended**

The most robust 1D implementation. Features:
- **Implicit Euler** PDE solver (sanity check)
- **"True JKO"** solver using quantile coordinates with **log-gap parameterization** (softmax)
- Avoids boundary artifacts by removing explicit monotonicity constraints
- Command-line arguments for all parameters
- Saves comparison plots with descriptive filenames

**Usage:**
```bash
python 1D_HeatEqation_final_version.py --alpha 0.6 --beta 0.8 --sigma1 0.01 --sigma2 0.2 --tau 0.0002 --n-cells 400 --steps 480
```

**Key parameters:**
- `--alpha`, `--beta`: Weights for initial Gaussian bumps
- `--sigma1`, `--sigma2`: Standard deviations of initial bumps
- `--tau`: Time step
- `--n-cells`: Spatial grid resolution
- `--steps`: Number of time steps
- `--n-quantiles`: Number of quantile points for JKO (default: 400)
- `--jko-max-iter`: Max iterations for JKO inner solver (default: 200)
- `--jko-tol`: Convergence tolerance for JKO (default: 1e-10)

**Outputs:**
- `plot_euler_alpha{alpha}_beta{beta}.png`: Euler solution snapshots
- `plot_jko_alpha{alpha}_beta{beta}.png`: JKO solution snapshots
- `alpha{alpha}_beta{beta}_sigma1_{sigma1}_sigma2_{sigma2}_tau{tau}_n{n}.png`: Side-by-side comparison

#### `1D_b.py`

An earlier cleaned-up version with:
- Newton-based JKO solver (with projection onto monotone set)
- More robust mass handling
- Diagnostics (mass error, entropy, mean)

#### `1D_a.py`

Initial implementation (likely superseded by later versions).

---

### 2D Heat Equation Solver

#### `2D_HeatEquation.py` ⭐ **Main 2D Implementation**

Implements the 2D heat equation on a square $[0,1] \times [0,1]$ with Neumann boundary conditions using:

1. **PDE sanity check**: Implicit Euler (ADI split or full unsplit)
2. **JKO via dynamic OT**: Benamou–Brenier formulation solved with Chambolle–Pock primal–dual algorithm

**Why dynamic OT?**
In 2D, we can't use the quantile trick. Instead, we reformulate $W_2^2(\rho_0, \rho_1)$ as:

$$
W_2^2(\rho_0, \rho_1) = \min_{\rho_t, m_t} \int_0^1 \int \frac{|m_t|^2}{\rho_t} \, dx \, dt
$$

subject to the continuity equation $\partial_t \rho_t + \nabla \cdot m_t = 0$ with $\rho(0) = \rho_0$, $\rho(1) = \rho_1$.

This gives us a **convex optimization problem** over space-time that we solve with primal–dual methods.

**Usage:**
```bash
python 2D_HeatEquation.py --n 48 --tau 5e-4 --steps 30 --euler-solver full --plot-common-scale
```

**Key parameters:**
- `--n`: Grid size (n×n cells)
- `--tau`: Outer time step
- `--steps`: Number of outer time steps
- `--euler-solver`: `adi` (default) or `full` (unsplit implicit Euler)
- `--transport-slices`: Inner OT time discretization (default: 8, try 16+ for better accuracy)
- `--pd-iters`: Max primal–dual iterations per JKO step (default: 800, try 4000+ for convergence)
- `--pd-primal-step`, `--pd-dual-step`: CP step sizes (default: 0.01, may need tuning)
- `--pd-verbose`: Print inner residual progress
- `--pd-debug-objective`: Print JKO objective (entropy + action) during inner solve
- `--plot-common-scale`: Use same color scale for Euler and JKO plots

**Outputs:**
- `compare2d_t{time}_n{n}.png`: Side-by-side comparison at each snapshot

**Current status:**
- ✅ Adjoint operators are correctly matched (adjoint test passes)
- ✅ No entropy collapse (JKO produces valid probability distributions)
- ⚠️ JKO diffusion is slower than Euler (likely under-convergence of inner solver)
- 🔧 Work in progress: tuning inner solver accuracy and step sizes

---

### 3D Simplex (Shahshahani Metric)

#### `3D_Simplex_Shahshahani.py` ⭐ **New (Starter Implementation)**

Implements diffusion on the **3D simplex** (4 components) using the **Shahshahani metric**, which corresponds to the JKO scheme for entropy under the Fisher/Shahshahani geometry. This script provides:

- 3D simplex grid discretization via a masked Cartesian cube
- Shahshahani diffusion tensor $B(x)$ for the 3D simplex
- **Implicit Euler** step (CG solve) as the JKO-equivalent update
- **Explicit Euler** step for sanity checks
- 2D marginal visualization (integrate over $x_3$)

**Usage:**
```bash
python 3D_Simplex_Shahshahani.py --n 32 --tau 1e-4 --steps 30 --scheme implicit --verbose
```

**Key parameters:**
- `--n`: Grid resolution per axis (masked to the simplex)
- `--tau`: Time step
- `--ne`: Effective population size (scales $B(x)$)
- `--scheme`: `implicit` (JKO-equivalent) or `explicit`
- `--cg-tol`, `--cg-max-iter`: Linear solver settings

---

## Mathematical Background

### The Heat Equation as a Gradient Flow

The 1D/2D heat equation:

$$
\partial_t \rho = \Delta \rho
$$

is the Wasserstein gradient flow of the **entropy functional**:

$$
F(\rho) = \int \rho(x) \log \rho(x) \, dx
$$

The first variation is $\delta F / \delta \rho = \log \rho + 1$, and the gradient flow in $W_2$ is:

$$
\partial_t \rho = \nabla \cdot \left( \rho \nabla \frac{\delta F}{\delta \rho} \right) = \nabla \cdot (\rho \nabla (\log \rho + 1)) = \Delta \rho
$$

### JKO Scheme (Minimizing Movement)

Given $\rho^k$ and time step $\tau$, the next iterate is:

$$
\rho^{k+1} \in \arg\min_{\rho \geq 0, \int \rho = 1} \left[ F(\rho) + \frac{1}{2\tau} W_2^2(\rho, \rho^k) \right]
$$

### 1D: Quantile Formulation

In 1D, we can rewrite everything in terms of the **quantile function** $Q(u) = F^{-1}(u)$:

- **Entropy**: $\int \rho \log \rho \, dx = -\int_0^1 \log Q'(u) \, du$ (up to constants)
- **Wasserstein**: $W_2^2(\rho, \rho^k) = \int_0^1 |Q(u) - Q^k(u)|^2 \, du$

This reduces the JKO step to a **convex optimization problem over monotone increasing functions** $Q(u)$.

### 2D: Dynamic OT Formulation

In 2D+, we use the **Benamou–Brenier** reformulation:

$$
W_2^2(\rho_0, \rho_1) = \min_{\rho_t, m_t} \int_0^1 \int \frac{|m_t(x)|^2}{\rho_t(x)} \, dx \, dt
$$

subject to:
- Continuity: $\partial_t \rho_t + \nabla \cdot m_t = 0$
- Boundary: $\rho(0) = \rho_0$, $\rho(1) = \rho_1$
- No-flux BCs: $m_t \cdot n = 0$ on boundaries

This is a **convex problem** that we solve with **Chambolle–Pock** (primal–dual) algorithm.

---

## Dependencies

- `numpy`: Numerical arrays and linear algebra
- `matplotlib`: Plotting (with `Agg` backend for headless operation)

Install with:
```bash
pip install numpy matplotlib
```

---

## Running Examples

### 1D: Quick Test
```bash
python 1D_HeatEqation_final_version.py --steps 100 --snapshot-every 20
```

### 1D: High Resolution
```bash
python 1D_HeatEqation_final_version.py --n-cells 800 --steps 1000 --tau 1e-5 --jko-max-iter 500
```

### 2D: Basic Run
```bash
python 2D_HeatEquation.py --n 32 --steps 20 --snapshot-every 5
```

### 2D: High Accuracy (Slow)
```bash
python 2D_HeatEquation.py --n 48 --transport-slices 32 --pd-iters 8000 \
  --pd-primal-step 0.002 --pd-dual-step 0.002 --pd-verbose --pd-debug-objective
```

---

## Diagnostics and Debugging

### What to Look For

1. **Mass conservation**: `sum` should be ≈ 1.0 at all times
2. **Entropy**: Should decrease over time (heat flow increases disorder)
3. **Max density**: Should decrease as diffusion spreads the mass
4. **Comparison plots**: Euler and JKO should be visually similar (they're theoretically equivalent)

### Common Issues

- **JKO entropy collapses to 0**: Inner solver not converging → increase `pd-iters`, check step sizes
- **JKO diffuses too slowly**: Inner solver under-converged → increase `transport-slices` and `pd-iters`
- **Adjoint error large**: Operator mismatch → check `[adjoint] relative_error` (should be ~1e-12)
- **Objective J increases**: Primal–dual not minimizing correctly → tune step sizes, check scaling

### Debug Flags

- `--pd-verbose`: Print continuity residual every `--pd-check-every` iterations
- `--pd-debug-objective`: Print entropy (E), action (A), and total objective (J) during inner solve
- `--plot-common-scale`: Use same color scale for fair visual comparison

---

## Adaptive landscape sketches (`AdptiveLandscapeSimulator/`)

Not JKO — static fitness geometry on the genotype simplex, used for intuition.

| File | Purpose |
|------|---------|
| `AdptiveLandscapeSimulator/ExAdaptiveLandscape.py` | $n$ loci, pairwise epistasis, slice plots of $\omega(g)$ |
| `AdptiveLandscapeSimulator/ExAdaptiveLandscape2Loci.py` | Two loci with LD coordinate $D$; genotype frequencies from $(f_1,f_2,D)$ |
| `3D_Simplex_Shahshahani.py` | Diffusion / JKO-style flow on the 3-simplex with the Shahshahani metric |

Config sketch: `my.yaml`.

---

## Future Work

1. **Non-Euclidean geometries**: Extend to manifolds (geodesic cost instead of Euclidean)
2. **Variable diffusion**: Fokker–Planck with space-dependent coefficients $D(x)$
3. **Better inner solvers**: Adaptive step sizes, diagonal preconditioning for primal–dual
4. **3D support**: Scale dynamic OT to 3D grids
5. **Sinkhorn alternative**: Implement entropic OT (Sinkhorn) as an alternative to dynamic OT

---

## References

- **JKO scheme**: Jordan, Kinderlehrer, Otto (1998). "The Variational Formulation of the Fokker–Planck Equation"
- **Dynamic OT**: Benamou, Brenier (2000). "A Computational Fluid Mechanics Solution to the Monge–Kantorovich Mass Transfer Problem"
- **Primal–Dual**: Chambolle, Pock (2011). "A First-Order Primal–Dual Algorithm for Convex Problems with Applications to Imaging"

---

## Notes

- The 1D implementations are **production-ready** and match Euler well.
- The 2D implementation is **experimental** and may require parameter tuning for convergence.
- All code uses **cell-centered grids** and **mass variables** (not density) for numerical stability.
