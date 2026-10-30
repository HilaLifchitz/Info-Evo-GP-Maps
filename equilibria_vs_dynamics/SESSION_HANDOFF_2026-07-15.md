# Session hand-off — 2026-07-15

Four-landscape comparison (2-loci Wright–Fisher, simplex master equation), all
sharing the **same endpoints** `logfit(00)=0`, `logfit(11)=2a`, differing only in
the two single-mutant intermediates. Focus: free fitness, information, and
first-passage dynamics as functions of selection strength `a`, mutation `u=v`, at
`N=30`.

## The four cases

`Σ = [[s1, ε12/2],[ε12/2, s2]]`, log-fitness `gᵀΣg = [0, s2, s1, s1+s2+ε12]`.

| case | Σ | (s1, s2, ε12) | logfit [00,01,10,11] | intermediates | colour |
|------|---|---------------|----------------------|---------------|--------|
| **A** sign epistasis        | `[[a, a],[a,-a]]`   | `(a, -a, 2a)`  | `[0,-a,+a,2a]` | one up, one down | red |
| **B** additive              | `[[a, 0],[0, a]]`   | `(a,  a, 0)`   | `[0,+a,+a,2a]` | both up          | blue |
| **C** reciprocal sign       | `[[-a,2a],[2a,-a]]` | `(-a,-a,4a)`   | `[0,-a,-a,2a]` | both down (valley) | green |
| **D** neutral intermediates | `[[0, a],[a, 0]]`   | `(0, 0, 2a)`   | `[0, 0, 0,2a]` | both flat         | purple |

## Free-fitness definitions used (the important ones)

- **F5 = "free fitness"** (the one we plot; monotone by construction because it is
  measured against the *true* chain stationary `ψ*`, and `D_KL(ψ(t)‖ψ*)` is a
  Lyapunov functional):
  ```
  F(ψ) = F* − (1/N) D_KL(ψ ‖ ψ*)
  F*   = <log W>_{ψ*} − (1/N) D_KL(ψ* ‖ φ)          # logB(α) EXCLUDED by default
  ```
  `<log W> = Σ_states ψ · log(Σ_g x_g ω_g)`; `φ` = neutral stationary.
  The Dirichlet-normaliser term `logB(α)` was dropped from `F*` because its
  cross-landscape variation is a nuisance that flipped the intuitive A-vs-B order;
  toggle with `free_fitness5_asymptote(..., include_logB=True)`.
- `F_N`, `F_boltz` (cond-α Gibbs) still exist in `compare_two_cases.py` but are NOT
  the headline metric: they measure divergence to an *approximate* equilibrium and
  therefore overshoot (non-monotone). See prior hand-off / transcript.
- **Information** `I = D_KL(ψ* ‖ φ)` [bits] — how sharply the equilibrium peaks vs
  neutral. This is the quantity driving the free-fitness differences.

## Key findings (N=30)

1. **Free-fitness ordering is `B > A > D > C` everywhere** (additive best, reciprocal
   worst). Since `<logW>` is nearly matched across cases, the ordering is set by the
   **information cost**: more valley-like intermediates ⇒ sharper equilibrium ⇒ more
   information ⇒ lower free fitness.
2. **Information ordering `C > D > A > B`** for `u ≲ 0.01` (flips to `C > A > D > B`
   at very high `u`). **C (reciprocal) always has the most information** — confirmed
   from `u=1e-4` up to `u=0.1`. C's info peaks near `u≈0.01` (~2.65 bits) then falls
   as mutation swamps selection (corner mass → 0).
3. **Gap knobs act oppositely**:
   - Increasing **`u`** *widens* the gaps (equilibrium stays polymorphic → path
     structure matters more). E.g. C's free-fitness shortfall from B grows to
     ~ −0.018 nats by `u=2e-3`.
   - Increasing **`a`** *closes* the gaps (all pile onto the same 11 corner; by
     `a ≳ 0.15` the four nearly coincide).
   - A and D swap: D has more information than A, but A has higher free fitness
     (its beneficial `10` intermediate lifts `<logW>`), so it's a genuine
     `<logW> − info/N` trade-off, not pure concentration.
4. **MFPT to the 11 basin (from pure 00), `u=2e-3`**: B fastest, and A/B/D all
   *speed up* with `a`. **C is the exception — its MFPT GROWS with `a`** (deeper
   valley to cross): 620 → 1,599 gen over `a=0.02→0.3`, ~22× slower than B at
   `a=0.3`. Classic reciprocal-sign valley-crossing signature.

## Scripts (all in `equilibria_vs_dynamics/`, 4-case aware)

- `adaptation_per_a.py` — 3-panel adaptation curves (information, mean log-fitness,
  free fitness) with 4 case-lines, **one row per `a`**. Defines the shared
  `CASE_DEFS / CASE_COLORS / CASE_ORDER / all_landscapes(a)` + `curves_for`.
  Run: `python adaptation_per_a.py --N 30 --u 2e-4`
- `adaptation_per_u.py` — same, **one row per `u`** (fixed `a`), imports the 4-case
  helpers. `--tmax` overrides the per-row `~4/u` horizon.
  Run: `python adaptation_per_u.py --a 0.05 --N 30`
- `summary_4cases.py` — equilibrium summary (fast, stationary only): information and
  free-fitness-relative-to-B, each vs `a` and vs `u`, 4 lines.
- `mfpt_4cases.py` — MFPT to the 11 basin (linear solve), 4 cases, vs `a` at fixed
  high `u` (from pure 00 and from φ).
- `info_compare.py` — earlier 2-case (A vs B) info comparison vs `a` and `u`.
- `compare_two_cases.py` — original A/B time-integration + all the free-fitness
  machinery (`free_fitness5_asymptote/_series`, `w_potential_mu/_condalpha`,
  `boltz_reference`, `mfpt_to_basin`, `cycle_affinity`).

## Output figures (results/…_20260715/)

- `adaptation_per_a_20260715/` — `adaptation_per_a_N30.png` (8 rows) + standalone
  `adaptation_a{…}_N30.png` per `a`. **Use the standalone ones**: the combined
  figure is very tall and looks compressed when downscaled.
- `adaptation_per_u_20260715/` — `adaptation_per_u_a0.05_N30.png` (5 rows, `u`
  1e-4…2e-3) + standalone per `u`.
- `adaptation_bigu_20260715/` — big-`u` rows (0.01, 0.05, 0.1), `tmax=10000`, all
  fully converged.
- `adaptation_longu5e4_20260715/` — single `u=5e-4` row at `tmax=60000` (checking
  whether C's slow curves finally plateau at their dotted `F*`). **This run was
  still finishing at hand-off time — verify the file exists tomorrow.**
- `summary_4cases_20260715/summary_4cases_N30.png` — 2×2 equilibrium summary.
- `mfpt_4cases_20260715/mfpt_4cases_N30_u0.002.png` + `mfpt.csv`.

## Open items / possible next steps

- Confirm the long `u=5e-4, tmax=60000` run finished and inspect whether C
  converges (if not, push tmax higher for C specifically).
- MFPT as a function of `u` at fixed `a` (mirror of the `a`-sweep).
- Optionally collapse gaps onto the effective strength `2N·a` to test a master
  curve.
- Re-anchor / reinstate `logB(α)` if a principled normaliser is wanted.
