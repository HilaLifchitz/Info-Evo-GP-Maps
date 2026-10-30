# Session handoff - 2026-07-13 (node-hopping / SSWM view of the 2-locus chain)

Plain text, no LaTeX (chat was not rendering it).

## The big idea we are chasing

The two-locus chain evolves a DISTRIBUTION psi(t) over the whole simplex of
population compositions x=(x00,x01,x10,x11). Two readings of the SAME object:

  - Eulerian: psi(t) is a probability cloud over all populations,
    psi(t+1) = P psi(t)   (P = column-stochastic transition matrix, S states).
  - Lagrangian ("one grain of sand"): follow ONE population. When u<<s and
    Nu<<1 it is almost always monomorphic (sitting at a CORNER of the simplex)
    and occasionally makes a fast substitution ALONG AN EDGE to a neighbour
    corner. This is the SSWM adaptive walk.

Claim (now verified): the corner-hopping picture is an EXACT superposition of
single-population paths, because P is LINEAR: psi(0)=sum_k w_k delta_k implies
psi(t)=sum_k w_k P^t delta_k. The nonlinearity (selection normalisation) lives
INSIDE a population; the law over populations is linear (master equation).

## Model recap (code in ../2loci/simplex.py)

State = integer count vector c=(c00,c01,c10,c11), sum=N, S=C(N+3,3) states.
log-fitness of genotype g = g^T Sigma g, with
  logfit = [00:0, 01:s2, 10:s1, 11:s1+s2+eps12],  Sigma=[[s1,eps12/2],[eps12/2,s2]].
One generation: selection -> recombination (LD D-shift, rate r) -> mutation
(u:0->1, v:1->0 per locus) -> multinomial drift. phi = neutral (Sigma=0)
stationary distribution.

Definitions used:
  - vertex "11" = single state c=(0,0,0,N), x=(0,0,0,1).
  - basin B_g   = { c : argmax_g' c_g' = g }  (set: vertex + near-monomorphic ring).
    The 4 basins partition all S states.
  - basin mass  = sum_{i in B_g} psi_i  (NOT the single vertex value).
  - MFPT(00->11) = expected generations to FIRST ENTER set B11 from c=(N,0,0,0);
    solves (I - P_row[T,T]) tau_T = 1, T = states not in B11.
  - free fitness F_N(psi) = <W>_psi + H(psi)/N, W = log mean fitness - mutation term
    (simplex.free_fitness_1 / w_potential_per_state).
  - lumped 4-node jump matrix Q[g,h]: put phi restricted to basin g, apply P once,
    read mass landing in basin h. Per-generation corner->corner hop probability.
  - cycle affinity (circulation) A = ln[ Q(00->10)Q(10->11)Q(11->01)Q(01->00)
                                        / Q(00->01)Q(01->11)Q(11->10)Q(10->00) ].
    A=0 <=> reversible loop (Kolmogorov). A!=0 = net circulation, sign=rotation.
  - detailed-balance residual: F[j,i]=phi_i P(i->j); rel asym = max|F-F^T|/max off-diag.

## Scripts written (all in equilibria_vs_dynamics/)

1. node_jump_analysis.py
   - Verifies corner concentration (support-size decomposition of psi(t)).
   - Basin flow from phi and from each pure corner (adaptive walk).
   - Lumped Q vs analytic SSWM rate Nu*pi_fix.
   - Detailed-balance test, r=0 vs r>0.
   - 3 landscapes: additive winner / sign epistasis / reciprocal-sign valley.

2. compare_two_cases.py
   - Controlled A-vs-B comparison (see below). Adds MFPT, free-fitness(t),
     cycle affinity, node-jump graphs.

3. compare_sweep.py
   - Robustness of A-vs-B across a=s1 in {0.05,0.1,0.15,0.2} and u in {1e-4,2e-4,1e-3}.
     Scalar metrics only (fast, no time integration).

4. basin_flow_u_r.py
   - basin-from-phi flow swept over u=v; monomorphism-vs-u curve; then swept over r.

Default regime: N=24, u=v=2e-4 (Nu=4.8e-3), r=0 unless noted.

## Key findings

CONCENTRATION ON NODES (node_jump_analysis)
  - phi corner (monomorphic) mass = 0.930 at u=2e-4; edges 0.069; interior 5e-6.
  - Through the WHOLE transient the monomorphic mass stays ~0.93-0.96. Simplex
    behaves as a 4-node graph (corners + edges only).

MONOMORPHISM vs MUTATION RATE (basin_flow_u_r)
  - Controlled by theta = 2Nu. corner mass: 0.982 (theta=0.002) -> 0.930 (0.01)
    -> 0.703 (0.048) -> 0.204 (0.24) -> ~0 (theta>=1, interior fills).
  - Low u: crisp SSWM node-hopping. High u: cloud smears to edges/faces/interior;
    equilibrium 11-basin also degrades (0.97 -> 0.71 as u: 1e-4 -> 3e-2).

NODE-JUMP MATRIX & EPISTATIC ROUTING (node_jump_analysis)
  - sign epistasis: adaptation CHANNELLED via 10 (Q(00->10) > Q(00->01)).
  - reciprocal sign (valley): 00 is metastable ~10^3 gen, then valley crossing to
    11; direct 00->11 one-step ~4.8e-5 (2-mutational, ~30x rarer).
  - Empirical Q matches SSWM ORDERING; magnitudes ~4x larger at finite Nu
    (standing polymorphism inflates one-step flux); lowering u sharpens agreement.

A-vs-B CONTROLLED COMPARISON (compare_two_cases; a=0.1)
  Case A (sign):     s1=a, s2=-a, eps=2a -> logfit [0,-0.1,+0.1,+0.2], additive part 0.
  Case B (additive): s1=a, s2=a,  eps=0  -> logfit [0,+0.1,+0.1,+0.2].
  Same base (00=0) and same optimum (11=+0.2). Results:
  - Equilibria nearly identical: psi* on 11 = 98.9% (A) vs 97.8% (B);
    KL_eq = 1.95 (A) vs 1.85 bits (B); F_N(psi*) = 0.211 (A) vs 0.215 (B).
  - Speed DIFFERS: MFPT(00->11) = 2221 gen (A) vs 1673 (B), ~33% slower for A.
    Reason: B has TWO uphill first steps out of 00 (supply ~4e-3/gen); A has ONE
    (00->10), the other (00->01) drops into the deleterious valley genotype.
  - Intermediates: B uses 01 and 10 symmetrically (overlap, peak ~0.25);
    A funnels all flux through 10 (peak ~0.37), 01 unused.
  - Free fitness overshoots its equilibrium value by ~0.004 then relaxes = a
    fingerprint of NON-reversibility (a true gradient flow would be monotone).
  - Circulation: cycle affinity A = -0.0145 (Case A) vs +0.0000 (Case B, forced by
    s1<->s2 mirror symmetry). Sign epistasis imprints a directed loop current.

ROBUSTNESS (compare_sweep) - qualitatively identical across all a and u:
  - MFPT ratio A/B ~ 1.25-1.33 (nearly constant; absolute MFPT scales ~1/u).
  - |affinity| grows ~a^3 for A, ~0 for B, and is INDEPENDENT of u (ratio cancels u).
  - KL_eq: A >= B, converge to ~2 bits (log2 4 ceiling) as a grows.
  - Holds even at u=1e-3 where corners hold only 70% (outside strict SSWM).

DETAILED BALANCE / RECOMBINATION
  - Chain is NOT strictly reversible: raw relative flux asymmetry ~13-17%,
    intrinsic to WF multinomial drift + per-locus mutation (NOT recombination:
    r=0 vs r=0.5 both ~0.150).
  - r has NO visible effect on the node flow in the SSWM regime, because
    near-monomorphic populations carry no standing LD (mean LD of phi ~1e-13 ~0),
    and edge states are single-locus polymorphisms (LD=0 by construction).
    Recombination only grips when BOTH loci segregate at once.

## Where the plots live (results/)

- node_jumps_sswm_20260713/  : concentration.png, basin_from_phi.png,
    walk_{additive,sign,valley}.png, jumps_{additive,sign,valley}.png, summary.txt
- compare_cases_20260713/     : adaptation_AB.png (KL, mean logfit, free fitness),
    basin_from_00_AB.png, basin_from_phi_AB.png, concentration_AB.png,
    jumps_A.png, jumps_B.png, summary.txt
- compare_sweep_20260713/     : sweep_summary.png, sweep.csv
- basin_flow_u_r_20260713/    : monomorphism_vs_u.png, basin_flow_u_sweep.png,
    basin_flow_r_sweep_additive.png, basin_flow_r_sweep_sign.png
- run logs: results/*_run.log

## Open next steps (candidates for tomorrow)

1. Decompose F_N time-derivative into reversible-descent + circulating parts, to
   SEE the non-gradient contribution behind the free-fitness overshoot.
2. Recombination WHERE IT MATTERS: rerun r-sweep at higher u (e.g. 5e-3) and on
   the reciprocal-sign VALLEY landscape (recombination of 01+10 -> 11 shortcut).
3. Larger N: confirm MFPT ratio (~1.3) and affinity are N-stable.
4. Push u lower to lock the empirical Q -> Nu*pi_fix SSWM calibration.
5. Optional: strict-vertex target for MFPT (instead of basin B11) to compare.
6. Tie back to project's core theme: same equilibrium score, different adaptation
   speed, controlled by epistatic ACCESSIBILITY (not the equilibrium).
