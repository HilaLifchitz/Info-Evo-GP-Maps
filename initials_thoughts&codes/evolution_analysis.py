#!/usr/bin/env python3
"""
Simple evolution analysis from neutral to selection using the existing Markov chain code.

This module provides standalone functions to analyze the evolution from a neutral 
steady state to a selection steady state, tracking D(t), I(t), and rest(t).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Markov import WrightFisherMarkovChain, compute_stationary_quiet

# Set professional seaborn style
sns.set_style("white")
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.0)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9


def evolve_from_neutral_to_selection(N, mu, s_target=0.0005, max_generations=10000, 
                                   convergence_tol=1e-8, save_every=10):
    """
    Evolve from neutral steady state (phi*) to selection steady state (psi*).
    
    This function implements your analysis:
    1. Calculate phi* (steady state for s=0) 
    2. Use phi* as psi(0) (initial condition)
    3. Run Markov process with s=s_target until convergence to psi*
    4. Track D(t), I(t), and rest(t) at each time step
    
    Parameters:
    -----------
    N : int
        Population size
    mu : float
        Mutation rate
    s_target : float, optional (default=0.0005)
        Target selection coefficient
    max_generations : int, optional (default=10000)
        Maximum number of generations
    convergence_tol : float, optional (default=1e-8)
        Convergence tolerance
    save_every : int, optional (default=10)
        Save distribution every N generations
        
    Returns:
    --------
    dict
        Dictionary containing all results and trajectories
    """
    
    print("="*80)
    print("EVOLUTION FROM NEUTRAL TO SELECTION")
    print("="*80)
    print(f"Parameters: N={N}, μ={mu}")
    print(f"Neutral (s=0) → Selection (s={s_target})")
    print(f"Neutral (s=0) → Selection (s={s_target})")
    print("="*80)
    
    # Step 1: Calculate phi* (neutral steady state, s=0)
    print("\nStep 1: Calculating φ* (neutral steady state, s=0)")
    phi_star = compute_stationary_quiet(N=N, s=0.0, mu=mu, method='direct')
    print(f"  φ* calculated: mean freq = {np.sum(np.arange(N+1) * phi_star) / N:.4f}")
    
    # Step 2: Calculate psi* (selection steady state)
    print(f"\nStep 2: Calculating ψ* (selection steady state, s={s_target})")
    psi_star = compute_stationary_quiet(N=N, s=s_target, mu=mu, method='direct')
    print(f"  ψ* calculated: mean freq = {np.sum(np.arange(N+1) * psi_star) / N:.4f}")
    
    # Step 3: Create selection chain for evolution
    print(f"\nStep 3: Creating selection chain for evolution")
    selection_chain = WrightFisherMarkovChain(N=N, s=s_target, mu=mu, quiet=True)
    selection_chain.construct_transition_matrix()
    
    # Step 4: Evolve from phi* (psi(0)) to psi*
    print(f"\nStep 4: Evolving from φ* to ψ*")
    print(f"  Initial condition: ψ(0) = φ*")
    print(f"  Target: ψ* (s={s_target})")
    print(f"  Tracking: D(t), I(t), and rest(t) every {save_every} generations")
    
    # Initialize with phi* (neutral steady state)
    psi_t = phi_star.copy()
    
    # Storage for trajectory and quantities
    psi_trajectory = [psi_t.copy()]
    generations = [0]
    D_t = []  # KL divergence D(t) = KL(ψ(t) || φ*)
    I_t = []  # KL divergence I(t) = KL(ψ(t) || ψ*)
    rest_t = []  # Cross-term rest = ∫ₓ ψ(t)(x) * log(ψ*(x)/φ*(x))
    
    # Calculate initial quantities
    epsilon = 1e-12
    phi_star_safe = phi_star + epsilon
    psi_star_safe = psi_star + epsilon
    phi_star_safe = phi_star_safe / np.sum(phi_star_safe)
    psi_star_safe = psi_star_safe / np.sum(psi_star_safe)
    
    # Initial D(0), I(0), and rest(0)
    psi_t_safe = psi_t + epsilon
    psi_t_safe = psi_t_safe / np.sum(psi_t_safe)
    
    D_0 = np.sum(psi_t_safe * np.log2(psi_t_safe / phi_star_safe))
    I_0 = np.sum(psi_t_safe * np.log2(psi_t_safe / psi_star_safe))
    rest_0 = np.sum(psi_t_safe * np.log2(psi_star_safe / phi_star_safe))
    
    D_t.append(D_0)
    I_t.append(I_0)
    rest_t.append(rest_0)
    
    print(f"  Initial quantities:")
    print(f"    D(0) = {D_0:.6f} bits")
    print(f"    I(0) = {I_0:.6f} bits")
    print(f"    rest(0) = {rest_0:.6f} bits")
    print(f"    Verification D(0) = I(0) + rest(0): {D_0:.6f} = {I_0 + rest_0:.6f} ✓")
    
    # Evolution loop
    converged = False
    final_generation = 0
    
    for generation in range(1, max_generations + 1):
        # Apply transition matrix: psi(t+1) = P * psi(t)
        psi_new = selection_chain.P @ psi_t
        
        # Check convergence to psi*
        error = np.linalg.norm(psi_new - psi_star, ord=1)  # L1 distance to target
        
        if error < convergence_tol:
            converged = True
            final_generation = generation
            print(f"  ✓ Converged to ψ* at generation {generation}")
            print(f"    Final error: {error:.2e}")
            break
        
        # Update for next iteration
        psi_t = psi_new
        
        # Calculate and save quantities every save_every generations
        if generation % save_every == 0:
            psi_trajectory.append(psi_t.copy())
            generations.append(generation)
            
            # Calculate D(t), I(t), and rest(t)
            psi_t_safe = psi_t + epsilon
            psi_t_safe = psi_t_safe / np.sum(psi_t_safe)
            
            D_current = np.sum(psi_t_safe * np.log2(psi_t_safe / phi_star_safe))
            I_current = np.sum(psi_t_safe * np.log2(psi_t_safe / psi_star_safe))
            rest_current = np.sum(psi_t_safe * np.log2(psi_star_safe / phi_star_safe))
            
            D_t.append(D_current)
            I_t.append(I_current)
            rest_t.append(rest_current)
            
            if generation % (save_every * 10) == 0:
                mean_freq = np.sum(np.arange(N + 1) * psi_t) / N
                # print(f"    Generation {generation}: mean freq = {mean_freq:.4f}, error = {error:.2e}")
                # print(f"      D({generation}) = {D_current:.6f}, I({generation}) = {I_current:.6f}, rest({generation}) = {rest_current:.6f}")
                # print(f"      Verification: D = I + rest → {D_current:.6f} = {I_current + rest_current:.6f} ✓")
                # print(f"      rest(t) change from t=0: {rest_current - rest_t[0]:.6f}")
    
    # If not converged, set final_generation to the last iteration
    if not converged:
        final_generation = generation

    # Add final state if not already saved
    if generations[-1] != final_generation:
        psi_trajectory.append(psi_t.copy())
        generations.append(final_generation)
        
        # Calculate final quantities
        psi_t_safe = psi_t + epsilon
        psi_t_safe = psi_t_safe / np.sum(psi_t_safe)
        
        D_final = np.sum(psi_t_safe * np.log2(psi_t_safe / phi_star_safe))
        I_final = np.sum(psi_t_safe * np.log2(psi_t_safe / psi_star_safe))
        rest_final = np.sum(psi_t_safe * np.log2(psi_star_safe / phi_star_safe))
        
        D_t.append(D_final)
        I_t.append(I_final)
        rest_t.append(rest_final)
    
    if not converged:
        print(f"  ⚠ Warning: Did not converge within {max_generations} generations")
        print(f"    Final error: {error:.2e}")
    
    # Final summary
    print(f"\nStep 5: Final Results")
    print(f"  Final quantities:")
    print(f"    D({final_generation}) = {D_t[-1]:.6f} bits")
    print(f"    I({final_generation}) = {I_t[-1]:.6f} bits")
    print(f"    rest({final_generation}) = {rest_t[-1]:.6f} bits")
    print(f"    Verification D = I + rest: {D_t[-1]:.6f} = {I_t[-1] + rest_t[-1]:.6f} ✓")
    
    # Calculate and print D_KL(ψ* || φ*)
    D_kl_psi_phi = np.sum(psi_star_safe * np.log2(psi_star_safe / phi_star_safe))
    print(f"\n  KL Divergence between steady states:")
    print(f"    D_KL(ψ* || φ*) = {D_kl_psi_phi:.6f} bits")
    print(f"    This should equal rest(t) = {rest_t[-1]:.6f} bits")
    print(f"    Difference: {abs(D_kl_psi_phi - rest_t[-1]):.2e}")
    
    # Prepare results
    results = {
        'phi_star': phi_star,
        'psi_star': psi_star,
        'psi_trajectory': psi_trajectory,
        'generations': generations,
        'D_t': D_t,
        'I_t': I_t,
        'rest_t': rest_t,
        'convergence_info': {
            'converged': converged,
            'final_generation': final_generation,
            'final_error': error if 'error' in locals() else None,
            's_target': s_target
        }
    }
    
    return results


def plot_evolution_analysis(evolution_results, N, mu=None):
    """
    Plot comprehensive analysis of the evolution from neutral to selection.
    Creates two figures: (1) distributions + mean frequency, (2) KL and components (2x2).
    
    Parameters:
    -----------
    evolution_results : dict
        Results from evolve_from_neutral_to_selection()
    N : int
        Population size
    mu : float, optional
        Mutation rate (for title display)
    """
    
    phi_star = evolution_results['phi_star']
    psi_star = evolution_results['psi_star']
    psi_trajectory = evolution_results['psi_trajectory']
    generations = evolution_results['generations']
    D_t = evolution_results['D_t']
    I_t = evolution_results['I_t']
    rest_t = evolution_results['rest_t']
    conv_info = evolution_results['convergence_info']
    
    # Calculate D_KL(ψ* || φ*) for the title
    epsilon = 1e-12
    phi_star_safe = phi_star + epsilon
    psi_star_safe = psi_star + epsilon
    phi_star_safe = phi_star_safe / np.sum(phi_star_safe)
    psi_star_safe = psi_star_safe / np.sum(psi_star_safe)
    D_kl_psi_phi = np.sum(psi_star_safe * np.log2(psi_star_safe / phi_star_safe))
    
    states = np.arange(N + 1)
    colors = sns.color_palette("husl", 3)
    
    # Calculate average of phi* and psi* means for vertical line
    phi_mean = np.sum(states * phi_star) / N
    psi_mean = np.sum(states * psi_star) / N
    mean_average = (phi_mean + psi_mean) / 2
    
    # =============================================================================
    # FIGURE 1: Distributions and Mean Frequency (2x1 layout)
    # =============================================================================
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)

    # Plot 1 (top): Compare phi*, psi*, and final psi(t) with average line
    # Build DataFrame for seaborn barplot (two distributions side-by-side)
    df_dist = pd.DataFrame({
        'state': np.concatenate([states, states]),
        'probability': np.concatenate([phi_star, psi_star]),
        'distribution': np.concatenate([np.repeat('φ* (neutral)', N + 1), np.repeat('ψ* (selection)', N + 1)])
    })
    sns.barplot(data=df_dist, x='state', y='probability', hue='distribution', ax=ax1, dodge=True)

    # Overlay final evolved ψ(t) as a line
    if len(psi_trajectory) > 0:
        final_psi = psi_trajectory[-1]
        df_final = pd.DataFrame({'state': states, 'probability': final_psi})
        sns.lineplot(data=df_final, x='state', y='probability', ax=ax1,
                     marker='o', linewidth=1.8, color=colors[2], dashes=False,
                     markerfacecolor=colors[2], markeredgecolor=colors[2],
                     errorbar=None, zorder=3,
                     label=f'ψ({conv_info["final_generation"]}) (evolved)')

    # No mean-average line as requested

    ax1.set_xlabel('Number of "1" alleles')
    ax1.set_ylabel('Probability')
    title_params = f'N={N}, s={conv_info["s_target"]}'
    if mu is not None:
        title_params += f', μ={mu}'
    ax1.set_title(f'Steady State Distributions Comparison\n{title_params}', fontsize=9)
    ax1.set_xticks([0, N])  # Only boundary ticks
    ax1.legend(frameon=True, fancybox=True, shadow=False)
    sns.despine(ax=ax1)

    # Plot 2 (bottom): Evolution of mean frequency over time
    if len(psi_trajectory) > 1:
        mean_frequencies = []
        for psi_t in psi_trajectory:
            mean_freq = np.sum(states * psi_t) / N
            mean_frequencies.append(mean_freq)

        ax2.plot(generations, mean_frequencies, '*--', color=colors[2], linewidth=2.5, markersize=4)
        ax2.axhline(phi_mean, color=colors[0], linestyle='--', alpha=0.7, linewidth=2,
                    label=f'φ* mean: {phi_mean:.3f}')
        ax2.axhline(psi_mean, color=colors[1], linestyle='--', alpha=0.7, linewidth=2,
                    label=f'ψ* mean: {psi_mean:.3f}')

        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Mean frequency of "1" allele')
        ax2.set_title('Evolution of Mean Frequency', fontsize=9)
        ax2.legend(frameon=True, fancybox=True, shadow=False)
        sns.despine(ax=ax2)

    fig1.show()

    # =============================================================================
    # FIGURE 2: KL + Components (2x2 layout)
    # =============================================================================
    fig2, ((ax3, ax4), (ax5, ax6)) = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # Left: D(t) and I(t)
    sns.lineplot(x=generations, y=D_t, ax=ax3,
                #  marker='.', linewidth=2.0, color=colors[0], dashes=False,
                #  errorbar=None,
                label='D(t) = KL(ψ(t) || φ*)', markersize=2)
    sns.lineplot(x=generations, y=I_t, ax=ax3,
                #  marker='x', linewidth=2.0, color=colors[1], dashes=False,
                #  errorbar=None,
                  label='I(t) = KL(ψ(t) || ψ*)', markersize=3)
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('KL Divergence [bits]')
    ax3.set_title(f'D(t) and I(t)\nD_KL(ψ* || φ*) = {D_kl_psi_phi:.4f} bits', fontsize=9)
    ax3.legend(frameon=True, fancybox=True, shadow=False)
    sns.despine(ax=ax3)

    # Right: Discrete derivatives of D(t) and I(t)
    if len(generations) > 1:
        gen_diff = np.diff(generations)
        D_derivative = np.diff(D_t) / gen_diff
        I_derivative = np.diff(I_t) / gen_diff
        gen_mid = generations[1:]

        sns.lineplot(x=gen_mid, y=D_derivative, ax=ax4,
                    #  marker='.', linewidth=2.0, color=colors[0], dashes=False,
                    #  errorbar=None, 
                     label='dD/dt', markersize=2)
        sns.lineplot(x=gen_mid, y=I_derivative, ax=ax4,
                    #  marker='x', linewidth=2.0, color=colors[1], dashes=False,
                    #  errorbar=None,
                      label='dI/dt', markersize=3)
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('|Derivative| [bits/generation]')
        ax4.set_title('Convergence Rates (log scale)', fontsize=9)
        ax4.axhline(0, linestyle='--', color='gray', linewidth=1, alpha=0.6)
        ax4.legend(frameon=True, fancybox=True, shadow=False)
        sns.despine(ax=ax4)

    # Left bottom: All components D(t), I(t), rest(t)
    if len(D_t) > 0 and len(I_t) > 0 and len(rest_t) > 0:
        sns.lineplot(x=generations, y=D_t, ax=ax5,
                    #  marker='.', linewidth=2.0, color=colors[0], dashes=False,
                    #  errorbar=None, 
                     label='D(t) = KL(ψ(t) || φ*)', markersize=2)
        sns.lineplot(x=generations, y=I_t, ax=ax5,
                    #  marker='x', linewidth=2.0, color=colors[1], dashes=False,
                    #  errorbar=None,
                      label='I(t) = KL(ψ(t) || ψ*)', markersize=3)
        sns.lineplot(x=generations, y=rest_t, ax=ax5,
                    #  marker='+', linewidth=2.0, color=colors[2], dashes=False,
                    #  errorbar=None,
                      label='rest(t) = E_ψ(t)[log(ψ*/φ*)]', markersize=3)
        ax5.set_xlabel('Generation')
        ax5.set_ylabel('KL Divergence [bits]')
        ax5.set_title('All Components Over Time', fontsize=9)
        ax5.legend(frameon=True, fancybox=True, shadow=False)
        sns.despine(ax=ax5)

    # Right bottom: Verification D(t) vs I(t) + rest(t)
    if len(D_t) > 0 and len(I_t) > 0 and len(rest_t) > 0:
        verification = [i + r for i, r in zip(I_t, rest_t)]
        differences = [abs(d - v) for d, v in zip(D_t, verification)]
        max_diff = max(differences)

        sns.lineplot(x=generations, y=D_t, ax=ax6,
                    #  marker='.', linewidth=2.0, color=colors[0], dashes=False,
                    #  errorbar=None,
                      label='D(t)', markersize=2)
        sns.lineplot(x=generations, y=verification, ax=ax6,
                    #  marker='x', linewidth=2.0, color=colors[1], dashes=False,
                    #  errorbar=None,
                      label='I(t) + rest(t)', markersize=3)
        ax6.text(0.02, 0.98, f'Max |D - (I + rest)|: {max_diff:.2e}',
                 transform=ax6.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
                 fontsize=9)
        ax6.set_xlabel('Generation')
        ax6.set_ylabel('KL Divergence [bits]')
        ax6.set_title('Verification: D(t) vs I(t) + rest(t)', fontsize=9)
        ax6.legend(frameon=True, fancybox=True, shadow=False)
        sns.despine(ax=ax6)

    fig2.show()

    return fig1, fig2


def compute_diffusion_stationary(N: int, s: float, mu: float, debug: bool = False) -> np.ndarray:
    """
    Compute diffusion-approximation steady-state density f*(p) on the discrete grid p=i/N.

    Formula (up to normalization):
        f*(p) ∝ p^(2Nμ-1) (1-p)^(2Nμ-1) exp(2Ns p)

    We evaluate this for p_i = i/N for i=0..N, then
    normalize by Z = Σ_i f*(p_i) to obtain a probability mass over {0..N}: f* -> f*/Z

    Returns a numpy array of length N+1 summing to 1.
    """
    states = np.arange(N + 1, dtype=float)
    p = states / float(N)

    # Numerical stability: clip p in (0,1) for the logarithms. Handle exact 0/1 carefully.
    eps = 1e-10
    p_clipped = np.clip(p, eps, 1.0 - eps)

    exponent = 2.0 * N * mu - 1.0
    selection_term = 2.0 * N * s
    
    if debug:
        print(f"DEBUG: N={N}, s={s}, mu={mu}")
        print(f"DEBUG: exponent = 2*N*mu - 1 = {exponent}")
        print(f"DEBUG: selection_term = 2*N*s = {selection_term}")
        print(f"DEBUG: p range: {p[0]:.6f} to {p[-1]:.6f}")
        print(f"DEBUG: p_clipped range: {p_clipped[0]:.2e} to {p_clipped[-1]:.2e}")
    
    # Use logs for stability
    log_f = exponent * np.log(p_clipped) + exponent * np.log(1.0 - p_clipped) + selection_term * p
    
    if debug:
        print(f"DEBUG: log_f range: {np.min(log_f):.3f} to {np.max(log_f):.3f}")
        print(f"DEBUG: log_f variance: {np.var(log_f):.6f}")

    # For exact boundaries, analytically: when exponent > 0, density → 0 at boundaries; when < 0, blows up.
    # Our clipping already handles numerics; keep log-space safe by recentring before exp.
    log_f -= np.max(log_f)
    unnormalized = np.exp(log_f)
    Z = np.sum(unnormalized)
    
    if debug:
        print(f"DEBUG: Z = {Z:.6f}")
        print(f"DEBUG: unnormalized range: {np.min(unnormalized):.2e} to {np.max(unnormalized):.2e}")
        print(f"DEBUG: Hit fallback? {Z == 0 or not np.isfinite(Z)}")
    
    if Z == 0 or not np.isfinite(Z):
        # Fallback to avoid NaNs
        if debug:
            print("DEBUG: Using fallback uniform distribution!")
        unnormalized = np.ones_like(unnormalized)
        Z = float(N + 1)
    f_star = unnormalized / Z
    
    if debug:
        print(f"DEBUG: f_star range: {np.min(f_star):.6f} to {np.max(f_star):.6f}")
        print(f"DEBUG: f_star sum: {np.sum(f_star):.6f}")
        print(f"DEBUG: Is uniform? {np.allclose(f_star, f_star[0])}")
    
    return f_star


def compute_diffusion_stationary_Iwasa(N: int, s: float, mu: float, W_AA: float = None, W_Aa: float = None, W_aa: float = None, debug: bool = False) -> np.ndarray:
    """
    Compute diffusion-approximation steady-state density n̂(x) using Iwasa formula.

    Formula: n̂(x) ∝ [x^(4Nμ-1) (1-x)^(4Nμ-1)] W(x)^(2N)
    where W(x) = W_AA x² + W_Aa 2x(1-x) + W_aa (1-x)²

    If fitness values not provided, uses standard Wright-Fisher fitnesses:
    W_AA = (1+s)², W_Aa = 1+s, W_aa = 1

    Returns a numpy array of length N+1 summing to 1.
    """
    states = np.arange(N + 1, dtype=float)
    x = states / float(N)

    # Default fitness values if not provided
    if W_AA is None:
        W_AA = (1 + s)**2
    if W_Aa is None:
        W_Aa = 1 + s
    if W_aa is None:
        W_aa = 1.0

    # Numerical stability: clip x in (0,1) for the logarithms
    eps = 1e-10
    x_clipped = np.clip(x, eps, 1.0 - eps)

    # Calculate W(x) = W_AA x² + W_Aa 2x(1-x) + W_aa (1-x)²
    W_x = W_AA * x**2 + W_Aa * 2 * x * (1 - x) + W_aa * (1 - x)**2

    exponent = 4.0 * N * mu - 1.0
    fitness_term = 2.0 * N

    if debug:
        print(f"DEBUG IWASA: N={N}, s={s}, mu={mu}")
        print(f"DEBUG IWASA: W_AA={W_AA}, W_Aa={W_Aa}, W_aa={W_aa}")
        print(f"DEBUG IWASA: exponent = 4*N*mu - 1 = {exponent}")
        print(f"DEBUG IWASA: fitness_term = 2*N = {fitness_term}")
        print(f"DEBUG IWASA: x range: {x[0]:.6f} to {x[-1]:.6f}")
        print(f"DEBUG IWASA: W(x) range: {np.min(W_x):.6f} to {np.max(W_x):.6f}")

    # Use logs for stability
    log_n = exponent * np.log(x_clipped) + exponent * np.log(1.0 - x_clipped) + fitness_term * np.log(W_x)

    if debug:
        print(f"DEBUG IWASA: log_n range: {np.min(log_n):.3f} to {np.max(log_n):.3f}")
        print(f"DEBUG IWASA: log_n variance: {np.var(log_n):.6f}")

    # Recentre and exponentiate
    log_n -= np.max(log_n)
    unnormalized = np.exp(log_n)
    Z = np.sum(unnormalized)

    if debug:
        print(f"DEBUG IWASA: Z = {Z:.6f}")
        print(f"DEBUG IWASA: unnormalized range: {np.min(unnormalized):.2e} to {np.max(unnormalized):.2e}")

    if Z == 0 or not np.isfinite(Z):
        # Fallback to avoid NaNs
        if debug:
            print("DEBUG IWASA: Using fallback uniform distribution!")
        unnormalized = np.ones_like(unnormalized)
        Z = float(N + 1)

    n_star = unnormalized / Z

    if debug:
        print(f"DEBUG IWASA: n̂* range: {np.min(n_star):.6f} to {np.max(n_star):.6f}")
        print(f"DEBUG IWASA: n̂* sum: {np.sum(n_star):.6f}")
        print(f"DEBUG IWASA: Is uniform? {np.allclose(n_star, n_star[0])}")

    return n_star


def compute_diffusion_stationary_4N(N: int, s: float, mu: float, debug: bool = False) -> np.ndarray:
    """
    Compute diffusion-approximation steady-state density f**(p) with 4N scaling.

    Formula: f**(p) ∝ p^(4Nμ-1) (1-p)^(4Nμ-1) exp(4Ns p)
    (Same as original f* but with 4N instead of 2N everywhere)

    Returns a numpy array of length N+1 summing to 1.
    """
    states = np.arange(N + 1, dtype=float)
    p = states / float(N)

    # Numerical stability: clip p in (0,1) for the logarithms
    eps = 1e-10
    p_clipped = np.clip(p, eps, 1.0 - eps)

    exponent = 4.0 * N * mu - 1.0  # 4N instead of 2N
    selection_term = 4.0 * N * s     # 4N instead of 2N
    
    if debug:
        print(f"DEBUG f**: N={N}, s={s}, mu={mu}")
        print(f"DEBUG f**: exponent = 4*N*mu - 1 = {exponent}")
        print(f"DEBUG f**: selection_term = 4*N*s = {selection_term}")

    # Use logs for stability
    log_f = exponent * np.log(p_clipped) + exponent * np.log(1.0 - p_clipped) + selection_term * p
    
    if debug:
        print(f"DEBUG f**: log_f range: {np.min(log_f):.3f} to {np.max(log_f):.3f}")

    # Recentre and exponentiate
    log_f -= np.max(log_f)
    unnormalized = np.exp(log_f)
    Z = np.sum(unnormalized)
    
    if Z == 0 or not np.isfinite(Z):
        # Fallback to avoid NaNs
        if debug:
            print("DEBUG f**: Using fallback uniform distribution!")
        unnormalized = np.ones_like(unnormalized)
        Z = float(N + 1)
        
    f_star_4N = unnormalized / Z
    
    if debug:
        print(f"DEBUG f**: f** range: {np.min(f_star_4N):.6f} to {np.max(f_star_4N):.6f}")
        print(f"DEBUG f**: Is uniform? {np.allclose(f_star_4N, f_star_4N[0])}")
    
    return f_star_4N


def plot_diffusion_comparison(N: int, s: float, mu: float, debug: bool = False):
    """
    Compare diffusion formulas: f*(x), f**(x), n̂*(x) Iwasa with (1+s)², n̂*(x) with 1+2s, and Markov ψ*.
    """
    # Compute all distributions
    f_star = compute_diffusion_stationary(N=N, s=s, mu=mu, debug=debug)
    f_star_4N = compute_diffusion_stationary_4N(N=N, s=s, mu=mu, debug=debug)
    n_star_squared = compute_diffusion_stationary_Iwasa(N=N, s=s, mu=mu, debug=debug)  # Uses (1+s)²
    n_star_linear = compute_diffusion_stationary_Iwasa(N=N, s=s, mu=mu, W_AA=1+2*s, W_Aa=1+s, W_aa=1.0, debug=debug)
    psi_star = compute_stationary_quiet(N=N, s=s, mu=mu, method='direct')

    # Calculate KL divergences first
    eps = 1e-12
    psi_safe = psi_star + eps
    psi_safe = psi_safe / np.sum(psi_safe)
    
    # Calculate KL divergences
    f_safe = f_star + eps
    f_safe = f_safe / np.sum(f_safe)
    kl_f = np.sum(psi_safe * np.log2(psi_safe / f_safe))
    
    f4N_safe = f_star_4N + eps
    f4N_safe = f4N_safe / np.sum(f4N_safe)
    kl_f4N = np.sum(psi_safe * np.log2(psi_safe / f4N_safe))
    
    n_sq_safe = n_star_squared + eps
    n_sq_safe = n_sq_safe / np.sum(n_sq_safe)
    kl_n_sq = np.sum(psi_safe * np.log2(psi_safe / n_sq_safe))
    
    n_lin_safe = n_star_linear + eps
    n_lin_safe = n_lin_safe / np.sum(n_lin_safe)
    kl_n_lin = np.sum(psi_safe * np.log2(psi_safe / n_lin_safe))

    # Create DataFrame with full KL divergences in labels (normal x-axis)
    states = np.arange(N + 1)
    df = pd.DataFrame({
        'state': np.concatenate([states, states, states, states, states]),
        'probability': np.concatenate([f_star, f_star_4N, n_star_squared, n_star_linear, psi_star]),
        'source': np.concatenate([
            np.repeat(f'f* (2N): D_KL(ψ*||f*)={kl_f:.3f}', N + 1), 
            np.repeat(f'f** (4N): D_KL(ψ*||f**)={kl_f4N:.3f}', N + 1),
            np.repeat(f'n̂* (1+s)²: D_KL(ψ*||n̂*)={kl_n_sq:.3f}', N + 1),
            np.repeat(f'n̂* (1+2s): D_KL(ψ*||n̂*)={kl_n_lin:.3f}', N + 1),
            np.repeat('ψ* (Markov)', N + 1)
        ])
    })

    fig, ax = plt.subplots(1, 1, figsize=(9, 5), constrained_layout=True, dpi=100)
    #sns.barplot(data=df, x='state', y='probability', hue='source', ax=ax, dodge=True, width=2.0,alpha=0.7, palette="tab10")
    sns.lineplot(data=df, x='state', y='probability', hue='source', ax=ax, marker='*', linewidth=1,  palette="Set1")

    ax.set_xlabel('Number of "1" alleles (i)')
    ax.set_ylabel('Probability (log scale)')
    ax.set_title(f'Distribution Comparison\nN={N}, s={s}, μ={mu}')
    
    # Normal x-axis behavior
    ax.set_xticks([0, N])  # Only boundary ticks
    
    #ax.set_yscale('log')  # Log scale for y-axis
    ax.set_yscale('linear')
    ax.legend(frameon=True, fancybox=True, shadow=False, fontsize=9, loc='upper center')
    sns.despine(ax=ax)
    
    print(f"\nKL Divergences D_KL(ψ* || g) [bits]:")
    print(f"  D_KL(ψ* || f*) = {kl_f:.6f}")
    print(f"  D_KL(ψ* || f**) = {kl_f4N:.6f}")
    print(f"  D_KL(ψ* || n̂*(1+s)²) = {kl_n_sq:.6f}")
    print(f"  D_KL(ψ* || n̂*(1+2s)) = {kl_n_lin:.6f}")
    print(f"\nBest approximation: {['f*', 'f**', 'n̂*(1+s)²', 'n̂*(1+2s)'][np.argmin([kl_f, kl_f4N, kl_n_sq, kl_n_lin])]}")
    
    fig.show()
    return fig, {
        'f_star': f_star,
        'f_star_4N': f_star_4N,
        'n_star_squared': n_star_squared, 
        'n_star_linear': n_star_linear,
        'psi_star': psi_star,
        'kl_divergences': {
            'f_star': kl_f,
            'f_star_4N': kl_f4N,
            'n_star_squared': kl_n_sq,
            'n_star_linear': kl_n_lin
        }
    }


def plot_diffusion_vs_markov(N: int, s: float, mu: float, debug: bool = True):
    """
    Compute diffusion-based f*(p) and Markov ψ* and plot them side-by-side as histograms.

    Not called from __main__; call manually when needed.
    """
    # Compute diffusion f*
    f_star = compute_diffusion_stationary(N=N, s=s, mu=mu, debug=debug)

    # Compute Markov ψ*
    psi_star = compute_stationary_quiet(N=N, s=s, mu=mu, method='direct')

    states = np.arange(N + 1)
    df = pd.DataFrame({
        'state': np.concatenate([states, states]),
        'probability': np.concatenate([f_star, psi_star]),
        'source': np.concatenate([np.repeat('Diffusion f*', N + 1), np.repeat('Markov ψ*', N + 1)])
    })

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
    sns.barplot(data=df, x='state', y='probability', hue='source', ax=ax, dodge=True)
    ax.set_xlabel('Number of "1" alleles (i)')
    ax.set_ylabel('Probability')
    ax.set_title(f'Diffusion f*(p=i/N) vs Markov ψ*\nN={N}, s={s}, μ={mu}')
    ax.set_xticks([0, N])  # Only boundary ticks
    ax.legend(frameon=True, fancybox=True, shadow=False)
    sns.despine(ax=ax)
    fig.show()
    return fig, {'diffusion_f_star': f_star, 'markov_psi_star': psi_star}


if __name__ == "__main__":
    # Compare diffusion formulas: f*(x) vs n̂*(x) from Iwasa
    print("="*80)
    print("DIFFUSION FORMULA COMPARISON")
    print("="*80)
    
    # Parameters
    N = 100
    mu = 0.0005
    s = 0.01
    
    print(f"Parameters: N={N}, μ={mu}, s={s}")
    print("Comparing 5 distributions:")
    print("  1. f*(x) ∝ x^(2Nμ-1) (1-x)^(2Nμ-1) exp(2Ns x)")
    print("  2. f**(x) ∝ x^(4Nμ-1) (1-x)^(4Nμ-1) exp(4Ns x)")
    print("  3. n̂*(x) with W_AA = (1+s)², W_Aa = 1+s, W_aa = 1")
    print("  4. n̂*(x) with W_AA = 1+2s, W_Aa = 1+s, W_aa = 1") 
    print("  5. ψ*(x) from Markov chain (exact)")
    print("  where n̂*(x) ∝ x^(4Nμ-1) (1-x)^(4Nμ-1) W(x)^(2N)")
    print("  and W(x) = W_AA x² + W_Aa 2x(1-x) + W_aa (1-x)²")
    
    # Run the comparison
    fig, data = plot_diffusion_comparison(N=N, s=s, mu=mu, debug=True)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
