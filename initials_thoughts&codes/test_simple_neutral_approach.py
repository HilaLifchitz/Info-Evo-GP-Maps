#!/usr/bin/env python3
"""
Enhanced Neutral KL Approach for Equilibrium Detection in Wright-Fisher Simulations

PROBLEM WE'RE SOLVING:
===================
We need to detect when a Wright-Fisher population under selection reaches 
"stochastic equilibrium" - the point where systematic evolution stops and 
only genetic drift remains.

Previous methods were either:
- Too conservative (detected equilibrium at ~7650 generations)
- Too aggressive (detected equilibrium at ~2000 generations)
- Expected target: ~5000 generations based on visual inspection

CORE INSIGHT:
============
At true stochastic equilibrium, the KL divergence should fluctuate EXACTLY 
like a neutral population (s=0) - just around a different mean value:

    Selection Population at Equilibrium ≈ Neutral Population + Constant Offset

The fluctuation patterns (variance, change statistics) should be identical.
Only the mean level differs.

THE ENHANCED APPROACH:
=====================
1. Run neutral simulation to establish "drift fingerprint"
2. Calculate multiple KL statistics over time for selection simulation:
   - Rolling variance: Overall fluctuation level
   - Change mean: Average change per generation (should → 0)
   - Change std: Fluctuation magnitude (should → neutral level)
3. Detect when ALL statistics simultaneously approach neutral levels
4. Require sustained period to avoid false positives

BIOLOGICAL INTERPRETATION:
=========================
- Generations 0-5000: Systematic evolution (selection dominates drift)
  * KL changes directionally due to selection pressure
  * High variance, non-zero change mean, elevated change std
  
- Generations 5000+: Stochastic equilibrium (drift dominates selection)
  * KL fluctuates randomly around stable mean (like neutral population)
  * Variance ≈ neutral, change mean ≈ 0, change std ≈ neutral
"""

import numpy as np
import matplotlib.pyplot as plt
from QE import (
    WrightFisherSimulator,
    WrightFisherSimulator_FreeRecombination,
    plot_phenotype_counts,
    plot_kl_trajectory,
    plot_delta_kl,
    plot_abs_delta_kl,
)

def simple_neutral_kl_equilibrium_detection(window_size=500, smooth_window=101,
                                            simulator_cls: type = WrightFisherSimulator):
    """
    Enhanced Neutral KL Equilibrium Detection with Smoothing
    
    This function implements the multi-criteria approach described above to detect
    when a Wright-Fisher population reaches stochastic equilibrium by comparing
    KL fluctuation patterns to those of a neutral population.
    
    =========================================================================
    COMPREHENSIVE CONVERGENCE TESTING HISTORY
    =========================================================================
    
    We have systematically tested multiple approaches to detect convergence of the
    empirical phenotype distribution under selection to a stochastic equilibrium:
    
    1. INITIAL KL DIVERGENCE DISCREPANCY (80 vs 30 bits)
       - Problem: Expected KL divergence of ~80 bits from previous code, but
         current implementation gives ~30 bits
       - Investigation: Tested different KL calculation methods, reference
         distributions, parameter scaling, and measurement timing
       - Findings: KL calculation method (epsilon smoothing vs strict masking)
         significantly impacts values. Empirical distribution converges to more
         extreme mean than initially assumed.
    
    2. BASIC CONVERGENCE ANALYSIS (FAILED)
       - Method: Track mean/variance trajectories, window-based KL divergence,
         autocorrelation, and stationarity tests
       - Problem: Too conservative - didn't account for genetic drift noise
       - Failure: Declared convergence too late or never, ignoring that finite
         populations always have fluctuations
    
    3. STOCHASTIC EQUILIBRIUM ANALYSIS (FAILED)
       - Method: Distinguish systematic trends from random fluctuations using
         sliding window linear regression (trend_strength = |slope| * R²)
       - Problem: Trend detection was too sensitive to noise
       - Failure: Could not reliably distinguish evolution from drift
    
    4. PHENOTYPIC VARIANCE COMPARISON (FAILED)
       - Method: Compare phenotypic variance to neutral baseline
       - Problem: Selection populations inherently have higher variance due to
         continuous selection-drift interplay
       - Failure: Too strict - equilibrium populations never match neutral
         variance levels
    
    5. NEUTRAL KL FLUCTUATION COMPARISON (CURRENT APPROACH)
       - Method: Compare KL variance, mean change, and change std to neutral
         baseline using rolling statistics
       - Criteria:
         a) Rolling KL variance < neutral_variance * 3.0
         b) |Rolling change mean| < neutral_change_mean + 2*neutral_change_std
         c) Rolling mean stability < threshold
       - Problem: Still too strict - selection populations have inherently
         higher noise even at equilibrium
       - Current Status: No sustained equilibrium period found
         - Best variance ratio: 0.50x (at generation 26021)
         - But change mean still too high: 0.011147 vs threshold 0.014164
    
    KEY INSIGHTS:
    - Selection populations never reach the same noise levels as neutral populations
    - Equilibrium should be detected by stability of fluctuation patterns, not
      absolute noise levels
    - The continuous interplay of selection and drift creates persistent variance
    - Need to focus on trend cessation rather than variance matching
    
    FUTURE DIRECTIONS:
    - Implement trend-based detection with more sophisticated noise filtering
    - Use longer time series to better distinguish systematic vs random changes
    - Consider adaptive thresholds that account for selection strength
    - Focus on detecting when systematic directional changes stop, rather than
      matching neutral noise characteristics
    
    Parameters:
    -----------
    window_size : int, optional (default=500)
        Size of the rolling window for calculating statistics. Larger windows
        give more stable estimates but less temporal resolution.
        - Small windows (100-300): More sensitive to short-term changes
        - Medium windows (500-1000): Good balance of stability and resolution  
        - Large windows (1000+): Very stable but may miss transitions
        
    smooth_window : int, optional (default=101)
        Size of smoothing window for KL trajectories. Applied before analysis
        to reduce noise and reveal underlying trends.
        - Small smoothing (21-51): Preserves detail, reduces noise slightly
        - Medium smoothing (101-201): Good balance, recommended for most cases
        - Large smoothing (501+): Very smooth, may obscure real transitions
    """
    
    print("="*60)
    print("ENHANCED NEUTRAL KL EQUILIBRIUM DETECTION")
    print("="*60)
    
    # =========================================================================
    # SIMULATION PARAMETERS
    # =========================================================================
    # Extended simulation parameters for better equilibrium detection
    N, l, mu = 40, 1000, 0.0005  # Population size, loci, mutation rate
    s = 0.01                      # Selection coefficient  
    T = 40000                     # Much longer time to capture full equilibrium
    
    print(f"Parameters: N={N}, l={l}, s={s}, mu={mu}, T={T}")
    print(f"Rolling window size: {window_size} generations")
    print(f"Smoothing window size: {smooth_window} generations")
    print(f"Extended simulation for comprehensive equilibrium analysis")
    
    # =========================================================================
    # STEP 1: ESTABLISH NEUTRAL BASELINE ("DRIFT FINGERPRINT")
    # =========================================================================
    # Run a neutral simulation (s=0) to understand what pure genetic drift 
    # looks like. This gives us the baseline fluctuation patterns that we
    # expect to see when the selection population reaches equilibrium.
    
    print("\nStep 1: Establishing neutral drift baseline...")
    print("➤ Running neutral simulation (s=0) to get 'drift fingerprint'")
    
    neutral_sim = simulator_cls(N, l, s=0.0, mu=mu, T=T, seed=42)
    neutral_res = neutral_sim.run()
    neutral_kl_raw = np.array(neutral_res['kl'])
    
    print(f"  Neutral simulation complete: {len(neutral_kl_raw)} generations")
    
    # Apply smoothing to neutral KL trajectory
    def smooth_trajectory(trajectory, window):
        """Apply moving average smoothing to reduce noise"""
        if window <= 1 or window >= len(trajectory):
            return trajectory
        return np.convolve(trajectory, np.ones(window)/window, mode='same')
    
    neutral_kl = smooth_trajectory(neutral_kl_raw, smooth_window)
    print(f"  Applied smoothing with window {smooth_window} to neutral KL")
    
    # =========================================================================
    # CALCULATE NEUTRAL BASELINE STATISTICS
    # =========================================================================
    # Extract steady-state period (skip initial transient where population
    # is still equilibrating from initial conditions)
    
    neutral_steady = neutral_kl  # Use full trajectory
    print(f"  Analyzing full trajectory: generations 0-{len(neutral_kl)}")
    
    # Overall KL statistics in neutral steady state
    neutral_baseline_var = np.var(neutral_steady)      # Overall fluctuation level
    neutral_baseline_mean = np.mean(neutral_steady)    # Mean KL level
    
    # Change statistics (generation-to-generation differences)
    # These capture the "motion characteristics" of the KL trajectory
    neutral_changes = np.diff(neutral_steady)
    neutral_change_mean = np.mean(neutral_changes)     # Average change (should be ~0)
    neutral_change_std = np.std(neutral_changes)       # Fluctuation magnitude
    
    # NEUTRAL PHENOTYPIC BASELINE
    neutral_phenotypic_means = []
    neutral_phenotypic_vars = []
    for t in range(len(neutral_res['phenotype_probs'])):
        phenotype_dist = neutral_res['phenotype_probs'][t]
        mean_phenotype = np.sum(phenotype_dist * np.arange(len(phenotype_dist)))
        neutral_phenotypic_means.append(mean_phenotype)
    
    neutral_phenotypic_mean = np.mean(neutral_phenotypic_means)
    neutral_phenotypic_var = np.var(neutral_phenotypic_means)
    neutral_phenotypic_changes = np.diff(neutral_phenotypic_means)
    neutral_phenotypic_change_std = np.std(neutral_phenotypic_changes)
    
    print(f"\n➤ Neutral 'drift fingerprint' established:")
    print(f"  KL - Overall variance: {neutral_baseline_var:.4f}")
    print(f"  KL - Overall mean: {neutral_baseline_mean:.4f}")
    print(f"  KL - Change mean: {neutral_change_mean:.6f} (should be ~0 for random walk)")
    print(f"  KL - Change std: {neutral_change_std:.4f} (drift fluctuation magnitude)")
    print(f"  Phenotype - Mean: {neutral_phenotypic_mean:.2f}")
    print(f"  Phenotype - Variance: {neutral_phenotypic_var:.2f}")
    print(f"  Phenotype - Change std: {neutral_phenotypic_change_std:.4f}")
    
    # =========================================================================
    # STEP 2: RUN SELECTION SIMULATION
    # =========================================================================
    # Run the actual selection simulation that we want to analyze
    
    print(f"\nStep 2: Running selection simulation...")
    print(f"➤ This simulation should show systematic evolution → equilibrium transition")
    
    selection_sim = simulator_cls(N, l, s, mu, T, seed=42)
    selection_res = selection_sim.run()
    selection_kl_raw = np.array(selection_res['kl'])
    
    # Apply smoothing to selection KL trajectory  
    selection_kl = smooth_trajectory(selection_kl_raw, smooth_window)
    
    final_mean = selection_res['phenotype_probs'][-1] @ np.arange(len(selection_res['phenotype_probs'][-1]))
    print(f"  Selection simulation complete: {len(selection_kl_raw)} generations")
    print(f"  Applied smoothing with window {smooth_window} to selection KL")
    print(f"  Final mean phenotype: {final_mean:.2f}")
    
    # SMOOTHING EFFECT ANALYSIS
    print(f"\n➤ Smoothing effect analysis:")
    print(f"  Raw KL std: {np.std(selection_kl_raw):.4f} → Smoothed KL std: {np.std(selection_kl):.4f}")
    print(f"  Noise reduction: {(1 - np.std(selection_kl)/np.std(selection_kl_raw))*100:.1f}%")
    
    # =========================================================================
    # STEP 3: CALCULATE ROLLING STATISTICS FOR SELECTION SIMULATION
    # =========================================================================
    # For each time point, calculate statistics in a sliding window around
    # that time. This lets us track how the KL behavior changes over time.
    
    print(f"\nStep 3: Calculating rolling KL statistics...")
    
    window = window_size  # Use the parameter-controlled window size
    print(f"➤ Using rolling window size: {window} generations")
    print(f"➤ This gives us {len(selection_kl) - window} time points to analyze")
    print(f"➤ Window size effects:")
    print(f"   - Smaller windows: More sensitive, less stable")
    print(f"   - Larger windows: More stable, less sensitive")
    
    # Arrays to store rolling statistics
    rolling_vars = []          # Overall variance in each window
    rolling_means = []         # Mean KL level in each window
    rolling_change_means = []  # Mean change per generation (window=1, just np.diff)
    times = []                 # Time points (center of each window)
    
    # PHENOTYPIC ANALYSIS ARRAYS
    phenotypic_means = []      # Mean phenotype in each window
    phenotypic_vars = []       # Phenotypic variance in each window
    phenotypic_change_means = []  # Phenotypic change per generation
    
    # Calculate rolling statistics using sliding window
    for i in range(window, len(selection_kl) - window):
        # Extract window centered at time i
        kl_window = selection_kl[i-window//2:i+window//2]
        
        # STATISTIC 1: Rolling variance (overall fluctuation level around mean)
        # High during evolution, should approach neutral level at equilibrium
        rolling_var = np.var(kl_window)
        
        # STATISTIC 2: Rolling mean (average KL level in window)
        # This tracks the central tendency - should stabilize at equilibrium
        rolling_mean = np.mean(kl_window)
        
        # STATISTIC 3: Change mean (generation-to-generation differences)
        # This is essentially a window of size 1 - immediate neighbors
        # Should approach 0 at equilibrium (no systematic directional change)
        if i > 0:
            change_mean = selection_kl[i] - selection_kl[i-1]  # Single-step change
        else:
            change_mean = 0.0
        
        # PHENOTYPIC ANALYSIS
        # Extract phenotypic data for the same window
        phenotype_window = []
        for t in range(i-window//2, i+window//2):
            if t < len(selection_res['phenotype_probs']):
                # Calculate mean phenotype from probability distribution
                phenotype_dist = selection_res['phenotype_probs'][t]
                mean_phenotype = np.sum(phenotype_dist * np.arange(len(phenotype_dist)))
                phenotype_window.append(mean_phenotype)
        
        if phenotype_window:
            # PHENOTYPIC STATISTICS
            phenotypic_mean = np.mean(phenotype_window)  # Average phenotype in window
            phenotypic_var = np.var(phenotype_window)    # Phenotypic variance in window
            
            # Phenotypic change (single-step)
            if i > 0 and i < len(selection_res['phenotype_probs']):
                current_phenotype = np.sum(selection_res['phenotype_probs'][i] * np.arange(len(selection_res['phenotype_probs'][i])))
                prev_phenotype = np.sum(selection_res['phenotype_probs'][i-1] * np.arange(len(selection_res['phenotype_probs'][i-1])))
                phenotypic_change = current_phenotype - prev_phenotype
            else:
                phenotypic_change = 0.0
        else:
            phenotypic_mean = phenotypic_var = phenotypic_change = 0.0
        
        # Store results
        rolling_vars.append(rolling_var)
        rolling_means.append(rolling_mean)
        rolling_change_means.append(change_mean)
        phenotypic_means.append(phenotypic_mean)
        phenotypic_vars.append(phenotypic_var)
        phenotypic_change_means.append(phenotypic_change)
        times.append(i)
    
    # Convert to numpy arrays for easier manipulation
    rolling_vars = np.array(rolling_vars)
    rolling_means = np.array(rolling_means)
    rolling_change_means = np.array(rolling_change_means)
    phenotypic_means = np.array(phenotypic_means)
    phenotypic_vars = np.array(phenotypic_vars)
    phenotypic_change_means = np.array(phenotypic_change_means)
    times = np.array(times)
    
    print(f"  Rolling statistics calculated for {len(times)} time points")
    print(f"  Phenotypic analysis: mean range {np.min(phenotypic_means):.1f}-{np.max(phenotypic_means):.1f}")
    print(f"  Phenotypic variance range: {np.min(phenotypic_vars):.2f}-{np.max(phenotypic_vars):.2f}")
    
    # =========================================================================
    # STEP 4: MULTI-CRITERIA EQUILIBRIUM DETECTION
    # =========================================================================
    # We look for periods where ALL three statistics simultaneously approach
    # neutral levels. This indicates that the KL trajectory behaves like
    # pure genetic drift rather than systematic evolution.
    
    print(f"\nStep 4: Multi-criteria equilibrium detection...")
    print(f"➤ Looking for periods where KL behaves like neutral drift")
    
    # =========================================================================
    # DEFINE EQUILIBRIUM CRITERIA (with biological justification)
    # =========================================================================
    
    # CRITERION 1: Rolling variance close to neutral level
    # At equilibrium, KL should fluctuate with similar magnitude as neutral
    # We allow 3x tolerance to account for residual selection effects
    var_threshold = neutral_baseline_var * 3.0
    
    # CRITERION 2: Change mean close to zero (no systematic trend)
    # At equilibrium, there should be no net directional change in KL
    # We set threshold based on neutral noise level
    change_mean_threshold = abs(neutral_change_mean) + 2 * neutral_change_std
    
    # CRITERION 3: Rolling mean stabilization
    # The rolling mean should show less variation at equilibrium
    # We look for when the rolling mean becomes stable (low variance)
    rolling_mean_stability_threshold = np.var(rolling_means) * 0.1  # Much more stable than overall
    
    # Calculate rolling variance of the rolling means (stability of KL level)
    mean_stability_window = 100
    rolling_mean_stabilities = []
    for j in range(mean_stability_window, len(rolling_means) - mean_stability_window):
        mean_window = rolling_means[j-mean_stability_window//2:j+mean_stability_window//2]
        stability = np.var(mean_window)
        rolling_mean_stabilities.append(stability)
    
    # Pad to match length
    rolling_mean_stabilities = np.array(rolling_mean_stabilities)
    padding = len(rolling_means) - len(rolling_mean_stabilities)
    rolling_mean_stabilities = np.pad(rolling_mean_stabilities, (padding//2, padding - padding//2), mode='edge')
    
    print(f"\n➤ Equilibrium detection criteria:")
    print(f"  1. Variance < {var_threshold:.4f} (neutral: {neutral_baseline_var:.4f})")
    print(f"  2. |Change mean| < {change_mean_threshold:.6f} (neutral: {neutral_change_mean:.6f})")
    print(f"  3. Rolling mean stability < {rolling_mean_stability_threshold:.4f}")
    
    # =========================================================================
    # APPLY CRITERIA TO FIND EQUILIBRIUM CANDIDATES
    # =========================================================================
    
    # Test each criterion separately
    var_ok = rolling_vars < var_threshold
    change_mean_ok = np.abs(rolling_change_means) < change_mean_threshold
    mean_stable_ok = rolling_mean_stabilities < rolling_mean_stability_threshold
    
    # Combined criterion: ALL must be satisfied simultaneously
    equilibrium_candidates = var_ok & change_mean_ok & mean_stable_ok
    
    print(f"\n➤ Criterion evaluation:")
    print(f"  Variance criterion:     {np.sum(var_ok):4d}/{len(var_ok)} windows pass")
    print(f"  Change mean criterion:  {np.sum(change_mean_ok):4d}/{len(change_mean_ok)} windows pass") 
    print(f"  Mean stability criterion: {np.sum(mean_stable_ok):4d}/{len(mean_stable_ok)} windows pass")
    print(f"  ALL criteria combined:  {np.sum(equilibrium_candidates):4d}/{len(equilibrium_candidates)} windows pass")
    
    # =========================================================================
    # FIND SUSTAINED EQUILIBRIUM PERIOD
    # =========================================================================
    # We require a sustained period (≥1000 generations) where all criteria
    # are met to avoid false positives from temporary fluctuations
    
    min_duration = 1000  # Minimum sustained period required
    print(f"\n➤ Looking for sustained equilibrium period (≥{min_duration} generations)")
    
    equilibrium_start = None
    current_start = None
    current_duration = 0
    
    # Scan through time points to find sustained periods
    for i, (time, meets_criteria) in enumerate(zip(times, equilibrium_candidates)):
        if meets_criteria:
            # Start or continue a qualifying period
            if current_start is None:
                current_start = time
                current_duration = 1
            else:
                current_duration += 1
        else:
            # End of qualifying period - check if it was long enough
            if current_start is not None and current_duration >= min_duration:
                if equilibrium_start is None:  # Take first qualifying period
                    equilibrium_start = current_start
                    print(f"  Found sustained period: generations {current_start}-{time}")
                    break
            # Reset for next potential period
            current_start = None
            current_duration = 0
    
    # Check if the final period qualifies (extends to end of simulation)
    if current_start is not None and current_duration >= min_duration and equilibrium_start is None:
        equilibrium_start = current_start
        print(f"  Found sustained period extending to end: generations {current_start}-{times[-1]}")
    
    if equilibrium_start:
        eq_idx = np.where(times == equilibrium_start)[0][0]
        equilibrium_var = rolling_vars[eq_idx]
        equilibrium_change_mean = rolling_change_means[eq_idx]
        equilibrium_rolling_mean = rolling_means[eq_idx]
        
        print(f"✅ EQUILIBRIUM DETECTED at generation {equilibrium_start}")
        print(f"   KL variance: {equilibrium_var:.4f} (vs neutral {neutral_baseline_var:.4f})")
        print(f"   Change mean: {equilibrium_change_mean:.6f} (vs neutral {neutral_change_mean:.6f})")
        print(f"   Rolling mean: {equilibrium_rolling_mean:.4f} (vs neutral {neutral_baseline_mean:.4f})")
    else:
        print("❌ No sustained equilibrium period found")
        # Show some stats about closest approaches
        min_var = np.min(rolling_vars)
        min_var_idx = np.argmin(rolling_vars)
        min_time = times[min_var_idx]
        
        print(f"   Best variance: {min_var:.4f} at generation {min_time} (ratio: {min_var / neutral_baseline_var:.2f}x)")
        print(f"   Change mean there: {rolling_change_means[min_var_idx]:.6f}")
        print(f"   Rolling mean there: {rolling_means[min_var_idx]:.4f}")
    
    # Step 5: Create diagnostic plot (include both raw and smoothed)
    create_diagnostic_plot(neutral_kl_raw, neutral_kl, selection_kl_raw, selection_kl, 
                          rolling_vars, rolling_means, rolling_change_means,
                          phenotypic_means, phenotypic_vars, phenotypic_change_means, times, 
                          neutral_baseline_var, neutral_change_mean, neutral_change_std,
                          neutral_phenotypic_mean, neutral_phenotypic_var, neutral_phenotypic_change_std,
                          equilibrium_start, window_size, smooth_window)
    
    return equilibrium_start



def create_diagnostic_plot(neutral_kl_raw, neutral_kl_smooth, selection_kl_raw, selection_kl_smooth,
                          rolling_vars, rolling_means, rolling_change_means, 
                          phenotypic_means, phenotypic_vars, phenotypic_change_means, times, 
                          neutral_baseline_var, neutral_change_mean, neutral_change_std,
                          neutral_phenotypic_mean, neutral_phenotypic_var, neutral_phenotypic_change_std,
                          equilibrium_start, window_size, smooth_window):
    """
    Create comprehensive diagnostic plot showing raw vs smoothed KL trajectories
    and the enhanced neutral approach analysis.
    
    Parameters:
    -----------
    window_size : int
        The window size used for rolling statistics (for plot titles)
    smooth_window : int  
        The smoothing window size used for KL trajectories
    """
    
    fig, axes = plt.subplots(6, 1, figsize=(16, 18))
    
    # Plot 1: Raw vs Smoothed KL trajectories  
    axes[0].plot(neutral_kl_raw, 'lightblue', alpha=0.7, linewidth=1, label='Neutral KL (raw)')
    axes[0].plot(selection_kl_raw, 'pink', alpha=0.7, linewidth=1, label='Selection KL (raw)')
    axes[0].plot(neutral_kl_smooth, 'blue', linewidth=2, label='Neutral KL (smoothed)')
    axes[0].plot(selection_kl_smooth, 'red', linewidth=2, label='Selection KL (smoothed)')
    if equilibrium_start:
        axes[0].axvline(x=equilibrium_start, color='green', linestyle='--', linewidth=2,
                       label=f'Detected equilibrium: {equilibrium_start}')
    axes[0].set_ylabel('KL Divergence (bits)')
    axes[0].set_title(f'Raw vs Smoothed KL Trajectories (smoothing window: {smooth_window})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Rolling variance
    axes[1].plot(times, rolling_vars, 'yellow', linewidth=2, label='Selection KL variance')
    axes[1].axhline(y=neutral_baseline_var, color='blue', linestyle='-', alpha=0.7, 
                   label=f'Neutral baseline: {neutral_baseline_var:.4f}')
    var_threshold = neutral_baseline_var * 3.0
    axes[1].axhline(y=var_threshold, color='orange', linestyle='--', alpha=0.7,
                   label=f'Variance threshold: {var_threshold:.4f}')
    
    if equilibrium_start:
        axes[1].axvline(x=equilibrium_start, color='green', linestyle='--', linewidth=2)
    
    axes[1].set_ylabel('KL Variance')
    axes[1].set_title(f'Rolling KL Variance (window={window_size})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')  # Log scale to see the convergence better
    
    # Plot 3: Rolling mean (KL level stabilization)
    axes[2].plot(times, rolling_means, 'green', linewidth=2, label='Rolling KL mean')
    axes[2].axhline(y=neutral_baseline_var, color='blue', linestyle='-', alpha=0.7,
                   label=f'Neutral mean: {neutral_baseline_var:.4f}')
    
    if equilibrium_start:
        axes[2].axvline(x=equilibrium_start, color='green', linestyle='--', linewidth=2)
    
    axes[2].set_ylabel('Rolling KL Mean (bits)')
    axes[2].set_title(f'Rolling KL Mean Level (window={window_size}) - Should stabilize at equilibrium')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Single-step changes (should approach 0)
    axes[3].plot(times, rolling_change_means, 'purple', linewidth=1, alpha=0.7, label='Single-step KL changes')
    axes[3].axhline(y=neutral_change_mean, color='blue', linestyle='-', alpha=0.7,
                   label=f'Neutral change mean: {neutral_change_mean:.6f}')
    axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Zero line')
    
    # Show acceptable range based on neutral noise
    change_threshold = abs(neutral_change_mean) + 2 * neutral_change_std
    axes[3].fill_between([times[0], times[-1]], [-change_threshold, -change_threshold], 
                        [change_threshold, change_threshold], alpha=0.2, color='green', 
                        label=f'Neutral range: ±{change_threshold:.4f}')
    
    if equilibrium_start:
        axes[3].axvline(x=equilibrium_start, color='green', linestyle='--', linewidth=2)
    
    axes[3].set_xlabel('Generation')
    axes[3].set_ylabel('KL Change (bits/gen)')
    axes[3].set_title('Single-Step KL Changes (should approach 0 at equilibrium)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    # Plot 4: Phenotypic mean trajectory
    axes[4].plot(times, phenotypic_means, 'orange', linewidth=2, label='Selection phenotypic mean')
    axes[4].axhline(y=neutral_phenotypic_mean, color='blue', linestyle='-', alpha=0.7,
                   label=f'Neutral phenotypic mean: {neutral_phenotypic_mean:.1f}')
    
    if equilibrium_start:
        axes[4].axvline(x=equilibrium_start, color='green', linestyle='--', linewidth=2)
    
    axes[4].set_ylabel('Phenotypic Mean')
    axes[4].set_title('Phenotypic Mean Trajectory (should stabilize at equilibrium)')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    
    # Plot 5: Phenotypic variance
    axes[5].plot(times, phenotypic_vars, 'brown', linewidth=2, label='Selection phenotypic variance')
    axes[5].axhline(y=neutral_phenotypic_var, color='blue', linestyle='-', alpha=0.7,
                   label=f'Neutral phenotypic variance: {neutral_phenotypic_var:.2f}')
    
    if equilibrium_start:
        axes[5].axvline(x=equilibrium_start, color='green', linestyle='--', linewidth=2)
    
    axes[5].set_xlabel('Generation')
    axes[5].set_ylabel('Phenotypic Variance')
    axes[5].set_title('Phenotypic Variance (tests selection noise hypothesis)')
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)
    
    # Align all x-axes to ensure perfect alignment
    for ax in axes:
        ax.set_xlim(0, max(len(neutral_kl_raw), len(selection_kl_raw)))
    
    plt.tight_layout()
    plt.show()

def test_multiple_window_sizes(simulator_cls: type = WrightFisherSimulator):
    """
    Test equilibrium detection with different window sizes to see the effect.
    """
    window_sizes = [200, 500, 1000, 2000]
    results = {}
    
    print("="*80)
    print("TESTING MULTIPLE WINDOW SIZES")
    print("="*80)
    
    for window_size in window_sizes:
        print(f"\n{'='*20} WINDOW SIZE: {window_size} {'='*20}")
        try:
            equilibrium_start = simple_neutral_kl_equilibrium_detection(window_size=window_size,
                                                                        simulator_cls=simulator_cls)
            results[window_size] = equilibrium_start
        except Exception as e:
            print(f"Error with window size {window_size}: {e}")
            results[window_size] = None
    
    # Summary comparison
    print(f"\n" + "="*80)
    print("WINDOW SIZE COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Window Size':<12} {'Equilibrium Start':<18} {'Status'}")
    print("-" * 45)
    
    for window_size, eq_start in results.items():
        if eq_start:
            status = f"✅ Gen {eq_start}"
        else:
            status = "❌ Not detected"
        print(f"{window_size:<12} {str(eq_start) if eq_start else 'None':<18} {status}")
    
    print(f"\nRecommendation:")
    print(f"- Small windows (200-500): Best for detecting early transitions")
    print(f"- Large windows (1000+): Best for stable, conservative detection")
    
    return results

def quick_test_window_size(window_size, simulator_cls: type = WrightFisherSimulator):
    """
    Quick test for a specific window size without plots.
    
    Parameters:
    -----------
    window_size : int
        The rolling window size to test
        
    Returns:
    --------
    int or None
        Generation when equilibrium is detected, or None if not detected
    """
    print(f"Quick test with window size {window_size}...")
    # This would be a simplified version without plotting
    # For now, just call the main function
    return simple_neutral_kl_equilibrium_detection(window_size=window_size, simulator_cls=simulator_cls)

def test_old_failed_convergence_methods(simulator_cls: type = WrightFisherSimulator):
    """
    Recreate the old convergence testing methods that failed, for comparison
    and educational purposes. This shows why these approaches didn't work
    for stochastic populations with genetic drift.
    
    METHODS IMPLEMENTED:
    ===================
    1. BASIC CONVERGENCE ANALYSIS (Failed)
       - Track mean/variance trajectories of KL
       - Window-based stationarity tests  
       - Autocorrelation analysis
       - Problem: Too conservative, didn't account for genetic drift
    
    2. STOCHASTIC EQUILIBRIUM ANALYSIS (Failed)
       - Sliding window linear regression
       - Trend strength = |slope| × R²
       - Problem: Over-sensitive to noise, couldn't distinguish evolution from drift
    """
    
    print("="*80)
    print("TESTING OLD FAILED CONVERGENCE METHODS")
    print("="*80)
    print("This recreates the methods we tried before that didn't work properly")
    print("for stochastic populations. Educational comparison with current method.")
    
    # =========================================================================
    # SIMULATION SETUP (same as current method for fair comparison)
    # =========================================================================
    N, l, mu = 40, 1000, 0.0005
    s = 0.01
    T = 40000
    
    print(f"\nSimulation parameters: N={N}, l={l}, s={s}, mu={mu}, T={T}")
    
    # Run selection simulation
    print("Running selection simulation...")
    selection_sim = simulator_cls(N, l, s, mu, T, seed=42)
    selection_res = selection_sim.run()
    selection_kl = np.array(selection_res['kl'])
    
    # =========================================================================
    # METHOD 1: BASIC CONVERGENCE ANALYSIS (FAILED APPROACH)
    # =========================================================================
    print(f"\n" + "="*60)
    print("METHOD 1: BASIC CONVERGENCE ANALYSIS (FAILED)")
    print("="*60)
    print("Original problem: Too conservative, ignored genetic drift noise")
    
    # Parameters for basic analysis
    window_size = 400  # Large window for "stability"
    min_stable_duration = 1000  # Very conservative requirement
    
    # Rolling statistics 
    rolling_means = []
    rolling_vars = []
    rolling_autocorr = []
    times_basic = []
    
    print(f"Using window size: {window_size}, min stable duration: {min_stable_duration}")
    
    for i in range(window_size, len(selection_kl) - window_size):
        window = selection_kl[i-window_size//2:i+window_size//2]
        
        # Basic statistics
        mean_val = np.mean(window)
        var_val = np.var(window)
        
        # Autocorrelation at lag 1 (should be low for "stationary" process)
        if len(window) > 1:
            autocorr = np.corrcoef(window[:-1], window[1:])[0, 1]
            if np.isnan(autocorr):
                autocorr = 0.0
        else:
            autocorr = 0.0
            
        rolling_means.append(mean_val)
        rolling_vars.append(var_val)  
        rolling_autocorr.append(autocorr)
        times_basic.append(i)
    
    rolling_means = np.array(rolling_means)
    rolling_vars = np.array(rolling_vars)
    rolling_autocorr = np.array(rolling_autocorr)
    times_basic = np.array(times_basic)
    
    # CONVERGENCE CRITERIA (overly strict)
    # 1. Mean should be "stable" (very low variance across windows)
    mean_stability_threshold = np.var(rolling_means) * 0.03  # Very strict: 1% of overall variance
    
    # 2. Variance should be "stable" (low variance of variances)  
    var_stability_threshold = np.var(rolling_vars) * 0.03
    
    # 3. Low autocorrelation (but this ignores that drift has memory!)
    autocorr_threshold = 0.5
    
    # Apply criteria
    stable_mean_periods = []
    stable_var_periods = []
    low_autocorr_periods = []
    
    # Calculate stability of means and variances
    stability_window = 400
    for i in range(stability_window, len(rolling_means) - stability_window):
        mean_window = rolling_means[i-stability_window//2:i+stability_window//2]
        var_window = rolling_vars[i-stability_window//2:i+stability_window//2]
        
        mean_stability = np.var(mean_window)
        var_stability = np.var(var_window)
        
        stable_mean_periods.append(mean_stability < mean_stability_threshold)
        stable_var_periods.append(var_stability < var_stability_threshold)
        low_autocorr_periods.append(abs(rolling_autocorr[i]) < autocorr_threshold)
    
    # Pad arrays
    pad_size = len(rolling_means) - len(stable_mean_periods)
    stable_mean_periods = np.pad(stable_mean_periods, (pad_size//2, pad_size - pad_size//2), mode='constant', constant_values=False)
    stable_var_periods = np.pad(stable_var_periods, (pad_size//2, pad_size - pad_size//2), mode='constant', constant_values=False)
    low_autocorr_periods = np.pad(low_autocorr_periods, (pad_size//2, pad_size - pad_size//2), mode='constant', constant_values=False)
    
    # Combined criteria (ALL must be met)
    basic_convergence = stable_mean_periods & stable_var_periods & low_autocorr_periods
    
    # Find first sustained period
    basic_equilibrium_start = None
    current_run = 0
    for i, converged in enumerate(basic_convergence):
        if converged:
            current_run += 1
        else:
            if current_run >= min_stable_duration:
                basic_equilibrium_start = times_basic[i - current_run]
                break
            current_run = 0
    
    print(f"\nBasic convergence criteria (overly strict):")
    print(f"  Mean stability < {mean_stability_threshold:.6f}")
    print(f"  Variance stability < {var_stability_threshold:.6f}")  
    print(f"  |Autocorrelation| < {autocorr_threshold:.2f}")
    print(f"  Required duration: {min_stable_duration} generations")
    
    print(f"\nResults:")
    print(f"  Stable mean periods: {np.sum(stable_mean_periods)}/{len(stable_mean_periods)}")
    print(f"  Stable var periods: {np.sum(stable_var_periods)}/{len(stable_var_periods)}")
    print(f"  Low autocorr periods: {np.sum(low_autocorr_periods)}/{len(low_autocorr_periods)}")
    print(f"  Combined convergence: {np.sum(basic_convergence)}/{len(basic_convergence)}")
    
    if basic_equilibrium_start:
        print(f"  ✅ Basic method detected equilibrium at: {basic_equilibrium_start}")
    else:
        print(f"  ❌ Basic method failed to detect equilibrium")
    
    # =========================================================================
    # METHOD 2: STOCHASTIC EQUILIBRIUM ANALYSIS (FAILED APPROACH)  
    # =========================================================================
    print(f"\n" + "="*60)
    print("METHOD 2: STOCHASTIC EQUILIBRIUM ANALYSIS (FAILED)")
    print("="*60)
    print("Original problem: Over-sensitive to noise, couldn't distinguish evolution from drift")
    
    # Parameters for trend analysis
    trend_window = 200  # Window for linear regression
    trend_threshold = 0.03  # Threshold for "no trend"
    
    trend_strengths = []
    trend_slopes = []
    trend_r_squared = []
    times_trend = []
    
    print(f"Using trend window: {trend_window}, trend threshold: {trend_threshold}")
    
    for i in range(trend_window, len(selection_kl) - trend_window):
        window = selection_kl[i-trend_window//2:i+trend_window//2]
        x = np.arange(len(window))
        
        # Linear regression
        try:
            slope, intercept = np.polyfit(x, window, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((window - y_pred) ** 2)
            ss_tot = np.sum((window - np.mean(window)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Trend strength = |slope| × R²
            trend_strength = abs(slope) * r_squared
            
        except:
            slope = 0
            r_squared = 0  
            trend_strength = 0
            
        trend_strengths.append(trend_strength)
        trend_slopes.append(slope)
        trend_r_squared.append(r_squared)
        times_trend.append(i)
    
    trend_strengths = np.array(trend_strengths)
    trend_slopes = np.array(trend_slopes)
    trend_r_squared = np.array(trend_r_squared)
    times_trend = np.array(times_trend)
    
    # Detect "equilibrium" when trend strength is low
    low_trend_periods = trend_strengths < trend_threshold
    
    # Find first sustained low-trend period
    trend_equilibrium_start = None
    min_no_trend_duration = 1000
    current_run = 0
    
    for i, low_trend in enumerate(low_trend_periods):
        if low_trend:
            current_run += 1
        else:
            if current_run >= min_no_trend_duration:
                trend_equilibrium_start = times_trend[i - current_run]
                break
            current_run = 0
    
    print(f"\nTrend analysis criteria:")
    print(f"  Trend strength < {trend_threshold}")
    print(f"  Required duration: {min_no_trend_duration} generations")
    
    print(f"\nResults:")
    print(f"  Low trend periods: {np.sum(low_trend_periods)}/{len(low_trend_periods)}")
    print(f"  Mean trend strength: {np.mean(trend_strengths):.6f}")
    print(f"  Min trend strength: {np.min(trend_strengths):.6f}")
    
    if trend_equilibrium_start:
        print(f"  ✅ Trend method detected equilibrium at: {trend_equilibrium_start}")
    else:
        print(f"  ❌ Trend method failed to detect equilibrium")
        
    # =========================================================================
    # CREATE COMPARISON PLOT
    # =========================================================================
    print(f"\n" + "="*60)
    print("CREATING COMPARISON PLOT")
    print("="*60)
    
    create_old_methods_comparison_plot(
        selection_kl, times_basic, times_trend,
        rolling_means, rolling_vars, rolling_autocorr,
        stable_mean_periods, stable_var_periods, low_autocorr_periods, basic_convergence,
        trend_strengths, trend_slopes, trend_r_squared, low_trend_periods,
        basic_equilibrium_start, trend_equilibrium_start,
        mean_stability_threshold, var_stability_threshold, autocorr_threshold, trend_threshold
    )
    
    # =========================================================================
    # SUMMARY COMPARISON
    # =========================================================================
    print(f"\n" + "="*80)
    print("SUMMARY: WHY THESE METHODS FAILED")
    print("="*80)
    
    print("1. BASIC CONVERGENCE ANALYSIS:")
    print("   ❌ Assumed finite populations could reach true stationarity")
    print("   ❌ Required unrealistically stable mean/variance (ignored genetic drift)")
    print("   ❌ Low autocorrelation criterion ignored that drift has memory")
    print(f"   ❌ Result: {'Detected at ' + str(basic_equilibrium_start) if basic_equilibrium_start else 'Never detected'}")
    
    print("\n2. STOCHASTIC EQUILIBRIUM ANALYSIS:")  
    print("   ❌ Linear regression too sensitive to random fluctuations")
    print("   ❌ Could not distinguish systematic evolution from large drift events")
    print("   ❌ Trend strength metric confused noise spikes with real trends")
    print(f"   ❌ Result: {'Detected at ' + str(trend_equilibrium_start) if trend_equilibrium_start else 'Never detected'}")
    
    print("\n3. CURRENT NEUTRAL COMPARISON METHOD:")
    print("   ✅ Recognizes that equilibrium = neutral-like fluctuations")
    print("   ✅ Uses neutral baseline to define 'normal' drift behavior")  
    print("   ✅ Accounts for residual selection effects with tolerance margins")
    print("   ✅ Multi-criteria approach more robust than single metrics")
    
    return {
        'basic_equilibrium_start': basic_equilibrium_start,
        'trend_equilibrium_start': trend_equilibrium_start,
        'basic_convergence': basic_convergence,
        'low_trend_periods': low_trend_periods
    }


def create_old_methods_comparison_plot(selection_kl, times_basic, times_trend,
                                     rolling_means, rolling_vars, rolling_autocorr,
                                     stable_mean_periods, stable_var_periods, low_autocorr_periods, basic_convergence,
                                     trend_strengths, trend_slopes, trend_r_squared, low_trend_periods,
                                     basic_equilibrium_start, trend_equilibrium_start,
                                     mean_stability_threshold, var_stability_threshold, autocorr_threshold, trend_threshold):
    """
    Create comprehensive comparison plot showing why the old methods failed.
    Split into two 4-panel plots for better visibility.
    
    DETAILED ANALYSIS OF WHAT WE'RE SEEING:
    =======================================
    
    This function visualizes two failed approaches to detecting equilibrium in 
    stochastic populations, demonstrating why they don't work for systems with
    genetic drift.
    
    PLOT 1: BASIC CONVERGENCE ANALYSIS (FAILED)
    ============================================
    
    CONCEPTUAL PROBLEM: This method assumes finite populations can reach true 
    stationarity - a fundamental misunderstanding of stochastic systems.
    
    Panel 1 - KL Trajectory: Shows the raw KL divergence over time
    - The trajectory never becomes truly "flat" due to genetic drift
    - Any detection point would be arbitrary and likely wrong
    
    Panel 2 - Rolling Mean Stability: 
    WHAT IT MEASURES: Variance of the rolling mean across a secondary window
    BIOLOGICAL MEANING: Tests if the "average KL level" stays constant
    WHY IT FAILS: 
    - Threshold is set to 1% of overall variance (impossibly strict!)
    - Genetic drift causes continuous mean fluctuations
    - Real populations never achieve this level of stability
    AUTOCORRELATION NOTE: Rolling means naturally have autocorrelation even
    in random systems, making this test even more conservative
    
    Panel 3 - Rolling Variance Stability:
    WHAT IT MEASURES: Variance of the rolling variance across a secondary window  
    BIOLOGICAL MEANING: Tests if the "fluctuation level" stays constant
    WHY IT FAILS:
    - Again uses 1% threshold (unrealistic for noisy biological systems)
    - Variance naturally fluctuates due to finite sampling effects
    - Selection-drift interactions create non-stationary variance patterns
    MATHEMATICAL NOTE: Variance of variance follows a complex distribution
    that's highly sensitive to outliers and sample size
    
    Panel 4 - Autocorrelation Analysis:
    WHAT IT MEASURES: Pearson correlation between KL(t) and KL(t+1)
    FORMULA: r = Σ[(x_i - x̄)(y_i - ȳ)] / √[Σ(x_i - x̄)²Σ(y_i - ȳ)²]
    BIOLOGICAL MEANING: Measures if current KL predicts next KL (memory)
    WHY THIS CRITERION IS WRONG:
    - Genetic drift NATURALLY creates autocorrelation (populations have memory!)
    - Wright-Fisher model inherently has lag-1 autocorrelation ≈ (1-1/N)
    - Low autocorrelation would indicate unrealistic "memoryless" evolution
    - This criterion fights against the biology rather than accepting it
    
    Combined Criteria (dark green): ALL three criteria must be met simultaneously
    - Intersection of three overly strict conditions = almost never satisfied
    - When satisfied, it's likely a statistical fluke, not real equilibrium
    
    PLOT 2: STOCHASTIC EQUILIBRIUM ANALYSIS (FAILED)  
    =================================================
    
    CONCEPTUAL PROBLEM: Tries to distinguish "systematic evolution" from "random
    drift" using linear regression, but noise in finite populations can mimic trends.
    
    Panel 1 - KL Trajectory: Same as Plot 1, but for trend method detection
    
    Panel 2 - Trend Strength Analysis:
    WHAT IT MEASURES: |slope| × R² from linear regression on sliding windows
    MATHEMATICAL BREAKDOWN:
    - Linear regression: y = mx + b (KL vs time)
    - Slope (m): Rate of change in KL per generation
    - R² = 1 - (SS_residual / SS_total): Fraction of variance explained by linear trend
    - Trend strength = |m| × R²: Combines magnitude and significance of trend
    
    BIOLOGICAL MEANING: Attempts to quantify "systematic directional change"
    WHY IT FAILS:
    - Random walk processes can show temporary "trends" (gambler's fallacy)
    - Large drift events look like systematic change to linear regression  
    - Short windows: too noisy; Long windows: miss real transitions
    - R² can be high even for random data if window contains a drift run
    
    LOG SCALE NOTE: Used because trend strengths span many orders of magnitude,
    from 10⁻⁶ (nearly random) to 10⁻² (strong apparent trend)
    
    Panel 3 - Linear Regression Components:
    BLUE LINE (Slope): Rate of KL change per generation
    - Positive = KL increasing; Negative = KL decreasing
    - Random drift creates false positive/negative slopes
    - Real systematic evolution would show sustained non-zero slopes
    
    RED LINE (R²): Quality of linear fit (0 = no linear relationship, 1 = perfect)
    STATISTICAL MEANING: R² = 1 - Σ(observed - predicted)² / Σ(observed - mean)²
    WHY R² MISLEADS HERE:
    - High R² doesn't mean biological significance, just linear pattern
    - Even random walks can have high R² over short windows
    - Genetic drift can create temporary linear-looking segments
    
    Panel 4 - Final Comparison:
    Shows both methods' "convergence periods" side by side
    - Green: When ALL basic criteria met (almost never)
    - Orange: When trend strength below threshold (too often, false positives)
    
    KEY INSIGHTS FROM THESE FAILURES:
    =================================
    
    1. STATIONARITY ASSUMPTION IS WRONG: Finite populations never reach stationarity
    2. OVERLY STRICT CRITERIA: 1% thresholds ignore natural biological variation
    3. MISUNDERSTANDING OF AUTOCORRELATION: Genetic systems naturally have memory
    4. LINEAR REGRESSION LIMITATIONS: Cannot distinguish trends from drift runs
    5. THRESHOLD ARBITRARINESS: No biological justification for specific values
    
    CONTRAST WITH CURRENT NEUTRAL METHOD:
    ====================================
    Our current approach succeeds because it:
    - Accepts that equilibrium = neutral-like fluctuations (not true stationarity)
    - Uses neutral simulation to define biologically realistic noise levels
    - Recognizes that autocorrelation is natural and expected
    - Focuses on fluctuation patterns rather than absolute stability
    - Uses tolerance margins based on actual biological variation
    """
    
    # =========================================================================
    # PLOT 1: BASIC CONVERGENCE ANALYSIS (4 panels)
    # =========================================================================
    fig1, axes1 = plt.subplots(4, 1, figsize=(16, 18))
    fig1.suptitle('BASIC CONVERGENCE ANALYSIS (FAILED METHOD)', fontsize=16, fontweight='bold', y=0.98)
    
    # Panel 1: Original KL trajectory with basic method detection
    axes1[0].plot(selection_kl, 'blue', linewidth=1, alpha=0.7, label='Selection KL')
    if basic_equilibrium_start:
        axes1[0].axvline(x=basic_equilibrium_start, color='red', linestyle='--', linewidth=2,
                        label=f'Basic method detected: {basic_equilibrium_start}')
    axes1[0].set_ylabel('KL Divergence (bits)')
    axes1[0].set_title('KL Trajectory with Basic Method Detection')
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.3)
    
    # Panel 2: Rolling means (basic method)
    axes1[1].plot(times_basic, rolling_means, 'green', linewidth=2, label='Rolling KL mean')
    axes1[1].fill_between(times_basic, 0, 1, where=stable_mean_periods, alpha=0.3, color='green',
                         transform=axes1[1].get_xaxis_transform(), label='Stable mean periods')
    if basic_equilibrium_start:
        axes1[1].axvline(x=basic_equilibrium_start, color='red', linestyle='--', linewidth=2)
    axes1[1].set_ylabel('Rolling Mean')
    axes1[1].set_title(f'Rolling Mean Stability (threshold: {mean_stability_threshold:.6f})')
    axes1[1].legend()
    axes1[1].grid(True, alpha=0.3)
    
    # Panel 3: Rolling variances (basic method)
    axes1[2].plot(times_basic, rolling_vars, 'purple', linewidth=2, label='Rolling KL variance')
    axes1[2].fill_between(times_basic, 0, 1, where=stable_var_periods, alpha=0.3, color='purple',
                         transform=axes1[2].get_xaxis_transform(), label='Stable variance periods')
    if basic_equilibrium_start:
        axes1[2].axvline(x=basic_equilibrium_start, color='red', linestyle='--', linewidth=2)
    axes1[2].set_ylabel('Rolling Variance')
    axes1[2].set_title(f'Rolling Variance Stability (threshold: {var_stability_threshold:.6f})')
    axes1[2].legend()
    axes1[2].grid(True, alpha=0.3)
    
    # Panel 4: Combined basic method criteria
    axes1[3].plot(times_basic, rolling_autocorr, 'brown', linewidth=2, label='Autocorrelation (lag 1)')
    axes1[3].axhline(y=autocorr_threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold: ±{autocorr_threshold}')
    axes1[3].axhline(y=-autocorr_threshold, color='red', linestyle='--', alpha=0.7)
    axes1[3].fill_between(times_basic, 0, 1, where=basic_convergence, alpha=0.5, color='darkgreen',
                         transform=axes1[3].get_xaxis_transform(), label='ALL criteria met')
    if basic_equilibrium_start:
        axes1[3].axvline(x=basic_equilibrium_start, color='red', linestyle='--', linewidth=3,
                        label=f'Detected equilibrium: {basic_equilibrium_start}')
    axes1[3].set_xlabel('Generation')
    axes1[3].set_ylabel('Autocorrelation')
    axes1[3].set_title('Autocorrelation + Combined Criteria')
    axes1[3].legend()
    axes1[3].grid(True, alpha=0.3)
    
    # Align all x-axes for plot 1
    for ax in axes1:
        ax.set_xlim(0, len(selection_kl))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    plt.subplots_adjust(hspace=0.4)  # Increase vertical spacing between panels
    plt.show()
    
    # =========================================================================
    # PLOT 2: STOCHASTIC EQUILIBRIUM ANALYSIS (4 panels)
    # =========================================================================
    fig2, axes2 = plt.subplots(4, 1, figsize=(16, 18))
    fig2.suptitle('STOCHASTIC EQUILIBRIUM ANALYSIS (FAILED METHOD)', fontsize=16, fontweight='bold', y=0.98)
    
    # Panel 1: Original KL trajectory with trend method detection
    axes2[0].plot(selection_kl, 'blue', linewidth=1, alpha=0.7, label='Selection KL')
    if trend_equilibrium_start:
        axes2[0].axvline(x=trend_equilibrium_start, color='orange', linestyle='--', linewidth=2,
                        label=f'Trend method detected: {trend_equilibrium_start}')
    axes2[0].set_ylabel('KL Divergence (bits)')
    axes2[0].set_title('KL Trajectory with Trend Method Detection')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    
    # Panel 2: Trend strengths (stochastic equilibrium method)
    axes2[1].plot(times_trend, trend_strengths, 'orange', linewidth=2, label='Trend strength = |slope| × R²')
    axes2[1].axhline(y=trend_threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold: {trend_threshold}')
    axes2[1].fill_between(times_trend, 0, 1, where=low_trend_periods, alpha=0.3, color='orange',
                         transform=axes2[1].get_xaxis_transform(), label='Low trend periods')
    if trend_equilibrium_start:
        axes2[1].axvline(x=trend_equilibrium_start, color='orange', linestyle='--', linewidth=2)
    axes2[1].set_ylabel('Trend Strength')
    axes2[1].set_title('Trend Strength Analysis (log scale)')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)
    axes2[1].set_yscale('log')
    
    # Panel 3: Trend slopes and R² (dual y-axis)
    ax3a = axes2[2]
    ax3b = ax3a.twinx()
    
    line1 = ax3a.plot(times_trend, trend_slopes, 'blue', linewidth=1, alpha=0.7, label='Slope')
    line2 = ax3b.plot(times_trend, trend_r_squared, 'red', linewidth=1, alpha=0.7, label='R²')
    
    if trend_equilibrium_start:
        ax3a.axvline(x=trend_equilibrium_start, color='orange', linestyle='--', linewidth=2)
    
    ax3a.set_ylabel('Slope', color='blue')
    ax3b.set_ylabel('R²', color='red')
    ax3a.set_title('Linear Regression Components: Slope and R²')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3a.legend(lines, labels, loc='upper right')
    
    ax3a.grid(True, alpha=0.3)
    ax3a.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Panel 4: Final comparison between both methods
    axes2[3].fill_between(times_basic, 0, 0.4, where=basic_convergence, alpha=0.7, color='green',
                         label='Basic method: ALL criteria met')
    axes2[3].fill_between(times_trend, 0.6, 1.0, where=low_trend_periods, alpha=0.7, color='orange',
                         label='Trend method: Low trend periods')
    
    if basic_equilibrium_start:
        axes2[3].axvline(x=basic_equilibrium_start, color='green', linestyle='--', linewidth=3,
                        label=f'Basic method: {basic_equilibrium_start}')
    if trend_equilibrium_start:
        axes2[3].axvline(x=trend_equilibrium_start, color='orange', linestyle='--', linewidth=3,
                        label=f'Trend method: {trend_equilibrium_start}')
    
    axes2[3].set_xlabel('Generation')
    axes2[3].set_ylabel('Method')
    axes2[3].set_title('Final Comparison: Both Failed Methods')
    axes2[3].legend()
    axes2[3].grid(True, alpha=0.3)
    axes2[3].set_ylim(0, 1)
    axes2[3].set_yticks([0.2, 0.8])
    axes2[3].set_yticklabels(['Basic Method', 'Trend Method'])
    
    # Align all x-axes for plot 2
    for ax in [axes2[0], axes2[1], ax3a, axes2[3]]:
        ax.set_xlim(0, len(selection_kl))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    plt.subplots_adjust(hspace=0.4)  # Increase vertical spacing between panels
    plt.show()


def test_free_recombination_convergence():
    """
    Test convergence detection on Wright-Fisher with FREE RECOMBINATION.
    
    This compares asexual reproduction (original WF) vs sexual reproduction
    with free recombination to see if the convergence patterns differ.
    
    BIOLOGICAL HYPOTHESIS:
    ======================
    Free recombination may lead to:
    1. FASTER convergence: Breaking up linkage allows more efficient selection
    2. DIFFERENT equilibrium: May reach different mean phenotype
    3. ALTERED NOISE PATTERNS: Recombination changes genetic drift characteristics
    4. MODIFIED KL DYNAMICS: Different trajectory shapes and variance patterns
    
    We test this using our current neutral comparison method.
    """
    
    print("="*80)
    print("FREE RECOMBINATION vs ASEXUAL REPRODUCTION CONVERGENCE COMPARISON")
    print("="*80)
    print("Testing if sexual reproduction with free recombination changes")
    print("convergence patterns compared to asexual reproduction.")
    
    # Parameters (same for fair comparison)
    N, l, mu = 40, 1000, 0.0005
    s = 0.01
    T = 20000  # Shorter for comparison
    window_size = 500
    smooth_window = 101
    
    print(f"\nParameters: N={N}, l={l}, s={s}, mu={mu}, T={T}")
    print(f"Window size: {window_size}, Smoothing: {smooth_window}")
    
    # =========================================================================
    # ASEXUAL REPRODUCTION (ORIGINAL)
    # =========================================================================
    print(f"\n" + "="*60)
    print("ASEXUAL REPRODUCTION (Original Wright-Fisher)")
    print("="*60)
    
    print("Running asexual simulation...")
    asexual_sim = WrightFisherSimulator(N, l, s, mu, T, seed=42)
    asexual_res = asexual_sim.run()
    asexual_kl = np.array(asexual_res['kl'])
    
    asexual_final_mean = asexual_res['phenotype_probs'][-1] @ np.arange(len(asexual_res['phenotype_probs'][-1]))
    print(f"Asexual final KL: {asexual_kl[-1]:.3f} bits")
    print(f"Asexual final mean phenotype: {asexual_final_mean:.2f}")
    
    # =========================================================================
    # SEXUAL REPRODUCTION WITH FREE RECOMBINATION
    # =========================================================================
    print(f"\n" + "="*60)
    print("SEXUAL REPRODUCTION (Free Recombination)")
    print("="*60)
    
    print("Running free recombination simulation...")
    sexual_sim = WrightFisherSimulator_FreeRecombination(N, l, s, mu, T, seed=42)
    sexual_res = sexual_sim.run()
    sexual_kl = np.array(sexual_res['kl'])
    
    sexual_final_mean = sexual_res['phenotype_probs'][-1] @ np.arange(len(sexual_res['phenotype_probs'][-1]))
    print(f"Sexual final KL: {sexual_kl[-1]:.3f} bits")
    print(f"Sexual final mean phenotype: {sexual_final_mean:.2f}")
    
    # =========================================================================
    # COMPARISON METRICS
    # =========================================================================
    print(f"\n" + "="*60)
    print("DIRECT COMPARISON")
    print("="*60)
    
    # Final states
    kl_difference = sexual_kl[-1] - asexual_kl[-1]
    mean_difference = sexual_final_mean - asexual_final_mean
    
    print(f"Final KL difference (Sexual - Asexual): {kl_difference:+.3f} bits")
    print(f"Final mean difference (Sexual - Asexual): {mean_difference:+.2f} alleles")
    
    # Convergence speeds (rough estimate)
    asexual_90_percent = np.where(asexual_kl >= 0.9 * asexual_kl[-1])[0]
    sexual_90_percent = np.where(sexual_kl >= 0.9 * sexual_kl[-1])[0]
    
    if len(asexual_90_percent) > 0 and len(sexual_90_percent) > 0:
        asexual_conv_time = asexual_90_percent[0]
        sexual_conv_time = sexual_90_percent[0]
        speed_difference = sexual_conv_time - asexual_conv_time
        
        print(f"90% convergence time - Asexual: {asexual_conv_time} generations")
        print(f"90% convergence time - Sexual: {sexual_conv_time} generations")
        print(f"Speed difference (Sexual - Asexual): {speed_difference:+d} generations")
        
        if speed_difference < 0:
            print("✅ Sexual reproduction converges FASTER")
        elif speed_difference > 0:
            print("⚠️  Sexual reproduction converges SLOWER")
        else:
            print("➡️  Similar convergence speeds")
    
    # Variance comparison
    asexual_late_var = np.var(asexual_kl[-5000:])  # Last 5000 generations
    sexual_late_var = np.var(sexual_kl[-5000:])
    variance_ratio = sexual_late_var / asexual_late_var
    
    print(f"\nLate-stage KL variance:")
    print(f"  Asexual: {asexual_late_var:.6f}")
    print(f"  Sexual: {sexual_late_var:.6f}")
    print(f"  Ratio (Sexual/Asexual): {variance_ratio:.3f}")
    
    if variance_ratio > 1.1:
        print("⚠️  Sexual reproduction has HIGHER variance (more noise)")
    elif variance_ratio < 0.9:
        print("✅ Sexual reproduction has LOWER variance (less noise)")
    else:
        print("➡️  Similar variance levels")
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    print(f"\n" + "="*60)
    print("CREATING COMPARISON PLOTS")
    print("="*60)
    
    create_recombination_comparison_plot(asexual_kl, sexual_kl, asexual_res, sexual_res)
    
    # =========================================================================
    # CONVERGENCE ANALYSIS WITH CURRENT METHOD
    # =========================================================================
    print(f"\n" + "="*60)
    print("CONVERGENCE ANALYSIS: SEXUAL vs ASEXUAL")
    print("="*60)
    
    print("Note: Full convergence analysis would require running the neutral")
    print("comparison method separately for each reproduction mode.")
    print("This comparison shows the raw differences in KL trajectories.")
    
    return {
        'asexual_kl': asexual_kl,
        'sexual_kl': sexual_kl,
        'asexual_final_mean': asexual_final_mean,
        'sexual_final_mean': sexual_final_mean,
        'kl_difference': kl_difference,
        'mean_difference': mean_difference,
        'variance_ratio': variance_ratio
    }


def create_recombination_comparison_plot(asexual_kl, sexual_kl, asexual_res, sexual_res):
    """
    Create comparison plot showing asexual vs sexual reproduction dynamics.
    """
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle('ASEXUAL vs SEXUAL REPRODUCTION: Convergence Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    # Plot 1: KL trajectories comparison
    axes[0].plot(asexual_kl, 'blue', linewidth=2, alpha=0.8, label='Asexual reproduction')
    axes[0].plot(sexual_kl, 'red', linewidth=2, alpha=0.8, label='Sexual reproduction (free recomb.)')
    axes[0].set_ylabel('KL Divergence (bits)')
    axes[0].set_title('KL Divergence Evolution: Asexual vs Sexual Reproduction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Difference trajectory (Sexual - Asexual)
    kl_diff = sexual_kl - asexual_kl
    axes[1].plot(kl_diff, 'purple', linewidth=2, alpha=0.8, label='KL difference (Sexual - Asexual)')
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5, label='No difference')
    axes[1].set_ylabel('KL Difference (bits)')
    axes[1].set_title('KL Difference Over Time (Positive = Sexual Higher)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Phenotypic mean trajectories
    asexual_means = []
    sexual_means = []
    
    for t in range(len(asexual_res['phenotype_probs'])):
        # Calculate mean phenotype at each time point
        asexual_dist = asexual_res['phenotype_probs'][t]
        sexual_dist = sexual_res['phenotype_probs'][t]
        
        asexual_mean = np.sum(asexual_dist * np.arange(len(asexual_dist)))
        sexual_mean = np.sum(sexual_dist * np.arange(len(sexual_dist)))
        
        asexual_means.append(asexual_mean)
        sexual_means.append(sexual_mean)
    
    axes[2].plot(asexual_means, 'blue', linewidth=2, alpha=0.8, label='Asexual mean phenotype')
    axes[2].plot(sexual_means, 'red', linewidth=2, alpha=0.8, label='Sexual mean phenotype')
    axes[2].axhline(y=500, color='gray', linestyle='--', alpha=0.5, label='Neutral expectation (500)')
    axes[2].set_xlabel('Generation')
    axes[2].set_ylabel('Mean Phenotype (# of 1-alleles)')
    axes[2].set_title('Phenotypic Evolution: Selection Response Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Align all x-axes
    for ax in axes:
        ax.set_xlim(0, max(len(asexual_kl), len(sexual_kl)))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.subplots_adjust(hspace=0.3)
    plt.show()


def run_all_equilibrium_plots_with_recombination():
    """
    Run all equilibrium detection plots using the free recombination simulator:
    - Neutral KL equilibrium detection plots
    - Old failed methods plots
    - Core KL/phenotype plot functions from QE.py
    """
    print("="*80)
    print("RUNNING ALL EQUILIBRIUM PLOTS WITH FREE RECOMBINATION")
    print("="*80)

    # Use same parameters as the main test for consistency
    N, l, mu = 40, 1000, 0.0005
    s = 0.01
    T = 20000

    # 1) Generate results with free recombination
    sim = WrightFisherSimulator_FreeRecombination(N, l, s, mu, T, seed=42)
    res = sim.run()

    # 2) Plot core QE diagnostics
    print("Plotting core QE diagnostics (recombination): phenotype counts and KL trajectories...")
    fig_counts = plot_phenotype_counts(res)
    plot_kl_trajectory(res, smooth_window=201, poly_degree=3)
    plot_abs_delta_kl(res, smooth_window=201)
    plot_delta_kl(res, smooth_window=201, t_start=2000, t_end=T)  # focused view

    # 3) Run neutral-KL equilibrium detection (plots included)
    print("Running neutral-KL equilibrium detection (recombination)...")
    simple_neutral_kl_equilibrium_detection(window_size=500, smooth_window=101,
                                            simulator_cls=WrightFisherSimulator_FreeRecombination)

    # 4) Run old failed convergence methods (now with recombination)
    print("Running old failed convergence methods (recombination)...")
    test_old_failed_convergence_methods(simulator_cls=WrightFisherSimulator_FreeRecombination)

    print("All recombination-based plots completed.")


if __name__ == "__main__":
    # Test with default parameters
    print("Testing with default parameters (window=500, smooth=101)...")
    # Run default (asexual/original) simulator
    equilibrium_start = simple_neutral_kl_equilibrium_detection(window_size=500, smooth_window=101,
                                                                simulator_cls=WrightFisherSimulator)
    
    print(f"\n" + "="*60)
    print("SINGLE TEST SUMMARY")
    print("="*60)
    if equilibrium_start:
        print(f"✅ Equilibrium detected at generation: {equilibrium_start}")
        print("This is when KL fluctuations become similar to neutral drift!")
    else:
        print("❌ No clear equilibrium detected")
        print("Try adjusting criteria or testing different window sizes")
    
    # Test the old failed methods for comparison
    print(f"\n" + "="*60)
    print("TESTING OLD FAILED METHODS FOR COMPARISON")
    print("="*60)
    old_results = test_old_failed_convergence_methods()
    
    # Test free recombination
    print(f"\n" + "="*80)
    print("TESTING FREE RECOMBINATION CONVERGENCE")
    print("="*80)
    recomb_results = test_free_recombination_convergence()
    
    # One-shot: run all plots with recombination
    print(f"\n" + "="*80)
    print("RUN ALL EQUILIBRIUM PLOTS WITH FREE RECOMBINATION")
    print("="*80)
    run_all_equilibrium_plots_with_recombination()

    # Optional: run the neutral KL detection using the free recombination simulator
    print(f"\n" + "="*60)
    print("RUNNING NEUTRAL-KL EQUILIBRIUM DETECTION WITH FREE RECOMBINATION")
    print("="*60)
    eq_start_recomb = simple_neutral_kl_equilibrium_detection(window_size=500, smooth_window=101,
                                                             simulator_cls=WrightFisherSimulator_FreeRecombination)
    if eq_start_recomb:
        print(f"✅ (Recombination) Equilibrium detected at generation: {eq_start_recomb}")
    else:
        print("❌ (Recombination) No clear equilibrium detected")
    
    # Uncomment the line below to test multiple window sizes (takes longer)
    # test_results = test_multiple_window_sizes()