#!/usr/bin/env python3
"""
New convergence analysis focusing on:
1. Multiple Wright-Fisher simulations with free recombination
2. Autocorrelation analysis at different time points
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from QE import WrightFisherSimulator_FreeRecombination
from scipy.stats import binom
from typing import List, Dict, Tuple
import os

def run_multiple_simulations(
    N: int = 40,
    l: int = 1000,
    s: float = 0.01,
    mu: float = 0.0005,
    T: int = 10000,
    n_sims: int = 30,
    base_seed: int = 42,
    results_dir: str = "convergence_results",
    case: str = "selection"
) -> List[Dict]:
    """Run multiple Wright-Fisher simulations with free recombination."""
    print(f"Running {n_sims} simulations with T={T} generations each...")
    results = []
    
    # Try to load existing results if any
    checkpoint_file = f"{results_dir}/checkpoint_{case}_sims.npz"
    if os.path.exists(checkpoint_file):
        print(f"Found checkpoint file. Loading existing results...")
        checkpoint = np.load(checkpoint_file, allow_pickle=True)
        results = list(checkpoint['results'])
        start_sim = len(results)
        print(f"Loaded {start_sim} existing simulations. Continuing from simulation {start_sim + 1}...")
    else:
        start_sim = 0
    
    try:
        for i in range(start_sim, n_sims):
            print(f"Running simulation {i+1}/{n_sims}...")
            sim = WrightFisherSimulator_FreeRecombination(
                N=N, l=l, s=s, mu=mu, T=T, seed=base_seed + i
            )
            results.append(sim.run())
            
            # Save checkpoint after each simulation
            checkpoint_data = {
                'results': results,
                'params': {
                    'N': N, 'l': l, 's': s, 'mu': mu,
                    'T': T, 'n_sims': n_sims, 'current_sim': i+1
                }
            }
            np.savez_compressed(checkpoint_file, **checkpoint_data)
            print(f"Checkpoint saved after simulation {i+1}")
    
    except Exception as e:
        print(f"\nError occurred during simulation {len(results) + 1}: {str(e)}")
        print("Saving current progress before exiting...")
        if results:  # Save what we have if there's anything
            checkpoint_data = {
                'results': results,
                'params': {
                    'N': N, 'l': l, 's': s, 'mu': mu,
                    'T': T, 'n_sims': n_sims, 'current_sim': len(results)
                }
            }
            np.savez_compressed(checkpoint_file, **checkpoint_data)
        raise  # Re-raise the exception after saving
    
    return results

def compute_average_dynamics(results: List[Dict]) -> Dict:
    """Compute average KL and phenotype dynamics across simulations."""
    n_sims = len(results)
    T = len(results[0]['kl'])
    l = results[0]['l']
    
    # Initialize arrays
    kl_matrix = np.zeros((n_sims, T))
    pheno_matrix = np.zeros((n_sims, T, l + 1))
    pheno_means = np.zeros((n_sims, T))  # Track mean phenotype
    pheno_vars = np.zeros((n_sims, T))   # Track phenotype variance
    
    # Fill matrices
    for i, res in enumerate(results):
        kl_matrix[i, :] = res['kl']
        pheno_matrix[i, :, :] = res['phenotype_probs']
        
        # Compute mean and variance of phenotype distribution at each time
        for t in range(T):
            x = np.arange(l + 1)
            p = res['phenotype_probs'][t]
            pheno_means[i, t] = np.sum(x * p)  # mean
            pheno_vars[i, t] = np.sum(x**2 * p) - pheno_means[i, t]**2  # variance
    
    # Compute statistics
    kl_mean = np.mean(kl_matrix, axis=0)
    kl_std = np.std(kl_matrix, axis=0)
    pheno_mean = np.mean(pheno_matrix, axis=0)
    pheno_std = np.std(pheno_matrix, axis=0)
    
    # Statistics of phenotype mean and variance
    pheno_mean_mean = np.mean(pheno_means, axis=0)  # mean of means
    pheno_mean_std = np.std(pheno_means, axis=0)    # std of means
    pheno_var_mean = np.mean(pheno_vars, axis=0)    # mean of variances
    pheno_var_std = np.std(pheno_vars, axis=0)      # std of variances
    
    return {
        'kl_mean': kl_mean,
        'kl_std': kl_std,
        'pheno_mean': pheno_mean,
        'pheno_std': pheno_std,
        'kl_matrix': kl_matrix,
        'pheno_matrix': pheno_matrix,
        'pheno_means': pheno_means,
        'pheno_vars': pheno_vars,
        'pheno_mean_mean': pheno_mean_mean,
        'pheno_mean_std': pheno_mean_std,
        'pheno_var_mean': pheno_var_mean,
        'pheno_var_std': pheno_var_std
    }

def compute_autocorrelation(series: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation up to max_lag."""
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    
    # Normalize series
    norm_series = series - mean
    
    # Compute autocorrelation for each lag
    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        # Correlation between series and lagged series
        acf[lag] = np.sum(norm_series[lag:] * norm_series[:(n-lag)]) / ((n - lag) * var)
    
    return acf

def analyze_autocorrelation_at_points(
    kl_matrix: np.ndarray,
    start_points: List[int],
    series_length: int = 5000,
    max_lag: int = 1000
) -> Dict:
    """Analyze autocorrelation at different starting points."""
    n_sims = kl_matrix.shape[0]
    n_points = len(start_points)
    
    # Initialize storage
    acf_matrix = np.zeros((n_points, n_sims, max_lag + 1))
    
    # Compute autocorrelation for each simulation at each start point
    for i, start in enumerate(start_points):
        for sim in range(n_sims):
            series = kl_matrix[sim, start:start+series_length]
            acf_matrix[i, sim, :] = compute_autocorrelation(series, max_lag)
    
    # Compute statistics
    acf_mean = np.mean(acf_matrix, axis=1)
    acf_std = np.std(acf_matrix, axis=1)
    
    return {
        'acf_mean': acf_mean,
        'acf_std': acf_std,
        'start_points': start_points,
        'acf_matrix': acf_matrix
    }

def plot_average_dynamics(avg_results: Dict, T: int, l: int, s: float):
    """Plot average KL and phenotype dynamics with confidence bands."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: KL dynamics
    t = np.arange(T + 1)
    ax1.plot(t, avg_results['kl_mean'], 'b-', label='Mean KL')
    ax1.fill_between(
        t,
        avg_results['kl_mean'] - avg_results['kl_std'],
        avg_results['kl_mean'] + avg_results['kl_std'],
        alpha=0.3, color='blue'
    )
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('KL Divergence (bits)')
    ax1.set_title(f'Average KL Dynamics (s={s}, ± 1 std)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Phenotype heatmap
    im = ax2.imshow(
        avg_results['pheno_mean'].T,
        aspect='auto',
        origin='lower',
        extent=[0, T+1, 0, l+1],
        cmap='viridis'
    )
    plt.colorbar(im, ax=ax2, label='Average probability')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Phenotype')
    ax2.set_title(f'Average Phenotype Distribution (s={s})')
    
    # Plot 3: Mean phenotype dynamics
    ax3.plot(t, avg_results['pheno_mean_mean'], 'g-', label='Mean phenotype')
    ax3.fill_between(
        t,
        avg_results['pheno_mean_mean'] - avg_results['pheno_mean_std'],
        avg_results['pheno_mean_mean'] + avg_results['pheno_mean_std'],
        alpha=0.3, color='green'
    )
    ax3.axhline(y=l/2, color='gray', linestyle='--', alpha=0.5, label='Neutral expectation')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Mean phenotype')
    ax3.set_title(f'Average Mean Phenotype (s={s}, ± 1 std)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Phenotype variance dynamics
    ax4.plot(t, avg_results['pheno_var_mean'], 'r-', label='Phenotype variance')
    ax4.fill_between(
        t,
        avg_results['pheno_var_mean'] - avg_results['pheno_var_std'],
        avg_results['pheno_var_mean'] + avg_results['pheno_var_std'],
        alpha=0.3, color='red'
    )
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Phenotype variance')
    ax4.set_title(f'Average Phenotype Variance (s={s}, ± 1 std)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

def plot_autocorrelation_analysis(acf_results: Dict):
    """Plot autocorrelation analysis results with focus on decay rate comparison."""
    n_points = len(acf_results['start_points'])
    lags = np.arange(acf_results['acf_mean'].shape[1])
    
    # Create figure with a main panel for decay comparison and smaller individual plots
    fig = plt.figure(figsize=(15, 12))
    
    # Main panel for decay comparison (larger)
    ax_main = plt.subplot2grid((n_points+2, 4), (0, 0), rowspan=2, colspan=3)
    colors = sns.color_palette("husl", n_points)
    
    # Plot all decay curves on main panel
    for i, (start, color) in enumerate(zip(acf_results['start_points'], colors)):
        ax_main.plot(lags, acf_results['acf_mean'][i], color=color,
                    label=f'Start t={start}', linewidth=2)
        # Add light confidence bands
        ax_main.fill_between(
            lags,
            acf_results['acf_mean'][i] - acf_results['acf_std'][i],
            acf_results['acf_mean'][i] + acf_results['acf_std'][i],
            alpha=0.1, color=color
        )
    
    ax_main.set_title('Comparison of Autocorrelation Decay Rates', fontsize=12, pad=10)
    ax_main.set_xlabel('Lag k', fontsize=10)
    ax_main.set_ylabel('Autocorrelation', fontsize=10)
    ax_main.grid(True, alpha=0.3)
    ax_main.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax_main.axhline(y=0.2, color='red', linestyle=':', alpha=0.3)
    ax_main.axhline(y=-0.2, color='red', linestyle=':', alpha=0.3)
    ax_main.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    # Smaller individual plots in a column
    for i, (start, color) in enumerate(zip(acf_results['start_points'], colors)):
        ax = plt.subplot2grid((n_points+2, 4), (i+2, 3))
        
        # Plot individual ACF
        ax.plot(lags, acf_results['acf_mean'][i], color=color, linewidth=1)
        ax.fill_between(
            lags,
            acf_results['acf_mean'][i] - acf_results['acf_std'][i],
            acf_results['acf_mean'][i] + acf_results['acf_std'][i],
            alpha=0.2, color=color
        )
        
        ax.set_title(f't={start}', fontsize=8)
        if i == len(acf_results['start_points'])-1:
            ax.set_xlabel('Lag k', fontsize=8)
        ax.set_ylabel('ACF', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.tick_params(labelsize=6)
    
    plt.tight_layout()
    plt.show()

def save_results(results: Dict, filename: str):
    """Save results to a numpy compressed file."""
    np.savez_compressed(filename, **results)

def load_results(filename: str) -> Dict:
    """Load results from a numpy compressed file."""
    data = np.load(filename)
    return {key: data[key] for key in data.files}

def main():
    import time
    start_time = time.time()
    
    print("="*80)
    print("CONVERGENCE ANALYSIS WITH TIMING")
    print("="*80)
    
    # Create results directory if it doesn't exist
    results_dir = "convergence_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Original parameters
    N = 40
    l = 1000
    mu = 0.0005
    T = 20000  # Reduced from 40000 to balance runtime and analysis depth
    n_sims = 100  # Original value
    
    # Store results for both cases
    all_results = {}
    
    # Run both selection and neutral simulations
    for s in [0.01, 0.0]:  # Selection and neutral cases
        case = "selection" if s > 0 else "neutral"
        print(f"\nRunning analysis for s={s}")
        print("="*40)
        
        # Run simulations
        sim_start = time.time()
        results = run_multiple_simulations(N=N, l=l, s=s, mu=mu, T=T, n_sims=n_sims)
        sim_end = time.time()
        print(f"Simulations completed in {sim_end - sim_start:.1f} seconds")
        
        # Save raw simulation results
        raw_data = {
            'results': results,
            'params': {
                'N': N, 'l': l, 's': s, 'mu': mu,
                'T': T, 'n_sims': n_sims
            }
        }
        save_results(raw_data, f"{results_dir}/raw_{case}_data.npz")
        print(f"Raw simulation data saved to {results_dir}/raw_{case}_data.npz")
        
        # Compute average dynamics
        avg_start = time.time()
        avg_results = compute_average_dynamics(results)
        avg_end = time.time()
        print(f"Average dynamics computed in {avg_end - avg_start:.1f} seconds")
        
        # Store results
        all_results[s] = avg_results
        
        # Plot average dynamics
        plot_start = time.time()
        plot_average_dynamics(avg_results, T, l, s)
        plot_end = time.time()
        print(f"Dynamics plots created in {plot_end - plot_start:.1f} seconds")
        
        # Autocorrelation analysis
        acf_start = time.time()
        # Generate points every 2000 generations
        start_points = list(range(1, T+1, 2000))
        acf_results = analyze_autocorrelation_at_points(
            avg_results['kl_matrix'],
            start_points,
            series_length=2000,  # Reduced from 5000
            max_lag=500  # Reduced from 1000
        )
        acf_end = time.time()
        print(f"Autocorrelation analysis completed in {acf_end - acf_start:.1f} seconds")
        
        # Save processed results
        processed_data = {
            'avg_results': avg_results,
            'acf_results': acf_results,
            'params': {
                'N': N, 'l': l, 's': s, 'mu': mu,
                'T': T, 'n_sims': n_sims,
                'series_length': 2000,
                'max_lag': 500,
                'start_points': start_points
            }
        }
        save_results(processed_data, f"{results_dir}/processed_{case}_data.npz")
        print(f"Processed results saved to {results_dir}/processed_{case}_data.npz")
        
        # Plot autocorrelation analysis
        plot_start = time.time()
        plot_autocorrelation_analysis(acf_results)
        plot_end = time.time()
        print(f"Autocorrelation plots created in {plot_end - plot_start:.1f} seconds")
    
    # Compare phenotype statistics between neutral and selected cases
    print("\nComparing phenotype statistics between neutral and selected cases...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    t = np.arange(T + 1)
    
    # Plot mean comparison
    ax1.plot(t, all_results[0.01]['pheno_mean_mean'], 'r-', label='Selected (s=0.01)')
    ax1.fill_between(t, 
                     all_results[0.01]['pheno_mean_mean'] - all_results[0.01]['pheno_mean_std'],
                     all_results[0.01]['pheno_mean_mean'] + all_results[0.01]['pheno_mean_std'],
                     alpha=0.2, color='red')
    
    ax1.plot(t, all_results[0.0]['pheno_mean_mean'], 'b-', label='Neutral (s=0.0)')
    ax1.fill_between(t,
                     all_results[0.0]['pheno_mean_mean'] - all_results[0.0]['pheno_mean_std'],
                     all_results[0.0]['pheno_mean_mean'] + all_results[0.0]['pheno_mean_std'],
                     alpha=0.2, color='blue')
    
    ax1.axhline(y=l/2, color='gray', linestyle='--', alpha=0.5, label='Neutral expectation')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Mean phenotype')
    ax1.set_title('Comparison of Mean Phenotype: Selected vs Neutral')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot variance comparison
    ax2.plot(t, all_results[0.01]['pheno_var_mean'], 'r-', label='Selected (s=0.01)')
    ax2.fill_between(t,
                     all_results[0.01]['pheno_var_mean'] - all_results[0.01]['pheno_var_std'],
                     all_results[0.01]['pheno_var_mean'] + all_results[0.01]['pheno_var_std'],
                     alpha=0.2, color='red')
    
    ax2.plot(t, all_results[0.0]['pheno_var_mean'], 'b-', label='Neutral (s=0.0)')
    ax2.fill_between(t,
                     all_results[0.0]['pheno_var_mean'] - all_results[0.0]['pheno_var_std'],
                     all_results[0.0]['pheno_var_mean'] + all_results[0.0]['pheno_var_std'],
                     alpha=0.2, color='blue')
    
    # Add text showing average variance difference
    avg_var_selected = np.mean(all_results[0.01]['pheno_var_mean'])
    avg_var_neutral = np.mean(all_results[0.0]['pheno_var_mean'])
    var_diff = avg_var_selected - avg_var_neutral
    ax2.text(0.02, 0.98, 
             f'Average variance difference\n(Selected - Neutral): {var_diff:.2f}',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Phenotype variance')
    ax2.set_title('Comparison of Phenotype Variance: Selected vs Neutral')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Save comparison results
    comparison_data = {
        'neutral_results': all_results[0.0],
        'selected_results': all_results[0.01],
        'params': {
            'N': N, 'l': l, 'mu': mu,
            'T': T, 'n_sims': n_sims
        }
    }
    save_results(comparison_data, f"{results_dir}/phenotype_comparison_data.npz")
    print(f"Comparison data saved to {results_dir}/phenotype_comparison_data.npz")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*80)
    print("TIMING SUMMARY")
    print("="*80)
    print(f"Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Parameters: N={N}, l={l}, T={T}, n_sims={n_sims}")
    print("Analyzed both s=0.01 (selection) and s=0 (neutral) cases")
    print("\nAll results have been saved in the 'convergence_results' directory:")
    print(f"  - Raw data: raw_selection_data.npz, raw_neutral_data.npz")
    print(f"  - Processed data: processed_selection_data.npz, processed_neutral_data.npz")
    print(f"  - Comparison data: phenotype_comparison_data.npz")

if __name__ == "__main__":
    main()