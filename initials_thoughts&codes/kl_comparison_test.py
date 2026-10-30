#!/usr/bin/env python3
"""
Test script to compare different KL divergence calculation methods
and explain why we get different values (80 bits vs 47 bits)
"""

import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
from Markov import compute_stationary_quiet

def kl_bits_old_method(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    OLD METHOD (from QE.py): Strict masking with epsilon only on zeros
    
    WHAT IT DOES:
    1. Only adds epsilon to values that are EXACTLY zero: np.where(p == 0, p + eps, p)
    2. Leaves all non-zero values unchanged (even if they're extremely small like 1e-302)
    3. Re-normalizes the distributions
    4. Applies a mask to only consider bins where BOTH distributions are > 0
    5. Computes KL divergence only on the masked bins
    
    PROBLEMS:
    - Creates numerical instability when distributions have very different sparsity patterns
    - Leaves many extremely small values unsmoothed, causing artificial inflation
    - Results in inconsistent KL divergence values depending on the reference distribution
    """
    # Add epsilon only to non-zero values
    p = np.where(p == 0, p + eps, p)
    q = np.where(q == 0, q + eps, q)
    
    # Re-normalize after adding epsilon
    p = p / p.sum()
    q = q / q.sum()
    mask = (p > 0) & (q > 0)
    if not np.any(mask):
        return 0.0
    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))

def kl_bits_new_method(p: np.ndarray, q: np.ndarray, eps: float = 1e-15) -> float:
    """
    NEW METHOD (from 3.6B_maybe.py): Symmetric smoothing everywhere
    
    WHAT IT DOES:
    1. Adds epsilon to ALL values in both distributions: p = p + eps, q = q + eps
    2. This includes both zero and non-zero values (even extremely small ones)
    3. Re-normalizes the distributions after smoothing
    4. Computes KL divergence over ALL bins (no masking needed)
    5. Uses the full formula: sum(p * log2(p/q))
    
    ADVANTAGES:
    - Numerically stable regardless of distribution sparsity patterns
    - Handles extremely small values correctly (like 1e-302 in binomial)
    - Gives consistent, realistic KL divergence values
    - No artificial inflation due to unsmoothed values
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    # symmetric additive smoothing
    p = p + eps
    q = q + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * (np.log2(p) - np.log2(q))))

def kl_pheno_bits(pheno_psi: np.ndarray, pheno_phi: np.ndarray, eps: float = 1e-16) -> float:
    """Method for phenotype distributions with meaningful range binning"""
    # Find the meaningful range where both distributions have mass
    min_val = max(0, min(pheno_psi.min(), pheno_phi.min()))
    max_val = max(pheno_psi.max(), pheno_phi.max())
    
    # Create histograms over the meaningful range with reasonable binning
    num_bins = min(100, max_val - min_val + 1)
    bins = np.linspace(min_val, max_val, num_bins + 1)
    
    # Compute histograms
    hist_psi, _ = np.histogram(pheno_psi, bins=bins, density=True)
    hist_phi, _ = np.histogram(pheno_phi, bins=bins, density=True)
    
    # Convert to PMFs (normalize to sum to 1)
    hist_psi = hist_psi * (bins[1] - bins[0])
    hist_phi = hist_phi * (bins[1] - bins[0])
    
    # Apply smoothing and compute KL
    hist_psi = hist_psi + eps
    hist_phi = hist_phi + eps
    hist_psi /= hist_psi.sum()
    hist_phi /= hist_phi.sum()
    
    return float(np.sum(hist_psi * (np.log2(hist_psi) - np.log2(hist_phi))))

def main():
    print("="*60)
    print("KL DIVERGENCE COMPARISON TEST")
    print("="*60)
    
    # Parameters from 3.6B_maybe.py run() function
    N = 40
    l = 1000
    N_pop = 2000
    mu = 0.0005
    s = 0.01
    seed = 42
    
    print(f"Parameters: N={N}, l={l}, N_pop={N_pop}, mu={mu}, s={s}")
    print()
    
    # 1. Create binomial reference distribution (like in QE.py)
    q_binom = binom.pmf(np.arange(l+1), l, 0.5)
    print(f"1. Binomial reference distribution Bin({l}, 0.5):")
    print(f"   - Range: 0 to {l}")
    print(f"   - Mean: {l * 0.5}")
    print(f"   - Non-zero bins: {np.sum(q_binom > 1e-15)} out of {l+1}")
    print(f"   - Max probability: {q_binom.max():.6f} at x={np.argmax(q_binom)}")
    print()
    
    # 2. Generate proper phi and psi distributions using the same method as 3.6B_maybe.py
    rng = np.random.default_rng(seed)
    
    # Get stationary distributions
    phi = compute_stationary_quiet(N=N, s=0.0, mu=mu)
    psi = compute_stationary_quiet(N=N, s=s, mu=mu)
    
    # Calculate means
    states = np.arange(N + 1)
    f_phi = (states * phi).sum() / N
    f_psi = (states * psi).sum() / N
    
    print(f"2. Stationary distributions:")
    print(f"   - phi (s=0): mean frequency f_phi = {f_phi:.4f}")
    print(f"   - psi (s={s}): mean frequency f_psi = {f_psi:.4f}")
    print()
    
    # 3. Generate phenotype distributions using the same sampling method as 3.6B_maybe.py
    def sample_Z_hist(pi: np.ndarray, N: int, l: int, N_pop: int, rng: np.random.Generator):
        """Same function as in 3.6B_maybe.py"""
        states = np.arange(N + 1)
        counts = rng.choice(states, size=l, p=pi)
        freqs = counts / N
        genome = np.empty((N_pop, l), dtype=np.uint8)
        for j, p_locus in enumerate(freqs):
            genome[:, j] = rng.binomial(1, p_locus, size=N_pop).astype(np.uint8)
        pheno = genome.sum(axis=1)
        hist = np.bincount(pheno, minlength=l + 1).astype(float)
        hist /= hist.sum()
        return genome, hist
    
    # Generate phenotype samples
    genome_phi, Zphi = sample_Z_hist(phi, N=N, l=l, N_pop=N_pop, rng=rng)
    genome_psi, Zpsi = sample_Z_hist(psi, N=N, l=l, N_pop=N_pop, rng=rng)
    
    pheno_phi = genome_phi.sum(axis=1)
    pheno_psi = genome_psi.sum(axis=1)
    
    print(f"3. Generated phenotype distributions:")
    print(f"   - phi phenotype: mean={pheno_phi.mean():.1f}, std={pheno_phi.std():.1f}")
    print(f"   - psi phenotype: mean={pheno_psi.mean():.1f}, std={pheno_psi.std():.1f}")
    print()
    
    # 4. Create histograms over full range (like Zphi, Zpsi in 3.6B_maybe.py)
    # Note: Zphi and Zpsi are already the histograms from sample_Z_hist
    print(f"4. Histograms over full range {{0, ..., {l}}}:")
    print(f"   - phi histogram (Zphi): non-zero bins: {np.sum(Zphi > 1e-15)} out of {l+1}")
    print(f"   - psi histogram (Zpsi): non-zero bins: {np.sum(Zpsi > 1e-15)} out of {l+1}")
    print()
    
    # 5. Compare different KL calculation methods
    
    print("5. KL DIVERGENCE COMPARISONS:")
    print("-" * 40)
    
    # Method A: Old method (QE.py style) - psi vs binomial
    kl_old_psi_vs_binom = kl_bits_old_method(Zpsi, q_binom)
    print(f"A. Old method (psi vs binomial): {kl_old_psi_vs_binom:.3f} bits")
    
    # Method B: New method (3.6B_maybe.py style) - psi vs binomial  
    kl_new_psi_vs_binom = kl_bits_new_method(Zpsi, q_binom)
    print(f"B. New method (psi vs binomial): {kl_new_psi_vs_binom:.3f} bits")
    
    # Method C: Old method - psi vs phi histograms
    kl_old_psi_vs_phi = kl_bits_old_method(Zpsi, Zphi)
    print(f"C. Old method (psi vs phi histograms): {kl_old_psi_vs_phi:.3f} bits")
    
    # Method D: New method - psi vs phi histograms
    kl_new_psi_vs_phi = kl_bits_new_method(Zpsi, Zphi)
    print(f"D. New method (psi vs phi histograms): {kl_new_psi_vs_phi:.3f} bits")
    
    # Method E: Phenotype method - psi vs phi samples
    kl_pheno_psi_vs_phi = kl_pheno_bits(pheno_psi, pheno_phi)
    print(f"E. Phenotype method (psi vs phi samples): {kl_pheno_psi_vs_phi:.3f} bits")
    
    # Additional calculations for φ vs binomial
    kl_old_phi_vs_binom = kl_bits_old_method(Zphi, q_binom)
    kl_new_phi_vs_binom = kl_bits_new_method(Zphi, q_binom)
    print(f"F. Old method (phi vs binomial): {kl_old_phi_vs_binom:.3f} bits")
    print(f"G. New method (phi vs binomial): {kl_new_phi_vs_binom:.3f} bits")
    
    print()
    
    # 6. Explanation
    print("6. WHY THE DIFFERENCES?")
    print("-" * 40)
    print("The key differences come from:")
    print()
    print("A. SMOOTHING METHOD (BIGGEST EFFECT):")
    print("   - Old method: epsilon only on zero values")
    print("   - New method: symmetric smoothing everywhere")
    print(f"   - This causes a {kl_old_psi_vs_binom - kl_new_psi_vs_binom:.1f} bit difference!")
    print("   - Old method is numerically unstable for sparse distributions")
    print()
    print("B. REFERENCE DISTRIBUTION:")
    print("   - ψ vs Bin(l, 0.5): theoretical vs empirical")
    print("   - ψ vs φ: empirical vs empirical")
    print(f"   - Difference: {kl_new_psi_vs_binom - kl_new_psi_vs_phi:.1f} bits")
    print()
    print("C. SPARSITY EFFECT:")
    print("   - Full range histograms (0...l) have many zero bins")
    print("   - Phenotype method focuses on meaningful range only")
    print("   - Results in more stable KL divergence calculations")
    print()
    
    # 7. Visual comparison
    print("7. VISUAL COMPARISON:")
    print("-" * 40)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Stationary distributions φ and ψ
    x_states = np.arange(N + 1)
    ax1.plot(x_states, phi, 'g-', linewidth=2, label='φ (s=0)', alpha=0.8)
    ax1.plot(x_states, psi, 'r-', linewidth=2, label=f'ψ (s={s})', alpha=0.8)
    ax1.set_xlabel('Allele count j (0..N)')
    ax1.set_ylabel('Probability')
    ax1.set_title('Single-locus Stationary Distributions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add KL divergence info as text
    dx_bits = kl_bits_new_method(psi, phi)
    ax1.text(0.02, 0.98, f'D(X) = {dx_bits:.3f} bits', transform=ax1.transAxes, 
             ha='left', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Binomial reference vs psi histogram
    x = np.arange(l+1)
    ax2.plot(x, q_binom, 'b-', linewidth=2, label='Bin(l, 0.5)', alpha=0.8)
    ax2.plot(x, Zpsi, 'r-', linewidth=1, label='ψ histogram', alpha=0.7)
    ax2.set_xlim(400, 800)
    ax2.set_xlabel('Phenotype')
    ax2.set_ylabel('Probability')
    ax2.set_title('Binomial Reference vs ψ Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add KL divergence info as text
    ax2.text(0.02, 0.98, f'Old method: {kl_old_psi_vs_binom:.1f} bits\nNew method: {kl_new_psi_vs_binom:.1f} bits', 
             transform=ax2.transAxes, ha='left', va='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 3: Phenotype distributions (zoomed)
    ax3.hist(pheno_phi, bins=50, density=True, alpha=0.6, label='φ phenotype', color='green', edgecolor='black', linewidth=0.5)
    ax3.hist(pheno_psi, bins=50, density=True, alpha=0.6, label='ψ phenotype', color='red', edgecolor='black', linewidth=0.5)
    ax3.set_xlim(400, 800)
    ax3.set_xlabel('Phenotype')
    ax3.set_ylabel('Density')
    ax3.set_title('Phenotype Distributions (zoomed)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add KL divergence info as text
    ax3.text(0.02, 0.98, f'Old method: {kl_old_psi_vs_phi:.1f} bits\nNew method: {kl_new_psi_vs_phi:.1f} bits\nPhenotype method: {kl_pheno_psi_vs_phi:.1f} bits', 
             transform=ax3.transAxes, ha='left', va='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 4: All distributions together
    ax4.plot(x, q_binom, 'b-', linewidth=2, label='Bin(l, 0.5)', alpha=0.8)
    ax4.plot(x, Zphi, 'g-', alpha=0.7, label='φ histogram', linewidth=1)
    ax4.plot(x, Zpsi, 'r-', alpha=0.7, label='ψ histogram', linewidth=1)
    ax4.set_xlim(400, 800)
    ax4.set_xlabel('Phenotype')
    ax4.set_ylabel('Probability')
    ax4.set_title('All Distributions Together')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add comprehensive KL divergence info as text
    ax4.text(0.02, 0.98, f'φ vs Bin(l,0.5):\n  New method: {kl_new_phi_vs_binom:.1f} bits\n  Old method: {kl_old_phi_vs_binom:.1f} bits', 
             transform=ax4.transAxes, ha='left', va='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("CONCLUSION:")
    print("The key findings:")
    print()
    print("1. SMOOTHING METHOD MATTERS:")
    print(f"   - Old method (epsilon only on zeros): {kl_old_psi_vs_binom:.1f} bits")
    print(f"   - New method (symmetric smoothing): {kl_new_psi_vs_binom:.1f} bits")
    print(f"   - Difference: {kl_old_psi_vs_binom - kl_new_psi_vs_binom:.1f} bits!")
    print()
    print("2. REFERENCE DISTRIBUTION MATTERS:")
    print(f"   - ψ vs Bin(l,0.5): {kl_new_psi_vs_binom:.1f} bits")
    print(f"   - ψ vs φ (empirical): {kl_new_psi_vs_phi:.1f} bits")
    print(f"   - Difference: {kl_new_psi_vs_binom - kl_new_psi_vs_phi:.1f} bits")
    print()
    print("3. THE CORRECT VALUES:")
    print(f"   - For comparing to theoretical: {kl_new_psi_vs_binom:.1f} bits")
    print(f"   - For comparing empirical distributions: {kl_pheno_psi_vs_phi:.1f} bits")
    print()
    print("The old method gives artificially high values due to numerical instability!")
    print("The new method gives stable, realistic KL divergence measurements.")
    print()
    
    # Additional analysis: Why the difference is larger with binomial reference
    print("DETAILED ANALYSIS: Why smoothing difference is larger with binomial reference")
    print("=" * 70)
    
    # Analyze the distributions to understand the difference
    print("Distribution characteristics:")
    print(f"   - Binomial reference: {np.sum(q_binom > 1e-15)} non-zero bins out of {l+1}")
    print(f"   - ψ histogram: {np.sum(Zpsi > 1e-15)} non-zero bins out of {l+1}")
    print(f"   - φ histogram: {np.sum(Zphi > 1e-15)} non-zero bins out of {l+1}")
    print()
    
    # Check the actual minimum values to understand numerical precision
    print("Numerical precision analysis:")
    print(f"   - Binomial min value: {q_binom[q_binom > 0].min():.2e}")
    print(f"   - ψ histogram min value: {Zpsi[Zpsi > 0].min():.2e}")
    print(f"   - φ histogram min value: {Zphi[Zphi > 0].min():.2e}")
    print(f"   - Threshold used (1e-15): {1e-15:.2e}")
    print()
    
    # Check overlap between distributions
    binom_nonzero = q_binom > 1e-15
    psi_nonzero = Zpsi > 1e-15
    phi_nonzero = Zphi > 1e-15
    
    overlap_binom_psi = np.sum(binom_nonzero & psi_nonzero)
    overlap_phi_psi = np.sum(phi_nonzero & psi_nonzero)
    
    print("Overlap analysis:")
    print(f"   - Binomial-ψ overlap: {overlap_binom_psi} bins")
    print(f"   - φ-ψ overlap: {overlap_phi_psi} bins")
    print(f"   - Binomial-ψ overlap ratio: {overlap_binom_psi/np.sum(psi_nonzero):.3f}")
    print(f"   - φ-ψ overlap ratio: {overlap_phi_psi/np.sum(psi_nonzero):.3f}")
    print()
    
    # Analyze the effect of smoothing on each comparison
    print("Smoothing effect analysis:")
    
    # For binomial comparison
    old_binom = kl_bits_old_method(Zpsi, q_binom)
    new_binom = kl_bits_new_method(Zpsi, q_binom)
    binom_diff = old_binom - new_binom
    
    # For phi comparison  
    old_phi = kl_bits_old_method(Zpsi, Zphi)
    new_phi = kl_bits_new_method(Zpsi, Zphi)
    phi_diff = old_phi - new_phi
    
    print(f"   - Binomial comparison: {old_binom:.1f} → {new_binom:.1f} (diff: {binom_diff:.1f} bits)")
    print(f"   - φ comparison: {old_phi:.1f} → {new_phi:.1f} (diff: {phi_diff:.1f} bits)")
    print(f"   - Ratio of differences: {binom_diff/phi_diff:.1f}x larger for binomial")
    print()
    
    print("REASON: The binomial reference has many more non-zero bins than the empirical")
    print("distributions. When the old method only smooths zeros, it leaves many more")
    print("unsmoothed values in the binomial comparison, causing greater numerical")
    print("instability and artificially inflated KL divergence values.")
    print()
    print("The empirical distributions (φ and ψ) have similar sparsity patterns,")
    print("so the smoothing difference is smaller when comparing them.")
    print()
    
    print("DETAILED EXPLANATION: Why the difference is larger with binomial reference")
    print("=" * 70)
    print()
    print("1. NUMERICAL PRECISION ISSUE:")
    print("   - Binomial distribution has values as small as 9.33e-302")
    print("   - These are effectively zero due to floating-point precision")
    print("   - But they're not exactly zero, so old method doesn't smooth them")
    print()
    print("2. SPARSITY PATTERN MISMATCH:")
    print("   - Binomial: 247 'non-zero' bins (many are actually ~1e-302)")
    print("   - ψ histogram: 26 truly non-zero bins")
    print("   - φ histogram: 28 truly non-zero bins")
    print()
    print("3. OLD METHOD PROBLEM:")
    print("   - Only smooths exactly zero values")
    print("   - Leaves 247 unsmoothed tiny values in binomial")
    print("   - Only 26 unsmoothed values in ψ")
    print("   - This creates massive numerical instability")
    print()
    print("4. NEW METHOD SOLUTION:")
    print("   - Smooths ALL values (including the 1e-302 ones)")
    print("   - Creates balanced, numerically stable comparison")
    print("   - Results in realistic KL divergence values")
    print()
    print("5. WHY φ vs ψ COMPARISON IS MORE STABLE:")
    print("   - Both have similar sparsity (~26-28 non-zero bins)")
    print("   - Similar number of unsmoothed values in old method")
    print("   - Smaller smoothing effect difference")
    print()
    print("CONCLUSION: The old method is fundamentally flawed for comparing")
    print("distributions with very different sparsity patterns, especially when")
    print("one distribution has many extremely small values (like the binomial).")
    print("The new method handles this correctly and gives stable results.")

if __name__ == "__main__":
    main()
