import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
from scipy.stats import binom

from Markov import compute_stationary_quiet

sns.set_style("white")
sns.set_palette("husl")



def kl_bits(p: np.ndarray, q: np.ndarray, eps: float = 1e-15) -> float:
	"""Return D_KL(p || q) in bits with safe symmetric smoothing and renormalization."""
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


def kl_pheno_bits(pheno_psi: np.ndarray, pheno_phi: np.ndarray, eps: float = 1e-15) -> float:
	"""Compute KL divergence between phenotype distributions in bits.
	
	This function properly handles phenotype distributions by:
	1. Finding the meaningful range where both distributions have mass
	2. Creating histograms over that range only
	3. Computing KL divergence with proper smoothing
	"""
	# Find the meaningful range where both distributions have mass
	min_val = max(0, min(pheno_psi.min(), pheno_phi.min()))
	max_val = max(pheno_psi.max(), pheno_phi.max())
	
	# Create histograms over the meaningful range with reasonable binning
	num_bins = min(100, max_val - min_val + 1)  # At most 100 bins
	bins = np.linspace(min_val, max_val, num_bins + 1)
	
	# Compute histograms
	hist_psi, _ = np.histogram(pheno_psi, bins=bins, density=True)
	hist_phi, _ = np.histogram(pheno_phi, bins=bins, density=True)
	
	# Convert to PMFs (normalize to sum to 1)
	hist_psi = hist_psi * (bins[1] - bins[0])  # Convert density to PMF
	hist_phi = hist_phi * (bins[1] - bins[0])
	
	# Apply smoothing and compute KL
	hist_psi = hist_psi + eps
	hist_phi = hist_phi + eps
	hist_psi /= hist_psi.sum()
	hist_phi /= hist_phi.sum()
	
	return float(np.sum(hist_psi * (np.log2(hist_psi) - np.log2(hist_phi))))


def kl_psi_vs_binomial(pheno_psi: np.ndarray, l: int, eps: float = 1e-15) -> float:
	"""Compute KL divergence between psi phenotype distribution and theoretical Bin(l, 0.5).
	
	This is the same comparison as in QE.py - comparing empirical selection distribution
	to the theoretical neutral binomial distribution.
	"""
	# Create theoretical binomial reference distribution
	q_binom = binom.pmf(np.arange(l+1), l, 0.5)
	
	# Create histogram of psi phenotype over full range {0, ..., l}
	hist_psi = np.bincount(pheno_psi, minlength=l+1).astype(float)
	hist_psi /= hist_psi.sum()
	
	# Compute KL divergence using the same method as QE.py
	# Add epsilon only to non-zero values
	hist_psi = np.where(hist_psi == 0, hist_psi + eps, hist_psi)
	q_binom = np.where(q_binom == 0, q_binom + eps, q_binom)
	
	# Re-normalize after adding epsilon
	hist_psi /= hist_psi.sum()
	q_binom /= q_binom.sum()
	
	# Apply mask and compute KL
	mask = (hist_psi > 0) & (q_binom > 0)
	if not np.any(mask):
		return 0.0
	return float(np.sum(hist_psi[mask] * np.log2(hist_psi[mask] / q_binom[mask])))


def sample_Z_hist(pi: np.ndarray, N: int, l: int, N_pop: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a phenotype histogram Z from π* by the locus/Bernoulli scheme.

    - Draw l locus-specific frequencies by sampling j ~ π and setting p=j/N
    - For each locus, sample N_pop Bernoulli(p) across individuals
    - Sum per row to get phenotype counts and return normalized histogram
    """
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


def plot_pheno_distributions(N: int, l: int, N_pop: int, mu: float, s: float, seed: int = 1) -> Tuple[np.ndarray, np.ndarray, float]:
    """Two-panel analysis plot and printout.

    Left panel: overlay single-locus stationary distributions φ and ψ over x = {0..N}.
    Right panel: overlay phenotype distributions (per-individual sums across loci) with ≥30 bins.

    Returns (pheno_phi, pheno_psi, D_Z_bits_from_samples).
    """
    rng = np.random.default_rng(seed)
    phi = compute_stationary_quiet(N=N, s=0.0, mu=mu)
    psi = compute_stationary_quiet(N=N, s=s, mu=mu)

    # Phenotype genomes and samples
    genome_phi, Zphi = sample_Z_hist(phi, N=N, l=l, N_pop=N_pop, rng=rng)
    genome_psi, Zpsi = sample_Z_hist(psi, N=N, l=l, N_pop=N_pop, rng=rng)

    pheno_phi = genome_phi.sum(axis=1)
    pheno_psi = genome_psi.sum(axis=1)

    # KLs and means
    DX_single_locus = kl_bits(psi, phi)
    states = np.arange(N + 1)
    f_phi = (states * phi).sum() / N
    f_psi = (states * psi).sum() / N
    DG_single_locus = kl_bits(np.array([1 - f_psi, f_psi]), np.array([1 - f_phi, f_phi]))
    
    # Use proper KL calculation for phenotype distributions
    DZ = kl_pheno_bits(pheno_psi, pheno_phi)
    DZ_old = kl_bits(Zpsi, Zphi)  # For comparison
    DZ_psi_vs_binom = kl_psi_vs_binomial(pheno_psi, l)  # QE.py style

    # Figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    color_phi, color_psi = sns.color_palette(n_colors=2)

    # Panel 1: single-locus distributions φ and ψ over {0..N}
    x = np.arange(N + 1)
    ax1.plot(x, phi, color=color_phi, linewidth=1.8, label='φ (s=0)')
    ax1.plot(x, psi, color=color_psi, linewidth=1.8, label='ψ (s>0)')
    ax1.set_xlabel('Allele count j (0..N)')
    ax1.set_ylabel('Probability')
    ax1.set_xlim(0, N)
    ax1.set_title('Single-locus stationary distributions')
    ax1.legend(frameon=True)

    # Panel 2: phenotype distributions from samples
    num_bins = max(30, min(l + 1, 200))
    pheno_max = int(max(pheno_phi.max(), pheno_psi.max()))
    lo, hi = 400, max(401, pheno_max + 20)

    sns.histplot(pheno_phi, bins=num_bins, stat="probability",
                 color=color_phi, alpha=0.4, element="step", fill=True, ax=ax2, label="φ phenotype")
    sns.histplot(pheno_psi, bins=num_bins, stat="probability",
                 color=color_psi, alpha=0.4, element="step", fill=True, ax=ax2, label="ψ phenotype")

    ax2.set_xlabel('Phenotype (sum over loci)')
    ax2.set_ylabel('Probability')
    ax2.set_xlim(lo, hi)
    ax2.set_title('Phenotype distributions (samples)')
    ax2.legend(frameon=True)
    ax2.text(0.5, 0.92, f'D(Z) = {DZ:.3f} bits (old: {DZ_old:.3f}, vs binom: {DZ_psi_vs_binom:.3f})', transform=ax2.transAxes,
             ha='center', va='top', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.show()

    # Print same diagnostics as run()
    print(f"phi: {phi}")
    print(f"psi: {psi}")
    print(f"Zphi: {Zphi}")
    print(f"Zpsi: {Zpsi}")
    print(f"D_X_bits: {DX_single_locus}")
    print(f"D_G_bits: {DG_single_locus}")
    print(f"D_Z_bits: {DZ} (old method: {DZ_old}, vs binomial: {DZ_psi_vs_binom:.3f})")
    print(f"params: {{'N': {N}, 'mu': {mu}, 's': {s}, 'l': {l}, 'N_pop': {N_pop}, 'seed': {seed}}}")
    print(f"means: {{'f_phi': {f_phi}, 'f_psi': {f_psi}}}")

    return pheno_phi, pheno_psi, DZ


def kl_from_pheno_samples(pheno_psi: np.ndarray, pheno_phi: np.ndarray, l: int, eps: float = 1e-15) -> float:
    """Compute KL(psi || phi) in bits from phenotype samples by forming PMFs over {0..l}."""
    hist_phi = np.bincount(pheno_phi, minlength=l + 1).astype(float)
    hist_psi = np.bincount(pheno_psi, minlength=l + 1).astype(float)
    # symmetric smoothing inside kl_bits; we normalize explicitly here as well
    hist_phi /= hist_phi.sum()
    hist_psi /= hist_psi.sum()
    return kl_bits(hist_psi, hist_phi, eps=eps)


def run(N: int = 40, mu: float = 0.0005, s: float = 0.01, l: int = 1000, N_pop: int = 20000, seed: int = 10):
    rng = np.random.default_rng(seed)

    # Stationary distributions without prints/plots
    phi = compute_stationary_quiet(N=N, s=0.0, mu=mu)
    psi = compute_stationary_quiet(N=N, s=s, mu=mu)

    # Divergences D(X) and D(G)
    DX_single_locus = kl_bits(psi, phi)
    states = np.arange(N + 1)
    f_phi = (states * phi).sum() / N
    f_psi = (states * psi).sum() / N
    DG_single_locus = kl_bits(np.array([1 - f_psi, f_psi]), np.array([1 - f_phi, f_phi]))

    # Phenotype distributions Z^phi and Z^psi
    genome_phi, Zphi = sample_Z_hist(phi, N=N, l=l, N_pop=N_pop, rng=rng)
    genome_psi, Zpsi = sample_Z_hist(psi, N=N, l=l, N_pop=N_pop, rng=rng)
    pheno_phi = genome_phi.sum(axis=1)
    pheno_psi = genome_psi.sum(axis=1)
    
    # Use the proper KL calculation for phenotype distributions
    DZ = kl_pheno_bits(pheno_psi, pheno_phi)
    
    # Also compute the old way for comparison
    DZ_old = kl_bits(Zpsi, Zphi)
    
    # Compute KL divergence between psi and theoretical binomial (like in QE.py)
    DZ_psi_vs_binom = kl_psi_vs_binomial(pheno_psi, l)
    
    print(f"KL divergence comparison:")
    print(f"  Old method (Zpsi vs Zphi): {DZ_old:.3f} bits")
    print(f"  New method (pheno_psi vs pheno_phi): {DZ:.3f} bits")
    print(f"  QE.py style (psi vs Bin(l,0.5)): {DZ_psi_vs_binom:.3f} bits")
    
    # Quick histogram to see the distributions
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist([pheno_psi, pheno_phi], bins=50, density=True, alpha=0.6, label=['ψ phenotype', 'φ phenotype'])
    plt.xlabel('Phenotype value')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Phenotype distributions')
    
    plt.subplot(1, 2, 2)
    plt.hist([pheno_psi, pheno_phi], bins=50, density=True, alpha=0.6, label=['ψ phenotype', 'φ phenotype'])
    plt.xlabel('Phenotype value')
    plt.ylabel('Density')
    plt.xlim(400, 600)  # Zoom in on the main mass
    plt.legend()
    plt.title('Phenotype distributions (zoomed)')
    plt.tight_layout()
    plt.show()

    # Plot Z distributions as histograms (PMFs), zoomed x-axis 400..750
    x = np.arange(l + 1)
    view_lo, view_hi = 400, 900
    mask = (x >= view_lo) & (x <= view_hi)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    color_phi, color_psi = sns.color_palette(n_colors=2)

    # Histogram bars with slight offset for visibility
    ax.bar(x[mask] , Zphi[mask], width=0.30, color=color_phi, alpha=0.6,
           edgecolor=color_phi, linewidth=0.3, label='φ^Z')
    ax.bar(x[mask] , Zpsi[mask], width=0.30, color=color_psi, alpha=0.6,
           edgecolor=color_psi, linewidth=0.3, label='ψ^Z')

    ax.set_xlabel('Phenotype z')
    ax.set_ylabel('Probability')
    ax.set_xlim(view_lo, view_hi)
    ax.set_title('Stationary phenotype distr.')
    ax.legend(frameon=True)
    ax.text(0.5, 0.92, f'D(Z) = {DZ:.4f} bits', transform=ax.transAxes, ha='center', va='top', fontsize=13,
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.tight_layout()
    plt.show()

    # Print all results to screen
    print(f"phi: {phi}")
    print(f"psi: {psi}")
    print(f"Zphi: {Zphi}")
    print(f"Zpsi: {Zpsi}")
    print(f"D_X_bits: {DX_single_locus}")
    print(f"D_G_bits: {DG_single_locus}")
    print(f"D_Z_bits: {DZ} (old method: {DZ_old}, vs binomial: {DZ_psi_vs_binom:.3f})")
    print(f"params: {{'N': {N}, 'mu': {mu}, 's': {s}, 'l': {l}, 'N_pop': {N_pop}, 'seed': {seed}}}")
    print(f"means: {{'f_phi': {f_phi}, 'f_psi': {f_psi}}}")

    # Only return the two variables
    return Zphi, Zpsi, phi, psi



if __name__ == '__main__':
     Zphi, Zpsi, phi, psi = run()
    #print('Done. KL divergences: ', {k: v for k, v in out.items() if k.startswith('D_')})
