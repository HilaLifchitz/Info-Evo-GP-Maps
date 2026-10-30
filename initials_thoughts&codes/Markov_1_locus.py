#!/usr/bin/env python3
"""
Markov Chain Analysis for Wright-Fisher Population Genetics

This module computes the stationary distribution of allele frequencies in a
finite Wright-Fisher population with selection and mutation using Markov chain
theory.

BIOLOGICAL MODEL:
================
- Haploid population of size N
- Two alleles: "0" (wildtype) and "1" (mutant/beneficial)
- Selection coefficient s: fitness of "1" allele relative to "0"
- Symmetric mutation rate μ: probability of allele flipping per generation
- Binomial sampling: genetic drift due to finite population size

MATHEMATICAL APPROACH:
=====================
The Wright-Fisher process with selection and mutation forms a finite Markov
chain on state space {0, 1, 2, ..., N} where each state represents the
number of "1" alleles in the population.

The transition probability from j "1" alleles to i "1" alleles is at any time step t:
    P[i,j] = Binomial(N, i, p')  = probablity of transitioning i <-- j
    
where p' is the frequency after selection and mutation:
    p_sel = (j/N * (1+s)) / (j/N * (1+s) + (1-j/N))
    p' = (1-μ) * p_sel + μ * (1-p_sel)

The stationary distribution π satisfies: π = P π with Σπ_i = 1 because P is column-stochastic
"""

import numpy as np
from scipy.stats import binom
from scipy.linalg import solve, norm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import kl_div

# Set seaborn style
sns.set_style("white")
sns.set_palette("husl")


class WrightFisherMarkovChain:
    """
    Wright-Fisher Markov chain with selection and mutation.
    
    This class constructs the transition matrix and computes the stationary
    distribution for a finite Wright-Fisher population under selection
    and symmetric mutation.
    
    Biological Parameters:
    ---------------------
    N : int
        Population size (number of haploid individuals)
    s : float  
        Selection coefficient. Fitness of "1" allele = 1+s relative to "0" allele = 1
        - s > 0: "1" allele is beneficial (directional selection)
        - s = 0: neutral evolution (no selection)
        - s < 0: "1" allele is deleterious
    mu : float
        Symmetric mutation rate per allele per generation
        - μ = 0: no mutation (absorbing boundaries at 0 and N)
        - μ > 0: mutation-selection-drift balance
        
    Mathematical Parameters:
    -----------------------
    tol : float, optional (default=1e-12)
        Convergence tolerance for iterative methods
    max_iter : int, optional (default=10000)
        Maximum iterations for iterative stationary distribution solver
    """
    
    def __init__(self, N, s, mu, tol=1e-12, max_iter=10000, quiet: bool = False):
        self.N = N              # Population size
        self.s = s              # Selection coefficient  
        self.mu = mu            # Mutation rate
        self.tol = tol          # Convergence tolerance
        self.max_iter = max_iter # Maximum iterations
        self.quiet = quiet      # Suppress prints/plots when True
        
        # Initialize arrays
        self.P = None           # Transition matrix (N+1) x (N+1)
        self.pi_star = None     # Stationary distribution
        self.converged = False  # Convergence flag
        self.iterations = 0     # Number of iterations used
        
        # Validate parameters
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate biological and mathematical parameters."""
        if self.N <= 0 or not isinstance(self.N, int):
            raise ValueError(f"Population size N must be positive integer, got {self.N}")
        if self.mu < 0 or self.mu > 1:
            raise ValueError(f"Mutation rate mu must be in [0,1], got {self.mu}")
        if self.s < -1:
            raise ValueError(f"Selection coefficient s must be > -1 for fitness > 0, got {self.s}")
        if self.tol <= 0:
            raise ValueError(f"Tolerance must be positive, got {self.tol}")
        if self.max_iter <= 0:
            raise ValueError(f"Max iterations must be positive, got {self.max_iter}")
    
    def _compute_transition_probability(self, j, i):
        """
        Compute transition probability from j "1" alleles to i "1" alleles.
        
        This implements the Wright-Fisher process:
        1. Selection: Change allele frequency based on fitness difference
        2. Mutation: Symmetric mutation between alleles
        3. Sampling: Binomial sampling due to finite population size
        
        Parameters:
        -----------
        j : int
            Current number of "1" alleles (state we're transitioning from)
        i : int  
            Next number of "1" alleles (state we're transitioning to)
            
        Returns:
        --------
        float
            Transition probability P[i,j]: i <-- j
        """
        
        # Current frequency of "1" allele
        x = j / self.N
        
        # STEP 1: Apply selection
        # Relative fitness of "1" allele = 1 + s
        # Relative fitness of "0" allele = 1
        # Frequency after selection = (freq * fitness) / (average fitness)
        if j == 0:
            # All "0" alleles: no selection effect
            p_sel = 0.0
        elif j == self.N:
            # All "1" alleles: no selection effect  
            p_sel = 1.0
        else:
            # Mixed population: selection changes frequency
            numerator = x * (1 + self.s)
            denominator = x * (1 + self.s) + (1 - x) * 1.0
            p_sel = numerator / denominator # relative fitness of j "1" alleles
        
        # STEP 2: Apply symmetric mutation
        # Mutation rate μ: probability that each allele flips
        # P("1" → "0") = μ, P("1" → "1") = 1-μ  
        # P("0" → "1") = μ, P("0" → "0") = 1-μ
        # Expected frequency after mutation:
        p_prime = (1 - self.mu) * p_sel + self.mu * (1 - p_sel) 
        # p_sel is the expected portion of the 1 allele, but from which (1-mu) wont flip to 0,
        # and mu times the portion of the 0 allele (=1-p_sel) will flip from 0 to 1.
        
        # STEP 3: Binomial sampling for next generation
        # Each of N individuals independently has probability p_prime of being "1"
        # Number of "1" alleles follows Binomial(N, p_prime)
        return binom.pmf(i, self.N, p_prime)
    
    def construct_transition_matrix(self):
        """
        Construct the (N+1) x (N+1) transition matrix P.
        
        Matrix element P[i,j] = probability of transitioning from j "1" alleles
        to i "1" alleles in one generation.
        
        Mathematical Properties:
        -----------------------
        - P is column-stochastic: each column sums to 1 (conservation of probability)
        - P is irreducible if μ > 0 (mutation connects all states)
        - P has unique stationary distribution if irreducible and finite
        - Boundary behavior:
          * μ = 0: states 0 and N are absorbing (fixation)
          * μ > 0: all states communicate (mutation-selection-drift balance)
        """
        
        if not self.quiet:
            print(f"Constructing transition matrix for N={self.N}, s={self.s}, μ={self.mu}")
        
        # Initialize transition matrix
        self.P = np.zeros((self.N + 1, self.N + 1))
        
        # Fill transition matrix
        for j in range(self.N + 1):  # Current state (columns)
            for i in range(self.N + 1):  # Next state (rows)
                self.P[i, j] = self._compute_transition_probability(j, i) 
                # P[i, j] = i <-- j :probability of transitioning from j "1" alleles to i "1" alleles in one generation
        
        # Verify row-stochastic property (debugging)
        column_sums = np.sum(self.P, axis=0)  # Sum over rows for each column
        if not np.allclose(column_sums, 1.0, atol=1e-10) and not self.quiet:
            print(f"Warning: Transition matrix not exactly row-stochastic")
            print(f"Column sums: min={np.min(column_sums):.2e}, max={np.max(column_sums):.2e}")
        
        if not self.quiet:
            print(f"Transition matrix constructed: shape {self.P.shape}")
        
        # Display boundary behavior
        if not self.quiet:
            if self.mu == 0:
                print("μ = 0: Absorbing boundaries at states 0 and N (fixation inevitable)")
            else:
                print(f"μ = {self.mu}: Irreducible chain (all states communicate)")
        
        # Optional visualizations/diagnostics
        if not self.quiet:
            self._plot_transition_matrix_heatmap()
            self._analyze_eigenvalues()
            
        return self.P
    
    def _plot_transition_matrix_heatmap(self):
        """
        Plot heatmap of the transition matrix P.
        
        This visualization helps understand the structure of the Markov chain:
        - Diagonal patterns: states tend to stay near themselves
        - Off-diagonal spread: genetic drift causes variance
        - Asymmetry: selection causes directional bias
        - Boundary behavior: absorption vs mutation effects
        """
        
        if self.P is None:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Full transition matrix
        im1 = ax1.imshow(self.P, cmap='Blues', aspect='equal', interpolation='nearest')
        ax1.set_xlabel('From state j (number of "1" alleles)')
        ax1.set_ylabel('To state i (number of "1" alleles)')
        ax1.set_title(f'Transition Matrix P\nN={self.N}, s={self.s}, μ={self.mu}')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Transition Probability P[i,j]: i <-- j')
        
        # Set ticks to show actual state numbers
        tick_positions = np.arange(0, self.N+1, max(1, self.N//5))
        ax1.set_xticks(tick_positions)
        ax1.set_yticks(tick_positions)
        ax1.set_xticklabels(tick_positions)
        ax1.set_yticklabels(tick_positions)
        
        # Plot 2: Log scale to see small probabilities
        P_log = np.log10(self.P + 1e-16)  # Add small value to avoid log(0)
        im2 = ax2.imshow(P_log, cmap='Blues', aspect='equal', interpolation='nearest')
        ax2.set_xlabel('From state j (number of "1" alleles)')
        ax2.set_ylabel('To state i (number of "1" alleles)')
        ax2.set_title('Transition Matrix P (Log Scale)\nReveals small probabilities')
        
        # Add colorbar for log scale
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('log₁₀(P[i,j]) : i <-- j')
        
        ax2.set_xticks(tick_positions)
        ax2.set_yticks(tick_positions)
        ax2.set_xticklabels(tick_positions)
        ax2.set_yticklabels(tick_positions)
        
        plt.tight_layout()
        plt.show()
    
    def _analyze_eigenvalues(self):
        """
        Detailed analysis of eigenvalues and eigenvectors of P^T.
        
        This is crucial for understanding the chain's behavior:
        - Eigenvalue 1: Must exist and be unique for irreducible chains
        - Second largest eigenvalue: Controls mixing time
        - Eigenvector for λ=1: The stationary distribution
        - Multiplicities: Indicate reducibility issues
        """
        
        if self.P is None:
            return
        
        if self.quiet:
            return
        print("\n" + "="*60)
        print("EIGENVALUE ANALYSIS OF P (RIGHT EIGENVECTORS)")
        print("="*60)
        
        # Compute all eigenvalues and eigenvectors (RIGHT eigenvectors of P)
        # We want π such that P π = π
        eigenvalues, eigenvectors = np.linalg.eig(self.P)
        
        # Sort by magnitude (descending)
        idx_sorted = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues_sorted = eigenvalues[idx_sorted]
        eigenvectors_sorted = eigenvectors[:, idx_sorted]
        
        print(f"Matrix size: {self.P.shape[0]}×{self.P.shape[1]}")
        print(f"Total eigenvalues found: {len(eigenvalues)}")
        
        # Show all eigenvalues
        print(f"\nAll eigenvalues (sorted by magnitude):")
        for i, eval in enumerate(eigenvalues_sorted):
            magnitude = np.abs(eval)
            if np.isreal(eval):
                print(f"  λ_{i:2d}: {eval.real:12.8f} (magnitude: {magnitude:.8f})")
            else:
                print(f"  λ_{i:2d}: {eval:.8f} (magnitude: {magnitude:.8f}) [COMPLEX]")
        
        # Find eigenvalues equal to 1 (within tolerance)
        tolerance = 1e-10
        unity_indices = np.where(np.abs(eigenvalues_sorted - 1.0) < tolerance)[0]
        
        print(f"\n" + "="*40)
        print(f"EIGENVALUES EQUAL TO 1 (tolerance: {tolerance})")
        print("="*40)
        print(f"Number of eigenvalues = 1: {len(unity_indices)}")
        
        if len(unity_indices) == 0:
            print("⚠️  ERROR: No eigenvalue equal to 1 found!")
            print("   This violates fundamental Markov chain theory.")
            print("   Check if P is column-stochastic.")
            return
        elif len(unity_indices) == 1:
            print("✅ GOOD: Exactly one eigenvalue = 1 (unique stationary distribution)")
        else:
            print(f"⚠️  WARNING: {len(unity_indices)} eigenvalues = 1 (reducible chain)")
            print("   Multiple stationary distributions possible.")
            print("   Chain may not mix properly between components.")
        
        # Analyze each unity eigenvalue
        for i, idx in enumerate(unity_indices):
            eval_unity = eigenvalues_sorted[idx]
            evec_unity = eigenvectors_sorted[:, idx]
            
            print(f"\nUnity eigenvalue #{i+1}:")
            print(f"  Value: {eval_unity:.12f}")
            print(f"  Index in sorted list: {idx}")
            
            # Convert eigenvector to probability distribution
            evec_real = np.real(evec_unity)
            
            # Handle sign ambiguity (eigenvectors are only defined up to sign)
            if np.sum(evec_real) < 0:
                evec_real = -evec_real
            
            # Check if all components are non-negative
            has_negative = np.any(evec_real < -1e-10)
            if has_negative:
                print(f"  ⚠️  Eigenvector has negative components!")
                print(f"     Min component: {np.min(evec_real):.2e}")
                print(f"     This suggests numerical issues or non-irreducible chain.")
            
            # Normalize to get probability distribution
            if np.sum(evec_real) > 1e-12:
                pi_candidate = evec_real / np.sum(evec_real)
                pi_candidate = np.maximum(pi_candidate, 0.0)  # Remove tiny negatives
                pi_candidate = pi_candidate / np.sum(pi_candidate)  # Renormalize
                
                # Compute statistics
                states = np.arange(self.N + 1)
                mean_count = np.sum(states * pi_candidate)
                mean_freq = mean_count / self.N
                
                print(f"  Stationary distribution candidate:")
                print(f"    Mean count: {mean_count:.4f}")
                print(f"    Mean frequency: {mean_freq:.4f}")
                print(f"    Probability vector: {pi_candidate}")
                
                # Verify it's actually stationary
                residual = np.linalg.norm(self.P.T @ pi_candidate - pi_candidate)
                print(f"    Verification ||P^T π - π||: {residual:.2e}")
                
                if residual < 1e-10:
                    print(f"    ✅ Valid stationary distribution")
                else:
                    print(f"    ⚠️  Not a valid stationary distribution")
            else:
                print(f"  ⚠️  Eigenvector sums to zero - cannot normalize")
        
        # Second largest eigenvalue (mixing time)
        if len(eigenvalues_sorted) > 1:
            second_largest = np.abs(eigenvalues_sorted[1])
            print(f"\n" + "="*40)
            print(f"MIXING TIME ANALYSIS")
            print("="*40)
            print(f"Second largest eigenvalue magnitude: {second_largest:.8f}")
            
            if second_largest >= 1.0:
                print(f"⚠️  WARNING: Second eigenvalue ≥ 1!")
                print(f"   This indicates multiple unity eigenvalues or numerical issues.")
            else:
                mixing_time = -1 / np.log(second_largest) if second_largest > 0 else np.inf
                print(f"Estimated mixing time: {mixing_time:.1f} generations")
                
                if mixing_time < 100:
                    print(f"✅ Fast mixing (good for numerical stability)")
                elif mixing_time < 1000:
                    print(f"⚠️  Moderate mixing (acceptable)")
                else:
                    print(f"⚠️  Slow mixing (may cause convergence issues)")
        
        print("="*60)
    
    def solve_stationary_direct(self):
        """
        Solve for stationary distribution using eigenvalue decomposition.
        
        The stationary distribution π is the right eigenvector of P with eigenvalue 1:
            π = P π
            
        We find this as the eigenvector corresponding to eigenvalue 1 of P.
        
        Mathematical Notes:
        ------------------
        - For irreducible finite Markov chains, eigenvalue 1 has algebraic 
          and geometric multiplicity 1, and the corresponding eigenvector
          has strictly positive components (Perron-Frobenius theorem)
        - For reducible chains (μ = 0), there may be multiple stationary
          distributions (absorbing states)
        """
        
        if self.P is None:
            raise ValueError("Must construct transition matrix first")
            
        if not self.quiet:
            print("Solving for stationary distribution (eigenvalue method)...")
        
        try:
            # Find eigenvalues and eigenvectors of P (RIGHT eigenvectors)
            # We want π such that P π = π
            eigenvalues, eigenvectors = np.linalg.eig(self.P)
            
            # Find eigenvalue = 1 (mathematically guaranteed to exist)
            unity_indices = np.where(np.abs(eigenvalues - 1.0) < 1e-10)[0]
            if not self.quiet:
                print(f"Found {len(unity_indices)} eigenvalue(s) = 1")
            
            # Use the first one
            idx = unity_indices[0]
            dominant_eigenvalue = eigenvalues[idx]
            stationary_eigenvector = np.real(eigenvectors[:, idx])
            
            # Ensure all probabilities are positive (flip sign if needed)
            if np.sum(stationary_eigenvector) < 0:
                stationary_eigenvector = -stationary_eigenvector
            
            # Normalize to get probabilities
            self.pi_star = stationary_eigenvector / np.sum(stationary_eigenvector)
            
            # Clean up tiny negative values (numerical artifacts)
            self.pi_star = np.maximum(self.pi_star, 0.0)
            self.pi_star = self.pi_star / np.sum(self.pi_star)
            
            self.converged = True
            self.iterations = 1  # Direct method
            
            # Verify solution
            residual = np.linalg.norm(self.P @ self.pi_star - self.pi_star)
            
            if not self.quiet:
                print(f"Eigenvalue solution complete:")
                print(f"  Dominant eigenvalue: {dominant_eigenvalue:.8f} (should be 1.0)")
                print(f"  Residual ||P π - π||: {residual:.2e}")
                print(f"  Normalization Σπ: {np.sum(self.pi_star):.6f}")
                print(f"  All probabilities ≥ 0: {np.all(self.pi_star >= 0)}")
                print(f"  Min probability: {np.min(self.pi_star):.2e}")
                print(f"  Max probability: {np.max(self.pi_star):.2e}")
            
        except np.linalg.LinAlgError as e:
            if not self.quiet:
                print(f"Eigenvalue method failed: {e}")
                print("Falling back to iterative method...")
            return self.solve_stationary_iterative()
            
        return self.pi_star
    
    def solve_stationary_iterative(self):
        """
        Solve for stationary distribution using iterative power method.
        
        The power method iteratively applies the transition matrix:
            π^(k+1) = P π^(k)
            
        For irreducible finite Markov chains, this converges to the unique
        stationary distribution regardless of initial condition.
        
        Algorithm:
        ----------
        1. Initialize π^(0) = uniform distribution
        2. Iterate: π^(k+1) = P π^(k)  
        3. Normalize: π^(k+1) = π^(k+1) / ||π^(k+1)||_1
        4. Check convergence: ||π^(k+1) - π^(k)|| < tolerance
        5. Return π^(k+1) when converged
        
        Convergence Properties:
        ----------------------
        - Rate: Geometric with ratio = |λ_2|, where λ_2 is second-largest eigenvalue
        - For well-conditioned problems: |λ_2| ≪ 1 → fast convergence
        - For nearly-reducible chains: |λ_2| ≈ 1 → slow convergence
        - Mutation helps: μ > 0 typically ensures |λ_2| < 1 - O(μ)
        """
        
        if self.P is None:
            raise ValueError("Must construct transition matrix first")
            
        if not self.quiet:
            print("Solving for stationary distribution (iterative method)...")
        
        # Initialize with uniform distribution
        pi = np.ones(self.N + 1) / (self.N + 1)
        
        for iteration in range(self.max_iter):
            # Apply transition matrix (transpose for left eigenvector)
            pi_new = self.P @ pi
            
            # Normalize to maintain probability distribution
            pi_new = pi_new / np.sum(pi_new)
            
            # Check convergence
            error = norm(pi_new - pi, ord=1)  # L1 norm (total variation distance)
            
            if error < self.tol:
                self.pi_star = pi_new
                self.converged = True
                self.iterations = iteration + 1
                
                if not self.quiet:
                    print(f"Iterative method converged in {self.iterations} iterations")
                    print(f"  Final error: {error:.2e}")
                    print(f"  Normalization: {np.sum(self.pi_star):.6f}")
                
                return self.pi_star
            
            pi = pi_new
            
            # Progress report for slow convergence
            if iteration > 0 and iteration % 1000 == 0 and not self.quiet:
                print(f"  Iteration {iteration}: error = {error:.2e}")
        
        # Did not converge
        if not self.quiet:
            print(f"Warning: Iterative method did not converge after {self.max_iter} iterations")
            print(f"Final error: {error:.2e} (tolerance: {self.tol})")
        self.pi_star = pi_new
        self.converged = False
        self.iterations = self.max_iter
        
        return self.pi_star
    
    def compute_stationary_distribution(self, method='direct'):
        """
        Compute the stationary distribution using specified method.
        
        Parameters:
        -----------
        method : str, optional (default='direct')
            Method for solving: 'direct' (linear algebra) or 'iterative' (power method)
            
        Returns:
        --------
        numpy.ndarray
            Stationary probabilities π[i] for i = 0, 1, ..., N
        """
        
        # Construct transition matrix if needed
        if self.P is None:
            self.construct_transition_matrix()
        
        # Solve using specified method
        if method == 'direct':
            return self.solve_stationary_direct()
        elif method == 'iterative':
            return self.solve_stationary_iterative()
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'direct' or 'iterative'")
    
    def get_stationary_statistics(self):
        """
        Compute summary statistics of the stationary distribution.
        
        Returns:
        --------
        dict
            Dictionary containing:
            - mean: Expected number of "1" alleles
            - variance: Variance in number of "1" alleles  
            - frequency_mean: Expected frequency of "1" allele
            - frequency_variance: Variance in frequency of "1" allele
            - mode: Most probable number of "1" alleles
            - probability_mass: Probability mass at boundaries
        """
        
        if self.pi_star is None:
            raise ValueError("Must compute stationary distribution first")
        
        # State space: number of "1" alleles
        states = np.arange(self.N + 1)
        
        # Moments of allele count distribution
        mean_count = np.sum(states * self.pi_star)
        variance_count = np.sum(states**2 * self.pi_star) - mean_count**2
        
        # Convert to frequency scale
        mean_freq = mean_count / self.N
        variance_freq = variance_count / self.N**2
        
        # Mode (most probable state)
        mode_idx = np.argmax(self.pi_star)
        mode_count = states[mode_idx]
        mode_probability = self.pi_star[mode_idx]
        
        # Boundary behavior
        prob_extinction = self.pi_star[0]      # P(all "0" alleles)
        prob_fixation = self.pi_star[self.N]   # P(all "1" alleles)
        
        return {
            'mean_count': mean_count,
            'variance_count': variance_count,
            'mean_frequency': mean_freq,
            'variance_frequency': variance_freq,
            'mode_count': mode_count,
            'mode_probability': mode_probability,
            'probability_extinction': prob_extinction,
            'probability_fixation': prob_fixation,
            'effective_support': np.sum(self.pi_star > 1e-6)  # States with non-negligible probability
        }
    
    def plot_stationary_distribution(self, show_stats=True, figsize=(14, 6)):
        """
        Plot the stationary distribution as a clear histogram.
        
        Parameters:
        -----------
        show_stats : bool, optional (default=True)
            Whether to display summary statistics on the plot
        figsize : tuple, optional (default=(14, 6))
            Figure size (width, height) in inches
        """
        
        if self.pi_star is None:
            raise ValueError("Must compute stationary distribution first")
        
        # Create single histogram plot
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # States (number of "1" alleles from 0 to N)
        states = np.arange(self.N + 1)
        
        # Create histogram bars with consistent color
        color = sns.color_palette()[0]  # Use first color from seaborn palette
        bars = ax.bar(states, self.pi_star, alpha=0.7, edgecolor='black', 
                     linewidth=0.5, color=color)
        
        # Formatting
        ax.set_xlabel('Number of "1" alleles', fontsize=12)
        ax.set_ylabel('Stationary probability', fontsize=12)
        ax.set_title(f'Steady-State Distribution: Wright-Fisher with Selection\n'
                    f'N={self.N}, s={self.s}, μ={self.mu}', fontsize=14, fontweight='bold')
        
        # Set x-axis to show all states
        ax.set_xlim(-0.5, self.N + 0.5)
        ax.set_xticks(np.arange(0, self.N+1, max(1, self.N//10)))  # Show reasonable number of ticks
        
        # Add statistics if requested
        if show_stats:
            stats = self.get_stationary_statistics()
            
            # Add vertical lines for mean and mode (but don't change bar colors)
            ax.axvline(stats['mean_count'], color='red', linestyle='--', linewidth=2,
                       label=f"Mean: {stats['mean_count']:.2f}")
            ax.axvline(stats['mode_count'], color='darkred', linestyle=':', linewidth=2,
                       label=f"Mode: {stats['mode_count']} (p={stats['mode_probability']:.3f})")
            
            # Add legend
            ax.legend(fontsize=11)
            
            # Text box with comprehensive statistics
            stats_text = (
                f"SOLUTION STATUS:\n"
                f"  Converged: {'Yes' if self.converged else 'No'}\n"
                f"  Method: {'Direct' if self.iterations == 1 else 'Iterative'}\n"
                f"\nDISTRIBUTION STATS:\n"
                f"  Mean count: {stats['mean_count']:.2f}\n"
                f"  Mean frequency: {stats['mean_frequency']:.4f}\n"
                f"  Std deviation: {np.sqrt(stats['variance_count']):.2f}\n"
                f"  Mode: {stats['mode_count']} (prob: {stats['mode_probability']:.3f})\n"
                f"\nBOUNDARY BEHAVIOR:\n"
                f"  P(extinction): {stats['probability_extinction']:.2e}\n"
                f"  P(fixation): {stats['probability_fixation']:.2e}\n"
                f"  States with p>0.1%: {stats['effective_support']}"
            )
            
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                    fontsize=9, family='monospace')
        
        # Improve layout
        plt.tight_layout()
        plt.show()
        
        return fig


def evolve_from_neutral_to_selection(N, mu, s_target=0.005, max_generations=10000, 
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
    s_target : float, optional (default=0.005)
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
                print(f"    Generation {generation}: mean freq = {mean_freq:.4f}, error = {error:.2e}")
                print(f"      D({generation}) = {D_current:.6f}, I({generation}) = {I_current:.6f}, rest({generation}) = {rest_current:.6f}")
                print(f"      Verification: D = I + rest → {D_current:.6f} = {I_current + rest_current:.6f} ✓")
    
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


def plot_evolution_analysis(evolution_results, N, figsize=(15, 12)):
    """
    Plot comprehensive analysis of the evolution from neutral to selection.
    
    Parameters:
    -----------
    evolution_results : dict
        Results from evolve_from_neutral_to_selection()
    N : int
        Population size
    figsize : tuple, optional (default=(15, 12))
        Figure size
    """
    
    phi_star = evolution_results['phi_star']
    psi_star = evolution_results['psi_star']
    psi_trajectory = evolution_results['psi_trajectory']
    generations = evolution_results['generations']
    D_t = evolution_results['D_t']
    I_t = evolution_results['I_t']
    rest_t = evolution_results['rest_t']
    conv_info = evolution_results['convergence_info']
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    states = np.arange(N + 1)
    
    # Plot 1: Compare phi*, psi*, and final psi(t)
    ax1.bar(states - 0.2, phi_star, width=0.4, alpha=0.7, 
            label='φ* (neutral, s=0)', color='blue')
    ax1.bar(states + 0.2, psi_star, width=0.4, alpha=0.7, 
            label='ψ* (selection, s=0.005)', color='red')
    
    # Add final evolved state
    if len(psi_trajectory) > 0:
        final_psi = psi_trajectory[-1]
        ax1.plot(states, final_psi, '-', color='green', linewidth=2,
                label=f'ψ({conv_info["final_generation"]}) (evolved)')
    
    ax1.set_xlabel('Number of "1" alleles')
    ax1.set_ylabel('Probability')
    ax1.set_title('Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: D(t) and I(t) on the same plot
    ax2.plot(generations, D_t, '-', color='blue', linewidth=2, 
            label='D(t) = KL(ψ(t) || φ*)')
    ax2.plot(generations, I_t, '-', color='red', linewidth=2, 
            label='I(t) = KL(ψ(t) || ψ*)')
    
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('KL Divergence [bits]')
    ax2.set_title('KL Divergences: D(t) and I(t)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Evolution of mean frequency over time
    if len(psi_trajectory) > 1:
        mean_frequencies = []
        for psi_t in psi_trajectory:
            mean_freq = np.sum(states * psi_t) / N
            mean_frequencies.append(mean_freq)
        
        ax3.plot(generations, mean_frequencies, '-', color='purple', linewidth=2)
        
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Mean frequency of "1" allele')
        ax3.set_title('Evolution of Mean Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Verification plot - D(t) vs I(t) + rest(t)
    if len(D_t) > 0 and len(I_t) > 0 and len(rest_t) > 0:
        verification = [i + r for i, r in zip(I_t, rest_t)]
        ax4.plot(generations, D_t, '-', color='blue', linewidth=2, 
                label='D(t)')
        ax4.plot(generations, verification, '-', color='green', linewidth=2, 
                label='I(t) + rest(t)')
        
        # Calculate and display the difference
        differences = [abs(d - v) for d, v in zip(D_t, verification)]
        max_diff = max(differences)
        ax4.text(0.02, 0.98, f'Max difference: {max_diff:.2e}', 
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax4.set_xlabel('Generation')
        ax4.set_ylabel('KL Divergence [bits]')
        ax4.set_title('Verification: D(t) = I(t) + rest(t)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig


def compute_kl_divergences(results_dict):
    """
    Compute KL divergences between distributions.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with scenario names as keys and WrightFisherMarkovChain objects as values
        
    Returns:
    --------
    dict : Dictionary with 'D(X)' and 'D(G)' values
    """
    # Get neutral and selection distributions
    neutral_chain = results_dict.get('Neutral')
    selection_chain = results_dict.get('Selection')
    
    if neutral_chain is None or selection_chain is None:
        print("Warning: Need both 'Neutral' and 'Selection' scenarios for KL divergence")
        return {'D(X)': 0.0, 'D(G)': 0.0}
    
    # D(X): KL divergence between full stationary distributions
    # D_KL(selection || neutral) = Σ p_selection * log(p_selection / p_neutral)
    q = neutral_chain.pi_star  # neutral distribution (reference)
    p = selection_chain.pi_star  # selection distribution (approximation)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-12
    p_safe = p + epsilon
    q_safe = q + epsilon
    
    # Normalize
    p_safe = p_safe / np.sum(p_safe)
    q_safe = q_safe / np.sum(q_safe)
    
    # Compute KL divergence D(X): D_KL(selection || neutral)
    kl_div_x = np.sum(p_safe * np.log2(p_safe / q_safe))
    
    # D(G): KL divergence between simplified 0/1 histograms
    # For each distribution, compute probability of 0 and 1 alleles

    # State space: number of "1" alleles
    states = np.arange(neutral_chain.N + 1)
    
    # Moments of allele count distribution
    mean_count_neutral = np.sum(states * neutral_chain.pi_star)
    mean_count_selection = np.sum(states * selection_chain.pi_star)
    
    # Convert to frequencies (divide by N)
    freq_1_neutral = mean_count_neutral / neutral_chain.N
    freq_1_selection = mean_count_selection / selection_chain.N
    
    # Create histograms [P(0), P(1)] - these should be probabilities that sum to 1
    hist_neutral = np.array([1 - freq_1_neutral, freq_1_neutral])  # [P(0), P(1)]
    hist_selection = np.array([1 - freq_1_selection, freq_1_selection])  # [P(0), P(1)]
    
    # Add epsilon and normalize to ensure proper probability distributions
    hist_neutral_safe = hist_neutral + epsilon
    hist_selection_safe = hist_selection + epsilon
    hist_neutral_safe = hist_neutral_safe / np.sum(hist_neutral_safe)
    hist_selection_safe = hist_selection_safe / np.sum(hist_selection_safe)
    
    # Compute KL divergence D(G): D_KL(selection_hist || neutral_hist)
    kl_div_g = np.sum(hist_selection_safe * np.log2(hist_selection_safe / hist_neutral_safe))
    
    return {'D(X)': kl_div_x, 'D(G)': kl_div_g}


def plot_log_comparison(results_dict, N, mu):
    """
    Create a standalone log scale comparison plot with KL divergence values.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with scenario names as keys and WrightFisherMarkovChain objects as values
    N : int
        Population size
    mu : float
        Mutation rate
    """
    
    # Compute KL divergences
    kl_values = compute_kl_divergences(results_dict)
    
    # Create single log scale plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Define colors for each scenario
    colors = sns.color_palette("husl", n_colors=len(results_dict))
    
    # States (number of "1" alleles from 0 to N)
    states = np.arange(N + 1)
    
    # Plot each scenario
    for idx, (scenario_name, chain) in enumerate(results_dict.items()):
        if chain.pi_star is not None:
            # Get statistics
            stats = chain.get_stationary_statistics()
            
            # Plot bars with offset for clarity
            width = 0.35
            offset = (idx - 0.5) * width
            
            bars = ax.bar(states + offset, chain.pi_star, width=width, 
                         alpha=0.7, edgecolor='black', linewidth=0.5,
                         color=colors[idx], label=f"{scenario_name} (s={chain.s})")
            
            # Add mean line
            ax.axvline(stats['mean_count'] + offset, color=colors[idx], 
                      linestyle='--', linewidth=2, alpha=0.8)
    
    # Log scale formatting
    ax.set_xlabel('Number of "1" alleles', fontsize=14)
    ax.set_ylabel('Stationary probability (log scale)', fontsize=14)
    ax.set_title(f'Marginal allele freq. distrib.\nN={N}, μ={mu}', 
                fontsize=16, fontweight='bold')
    ax.set_xlim(-1, N + 1)
    ax.set_xticks(np.arange(0, N+1, max(1, N//10)))
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    
    # Add KL divergence values prominently in the center
    kl_text = f'D(X) = {kl_values["D(X)"]:.4f} bits\nD(G) = {kl_values["D(G)"]:.4f} bits'
    ax.text(0.5, 0.95, kl_text, transform=ax.transAxes, fontsize=14, fontweight='bold',
            horizontalalignment='center', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return fig


def plot_comparison(results_dict, N, mu):
    """
    Create comparison plots with neutral and selection scenarios on the same plot.
    Creates both linear and log scale versions.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with scenario names as keys and WrightFisherMarkovChain objects as values
    N : int
        Population size
    mu : float
        Mutation rate
    """
    
    # Compute KL divergences
    kl_values = compute_kl_divergences(results_dict)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Define colors for each scenario
    colors = sns.color_palette("husl", n_colors=len(results_dict))
    
    # States (number of "1" alleles from 0 to N)
    states = np.arange(N + 1)
    
    # Plot each scenario on both subplots
    for idx, (scenario_name, chain) in enumerate(results_dict.items()):
        if chain.pi_star is not None:
            # Get statistics
            stats = chain.get_stationary_statistics()
            
            # Plot bars with offset for clarity
            width = 0.35
            offset = (idx - 0.5) * width
            
            # Linear scale plot
            bars1 = ax1.bar(states + offset, chain.pi_star, width=width, 
                           alpha=0.7, edgecolor='black', linewidth=0.5,
                           color=colors[idx], label=f"{scenario_name} (s={chain.s})")
            
            # Log scale plot
            bars2 = ax2.bar(states + offset, chain.pi_star, width=width, 
                           alpha=0.7, edgecolor='black', linewidth=0.5,
                           color=colors[idx], label=f"{scenario_name} (s={chain.s})")
            
            # Add mean lines
            ax1.axvline(stats['mean_count'] + offset, color=colors[idx], 
                       linestyle='--', linewidth=2, alpha=0.8)
            ax2.axvline(stats['mean_count'] + offset, color=colors[idx], 
                       linestyle='--', linewidth=2, alpha=0.8)
    
    # Linear scale formatting
    ax1.set_xlabel('Number of "1" alleles', fontsize=14)
    ax1.set_ylabel('Stationary probability', fontsize=14)
    ax1.set_title(f'Comparison: Neutral vs Selection (Linear Scale)\nN={N}, μ={mu}', 
                 fontsize=16, fontweight='bold')
    ax1.set_xlim(-1, N + 1)
    ax1.set_xticks(np.arange(0, N+1, max(1, N//10)))
    ax1.legend(fontsize=12)
    
    # Log scale formatting
    ax2.set_xlabel('Number of "1" alleles', fontsize=14)
    ax2.set_ylabel('Stationary probability (log scale)', fontsize=14)
    ax2.set_title(f'Comparison: Neutral vs Selection (Log Scale)\nN={N}, μ={mu}', 
                 fontsize=16, fontweight='bold')
    ax2.set_xlim(-1, N + 1)
    ax2.set_xticks(np.arange(0, N+1, max(1, N//10)))
    ax2.set_yscale('log')
    ax2.legend(fontsize=12)
    
    # Add KL divergence values to log scale plot (like in reference image)
    kl_text = f'D(X) = {kl_values["D(X)"]:.3f} bits\nD(G) = {kl_values["D(G)"]:.3f} bits'
    ax2.text(0.5, 0.95, kl_text, transform=ax2.transAxes, fontsize=14, fontweight='bold',
             horizontalalignment='center', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Add parameter info in text box on first subplot
    textstr = f'Population size: N = {N}\nMutation rate: μ = {mu}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.show()
    
    return fig


# Convenience quiet API
def compute_stationary_quiet(N: int, s: float, mu: float, method: str = 'direct') -> np.ndarray:
    """Return π* without any prints/plots."""
    mc = WrightFisherMarkovChain(N=N, s=s, mu=mu, quiet=True)
    pi = mc.compute_stationary_distribution(method=method)
    return np.asarray(pi, dtype=float)


def run_markov_analysis_example():
    """
    Example analysis demonstrating the Markov chain approach.
    
    This function runs several example cases to show different biological
    scenarios and compares with Wright-Fisher simulation expectations.
    """
    
    print("="*80)
    print("WRIGHT-FISHER MARKOV CHAIN ANALYSIS")
    print("="*80)
    
    # Example parameters - Compare neutral vs. weak selection
    N = 40  # As requested by user
    mu = 0.0005  # Fixed mutation rate
    scenarios = [
        {'s': 0.0, 'mu': mu, 'name': 'Neutral'},
        {'s': 0.01, 'mu': mu, 'name': 'Selection'}
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n" + "="*60)
        print(f"SCENARIO: {scenario['name']}")
        print(f"N={N}, s={scenario['s']}, μ={scenario['mu']}")
        print("="*60)
        
        # Create Markov chain
        mc = WrightFisherMarkovChain(N=N, s=scenario['s'], mu=scenario['mu'])
        
        # Compute stationary distribution
        pi_star = mc.compute_stationary_distribution(method='direct')
        
        # Get statistics
        stats = mc.get_stationary_statistics()
        
        # Store results
        results[scenario['name']] = {
            'markov_chain': mc,
            'pi_star': pi_star,
            'stats': stats
        }
        
        # Print key results
        print(f"\nKey Results:")
        print(f"  Mean allele frequency: {stats['mean_frequency']:.4f}")
        print(f"  Standard deviation: {np.sqrt(stats['variance_frequency']):.4f}")
        print(f"  Most probable count: {stats['mode_count']}")
        print(f"  P(extinction): {stats['probability_extinction']:.2e}")
        print(f"  P(fixation): {stats['probability_fixation']:.2e}")
        
        # Biological interpretation
        if scenario['s'] > 0:
            print(f"  → Selection favors '1' allele (s={scenario['s']})")
            if scenario['mu'] > 0:
                print(f"  → Mutation-selection balance expected")
            else:
                print(f"  → Fixation of '1' allele expected")
        elif scenario['s'] == 0:
            print(f"  → Neutral evolution")
            if scenario['mu'] > 0:
                print(f"  → Mutation-drift balance around 50%")
        
        # Plot distribution
        print(f"  Plotting stationary distribution...")
        mc.plot_stationary_distribution()
    
    # Summary comparison
    print(f"\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    print(f"{'Scenario':<35} {'Mean Freq':<12} {'Std Dev':<12} {'Mode':<8}")
    print("-" * 70)
    
    for name, result in results.items():
        stats = result['stats']
        print(f"{name:<35} {stats['mean_frequency']:<12.4f} "
              f"{np.sqrt(stats['variance_frequency']):<12.4f} "
              f"{stats['mode_count']:<8d}")
    
    # Create comparison plot
    print("\n" + "="*80)
    print("CREATING COMPARISON PLOT")
    print("="*80)
    mc_objects = {name: result['markov_chain'] for name, result in results.items()}
    
    # Compute and display KL divergences
    kl_values = compute_kl_divergences(mc_objects)
    print(f"KL Divergence Analysis:")
    print(f"  D(X) = {kl_values['D(X)']:.6f} bits (full distribution)")
    print(f"  D(G) = {kl_values['D(G)']:.6f} bits (0/1 histogram)")
    print(f"  Direction: D_KL(neutral || selection)")
    
    # Create standalone log scale plot
    plot_log_comparison(mc_objects, N, mu)
    
    return results

    def evolve_from_neutral_to_selection(self, s_target=0.005, max_generations=10000, 
                                       convergence_tol=1e-8, save_every=10):
        """
        Evolve from neutral steady state (phi*) to selection steady state (psi*).
        
        This method implements the analysis you described:
        1. Calculate phi* (steady state for s=0) 
        2. Use phi* as psi(0) (initial condition)
        3. Run Markov process with s=s_target until convergence to psi*
        4. Track evolution and make comparisons
        
        Parameters:
        -----------
        s_target : float, optional (default=0.005)
            Target selection coefficient to evolve towards
        max_generations : int, optional (default=10000)
            Maximum number of generations to run
        convergence_tol : float, optional (default=1e-8)
            Convergence tolerance for psi* calculation
        save_every : int, optional (default=10)
            Save distribution every N generations for analysis
            
        Returns:
        --------
        dict
            Dictionary containing:
            - phi_star: Neutral steady state distribution (s=0)
            - psi_star: Selection steady state distribution (s=s_target)
            - psi_trajectory: List of distributions over time
            - generations: List of generation numbers
            - D_t: KL divergence D(t) = KL(ψ(t) || φ*)
            - I_t: KL divergence I(t) = KL(ψ(t) || ψ*)
            - rest_t: Cross-term rest = ∫ₓ ψ(t)(x) * log(ψ*(x)/φ*(x))
            - convergence_info: Information about convergence
        """
        
        if not self.quiet:
            print("="*80)
            print("EVOLUTION FROM NEUTRAL TO SELECTION")
            print("="*80)
            print(f"Parameters: N={self.N}, μ={self.mu}")
            print(f"Neutral (s=0) → Selection (s={s_target})")
            print("="*80)
        
        # Step 1: Calculate phi* (neutral steady state, s=0)
        if not self.quiet:
            print("\nStep 1: Calculating φ* (neutral steady state, s=0)")
        
        neutral_chain = WrightFisherMarkovChain(N=self.N, s=0.0, mu=self.mu, quiet=True)
        phi_star = neutral_chain.compute_stationary_distribution(method='direct')
        
        if not self.quiet:
            neutral_stats = neutral_chain.get_stationary_statistics()
            print(f"  φ* calculated: mean freq = {neutral_stats['mean_frequency']:.4f}")
            print(f"  Mode: {neutral_stats['mode_count']}, P(extinction) = {neutral_stats['probability_extinction']:.2e}")
        
        # Step 2: Create selection chain (s=s_target)
        if not self.quiet:
            print(f"\nStep 2: Creating selection chain (s={s_target})")
        
        selection_chain = WrightFisherMarkovChain(N=self.N, s=s_target, mu=self.mu, quiet=True)
        selection_chain.construct_transition_matrix()
        
        # Step 3: Calculate psi* (selection steady state) for comparison
        if not self.quiet:
            print(f"\nStep 3: Calculating ψ* (selection steady state, s={s_target})")
        
        psi_star = selection_chain.compute_stationary_distribution(method='direct')
        
        if not self.quiet:
            selection_stats = selection_chain.get_stationary_statistics()
            print(f"  ψ* calculated: mean freq = {selection_stats['mean_frequency']:.4f}")
            print(f"  Mode: {selection_stats['mode_count']}, P(extinction) = {selection_stats['probability_extinction']:.2e}")
        
        # Step 4: Evolve from phi* (psi(0)) to psi*
        if not self.quiet:
            print(f"\nStep 4: Evolving from φ* to ψ*")
            print(f"  Initial condition: ψ(0) = φ*")
            print(f"  Target: ψ* (s={s_target})")
            print(f"  Max generations: {max_generations}")
            print(f"  Convergence tolerance: {convergence_tol}")
            print(f"  Tracking: D(t), I(t), and rest at every {save_every} generations")
        
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
        
        if not self.quiet:
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
            error = norm(psi_new - psi_star, ord=1)  # L1 distance to target
            
            if error < convergence_tol:
                converged = True
                final_generation = generation
                if not self.quiet:
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
                
                if not self.quiet and generation % (save_every * 10) == 0:
                    # Calculate current statistics
                    states = np.arange(self.N + 1)
                    mean_count = np.sum(states * psi_t)
                    mean_freq = mean_count / self.N
                    print(f"    Generation {generation}: mean freq = {mean_freq:.4f}, error = {error:.2e}")
                    print(f"      D({generation}) = {D_current:.6f}, I({generation}) = {I_current:.6f}, rest({generation}) = {rest_current:.6f}")
                    print(f"      Verification: D = I + rest → {D_current:.6f} = {I_current + rest_current:.6f} ✓")
        
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
        
        if not converged and not self.quiet:
            print(f"  ⚠ Warning: Did not converge within {max_generations} generations")
            print(f"    Final error: {error:.2e}")
        
        # Step 5: Analysis and comparisons
        if not self.quiet:
            print(f"\nStep 5: Analysis and Comparisons")
            
            # Final verification
            print(f"  Final quantities:")
            print(f"    D({final_generation}) = {D_t[-1]:.6f} bits")
            print(f"    I({final_generation}) = {I_t[-1]:.6f} bits")
            print(f"    rest({final_generation}) = {rest_t[-1]:.6f} bits")
            print(f"    Verification D = I + rest: {D_t[-1]:.6f} = {I_t[-1] + rest_t[-1]:.6f} ✓")
            
            # Mean frequency evolution
            final_mean = np.sum(np.arange(self.N + 1) * psi_t) / self.N
            phi_mean = np.sum(np.arange(self.N + 1) * phi_star) / self.N
            psi_mean = np.sum(np.arange(self.N + 1) * psi_star) / self.N
            
            print(f"  Mean frequency evolution:")
            print(f"    φ* (neutral): {phi_mean:.4f}")
            print(f"    ψ* (selection): {psi_mean:.4f}")
            print(f"    ψ({final_generation}) (final): {final_mean:.4f}")
            print(f"    Change: {final_mean - phi_mean:.4f}")
        
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
            },
            'neutral_chain': neutral_chain,
            'selection_chain': selection_chain
        }
        
        return results
    
    def plot_evolution_analysis(self, evolution_results, figsize=(15, 12)):
        """
        Plot comprehensive analysis of the evolution from neutral to selection.
        
        Parameters:
        -----------
        evolution_results : dict
            Results from evolve_from_neutral_to_selection()
        figsize : tuple, optional (default=(15, 12))
            Figure size
        """
        
        phi_star = evolution_results['phi_star']
        psi_star = evolution_results['psi_star']
        psi_trajectory = evolution_results['psi_trajectory']
        generations = evolution_results['generations']
        D_t = evolution_results['D_t']
        I_t = evolution_results['I_t']
        rest_t = evolution_results['rest_t']
        conv_info = evolution_results['convergence_info']
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        states = np.arange(self.N + 1)
        
        # Plot 1: Compare phi*, psi*, and final psi(t)
        ax1.bar(states - 0.2, phi_star, width=0.4, alpha=0.7, 
                label='φ* (neutral, s=0)', color='blue')
        ax1.bar(states + 0.2, psi_star, width=0.4, alpha=0.7, 
                label='ψ* (selection, s=0.005)', color='red')
        
        # Add final evolved state
        if len(psi_trajectory) > 0:
            final_psi = psi_trajectory[-1]
            ax1.plot(states, final_psi, 'o-', color='green', markersize=4,
                    label=f'ψ({conv_info["final_generation"]}) (evolved)')
        
        ax1.set_xlabel('Number of "1" alleles')
        ax1.set_ylabel('Probability')
        ax1.set_title('Distribution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: D(t) and I(t) on the same plot
        ax2.plot(generations, D_t, '-', color='blue', linewidth=2, 
                label='D(t) = KL(ψ(t) || φ*)')
        ax2.plot(generations, I_t, '-', color='red', linewidth=2, 
                label='I(t) = KL(ψ(t) || ψ*)')
        
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('KL Divergence [bits]')
        ax2.set_title('KL Divergences: D(t) and I(t)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Evolution of mean frequency over time
        if len(psi_trajectory) > 1:
            mean_frequencies = []
            for psi_t in psi_trajectory:
                mean_freq = np.sum(states * psi_t) / self.N
                mean_frequencies.append(mean_freq)
            
            ax3.plot(generations, mean_frequencies, '-', color='purple', linewidth=2)
            
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Mean frequency of "1" allele')
            ax3.set_title('Evolution of Mean Frequency')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Verification plot - D(t) vs I(t) + rest(t)
        if len(D_t) > 0 and len(I_t) > 0 and len(rest_t) > 0:
            verification = [i + r for i, r in zip(I_t, rest_t)]
            ax4.plot(generations, D_t, 'o-', color='blue', linewidth=2, 
                    label='D(t)', markersize=4)
            ax4.plot(generations, verification, 's-', color='green', linewidth=2, 
                    label='I(t) + rest(t)', markersize=4)
            
            # Calculate and display the difference
            differences = [abs(d - v) for d, v in zip(D_t, verification)]
            max_diff = max(differences)
            ax4.text(0.02, 0.98, f'Max difference: {max_diff:.2e}', 
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax4.set_xlabel('Generation')
            ax4.set_ylabel('KL Divergence [bits]')
            ax4.set_title('Verification: D(t) = I(t) + rest(t)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig



def run_evolution_example():
    """
    Example demonstrating the evolution from neutral to selection analysis.
    """
    print("="*80)
    print("EVOLUTION FROM NEUTRAL TO SELECTION EXAMPLE")
    print("="*80)
    
    # Parameters
    N = 40
    mu = 0.0005
    s_target = 0.005
    
    print(f"Parameters: N={N}, μ={mu}, s_target={s_target}")
    
    # Run the evolution analysis using the standalone function
    evolution_results = evolve_from_neutral_to_selection(
        N=N, 
        mu=mu,
        s_target=s_target, 
        max_generations=1000, 
        convergence_tol=1e-6, 
        save_every=10
    )
    
    # Plot the results
    plot_evolution_analysis(evolution_results, N)
    
    return evolution_results


if __name__ == "__main__":
    # Run example analysis
    results = run_markov_analysis_example()
    
    # Run evolution example
    print("\n" + "="*80)
    print("RUNNING EVOLUTION EXAMPLE")
    print("="*80)
    evolution_results = run_evolution_example()
