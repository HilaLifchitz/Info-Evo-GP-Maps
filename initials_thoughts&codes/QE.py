
from math import exp, sqrt, pi
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom


#################################################################
# Distribution for the neutral distribution phi(t)
#################################################################

def variance_t(v0_hat, v0, N, t): # variance of the neutral distribution phi(t)
    """
    Compute the variance at time t according to:
    v_phi(t) = v0_hat / 2**t + v0 * sum_{i=1}^t ((1 - 1/N)**i) / 2**(t-i))
    where v0_hat is the variance of the ancestral distribution, 
    v0 is the segeration variance, 
    N is the number of samples, 
    and t is the time.
    """
    sum_term = sum(((1 - 1/N)**i) / 2**(t - i) for i in range(1, t + 1))
    return v0_hat / 2**t + v0 * sum_term


def normal_pdf_at_t(z0, v0_hat, v0, N, t):
    """
    Returns a function f(x) that computes the normal distribution PDF at x
    with mean z0 and variance v_phi(t).
    """
    var = variance_t(v0_hat, v0, N, t)
    def f(x):
        return (1 / sqrt(2 * pi * var)) * exp(-((x - z0) ** 2) / (2 * var))
    return f


def normal_pdf_at_t_scipy(z0, v0_hat, v0, N, t):
    var = variance_t(v0_hat, v0, N, t)
    std = var ** 0.5
    def f(x):
        return norm.pdf(x, loc=z0, scale=std)
    return f




#phi = normal_pdf_at_t_scipy(z0, v0_hat, v0, N, t)
# sanity check
# x = np.linspace(-5, 5, 10000)
# y = phi (x)
# plt.plot(x, y)
# plt.show()

#################################################################
# Distribution for the under selection distribution psi(t)
#################################################################



def fitness_directional(z, beta=1):
    """
    Directional selection fitness: fitness_d(z) = exp(beta * z)
    """
    return exp(beta * z)


def fitness_stabilizing(z, mu_s, v_s):
    """
    Stabilizing selection fitness: fitness_s(z) = N(z; mu_s, v_s)
    Returns the value of the normal PDF at z with mean mu_s and variance v_s.
    """
    std = sqrt(v_s)
    return norm.pdf(z, loc=mu_s, scale=std)

#################################################################
# Simulation of the evolution of trait distributions
#################################################################

class TraitEvolutionSimulator_directional:
    """
    Simulates the evolution of trait distributions under neutral and directional selection,
    and computes the KL divergence between them over time.
    """
    def __init__(self, z0, v0_hat, v0, N, beta, T):
        self.z0 = z0
        self.v0_hat = v0_hat
        self.v0 = v0
        self.N = N
        self.beta = beta
        self.T = T  # number of time steps

        # Storage for results
        self.z_phi = [z0]  # mean for neutral
        self.v_phi = [v0_hat]  # variance for neutral
        self.z_psi = [z0]  # mean for selection
        self.v_psi = [v0_hat]  # variance for selection
        self.kl = []  # KL divergence at each t
        self.kl.append(0)

    def variance_t(self, v0_hat, v0, N, t):
        sum_term = sum(((1 - 1/N)**i) / 2**(t - i) for i in range(1, t + 1))
        return v0_hat / 2**t + v0 * sum_term

    def step(self):
        for t in range(1,self.T):
            # Neutral
            v_phi_t = self.variance_t(self.v0_hat, self.v0, self.N, t)
            self.v_phi.append(v_phi_t)
            self.z_phi.append(self.z0)  # mean stays constant for neutral
            # Selection
            v_psi_t = v_phi_t #
            self.v_psi.append(v_psi_t)
            if t == 0:
                z_psi_t = self.z0
            else:
                z_psi_t = self.beta * self.v_psi[t-1] + self.z_psi[t-1]
            self.z_psi.append(z_psi_t)
            # KL divergence (after both distributions are updated for this step)
            kl_t = self.kl_divergence(self.z_psi[t], self.v_psi[t], self.z_phi[t], self.v_phi[t])
            self.kl.append(kl_t)

    @staticmethod
    # Dkl(psi||phi)
    # KL divergence between two univariate normal distributions:
    # For p(x) = N(mu0, var0), q(x) = N(mu1, var1):
    # D_KL(p || q) = 0.5 * [ var0/var1 + (mu1-mu0)^2/var1 - 1 + ln(var1/var0) ]
    # This is the closed-form analytical solution for the KL divergence between two Gaussians.
    # It avoids numerical integration and is widely used in statistics, machine learning, and information theory.
    def kl_divergence(mu0, var0, mu1, var1):
        # KL(N0 || N1)
        return 0.5 * (var0/var1 + (mu1-mu0)**2/var1 - 1 + np.log(var1/var0))

    def run(self):
        self.step()
        return {
            'z_phi': self.z_phi,
            'v_phi': self.v_phi,
            'z_psi': self.z_psi,
            'v_psi': self.v_psi,
            'kl': self.kl
        }

##################################################################################
# Stabilizing selection
##################################################################################

class StabilizingSelectionSimulator:
    """
    Simulates the evolution of trait distributions under stabilizing selection,
    and computes the KL divergence between them over time.
    The neutral distribution remains constant.

    reminder:
    phi(0)=psi(0)=N(z0, v0_hat)=phi(t) for all t
    v0_hat = the segregation variance of the ancestral distribution 
    N = the size of the population
    z -> N(z; mu_s,v_s) = the stabilizing selection fitness function
    T = the number of time steps
    
    """
    def __init__(self, z0, v0_hat, v0, N, v_s, mu_s, T):
        self.z0 = z0
        self.v0_hat = v0_hat
        self.v0 = v0
        self.N = N
        self.v_s = v_s
        self.mu_s = mu_s
        self.T = T  
        

        # Storage for results
        self.z_phi = [z0] * (T+1)  # mean for neutral (constant)
        self.v_phi = [v0_hat] * (T+1)  # variance for neutral (constant)
        self.z_psi = [z0]  # mean for selection
        self.v_psi = [v0_hat]  # variance for selection
        self.kl = []  # KL divergence at each t
        self.kl.append(0)

    def step(self):
        for t in range(1,self.T):
            # Update psi variance
            v_psi_t = self.v_psi[t-1]
            v_psi_next = 0.5 * (v_psi_t * self.v_s) / (v_psi_t + self.v_s) \
                + self.v0 * (1 - 1/self.N)**(t+1)
            self.v_psi.append(v_psi_next)
            # Update psi mean
            z_psi_t = self.z_psi[t-1]
            z_psi_next = (self.v_s * z_psi_t + self.mu_s * v_psi_t) / (v_psi_t + self.v_s)
            self.z_psi.append(z_psi_next)
            # KL divergence (after update)
            kl_t = self.kl_divergence(
                z_psi_next, v_psi_next, self.z_phi[t], self.v_phi[t]
            )
            self.kl.append(kl_t)


    @staticmethod
    # Dkl(psi||phi)
    # KL divergence between two univariate normal distributions:
    # For p(x) = N(mu0, var0), q(x) = N(mu1, var1):
    # D_KL(p || q) = 0.5 * [ var0/var1 + (mu1-mu0)^2/var1 - 1 + ln(var1/var0) ]
    # This is the closed-form analytical solution for the KL divergence between two Gaussians.
    # It avoids numerical integration and is widely used in statistics, machine learning, and information theory.
    def kl_divergence(mu0, var0, mu1, var1):
        # KL(N0 || N1)
        return 0.5 * (var0/var1 + (mu1-mu0)**2/var1 - 1 + np.log(var1/var0))

    def run(self):
        self.step()
        return {
            'z_phi': self.z_phi,
            'v_phi': self.v_phi,
            'z_psi': self.z_psi,
            'v_psi': self.v_psi,
            'kl': self.kl
        }






##################################################################################
# Plotting the results
##################################################################################


def plot_kl(results):
    """
    Plot the KL divergence D(Z^t) over time from the simulation results.
    """
    kl = results['kl']
    steps = list(range(len(kl)))
    plt.figure(figsize=(8, 5))
    plt.plot(steps, kl, marker='o', linestyle='-', alpha=0.7)
    plt.title(r"$D(Z^t)$")
    plt.xlabel("Iteration Step")
    plt.ylabel("Bits")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()



def plot_distributions_at_t(results, t):
    """
    Plot the neutral (phi) and selected (psi) distributions at time t.
    """
    z_phi = results['z_phi'][t]
    v_phi = results['v_phi'][t]
    z_psi = results['z_psi'][t]
    v_psi = results['v_psi'][t]
    kl = results['kl'][t]   # KL is computed after t=0

    x = np.linspace(
        min(z_phi - 4 * np.sqrt(v_phi), z_psi - 4 * np.sqrt(v_psi)),
        max(z_phi + 4 * np.sqrt(v_phi), z_psi + 4 * np.sqrt(v_psi)),
        500
    )
    phi_pdf = norm.pdf(x, loc=z_phi, scale=np.sqrt(v_phi))
    psi_pdf = norm.pdf(x, loc=z_psi, scale=np.sqrt(v_psi))

    plt.figure(figsize=(4, 4))
    plt.plot(x, phi_pdf, 'b--', label=r'$\phi^{Z^t}$')
    plt.plot(x, psi_pdf, 'r:', label=r'$\psi^{Z^t}$')
    plt.fill_between(x, phi_pdf, color='blue', alpha=0.2)
    plt.fill_between(x, psi_pdf, color='orange', alpha=0.2)
    plt.xlabel("Trait Value")
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"Time Step: {t}, KL Divergence: {kl:.5f}")
    plt.tight_layout()
    plt.show()

    
#################################################################
# An experiment
#################################################################
z0 = 1.5
v0_hat = 0.64
v0 = 0.2
N = 10000
T = 10
beta = 1

simulator_directional = TraitEvolutionSimulator_directional(z0, v0_hat, v0, N, beta, T)
results_directional = simulator_directional.run()

mu_s = 1
v_s = v0_hat/5

simulator_stabilizing = StabilizingSelectionSimulator(z0, v0_hat, v0, N, v_s, mu_s, T)
results_stabilizing = simulator_stabilizing.run()


##################################################################################
# OK REESTABLISHING THE QE WORK IS DONE 
##########################################################################  

# Miso's image 3.6

# subplot A
class WrightFisherSimulator:
    """
    Simulates a haploid Wright-Fisher population with mutation and (optionally) directional selection.
    Tracks the empirical phenotype distribution (number of 1-alleless per individual) and KL divergence to Binom(l, 0.5) at each step.
    """
        # === Reproduction Step ===
    # 1. Compute fitness of each individual: w_i = (1 + s)^{# of 1 alleles}
    # 2. Normalize fitness to get probabilities: rel_fit_i = w_i / sum(w)
    # 3. Sample N parents with replacement, weighted by rel_fit
    # 4. Copy their genomes to offspring
    # 5. Mutate offspring: each locus has probability mu to flip (0 ↔ 1)
    # Result: new genomes array becomes the next generation

    def __init__(self, N, l, s, mu, T, seed=999):
        self.N = N  # population size
        self.l = l  # number of loci
        self.s = s  # selection coefficient (0 for neutral)
        self.mu = mu  # mutation rate per locus per generation
        self.T = T  # number of generations
        self.rng = np.random.default_rng(seed) # random object generator
        # Storage
        self.phenotype_counts = []  # list of arrays: counts of each phenotype (0..l) at each t
        self.phenotype_probs = []   # list of arrays: empirical probability of each phenotype at each t
        self.kl = []                # KL divergence to Binom(l, 0.5) at each t

    def _fitness(self, genomes):
        # genomes: (N, l) array
        g_sum = genomes.sum(axis=1)  # number of 1s per individual
        w = (1 + self.s) ** g_sum
        return w

    def _relative_fitness(self, w):
        return w / w.sum()

    def _mutate(self, genomes):
        # genomes: (N, l) array
        mutation_mask = self.rng.random(genomes.shape) < self.mu # chances of being TRUE ~= mu, all else is FLASE
        genomes_flipped = np.where(mutation_mask, 1 - genomes, genomes) # if TRUE flips 1 <-->0
        return genomes_flipped

    def _empirical_distribution(self, genomes):
        # Returns: counts (length l+1), probs (length l+1)
        pheno = genomes.sum(axis=1) # sums each row= genom of an individual --> its sum is thoe phenotypic value
        counts = np.bincount(pheno, minlength=self.l+1) #counts[x]= how many individuals have counts == x
        probs = counts / self.N
        return counts, probs

    def _kl_divergence(self, p, q):
        # p, q: arrays of probabilities (length l+1)
        # D_KL(p || q) - NEW METHOD: strict masking, no epsilon
 

        # Add epsilon only to non-zero values
        epsilon = 1e-12 # small number to avoid log(0)
        p = np.where(p == 0, p + epsilon, p) # add epsilon to zero values
        q = np.where(q == 0, q + epsilon, q) # add epsilon to zero values
        
        # Re-normalize after adding epsilon
        p = p / p.sum()  # normalize
        q = q / q.sum()  # normalize
        mask = (p > 0) & (q > 0)  # Only where BOTH are non-zero (which should be everywhere, but just in case)
        if not np.any(mask):
            return 0.0
        return np.sum(p[mask] * np.log2(p[mask] / q[mask])) # bits = the information units in log2

    def run(self):
        # Initialize population: N x l, each locus Bernoulli(0.5)
        genomes = self.rng.integers(0, 2, size=(self.N, self.l))
        # Binomial reference distribution q=phi
        q = binom.pmf(np.arange(self.l+1), self.l, 0.5)
        for t in range(self.T+1):
            counts, probs = self._empirical_distribution(genomes)
            self.phenotype_counts.append(counts) # the phenotype itself is how many 1- alleles one has
            self.phenotype_probs.append(probs)
            # KL divergence between empirical distribution and reference distribution
            kl = self._kl_divergence(probs, q)
            self.kl.append(kl)
            if t == self.T:
                break
            # Fitness
            w = self._fitness(genomes)
            rel_fit = self._relative_fitness(w)
            # Parent selection
            parent_indices = self.rng.choice(self.N, size=self.N, p=rel_fit)
            offspring = genomes[parent_indices].copy()
            # Mutation
            genomes = self._mutate(offspring)
        return {
            'phenotype_counts': self.phenotype_counts,
            'phenotype_probs': self.phenotype_probs,
            'kl': self.kl,
            'N': self.N,
            'l': self.l,
            'mu': self.mu,
            's': self.s
        }

class WrightFisherSimulator_FreeRecombination:
    """
    Wright-Fisher simulator with FREE RECOMBINATION between all loci.
    
    Key difference from standard WF: Each offspring has TWO parents instead of one!
    
    Reproduction process:
    1. Select father with probability proportional to fitness
    2. Select mother with probability proportional to fitness  
    3. For each locus, offspring inherits from father (50%) or mother (50%)
    4. Apply mutation to offspring genome
    
    This models sexual reproduction with free recombination, which can significantly
    affect the dynamics of trait evolution and potentially change convergence patterns.
    
    Biological significance:
    - Breaks up linkage between beneficial/deleterious alleles
    - Increases effective recombination rate to maximum (0.5 per locus pair)
    - May lead to faster equilibration or different equilibrium distributions
    - Classic model in population genetics for sexual vs asexual reproduction
    """
    
    def __init__(self, N, l, s, mu, T, seed=999):
        self.N = N  # population size
        self.l = l  # number of loci
        self.s = s  # selection coefficient (0 for neutral)
        self.mu = mu  # mutation rate per locus per generation
        self.T = T  # number of generations
        self.rng = np.random.default_rng(seed) # random object generator
        # Storage
        self.phenotype_counts = []  # list of arrays: counts of each phenotype (0..l) at each t
        self.phenotype_probs = []   # list of arrays: empirical probability of each phenotype at each t
        self.kl = []                # KL divergence to Binom(l, 0.5) at each t

    def _fitness(self, genomes):
        # genomes: (N, l) array
        g_sum = genomes.sum(axis=1)  # number of 1s per individual
        w = (1 + self.s) ** g_sum
        return w

    def _relative_fitness(self, w):
        return w / w.sum()

    def _mutate(self, genomes):
        # genomes: (N, l) array
        mutation_mask = self.rng.random(genomes.shape) < self.mu # chances of being TRUE ~= mu, all else is FALSE
        genomes_flipped = np.where(mutation_mask, 1 - genomes, genomes) # if TRUE flips 1 <-->0
        return genomes_flipped

    def _empirical_distribution(self, genomes):
        # Returns: counts (length l+1), probs (length l+1)
        g_sum = genomes.sum(axis=1) # sums each row = genome of an individual
        counts = np.bincount(g_sum, minlength=self.l+1) #counts[x]= how many individuals have counts == x
        probs = counts / self.N
        return counts, probs

    def _kl_divergence(self, p, q):
        # p, q: arrays of probabilities (length l+1)
        # D_KL(p || q) - Same method as original WF simulator
        epsilon = 1e-12 # small number to avoid log(0)
        p = np.where(p == 0, p + epsilon, p) # add epsilon to zero values
        q = np.where(q == 0, q + epsilon, q) # add epsilon to zero values
        
        # Re-normalize after adding epsilon
        p = p / p.sum()  # normalize
        q = q / q.sum()  # normalize
        mask = (p > 0) & (q > 0)  # Only where BOTH are non-zero (which should be everywhere, but just in case)
        if not np.any(mask):
            return 0.0
        return np.sum(p[mask] * np.log2(p[mask] / q[mask])) # bits = the information units in log2

    def _reproduce_with_recombination(self, genomes):
        """
        TWO-PARENT REPRODUCTION WITH FREE RECOMBINATION
        
        For each offspring:
        1. Select father proportional to fitness
        2. Select mother proportional to fitness  
        3. At each locus: 50% chance inherit from father, 50% from mother
        
        This breaks up all linkage and maximizes recombination rate.
        """
        # Calculate fitness for all individuals
        w = self._fitness(genomes)
        rel_fit = self._relative_fitness(w)
        
        # Create offspring array
        offspring = np.zeros((self.N, self.l), dtype=int)
        
        # For each offspring, select two parents and recombine
        for offspring_idx in range(self.N):
            # Select father and mother independently, weighted by fitness
            father_idx = self.rng.choice(self.N, p=rel_fit)
            mother_idx = self.rng.choice(self.N, p=rel_fit)
            
            # Get parental genomes
            father_genome = genomes[father_idx]
            mother_genome = genomes[mother_idx]
            
            # For each locus, inherit from father (50%) or mother (50%)
            inheritance_mask = self.rng.random(self.l) < 0.5  # True = inherit from father
            
            offspring_genome = np.where(inheritance_mask, father_genome, mother_genome)
            offspring[offspring_idx] = offspring_genome
            
        return offspring

    def run(self):
        # Initialize population: N x l, each locus Bernoulli(0.5)
        genomes = self.rng.integers(0, 2, size=(self.N, self.l))
        # Binomial reference distribution q=phi
        q = binom.pmf(np.arange(self.l+1), self.l, 0.5)
        
        for t in range(self.T+1):
            counts, probs = self._empirical_distribution(genomes)
            self.phenotype_counts.append(counts) # the phenotype itself is how many 1- alleles one has
            self.phenotype_probs.append(probs)
            # KL divergence between empirical distribution and reference distribution
            kl = self._kl_divergence(probs, q)
            self.kl.append(kl)
            if t == self.T:
                break
                
            # === NEW REPRODUCTION STEP WITH FREE RECOMBINATION ===
            offspring = self._reproduce_with_recombination(genomes)
            
            # Mutation (same as before)
            genomes = self._mutate(offspring)
            
        return {
            'phenotype_counts': self.phenotype_counts,
            'phenotype_probs': self.phenotype_probs,
            'kl': self.kl,
            'N': self.N,
            'l': self.l,
            'mu': self.mu,
            's': self.s
        }


N=40
l=1000
s=0.01
mu=0.02/40 # 0.0005
T=1500

sim = WrightFisherSimulator(N, l, s, mu, T)
res = sim.run()



def kl_divergence(p, q):
    """Strict masking method - no epsilon smoothing"""
    p = p / p.sum()
    q = q / q.sum()
    mask = (p > 0) & (q > 0)  # Only where BOTH are non-zero
    if not np.any(mask):
        return 0.0
    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))


def kl_divergence_epsilon(p, q):
        # p, q: arrays of probabilities (length l+1)
        # D_KL(p || q)
        #p=p+1e-12 # avoid log(0) # EITHER THIS OR THE MASK BELOW
    mask = p > 0
        #return np.sum(p[mask] * np.log(p[mask] / q[mask])) # Nats =  the information units with natural log
    return np.sum(p[mask] * np.log2(p[mask] / q[mask])) # bits = the information units in log2

# # Stack phenotype probabilities into a 2D array: shape (T+1, l+1)
# phenotype_matrix = np.array(res['phenotype_probs'])  # shape: (T+1, l+1)

# fig, ax = plt.subplots(figsize=(12, 7))
# ax.set_facecolor('white')
# fig.patch.set_facecolor('white')
# plt.imshow(
#     phenotype_matrix.T,  # transpose so y-axis is phenotype
#     aspect='auto',
#     origin='lower',
#     cmap='Reds',
#     interpolation='nearest',
#     extent=[0, phenotype_matrix.shape[0], 0, phenotype_matrix.shape[1]]
# )
# plt.colorbar(label='Proportion of Population')
# plt.xlabel('Generation')
# plt.ylabel('Number of 1 alleles (z)')
# plt.title('Stochastic Phenotype Trajectory (Proportions)')
# plt.show()
###########################################################################
# Miso's image 3.6- PLOTTING!!

# counts
def plot_phenotype_counts(res): 
    """
    Plot heatmap of phenotype evolution over time.
    
    Shows stochastic trajectory of phenotype counts (number of 1-alleles per individual)
    across generations. X-axis: generations, Y-axis: phenotype values (400-800),
    color intensity: number of individuals with each phenotype.
    
    Reveals: initial binomial distribution → directional selection shift → 
    stochastic equilibrium with drift fluctuations.
    """
    phenotype_matrix = np.array(res['phenotype_counts'])  # shape: (T+1, l+1)
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Restrict to z-values 400-800
    z_min, z_max = 400, 800
    phenotype_matrix = phenotype_matrix[:, z_min:z_max+1]  # restrict to desired range

    # Custom colormap: start with white
    cmap = plt.get_cmap('Reds').copy()
    cmap.set_under('white')

    im = ax.imshow(
        phenotype_matrix.T,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        interpolation='nearest',
        extent=[0, phenotype_matrix.shape[0], z_min, z_max+1],
        alpha=1,
        vmin=0.01  # so that zero values are truly white
    )
    fig.colorbar(im, ax=ax, label='Number of Individuals')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Number of 1 alleles (z)')
    ax.set_title('Stochastic Phenotype Trajectory')
    ax.set_ylim(z_min, z_max)

    # Add parameter legend (LaTeX formatted)
    param_text = (
        r"$N = %d$" "\n"
        r"$l = %d$" "\n"
        r"$\mu = %.1e$" "\n"
        r"$s = %.3f$"
    ) % (res['N'], res['l'], res['mu'], res['s'])
    ax.text(
        0.02, 0.98, param_text,
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    return fig


def moving_average(data, window_size):
    """
    Compute the moving average of the data with the given window size.
    For each point, it averages over a window of 'window_size' centered at that point.
    The result is shorter by window_size-1 points.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def plot_kl_trajectory(res, smooth_window=201, poly_degree=3):
    """
    Plot KL divergence over generations, with a moving average and a polynomial fit.
    """
    kl = np.array(res['kl'])
    generations = np.arange(len(kl))
    plt.figure(figsize=(10, 5))
    plt.plot(generations, kl, linestyle='-', color='tab:blue', alpha=0.5, label='KL')
    
    # Moving average smoothing
    if smooth_window > 1 and smooth_window < len(kl):
        kl_smooth = moving_average(kl, smooth_window)
        offset = (smooth_window - 1) // 2
        gen_smooth = generations[offset:offset+len(kl_smooth)]
        plt.plot(gen_smooth, kl_smooth, 'k--', linewidth=2, label=f'Smoothed KL (window={smooth_window})')
    
    # Polynomial fit
    if poly_degree is not None and poly_degree > 0:
        coeffs = np.polyfit(generations, kl, deg=poly_degree)
        kl_poly = np.polyval(coeffs, generations)
        plt.plot(generations, kl_poly, 'r-', linewidth=2, label=f'Poly fit (deg={poly_degree})')
    
    plt.xlabel('Generation')
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence vs. Generation')
    plt.grid(True, alpha=0.3)
    plt.legend()
    # Add parameter legend (LaTeX formatted)
    param_text = (
        r"$N = %d$" "\n"
        r"$l = %d$" "\n"
        r"$\mu = %.1e$" "\n"
        r"$s = %.3f$"
    ) % (res['N'], res['l'], res['mu'], res['s'])
    plt.gca().text(
        0.02, 0.98, param_text,
        transform=plt.gca().transAxes,
        fontsize=16,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    plt.tight_layout()
    plt.show()


def plot_abs_delta_kl(res, smooth_window=201):
    kl = np.array(res['kl'])
    abs_delta_kl = np.abs(np.diff(kl))
    generations = np.arange(1, len(kl))
    # Smoothing
    if smooth_window > 1 and smooth_window < len(abs_delta_kl):
        abs_delta_kl_smooth = moving_average(abs_delta_kl, smooth_window)
        offset = (smooth_window - 1) // 2
        gen_smooth = generations[offset:offset+len(abs_delta_kl_smooth)]
    else:
        abs_delta_kl_smooth = abs_delta_kl
        gen_smooth = generations

    plt.figure(figsize=(10, 4))
    plt.plot(generations, abs_delta_kl, color='purple', alpha=0.4, label=r'$|\Delta D(Z)|$')
    plt.plot(gen_smooth, abs_delta_kl_smooth, color='orange', linewidth=2, label=f'Smoothed $|\\Delta D(Z)|$ (window={smooth_window})')
    plt.xlabel('Generation')
    plt.ylabel(r'$|\Delta D(Z)|$ (bits)')
    plt.title('Absolute Change in KL Divergence per Generation')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_delta_kl(res, smooth_window=201, t_start=None, t_end=None):
    """
    Plot the raw difference in KL divergence (D(Z)) between generations.
    
    Parameters:
    - res: simulation results
    - smooth_window: window size for smoothing (default: 201)
    - t_start: start generation for x-axis focus (default: None = start from beginning)
    - t_end: end generation for x-axis focus (default: None = end at final generation)
    """

    kl = np.array(res['kl'])
    delta_kl = np.diff(kl)
    generations = np.arange(1, len(kl))
    
    # Apply time window focus if specified
    if t_start is not None or t_end is not None:
        if t_start is None:
            t_start = 1
        if t_end is None:
            t_end = len(kl) - 1
        
        # Validate time window
        t_start = max(1, min(t_start, len(kl) - 1))
        t_end = max(t_start, min(t_end, len(kl) - 1))
        
        # Filter data to time window
        mask = (generations >= t_start) & (generations <= t_end)
        generations = generations[mask]
        delta_kl = delta_kl[mask]
    
    # Smoothing
    if smooth_window > 1 and smooth_window < len(delta_kl):
        delta_kl_smooth = moving_average(delta_kl, smooth_window)
        offset = (smooth_window - 1) // 2
        gen_smooth = generations[offset:offset+len(delta_kl_smooth)]
    else:
        delta_kl_smooth = delta_kl
        gen_smooth = generations

    plt.figure(figsize=(10, 4))
    plt.plot(generations, delta_kl, color='teal', alpha=0.4, label=r'$\Delta D(Z)$')
    plt.plot(gen_smooth, delta_kl_smooth, color='red', linewidth=2, label=f'Smoothed $\\Delta D(Z)$ (window={smooth_window})')
    plt.xlabel('Generation')
    plt.ylabel(r'$\Delta D(Z)$ (bits)')
    
    # Update title based on time window
    if t_start is not None or t_end is not None:
        plt.title(f'Change in KL Divergence per Generation (t={t_start}-{t_end})')
    else:
        plt.title('Change in KL Divergence per Generation')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
