#!/usr/bin/env python3

import numpy as np
from scipy.stats import binom

def quick_test_for_80_bits():
    """Quick tests to find parameter combination giving ~80 bits"""
    print("QUICK TEST: Finding parameter combination for ~80 bits KL")
    print("="*60)
    
    # Test different selection strengths with your current setup
    s_values = [0.015, 0.02, 0.025, 0.03, 0.035]
    
    print("1. Testing different selection strengths (s):")
    for s in s_values:
        # Quick simulation
        N, l, mu, T = 40, 1000, 0.0005, 1000
        
        np.random.seed(42)
        genomes = np.random.randint(0, 2, size=(N, l))
        q = binom.pmf(np.arange(l+1), l, 0.5)
        
        for t in range(T):
            if t == T-1:
                g_sum = genomes.sum(axis=1)
                counts = np.bincount(g_sum, minlength=l+1)
                probs = counts / N
                
                kl = np.sum(probs[probs>0] * np.log2(probs[probs>0] / q[probs>0]))
                mean_emp = np.sum(probs * np.arange(len(probs)))
                
                print(f"  s={s:.3f}: KL={kl:6.2f} bits, mean={mean_emp:6.1f}")
                
                if 75 <= kl <= 85:
                    print(f"    ✅ BINGO! s={s} gives ~80 bits!")
                break
            
            # Evolution
            g_sum = genomes.sum(axis=1)
            w = (1 + s) ** g_sum
            w = w / w.sum()
            
            parent_indices = np.random.choice(N, size=N, p=w)
            offspring = genomes[parent_indices].copy()
            
            mutation_mask = np.random.random(offspring.shape) < mu
            genomes = np.where(mutation_mask, 1 - offspring, offspring)
    
    # Test different reference distributions
    print(f"\n2. Testing different reference distributions:")
    
    # Create empirical distribution from current parameters
    N, l, s, mu, T = 40, 1000, 0.01, 0.0005, 1000
    np.random.seed(42)
    genomes = np.random.randint(0, 2, size=(N, l))
    
    for t in range(T):
        if t == T-1:
            g_sum = genomes.sum(axis=1)
            counts = np.bincount(g_sum, minlength=l+1)
            probs = counts / N
            break
        
        g_sum = genomes.sum(axis=1)
        w = (1 + s) ** g_sum
        w = w / w.sum()
        
        parent_indices = np.random.choice(N, size=N, p=w)
        offspring = genomes[parent_indices].copy()
        
        mutation_mask = np.random.random(offspring.shape) < mu
        genomes = np.where(mutation_mask, 1 - offspring, offspring)
    
    # Test different reference distributions
    references = [
        (0.45, 1000), (0.4, 1000), (0.35, 1000),
        (0.5, 800), (0.5, 1200), (0.5, 1500)
    ]
    
    for p, l_ref in references:
        if l_ref <= 1000:  # Can't have reference longer than empirical
            q = binom.pmf(np.arange(1001), l_ref, p)
            if len(q) < len(probs):
                q = np.pad(q, (0, len(probs) - len(q)), 'constant')
            else:
                q = q[:len(probs)]
            
            q = q / q.sum()  # Normalize
            
            mask = (probs > 0) & (q > 0)
            if np.any(mask):
                kl = np.sum(probs[mask] * np.log2(probs[mask] / q[mask]))
                print(f"  Binom(l={l_ref}, p={p}): KL={kl:6.2f} bits")
                
                if 75 <= kl <= 85:
                    print(f"    ✅ BINGO! Reference Binom({l_ref}, {p}) gives ~80 bits!")

    print(f"\n3. Most likely explanation:")
    print(f"   - Your old code used stronger selection (s ≈ 0.02-0.03)")
    print(f"   - OR different reference distribution parameters")
    print(f"   - Check your old binomial reference: binom.pmf(range(???), ???, ???)")

if __name__ == "__main__":
    quick_test_for_80_bits()