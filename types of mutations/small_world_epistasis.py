"""
Generate a small-world epistasis matrix and test mutation-effect variability.

Biological model
----------------
Each node is a locus. Allele 0 is the wild type and allele 1 is the mutation.
The script first creates a connected Watts-Strogatz small-world graph with
NetworkX. The graph topology decides which pairwise epistatic entries in M are
allowed to be nonzero.

For each graph edge (i, j), we sample an epistatic effect epsilon_ij. Because
g.T @ M @ g counts each undirected pair twice, the matrix stores:

    M[i, j] = M[j, i] = epsilon_ij / 2

The diagonal stores single-locus selection coefficients:

    M[i, i] = s_i

For each haploid genotype g in {0,1}^n:

    log_fitness(g) = g.T @ M @ g
    fitness(g)     = exp(g.T @ M @ g)

House-of-cards-style question
-----------------------------
For each locus i, and for all 2^(n-1) backgrounds where g_i = 0, we flip that
locus from 0 to 1 and measure:

    effect_i(g)   = log_fitness(g with g_i = 1) - log_fitness(g)
    surprise_i(g) = effect_i(g) - average_background_effect_i

If the effect variance/surprise is large, the mutation has strongly
background-dependent effects. If parent and mutant log fitness are weakly
correlated, the landscape is closer to the classical House-of-Cards idea where
neighboring genotypes behave almost independently.

Example with normally distributed effects:
    python3 small_world_epistasis.py \
        --n 12 \
        --k 4 \
        --rewire-prob 0.05 \
        --edge-distribution normal \
        --edge-mean 0.0 \
        --edge-std 0.1 \
        --selection-distribution normal \
        --selection-mean 0.0 \
        --selection-std 0.05 \
        --seed 42 \
        --results-dir results \
        --out-prefix sw12

This creates:
    results/YYYY-MM-DD_HH-MM-SS/sw12_M.npy
    results/YYYY-MM-DD_HH-MM-SS/sw12_M.csv
    results/YYYY-MM-DD_HH-MM-SS/sw12_edges.csv
    results/YYYY-MM-DD_HH-MM-SS/sw12_selection_coefficients.csv
    results/YYYY-MM-DD_HH-MM-SS/sw12_genotype_fitness.csv  # only when 2^n is not too large
    results/YYYY-MM-DD_HH-MM-SS/sw12_mutation_surprise.csv  # only when 2^n is not too large
    results/YYYY-MM-DD_HH-MM-SS/sw12_mutation_surprise_by_locus.csv
    results/YYYY-MM-DD_HH-MM-SS/sw12_metrics.json
    results/YYYY-MM-DD_HH-MM-SS/sw12_run_report.txt
    results/YYYY-MM-DD_HH-MM-SS/sw12_equilibria_config.yaml
    results/YYYY-MM-DD_HH-MM-SS/plots/
"""
# This script generates a weighted small-world epistatic interaction network
# between loci and converts it into an epistasis matrix M.
#
# The biological edge weight epsilon_ij represents the pairwise epistatic
# interaction between loci i and j. Because the quadratic form g.T @ M @ g
# counts each undirected pair twice, we store the interaction in M as
#
#     M[i, j] = M[j, i] = epsilon_ij / 2.
#
# With haploid genotypes g in {0,1}^n, this gives
#
#     g.T @ M @ g
#       = sum_i M[i, i] g_i
#         + sum_{i<j} epsilon_ij g_i g_j.
#
# The diagonal entries M[i, i] are the single-locus selection coefficients s_i.
# Therefore the genotype fitness used here is
#
#     log_fitness(g) = g.T @ M @ g
#     fitness(g)     = exp(g.T @ M @ g).
#
# The script also computes "mutation surprise" for 0 -> 1 mutations. For each
# locus i and each genetic background with g_i = 0, it compares the log-fitness
# before and after flipping that locus:
#
#     effect_i(g) = log_fitness(g with g_i = 1) - log_fitness(g).
#
# The surprise is the deviation of this effect from the average effect of that
# mutation over all backgrounds:
#
#     surprise_i(g) = effect_i(g) - average_background_effect_i.
#
# Large surprise variance means that the effect of a mutation depends strongly
# on the rest of the genotype, i.e. strong background-dependent epistasis.
#
# Note: by default, this averages over all possible haploid backgrounds
# uniformly. For large n, full enumeration over 2^n genotypes is skipped, and
# Monte Carlo or equilibrium-weighted summaries may be more appropriate.

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

MetricValue = float | bool | int | str


def generate_connected_small_world_graph(
    n: int,
    k: int,
    rewire_prob: float,
    rng: np.random.Generator,
    max_attempts: int = 100,
) -> nx.Graph:
    """
    Generate a connected Watts-Strogatz graph with networkx.

    k is the number of nearest neighbors in the starting ring lattice.
    NetworkX requires k to be even for the usual symmetric ring construction.
    """
    if n < 3:
        raise ValueError("n must be at least 3")
    if k < 2 or k >= n:
        raise ValueError("k must satisfy 2 <= k < n")
    if k % 2 != 0:
        raise ValueError("k must be even for this ring-lattice construction")
    if not 0.0 <= rewire_prob <= 1.0:
        raise ValueError("rewire_prob must be between 0 and 1")

    for _ in range(max_attempts):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        graph = nx.watts_strogatz_graph(n=n, k=k, p=rewire_prob, seed=seed)
        if nx.is_connected(graph):
            return graph
    raise RuntimeError(f"Could not generate a connected graph after {max_attempts} attempts")


def weighted_adjacency_from_graph(
    graph: nx.Graph,
    rng: np.random.Generator,
    distribution: str = "uniform",
    weight_low: float = -1.0,
    weight_high: float = 1.0,
    weight_mean: float = 0.0,
    weight_std: float = 0.1,
) -> np.ndarray:
    """Assign one random undirected weight to every edge."""
    n = graph.number_of_nodes()
    adjacency = np.zeros((n, n), dtype=float)
    weights = sample_random_values(
        rng,
        size=graph.number_of_edges(),
        distribution=distribution,
        low=weight_low,
        high=weight_high,
        mean=weight_mean,
        std=weight_std,
        label="edge weights",
    )
    for (i, j), weight in zip(graph.edges(), weights):
        adjacency[i, j] = weight
        adjacency[j, i] = weight
    return adjacency


def sample_random_values(
    rng: np.random.Generator,
    size: int,
    distribution: str,
    low: float,
    high: float,
    mean: float,
    std: float,
    label: str,
) -> np.ndarray:
    """Sample model parameters from a named distribution."""
    if distribution == "uniform":
        if low > high:
            raise ValueError(f"{label}: low must be <= high")
        return rng.uniform(low, high, size=size)
    if distribution == "normal":
        if std < 0:
            raise ValueError(f"{label}: std must be non-negative")
        return rng.normal(mean, std, size=size)
    raise ValueError(f"{label}: distribution must be 'uniform' or 'normal'")


def epistasis_matrix_from_weighted_adjacency(
    weighted_adjacency: np.ndarray,
    diagonal_selection: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Convert undirected edge weights epsilon_ij to M_ij = epsilon_ij / 2."""
    weighted_adjacency = np.asarray(weighted_adjacency, dtype=float)
    if weighted_adjacency.ndim != 2 or weighted_adjacency.shape[0] != weighted_adjacency.shape[1]:
        raise ValueError("weighted_adjacency must be a square matrix")

    n = weighted_adjacency.shape[0]
    M = 0.5 * weighted_adjacency
    diagonal = np.asarray(diagonal_selection, dtype=float)
    if diagonal.ndim == 0:
        np.fill_diagonal(M, float(diagonal))
    elif diagonal.shape == (n,):
        np.fill_diagonal(M, diagonal)
    else:
        raise ValueError(f"diagonal_selection must be a scalar or length-{n} vector")
    return M


def random_selection_coefficients(
    n: int,
    rng: np.random.Generator,
    distribution: str,
    selection_low: float,
    selection_high: float,
    selection_mean: float,
    selection_std: float,
) -> np.ndarray:
    """Draw random single-locus coefficients s_i for the diagonal of M."""
    return sample_random_values(
        rng,
        size=n,
        distribution=distribution,
        low=selection_low,
        high=selection_high,
        mean=selection_mean,
        std=selection_std,
        label="selection coefficients",
    )


def all_haploid_genotypes(n: int, max_genotypes: int) -> np.ndarray:
    """All haploid genotypes g in {0,1}^n, if 2^n is computationally reasonable."""
    n_genotypes = 2**n
    if n_genotypes > max_genotypes:
        raise ValueError(
            f"Cannot enumerate 2^{n} = {n_genotypes} genotypes; "
            f"increase --max-genotypes above {max_genotypes} if this is intentional."
        )
    indices = np.arange(n_genotypes, dtype=np.uint64)[:, None]
    bit_positions = np.arange(n - 1, -1, -1, dtype=np.uint64)
    return ((indices >> bit_positions) & 1).astype(np.int8)


def genotype_fitness_table(M: np.ndarray, max_genotypes: int) -> np.ndarray | None:
    """
    Return rows [genotype bits..., log_fitness, fitness] for all haploid genotypes.

    log_fitness(g) = g.T @ M @ g
    fitness(g) = exp(log_fitness(g))
    """
    n = M.shape[0]
    try:
        G = all_haploid_genotypes(n, max_genotypes)
    except ValueError:
        return None

    G_float = G.astype(float)
    log_fitness = np.einsum("bi,ij,bj->b", G_float, M, G_float)
    fitness = np.exp(log_fitness)
    return np.column_stack([G, log_fitness, fitness])


def information_epistasis_table(
    M: np.ndarray,
    max_genotypes: int,
    allele_frequency: float,
) -> np.ndarray | None:
    """
    Compute D(g_k | g_-k) - D(g_k) for each locus.

    Baseline phi is an independent Bernoulli genotype distribution with
    P(g_i=1)=allele_frequency. Psi is the fitness-weighted distribution:

        psi(g) proportional to phi(g) * exp(g.T @ M @ g)

    The returned columns are:
        locus, D_gk, D_gk_given_background, information_epistasis
    """
    if not 0.0 < allele_frequency < 1.0:
        raise ValueError("--allele-frequency must be strictly between 0 and 1")

    n = M.shape[0]
    try:
        G = all_haploid_genotypes(n, max_genotypes)
    except ValueError:
        return None

    G_float = G.astype(float)
    log_fitness = np.einsum("bi,ij,bj->b", G_float, M, G_float)
    log_phi = np.sum(
        G * np.log(allele_frequency) + (1 - G) * np.log(1.0 - allele_frequency),
        axis=1,
    )
    log_unnormalized = log_phi + log_fitness
    log_unnormalized -= float(np.max(log_unnormalized))
    psi = np.exp(log_unnormalized)
    psi /= float(np.sum(psi))

    genotype_to_index = {tuple(row.tolist()): index for index, row in enumerate(G)}
    phi_gk = np.array([1.0 - allele_frequency, allele_frequency])
    rows: list[list[float]] = []
    for locus in range(n):
        psi_gk = np.array(
            [
                float(np.sum(psi[G[:, locus] == 0])),
                float(np.sum(psi[G[:, locus] == 1])),
            ]
        )
        D_gk = kl_divergence(psi_gk, phi_gk)

        D_conditional = 0.0
        for background_index, genotype in enumerate(G):
            if genotype[locus] != 0:
                continue
            mutant = genotype.copy()
            mutant[locus] = 1
            mutant_index = genotype_to_index[tuple(mutant.tolist())]
            psi_pair = np.array([psi[background_index], psi[mutant_index]], dtype=float)
            psi_background = float(np.sum(psi_pair))
            if psi_background == 0.0:
                continue
            psi_conditional = psi_pair / psi_background
            D_conditional += psi_background * kl_divergence(psi_conditional, phi_gk)

        rows.append([float(locus), D_gk, D_conditional, D_conditional - D_gk])

    return np.array(rows, dtype=float)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q), ignoring zero-probability p entries."""
    mask = p > 0.0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def mutation_surprise_tables(
    M: np.ndarray,
    max_genotypes: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Measure background-dependence of each 0->1 mutation.

    For every locus i and every background g with g_i = 0:

        effect_i(g) = log_fitness(g with i=1) - log_fitness(g)
                    = s_i + sum_{j != i} epsilon_ij g_j
        surprise_i(g) = effect_i(g) - average_background_effect_i

    Large surprise variance means the effect of mutating locus i depends strongly
    on the rest of the genotype, which is the house-of-cards-like signal here.
    """
    n = M.shape[0]
    try:
        G = all_haploid_genotypes(n, max_genotypes)
    except ValueError:
        return None, None

    G_float = G.astype(float)
    log_fitness = np.einsum("bi,ij,bj->b", G_float, M, G_float)
    genotype_to_index = {tuple(row.tolist()): index for index, row in enumerate(G)}

    rows: list[list[float]] = []
    summary_rows: list[list[float]] = []
    for locus in range(n):
        locus_rows: list[list[float]] = []
        for background_index, genotype in enumerate(G):
            if genotype[locus] != 0:
                continue

            mutant = genotype.copy()
            mutant[locus] = 1
            mutant_index = genotype_to_index[tuple(mutant.tolist())]

            background_log_fitness = float(log_fitness[background_index])
            mutant_log_fitness = float(log_fitness[mutant_index])
            log_effect = mutant_log_fitness - background_log_fitness
            locus_rows.append(
                [
                    float(locus),
                    float(background_index),
                    float(mutant_index),
                    background_log_fitness,
                    mutant_log_fitness,
                    log_effect,
                ]
                + genotype.astype(float).tolist()
            )

        effects = np.array([row[5] for row in locus_rows], dtype=float)
        background_log_fitnesses = np.array([row[3] for row in locus_rows], dtype=float)
        mutant_log_fitnesses = np.array([row[4] for row in locus_rows], dtype=float)
        average_effect = float(np.mean(effects))
        surprise = effects - average_effect
        effect_sd = float(np.std(effects))
        effect_variance = float(np.var(effects))
        surprise_sd = float(np.std(surprise))
        parent_mutant_correlation = safe_correlation(background_log_fitnesses, mutant_log_fitnesses)
        background_effect_correlation = safe_correlation(background_log_fitnesses, effects)

        for row, surprise_value in zip(locus_rows, surprise):
            z_score = 0.0 if surprise_sd == 0.0 else float(surprise_value / surprise_sd)
            rows.append(row[:6] + [average_effect, float(surprise_value), abs(float(surprise_value)), z_score] + row[6:])

        summary_rows.append(
            [
                float(locus),
                float(len(effects)),
                average_effect,
                effect_sd,
                effect_variance,
                float(np.mean(np.abs(surprise))),
                float(np.sqrt(np.mean(surprise**2))),
                float(np.min(effects)),
                float(np.max(effects)),
                parent_mutant_correlation,
                background_effect_correlation,
            ]
        )

    return np.array(rows, dtype=float), np.array(summary_rows, dtype=float)


def safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson correlation, or nan when one vector has no variation."""
    if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def add_fitness_metrics(
    metrics: dict[str, MetricValue],
    fitness_table: np.ndarray | None,
    mutation_summary_table: np.ndarray | None,
    information_table: np.ndarray | None,
    n: int,
) -> None:
    """Add genotype-fitness summary statistics to the graph metrics."""
    metrics["n_possible_haploid_genotypes"] = 2**n
    metrics["genotype_fitness_enumerated"] = fitness_table is not None
    if fitness_table is None:
        return

    log_fitness = fitness_table[:, -2]
    fitness = fitness_table[:, -1]
    metrics.update(
        {
            "min_log_fitness": float(np.min(log_fitness)),
            "mean_log_fitness": float(np.mean(log_fitness)),
            "max_log_fitness": float(np.max(log_fitness)),
            "min_fitness": float(np.min(fitness)),
            "mean_fitness": float(np.mean(fitness)),
            "max_fitness": float(np.max(fitness)),
        }
    )

    if mutation_summary_table is None:
        metrics["mutation_surprise_enumerated"] = False
        return

    metrics["mutation_surprise_enumerated"] = True
    metrics["mean_locus_effect_sd"] = float(np.mean(mutation_summary_table[:, 3]))
    metrics["mean_locus_effect_variance"] = float(np.mean(mutation_summary_table[:, 4]))
    metrics["max_locus_effect_variance"] = float(np.max(mutation_summary_table[:, 4]))
    metrics["mean_locus_absolute_surprise"] = float(np.mean(mutation_summary_table[:, 5]))
    metrics["mean_locus_rms_surprise"] = float(np.mean(mutation_summary_table[:, 6]))
    metrics["max_locus_rms_surprise"] = float(np.max(mutation_summary_table[:, 6]))
    metrics["mean_parent_mutant_log_fitness_correlation"] = nanmean_or_nan(mutation_summary_table[:, 9])
    metrics["mean_background_log_fitness_effect_correlation"] = nanmean_or_nan(mutation_summary_table[:, 10])

    if information_table is None:
        metrics["information_epistasis_enumerated"] = False
        return

    metrics["information_epistasis_enumerated"] = True
    metrics["mean_information_epistasis"] = float(np.mean(information_table[:, 3]))
    metrics["max_information_epistasis"] = float(np.max(information_table[:, 3]))


def nanmean_or_nan(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if len(finite_values) == 0:
        return float("nan")
    return float(np.mean(finite_values))


def add_skipped_fitness_metrics(metrics: dict[str, MetricValue], n: int) -> None:
    metrics["n_possible_haploid_genotypes"] = 2**n
    metrics["genotype_fitness_enumerated"] = False
    metrics["mutation_surprise_enumerated"] = False
    metrics["information_epistasis_enumerated"] = False


def graph_metrics(weighted_adjacency: np.ndarray, rng: np.random.Generator, comparison_graphs: int = 20) -> dict[str, MetricValue]:
    """Compute small-world diagnostics using networkx."""
    adjacency = np.asarray(weighted_adjacency != 0, dtype=bool)
    graph = nx.from_numpy_array(adjacency)
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    connected = nx.is_connected(graph)
    clustering = float(nx.average_clustering(graph))
    path_length = float(nx.average_shortest_path_length(graph)) if connected else float("inf")

    random_clusterings: list[float] = []
    random_path_lengths: list[float] = []
    for _ in range(comparison_graphs):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        random_graph = nx.gnm_random_graph(n=n, m=m, seed=seed)
        if not nx.is_connected(random_graph):
            continue
        random_clusterings.append(float(nx.average_clustering(random_graph)))
        random_path_lengths.append(float(nx.average_shortest_path_length(random_graph)))

    C_rand = float(np.mean(random_clusterings)) if random_clusterings else float("nan")
    L_rand = float(np.mean(random_path_lengths)) if random_path_lengths else float("nan")
    sigma = float("nan")
    if C_rand > 0 and L_rand > 0 and np.isfinite(path_length):
        sigma = float((clustering / C_rand) / (path_length / L_rand))

    return {
        "n": n,
        "edges": m,
        "average_degree": float(2 * m / n),
        "density": float(2 * m / (n * (n - 1))),
        "connected": connected,
        "average_clustering": clustering,
        "average_shortest_path_length": path_length,
        "random_average_clustering": C_rand,
        "random_average_shortest_path_length": L_rand,
        "small_world_sigma": sigma,
        "small_world_like": bool(connected and np.isfinite(sigma) and sigma > 1.0),
    }


def write_outputs(
    prefix: Path,
    weighted_adjacency: np.ndarray,
    M: np.ndarray,
    metrics: dict[str, MetricValue],
    selection_coefficients: np.ndarray,
    fitness_table: np.ndarray | None,
    mutation_surprise_table: np.ndarray | None,
    mutation_summary_table: np.ndarray | None,
    information_table: np.ndarray | None,
    n_samples: int,
) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(prefix.with_name(prefix.name + "_weighted_adjacency.npy"), weighted_adjacency)
    np.save(prefix.with_name(prefix.name + "_M.npy"), M)
    np.savetxt(prefix.with_name(prefix.name + "_M.csv"), M, delimiter=",")
    np.save(prefix.with_name(prefix.name + "_selection_coefficients.npy"), selection_coefficients)

    with prefix.with_name(prefix.name + "_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    with prefix.with_name(prefix.name + "_edges.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["i", "j", "edge_weight_epsilon", "M_ij"])
        for i, j in _edges_from_adjacency(weighted_adjacency != 0):
            writer.writerow([i, j, weighted_adjacency[i, j], M[i, j]])

    with prefix.with_name(prefix.name + "_selection_coefficients.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["i", "s_i", "M_ii"])
        for i, selection in enumerate(selection_coefficients):
            writer.writerow([i, selection, M[i, i]])

    if fitness_table is not None:
        n = M.shape[0]
        with prefix.with_name(prefix.name + "_genotype_fitness.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"g_{i}" for i in range(n)] + ["log_fitness_gTMg", "fitness_exp_gTMg"])
            writer.writerows(fitness_table)

    if mutation_surprise_table is not None:
        n = M.shape[0]
        with prefix.with_name(prefix.name + "_mutation_surprise.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "locus",
                    "background_index",
                    "mutant_index",
                    "background_log_fitness",
                    "mutant_log_fitness",
                    "log_fitness_effect",
                    "average_locus_effect",
                    "surprise",
                    "absolute_surprise",
                    "surprise_z_score",
                ]
                + [f"background_g_{i}" for i in range(n)]
            )
            writer.writerows(mutation_surprise_table)

    if mutation_summary_table is not None:
        with prefix.with_name(prefix.name + "_mutation_surprise_by_locus.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "locus",
                    "n_backgrounds",
                    "average_log_fitness_effect",
                    "effect_sd",
                    "effect_variance",
                    "mean_absolute_surprise",
                    "rms_surprise",
                    "min_log_fitness_effect",
                    "max_log_fitness_effect",
                    "parent_mutant_log_fitness_correlation",
                    "background_log_fitness_effect_correlation",
                ]
            )
            writer.writerows(mutation_summary_table)

    if information_table is not None:
        with prefix.with_name(prefix.name + "_information_epistasis_by_locus.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["locus", "D_gk", "D_gk_given_background", "information_epistasis"])
            writer.writerows(information_table)

    config_path = prefix.with_name(prefix.name + "_equilibria_config.yaml")
    m_path = prefix.with_name(prefix.name + "_M.npy")
    config_path.write_text(
        "\n".join(
            [
                "# Generated by small_world_epistasis.py",
                f"n_loci: {M.shape[0]}",
                "N: 50",
                "ploidy: haploid",
                "u: 1.0e-5",
                "v: 1.0e-5",
                "h: 0.5",
                "seed: 42",
                'out: ""',
                f'sigma_file: "{m_path.resolve()}"',
                "explore:",
                "  method: mc",
                f"  n_samples: {n_samples}",
                "  grid_resolution: 17",
                "  x_eps: 1.0e-8",
                "",
            ]
        )
    )
    write_run_report(
        prefix,
        weighted_adjacency,
        M,
        metrics,
        selection_coefficients,
        mutation_summary_table,
        information_table,
    )
    write_plots(
        prefix,
        weighted_adjacency,
        selection_coefficients,
        fitness_table,
        mutation_surprise_table,
        mutation_summary_table,
        information_table,
    )


def write_run_report(
    prefix: Path,
    weighted_adjacency: np.ndarray,
    M: np.ndarray,
    metrics: dict[str, MetricValue],
    selection_coefficients: np.ndarray,
    mutation_summary_table: np.ndarray | None,
    information_table: np.ndarray | None,
) -> None:
    """Write one human-readable text file with setup, sampled M, and summaries."""
    lines: list[str] = []
    lines.append("Small-world epistasis run report")
    lines.append("=" * 34)
    lines.append("")

    lines.append("Model convention")
    lines.append("----------------")
    lines.append("Allele 0 is wild type; allele 1 is the mutation.")
    lines.append("The Watts-Strogatz graph chooses which off-diagonal M entries are nonzero.")
    lines.append("For each edge (i, j), an epsilon_ij is sampled and stored symmetrically:")
    lines.append("    M[i, j] = M[j, i] = epsilon_ij / 2")
    lines.append("The diagonal stores single-locus coefficients s_i:")
    lines.append("    M[i, i] = s_i")
    lines.append("Fitness is computed as omega(g) = exp(g.T @ M @ g).")
    lines.append("")

    lines.append("Definitions of mutation-effect statistics")
    lines.append("-----------------------------------------")
    lines.append("For a locus i, we examine every background with g_i = 0.")
    lines.append("For each background, effect_i(g) = log_fitness(mutant) - log_fitness(background).")
    lines.append("average_log_fitness_effect: mean of effect_i(g) over all 2^(n-1) backgrounds.")
    lines.append("effect_sd: standard deviation of effect_i(g); same units as log fitness.")
    lines.append("effect_variance: variance of effect_i(g); equal to effect_sd squared.")
    lines.append("surprise: effect_i(g) - average_log_fitness_effect for that locus.")
    lines.append("mean_absolute_surprise: mean(abs(surprise)); typical absolute deviation from the average effect.")
    lines.append("rms_surprise: sqrt(mean(surprise^2)); RMS deviation from the average effect.")
    lines.append("min/max_log_fitness_effect: smallest/largest effect seen across backgrounds.")
    lines.append("parent_mutant_log_fitness_correlation: Pearson correlation between background and mutant log fitness.")
    lines.append("background_log_fitness_effect_correlation: Pearson correlation between background log fitness and mutation effect.")
    lines.append("D_gk: KL divergence between the selected allele distribution psi(g_k) and baseline phi(g_k).")
    lines.append("D_gk_given_background: expected KL divergence for g_k after conditioning on g_-k.")
    lines.append("information_epistasis: D(g_k | g_-k) - D(g_k), the extra information gained from knowing the background.")
    lines.append("")

    lines.append("Run settings and analysis metrics")
    lines.append("---------------------------------")
    lines.append(json.dumps(metrics, indent=2))
    lines.append("")

    lines.append("Nonzero epistatic edges")
    lines.append("-----------------------")
    lines.append("i,j,epsilon_ij,M_ij,M_ji")
    for i, j in _edges_from_adjacency(weighted_adjacency != 0):
        lines.append(f"{i},{j},{weighted_adjacency[i, j]:.12g},{M[i, j]:.12g},{M[j, i]:.12g}")
    lines.append("")

    lines.append("Diagonal selection coefficients")
    lines.append("-------------------------------")
    lines.append("i,s_i,M_ii")
    for i, selection in enumerate(selection_coefficients):
        lines.append(f"{i},{selection:.12g},{M[i, i]:.12g}")
    lines.append("")

    lines.append("Full M matrix")
    lines.append("-------------")
    lines.append(np.array2string(M, precision=8, suppress_small=False, max_line_width=160))
    lines.append("")

    if mutation_summary_table is not None:
        lines.append("Mutation-effect summary by locus")
        lines.append("--------------------------------")
        lines.append(
            "locus,n_backgrounds,average_log_fitness_effect,effect_sd,effect_variance,"
            "mean_absolute_surprise,rms_surprise,min_log_fitness_effect,max_log_fitness_effect,"
            "parent_mutant_log_fitness_correlation,background_log_fitness_effect_correlation"
        )
        for row in mutation_summary_table:
            lines.append(",".join(f"{value:.12g}" for value in row))
    else:
        lines.append("Mutation-effect summary by locus")
        lines.append("--------------------------------")
        lines.append("Skipped because 2^n exceeded --max-genotypes.")
    lines.append("")

    if information_table is not None:
        lines.append("Information epistasis by locus")
        lines.append("------------------------------")
        lines.append("locus,D_gk,D_gk_given_background,information_epistasis")
        for row in information_table:
            lines.append(",".join(f"{value:.12g}" for value in row))
    else:
        lines.append("Information epistasis by locus")
        lines.append("------------------------------")
        lines.append("Skipped because 2^n exceeded --max-genotypes.")
    lines.append("")

    prefix.with_name(prefix.name + "_run_report.txt").write_text("\n".join(lines))


def write_plots(
    prefix: Path,
    weighted_adjacency: np.ndarray,
    selection_coefficients: np.ndarray,
    fitness_table: np.ndarray | None,
    mutation_surprise_table: np.ndarray | None,
    mutation_summary_table: np.ndarray | None,
    information_table: np.ndarray | None,
) -> None:
    """Write per-locus and across-locus plots into a plots subdirectory."""
    plots_dir = prefix.parent / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    write_epistatic_network_plot(prefix, plots_dir, weighted_adjacency, selection_coefficients)

    if fitness_table is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        log_fitness = fitness_table[:, -2]
        ax.hist(log_fitness, bins=50, color="#4c72b0", edgecolor="black", alpha=0.85)
        ax.axvline(np.mean(log_fitness), color="red", linestyle="--", label="mean")
        ax.set_title("Log-fitness distribution across all genotypes")
        ax.set_xlabel("log fitness = g.T @ M @ g")
        ax.set_ylabel("Number of genotypes")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"{prefix.name}_all_genotypes_log_fitness.png", dpi=150)
        plt.close(fig)

    if mutation_surprise_table is None or mutation_summary_table is None:
        return

    for locus in mutation_summary_table[:, 0].astype(int):
        locus_rows = mutation_surprise_table[mutation_surprise_table[:, 0] == float(locus)]
        if len(locus_rows) == 0:
            continue

        background_log_fitness = locus_rows[:, 3]
        mutant_log_fitness = locus_rows[:, 4]
        log_effect = locus_rows[:, 5]
        surprise = locus_rows[:, 7]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Locus {locus}: 0 -> 1 mutation effects across backgrounds")

        axes[0, 0].hist(log_effect, bins=40, color="#4c72b0", edgecolor="black", alpha=0.8)
        axes[0, 0].axvline(np.mean(log_effect), color="red", linestyle="--", label="mean")
        axes[0, 0].set_title("Distribution of log-fitness effect")
        axes[0, 0].set_xlabel("log_fitness(mutant) - log_fitness(background)")
        axes[0, 0].set_ylabel("Number of backgrounds")
        axes[0, 0].legend()

        axes[0, 1].hist(surprise, bins=40, color="#55a868", edgecolor="black", alpha=0.8)
        axes[0, 1].axvline(0.0, color="red", linestyle="--", label="average effect")
        axes[0, 1].set_title("Distribution of surprise")
        axes[0, 1].set_xlabel("effect - average_locus_effect")
        axes[0, 1].set_ylabel("Number of backgrounds")
        axes[0, 1].legend()

        axes[1, 0].scatter(background_log_fitness, log_effect, s=8, alpha=0.5)
        axes[1, 0].set_title("Background fitness vs mutation effect")
        axes[1, 0].set_xlabel("background log fitness")
        axes[1, 0].set_ylabel("log-fitness effect")

        parent_mutant_density = axes[1, 1].hist2d(background_log_fitness, mutant_log_fitness, bins=45, cmap="viridis")
        fig.colorbar(parent_mutant_density[3], ax=axes[1, 1], label="Number of backgrounds")
        axes[1, 1].set_title("Parent vs mutant log fitness density")
        axes[1, 1].set_xlabel("background log fitness")
        axes[1, 1].set_ylabel("mutant log fitness")

        fig.tight_layout()
        fig.savefig(plots_dir / f"{prefix.name}_locus_{locus:03d}_effects.png", dpi=150)
        plt.close(fig)

    loci = mutation_summary_table[:, 0].astype(int)
    average_effect = mutation_summary_table[:, 2]
    effect_sd = mutation_summary_table[:, 3]
    effect_variance = mutation_summary_table[:, 4]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Mutation-effect summary across loci")
    x = np.arange(len(loci))

    axes[0].bar(x, average_effect)
    axes[0].set_title("Average effect")
    axes[0].set_ylabel("mean log-fitness effect")

    axes[1].bar(x, effect_variance)
    axes[1].set_title("Effect variance")
    axes[1].set_ylabel("variance")

    for axis in axes:
        axis.set_xlabel("locus")
        axis.set_xticks(x)
        axis.set_xticklabels(loci)

    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix.name}_summary_by_locus.png", dpi=150)
    plt.close(fig)

    if information_table is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        info_loci = information_table[:, 0].astype(int)
        ax.bar(np.arange(len(info_loci)), information_table[:, 3])
        ax.set_title("Information epistasis by locus")
        ax.set_xlabel("locus")
        ax.set_ylabel("D(g_k | g_-k) - D(g_k)")
        ax.set_xticks(np.arange(len(info_loci)))
        ax.set_xticklabels(info_loci)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{prefix.name}_information_epistasis_by_locus.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    parent_mutant_correlation = mutation_summary_table[:, 9]
    scatter = ax.scatter(
        effect_sd,
        np.nan_to_num(parent_mutant_correlation, nan=np.nan),
        c=loci,
        cmap="tab20",
        s=70,
        edgecolor="black",
    )
    for locus, x_value, y_value in zip(loci, effect_sd, parent_mutant_correlation):
        if np.isfinite(y_value):
            ax.annotate(str(locus), (x_value, y_value), textcoords="offset points", xytext=(5, 5), fontsize=8)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("locus")
    ax.set_title("House-of-cards-style diagnostic")
    ax.set_xlabel("effect_sd: mutation-effect variability across backgrounds")
    ax.set_ylabel("parent-mutant log-fitness correlation")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix.name}_house_of_cards_diagnostic.png", dpi=150)
    plt.close(fig)


def write_epistatic_network_plot(
    prefix: Path,
    plots_dir: Path,
    weighted_adjacency: np.ndarray,
    selection_coefficients: np.ndarray,
) -> None:
    """Plot the epistatic network: nodes are loci, edges are nonzero epsilon_ij."""
    graph = nx.from_numpy_array(weighted_adjacency)
    pos = nx.circular_layout(graph, scale=5.0)

    edge_values = np.array([weighted_adjacency[i, j] for i, j in graph.edges()], dtype=float)
    if len(edge_values) == 0:
        edge_colors: list[float] | np.ndarray = []
        edge_widths: list[float] | np.ndarray = []
    else:
        max_abs_edge = float(np.max(np.abs(edge_values)))
        edge_colors = edge_values
        edge_widths = 1.0 + 4.0 * np.abs(edge_values) / max_abs_edge if max_abs_edge > 0.0 else np.ones_like(edge_values)

    max_abs_selection = float(np.max(np.abs(selection_coefficients))) if len(selection_coefficients) else 0.0
    node_vmin = -max_abs_selection if max_abs_selection > 0.0 else -1.0
    node_vmax = max_abs_selection if max_abs_selection > 0.0 else 1.0

    fig, ax = plt.subplots(figsize=(16, 14))
    nodes = nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=selection_coefficients,
        cmap="coolwarm",
        vmin=node_vmin,
        vmax=node_vmax,
        node_size=1000,
        edgecolors="black",
        linewidths=1.2,
        ax=ax,
    )
    edges = nx.draw_networkx_edges(
        graph,
        pos,
        edge_color=edge_colors,
        edge_cmap=plt.cm.PiYG,
        edge_vmin=-float(np.max(np.abs(edge_values))) if len(edge_values) else -1.0,
        edge_vmax=float(np.max(np.abs(edge_values))) if len(edge_values) else 1.0,
        width=edge_widths,
        ax=ax,
    )

    node_labels = {i: f"{i}\ns={selection_coefficients[i]:.2g}" for i in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8, ax=ax)

    edge_labels = {(i, j): f"{weighted_adjacency[i, j]:.2g}" for i, j in graph.edges()}
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        ax=ax,
    )

    fig.colorbar(nodes, ax=ax, label="diagonal selection coefficient s_i")
    if edges is not None and len(edge_values):
        fig.colorbar(edges, ax=ax, label="edge epistasis epsilon_ij")

    ax.set_title("Epistatic network: edges are nonzero M_ij entries")
    ax.text(
        0.5,
        -0.08,
        "Node labels show locus and s_i = M_ii. Edge labels show epsilon_ij; stored matrix entries are M_ij = M_ji = epsilon_ij / 2.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{prefix.name}_epistatic_network.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _edges_from_adjacency(adjacency: np.ndarray) -> list[tuple[int, int]]:
    return [(i, j) for i in range(adjacency.shape[0]) for j in range(i + 1, adjacency.shape[1]) if adjacency[i, j]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a weighted small-world graph and epistasis matrix M.")
    parser.add_argument("--n", type=int, required=True, help="Number of loci/nodes.")
    parser.add_argument("--k", type=int, default=4, help="Even number of nearest neighbors in the starting ring lattice.")
    parser.add_argument("--rewire-prob", type=float, default=0.1, help="Watts-Strogatz rewiring probability.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--edge-distribution", choices=["uniform", "normal"], default="uniform", help="Distribution for nonzero edge effects epsilon_ij.")
    parser.add_argument("--weight-low", type=float, default=-1.0, help="Minimum epsilon_ij when --edge-distribution uniform.")
    parser.add_argument("--weight-high", type=float, default=1.0, help="Maximum epsilon_ij when --edge-distribution uniform.")
    parser.add_argument("--edge-mean", type=float, default=0.0, help="Mean epsilon_ij when --edge-distribution normal.")
    parser.add_argument("--edge-std", type=float, default=0.1, help="Standard deviation of epsilon_ij when --edge-distribution normal.")
    parser.add_argument("--diagonal-selection", type=float, default=0.0, help="Constant value for all M[i,i] when random s_i is disabled.")
    parser.add_argument("--selection-distribution", choices=["constant", "uniform", "normal"], default="constant", help="Distribution for diagonal selection coefficients s_i.")
    parser.add_argument("--random-selection-coefficients", action="store_true", help="Legacy shortcut for --selection-distribution uniform.")
    parser.add_argument("--selection-low", type=float, default=-0.1, help="Minimum s_i when --selection-distribution uniform.")
    parser.add_argument("--selection-high", type=float, default=0.1, help="Maximum s_i when --selection-distribution uniform.")
    parser.add_argument("--selection-mean", type=float, default=0.0, help="Mean s_i when --selection-distribution normal.")
    parser.add_argument("--selection-std", type=float, default=0.05, help="Standard deviation of s_i when --selection-distribution normal.")
    parser.add_argument("--allele-frequency", type=float, default=0.5, help="Baseline P(g_i=1)=x used for information-epistasis calculations.")
    parser.add_argument("--max-genotypes", type=int, default=65536, help="Largest 2^n genotype table to enumerate and save.")
    parser.add_argument("--comparison-graphs", type=int, default=20, help="Random graphs used for small-world comparison.")
    parser.add_argument("--n-samples", type=int, default=80000, help="Monte Carlo samples in generated equilibrium config.")
    parser.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory where all output files are written.")
    parser.add_argument("--out-prefix", type=Path, default=Path("small_world"), help="Output file prefix inside --results-dir.")
    return parser.parse_args()


def output_prefix_from_args(args: argparse.Namespace) -> Path:
    """Keep every run in its own timestamped folder inside the results directory."""
    if args.out_prefix.is_absolute():
        raise ValueError("--out-prefix must be relative so outputs stay inside --results-dir")
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return args.results_dir / run_timestamp / args.out_prefix


# ---------------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------------
#
# Run from this folder:
#
#   python3 small_world_epistasis.py \
#       --n 12 \
#       --k 4 \
#       --rewire-prob 0.05 \
#       --edge-distribution normal \
#       --edge-mean 0.0 \
#       --edge-std 0.1 \
#       --selection-distribution normal \
#       --selection-mean 0.0 \
#       --selection-std 0.05 \
#       --allele-frequency 0.5 \
#       --max-genotypes 65536 \
#       --comparison-graphs 20 \
#       --seed 42 \
#       --results-dir results \
#       --out-prefix sw12
#
# What the command means:
#
#   --n
#       Number of loci. The genotype space has 2^n binary genotypes.
#
#   --k
#       Number of nearest neighbors in the starting Watts-Strogatz ring.
#       Must be even. The graph starts locally connected, then some edges are
#       rewired into long-distance shortcuts.
#
#   --rewire-prob
#       Watts-Strogatz rewiring probability. Small values usually preserve high
#       clustering while shortening paths.
#
#   --edge-distribution
#       Distribution used to sample epsilon_ij for graph edges. These are the
#       pairwise epistatic effects. Supported values: uniform, normal.
#
#   --selection-distribution
#       Distribution used to sample diagonal s_i values. Supported values:
#       constant, uniform, normal.
#
#   --results-dir
#       Parent output directory. Each run creates a timestamped subfolder here,
#       for example results/2026-05-05_13-45-12/.
#
#   --out-prefix
#       Prefix for all output files. With --results-dir results --out-prefix sw12,
#       outputs include results/YYYY-MM-DD_HH-MM-SS/sw12_M.csv,
#       results/YYYY-MM-DD_HH-MM-SS/sw12_metrics.json, and
#       results/YYYY-MM-DD_HH-MM-SS/sw12_run_report.txt.
#
# Main outputs:
#
#   *_M.csv
#       Full epistasis matrix M, including zeros and diagonal s_i.
#
#   *_edges.csv
#       Nonzero graph edges, sampled epsilon_ij, and stored M_ij = epsilon_ij / 2.
#
#   *_mutation_surprise_by_locus.csv
#       Per-locus statistics over all 2^(n-1) backgrounds with g_i = 0:
#       average effect, effect variance, surprise, and correlation diagnostics.
#
#   *_information_epistasis_by_locus.csv
#       D(g_k | g_-k) - D(g_k), an information-theoretic epistasis measure.
#
#   *_run_report.txt
#       One readable text report with the network setup, distributions, final M,
#       and analysis summaries.
#
#   plots/
#       Per-locus mutation-effect plots plus across-locus summary plots.
#
def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    output_prefix = output_prefix_from_args(args)

    graph = generate_connected_small_world_graph(args.n, args.k, args.rewire_prob, rng)
    weighted_adjacency = weighted_adjacency_from_graph(
        graph,
        rng,
        distribution=args.edge_distribution,
        weight_low=args.weight_low,
        weight_high=args.weight_high,
        weight_mean=args.edge_mean,
        weight_std=args.edge_std,
    )
    selection_distribution = args.selection_distribution
    if args.random_selection_coefficients and selection_distribution == "constant":
        selection_distribution = "uniform"

    if selection_distribution == "constant":
        selection_coefficients = np.full(args.n, args.diagonal_selection, dtype=float)
    else:
        selection_coefficients = random_selection_coefficients(
            args.n,
            rng,
            selection_distribution,
            args.selection_low,
            args.selection_high,
            args.selection_mean,
            args.selection_std,
        )

    M = epistasis_matrix_from_weighted_adjacency(weighted_adjacency, selection_coefficients)
    metrics = graph_metrics(weighted_adjacency, rng, comparison_graphs=args.comparison_graphs)
    metrics.update(
        {
            "seed": args.seed,
            "watts_strogatz_k": args.k,
            "watts_strogatz_rewire_prob": args.rewire_prob,
            "results_dir": str(args.results_dir),
            "out_prefix": str(args.out_prefix),
            "run_output_dir": str(output_prefix.parent),
            "plots_dir": str(output_prefix.parent / "plots"),
            "edge_distribution": args.edge_distribution,
            "edge_uniform_low": args.weight_low,
            "edge_uniform_high": args.weight_high,
            "edge_normal_mean": args.edge_mean,
            "edge_normal_std": args.edge_std,
            "selection_distribution": selection_distribution,
            "selection_constant": args.diagonal_selection,
            "selection_uniform_low": args.selection_low,
            "selection_uniform_high": args.selection_high,
            "selection_normal_mean": args.selection_mean,
            "selection_normal_std": args.selection_std,
            "allele_frequency_for_information_epistasis": args.allele_frequency,
        }
    )
    fitness_table = genotype_fitness_table(M, args.max_genotypes)
    mutation_surprise_table, mutation_summary_table = mutation_surprise_tables(M, args.max_genotypes)
    information_table = information_epistasis_table(M, args.max_genotypes, args.allele_frequency)
    if fitness_table is None:
        add_skipped_fitness_metrics(metrics, args.n)
    else:
        add_fitness_metrics(metrics, fitness_table, mutation_summary_table, information_table, args.n)

    write_outputs(
        output_prefix,
        weighted_adjacency,
        M,
        metrics,
        selection_coefficients,
        fitness_table,
        mutation_surprise_table,
        mutation_summary_table,
        information_table,
        args.n_samples,
    )

    print(json.dumps(metrics, indent=2))
    print(f"\nRun output directory: {output_prefix.parent}")
    print(f"\nSaved M to: {output_prefix.with_name(output_prefix.name + '_M.npy')}")
    print(f"Saved full M as CSV to: {output_prefix.with_name(output_prefix.name + '_M.csv')}")
    print(f"Saved run report to: {output_prefix.with_name(output_prefix.name + '_run_report.txt')}")
    print(f"Saved plots to: {output_prefix.parent / 'plots'}")
    print(f"Saved selection coefficients to: {output_prefix.with_name(output_prefix.name + '_selection_coefficients.csv')}")
    if fitness_table is not None:
        print(f"Saved genotype fitness table to: {output_prefix.with_name(output_prefix.name + '_genotype_fitness.csv')}")
        print(f"Saved mutation surprise table to: {output_prefix.with_name(output_prefix.name + '_mutation_surprise.csv')}")
        print(f"Saved mutation surprise summary to: {output_prefix.with_name(output_prefix.name + '_mutation_surprise_by_locus.csv')}")
        print(f"Saved information epistasis summary to: {output_prefix.with_name(output_prefix.name + '_information_epistasis_by_locus.csv')}")
    else:
        print("Skipped genotype fitness and mutation surprise tables because 2^n exceeds --max-genotypes.")
    print(f"Saved equilibrium config to: {output_prefix.with_name(output_prefix.name + '_equilibria_config.yaml')}")


if __name__ == "__main__":
    main()
