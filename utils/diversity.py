from __future__ import annotations

from typing import List
import numpy as np


def jensen_shannon_divergence_batched(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Calculate the Jensen-Shannon divergence between two batches of probability distributions.

    JS(P || Q) = 1/2 * KL(P || M) + 1/2 * KL(Q || M)
    where M = 1/2 * (P + Q)

    Args:
        p, q: numpy arrays of shape [batch, actions].
              Must be non-negative; will be normalized internally.

    Returns:
        numpy array of shape [batch] containing JSD values in [0, 1] (log base-2).
    """
    # Defensive clipping to avoid log(0) and ensure numerical stability
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)

    # Ensure distributions sum to 1 over the action dimension
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)

    m = 0.5 * (p + q)

    def kl_batched(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Vectorized KL Divergence using log base-2 for [0, 1] boundedness
        return np.sum(a * np.log2(a / b), axis=-1)

    return 0.5 * kl_batched(p, m) + 0.5 * kl_batched(q, m)


def calculate_population_diversity(agent_distributions: List[np.ndarray]) -> float:
    """
    Calculate the mean pairwise JSD across all agent pairs in a population.
    This serves as a high-fidelity metric for behavioral diversity.

    Args:
        agent_distributions: List of numpy arrays, each of shape [batch, actions].
                             One array per agent in the population.

    Returns:
        A float representing the mean pairwise diversity in [0, 1].
        Returns 0.0 if fewer than 2 agents are provided.
    """
    num_agents = len(agent_distributions)
    if num_agents < 2:
        return 0.0

    total_js = 0.0
    pair_count = 0

    # Iterate through all unique pairs (i, j)
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            # Compute vectorized JSD for the entire batch of observations
            pair_js_batch = jensen_shannon_divergence_batched(
                agent_distributions[i],
                agent_distributions[j]
            )

            # Average JSD across the probe batch for this specific pair
            total_js += float(np.mean(pair_js_batch))
            pair_count += 1

    return total_js / pair_count