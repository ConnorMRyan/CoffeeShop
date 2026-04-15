from __future__ import annotations

import math
from typing import Any, Dict, List
import numpy as np


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate the Jensen-Shannon divergence between two probability distributions.

    JS(P || Q) = 1/2 * KL(P || M) + 1/2 * KL(Q || M)
    where M = 1/2 * (P + Q)

    Args:
        p: Probability distribution (numpy array).
        q: Probability distribution (numpy array).

    Returns:
        JS divergence value in [0, 1] (float, log base-2).
    """
    # Clip only the lower bound to avoid log(0).
    # Do NOT clip to 1.0 — upper-clipping distorts near-deterministic
    # distributions by compressing probability mass before renormalisation.
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    # Re-normalize
    p /= p.sum()
    q /= q.sum()

    m = 0.5 * (p + q)

    def kl_divergence(a: np.ndarray, b: np.ndarray) -> float:
        # log base-2 ensures JSD is bounded in [0, 1]
        return float(np.sum(a * np.log2(a / b)))

    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def calculate_population_diversity(agent_distributions: List[np.ndarray]) -> float:
    """Calculate the average JS divergence across all agent pairs in a population.

    Returns the diversity value bounded in [0, 1] using log base 2.
    """
    if not agent_distributions or len(agent_distributions) < 2:
        return 0.0

    num_agents = len(agent_distributions)
    total_js = 0.0
    count = 0

    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            batch_js = []
            for k in range(agent_distributions[i].shape[0]):
                # jensen_shannon_divergence already uses log2, so result is in [0, 1]
                js = jensen_shannon_divergence(agent_distributions[i][k], agent_distributions[j][k])
                batch_js.append(js)
            total_js += float(np.mean(batch_js))
            count += 1

    return total_js / count if count > 0 else 0.0
