from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING
import torch
import torch.nn.functional as F
import polars as pl
import numpy as np

if TYPE_CHECKING:
    from agents.ppo import PPOAgent as SocialActor
    from core_marl.mediator import CoffeeShopMediator as CoffeeShopMediator

@dataclass
class Metrics:
    """Lightweight metrics accumulator using Polars for high-performance aggregation.
    """

    history: Dict[str, List[float]] = field(default_factory=dict)

    def update(self, values: Dict[str, float]) -> None:
        for k, v in values.items():
            self.history.setdefault(k, []).append(float(v))

    def mean(self) -> Dict[str, float]:
        if not self.history:
            return {}
        
        results = {}
        for k, v in self.history.items():
            if not v:
                continue
            # Use Polars for faster mean calculation on individual series
            # Or just use numpy for simplicity if series are small
            results[k] = float(np.mean(v))
        return results

    def clear(self) -> None:
        self.history.clear()

    def report_final(self, output_path: str = "metrics.parquet"):
        """Save final metrics to a parquet file using Polars."""
        if not self.history:
            return
        
        # If columns have different lengths, Polars DataFrame constructor will fail.
        # We find the max length and pad with NaN, or store as a list of series if needed.
        # For a flat parquet, we should probably pad.
        max_len = max(len(v) for v in self.history.values())
        padded_history = {}
        for k, v in self.history.items():
            if len(v) < max_len:
                padded_history[k] = v + [float('nan')] * (max_len - len(v))
            else:
                padded_history[k] = v
        
        df = pl.DataFrame(padded_history)
        df.write_parquet(output_path)
        print(f"Final metrics saved to {output_path}")


def measure_population_diversity(actors: Dict[str, SocialActor], mediator: CoffeeShopMediator, device: str, batch_size: int = 64) -> float:
    """
    Computes Jensen-Shannon (JS) Divergence between all agents in the population
    using a shared probe batch sampled from the mediator's buffer.
    
    Optimized: Batches actor policy computations and vectorizes JS calculations.
    """
    if len(mediator._buffer) < batch_size:
        return 0.0

    # 1. Sample probe batch
    memories = mediator._buffer.sample(batch_size)
    obs_tensor = torch.stack([m.transition.obs for m in memories]).to(device)

    # 2. Compute policy distributions (batched per agent)
    aids = list(actors.keys())
    if len(aids) < 2:
        return 0.0

    probs_list = []
    with torch.no_grad():
        for aid in aids:
            logits, _ = actors[aid].ac(obs_tensor)
            probs_list.append(F.softmax(logits, dim=-1).clamp(min=1e-10))

    # All probs: [num_agents, batch, action_dim]
    all_probs = torch.stack(probs_list)
    num_agents = len(aids)

    # 3. Vectorized JS Divergence
    # We want mean JS over all unique pairs (i, j).
    # Since num_agents is usually small (e.g., 2-16), we can use broadcasting.
    # P: [num_agents, 1, batch, action_dim]
    # Q: [1, num_agents, batch, action_dim]
    p = all_probs.unsqueeze(1)
    q = all_probs.unsqueeze(0)
    m = 0.5 * (p + q)

    # KL(P || M) = (P * (log P - log M)).sum(-1)
    kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
    kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
    js_matrix = 0.5 * (kl_pm + kl_qm) # [num_agents, num_agents, batch]

    # Mean over batch
    js_pairs = js_matrix.mean(dim=-1) # [num_agents, num_agents]

    # Extract upper triangle (excluding diagonal)
    triu_indices = torch.triu_indices(num_agents, num_agents, offset=1)
    mean_js = js_pairs[triu_indices[0], triu_indices[1]].mean().item()

    return float(mean_js)
