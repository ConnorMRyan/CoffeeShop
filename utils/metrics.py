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


from .diversity import calculate_population_diversity

def compute_population_diversity(actors: Dict[str, SocialActor], mediator: CoffeeShopMediator, device: str, batch_size: int = 64) -> float:
    """
    Computes Jensen-Shannon (JS) Divergence between all agents in the population
    using a shared probe batch sampled from the mediator's buffer.
    """
    # Sample probe batch from mediator (which holds highest-priority memories)
    # Actually, mediator.critic evaluates TD-error, but the buffer is what stores them.
    # In the modern script/train.py, we don't have a centralized buffer yet, 
    # but we can sample from the local agent rollout buffers if needed.
    # For now, if no buffer is passed, return 0.
    
    # Check if any actor has data in their rollout buffer
    any_actor = next(iter(actors.values()))
    if not hasattr(any_actor, 'buffer') or len(any_actor.buffer) < batch_size:
        return 0.0

    # Sample observations from the first agent's buffer as a probe
    obs_batch = any_actor.buffer.obs[:batch_size].to(device)

    agent_distributions = []
    with torch.no_grad():
        for aid, actor in actors.items():
            # ActorCritic output: (logits, value)
            logits, _ = actor.model(obs_batch)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            agent_distributions.append(probs)

    return calculate_population_diversity(agent_distributions)
