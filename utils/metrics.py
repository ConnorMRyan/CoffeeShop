from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from agents.ppo import PPOAgent as SocialActor
    from core_marl.mediator import CoffeeShopMediator

@dataclass
class Metrics:
    """Lightweight metrics accumulator.

    Example:
        m = Metrics()
        m.update({'reward': 1.0})
        mean = m.mean()
    """

    history: Dict[str, List[float]] = field(default_factory=dict)

    def update(self, values: Dict[str, float]) -> None:
        for k, v in values.items():
            self.history.setdefault(k, []).append(float(v))

    def mean(self) -> Dict[str, float]:
        return {k: (sum(vs) / len(vs) if vs else 0.0) for k, vs in self.history.items()}

    def clear(self) -> None:
        self.history.clear()


def measure_population_diversity(actors: Dict[str, SocialActor], mediator: CoffeeShopMediator, device: str, batch_size: int = 64) -> float:
    """
    Computes Jensen-Shannon (JS) Divergence between all pairs of agents in the population
    using a shared probe batch sampled from the mediator's buffer.
    """
    if len(mediator._buffer) < batch_size:
        return 0.0

    # 1. Sample probe batch (observations only)
    memories = mediator._buffer.sample(batch_size)
    # memories are ScoredMemory, which contain Transition, which has .obs
    obs_list = [m.transition.obs for m in memories]
    # transitions contain obs for a specific agent; we assume they are compatible
    # across agents in the same environment type.
    obs_tensor = torch.stack(obs_list).to(device)

    # 2. Compute policy distributions for all agents on this batch
    # We use agent.ac.forward(obs) to get logits
    agent_probs = {}
    with torch.no_grad():
        for aid, actor in actors.items():
            logits, _ = actor.ac(obs_tensor)
            probs = F.softmax(logits, dim=-1)
            agent_probs[aid] = probs

    # 3. Compute pairwise JS Divergence
    # JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5 * (P + Q)
    aids = list(actors.keys())
    if len(aids) < 2:
        return 0.0

    total_js = 0.0
    pairs = 0

    def kl_div(p, q):
        # p, q: [batch, action_dim]
        return (p * (p.log() - q.log())).sum(dim=-1)

    for i in range(len(aids)):
        for j in range(i + 1, len(aids)):
            p = agent_probs[aids[i]]
            q = agent_probs[aids[j]]
            m = 0.5 * (p + q)
            
            # Use small epsilon to avoid log(0)
            eps = 1e-10
            p = p.clamp(min=eps)
            q = q.clamp(min=eps)
            m = m.clamp(min=eps)

            js = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
            total_js += js.mean().item()
            pairs += 1

    return total_js / pairs if pairs > 0 else 0.0
