from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 64


class PPOAgent:
    """Minimal PPO agent placeholder.

    This is a template. Wire it to your model, optimizer, and rollout collection.
    """

    def __init__(self, obs_space: Any | None = None, act_space: Any | None = None, config: PPOConfig | None = None) -> None:
        self.obs_space = obs_space
        self.act_space = act_space
        self.config = config or PPOConfig()

    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> Dict[str, Any]:
        """Return actions per agent_id. This is a stub returning no-ops."""
        return {aid: None for aid in obs.keys()}

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform a training update from a batch.

        Returns scalar metrics.
        """
        return {"loss_policy": 0.0, "loss_value": 0.0}
