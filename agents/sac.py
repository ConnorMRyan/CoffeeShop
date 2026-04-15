from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch


@dataclass
class SACConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2  # entropy temperature (fixed for stub)
    target_update_interval: int = 1
    batch_size: int = 256


class SACAgent:
    """Minimal SAC agent placeholder for MARL.

    This is a template to be wired to networks, replay buffer, and optimizers.
    """

    def __init__(self, obs_space: Any | None = None, act_space: Any | None = None, config: SACConfig | None = None) -> None:
        self.obs_space = obs_space
        self.act_space = act_space
        self.config = config or SACConfig()

    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> Dict[str, Any]:
        """Return actions per agent_id. This stub returns no-ops."""
        return {aid: {"action": 0, "val": 0.0, "logp": 0.0} for aid in obs.keys()}

    def behavior_cloning_update(self, aid: str, obs_batch: torch.Tensor, act_batch: torch.Tensor, omega: Any) -> float:
        """Stub for auxiliary distillation update."""
        return 0.0

    def get_action_dist(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Get the action distribution for the given observation.
        
        Returns a dict of action probabilities (numpy arrays) keyed by agent_id.
        In this stub, we return a uniform distribution over dummy actions.
        """
        # Example: Assume 4 actions
        num_actions = 4
        dist = np.ones(num_actions) / num_actions
        return {aid: dist for aid in obs.keys()}

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform a training update from a batch.

        Returns scalar metrics.
        """
        return {"loss_q": 0.0, "loss_pi": 0.0, "alpha": self.config.alpha}
