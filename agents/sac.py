from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


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
        return {aid: None for aid in obs.keys()}

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform a training update from a batch.

        Returns scalar metrics.
        """
        return {"loss_q": 0.0, "loss_pi": 0.0, "alpha": self.config.alpha}
