from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch

@dataclass
class SocialActorConfig:
    """Configuration for a PPOActor."""
    id: str
    omega_init: float = 0.1
    omega_learnable: bool = False

class SocialActor:
    """Represents a single agent participating in social (multi-agent) interaction.

    It delegates action selection to a provided base agent, while internally
    managing its "Openness" (omega) to foreign experiences.
    """

    def __init__(self, agent: Any, config: SocialActorConfig) -> None:
        self.agent = agent
        self.config = config
        
        # Openness parameter (omega)
        if self.config.omega_learnable:
            self.omega = torch.nn.Parameter(torch.tensor([self.config.omega_init], dtype=torch.float32))
            # Requires integration with an optimizer
            self.optimizer = torch.optim.Adam([self.omega], lr=1e-3)
        else:
            self.omega = self.config.omega_init

    @property
    def id(self) -> str:
        return self.config.id

    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> Any:
        """Select an action for this actor given its own observation."""
        return self.agent.act({self.id: obs}, deterministic=deterministic).get(self.id)
        
    def get_omega(self, as_tensor: bool = False) -> Any:
        if isinstance(self.omega, torch.Tensor):
            # Sigmoid bound to [0, 1] if learned
            omega_val = torch.sigmoid(self.omega)
            return omega_val if as_tensor else omega_val.item()
        return torch.tensor([self.omega], device=self.agent.device) if as_tensor else self.omega

    def incorporate_shared_experience(self, obs_batch: torch.Tensor, act_batch: torch.Tensor) -> torch.Tensor:
        """Perform auxiliary distillation (Behavior Cloning) from shared experiences.
        
        Gated by the agent's current openness parameter `omega`.
        """
        current_omega = self.get_omega(as_tensor=True)
        
        # Delegate BC computation to the underlying agent (PPOAgent)
        if hasattr(self.agent, "behavior_cloning_update"):
            loss = self.agent.behavior_cloning_update(self.id, obs_batch, act_batch, current_omega)
            return loss
        return torch.tensor(0.0, device=obs_batch.device)
