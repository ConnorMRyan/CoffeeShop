from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch


class SocialEnvWrapper(ABC):
    """Abstract base class for CoffeeShop multi-agent environments.

    Contract:
      - All inputs/outputs are dictionaries keyed by `agent_id` strings.
      - Observations are PyTorch tensors.
      - `step()` returns the Gymnasium v0.29 style tuple plus an `infos` dict
        that MUST include `"sparse_rewards"` mapping each `agent_id` to float.
    """

    # ---- Required properties ----
    @property
    @abstractmethod
    def agent_ids(self) -> List[str]:
        """Stable ordering of agent identifiers (e.g., ["agent_0", "agent_1"])."""
        raise NotImplementedError

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def action_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def global_obs_dim(self) -> int:
        raise NotImplementedError

    # ---- Core API ----
    @abstractmethod
    def reset(self) -> Tuple[Dict[str, torch.Tensor], dict]:
        """Reset the environment and return initial observations per agent."""
        raise NotImplementedError

    @abstractmethod
    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, torch.Tensor],  # observations per agent
        Dict[str, float],         # rewards per agent (dense)
        Dict[str, bool],          # terminated per agent
        Dict[str, bool],          # truncated per agent
        dict,                     # infos dict; MUST include top-level key "sparse_rewards"
    ]:
        """Perform one environment step given discrete actions per agent."""
        raise NotImplementedError

    @abstractmethod
    def get_global_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Assemble a single global observation tensor from per-agent obs."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release any resources (noop by default)."""
        raise NotImplementedError
