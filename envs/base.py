from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
import gymnasium as gym

class SocialEnvWrapper(ABC):
    """Abstract base class for all environments in CoffeeShop."""

    def __init__(self, agent_ids: List[str]):
        self.agent_ids = agent_ids

    @property
    @abstractmethod
    def obs_space(self) -> gym.Space:
        pass

    @property
    @abstractmethod
    def act_space(self) -> gym.Space:
        pass

    @abstractmethod
    def reset(self, seed: int | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pass

    @abstractmethod
    def step(self, actions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        pass

    @abstractmethod
    def get_global_obs(self) -> Any:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
