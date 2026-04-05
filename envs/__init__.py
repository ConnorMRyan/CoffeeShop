from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class SocialEnvWrapper(ABC):
    """Base interface to normalize multi-agent envs for CoffeeShop.

    Implementations should wrap a specific environment (e.g., Overcooked, Crafter)
    and expose a consistent multi-agent step API.
    """

    @property
    @abstractmethod
    def agent_ids(self) -> List[str]:
        """List of agent identifiers present in the environment."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, seed: int | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment.

        Returns a tuple of (observations, infos), both dicts keyed by agent_id.
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self, actions: Dict[str, Any]
    ) -> Tuple[
        Dict[str, Any],  # observations per agent
        Dict[str, float],  # rewards per agent
        Dict[str, bool],   # terminated flags per agent
        Dict[str, bool],   # truncated flags per agent
        Dict[str, Any],    # infos per agent
    ]:
        """Perform one environment step given multi-agent actions."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release resources if any."""
        raise NotImplementedError
