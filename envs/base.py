from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class SocialEnvWrapper(ABC):
    """Abstract base class for environment wrappers in CoffeeShop.

    All environment wrappers must accept and return data using a dictionary pattern
    keyed by agent IDs (e.g., {"agent_0": data, "agent_1": data}). This ensures
    consistent multi-agent data routing throughout the pipeline.
    """

    def __init__(self, env_name: str, agent_ids: List[str]) -> None:
        self.env_name = env_name
        self.agent_ids = agent_ids

    @abstractmethod
    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset the environment.

        Returns a tuple of (obs_dict, infos_dict), where each is a dictionary
        keyed by agent_id with the corresponding observation or info data.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, actions_dict: Dict[str, Any]) -> Tuple[
        Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]
    ]:
        """Perform one environment step given multi-agent actions.

        Args:
            actions_dict: Dictionary keyed by agent_id with action data.

        Returns:
            A tuple of (obs_dict, rewards_dict, terminated_dict, truncated_dict, infos_dict),
            where each is a dictionary keyed by agent_id with the corresponding data.
            All outputs must strictly follow the dictionary pattern.
        """
        raise NotImplementedError

    @abstractmethod
    def get_global_obs(self, obs_dict: Dict[str, Any], env_id: int = 0) -> Any:
        """Return a global observation for the centralized critic.

        In some environments, this might be a concatenation of all local
        observations; in others, it's a dedicated global state.
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release resources if any."""
        raise NotImplementedError
