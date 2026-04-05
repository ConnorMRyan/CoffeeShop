from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from envs import SocialEnvWrapper


@dataclass
class MediatorConfig:
    """Configuration for the CoffeeShop Mediator."""
    shared_reward: bool = True


class Mediator:
    """Coordinates multi-agent interaction between env and agents.

    Responsibilities:
      - Own an environment wrapper implementing `SocialEnvWrapper`.
      - Provide a simple loop interface: reset -> step(actions) -> outputs.
      - Optionally perform reward shaping or mediation logic.
    """

    def __init__(self, env: SocialEnvWrapper, config: MediatorConfig | None = None) -> None:
        self.env = env
        self.config = config or MediatorConfig()

    @property
    def agent_ids(self) -> List[str]:
        return self.env.agent_ids

    def reset(self, seed: int | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self.env.reset(seed=seed)

    def step(
        self, actions: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        # Hook for mediation logic could go here (e.g., social shaping)
        obs, rewards, terminated, truncated, infos = self.env.step(actions)
        if self.config.shared_reward:
            # Example: average reward across agents
            if len(rewards) > 0:
                avg = sum(rewards.values()) / len(rewards)
                rewards = {aid: float(avg) for aid in rewards}
        return obs, rewards, terminated, truncated, infos

    def close(self) -> None:
        self.env.close()
