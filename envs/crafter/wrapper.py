from __future__ import annotations
from typing import Any, Dict, List, Tuple

try:
    import crafter  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    crafter = None  # type: ignore

from envs import SocialEnvWrapper


class CrafterWrapper(SocialEnvWrapper):
    """Placeholder wrapper for Crafter benchmark.

    Normalizes Crafter's single-agent environment to a 1-agent multi-agent API.
    """

    def __init__(self, seed: int | None = None, reward: bool = True) -> None:
        if crafter is None:
            raise ImportError("crafter is not installed. Enable it in requirements to use this wrapper.")
        # NOTE: Keep a placeholder; real integration can wrap gymnasium env
        self._agent_ids = ["agent_0"]
        self._env = None  # to be instantiated when dependency is available
        self._seed = seed
        self._reward = reward

    @property
    def agent_ids(self) -> List[str]:
        return self._agent_ids

    def reset(self, seed: int | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Placeholder observation
        obs = {"agent_0": None}
        infos = {"agent_0": {"note": "Crafter environment not fully wired yet"}}
        return obs, infos

    def step(
        self, actions: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        obs = {"agent_0": None}
        rewards = {"agent_0": 0.0}
        terminated = {"agent_0": False}
        truncated = {"agent_0": False}
        infos = {"agent_0": {"note": "Crafter environment not fully wired yet"}}
        return obs, rewards, terminated, truncated, infos

    def close(self) -> None:
        pass
