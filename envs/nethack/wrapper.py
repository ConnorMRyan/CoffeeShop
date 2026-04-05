from __future__ import annotations
from typing import Any, Dict, List, Tuple

try:
    import nle  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    nle = None  # type: ignore

from envs import SocialEnvWrapper


class NetHackWrapper(SocialEnvWrapper):
    """Placeholder wrapper for NetHack Learning Environment (NLE).

    For now this is a stub that presents a 1-agent multi-agent API.
    """

    def __init__(self, seed: int | None = None) -> None:
        if nle is None:
            raise ImportError("nle is not installed. Enable it in requirements to use this wrapper.")
        self._agent_ids = ["agent_0"]
        self._env = None  # real integration can wrap gymnasium NLE env later
        self._seed = seed

    @property
    def agent_ids(self) -> List[str]:
        return self._agent_ids

    def reset(self, seed: int | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        obs = {"agent_0": None}
        infos = {"agent_0": {"note": "NLE environment not fully wired yet"}}
        return obs, infos

    def step(
        self, actions: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        obs = {"agent_0": None}
        rewards = {"agent_0": 0.0}
        terminated = {"agent_0": False}
        truncated = {"agent_0": False}
        infos = {"agent_0": {"note": "NLE environment not fully wired yet"}}
        return obs, rewards, terminated, truncated, infos

    def close(self) -> None:
        pass
