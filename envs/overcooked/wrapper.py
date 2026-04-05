from __future__ import annotations
from typing import Any, Dict, Tuple

try:
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    from overcooked_ai_py.mdp.actions import Action
except Exception:  # pragma: no cover - optional dependency
    OvercookedGridworld = None  # type: ignore
    Action = None  # type: ignore

from envs.base import SocialEnvWrapper


class OvercookedSocialWrapper(SocialEnvWrapper):
    """Overcooked-AI wrapper for CoffeeShop's multi-agent API.

    Wraps the Overcooked environment to provide observations, rewards, and other data
    in the required dictionary format keyed by agent IDs.
    """

    def __init__(self) -> None:
        if OvercookedGridworld is None:
            raise ImportError(
                "overcooked-ai is not installed. Add it to requirements or install manually."
            )
        super().__init__("overcooked", ["agent_0", "agent_1"])
        self.mdp = OvercookedGridworld.from_layout_name("cramped_room", horizon=400)
        self._state = None

    def reset(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._state = self.mdp.get_standard_start_state()
        obs = {aid: self.mdp.lossless_state_encoding(self._state) for aid in self.agent_ids}
        infos = {aid: {} for aid in self.agent_ids}
        return obs, infos

    def step(self, actions_dict: Dict[str, Any]) -> Tuple[
        Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]
    ]:
        a0 = self._to_env_action(actions_dict.get("agent_0"))
        a1 = self._to_env_action(actions_dict.get("agent_1"))
        next_state, info = self.mdp.get_state_transition(self._state, (a0, a1))
        self._state = next_state
        obs = {aid: self.mdp.lossless_state_encoding(self._state) for aid in self.agent_ids}
        sparse_reward = sum(info['sparse_reward_by_agent'])
        rewards = {aid: float(sparse_reward) for aid in self.agent_ids}  # shared sparse reward
        terminated = {aid: False for aid in self.agent_ids}  # horizon not reached
        truncated = {aid: False for aid in self.agent_ids}
        infos = {aid: info for aid in self.agent_ids}
        return obs, rewards, terminated, truncated, infos

    def close(self) -> None:
        pass

    def _to_env_action(self, a: Any):
        if Action is None:
            raise RuntimeError("Overcooked Action enum unavailable")
        if isinstance(a, str):
            mapping = {
                "stay": (0, 0),
                "up": (0, -1),
                "down": (0, 1),
                "left": (-1, 0),
                "right": (1, 0),
                "interact": 'interact',
            }
            return mapping.get(a, (0, 0))
        return (0, 0)
