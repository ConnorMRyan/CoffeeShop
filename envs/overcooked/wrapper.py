from __future__ import annotations
from typing import Any, Dict, Tuple, List
import numpy as np

try:
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    from overcooked_ai_py.mdp.actions import Action
except Exception:  # pragma: no cover - optional dependency
    OvercookedGridworld = None  # type: ignore
    Action = None  # type: ignore

from envs import SocialEnvWrapper


class DummySpace:
    def __init__(self, shape, n=None):
        self.shape = shape
        self.n = n

class OvercookedSocialWrapper(SocialEnvWrapper):
    """Overcooked-AI wrapper for CoffeeShop's multi-agent API.

    Wraps the Overcooked environment to provide observations, rewards, and other data
    in the required dictionary format keyed by agent IDs.
    """

    def __init__(self, layout_name: str = "cramped_room") -> None:
        if OvercookedGridworld is None:
            raise ImportError(
                "overcooked-ai is not installed. Add it to requirements or install manually."
            )
        self._agent_ids = ["agent_0", "agent_1"]
        self.horizon = 400
        self._step_count = 0
        self.mdp = OvercookedGridworld.from_layout_name(layout_name, horizon=self.horizon)
        self._state = None
        
        # We manually compute the flattened size of lossless encoding. For cramped_room with 2 agents, it's typically 96.
        # We use a dummy space object to keep it dependency-light, or import gym.spaces if installed.
        dummy_state = self.mdp.get_standard_start_state()
        dummy_obs = self.mdp.lossless_state_encoding(dummy_state)[0]
        flat_dim = np.prod(np.array(dummy_obs).shape)
        
        self.obs_space = DummySpace(shape=(flat_dim,))
        self.act_space = DummySpace(shape=(), n=6)

    def _flatten_obs(self, obs: list) -> np.ndarray:
        return np.array(obs, dtype=np.float32).flatten()

    def reset(self, seed: int | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        self._step_count = 0
        self._state = self.mdp.get_standard_start_state()
        encodings = self.mdp.lossless_state_encoding(self._state)
        # Overcooked returning list of obs per agent (e.g. index 0 -> agent_0, index 1 -> agent_1)
        obs = {
            self.agent_ids[0]: self._flatten_obs(encodings[0]),
            self.agent_ids[1]: self._flatten_obs(encodings[1])
        }
        infos = {aid: {} for aid in self.agent_ids}
        return obs, infos

    def step(self, actions_dict: Dict[str, Any]) -> Tuple[
        Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]
    ]:
        a0 = self._to_env_action(actions_dict.get("agent_0"))
        a1 = self._to_env_action(actions_dict.get("agent_1"))
        next_state, info = self.mdp.get_state_transition(self._state, (a0, a1))
        self._state = next_state
        
        encodings = self.mdp.lossless_state_encoding(self._state)
        obs = {
            self.agent_ids[0]: self._flatten_obs(encodings[0]),
            self.agent_ids[1]: self._flatten_obs(encodings[1])
        }
        
        sparse_reward = sum(info['sparse_reward_by_agent'])
        rewards = {aid: float(sparse_reward) for aid in self.agent_ids}  # shared sparse reward
        terminated = {aid: False for aid in self.agent_ids}
        self._step_count += 1
        is_truncated = self._step_count >= self.horizon
        truncated = {aid: is_truncated for aid in self.agent_ids}
        infos = {aid: info for aid in self.agent_ids}
        return obs, rewards, terminated, truncated, infos

    @property
    def agent_ids(self) -> List[str]:
        return self._agent_ids

    def close(self) -> None:
        pass

    def _to_env_action(self, a: Any):
        if Action is None:
            raise RuntimeError("Overcooked Action enum unavailable")
        # Action map: 0->stay, 1->up, 2->down, 3->right, 4->left, 5->interact
        # Note: mapping direction values correctly for overcooked mdp
        mapping = {
            0: (0, 0),
            1: (0, -1),
            2: (0, 1),
            3: (1, 0),
            4: (-1, 0),
            5: 'interact',
        }
        if isinstance(a, str):
            str_mapping = {
                "stay": (0, 0), "up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0), "interact": 'interact',
            }
            return str_mapping.get(a, (0, 0))
        return mapping.get(a, (0, 0))
