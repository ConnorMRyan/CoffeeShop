"""
envs/crafter/wrapper.py — Crafter Multi-Agent Virtual Integration
==================================================================
CoffeeShop wrapper that supports multiple parallel Crafter instances
to satisfy the SocialEnvWrapper API for multi-agent evaluation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from envs import SocialEnvWrapper

try:
    import crafter as _crafter_module   # type: ignore
    _CRAFTER_AVAILABLE = True
except Exception:                       # pragma: no cover
    _CRAFTER_AVAILABLE = False

# Fixed constants derived from Crafter's observation and action specs.
_IMG_H      = 64
_IMG_W      = 64
_IMG_C      = 3
_OBS_DIM    = _IMG_H * _IMG_W * _IMG_C   # 12 288
_ACTION_DIM = 17                          # Discrete(17)


class CrafterSocialWrapper(SocialEnvWrapper):
    """
    Adapts Crafter to a multi-agent API by running N independent environments.
    """

    def __init__(
            self,
            num_agents:  int        = 1,
            length:      int        = 10_000,
            seed:        int | None = None,
            render_mode: str | None = None,
            reward:      bool       = True,
            **kwargs,
    ) -> None:
        if not _CRAFTER_AVAILABLE:
            raise ImportError("crafter is not installed. Run: pip install crafter>=1.8.0")

        self._num_agents = num_agents
        self._agent_ids = [f"agent_{i}" for i in range(num_agents)]
        self._length    = length

        # Initialize one environment per agent seat
        self._envs = [
            _crafter_module.Env(length=length, seed=(seed + i if seed is not None else None), reward=reward)
            for i in range(num_agents)
        ]

        # Track achievements separately for each agent to detect new unlocks
        self._prev_achievements: List[Dict[str, int]] = [{} for _ in range(num_agents)]

    @property
    def agent_ids(self) -> List[str]:
        return self._agent_ids

    @property
    def obs_dim(self) -> int:
        return _OBS_DIM

    @property
    def action_dim(self) -> int:
        return _ACTION_DIM

    @property
    def global_obs_dim(self) -> int:
        # In this virtual MA setup, global obs isn't strictly defined across envs,
        # so we return a flattened local obs for compatibility.
        return _OBS_DIM

    def reset(
            self,
            seed:    Optional[int]  = None,
            options: Optional[dict] = None,
    ) -> Tuple[Dict[str, torch.Tensor], dict]:
        obs_dict = {}
        for i, env in enumerate(self._envs):
            # Deterministic offset for multi-agent seeds
            if seed is not None:
                env.seed(seed + i)

            raw_obs = env.reset()
            obs_dict[f"agent_{i}"] = self._encode(raw_obs)
            self._prev_achievements[i] = {}

        return obs_dict, {}

    def step(
            self,
            actions: Dict[str, int],
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        dict,
    ]:
        obs_dict, rew_dict, term_dict, trunc_dict = {}, {}, {}, {}

        # Aggregated info for the CoffeeShop mediator
        all_sparse_rewards = {}
        all_new_achievements = []
        any_delivery = False

        for i, env in enumerate(self._envs):
            aid = f"agent_{i}"
            action = int(actions[aid])

            raw_obs, reward, done, info = env.step(action)

            obs_dict[aid] = self._encode(raw_obs)
            rew_dict[aid] = float(reward)

            # Terminated vs truncated logic
            discount = float(info.get("discount", 1.0))
            term_dict[aid] = done and (discount == 0.0)
            trunc_dict[aid] = done and (discount == 1.0)

            # Achievement tracking
            ach = info.get("achievements", {})
            new_ach = [k for k, v in ach.items() if v > self._prev_achievements[i].get(k, 0)]
            self._prev_achievements[i] = dict(ach)

            # CoffeeShop specific signaling
            sparse_val = max(0.0, float(reward))
            all_sparse_rewards[aid] = sparse_val
            if sparse_val > 0:
                any_delivery = True
            all_new_achievements.extend(new_ach)

        step_infos = {
            "sparse_rewards": all_sparse_rewards,
            "has_delivery": any_delivery,
            "new_achievements": all_new_achievements,
            "individual_infos": [env.metadata for env in self._envs] # Pass through raw metadata
        }

        return obs_dict, rew_dict, term_dict, trunc_dict, step_infos

    def get_global_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # For simplicity in virtual MA, we just use agent_0's view as global
        return obs_dict["agent_0"]

    def render(self) -> None:
        # To see them both, you'd need to tile the numpy arrays from env.render()
        # For now, we'll keep it as a noop to avoid blocking the eval script.
        pass

    def close(self) -> None:
        for env in self._envs:
            env.close()

    def _encode(self, raw_obs: np.ndarray) -> torch.Tensor:
        from einops import rearrange
        # Normalization and HWC -> CHW transposition via einops
        arr = torch.as_tensor(raw_obs, dtype=torch.float32) / 255.0
        # Explicitly move to CHW then flatten
        chw = rearrange(arr, 'h w c -> c h w')
        return chw.reshape(-1)

# Alias for import compatibility
CrafterWrapper = CrafterSocialWrapper