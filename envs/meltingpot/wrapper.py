"""
envs/meltingpot/wrapper.py — DeepMind MeltingPot integration
============================================================

Lightweight adapter that exposes MeltingPot substrates through CoffeeShop's
`SocialEnvWrapper` interface.

Design goals
------------
- Zero new training-time concepts: same dict-based multi-agent API as
  Overcooked, so it works with the existing vectorized runner and PPO stack.
- Safe optional dependency: raises a helpful ImportError if MeltingPot is not
  installed, without breaking other environments.
- Conservative observation encoding: pick a single ndarray channel per player
  (prefer RGB if present), flatten to float32 in [0, 1]. This keeps the
  encoder side simple for initial testing. You can switch to CNN by setting
  `--encoder=cnn --img_h/--img_w` if you want to treat observations as images
  end-to-end.

Notes
-----
MeltingPot uses the `dm_env` API. `reset()` and `step()` return a `TimeStep`
object with fields: `step_type`, `observation`, `reward`, `discount`.
- `observation` is typically a mapping from player index to an observation
  dict per player. We sample a single ndarray channel per player to build the
  policy observation. Preference order: 'RGB', then first ndarray found.
- `reward` is per-player (list/array). We return dense rewards per agent, and
  also expose a `sparse_rewards` dict that keeps only the positive component
  (>=0) to act as a sparse signal for the CoffeeShopMediator.
- Episode termination is signalled when `step_type` is LAST. We do not have a
  reliable way to distinguish true-terminal vs. timeout here, so we report
  `terminated=True` and `truncated=False` for LAST steps.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

from envs import SocialEnvWrapper

try:  # Optional dependency
    from meltingpot.python import substrate as mp_substrate  # type: ignore
    import dm_env  # type: ignore
    _MP_AVAILABLE = True
except Exception:  # pragma: no cover
    mp_substrate = None  # type: ignore
    dm_env = None  # type: ignore
    _MP_AVAILABLE = False


class MeltingPotWrapper(SocialEnvWrapper):
    """DeepMind MeltingPot substrate wrapper for CoffeeShop.

    Parameters
    ----------
    scenario : str
        Name of the MeltingPot scenario (e.g., 'clean_up', 'harvest').
    horizon : int
        Maximum episode length. Passed through to the substrate config where
        supported; otherwise used to early-stop on our side.
    render_mode : str | None
        Accepted for API compatibility. Not currently used.
    obs_key : str | None
        Specific observation key to use per player (e.g., 'RGB'). If None,
        we auto-select 'RGB' if present, else the first ndarray in the per-
        player observation dict.
    """

    def __init__(
        self,
        scenario: str = "clean_up",
        horizon: int = 400,
        render_mode: Optional[str] = None,
        obs_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if not _MP_AVAILABLE:
            raise ImportError(
                "meltingpot is not installed. Install with: pip install meltingpot"
            )

        # Build substrate from scenario config
        config = mp_substrate.get_config(scenario)
        # Try to override episode length if supported by config
        try:
            if hasattr(config, "lab2d_settings") and isinstance(config.lab2d_settings, dict):
                config.lab2d_settings.setdefault("episodeLength", horizon)
        except Exception:
            pass  # If the config format changes, we still proceed.
        self._env = mp_substrate.build(config)

        # Agent ids are 0..N-1 in stable order
        try:
            n_players = int(self._env.num_players)
        except Exception:
            # Fallback: infer from action_spec structure if attr missing
            spec = self._env.action_spec()
            n_players = len(spec) if isinstance(spec, (list, tuple)) else int(getattr(spec, "num_values", 1))
        self._agent_ids: List[str] = [f"agent_{i}" for i in range(n_players)]

        self._render_mode = render_mode
        self._horizon = int(horizon)
        self._obs_key = obs_key

        # Reset once to infer dimensions from a live observation
        ts = self._env.reset()
        sample_obs = self._extract_player_obs(ts.observation)
        # Ensure homogeneous shapes across players
        first = sample_obs[self._agent_ids[0]]
        self._obs_dim = int(first.size)
        self._global_obs_dim = self._obs_dim * len(self._agent_ids)

        # Action dim from per-player action spec (assumed homogeneous)
        action_spec = self._per_player_action_spec()
        self._action_dim = int(getattr(action_spec, "num_values", 0) or getattr(action_spec, "num_actions", 0))
        if self._action_dim <= 0:
            # Try dict-like spec
            try:
                self._action_dim = int(action_spec.maximum - action_spec.minimum + 1)
            except Exception:
                raise ValueError("Unsupported action spec; expected discrete action space.")

        # Internal step counter for horizon limiting (belt-and-braces)
        self._step_in_episode = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def agent_ids(self) -> List[str]:
        return self._agent_ids

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def global_obs_dim(self) -> int:
        return self._global_obs_dim

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # MeltingPot/dm_env reset does not take a seed directly (env is seeded via config)
        ts = self._env.reset()
        self._step_in_episode = 0
        obs_dict = self._extract_player_obs(ts.observation)
        return obs_dict, {}

    def step(self, actions: Dict[str, int]):
        # Order the joint action by player index
        joint: List[int] = [int(actions[aid]) for aid in self._agent_ids]
        ts: "dm_env.TimeStep" = self._env.step(joint)
        self._step_in_episode += 1

        obs_dict = self._extract_player_obs(ts.observation)

        # Rewards may be list/np.ndarray or scalar if shared; coerce per-agent
        rewards_np = np.asarray(ts.reward if ts.reward is not None else np.zeros(len(self._agent_ids), dtype=np.float32), dtype=np.float32)
        if rewards_np.ndim == 0:
            rewards_np = np.full((len(self._agent_ids),), float(rewards_np), dtype=np.float32)

        rewards: Dict[str, float] = {aid: float(rewards_np[i]) for i, aid in enumerate(self._agent_ids)}
        sparse: Dict[str, float] = {aid: float(max(0.0, rewards_np[i])) for i, aid in enumerate(self._agent_ids)}

        is_last = bool(ts.last()) if hasattr(ts, "last") else (getattr(ts, "step_type", None) == getattr(dm_env, "StepType", object) .LAST if dm_env else False)
        terminated = {aid: is_last for aid in self._agent_ids}
        truncated = {aid: False for aid in self._agent_ids}

        infos = {
            "sparse_rewards": sparse,
            "has_delivery": any(v > 0.0 for v in sparse.values()),
            "mp_discount": float(ts.discount) if ts.discount is not None else 1.0,
        }

        # If we track our own horizon, early truncate
        if self._horizon and self._step_in_episode >= self._horizon and not is_last:
            truncated = {aid: True for aid in self._agent_ids}
            terminated = {aid: False for aid in self._agent_ids}

        return obs_dict, rewards, terminated, truncated, infos

    def get_global_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [obs_dict[aid] for aid in self._agent_ids]
        return torch.cat(parts, dim=-1)

    def render(self) -> None:
        # No-op: headless by default. Hook up RGB frames here if desired.
        pass

    def close(self) -> None:
        if hasattr(self, "_env") and self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _per_player_action_spec(self):
        spec = self._env.action_spec()
        if isinstance(spec, (list, tuple)):
            return spec[0]
        return spec

    def _extract_player_obs(self, ts_observation: Any) -> Dict[str, torch.Tensor]:
        """Pick one ndarray channel per player and flatten to float32 tensor.

        Preference order within a player's observation dict: 'RGB' -> first ndarray.
        If observation is already a list/array per player, use it directly if ndarray.
        """
        result: Dict[str, torch.Tensor] = {}

        # Case A: mapping from player index to obs-dict per player
        if isinstance(ts_observation, dict) and all(isinstance(k, (int, np.integer)) for k in ts_observation.keys()):
            for i, aid in enumerate(self._agent_ids):
                per = ts_observation.get(i)
                arr = self._select_array_from_obs(per)
                result[aid] = self._to_tensor(arr)
            return result

        # Case B: list/tuple of per-player obs-dicts
        if isinstance(ts_observation, (list, tuple)):
            for i, aid in enumerate(self._agent_ids):
                per = ts_observation[i]
                arr = self._select_array_from_obs(per)
                result[aid] = self._to_tensor(arr)
            return result

        # Case C: single obs shared — give each agent its own tensor.
        # arr.copy() is required because _to_tensor calls torch.from_numpy()
        # which shares memory with the numpy array; without a copy every agent
        # would alias the same underlying buffer.
        arr = self._select_array_from_obs(ts_observation)
        for aid in self._agent_ids:
            result[aid] = self._to_tensor(arr.copy())
        return result

    def _select_array_from_obs(self, per_player_obs: Any) -> np.ndarray:
        # If obs is already an ndarray, use it
        if isinstance(per_player_obs, np.ndarray):
            return per_player_obs
        # If dict-like, prefer specified key, then 'RGB', else first ndarray value
        if isinstance(per_player_obs, dict):
            if self._obs_key and self._obs_key in per_player_obs and isinstance(per_player_obs[self._obs_key], np.ndarray):
                return per_player_obs[self._obs_key]
            if "RGB" in per_player_obs and isinstance(per_player_obs["RGB"], np.ndarray):
                return per_player_obs["RGB"]
            for v in per_player_obs.values():
                if isinstance(v, np.ndarray):
                    return v
        # Last resort: try to convert to array
        return np.asarray(per_player_obs)

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        a = np.asarray(arr)
        if a.dtype != np.float32:
            if np.issubdtype(a.dtype, np.integer) and a.size > 0 and a.max() > 1:
                a = a.astype(np.float32) / 255.0
            else:
                a = a.astype(np.float32)
        flat = a.reshape(-1)
        return torch.from_numpy(flat)
