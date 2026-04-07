"""
envs/overcooked/wrapper.py  —  Real Overcooked-AI integration
================================================================
Fixed reward handling, dynamic observation sizing, safer encoding,
cleaner info logging, and correct terminated/truncated signalling
for CoffeeShop's SocialEnvWrapper API.

Bug fixes over previous version
---------------------------------
1.  terminated / truncated  — OvercookedEnv sets done=True for both true
    task completion and horizon expiry.  We now inspect info["done_info"]
    for a "timeout" key to disambiguate:
      • horizon expiry  → terminated=False, truncated=True
      • genuine done    → terminated=True,  truncated=False
    This matters for PPO's bootstrapping: a truncated episode should still
    bootstrap V(s'), while a terminated one should not.

2.  next_state vs self._env.state  — OvercookedEnv.step() returns the raw
    joint next-state object as its first return value.  We pass that directly
    to _encode_state rather than re-reading self._env.state, which could
    theoretically diverge if the environment is stepped again before encoding
    (e.g. in a vectorised wrapper).

3.  team_reward double-counted  — the previous code gave every agent the
    full team_reward, inflating the reward signal by N_agents.  team_reward
    is a shared delivery bonus; each agent receives an equal 1/N share.
    Shaped rewards remain per-agent and are added on top of each agent's share.
"""

from __future__ import annotations

import contextlib
import io
import logging
from typing import Dict, Tuple

import numpy as np
import torch

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import Action, OvercookedGridworld
from envs import SocialEnvWrapper

# Silence noisy internal planner logs globally
logging.getLogger("overcooked_ai_py.planning.planners").setLevel(logging.WARNING)

# Pre-build the action lookup once at import time
_ACTION_MAP = Action.ALL_ACTIONS

# Number of agents is fixed for Overcooked-AI (always a two-player game)
_N_AGENTS = 2


class OvercookedSocialWrapper(SocialEnvWrapper):
    """
    Live Overcooked-AI wrapper conforming to CoffeeShop's SocialEnvWrapper API.
    """

    _ACTION_DIM = 6

    def __init__(
            self,
            layout_name: str = "cramped_room",
            horizon: int = 400,
            reward_shaping_factor: float = 1.0,
            render_mode: str | None = None,
            **kwargs,
    ) -> None:
        self._layout_name = layout_name
        self._horizon = horizon
        self._reward_shaping_factor = reward_shaping_factor
        self._render_mode = render_mode
        self._agent_ids = ["agent_0", "agent_1"]

        # Suppress Overcooked-AI startup chatter
        with contextlib.redirect_stdout(io.StringIO()):
            self._mdp = OvercookedGridworld.from_layout_name(layout_name)
            self._env = OvercookedEnv.from_mdp(
                self._mdp,
                horizon=horizon,
                info_level=0,
            )
            self._env.reset()

        # Dynamically infer observation dimensions from the actual layout
        sample_raw = self._mdp.lossless_state_encoding(self._env.state)
        self._obs_dim = int(np.asarray(sample_raw[0], dtype=np.float32).size)
        self._global_obs_dim = self._obs_dim * len(self._agent_ids)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def agent_ids(self):
        return self._agent_ids

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._ACTION_DIM

    @property
    def global_obs_dim(self) -> int:
        return self._global_obs_dim

    # ------------------------------------------------------------------
    # Gym-style API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Overcooked-AI does not expose Gym-style per-reset seeding here,
        so the seed argument is accepted for API compatibility but unused.
        """
        with contextlib.redirect_stdout(io.StringIO()):
            self._env.reset()
        obs = self._encode_state(self._env.state)
        return obs, {}

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
        # Convert discrete policy outputs into Overcooked joint actions
        joint_action = tuple(_ACTION_MAP[actions[aid]] for aid in self._agent_ids)

        # Step the underlying environment.
        # FIX (bug 2): use the returned next_state directly rather than
        # re-reading self._env.state after the step, which could diverge
        # in asynchronous / vectorised settings.
        next_state, team_reward, done, info = self._env.step(joint_action)

        # Safely extract per-agent reward components
        shaped = info.get("shaped_r_by_agent", [0.0] * _N_AGENTS)
        sparse = info.get("sparse_r_by_agent",  [0.0] * _N_AGENTS)

        # FIX (bug 3): team_reward is a shared delivery bonus.  Giving every
        # agent the full amount inflates the reward by N_agents.  Each agent
        # receives an equal 1/N share, then adds their own shaped guidance.
        team_reward_per_agent = float(team_reward) / _N_AGENTS
        rewards = {
            aid: team_reward_per_agent + float(shaped[i]) * self._reward_shaping_factor
            for i, aid in enumerate(self._agent_ids)
        }

        # Preserve sparse rewards for mediator / gossip logic
        sparse_rewards  = {aid: float(sparse[i])  for i, aid in enumerate(self._agent_ids)}
        shaped_rewards  = {aid: float(shaped[i])  for i, aid in enumerate(self._agent_ids)}

        # FIX (bug 1): disambiguate horizon expiry from genuine termination.
        # OvercookedEnv populates info["done_info"] with a "timeout" key when
        # the episode ends because the horizon was reached.  If that key is
        # present and truthy the episode was truncated, not terminated.
        done_info = info.get("done_info", {})
        is_timeout = bool(done_info.get("timeout", False))

        if done and is_timeout:
            # Horizon expiry: episode should be bootstrapped by PPO
            terminated = {aid: False for aid in self._agent_ids}
            truncated  = {aid: True  for aid in self._agent_ids}
        else:
            # Genuine task completion (or environment error)
            terminated = {aid: bool(done) for aid in self._agent_ids}
            truncated  = {aid: False      for aid in self._agent_ids}

        infos = {
            "team_reward":    float(team_reward),
            "sparse_rewards": sparse_rewards,
            "shaped_rewards": shaped_rewards,
            "has_delivery":   any(s > 0 for s in sparse),
            "is_timeout":     is_timeout,
        }

        return self._encode_state(next_state), rewards, terminated, truncated, infos

    def get_global_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([obs_dict[aid] for aid in self._agent_ids], dim=-1)

    def render(self) -> None:
        """
        Placeholder. OvercookedEnv.render() typically needs additional setup
        for a live window; keeping this quiet for training compatibility.
        """
        if self._render_mode == "human":
            pass

    def close(self):
        if hasattr(self, "_env") and hasattr(self._env, "close"):
            self._env.close()
        else:
            # OvercookedEnv often doesn't need a formal close,
            # so we can just pass or log a debug message.
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_state(self, state) -> Dict[str, torch.Tensor]:
        """
        Flatten the lossless Overcooked state encoding into per-agent vectors.
        """
        raw = self._mdp.lossless_state_encoding(state)

        encoded = {}
        for i, aid in enumerate(self._agent_ids):
            arr = np.asarray(raw[i], dtype=np.float32).reshape(-1)
            encoded[aid] = torch.from_numpy(arr)
        return encoded