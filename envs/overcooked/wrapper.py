from __future__ import annotations

import contextlib
import io
import logging
import os
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

# ---------------------------------------------------------------------------
# Custom layout resolution
# ---------------------------------------------------------------------------

_CUSTOM_LAYOUTS_DIR = os.path.join(os.path.dirname(__file__), "layouts")


def _load_mdp(layout_name: str) -> OvercookedGridworld:
    """
    Load an OvercookedGridworld by name.  Checks the project-local
    envs/overcooked/layouts/ directory first so custom layouts can be added
    without touching the installed overcooked_ai_py package.
    """
    custom_path = os.path.join(_CUSTOM_LAYOUTS_DIR, layout_name + ".layout")
    if os.path.isfile(custom_path):
        with open(custom_path) as f:
            params = eval(f.read())  # same eval-based format as overcooked's own layouts
        # Strip indentation whitespace from each row; drop empty lines produced
        # by leading/trailing newlines in triple-quoted strings.
        grid = [row.strip() for row in params["grid"].split("\n") if row.strip()]
        del params["grid"]
        params["layout_name"] = layout_name
        return OvercookedGridworld.from_grid(grid, params)
    return OvercookedGridworld.from_layout_name(layout_name)

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
            inactivity_penalty_magnitude: float = 0.01,
            inactivity_steps: int = 3,
            render_mode: str | None = None,
            **kwargs,
    ) -> None:
        self._layout_name = layout_name
        self._horizon = horizon
        self._reward_shaping_factor = reward_shaping_factor
        self._inactivity_penalty_magnitude = inactivity_penalty_magnitude
        self._inactivity_steps = inactivity_steps
        self._render_mode = render_mode
        self._agent_ids = ["agent_0", "agent_1"]

        # Inactivity tracking
        self._prev_positions: Dict[str, Tuple[int, int]] = {}
        self._inactivity_counts: Dict[str, int] = {aid: 0 for aid in self._agent_ids}
        self._total_inactivity_penalty = 0.0

        # Suppress Overcooked-AI startup chatter
        with contextlib.redirect_stdout(io.StringIO()):
            self._mdp = _load_mdp(layout_name)
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
        
        # Reset inactivity tracking
        self._prev_positions = {
            aid: self._env.state.players[i].position 
            for i, aid in enumerate(self._agent_ids)
        }
        self._inactivity_counts = {aid: 0 for aid in self._agent_ids}
        self._total_inactivity_penalty = 0.0

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
        next_state, team_reward, done, info = self._env.step(joint_action)

        # Safely extract per-agent reward components
        shaped = info.get("shaped_r_by_agent", [0.0] * _N_AGENTS)
        sparse = info.get("sparse_r_by_agent",  [0.0] * _N_AGENTS)

        # team_reward is a shared delivery bonus.  Giving every
        # agent the full amount inflates the reward by N_agents.  Each agent
        # receives an equal 1/N share, then adds their own shaped guidance.
        team_reward_per_agent = float(team_reward) / _N_AGENTS
        rewards = {
            aid: team_reward_per_agent + float(shaped[i]) * self._reward_shaping_factor
            for i, aid in enumerate(self._agent_ids)
        }

        # Inactivity and Stall Logic
        step_inactivity_penalty = 0.0
        step_stalls = {aid: 0 for aid in self._agent_ids}
        for i, aid in enumerate(self._agent_ids):
            curr_pos = next_state.players[i].position
            prev_pos = self._prev_positions.get(aid)
            is_interacting = (joint_action[i] == Action.INTERACT)
            
            # A "stall" is either Action.STAY (index 0) OR moving into a wall (curr_pos == prev_pos when moving)
            is_stay = (actions[aid] == 0)
            is_into_wall = (curr_pos == prev_pos and not is_stay and not is_interacting)
            if is_stay or is_into_wall:
                step_stalls[aid] = 1

            if curr_pos == prev_pos and not is_interacting:
                self._inactivity_counts[aid] += 1
            else:
                self._inactivity_counts[aid] = 0
            
            if self._inactivity_counts[aid] >= self._inactivity_steps:
                penalty = self._inactivity_penalty_magnitude
                step_inactivity_penalty += penalty
                rewards[aid] -= penalty
            
            self._prev_positions[aid] = curr_pos

        self._total_inactivity_penalty += step_inactivity_penalty

        # Preserve sparse rewards for mediator / gossip logic
        sparse_rewards  = {aid: float(sparse[i])  for i, aid in enumerate(self._agent_ids)}
        shaped_rewards  = {aid: float(shaped[i])  for i, aid in enumerate(self._agent_ids)}

        # Disambiguate horizon expiry from genuine termination.
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
            "team_reward":        float(team_reward),
            "sparse_rewards":     sparse_rewards,
            "shaped_rewards":     shaped_rewards,
            "inactivity_penalty": step_inactivity_penalty,
            "total_inactivity_penalty": self._total_inactivity_penalty,
            "action_entropy_stall": step_stalls,
            "has_delivery":       any(s > 0 for s in sparse),
            "is_timeout":         is_timeout,
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