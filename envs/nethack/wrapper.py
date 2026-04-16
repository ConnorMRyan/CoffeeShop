"""
envs/nethack/wrapper.py — NetHack Learning Environment (NLE) integration
=========================================================================
Single-agent NLE environment wrapped to CoffeeShop's SocialEnvWrapper API.

NLE 0.9.x ships as a gym (not gymnasium) environment.  This wrapper
handles the old-style 4-tuple step return and converts it to gymnasium
semantics (separate terminated / truncated flags, infos dict at reset).

Observation encoding
--------------------
Two components concatenated into a flat float32 vector:

  1. Glyphs  (21 × 79 = 1 659 values)
     The dungeon map as glyph IDs, normalized to [0, 1] by dividing by
     nle.nethack.MAX_GLYPH (≈ 5 976).

  2. BL-stats (27 values)
     Scalar statistics from NetHack's bottom line (HP, depth, gold, …).
     Divided by _BLSTAT_SCALE (per-field) then clipped to [-5, 5].

Total obs_dim  = 1 686.
global_obs_dim = obs_dim  (single-agent env; global obs = local obs).

Action space
------------
Mirrors NLE's Discrete action space.  action_dim is read from the live
environment so it adapts automatically to any NLE variant (Score, Challenge,
Staircase, …).

Terminated vs truncated
-----------------------
NLE 0.9.x (gym 0.23) sets info["end_status"] to one of three StepStatus
enum values:
  • ABORTED (-1) — horizon / step limit reached → truncated=True
  • DEATH   ( 1) — player died                 → terminated=True
  • RUNNING ( 0) — episode still ongoing         → both False

is_ascended=True in info is also a genuine termination (player won).

Sparse reward
-------------
NLE's reward is the per-step score delta — naturally sparse (mostly 0,
positive on kills, item discoveries, etc.).  We surface it as both the
dense reward and the sparse_reward in infos so the CoffeeShopMediator can treat
score events as high-priority memories.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from envs import SocialEnvWrapper

try:
    import gym          # NLE 0.9.x registers its envs with old gym
    import nle          # noqa: F401 — side-effect: registers NLE gym envs
    import nle.nethack  # for MAX_GLYPH constant
    _NLE_AVAILABLE = True
except Exception:       # pragma: no cover
    _NLE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Per-field normalization scale for the 27 BL-stat values.
# Chosen to put typical in-game values in roughly [0, 1]; extreme values are
# clipped to [-5, 5] after division.  Order matches NLE's blstats layout:
# str_pct, str, dex, con, int, wis, cha, score, hp, maxhp, depth, gold,
# energy, max_energy, ac, monster_lvl, exp_lvl, exp_pts, time, hunger,
# carry_cap, dungeon_num, level_num, condition, align, unused, unused
# ---------------------------------------------------------------------------
_BLSTAT_SCALE = np.array([
    100.0,    # strength_percentage   (0–100)
     25.0,    # strength              (3–25)
     25.0,    # dexterity
     25.0,    # constitution
     25.0,    # intelligence
     25.0,    # wisdom
     25.0,    # charisma
    1e5,      # score                 (unbounded; large divisor keeps it small)
    1000.0,   # hitpoints
    1000.0,   # max_hitpoints
     60.0,    # depth                 (≤ 60 in Gehennom)
    1e5,      # gold
    1000.0,   # energy
    1000.0,   # max_energy
     40.0,    # armor_class           (can be negative; centred around 0)
     30.0,    # monster_level
     30.0,    # experience_level      (1–30)
    1e7,      # experience_points
    1e6,      # time                  (step counter)
      8.0,    # hunger_state          (0 = Satiated … 7 = Starved)
    1000.0,   # carrying_capacity
     10.0,    # dungeon_number        (0–9 in vanilla)
     60.0,    # level_number
     32.0,    # condition             (bitmask)
      3.0,    # alignment             (0–2)
      1.0,    # unused
      1.0,    # unused
], dtype=np.float32)

_N_BLSTATS = 27


class NLESocialWrapper(SocialEnvWrapper):
    """
    Live NetHack Learning Environment wrapper for CoffeeShop.

    Exposes a single-agent multi-agent API (agent_ids = ["agent_0"]) so
    it slots into the same training and evaluation pipelines as Overcooked.

    Parameters
    ----------
    env_name : str
        Any registered NLE gym environment id.
        Defaults to "NetHackScore-v0" (score-maximisation task).
    horizon : int
        Maximum steps per episode.  NLE's built-in TimeLimit is used if
        the chosen env already wraps one; otherwise we apply our own cap
        via max_episode_steps.
    render_mode : str | None
        Accepted for API compatibility but ignored — NLE renders to a
        terminal and does not support a headless render_mode kwarg.
    seed : int | None
        Random seed forwarded to env.seed() at reset time.
    """

    def __init__(
            self,
            env_name:    str       = "NetHackScore-v0",
            horizon:     int       = 500,
            render_mode: str | None = None,   # accepted, unused
            seed:        int | None = None,
            **kwargs,
    ) -> None:
        if not _NLE_AVAILABLE:
            raise ImportError(
                "nle is not installed. "
                "Run: pip install nle==0.9.1  (requires cmake + build-essential)"
            )

        self._agent_ids = ["agent_0"]
        self._seed      = seed
        self._horizon   = horizon

        # Build the underlying gym env.  Pass max_episode_steps so the
        # TimeLimit wrapper is applied at the requested horizon length.
        self._env = gym.make(env_name, max_episode_steps=horizon)

        # Seed the env (gym 0.21-style; NLE 0.9.x supports this)
        if seed is not None:
            self._env.seed(seed)

        # ── Derive fixed dimensions from the live environment ──────────────
        # Glyph plane: always 21 × 79 in NLE
        self._glyph_rows = 21
        self._glyph_cols = 79
        self._n_glyphs   = self._glyph_rows * self._glyph_cols   # 1 659

        # Max glyph ID for normalization (NLE constant)
        self._max_glyph = float(nle.nethack.MAX_GLYPH)

        self._obs_dim        = self._n_glyphs + _N_BLSTATS       # 1 686
        self._global_obs_dim = self._obs_dim                     # single agent
        self._action_dim     = int(self._env.action_space.n)

    # ------------------------------------------------------------------
    # SocialEnvWrapper properties
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
    # Core API
    # ------------------------------------------------------------------

    def reset(
            self,
            seed:    Optional[int] = None,
            options: Optional[dict] = None,
    ) -> Tuple[Dict[str, torch.Tensor], dict]:
        if seed is not None:
            self._env.seed(seed)

        # NLE 0.9.x old-gym reset: returns obs dict only
        raw_obs = self._env.reset()
        obs_tensor = self._encode(raw_obs)

        return {"agent_0": obs_tensor}, {}

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
        action = int(actions["agent_0"])

        # NLE 0.9.x old-gym step: 4-tuple (obs, reward, done, info)
        raw_obs, reward, done, info = self._env.step(action)

        obs_tensor = self._encode(raw_obs)
        reward_f   = float(reward)

        # ── Terminated vs truncated ────────────────────────────────────────
        # NLE 0.9.x (gym 0.23) reports end_status as a StepStatus enum:
        #   ABORTED (-1) : step limit hit   → truncated (bootstrap V(s'))
        #   DEATH   ( 1) : player died      → terminated
        #   RUNNING ( 0) : still going      → both False
        # is_ascended=True is also a genuine termination (win condition).
        end_status    = int(info.get("end_status", 0))
        is_ascended   = bool(info.get("is_ascended", False))
        is_truncated  = done and (end_status == -1)
        is_terminated = done and (end_status == 1 or is_ascended)

        terminated = {"agent_0": is_terminated}
        truncated  = {"agent_0": is_truncated}

        # ── Reward decomposition ───────────────────────────────────────────
        # NLE's reward is the score delta per step — naturally sparse.
        # We expose it as both dense reward and sparse_reward so the
        # CoffeeShopMediator treats positive score events as high-priority memories.
        sparse_reward = max(0.0, reward_f)   # only positive score events

        step_infos = {
            "sparse_rewards": {"agent_0": sparse_reward},
            "has_delivery":   sparse_reward > 0.0,
            "nle_info":       info,
        }

        return (
            {"agent_0": obs_tensor},
            {"agent_0": reward_f},
            terminated,
            truncated,
            step_infos,
        )

    def get_global_obs(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Single-agent: global obs is just the agent's own obs.
        return obs_dict["agent_0"]

    def render(self) -> None:
        # NLE renders to a terminal; no-op here to avoid corrupting output.
        pass

    def close(self) -> None:
        if hasattr(self, "_env") and self._env is not None:
            self._env.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, raw_obs: dict) -> torch.Tensor:
        """
        Flatten and normalise an NLE observation dict into a 1-D float32
        tensor of shape (obs_dim,).

        Components
        ----------
        glyphs  : (21, 79) int16  → flattened, divided by MAX_GLYPH → float32
        blstats : (27,)    int64  → divided per-field by _BLSTAT_SCALE,
                                    clipped to [-5, 5] → float32
        """
        glyphs = np.asarray(raw_obs["glyphs"], dtype=np.float32).reshape(-1)
        glyphs /= self._max_glyph

        blstats = np.asarray(raw_obs["blstats"], dtype=np.float32)
        blstats = np.clip(blstats / _BLSTAT_SCALE, -5.0, 5.0)

        combined = np.concatenate([glyphs, blstats], axis=0)
        return torch.from_numpy(combined)
