from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from utils import get_logger
from utils.checkpointing import Checkpointer
from utils.factory import make_env
from utils.evaluation import RandomPartner, ActorFromCheckpoint, run_episode
from envs import SocialEnvWrapper

log = get_logger()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _mean_over_eps(
        env: SocialEnvWrapper,
        policy_left,
        policy_right,
        episodes: int,
        render: bool,
) -> float:
    soups = [
        run_episode(env, policy_left, policy_right, horizon=400, render=render)[0]
        for _ in range(episodes)
    ]
    return float(np.mean(soups))


def _load_policy_pair(
        obs_dim: int,
        action_dim: int,
        ckpt: str,
) -> Tuple[ActorFromCheckpoint, ActorFromCheckpoint]:
    """Load the canonical seat-0 and seat-1 policies from one checkpoint."""
    left = ActorFromCheckpoint(
        obs_dim, action_dim, ckpt,
        key_hints=("env0_agent_0",),
    )
    right = ActorFromCheckpoint(
        obs_dim, action_dim, ckpt,
        key_hints=("env0_agent_1",),
    )
    return left, right


# ---------------------------------------------------------------------------
# Evaluation orchestrator
# ---------------------------------------------------------------------------

def evaluate(
        env_name: str,
        layout: str,
        episodes: int,
        ckpt_a: Optional[str],
        ckpt_b: Optional[str],
        render: bool = False,
) -> Dict[str, float]:
    if not ckpt_a:
        raise ValueError("--ckpt_a is required for meaningful evaluation.")
    if not ckpt_b:
        raise ValueError(
            "--ckpt_b is required for cross-play evaluation. "
            "Do not reuse ckpt_a implicitly; that collapses XP into self-play-like behavior."
        )

    if env_name == "overcooked":
        env_params = {"layout_name": layout}
    elif env_name == "meltingpot":
        env_params = {"scenario": "clean_up"}
    else:
        env_params = {}
    env = make_env(env_name, env_params, render_mode=("human" if render else None))

    obs_dim = env.obs_dim
    action_dim = env.action_dim

    # Canonical seat-specialized policies from each run.
    a_left, a_right = _load_policy_pair(obs_dim, action_dim, ckpt_a)
    b_left, b_right = _load_policy_pair(obs_dim, action_dim, ckpt_b)
    rand_pol = RandomPartner(action_dim)

    # ── Protocols ─────────────────────────────────────────────────────────
    # Self-play: canonical within-checkpoint pairing.
    sp_a = _mean_over_eps(env, a_left, a_right, episodes, render)
    sp_b = _mean_over_eps(env, b_left, b_right, episodes, render)
    sp_mean = float(np.mean([sp_a, sp_b]))

    # Cross-play: evaluate both cross-seat combinations.
    # These are the important zero-shot coordination tests across runs.
    xp_a0_b1 = _mean_over_eps(env, a_left, b_right, episodes, render)
    xp_b0_a1 = _mean_over_eps(env, b_left, a_right, episodes, render)
    xp_mean = float(np.mean([xp_a0_b1, xp_b0_a1]))

    # Optional same-slot stress tests. These can diagnose slot overfitting,
    # but they are not the primary XP metric.
    xp_same_left = _mean_over_eps(env, a_left, b_left, episodes, render)
    xp_same_right = _mean_over_eps(env, a_right, b_right, episodes, render)

    # Robustness: agent from A with a random teammate in the opposite seat.
    rb_a_left = _mean_over_eps(env, a_left, rand_pol, episodes, render)
    rb_rand_a_right = _mean_over_eps(env, rand_pol, a_right, episodes, render)
    rb_mean = float(np.mean([rb_a_left, rb_rand_a_right]))

    env.close()

    results = {
        "SP/A": sp_a,
        "SP/B": sp_b,
        "SP/mean_soups_per_400": sp_mean,
        "XP/A0_B1": xp_a0_b1,
        "XP/B0_A1": xp_b0_a1,
        "XP/mean_soups_per_400": xp_mean,
        "XP/same_slot_left": xp_same_left,
        "XP/same_slot_right": xp_same_right,
        "RB/A0_rand1": rb_a_left,
        "RB/rand0_A1": rb_rand_a_right,
        "RB/mean_soups_per_400": rb_mean,
        "eval/cross_play_score": xp_mean,
    }

    for k, v in results.items():
        log.info({k: float(v)})

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CoffeeShop Cross-Play Evaluation")
    parser.add_argument("--env", default="overcooked",
                        choices=["overcooked", "crafter", "nethack", "aisaac", "meltingpot"])
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--mp_scenario", default="clean_up",
                        help="MeltingPot scenario (when --env meltingpot)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--ckpt_a", type=str, required=True,
                        help="Path to checkpoint A")
    parser.add_argument("--ckpt_b", type=str, required=True,
                        help="Path to checkpoint B")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    # Build env with appropriate params
    params = {"layout_name": args.layout} if args.env == "overcooked" else ({"scenario": args.mp_scenario} if args.env == "meltingpot" else {})
    env = make_env(args.env, params, render_mode=("human" if args.render else None))

    # Run evaluate using the constructed env settings
    evaluate(args.env, args.layout, args.episodes, args.ckpt_a, args.ckpt_b, render=args.render)


if __name__ == "__main__":
    main()
