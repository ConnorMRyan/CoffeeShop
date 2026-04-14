from __future__ import annotations

import sys
import os

import argparse
from typing import Any, Dict, Optional, List

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
        actors: Dict[str, Any],
        episodes: int,
        render: bool,
) -> Dict[str, float]:
    """Runs multiple episodes and averages the results."""
    scores = []
    deliveries = []

    for _ in range(episodes):
        res = run_episode(env, actors, horizon=400, render=render)
        scores.append(res.get("total_team_score", 0.0))
        # Keep backward compatibility for Overcooked metrics
        deliveries.append(res.get("total_deliveries", 0.0))

    return {
        "mean_team_score": float(np.mean(scores)),
        "mean_deliveries": float(np.mean(deliveries))
    }

def _load_population(
        obs_dim: int,
        action_dim: int,
        ckpt: str,
        agent_ids: List[str]
) -> Dict[str, ActorFromCheckpoint]:
    """Loads a full population of actors from a single checkpoint."""
    return {
        aid: ActorFromCheckpoint(obs_dim, action_dim, ckpt, key_hints=(aid,))
        for aid in agent_ids
    }

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

    if not ckpt_a or not ckpt_b:
        raise ValueError("Both --ckpt_a and --ckpt_b are required for cross-play evaluation.")

    # Build Env
    if env_name == "overcooked":
        env_params = {"layout_name": layout}
    elif env_name == "meltingpot":
        env_params = {"scenario": "clean_up"}
    else:
        env_params = {}

    env = make_env(env_name, env_params, render_mode=("human" if render else None))

    obs_dim = env.obs_dim
    action_dim = env.action_dim
    agent_ids = env.agent_ids

    # Load Full Populations
    pop_a = _load_population(obs_dim, action_dim, ckpt_a, agent_ids)
    pop_b = _load_population(obs_dim, action_dim, ckpt_b, agent_ids)

    # ── Protocols ─────────────────────────────────────────────────────────
    results = {}

    # 1. Self-Play (SP)
    sp_a = _mean_over_eps(env, pop_a, episodes, render)
    sp_b = _mean_over_eps(env, pop_b, episodes, render)

    results["SP/A_team_score"] = sp_a["mean_team_score"]
    results["SP/B_team_score"] = sp_b["mean_team_score"]
    results["SP/mean_score"] = float(np.mean([sp_a["mean_team_score"], sp_b["mean_team_score"]]))

    # 2. Cross-Play (XP)
    # We dynamically test Cross-Play by taking Agent 0 from A and everyone else from B, and vice-versa.
    xp_scores = []

    for target_aid in agent_ids:
        # Team 1: Target agent from A, rest from B
        xp_team_1 = {aid: (pop_a[aid] if aid == target_aid else pop_b[aid]) for aid in agent_ids}
        res_1 = _mean_over_eps(env, xp_team_1, episodes, render)
        results[f"XP/A_{target_aid}_B_others"] = res_1["mean_team_score"]
        xp_scores.append(res_1["mean_team_score"])

        # Team 2: Target agent from B, rest from A
        xp_team_2 = {aid: (pop_b[aid] if aid == target_aid else pop_a[aid]) for aid in agent_ids}
        res_2 = _mean_over_eps(env, xp_team_2, episodes, render)
        results[f"XP/B_{target_aid}_A_others"] = res_2["mean_team_score"]
        xp_scores.append(res_2["mean_team_score"])

    results["eval/cross_play_score"] = float(np.mean(xp_scores))

    # 3. Robustness (Random Partner)
    # Target agent from A, rest are random
    rb_scores = []
    for target_aid in agent_ids:
        rb_team = {aid: (pop_a[aid] if aid == target_aid else RandomPartner(action_dim)) for aid in agent_ids}
        res_rb = _mean_over_eps(env, rb_team, episodes, render)
        results[f"RB/A_{target_aid}_Rand_others"] = res_rb["mean_team_score"]
        rb_scores.append(res_rb["mean_team_score"])

    results["RB/mean_score"] = float(np.mean(rb_scores))

    env.close()

    # Log results
    for k, v in results.items():
        log.info(f"{k}: {float(v):.4f}")

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

    evaluate(args.env, args.layout, args.episodes, args.ckpt_a, args.ckpt_b, render=args.render)

if __name__ == "__main__":
    main()