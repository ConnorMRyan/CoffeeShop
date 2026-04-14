"""
coffeeshop/evaluate.py  —  CoffeeShop Cross-Play Evaluation
==========================================================
Robust evaluation with forced Multi-Agent Crafter initialization.
"""

from __future__ import annotations

import sys
import os

import argparse
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from utils import get_logger
from utils.checkpointing import Checkpointer
from utils.factory import make_env
from utils.evaluation import ActorFromCheckpoint, RandomPartner, run_episode
from envs import SocialEnvWrapper

log = get_logger()

# ---------------------------------------------------------------------------
# Episode & Evaluation logic
# ---------------------------------------------------------------------------

def _load_policy_pair(obs_dim: int, action_dim: int, ckpt: str) -> Tuple[ActorFromCheckpoint, ActorFromCheckpoint]:
    left = ActorFromCheckpoint(obs_dim, action_dim, ckpt, key_hints=("env0_agent_0", "env1_agent_0"))
    try:
        right = ActorFromCheckpoint(obs_dim, action_dim, ckpt, key_hints=("env0_agent_1", "env1_agent_1"))
    except KeyError:
        log.info(f"--- Mapping same policy to Seat 1 for {os.path.basename(ckpt)} ---")
        right = ActorFromCheckpoint(obs_dim, action_dim, ckpt, key_hints=("env1_agent_0", "env0_agent_0"))
    return left, right


def _evaluate_pair(env, p_left, p_right, episodes, render) -> Tuple[float, float]:
    actors = {env.agent_ids[0]: p_left, env.agent_ids[1]: p_right}
    results = [run_episode(env, actors, render=render) for _ in range(episodes)]
    a0_id, a1_id = env.agent_ids[0], env.agent_ids[1]
    return (
        float(np.mean([r["agent_rewards"][a0_id] for r in results])),
        float(np.mean([r["agent_rewards"][a1_id] for r in results]))
    )


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="crafter")
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--ckpt_a", type=str, required=True)
    parser.add_argument("--ckpt_b", type=str, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    # Pass num_agents=2 directly into the factory call
    env = make_env(args.env, {"num_agents": 2, "layout_name": args.layout}, render_mode=("human" if args.render else None))
    log.info(f"Evaluation Mode: {args.env.upper()}")
    log.info(f"Agents detected in environment: {env.agent_ids}")

    a_left, a_right = _load_policy_pair(env.obs_dim, env.action_dim, args.ckpt_a)
    b_left, b_right = _load_policy_pair(env.obs_dim, env.action_dim, args.ckpt_b)
    rand_pol = RandomPartner(env.action_dim)

    log.info("Running SP/A (Self-Play Baseline)...")
    _, sp_a = _evaluate_pair(env, a_left, a_right, args.episodes, args.render)

    log.info("Running SP/B (Self-Play Mature)...")
    _, sp_b = _evaluate_pair(env, b_left, b_right, args.episodes, args.render)

    log.info("Running XP (Cross-Play Coordination)...")
    _, xp_a0_b1 = _evaluate_pair(env, a_left, b_right, args.episodes, args.render)
    _, xp_b0_a1 = _evaluate_pair(env, b_left, a_right, args.episodes, args.render)

    log.info("Running RB (Robustness vs Random)...")
    _, rb_b = _evaluate_pair(env, b_left, rand_pol, args.episodes, args.render)

    log.info("-" * 40)
    log.info(f"RESULTS FOR {args.env.upper()}")
    log.info(f"SP/A Score:     {sp_a:.4f}")
    log.info(f"SP/B Score:     {sp_b:.4f}")
    log.info(f"XP Mean Score:  {(xp_a0_b1 + xp_b0_a1) / 2:.4f}")
    log.info(f"RB/B vs Rand:   {rb_b:.4f}")
    log.info("-" * 40)

    env.close()

if __name__ == "__main__":
    evaluate()