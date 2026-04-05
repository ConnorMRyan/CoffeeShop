from __future__ import annotations

import argparse
from typing import Any, Dict

from utils import get_logger, Metrics
from core_marl import Mediator
from envs import SocialEnvWrapper

# Lazy imports to keep optional deps optional

def make_env(name: str, params: Dict[str, Any]) -> SocialEnvWrapper:
    name = name.lower()
    if name == "overcooked":
        from envs.overcooked.wrapper import OvercookedWrapper
        return OvercookedWrapper(**params)
    if name == "crafter":
        from envs.crafter.wrapper import CrafterWrapper
        return CrafterWrapper(**params)
    if name == "nethack":
        from envs.nethack.wrapper import NetHackWrapper
        return NetHackWrapper(**params)
    if name == "aisaac":
        from envs.aisaac.wrapper import AIsaacWrapper
        return AIsaacWrapper(**params)
    raise ValueError(f"Unknown env name: {name}")


def make_agent(name: str, obs_space=None, act_space=None):
    name = name.lower()
    if name == "ppo":
        from agents.ppo import PPOAgent
        return PPOAgent(obs_space, act_space)
    if name == "sac":
        from agents.sac import SACAgent
        return SACAgent(obs_space, act_space)
    raise ValueError(f"Unknown agent name: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="overcooked")
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--agent", default="ppo")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    log = get_logger()

    # Build environment and mediator
    env = make_env(args.env, {"layout_name": args.layout} if args.env == "overcooked" else {})
    mediator = Mediator(env)

    # Build a trivial agent (no-op stub) and metrics
    agent = make_agent(args.agent)
    metrics = Metrics()

    obs, infos = mediator.reset()

    for t in range(args.steps):
        actions = agent.act(obs)
        next_obs, rewards, terminated, truncated, infos = mediator.step(actions)

        # Simple logging/metrics
        metrics.update({"reward": sum(rewards.values())})
        if (t + 1) % 100 == 0:
            log.info({"step": t + 1, **metrics.mean()})
            metrics.clear()

        obs = next_obs
        if any(terminated.values()) or any(truncated.values()):
            obs, infos = mediator.reset()

    mediator.close()
    log.info("Training stub complete.")


if __name__ == "__main__":
    main()
