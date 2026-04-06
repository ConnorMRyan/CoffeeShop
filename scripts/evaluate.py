from __future__ import annotations

import argparse
from typing import Any, Dict

from utils import get_logger
from envs import SocialEnvWrapper


def make_env(name: str, params: Dict[str, Any]) -> SocialEnvWrapper:
    name = name.lower()
    if name == "overcooked":
        from envs.overcooked.wrapper import OvercookedSocialWrapper
        return OvercookedSocialWrapper(**params)
    if name == "crafter":
        from envs.crafter.wrapper import CrafterWrapper  # may be unavailable
        return CrafterWrapper(**params)
    if name == "nethack":
        from envs.nethack.wrapper import NetHackWrapper  # may be unavailable
        return NetHackWrapper(**params)
    if name == "aisaac":
        from envs.aisaac.wrapper import AIsaacWrapper
        return AIsaacWrapper(**params)
    raise ValueError(f"Unknown env name: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="overcooked")
    parser.add_argument("--layout", default="cramped_room")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    log = get_logger("CoffeeShop.Eval")

    env = make_env(args.env, {"layout_name": args.layout} if args.env == "overcooked" else {})

    for ep in range(args.episodes):
        obs, infos = env.reset()
        ep_return = 0.0
        steps = 0
        done = False
        while not done and steps < 1000:
            # Random/no-op placeholder policy: do nothing
            actions = {aid: "stay" for aid in env.agent_ids}
            obs, rewards, terminated, truncated, infos = env.step(actions)
            ep_return += sum(rewards.values())
            steps += 1
            done = any(terminated.values()) or any(truncated.values())
        log.info({"episode": ep + 1, "return": ep_return, "steps": steps})

    env.close()
    log.info("Evaluation stub complete.")


if __name__ == "__main__":
    main()
