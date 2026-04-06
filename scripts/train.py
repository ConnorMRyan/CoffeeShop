"""
scripts/train.py  —  CoffeeShop MARL training entry-point
Fixed with --render support and proper Gymnasium mode handling.
"""

from __future__ import annotations
import argparse
from collections import defaultdict
from typing import Any, Dict, Tuple
import numpy as np
import torch
import time
import logging

from utils import get_logger, Metrics
from utils.checkpointing import Checkpointer
from core_marl.mediator import CoffeeShopMediator
from agents.ppo import PPOAgent as SocialActor
from envs import SocialEnvWrapper

logging.getLogger('overcooked_ai_py.planning.planners').setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_env(name: str, params: Dict[str, Any], render_mode: str | None = None) -> SocialEnvWrapper:
    """Instantiate the correct SocialEnvWrapper subclass by name."""
    name = name.lower()
    # Add render_mode to params
    params["render_mode"] = render_mode

    if name == "overcooked":
        from envs.overcooked.wrapper import OvercookedSocialWrapper
        return OvercookedSocialWrapper(**params)
    if name == "crafter":
        from envs.crafter.wrapper import CrafterWrapper
        return CrafterWrapper(**params)
    if name == "nethack":
        from envs.nethack.wrapper import NetHackWrapper
        return NetHackWrapper(**params)
    if name == "aisaac":
        from envs.aisaac.wrapper import AIsaacWrapper
        return AIsaacWrapper(**params)
    raise ValueError(f"Unknown env name: {name!r}")

def make_actors(env: SocialEnvWrapper, mediator: CoffeeShopMediator, args: argparse.Namespace) -> Dict[str, SocialActor]:
    return {
        aid: SocialActor(
            agent_id        = aid,
            obs_dim         = env.obs_dim,
            action_dim      = env.action_dim,
            global_obs_dim  = env.global_obs_dim,
            mediator        = mediator,
            gamma           = args.gamma,
            lam             = args.lam,
            clip_eps        = args.clip_eps,
            c_vf            = args.c_vf,
            c_ent           = args.c_ent,
            ppo_epochs      = args.ppo_epochs,
            mini_batch_size = args.mini_batch_size,
            lr              = args.lr,
            push_every      = args.push_every,
            device          = args.device,
        )
        for aid in env.agent_ids
    }

class VectorSocialRunner:
    """
    Lightweight vectorized manager for N SocialEnvWrapper instances.
    Steps each env sequentially but allows easy swap to gymnasium.vector later.
    """
    def __init__(self, make_env_fn, num_envs: int, env_name: str, env_params: Dict[str, Any], render_mode: str | None):
        self.num_envs = num_envs
        self.envs = [make_env_fn(env_name, dict(env_params), render_mode=render_mode) for _ in range(num_envs)]
        # Build global agent ids: env{e}_agent_0/1
        self.agent_ids = [f"env{e}_agent_{i}" for e in range(num_envs) for i in range(len(self.envs[e].agent_ids))]
        # Mapping from global aid to (env_idx, local_aid)
        self._aid_map = {}
        for e, env in enumerate(self.envs):
            for i, local_aid in enumerate(env.agent_ids):
                self._aid_map[f"env{e}_agent_{i}"] = (e, local_aid)

        # Assume homogeneous dims
        self.obs_dim = self.envs[0].obs_dim
        self.action_dim = self.envs[0].action_dim
        self.global_obs_dim = self.envs[0].global_obs_dim

    def reset(self):
        obs = {}
        infos_all = {}
        for e, env in enumerate(self.envs):
            o, info = env.reset()
            for i, local_aid in enumerate(env.agent_ids):
                obs[f"env{e}_agent_{i}"] = o[local_aid]
            infos_all[e] = info
        return obs, infos_all

    def reset_env(self, env_idx: int):
        o, info = self.envs[env_idx].reset()
        return o, info

    def step(self, actions: Dict[str, int]):
        next_obs = {}
        rewards = {}
        terminated = {}
        truncated = {}
        infos_all = { }
        for e, env in enumerate(self.envs):
            # Extract actions for this env's agents
            env_actions = {}
            for i, local_aid in enumerate(env.agent_ids):
                gaid = f"env{e}_agent_{i}"
                env_actions[local_aid] = actions[gaid]
            nobs, r, term, trunc, info = env.step(env_actions)
            for i, local_aid in enumerate(env.agent_ids):
                gaid = f"env{e}_agent_{i}"
                next_obs[gaid] = nobs[local_aid]
                rewards[gaid] = r[local_aid]
                terminated[gaid] = term[local_aid]
                truncated[gaid] = trunc[local_aid]
            infos_all[e] = info
        return next_obs, rewards, terminated, truncated, infos_all

    def get_global_obs(self, obs_dict: Dict[str, torch.Tensor], env_idx: int) -> torch.Tensor:
        # Delegate to env's own get_global_obs using local ids
        env = self.envs[env_idx]
        local_obs = {aid: obs_dict[f"env{env_idx}_agent_{i}"] for i, aid in enumerate(env.agent_ids)}
        return env.get_global_obs(local_obs)

    def render(self):
        for env in self.envs:
            env.render()

    def close(self):
        for env in self.envs:
            env.close()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CoffeeShop MARL trainer")

    p.add_argument("--env",    default="overcooked", choices=["overcooked", "crafter", "nethack", "aisaac"])
    p.add_argument("--layout", default="cramped_room", help="Layout name (Overcooked only)")
    p.add_argument("--render", action="store_true", help="Enable GUI visualization (Slows down training)")
    p.add_argument("--num_envs", type=int, default=2, help="Number of parallel environments (Overcooked instances)")

    p.add_argument("--steps",                  type=int,   default=2_000_000, dest="total_steps")
    p.add_argument("--update_interval",        type=int,   default=512)
    p.add_argument("--critic_update_interval", type=int,   default=256)
    p.add_argument("--log_interval",           type=int,   default=100)

    p.add_argument("--push_every",      type=int,   default=128)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--lam",             type=float, default=0.95)
    p.add_argument("--clip_eps",        type=float, default=0.2)
    p.add_argument("--c_vf",            type=float, default=0.5)
    p.add_argument("--c_ent",           type=float, default=0.01)
    p.add_argument("--ppo_epochs",      type=int,   default=4)
    p.add_argument("--mini_batch_size", type=int,   default=64)
    p.add_argument("--lr",              type=float, default=3e-4)

    p.add_argument("--synergy_alpha", type=float, default=0.3)
    p.add_argument("--epsilon_td",    type=float, default=0.05)

    p.add_argument("--device", default="cpu")
    p.add_argument("--seed",   type=int, default=42)

    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log = get_logger()

    # Initialize checkpointer with absolute path for guaranteed checkpointing
    checkpointer = Checkpointer(dirpath="/home/mynam/CoffeeShop/checkpoints")

    # ── Environments (vectorized) ─────────────────────────────────────────
    env_params: Dict[str, Any] = (
        {"layout_name": args.layout} if args.env == "overcooked" else {}
    )
    rmode = "human" if args.render else None
    runner = VectorSocialRunner(make_env, args.num_envs, args.env, env_params, render_mode=rmode)

    # ── Mediator (singleton) ──────────────────────────────────────────────
    mediator = CoffeeShopMediator(
        global_obs_dim  = runner.global_obs_dim,
        gamma           = args.gamma,
        epsilon_td      = args.epsilon_td,
        synergy_alpha   = args.synergy_alpha,
        device          = args.device,
    )

    # ── Actors (2 per env) ────────────────────────────────────────────────
    actors: Dict[str, SocialActor] = {
        aid: SocialActor(
            agent_id        = aid,
            obs_dim         = runner.obs_dim,
            action_dim      = runner.action_dim,
            global_obs_dim  = runner.global_obs_dim,
            mediator        = mediator,
            gamma           = args.gamma,
            lam             = args.lam,
            clip_eps        = args.clip_eps,
            c_vf            = args.c_vf,
            c_ent           = args.c_ent,
            ppo_epochs      = args.ppo_epochs,
            mini_batch_size = args.mini_batch_size,
            lr              = args.lr,
            push_every      = args.push_every,
            device          = args.device,
        ) for aid in runner.agent_ids
    }

    metrics     = Metrics()
    ep_rewards  = defaultdict(float)
    ppo_metrics = defaultdict(list)

    obs_dict, infos_all = runner.reset()

    log.info({
        "event":       "training_start",
        "env":         args.env,
        "render":      args.render,
        "num_envs":    args.num_envs,
        "total_steps": args.total_steps,
        "agents":      list(actors.keys()),
    })

    # ── Training loop ─────────────────────────────────────────────────────
    for t in range(args.total_steps):
        # Compute global obs per env
        env_global_obs: Dict[int, torch.Tensor] = {
            e: runner.get_global_obs(obs_dict, e) for e in range(runner.num_envs)
        }

        # Select actions for every agent
        actions: Dict[str, int] = {}
        for aid, actor in actors.items():
            with torch.no_grad():
                action = actor.ac.act(obs_dict[aid].to(args.device))[0]
                actions[aid] = action.item()

        next_obs, rewards, terminated, truncated, infos_all = runner.step(actions)

        if args.render:
            runner.render()
            time.sleep(0.005)

        # Build sparse rewards per global agent id and compute per-env soup deliveries
        sparse_rewards: Dict[str, float] = {}
        soups_counts: list[float] = []
        for e in range(runner.num_envs):
            info = infos_all.get(e, {})
            sr_local = info.get("sparse_rewards", {})
            env_soups = 0.0
            # Map local -> global ids
            if e < len(runner.envs):
                env = runner.envs[e]
                for i, local_aid in enumerate(env.agent_ids):
                    gaid = f"env{e}_agent_{i}"
                    val = float(sr_local.get(local_aid, 0.0))
                    sparse_rewards[gaid] = val
                    env_soups += max(0.0, val)
            soups_counts.append(env_soups)
        # Log mean soup delivery count across envs as cross-play proxy
        if soups_counts:
            metrics.update({"eval/cross_play_score": float(np.mean(soups_counts))})

        # Push to each actor
        for aid, actor in actors.items():
            # Parse env index from global id: env{e}_agent_{i}
            try:
                env_idx = int(aid.split("_")[0][3:])
            except Exception:
                env_idx = 0
            done = bool(terminated.get(aid, False) or truncated.get(aid, False))
            actor.step(
                env_id        = env_idx,
                obs           = obs_dict[aid],
                global_obs    = env_global_obs[env_idx],
                reward        = float(rewards[aid]),
                sparse_reward = float(sparse_rewards.get(aid, 0.0)),
                done          = done,
            )
            ep_rewards[aid] += float(rewards[aid])

        metrics.update({"reward": float(sum(rewards.values()))})
        obs_dict = next_obs

        # Handle per-env episode ends and reset
        for e in range(runner.num_envs):
            num_agents_e = len(runner.envs[e].agent_ids)
            # If any agent in env e is done, treat as episode end for that env
            env_done = any(terminated.get(f"env{e}_agent_{i}", False) or truncated.get(f"env{e}_agent_{i}", False)
                           for i in range(num_agents_e))
            if env_done:
                for i in range(num_agents_e):
                    gaid = f"env{e}_agent_{i}"
                    ppo_metrics[f"ep_return/{gaid}"].append(ep_rewards[gaid])
                    ep_rewards[gaid] = 0.0
                # Reset only this env and update obs_dict entries
                o, info = runner.reset_env(e)
                for i, local_aid in enumerate(runner.envs[e].agent_ids):
                    obs_dict[f"env{e}_agent_{i}"] = o[local_aid]

        if (t + 1) % args.update_interval == 0:
            for aid, actor in actors.items():
                m = actor.update()
                for k, v in m.items():
                    ppo_metrics[f"{k}/{aid}"].append(v)

        if (t + 1) % args.critic_update_interval == 0:
            buffer_list = list(mediator._buffer)
            if len(buffer_list) >= 64:
                recent = [sm.transition for sm in buffer_list[-256:]]
                critic_loss = mediator.update_critic(recent)
                ppo_metrics["mediator/critic_loss"].append(critic_loss)

        if (t + 1) % args.log_interval == 0:
            log.info({"step": t + 1, **metrics.mean()})
            metrics.clear()
            for k, v in ppo_metrics.items():
                if v:
                    log.info({"step": t + 1, k: float(np.mean(v))})
            for aid, actor in actors.items():
                log.info({"step": t + 1, f"omega/{aid}": actor.openness.value})
            ppo_metrics.clear()

        # --- Guaranteed checkpointing every 10k steps ---
        global_step = t + 1
        if global_step % 10000 == 0:
            # Map the first four global agents deterministically to agent_0..agent_3
            mapping = [
                "env0_agent_0",
                "env0_agent_1",
                "env1_agent_0",
                "env1_agent_1",
            ]
            state = { }
            for i, gaid in enumerate(mapping):
                key = f"agent_{i}"
                if gaid in actors:
                    # Save the per-agent Actor-Critic network weights
                    state[key] = actors[gaid].ac.state_dict()
            # Always include mediator critic and current step
            state.update({
                "mediator": mediator.critic.state_dict(),
                "step": global_step,
            })
            path = checkpointer.save(state, filename=f"checkpoint_{global_step}.pt")
            print(f"--- CHECKPOINT SAVED TO {path} ---")

    runner.close()
    log.info({"event": "training_complete", "total_steps": args.total_steps})

if __name__ == "__main__":
    main()