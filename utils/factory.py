from __future__ import annotations
import argparse
import os
import torch
from typing import Any, Dict, Optional, TYPE_CHECKING, Union
from omegaconf import DictConfig
from envs import SocialEnvWrapper

if TYPE_CHECKING:
    from agents.ppo import PPOAgent as SocialActor
    from core_marl.mediator import CoffeeShopMediator

def make_env(name: str, params: Dict[str, Any], render_mode: Optional[str] = None) -> SocialEnvWrapper:
    """Factory function to instantiate the correct environment wrapper."""
    name = name.lower()
    # Convert DictConfig to dict if necessary
    params = dict(params)
    params["render_mode"] = render_mode
    if name == "overcooked":
        from envs.overcooked.wrapper import OvercookedSocialWrapper
        return OvercookedSocialWrapper(**params)
    if name == "crafter":
        from envs.crafter.wrapper import CrafterWrapper
        return CrafterWrapper(**params)
    if name == "nethack":
        from envs.nethack.wrapper import NLESocialWrapper
        return NLESocialWrapper(**params)
    if name == "aisaac":
        from envs.aisaac.wrapper import AIsaacWrapper
        return AIsaacWrapper(**params)
    raise ValueError(f"Unknown env name: {name!r}")

def _env_idx(agent_id: str) -> int:
    """Extract environment index from a global agent ID (e.g., 'env0_agent_1' -> 0)."""
    return int(agent_id.split("_")[0][3:])

def make_actors(runner: "VectorSocialRunner", mediator: CoffeeShopMediator, cfg: DictConfig, distributed: bool = False) -> Dict[str, SocialActor]:
    """Factory to create SocialActor (PPOAgent) instances."""
    from agents.ppo import PPOAgent as SocialActor
    from torch.nn.parallel import DistributedDataParallel as DDP
    a_cfg = cfg.agent
    img_shape = (a_cfg.img_c, a_cfg.img_h, a_cfg.img_w) if a_cfg.encoder == "cnn" else None
    
    actors = {}
    for aid in runner.agent_ids:
        actor = SocialActor(
            agent_id=aid, obs_space=runner.obs_dim, act_space=runner.action_dim,
            global_obs_dim=runner.global_obs_dim, mediator=mediator,
            gamma=a_cfg.gamma, lam=a_cfg.lam, clip_eps=a_cfg.clip_eps,
            c_vf=a_cfg.c_vf, c_ent=a_cfg.c_ent, ppo_epochs=a_cfg.ppo_epochs,
            mini_batch_size=a_cfg.mini_batch_size, lr=a_cfg.lr, push_every=cfg.trainer.push_every,
            hidden=a_cfg.hidden, encoder=a_cfg.encoder, img_shape=img_shape, device=cfg.trainer.device,
            vanilla=bool(a_cfg.get("vanilla", False)),
        )
        if distributed:
            actor.ac = DDP(actor.ac, device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None)
        actors[aid] = actor
    return actors

class VectorSocialRunner:
    """Manages parallel SocialEnvWrapper instances."""
    def __init__(self, make_env_fn, num_envs: int, env_name: str, env_params: Dict[str, Any], render_mode: str | None):
        self.num_envs = num_envs
        self.envs = [make_env_fn(env_name, dict(env_params), render_mode=render_mode) for _ in range(num_envs)]
        # Add rank-based seed offset to sub-environments
        for i, env in enumerate(self.envs):
            rank = int(os.environ.get("RANK", 0))
            env.reset(seed=rank * 1000 + i)
        self.agent_ids = [f"env{e}_agent_{i}" for e in range(num_envs) for i in range(len(self.envs[e].agent_ids))]
        self.obs_dim, self.action_dim, self.global_obs_dim = self.envs[0].obs_dim, self.envs[0].action_dim, self.envs[0].global_obs_dim

    def reset(self):
        obs, infos_all = {}, {}
        for e, env in enumerate(self.envs):
            o, info = env.reset()
            # Cache the agent ID list
            e_aids = env.agent_ids
            for i, local_aid in enumerate(e_aids): obs[f"env{e}_agent_{i}"] = o[local_aid]
            infos_all[e] = info
        return obs, infos_all

    def reset_env(self, env_idx: int): return self.envs[env_idx].reset()

    def step(self, actions: Dict[str, int]):
        next_obs, rewards, terminated, truncated, infos_all = {}, {}, {}, {}, {}
        for e, env in enumerate(self.envs):
            # Pre-indexed list of agent IDs is faster than enumerate(env.agent_ids)
            e_aids = env.agent_ids
            env_actions = {local_aid: actions[f"env{e}_agent_{i}"] for i, local_aid in enumerate(e_aids)}
            nobs, r, term, trunc, info = env.step(env_actions)
            for i, local_aid in enumerate(e_aids):
                gaid = f"env{e}_agent_{i}"
                next_obs[gaid], rewards[gaid], terminated[gaid], truncated[gaid] = nobs[local_aid], r[local_aid], term[local_aid], trunc[local_aid]
            infos_all[e] = info
        return next_obs, rewards, terminated, truncated, infos_all

    def get_global_obs(self, obs_dict: Dict[str, torch.Tensor], env_idx: int) -> torch.Tensor:
        env = self.envs[env_idx]
        # Avoid list comprehension and intermediate dict if possible, but Wrapper expects dict
        local_obs = {aid: obs_dict[f"env{env_idx}_agent_{i}"] for i, aid in enumerate(env.agent_ids)}
        return env.get_global_obs(local_obs)

    def close(self):
        for env in self.envs: env.close()
