"""
scripts/train.py  —  CoffeeShop MARL training entry-point
==========================================================

Fixes included
--------------
11. Robust Checkpoint Loading:
    Wraps Mediator loading in a try-except. If there is a size mismatch
    (e.g., loading an MLP checkpoint into a CNN model), it allows
    Actors to load while the Mediator starts fresh.

12. Critic-Quality Feedback Loop:
    The mediator's critic_loss is fed back into each SocialActor.
    This enables the "Verifiable Trust" logic in ppo.py.

13. Clean Structured Logging:
    Logs per-agent metrics on individual lines with ω, Critic Loss,
    and Quality (Q) visible at high precision.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import logging
from collections import defaultdict
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import OmegaConf

from utils import get_logger, Metrics, TBWriter, WandbWriter
from utils.checkpointing import Checkpointer
from utils.factory import make_env, _env_idx, make_actors, VectorSocialRunner
from utils.metrics import measure_population_diversity
from core_marl.mediator import CoffeeShopMediator
from agents.ppo import PPOAgent as SocialActor
from envs import SocialEnvWrapper

logging.getLogger("overcooked_ai_py.planning.planners").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------

def shuffle_partners(runner: VectorSocialRunner, actors: Dict[str, SocialActor], log, step: int) -> None:
    num_envs, num_local_agents = runner.num_envs, len(runner.envs[0].agent_ids)
    for i in range(num_local_agents):
        slot = [f"env{e}_agent_{i}" for e in range(num_envs)]
        rotated = slot[-1:] + slot[:-1]
        tmp = {new_key: actors[old_key] for new_key, old_key in zip(slot, rotated)}
        actors.update(tmp)
        for new_key in slot: actors[new_key].agent_id = new_key
    log.info({"event": "partner_shuffle", "step": step})

def load_config() -> Any:
    """Load and merge hierarchical configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="overcooked")
    parser.add_argument("--agent", type=str, default="ppo")
    parser.add_argument("--mediator", type=str, default="default")
    parser.add_argument("--trainer", type=str, default="default")
    parser.add_argument("overrides", nargs="*", help="Any config overrides (e.g. trainer.device=cuda)")
    args = parser.parse_args()

    # Base configs
    cfg = OmegaConf.create({
        "env": OmegaConf.load(f"configs/env/{args.env}.yaml"),
        "agent": OmegaConf.load(f"configs/agent/{args.agent}.yaml"),
        "mediator": OmegaConf.load(f"configs/mediator/{args.mediator}.yaml"),
        "trainer": OmegaConf.load(f"configs/trainer/{args.trainer}.yaml"),
    })

    # CLI Overrides
    if args.overrides:
        cli_cfg = OmegaConf.from_dotlist(args.overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg

def main() -> None:
    cfg = load_config()
    t_cfg, e_cfg, a_cfg, m_cfg = cfg.trainer, cfg.env, cfg.agent, cfg.mediator

    torch.manual_seed(t_cfg.seed); np.random.seed(t_cfg.seed); log = get_logger()
    run_id = t_cfg.run_id or time.strftime("%Y%m%d_%H%M%S")
    checkpointer = Checkpointer(dirpath=os.path.abspath(t_cfg.checkpoint_dir), run_id=run_id)
    log.info({"event": "run_start", "run_id": run_id, "save_dir": str(checkpointer.run_dir)})

    tb = TBWriter(log_dir=str(checkpointer.run_dir / "tb"))
    wb = WandbWriter(
        config=OmegaConf.to_container(cfg, resolve=True),
        project=t_cfg.tracking.project,
        entity=t_cfg.tracking.entity,
        run_id=run_id,
        use_wandb=t_cfg.tracking.use_wandb
    )

    runner = VectorSocialRunner(make_env, t_cfg.num_envs, e_cfg.name, e_cfg.params, "human" if t_cfg.render else None)
    
    img_shape = ((m_cfg.img_c, m_cfg.img_h, m_cfg.img_w) if m_cfg.encoder == "cnn" else None)
    mediator = CoffeeShopMediator(
        global_obs_dim=runner.global_obs_dim,
        buffer_capacity=m_cfg.buffer_capacity,
        gamma=m_cfg.gamma,
        epsilon_td=m_cfg.epsilon_td,
        synergy_alpha=m_cfg.synergy_alpha,
        critic_lr=m_cfg.critic_lr,
        critic_hidden=m_cfg.critic_hidden,
        encoder=m_cfg.encoder,
        img_shape=img_shape,
        device=t_cfg.device
    )
    actors = make_actors(runner, mediator, cfg)

    # --- FIX 11: Robust Loading ---
    if t_cfg.checkpoint_path and os.path.isfile(t_cfg.checkpoint_path):
        ckpt = torch.load(t_cfg.checkpoint_path, map_location=t_cfg.device)
        for aid, actor in actors.items():
            if aid in ckpt: actor.ac.load_state_dict(ckpt[aid])
        if "mediator" in ckpt:
            try:
                mediator.critic.load_state_dict(ckpt["mediator"])
            except RuntimeError as e:
                log.info(f"Mediator state mismatch: {e}. Starting fresh.")

    metrics, ep_rewards, ppo_metrics = Metrics(), defaultdict(float), defaultdict(list)
    obs_dict, _ = runner.reset()

    for t in range(t_cfg.total_steps):
        env_global_obs = {e: runner.get_global_obs(obs_dict, e) for e in range(runner.num_envs)}
        actions = {aid: actor.act(_env_idx(aid), obs_dict[aid], env_global_obs[_env_idx(aid)]).item() for aid, actor in actors.items()}
        n_obs, rew, term, trunc, infos = runner.step(actions)
        n_env_global = {e: runner.get_global_obs(n_obs, e) for e in range(runner.num_envs)}

        for aid, actor in actors.items():
            e = _env_idx(aid)
            sr = float(infos.get(e, {}).get("sparse_rewards", {}).get(aid.split('_')[-1], 0.0))
            actor.observe_outcome(n_env_global[e], float(rew[aid]), sr, bool(term[aid]), bool(trunc[aid]))
            ep_rewards[aid] += float(rew[aid])
        metrics.update({"reward": float(sum(rew.values()))}); obs_dict = n_obs

        for e in range(runner.num_envs):
            if any(term.get(f"env{e}_agent_{i}", False) or trunc.get(f"env{e}_agent_{i}", False) for i in range(len(runner.envs[e].agent_ids))):
                for i in range(len(runner.envs[e].agent_ids)):
                    gaid = f"env{e}_agent_{i}"
                    ppo_metrics[f"ep_return/{gaid}"].append(ep_rewards[gaid]); ep_rewards[gaid] = 0.0
                o, _ = runner.reset_env(e)
                for i, l_aid in enumerate(runner.envs[e].agent_ids): obs_dict[f"env{e}_agent_{i}"] = o[l_aid]

        if (t + 1) % t_cfg.update_interval == 0:
            for aid, actor in actors.items():
                m = actor.update(last_obs=obs_dict[aid].to(t_cfg.device))
                for k, v in m.items(): ppo_metrics[f"{k}/{aid}"].append(v)

        if (t + 1) % t_cfg.critic_update_interval == 0 and len(mediator._buffer) >= 64:
            c_loss = float(mediator.update_critic(mediator._buffer.sample(256)))
            ppo_metrics["mediator/critic_loss"].append(c_loss)
            for actor in actors.values(): actor._last_critic_loss = c_loss

        if t_cfg.shuffle_interval > 0 and (t + 1) % t_cfg.shuffle_interval == 0: shuffle_partners(runner, actors, log, t + 1)

        if (t + 1) % t_cfg.log_interval == 0:
            step = t + 1
            m_avg = metrics.mean()
            log.info(f"[STEP {step}] Global | Reward: {m_avg.get('reward', 0.0):.4f}")
            tb.add_scalars(m_avg, step)
            wb.log(m_avg, step)
            metrics.clear()
            
            p_avg = {k: float(np.mean(v)) for k, v in ppo_metrics.items() if v}
            for aid in actors.keys():
                a_m = {k.split('/')[0]: v for k, v in p_avg.items() if k.endswith(aid)}
                om, c_l = actors[aid].openness.value, float(getattr(actors[aid], "_last_critic_loss", 1.0))
                q = max(0.0, 1.0 - c_l / 2.0)
                m_s = " | ".join([f"{k}: {v:.4f}" for k, v in a_m.items()])
                log.info(f"[STEP {step}] {aid} | ω: {om:.4f} | loss: {c_l:.4f} | Q: {q:.4f} | {m_s}")
            if "mediator/critic_loss" in p_avg: log.info(f"[STEP {step}] Mediator | Critic Loss: {p_avg['mediator/critic_loss']:.4f}")
            
            # --- Population Diversity Logging ---
            div = measure_population_diversity(actors, mediator, t_cfg.device)
            log.info(f"[STEP {step}] Population | Diversity (JS): {div:.4f}")
            tb.add_scalar("population/diversity", div, step)
            wb.log({"population/diversity": div}, step)

            tb.add_scalars(p_avg, step)
            wb.log(p_avg, step)
            ppo_metrics.clear()

        if (t + 1) % 10_000 == 0:
            state = {aid: actors[aid].ac.state_dict() for aid in actors}
            state.update({"mediator": mediator.critic.state_dict(), "step": t + 1})
            checkpointer.save(state, filename=f"checkpoint_{t+1}.pt")
    runner.close(); tb.close(); wb.close()

if __name__ == "__main__":
    main()