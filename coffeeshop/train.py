from __future__ import annotations

import sys
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import polars as pl

import hydra
from omegaconf import DictConfig, OmegaConf
import time
import logging
from collections import defaultdict
from typing import Any, Dict

import numpy as np
import torch
from einops import rearrange

from utils import get_logger, Metrics, TBWriter, WandbWriter
from utils.checkpointing import Checkpointer
from utils.factory import make_env, _env_idx, make_actors, VectorSocialRunner
from utils.metrics import measure_population_diversity
from core_marl import CoffeeShopMediator
from core_marl.memory import ScoredMemory
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

def setup_distributed(backend: str = "gloo") -> tuple[int, int]:
    if not dist.is_available():
        return 0, 1
    if not dist.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if "cuda" in backend and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        else:
            dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        return rank, world_size
    return dist.get_rank(), dist.get_world_size()

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # 0. Distributed Setup
    rank, world_size = setup_distributed(backend=cfg.run.get("dist_backend", "gloo"))
    is_main_process = (rank == 0)

    # Use hydra.utils.get_original_cwd() for path compatibility
    original_cwd = hydra.utils.get_original_cwd()
    # os.chdir(original_cwd) # Hydra handles run dir, but if we need relative paths to work:

    t_cfg, e_cfg, a_cfg, m_cfg = cfg.trainer, cfg.env, cfg.agent, cfg.mediator

    # Apply rank to seed for uniqueness across processes
    base_seed = t_cfg.seed + rank
    torch.manual_seed(base_seed); np.random.seed(base_seed); log = get_logger()
    run_id = t_cfg.run_id or time.strftime("%Y%m%d_%H%M%S")
    checkpointer = Checkpointer(
        dirpath=os.path.abspath(t_cfg.checkpoint_dir),
        run_id=run_id,
        run_args=OmegaConf.to_container(cfg, resolve=True)
    )
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
        device=t_cfg.device,
        tau=m_cfg.get("tau", 5.0),
        tau_min=m_cfg.get("tau_min", 0.5),
        baseline_max_jump=m_cfg.get("baseline_max_jump", 2.0),
        target_tau=m_cfg.get("target_tau", 0.005),
    )
    actors = make_actors(runner, mediator, cfg, distributed=(world_size > 1))

    # --- Robust Loading ---
    if t_cfg.checkpoint_path and os.path.isfile(t_cfg.checkpoint_path):
        ckpt = torch.load(t_cfg.checkpoint_path, map_location=t_cfg.device)
        for aid, actor in actors.items():
            if aid in ckpt: actor.ac.load_state_dict(ckpt[aid])
        if "mediator" in ckpt:
            try:
                mediator.critic.load_state_dict(ckpt["mediator"])
            except RuntimeError as e:
                log.info(f"Mediator state mismatch: {e}. Starting fresh.")

    metrics, ep_rewards_env, ep_rewards_social, ep_stalls, ep_penalties, ppo_metrics = Metrics(), defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(list)
    obs_dict, _ = runner.reset()

    # Pre-compute env index for each agent ID — avoids repeated string parsing
    # in the 2M-step hot loop (e.g. "env0_agent_1" → 0).
    env_idx_cache: Dict[str, int] = {aid: _env_idx(aid) for aid in actors}
    # Pre-parse local agent IDs ("env0_agent_1" -> "agent_1")
    local_aid_cache: Dict[str, str] = {aid: aid.split("_", 1)[1] for aid in actors}

    # Pre-select device and non_blocking for performance
    device = torch.device(t_cfg.device)
    non_blocking = t_cfg.get("non_blocking", False)

    start_time = time.time()
    n_env_global = None
    for t in range(t_cfg.total_steps):
        step = t + 1
        env_global_obs = n_env_global if n_env_global is not None else {e: runner.get_global_obs(obs_dict, e) for e in range(runner.num_envs)}
        
        # Batch observation transfer to device
        for aid, o in obs_dict.items():
            obs_dict[aid] = o.to(device, non_blocking=non_blocking)

        actions = {aid: actor.act(env_idx_cache[aid], obs_dict[aid], env_global_obs[env_idx_cache[aid]]).item() for aid, actor in actors.items()}
        n_obs, rew, term, trunc, infos = runner.step(actions)

        # Handle reward averaging via the mediator to stop data leakage
        # The environment now returns raw rewards, and the mediator averages them
        # after capturing them for the centralized critic.
        if not t_cfg.get("baseline", False):
            rew = mediator.step(rew)

        # Batch next observation transfer to device
        for aid, o in n_obs.items():
            n_obs[aid] = o.to(device, non_blocking=non_blocking)

        n_env_global = {e: runner.get_global_obs(n_obs, e) for e in range(runner.num_envs)}

        for aid, actor in actors.items():
            e = env_idx_cache[aid]
            info_e = infos.get(e, {})
            l_aid = local_aid_cache[aid]
            sr = float(info_e.get("sparse_rewards", {}).get(l_aid, 0.0))
            actor.observe_outcome(n_env_global[e], float(rew[aid]), sr, bool(term[aid]), bool(trunc[aid]))
            
            # ep_return_env (raw game reward) vs ep_return_social (mediator bonus)
            # The agent's act/observe loop consumes "rew[aid]" which is team_reward/N + shaped_reward.
            # Social bonus is applied by the mediator later, but the agent's learned openness 
            # determines how much it values social stories via BC loss.
            # To track "Self" vs "Social", we log env reward (rew[aid]) and social signal (sr) separately.
            ep_rewards_env[aid] += float(rew[aid])
            ep_rewards_social[aid] += sr
            ep_stalls[aid] += float(info_e.get("action_entropy_stall", {}).get(l_aid, 0.0))
            
            # Accumulate inactivity penalty per agent for granular logging.
            # OvercookedSocialWrapper applies penalty to both reward and sparse_reward.
            # We track the total per-episode penalty here.
        metrics.update({
            "reward": float(sum(rew.values())),
        })
        obs_dict = n_obs
        # Snapshot observations *before* any episode resets so that
        # actor.update() bootstraps from the true post-rollout state, not
        # the initial obs of a newly reset episode.
        last_obs_snapshot = {aid: obs_dict[aid] for aid in actors}

        for e in range(runner.num_envs):
            ep_penalties[e] += float(infos.get(e, {}).get("inactivity_penalty", 0.0))
            if any(term.get(f"env{e}_agent_{i}", False) or trunc.get(f"env{e}_agent_{i}", False) for i in range(len(runner.envs[e].agent_ids))):
                ppo_metrics["env/inactivity_penalty"].append(ep_penalties[e])
                ep_penalties[e] = 0.0
                for i in range(len(runner.envs[e].agent_ids)):
                    gaid = f"env{e}_agent_{i}"
                    ppo_metrics[f"ep_return_env/{gaid}"].append(ep_rewards_env[gaid])
                    ppo_metrics[f"ep_return_social/{gaid}"].append(ep_rewards_social[gaid])
                    ppo_metrics[f"action_entropy_stall/{gaid}"].append(ep_stalls[gaid])
                    ep_rewards_env[gaid] = 0.0
                    ep_rewards_social[gaid] = 0.0
                    ep_stalls[gaid] = 0.0

                o, _ = runner.reset_env(e)
                for i, l_aid in enumerate(runner.envs[e].agent_ids):
                    gaid = f"env{e}_agent_{i}"
                    obs_dict[gaid] = o[l_aid]
                # After an environment reset, global observations for that env must be re-computed
                # for the next step's act() call.
                n_env_global[e] = runner.get_global_obs(obs_dict, e)

        if (t + 1) % t_cfg.update_interval == 0:
            # Meeting interval: push transitions across agents and aggregate TD errors
            if not t_cfg.get("baseline", False):
                # 1. Collect transitions from all agents
                all_trans = []
                for aid in actors:
                    all_trans.append(actors[aid].buffer.to_transitions(aid))
                
                # Ensure we have data and all buffers have the same length
                if all_trans and all_trans[0]:
                    num_timesteps = len(all_trans[0])
                    num_actors = len(actors)
                    
                    # Flatten transitions for batched evaluation: (timesteps * actors)
                    flattened_trans = []
                    for ts in range(num_timesteps):
                        for a_idx in range(num_actors):
                            flattened_trans.append(all_trans[a_idx][ts])
                    
                    # 2. Batched evaluation of TD errors
                    td_errors = mediator.evaluate_transitions(flattened_trans) # [T * A]
                    
                    # 3. Aggregate TD errors per timestep using einops.reduce
                    # Reshape to (timesteps, actors) and take max
                    from einops import reduce
                    agg_td_errors = reduce(td_errors, '(t a) -> t', 'max', a=num_actors)
                    # We still need best_agent_indices for step 4
                    td_errors_reshaped = rearrange(td_errors, '(t a) -> t a', a=num_actors)
                    best_agent_indices = td_errors_reshaped.argmax(dim=1)
                    
                    # 4. Create batch_to_share with one transition per timestep (the "best" one)
                    batch_to_share = []
                    for ts in range(num_timesteps):
                        best_a_idx = best_agent_indices[ts].item()
                        best_t = all_trans[best_a_idx][ts]
                        best_td = agg_td_errors[ts].item()
                        
                        # Re-calculate social bonus and priority for the best transition
                        social_bonus = (
                            mediator._compute_synergy_bonus(best_t.agent_id, best_t.sparse_reward)
                            if best_t.sparse_reward > 0.0 else 0.0
                        )
                        priority = mediator._compute_priority(best_td, social_bonus, best_t)
                        batch_to_share.append(
                            ScoredMemory(best_t, best_td, social_bonus, priority)
                        )
                    
                    # 5. Push aggregated batch to mediator
                    mediator.push_scored_batch(batch_to_share)

            for aid, actor in actors.items():
                # End of rollout: fetch V(last_obs) for bootstrapping
                if trunc[aid]:
                    with torch.no_grad():
                        _, v_last = actor.model(last_obs_snapshot[aid].to(device).unsqueeze(0))
                        actor.buffer._last_val = v_last.item()

                # update() now uses buffer._last_val for bootstrapping
                m = actor.update(last_obs=last_obs_snapshot[aid].to(t_cfg.device))
                actor.buffer.clear()
                for k, v in m.items(): ppo_metrics[f"{k}/{aid}"].append(v)

        if not t_cfg.get("baseline", False):
            if (t + 1) % t_cfg.critic_update_interval == 0 and len(mediator._buffer) >= 64:
                c_loss = float(mediator.update_critic(mediator._buffer.sample(256)))
                ppo_metrics["mediator/critic_loss"].append(c_loss)

        if t_cfg.shuffle_interval > 0 and (t + 1) % t_cfg.shuffle_interval == 0: shuffle_partners(runner, actors, log, t + 1)

        if (t + 1) % t_cfg.log_interval == 0:
            elapsed = time.time() - start_time
            sps = (t + 1) / max(elapsed, 1e-6)
            m_avg = metrics.mean()
            m_avg["perf/sps"] = sps
            log.info(f"[STEP {step}] Global | Reward: {m_avg.get('reward', 0.0):.4f} | SPS: {sps:.1f}")
            tb.add_scalars(m_avg, step)
            wb.log(m_avg, step)
            metrics.clear()
            
            p_avg = {k: float(np.mean(v)) for k, v in ppo_metrics.items() if v}
            # Global mediator critic loss for Q calculation
            global_c_loss = p_avg.get("mediator/critic_loss", 1.0)
            
            for aid in actors.keys():
                # Filter metrics for this specific agent
                a_m = {k.split('/')[0]: v for k, v in p_avg.items() if k.endswith(aid)}
                # Also include returns and stall counter which are prefixed differently
                a_m.update({k.split('/')[0]: v for k, v in p_avg.items() if k.endswith(aid) and '/' in k})
                
                om = actors[aid].openness.value
                # Q is a measure of social trust (global signal), but doesn't overwrite agent metrics
                q = mediator.get_verifiable_trust(global_c_loss)
                m_s = " | ".join([f"{k}: {v:.4f}" for k, v in a_m.items() if k not in ["ep_return_env", "ep_return_social", "action_entropy_stall"]])
                ret_s = f"EnvR: {a_m.get('ep_return_env', 0.0):.2f} | SocR: {a_m.get('ep_return_social', 0.0):.2f} | Stall: {a_m.get('action_entropy_stall', 0.0):.1f}"
                log.info(f"[STEP {step}] {aid} | ω: {om:.4f} | Q: {q:.4f} | {ret_s} | {m_s}")
            if "mediator/critic_loss" in p_avg: log.info(f"[STEP {step}] Mediator | Critic Loss: {p_avg['mediator/critic_loss']:.4f}")
            
            is_vanilla = t_cfg.get("baseline", False)

            # --- Population Diversity Logging (social runs only) ---
            if not is_vanilla:
                div = measure_population_diversity(actors, mediator, t_cfg.device)
                log.info(f"[STEP {step}] Population | Diversity (JS): {div:.4f}")
                tb.add_scalar("population/diversity", div, step)
                wb.log({"population/diversity": div}, step)

                # --- Social Sabbatical Intervention ---
                sabbatical_agent = mediator.enforce_diversity(div, actors)
                if sabbatical_agent:
                    log.info({"event": "sabbatical_trigger", "agent": sabbatical_agent, "msg": f"🏝️ SABBATICAL TRIGGERED. Agent {sabbatical_agent} has been sent on a solo mission to break the learning plateau."})

                # --- Broadcast Staleness Diagnostic ---
                sample_aid = list(actors.keys())[0]
                mediator.broadcast(sample_aid, actors[sample_aid].openness.value, n=0)
                b_stats = mediator._last_broadcast_stats
                if b_stats:
                    pct_fresh = b_stats["n_fresh"] / max(b_stats["n_not_self"], 1)
                    log.info(f"[STEP {step}] Broadcast | Fresh: {b_stats['n_fresh']} ({pct_fresh:.1%}) | Baseline: {b_stats['baseline']:.4f}")
                    tb.add_scalar("broadcast/n_fresh", b_stats["n_fresh"], step)
                    tb.add_scalar("broadcast/baseline", b_stats["baseline"], step)
                    tb.add_scalar("broadcast/pct_fresh", pct_fresh, step)
                    wb.log({
                        "broadcast/n_fresh": b_stats["n_fresh"],
                        "broadcast/baseline": b_stats["baseline"],
                        "broadcast/pct_fresh": pct_fresh
                    }, step)

                # --- Priority Distribution Diagnostic ---
                if (t + 1) % 10_000 == 0:
                    stats = mediator.get_priority_stats()
                    log.info(f"[STEP {t+1}] Buffer Priority | Min: {stats['min']:.4f} | Max: {stats['max']:.4f} | Mean: {stats['mean']:.4f} | Median: {stats['median']:.4f}")
                    for k, v in stats.items():
                        tb.add_scalar(f"buffer/priority_{k}", v, t + 1)
                        wb.log({f"buffer/priority_{k}": v}, t + 1)

            tb.add_scalars(p_avg, step)
            tb.flush()
            wb.log(p_avg, step)
            ppo_metrics.clear()

        if (t + 1) % 10_000 == 0:
            state = {aid: actors[aid].ac.state_dict() for aid in actors}
            state.update({"mediator": mediator.critic.state_dict(), "step": t + 1})
            # Explicitly include run_args in the state dict just in case the checkpointer.save 
            # logic is bypassed or if we want extra redundancy in the saved file.
            state["run_args"] = OmegaConf.to_container(cfg, resolve=True)
            checkpointer.save(state, filename=f"checkpoint_{t+1}.pt")
    runner.close(); tb.close(); wb.close()
    if is_main_process:
        metrics.report_final(f"metrics_rank{rank}.parquet")

if __name__ == "__main__":
    main()