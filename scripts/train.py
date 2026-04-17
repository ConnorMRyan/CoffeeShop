from __future__ import annotations

import os
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict, List, Optional

from utils import get_logger, Metrics
from utils.diversity import calculate_population_diversity
from core_marl import CoffeeShopMediator, SocialActor, SocialActorConfig, ExperienceBuffer, SharedExperienceBuffer
from envs import SocialEnvWrapper
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from einops import rearrange, reduce

# Lazy imports to keep optional deps optional

def setup_distributed(backend: str = "gloo") -> tuple[int, int]:
    if not dist.is_available():
        return 0, 1
    
    # Check if we are running under torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
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
    else:
        # Not running in distributed mode
        return 0, 1

def make_env(name: str, params: Dict[str, Any], seed: Optional[int] = None) -> SocialEnvWrapper:
    name = name.lower()
    if name == "overcooked":
        from envs.overcooked.wrapper import OvercookedSocialWrapper
        env = OvercookedSocialWrapper(**params)
    elif name == "crafter":
        from envs.crafter.wrapper import CrafterWrapper
        env = CrafterWrapper(**params)
    elif name == "nethack":
        from envs.nethack.wrapper import NLESocialWrapper
        env = NLESocialWrapper(**params)
    elif name == "aisaac":
        from envs.aisaac.wrapper import AIsaacWrapper
        env = AIsaacWrapper(**params)
    else:
        raise ValueError(f"Unknown env name: {name}")
    
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_agent(name: str, obs_space=None, act_space=None, distributed: bool = False):
    name = name.lower()
    if name == "ppo":
        from agents.ppo import PPOAgent
        agent = PPOAgent(obs_space, act_space)
        if distributed:
            # Wrap model in DDP
            agent.model = DDP(agent.model)
        return agent
    if name == "sac":
        from agents.sac import SACAgent
        return SACAgent(obs_space, act_space)
    raise ValueError(f"Unknown agent name: {name}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # 0. Distributed Setup
    rank, world_size = setup_distributed(backend=cfg.run.get("dist_backend", "gloo"))
    is_main_process = (rank == 0)

    # Set deterministic run if seed provided
    if cfg.run.seed is not None:
        torch.manual_seed(cfg.run.seed + rank)
        np.random.seed(cfg.run.seed + rank)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Use hydra.utils.get_original_cwd() for path compatibility
    original_cwd = hydra.utils.get_original_cwd()
    
    log = get_logger()
    if not is_main_process:
        # Suppress logging for non-main processes if needed, 
        # or just let them log with rank prefix.
        pass

    # 1. Build environment and mediator
    # Apply seed offset for diversity in parallel envs
    base_seed = cfg.run.get("seed", 42)
    env_seed = base_seed + rank
    
    env_params = {}
    if cfg.env.name == "overcooked":
        env_params["layout_name"] = cfg.env.layout_name
        
    env = make_env(cfg.env.name, env_params, seed=env_seed)
    mediator = CoffeeShopMediator(env)

    # 2. Build base agent
    # We pass distributed=True to wrap the internal model in DDP
    base_agent = make_agent(cfg.agent.name, env.obs_space, env.act_space, distributed=(world_size > 1))
    
    # Wrap in SocialActors
    actors = {
        aid: SocialActor(base_agent, SocialActorConfig(id=aid, omega_init=0.1, omega_learnable=False))
        for aid in mediator.agent_ids
    }
    
    metrics = Metrics()
    
    # Buffers
    local_buffer = ExperienceBuffer()
    shared_buffer = SharedExperienceBuffer(capacity=5000, gamma_time=0.999)

    obs, infos = mediator.reset()
    
    # Store temporary transitions before stepping
    last_act_info = {}

    for t in range(cfg.run.steps):
        # 1. Actors act
        env_actions = {}
        act_info_dict = {}
        
        for aid, actor in actors.items():
            act_info = actor.act(obs[aid])
            if act_info is not None:
                act_info_dict[aid] = act_info
                env_actions[aid] = act_info["action"]

        # 2. Env step
        next_obs, rewards, terminated, truncated, infos = mediator.step(env_actions)

        # 3. Store local transitions for PPO AND for social sharing
        local_buffer.add(obs, env_actions, rewards, terminated, truncated, next_obs, infos)
        
        for aid in mediator.agent_ids:
            done = terminated[aid] or truncated[aid]
            if aid in act_info_dict:
                base_agent.store_transition(
                    aid=aid, obs=obs[aid], act=act_info_dict[aid]["action"], 
                    reward=rewards[aid], val=act_info_dict[aid]["val"], 
                    logp=act_info_dict[aid]["logp"], terminal=done
                )

        # 4. Trigger PPO Update
        if (t + 1) % cfg.run.horizon == 0:
            # Calculate last values for bootstrapping
            last_vals = {}
            for aid, single_obs in obs.items():
                o = torch.as_tensor(single_obs, dtype=torch.float32, device=base_agent.device)
                with torch.no_grad():
                    _, val = base_agent.model(o)
                    last_vals[aid] = val.item()
            
            update_metrics = base_agent.update(last_vals=last_vals)
            metrics.update(update_metrics)
            
        # 5. The Meeting (Asynchronous Social Sharing)
        if (t + 1) % cfg.run.meeting_interval == 0:
            batch_to_share = local_buffer.export()
            local_buffer.clear()
            
            # CoffeeShopMediator evaluates TD-error
            td_errors, critic_loss = mediator.evaluate_and_prioritize(batch_to_share)
            metrics.update({"mediator_critic_loss": critic_loss})
            
            if td_errors.numel() > 0:
                # Needs exactly one priority per timestep, so reduce across agents
                # by taking the max TD-error per step.
                num_agents = len(mediator.agent_ids)
                if td_errors.numel() == len(batch_to_share.observations) * num_agents:
                    priorities = reduce(td_errors, '(t a) -> t', 'max', a=num_agents).detach().cpu().tolist()
                else:
                    priorities = td_errors.detach().cpu().tolist()  # already per-step
                shared_buffer.add(priorities, timestamp=t, batch=batch_to_share)
                
                # Sample top M experiences to distill back to actors
                distillation_batch = shared_buffer.sample_top(current_time=t, n=32)
                
                # Format to tensors
                flat_obs, flat_acts = [], []
                for b_t in range(len(distillation_batch.observations)):
                    for aid in mediator.agent_ids:
                        if aid in distillation_batch.observations[b_t]:
                            flat_obs.append(distillation_batch.observations[b_t][aid])
                            flat_acts.append(distillation_batch.actions[b_t][aid])
                            
                if flat_obs:
                    # Minimize redundant numpy conversions by stack directly to tensor if possible
                    # but observations are currently numpy from env.wrapper
                    t_obs = torch.as_tensor(np.stack(flat_obs), dtype=torch.float32, device=base_agent.device)
                    t_act = torch.as_tensor(np.stack(flat_acts), dtype=torch.long, device=base_agent.device)
                    
                    bc_losses = {}
                    for aid, actor in actors.items():
                        bc_loss = actor.incorporate_shared_experience(t_obs, t_act)
                        bc_losses[f"{aid}_bc_loss"] = bc_loss
                        bc_losses[f"{aid}_omega"] = actor.get_omega()
                        
                        # Step omega optimizer if learnable
                        if actor.config.omega_learnable:
                            actor.optimizer.step()
                            actor.optimizer.zero_grad()
                            
                    metrics.update(bc_losses)

        # Simple logging/metrics
        metrics.update({"reward": sum(rewards.values())})
        
        if (t + 1) % 100 == 0:
            log.info({"step": t + 1, **metrics.mean()})
            metrics.clear()

        obs = next_obs
        if any(terminated.values()) or any(truncated.values()):
            obs, infos = mediator.reset()

    if is_main_process:
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        metrics.report_final(os.path.join(output_dir, f"metrics_rank{rank}.parquet"))
    
    mediator.close()
    log.info("Training complete.")

if __name__ == "__main__":
    main()
