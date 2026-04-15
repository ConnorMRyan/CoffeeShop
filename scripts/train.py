from __future__ import annotations

import argparse
from typing import Any, Dict, List

from utils import get_logger, Metrics
from utils.diversity import calculate_population_diversity
from core_marl import Mediator
from envs import SocialEnvWrapper
from core_marl.social_actor import SocialActor, SocialActorConfig
from core_marl.experience_buffer import ExperienceBuffer, SharedExperienceBuffer
import numpy as np
import torch
from einops import rearrange, reduce

# Lazy imports to keep optional deps optional

def make_env(name: str, params: Dict[str, Any]) -> SocialEnvWrapper:
    name = name.lower()
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
    parser.add_argument("--horizon", type=int, default=200) # Local rollout horizon
    parser.add_argument("--meeting_interval", type=int, default=400) # When to share experiences
    args = parser.parse_args()

    log = get_logger()

    # Build environment and mediator
    env = make_env(args.env, {"layout_name": args.layout} if args.env == "overcooked" else {})
    mediator = Mediator(env)

    # Build base PPO agent
    base_agent = make_agent(args.agent, env.obs_space, env.act_space)
    
    # Wrap in SocialActors (decentralized accounts, shared policy weights mapping for simplicity)
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

    for t in range(args.steps):
        # 1. Actors act
        env_actions = {}
        act_info_dict = {}
        
        # Vectorized actor calls (if agent supports batching) - for now, per-agent
        # But we minimize data transfer within the loop
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
        if (t + 1) % args.horizon == 0:
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
        if (t + 1) % args.meeting_interval == 0:
            batch_to_share = local_buffer.export()
            local_buffer.clear()
            
            # Mediator evaluates TD-error
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

    mediator.close()
    log.info("Training complete.")

if __name__ == "__main__":
    main()
