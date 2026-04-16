# Run after: pip install -e .[test]
# Tests import cleanly because the package is installed in editable mode.
import pytest
pytest.importorskip("overcooked_ai_py")

import torch
import numpy as np

from core_marl import CoffeeShopMediator
from core_marl.social_actor import SocialActor, SocialActorConfig
from agents.ppo import PPOAgent
from envs.overcooked.wrapper import OvercookedSocialWrapper

def test_full_pipeline_gradient_flow():
    # 1. Setup
    env = OvercookedSocialWrapper(layout_name="cramped_room")
    mediator = CoffeeShopMediator(env)
    agent = PPOAgent(env.obs_space, env.act_space)
    actors = {aid: SocialActor(agent, SocialActorConfig(id=aid, omega_init=0.5, omega_learnable=True)) 
              for aid in mediator.agent_ids}
    
    # 2. Deterministic Rollout (Verification of shapes)
    obs, _ = mediator.reset()
    for _ in range(10):
        actions = {}
        act_infos = {}
        for aid, actor in actors.items():
            info = actor.act(obs[aid], deterministic=True)
            actions[aid], act_infos[aid] = info["action"], info
        
        next_obs, rewards, terminated, truncated, _ = mediator.step(actions)
        for aid in mediator.agent_ids:
            agent.store_transition(aid, obs[aid], actions[aid], rewards[aid], 
                                   act_infos[aid]["val"], act_infos[aid]["logp"], 
                                   terminated[aid] or truncated[aid])
        obs = next_obs

    # 3. PPO Update Gradient Flow
    initial_weights = [p.clone() for p in agent.model.parameters()]
    agent.update(last_vals={aid: 0.0 for aid in mediator.agent_ids})
    
    for p_init, p_new in zip(initial_weights, agent.model.parameters()):
        assert not torch.equal(p_init, p_new), "PPO update did not change weights"

    # 4. Social Influence (BC) Gradient Flow & Omega Learning
    # Create fake shared batch
    from core_marl.experience_buffer import ExperienceBatch
    fake_batch = ExperienceBatch(observations=[obs]*2, actions=[actions]*2, rewards=[rewards]*2,
                                 terminated=[terminated]*2, truncated=[truncated]*2, next_observations=[next_obs]*2)
    
    t_obs = torch.as_tensor(np.stack([obs["agent_0"]]*4), dtype=torch.float32, device=agent.device)
    t_act = torch.as_tensor(np.array([0,1,2,3]), dtype=torch.long, device=agent.device)
    
    for aid, actor in actors.items():
        omega_init = actor.omega.clone()
        actor.agent.optimizer.zero_grad()
        actor.optimizer.zero_grad()
        
        loss = actor.incorporate_shared_experience(t_obs, t_act)
        assert loss != 0, "BC loss should be non-zero"
        
        # Check gradient in omega
        assert actor.omega.grad is not None, f"Omega gradient not found for {aid}"
        assert actor.omega.grad.abs().sum() > 0, f"Omega gradient is zero for {aid}"
        
        actor.optimizer.step()
        assert not torch.equal(omega_init, actor.omega), f"Omega not updated for {aid}"
    
    env.close()
    print("✅ Integration test passed: Gradients flow through PPO and Social Influence (Omega).")

if __name__ == "__main__":
    test_full_pipeline_gradient_flow()
