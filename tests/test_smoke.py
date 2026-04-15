import pytest
import torch
import torch.nn as nn
from core_marl.mediator import CoffeeShopMediator
from core_marl.memory import PrioritizedBuffer, Transition, ScoredMemory
from core_marl.social_actor import SocialActor
from agents.ppo import PPOAgent
from utils.metrics import compute_population_diversity

def test_imports():
    """Verify that core modules import without error."""
    # These imports are already done at the top level, but we keep them here for Task 5 compliance.
    from core_marl.mediator import CoffeeShopMediator
    from core_marl.memory import PrioritizedBuffer
    from core_marl.social_actor import SocialActor
    from agents.ppo import PPOAgent
    from utils.metrics import compute_population_diversity

def test_ppo_agent_instantiation():
    """Construct a PPOAgent with a minimal config and verify it does not raise."""
    obs_dim = 8
    act_dim = 4
    global_obs_dim = 16
    
    # Minimal config for mediator
    mediator = CoffeeShopMediator(global_obs_dim=global_obs_dim)
    
    agent = PPOAgent(
        agent_id="test_agent",
        obs_space=obs_dim,
        act_space=act_dim,
        global_obs_dim=global_obs_dim,
        mediator=mediator
    )
    assert agent.agent_id == "test_agent"
    assert agent.model is not None

def test_buffer_push_sample():
    """Instantiate PrioritizedBuffer, push dummy objects, and verify length."""
    capacity = 100
    buffer = PrioritizedBuffer(capacity=capacity)
    
    obs_dim = 4
    global_obs_dim = 8
    
    for i in range(10):
        t = Transition(
            agent_id="agent_0",
            env_id=0,
            obs=torch.zeros(obs_dim),
            global_obs=torch.zeros(global_obs_dim),
            next_global_obs=torch.zeros(global_obs_dim),
            action=torch.tensor(0),
            log_prob=torch.tensor(0.0),
            reward=1.0,
            sparse_reward=0.0,
            value_est=0.5,
            done=False
        )
        # ScoredMemory(transition, td_error, social_bonus, priority)
        memory = ScoredMemory(transition=t, td_error=0.1, social_bonus=0.0, priority=float(i))
        buffer.push(memory)
        
    assert len(buffer) == 10
    
    samples = buffer.sample(5)
    assert len(samples) == 5
    for s in samples:
        assert isinstance(s, ScoredMemory)

def test_openness_parameter():
    """Verify that PPOAgent's omega parameter is an nn.Parameter and within range."""
    obs_dim = 8
    act_dim = 4
    global_obs_dim = 16
    mediator = CoffeeShopMediator(global_obs_dim=global_obs_dim)
    
    agent = PPOAgent(
        agent_id="test_agent",
        obs_space=obs_dim,
        act_space=act_dim,
        global_obs_dim=global_obs_dim,
        mediator=mediator
    )
    
    # In PPOAgent, openness is a LearnedOpenness object (nn.Module)
    # The actual parameter is agent.openness._raw
    assert isinstance(agent.openness._raw, nn.Parameter)
    
    # The value property returns the sigmoid of _raw, which is omega
    omega = agent.openness.value
    assert 0.05 <= omega <= 0.75
