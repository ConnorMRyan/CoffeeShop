import pytest
import torch
import numpy as np
from core_marl.social_actor import SocialActor, SocialActorConfig
from core_marl.experience_buffer import ExperienceBuffer, ExperienceBatch
from agents.ppo import PPOAgent, PPOConfig


class FakeObsSpace:
    """Minimal gym-like obs space with a shape."""
    def __init__(self, shape):
        self.shape = shape


class FakeActSpace:
    """Minimal gym-like discrete action space."""
    def __init__(self, n):
        self.n = n


def test_imports():
    """Verify that core modules import without error."""
    from core_marl.mediator import CoffeeShopMediator
    from core_marl.social_actor import SocialActor
    from core_marl.experience_buffer import ExperienceBuffer
    from agents.ppo import PPOAgent
    from utils.metrics import Metrics


def test_ppo_agent_instantiation():
    """Construct a PPOAgent with a minimal config and verify it does not raise."""
    obs_space = FakeObsSpace((8,))
    act_space = FakeActSpace(4)

    agent = PPOAgent(obs_space=obs_space, act_space=act_space)
    assert agent.model is not None
    assert agent.optimizer is not None


def test_ppo_agent_custom_config():
    """PPOAgent respects a custom PPOConfig."""
    obs_space = FakeObsSpace((16,))
    act_space = FakeActSpace(6)
    config = PPOConfig(lr=1e-3, hidden_size=32, update_epochs=2)

    agent = PPOAgent(obs_space=obs_space, act_space=act_space, config=config)
    assert agent.config.lr == 1e-3
    assert agent.config.hidden_size == 32


def test_ppo_agent_act():
    """PPOAgent.act returns action/val/logp for a single observation."""
    obs_space = FakeObsSpace((8,))
    act_space = FakeActSpace(4)
    agent = PPOAgent(obs_space=obs_space, act_space=act_space)

    obs = np.zeros(8, dtype=np.float32)
    result = agent.act({"agent_0": obs})
    assert "agent_0" in result
    info = result["agent_0"]
    assert "action" in info and "val" in info and "logp" in info
    assert 0 <= info["action"] < 4


def test_ppo_store_and_update():
    """PPOAgent accumulates transitions and runs a PPO update without error."""
    obs_space = FakeObsSpace((8,))
    act_space = FakeActSpace(4)
    agent = PPOAgent(obs_space=obs_space, act_space=act_space)

    for _ in range(20):
        agent.store_transition(
            aid="agent_0",
            obs=np.zeros(8, dtype=np.float32),
            act=0,
            reward=1.0,
            val=0.5,
            logp=-1.0,
            terminal=False,
        )

    metrics = agent.update(last_vals={"agent_0": 0.0})
    assert "loss_policy" in metrics
    assert "loss_value" in metrics


def test_social_actor_omega():
    """SocialActor exposes a non-negative omega value."""
    obs_space = FakeObsSpace((8,))
    act_space = FakeActSpace(4)
    agent = PPOAgent(obs_space=obs_space, act_space=act_space)
    actor = SocialActor(agent, SocialActorConfig(id="agent_0", omega_init=0.1, omega_learnable=False))

    omega = actor.get_omega()
    assert isinstance(omega, float)
    assert omega >= 0.0


def test_social_actor_bc_loss():
    """SocialActor.incorporate_shared_experience returns a differentiable tensor."""
    obs_space = FakeObsSpace((8,))
    act_space = FakeActSpace(4)
    agent = PPOAgent(obs_space=obs_space, act_space=act_space)
    actor = SocialActor(agent, SocialActorConfig(id="agent_0", omega_init=0.1, omega_learnable=False))

    # Tensors must be on the same device as the agent's model (mirrors train loop behaviour)
    obs_batch = torch.zeros(4, 8, device=agent.device)
    act_batch = torch.zeros(4, dtype=torch.long, device=agent.device)
    loss = actor.incorporate_shared_experience(obs_batch, act_batch)

    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad


def test_experience_buffer_add_export():
    """ExperienceBuffer retains transitions and exports them correctly."""
    buf = ExperienceBuffer(capacity=50)
    obs = {"agent_0": np.zeros(8, dtype=np.float32)}
    actions = {"agent_0": 0}
    rewards = {"agent_0": 1.0}
    terminated = {"agent_0": False}
    truncated = {"agent_0": False}
    next_obs = {"agent_0": np.ones(8, dtype=np.float32)}
    infos = {}

    for _ in range(10):
        buf.add(obs, actions, rewards, terminated, truncated, next_obs, infos)

    assert len(buf) == 10
    batch = buf.export()
    assert len(batch.observations) == 10
