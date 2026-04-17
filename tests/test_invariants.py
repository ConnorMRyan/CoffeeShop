import torch
import numpy as np
from hypothesis import given, strategies as st
from core_marl.experience_buffer import ExperienceBuffer, ExperienceBatch
import pytest

# Strategies for generating multi-agent observations and actions
@st.composite
def multi_agent_obs(draw, num_agents, obs_dim):
    return {f"agent_{i}": draw(st.lists(st.floats(min_value=-10.0, max_value=10.0), min_size=obs_dim, max_size=obs_dim)) for i in range(num_agents)}

@st.composite
def multi_agent_actions(draw, num_agents):
    return {f"agent_{i}": draw(st.integers(min_value=0, max_value=5)) for i in range(num_agents)}

@st.composite
def multi_agent_rewards(draw, num_agents):
    return {f"agent_{i}": draw(st.floats(min_value=-1.0, max_value=1.0)) for i in range(num_agents)}

@st.composite
def multi_agent_bools(draw, num_agents):
    return {f"agent_{i}": draw(st.booleans()) for i in range(num_agents)}

@given(
    num_agents=st.integers(min_value=1, max_value=8),
    obs_dim=st.integers(min_value=1, max_value=64),
    batch_size=st.integers(min_value=1, max_value=32),
    num_insertions=st.integers(min_value=32, max_value=64)
)
def test_experience_buffer_invariant(num_agents, obs_dim, batch_size, num_insertions):
    buffer = ExperienceBuffer(capacity=100)
    
    # Insert random experiences
    for _ in range(num_insertions):
        obs = {f"agent_{i}": np.random.randn(obs_dim).astype(np.float32) for i in range(num_agents)}
        actions = {f"agent_{i}": np.random.randint(0, 5) for i in range(num_agents)}
        rewards = {f"agent_{i}": np.random.rand() for i in range(num_agents)}
        terminated = {f"agent_{i}": False for i in range(num_agents)}
        truncated = {f"agent_{i}": False for i in range(num_agents)}
        next_obs = {f"agent_{i}": np.random.randn(obs_dim).astype(np.float32) for i in range(num_agents)}
        
        buffer.add(obs, actions, rewards, terminated, truncated, next_obs)
    
    # Export and check batch
    batch = buffer.export()
    assert isinstance(batch, ExperienceBatch)
    assert len(batch.observations) == min(num_insertions, 100)
    
    # Verify tensor shapes after stack
    # In Phase 2 this will be unified with tensordict, but for now we verify current logic
    agent_ids = [f"agent_{i}" for i in range(num_agents)]
    
    # Simulate sampling/stacking logic often used in trainers
    # Goal: Ensure (batch_size, num_agents, obs_dim)
    
    # We'll sample a sub-batch of size 'batch_size' from the exported batch
    actual_batch_len = len(batch.observations)
    sample_indices = np.random.choice(actual_batch_len, size=min(batch_size, actual_batch_len), replace=False)
    
    sampled_obs = [batch.observations[i] for i in sample_indices]
    
    # Convert to (batch, agents, obs_dim)
    tensor_list = []
    for obs_dict in sampled_obs:
        agent_obs = [obs_dict[aid] for aid in agent_ids]
        tensor_list.append(np.stack(agent_obs))
    
    final_tensor = torch.as_tensor(np.stack(tensor_list))
    
    expected_shape = (len(sampled_obs), num_agents, obs_dim)
    assert final_tensor.shape == expected_shape, f"Expected {expected_shape}, got {final_tensor.shape}"

if __name__ == "__main__":
    # To run with hypothesis from CLI if needed
    test_experience_buffer_invariant()
