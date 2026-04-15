import pytest
import torch
from core_marl.memory import PrioritizedBuffer, ScoredMemory, Transition

def test_centralized_buffer_capacity():
    buffer = PrioritizedBuffer(capacity=10)
    assert len(buffer) == 0

    # Push 15 items
    for i in range(15):
        t = Transition(
            agent_id="a1",
            env_id=0,
            obs=torch.zeros(4),
            global_obs=torch.zeros(4),
            next_global_obs=torch.zeros(4),
            action=torch.tensor(0),
            log_prob=torch.tensor(0.0),
            reward=float(i),
            sparse_reward=float(i),
            value_est=0.0,
            done=False
        )
        sm = ScoredMemory(transition=t, td_error=float(i), social_bonus=0.0, priority=float(i))
        buffer.push(sm)
    
    # Capacity should not exceed 10
    assert len(buffer) == 10
    # Checking Priority Heap behavior
    # The first 5 items (priority 0..4) should be evicted. Items 5 to 14 should remain.
    # We can inspect the reward to identify them.
    rewards = [m.transition.reward for _, _, m in buffer._heap]
    assert 14.0 in rewards
    assert 0.0 not in rewards

def test_centralized_buffer_sample():
    buffer = PrioritizedBuffer(capacity=100)
    for i in range(20):
        t = Transition(
            agent_id="a1",
            env_id=0,
            obs=torch.zeros(4),
            global_obs=torch.zeros(4),
            next_global_obs=torch.zeros(4),
            action=torch.tensor(0),
            log_prob=torch.tensor(0.0),
            reward=float(i),
            sparse_reward=float(i),
            value_est=0.0,
            done=False
        )
        sm = ScoredMemory(transition=t, td_error=0.1, social_bonus=0.0, priority=float(i))
        buffer.push(sm)
        
    sampled = buffer.sample(n=5)
    assert len(sampled) == 5
    assert isinstance(sampled[0], ScoredMemory)
