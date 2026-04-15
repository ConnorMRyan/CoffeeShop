import pytest
import torch
import time
import heapq
from core_marl.experience_buffer import SharedExperienceBuffer, ExperienceBuffer
from core_marl.memory import ScoredMemory, Transition

def test_experience_buffer_deque():
    capacity = 10
    buffer = ExperienceBuffer(capacity=capacity)
    
    for i in range(15):
        buffer.add({}, {}, {"a": float(i)}, {"a": False}, {"a": False}, {})
    
    assert len(buffer) == 10
    batch = buffer.export()
    # The first 5 should be evicted, so rewards start at 5.0
    assert batch.rewards[0]["a"] == 5.0
    assert batch.rewards[-1]["a"] == 14.0

def test_shared_experience_buffer_add():
    capacity = 5
    buffer = SharedExperienceBuffer(capacity=capacity)
    
    memories = []
    for i in range(10):
        t = Transition(
            agent_id="a1", env_id=0, obs=torch.zeros(1), global_obs=torch.zeros(1),
            next_global_obs=torch.zeros(1), action=torch.tensor(0),
            log_prob=torch.tensor(0.0), reward=float(i), sparse_reward=0.0,
            value_est=0.0, done=False
        )
        memories.append(ScoredMemory(transition=t, td_error=0.0, social_bonus=0.0, priority=float(i)))
    
    buffer.add(memories)
    assert len(buffer) == capacity
    
    # Check that it kept the highest priority ones (5-9)
    stored_priorities = sorted([m.priority for m in buffer._storage])
    assert stored_priorities == [5.0, 6.0, 7.0, 8.0, 9.0]

def test_shared_experience_buffer_sample_top():
    buffer = SharedExperienceBuffer(capacity=10)
    
    # Memory 1: Priority 10, just now
    t1 = Transition(
        agent_id="a1", env_id=0, obs=torch.zeros(1), global_obs=torch.zeros(1),
        next_global_obs=torch.zeros(1), action=torch.tensor(0),
        log_prob=torch.tensor(0.0), reward=1.0, sparse_reward=0.0,
        value_est=0.0, done=False, timestamp=time.monotonic()
    )
    m1 = ScoredMemory(transition=t1, td_error=0.0, social_bonus=0.0, priority=10.0)
    
    # Memory 2: Priority 20, but old (should decay below m1)
    # If gamma_time = 0.5, and dt = 2, decayed_priority = 20 * (0.5^2) = 5.0
    t2 = Transition(
        agent_id="a1", env_id=0, obs=torch.zeros(1), global_obs=torch.zeros(1),
        next_global_obs=torch.zeros(1), action=torch.tensor(0),
        log_prob=torch.tensor(0.0), reward=2.0, sparse_reward=0.0,
        value_est=0.0, done=False, timestamp=time.monotonic() - 2.0
    )
    m2 = ScoredMemory(transition=t2, td_error=0.0, social_bonus=0.0, priority=20.0)
    
    buffer.add([m1, m2])
    
    # Sample top 1 with gamma_time=0.5
    sampled = buffer.sample_top(n=1, gamma_time=0.5)
    assert len(sampled) == 1
    assert sampled[0].priority == 10.0 # m1 should be top despite lower raw priority
    
    # Sample top 1 with gamma_time=1.0 (no decay)
    sampled_no_decay = buffer.sample_top(n=1, gamma_time=1.0)
    assert sampled_no_decay[0].priority == 20.0 # m2 should be top
