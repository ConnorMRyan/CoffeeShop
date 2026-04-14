#!/usr/bin/env python3
"""Quick test of CoffeeShopMediator core API with dummy transitions (no env deps)."""

import sys
import os

import torch
from core_marl.mediator import CoffeeShopMediator
from core_marl.memory import Transition

print("1. Initializing CoffeeShop Mediator...")
GLOBAL_OBS_DIM = 16
mediator = CoffeeShopMediator(global_obs_dim=GLOBAL_OBS_DIM, buffer_capacity=100)

print("2. Creating and pushing dummy transitions...")
for i in range(5):
    t = Transition(
        agent_id        = "agent_0",
        env_id          = 0,
        obs             = torch.zeros(4),
        global_obs      = torch.zeros(GLOBAL_OBS_DIM),
        next_global_obs = torch.zeros(GLOBAL_OBS_DIM),
        action          = torch.tensor(0),
        log_prob        = torch.tensor(0.0),
        reward          = float(i % 2),
        sparse_reward   = float(i % 2),
        value_est       = 0.5,
        done            = False,
    )
    mediator.push(t)
print("✅ Pushed 5 transitions.")

print("3. Updating critic with a batch sampled from the buffer...")
# update_critic now expects List[ScoredMemory]; sample directly from the buffer.
batch = mediator._buffer.sample(5)
assert len(batch) > 0, "Buffer should contain pushed transitions"
loss = mediator.update_critic(batch)
print(f"✅ Critic update ran. Loss: {loss:.4f}")

print("4. Broadcasting top memories to a peer with full openness...")
peers = mediator.broadcast(requesting_agent_id="agent_2", openness=1.0, n=3)
print(f"✅ Broadcast returned {len(peers)} memories (<=3).")

print("\n🏁 Mediator API smoke test complete!")
