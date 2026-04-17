from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
import numpy as np
from tensordict import TensorDict

@dataclass
class ExperienceBatch:
    """Container for a batch of multi-agent experience."""
    observations: List[Dict[str, Any]] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    rewards: List[Dict[str, float]] = field(default_factory=list)
    terminated: List[Dict[str, bool]] = field(default_factory=list)
    truncated: List[Dict[str, bool]] = field(default_factory=list)
    next_observations: List[Dict[str, Any]] = field(default_factory=list)
    infos: List[Dict[str, Any]] = field(default_factory=list)

    def to_tensordict(self) -> TensorDict:
        """Convert the batch to a TensorDict for efficient processing."""
        # Multi-agent dicts are converted to nested TensorDicts
        # Observations: (T, num_agents, obs_dim) or (T, agents) dict
        
        # We need to know the agent IDs to group them properly
        if not self.observations:
            return TensorDict({}, batch_size=[0])
            
        agent_ids = list(self.observations[0].keys())
        T = len(self.observations)
        
        data = {}
        for key in ["observations", "actions", "rewards", "terminated", "truncated", "next_observations"]:
            val_list = getattr(self, key)
            if not val_list: continue
            
            agent_data = {}
            for aid in agent_ids:
                # Stack across time T
                stacked = np.stack([t_dict[aid] for t_dict in val_list])
                agent_data[aid] = torch.as_tensor(stacked)
            
            data[key] = TensorDict(agent_data, batch_size=[T])
            
        return TensorDict(data, batch_size=[T])

@dataclass
class PrioritizedMemory:
    priority: float
    timestamp: int
    data: Tuple[Any, ...]
    
    def __lt__(self, other):
        # We want a min-heap to efficiently discard lowest priorities during add
        return self.priority < other.priority

class SharedExperienceBuffer:
    """Centralized buffer that holds prioritized shared memories.
    
    Memories are prioritized by TD-error, and penalized by age (recency decay).
    """

    def __init__(self, capacity: int = 10_000, gamma_time: float = 0.999) -> None:
        self.capacity = capacity
        # Using a heap to automatically sort by highest priority
        self._storage: List[PrioritizedMemory] = []
        self.gamma_time = gamma_time

    def add(self, priorities: List[float], timestamp: int, batch: ExperienceBatch) -> None:
        """Add a batch of experiences with pre-computed TD-error priorities."""
        for t, priority in enumerate(priorities):
            # Extract individual transition
            transition = (
                batch.observations[t],
                batch.actions[t],
                batch.rewards[t],
                batch.terminated[t],
                batch.truncated[t],
                batch.next_observations[t],
                batch.infos[t] if batch.infos else {}
            )
            item = PrioritizedMemory(priority, timestamp, transition)
            
            if len(self._storage) < self.capacity:
                heapq.heappush(self._storage, item)
            else:
                # Use heappushpop to maintain capacity efficiently.
                # Since index 0 is the smallest priority (min-heap),
                # heappushpop will replace the smallest with the new one if new is larger,
                # or keep the smallest and discard the new one if new is even smaller.
                heapq.heappushpop(self._storage, item)

    def sample_top(self, current_time: int, n: int = 32) -> ExperienceBatch:
        """Sample the highest priority experiences, applying recency decay.

        Priorities are decayed transiently via a key function — stored
        priorities are NEVER mutated, preventing compounding decay on
        repeated calls.
        """
        def effective_priority(mem: PrioritizedMemory) -> float:
            age = max(0, current_time - mem.timestamp)
            return mem.priority * (self.gamma_time ** age)

        top_n = heapq.nlargest(n, self._storage, key=effective_priority)
        
        obs_list, act_list, rew_list, term_list, trunc_list, next_obs_list, infos_list = [], [], [], [], [], [], []
        
        for mem in top_n:
            obs, actions, rewards, terminated, truncated, next_obs, infos = mem.data
            obs_list.append(obs)
            act_list.append(actions)
            rew_list.append(rewards)
            term_list.append(terminated)
            trunc_list.append(truncated)
            next_obs_list.append(next_obs)
            infos_list.append(infos)

        return ExperienceBatch(
            observations=obs_list,
            actions=act_list,
            rewards=rew_list,
            terminated=term_list,
            truncated=trunc_list,
            next_observations=next_obs_list,
            infos=infos_list,
        )

    def __len__(self) -> int:
        return len(self._storage)

class ExperienceBuffer:
    """Simple in-memory multi-agent experience buffer (for local rollout before sharing).

    Uses collections.deque for O(1) append and automatic capacity eviction
    instead of O(n) list.pop(0).
    """
    def __init__(self, capacity: int = 1_000) -> None:
        self.capacity = capacity
        self._storage: deque = deque(maxlen=capacity)

    def add(self, obs, actions, rewards, terminated, truncated, next_obs, infos=None) -> None:
        # deque with maxlen automatically evicts the oldest entry — no pop(0) needed
        self._storage.append((obs, actions, rewards, terminated, truncated, next_obs or {}, infos or {}))

    def clear(self) -> None:
        self._storage.clear()

    def __len__(self) -> int:
        return len(self._storage)

    def export(self) -> ExperienceBatch:
        obs_list, act_list, rew_list, term_list, trunc_list, next_obs_list, infos_list = [], [], [], [], [], [], []
        for obs, actions, rewards, terminated, truncated, next_obs, infos in self._storage:
            obs_list.append(obs)
            act_list.append(actions)
            rew_list.append(rewards)
            term_list.append(terminated)
            trunc_list.append(truncated)
            next_obs_list.append(next_obs)
            infos_list.append(infos)
        return ExperienceBatch(
            observations=obs_list, actions=act_list, rewards=rew_list, terminated=term_list,
            truncated=trunc_list, next_observations=next_obs_list, infos=infos_list,
        )
