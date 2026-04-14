from __future__ import annotations
import heapq
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import threading
import torch

if TYPE_CHECKING:
    from core_marl.mediator import CoffeeShopMediator

@dataclass
class Transition:
    """One environment step, as pushed by a PPOActor."""
    agent_id:        str
    env_id:          int                   # which vectorized env this came from
    obs:             torch.Tensor          # local observation
    global_obs:      torch.Tensor          # full env state at s  (for centralized critic)
    next_global_obs: torch.Tensor          # full env state at s' (required for correct TD)
    action:          torch.Tensor
    log_prob:        torch.Tensor          # π_old(a|s)
    reward:          float
    sparse_reward:   float                 # isolated sparse signal (e.g. dish served)
    value_est:       float                 # actor's own V(s) estimate at collection time
    done:            bool
    truncated:       bool = False
    timestamp:       float = field(default_factory=time.monotonic)

@dataclass(frozen=True)
class ScoredMemory:
    """A Transition annotated with Mediator-computed priority."""
    transition:   Transition
    td_error:     float
    social_bonus: float                    # extra reward credited from peer synergy
    priority:     float                    # composite score used for broadcasting

class PrioritizedBuffer:
    """
    Fixed-capacity min-heap buffer that maintains the highest-priority
    transitions seen across the entire population.

    This gives the γ_time property from the paper: as the population improves
    and pushes higher-priority memories, older low-value transitions are
    progressively displaced.  Insertion order plays no role in eviction.

    A monotone counter is used as a tiebreaker so ScoredMemory objects are
    never compared directly (they contain tensors, which are not orderable).
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._heap: List[tuple] = []   # (priority, counter, ScoredMemory)
        self._counter = 0
        self._lock = threading.Lock()

    def push(self, memory: ScoredMemory) -> None:
        with self._lock:
            entry = (memory.priority, self._counter, memory)
            self._counter += 1
            if len(self._heap) < self._capacity:
                heapq.heappush(self._heap, entry)
            elif memory.priority > self._heap[0][0]:
                # New memory beats the current worst — replace it.
                heapq.heapreplace(self._heap, entry)
            # else: new memory is worse than the current minimum; discard.

    def sample(self, n: int) -> List[ScoredMemory]:
        """Return a random sample of up to n entries."""
        with self._lock:
            if not self._heap:
                return []
            if len(self._heap) <= n:
                return [m for _, _, m in self._heap]
            indices = np.random.choice(len(self._heap), size=n, replace=False)
            return [self._heap[i][2] for i in indices]

    def __iter__(self):
        with self._lock:
            # Return a copy of the list of memories for safe iteration
            # This avoids holding the lock throughout the entire iteration
            # which could be slow.
            memories = [m for _, _, m in self._heap]
        return iter(memories)

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)
