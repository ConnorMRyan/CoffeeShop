from __future__ import annotations
import heapq
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from core_marl.memory import ScoredMemory


@dataclass
class ExperienceBatch:
    """Container for a batch of multi-agent experience.

    Each field is a list aligned by time steps; values are per-agent dicts.
    """
    observations: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    rewards: List[Dict[str, float]]
    terminated: List[Dict[str, bool]]
    truncated: List[Dict[str, bool]]
    next_observations: List[Dict[str, Any]]
    infos: List[Dict[str, Any]] = field(default_factory=list)


class ExperienceBuffer:
    """Simple in-memory multi-agent experience buffer (rollout-style)."""

    def __init__(self, capacity: int = 1_000) -> None:
        self.capacity = capacity
        # deque(maxlen=capacity) gives O(1) FIFO eviction vs list.pop(0) O(n).
        self._storage: deque = deque(maxlen=capacity)

    def add(
        self,
        obs: Dict[str, Any],
        actions: Dict[str, Any],
        rewards: Dict[str, float],
        terminated: Dict[str, bool],
        truncated: Dict[str, bool],
        next_obs: Dict[str, Any],
        infos: Dict[str, Any] | None = None,
    ) -> None:
        self._storage.append((obs, actions, rewards, terminated, truncated, next_obs if next_obs is not None else {}, infos or {}))

    def clear(self) -> None:
        self._storage.clear()

    def __len__(self) -> int:  # noqa: D401
        return len(self._storage)

    def export(self) -> ExperienceBatch:
        obs_list: List[Dict[str, Any]] = []
        act_list: List[Dict[str, Any]] = []
        rew_list: List[Dict[str, float]] = []
        term_list: List[Dict[str, bool]] = []
        trunc_list: List[Dict[str, bool]] = []
        next_obs_list: List[Dict[str, Any]] = []
        infos_list: List[Dict[str, Any]] = []

        for obs, actions, rewards, terminated, truncated, next_obs, infos in self._storage:
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


class SharedExperienceBuffer:
    """Buffer for multi-agent experience shared across the population.

    Uses a priority-based system to keep the most valuable transitions.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._storage: List[ScoredMemory] = []
        self._lock = threading.Lock()

    def add(self, memories: List[ScoredMemory]) -> None:
        """Add a list of ScoredMemory objects to the buffer."""
        with self._lock:
            self._storage.extend(memories)
            # Post-loop heap trim: keep only the top capacity elements by priority.
            if len(self._storage) > self.capacity:
                self._storage = heapq.nlargest(self.capacity, self._storage, key=lambda x: x.priority)

    def sample_top(self, n: int, gamma_time: float = 0.99) -> List[ScoredMemory]:
        """Compute priority decay transiently and return the top n."""
        with self._lock:
            if not self._storage:
                return []

            current_time = time.monotonic()
            scored_list = []
            for mem in self._storage:
                dt = current_time - mem.transition.timestamp
                decayed_priority = mem.priority * (gamma_time ** dt)
                scored_list.append((decayed_priority, mem))

            scored_list.sort(key=lambda x: x[0], reverse=True)
            return [mem for _, mem in scored_list[:n]]

    def __len__(self) -> int:
        with self._lock:
            return len(self._storage)
