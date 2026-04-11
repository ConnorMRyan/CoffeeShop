from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


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
