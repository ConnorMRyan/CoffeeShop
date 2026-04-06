"""
CoffeeShop Mediator — Off-Policy TD-Error Evaluation & Social Bonus Shaping
============================================================================
Architecture role:
  - Accepts asynchronous trajectory pushes from decentralized PPO SocialActors.
  - Maintains a centralized prioritized replay buffer keyed by agent_id.
  - Computes TD-errors to rank memory novelty / surprise.
  - Applies a *Social Bonus*: rewards peer agents for synergistic contributions.
  - Global Gossip/Skepticism Trigger — monitors global reward trends and
    reduces trust (effective openness) if performance stagnates or declines.

Bug fixes over previous version
--------------------------------
1. _is_stale        — now compares raw transition signal (reward + sparse_reward)
                      against requester's baseline, not sender's, and does not
                      mix in the Mediator-computed social_bonus.

2. _recency_penalty — penalty is now computed relative to the *requesting*
                      agent's baseline (passed in), not the stored transition's
                      sender baseline.  A separate _sender_penalty helper
                      remains for priority scoring at push() time where the
                      sender baseline is appropriate.

3. get_gossip_factor — no longer mutates _atb_reward; that update lives
                       exclusively in push() so there is one authoritative
                       write path.

4. update_critic    — TD target now uses the *next-state* value estimate
                       V(s') from a forward pass through the critic rather
                       than re-using the actor's V(s) estimate.  A separate
                       next_global_obs field is required on Transition for
                       this; see dataclass below.

5. synergy_score    — when all peer values are identical (spread ≤ 1e-8) the
                       score is 0.5 regardless of sign, avoiding a spurious
                       maximum bonus for perfectly equal peers.
"""

from __future__ import annotations

import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

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
    timestamp:       float = field(default_factory=time.monotonic)


@dataclass
class ScoredMemory:
    """A Transition annotated with Mediator-computed priority."""
    transition:   Transition
    td_error:     float
    social_bonus: float                    # extra reward credited from peer synergy
    priority:     float                    # composite score used for broadcasting


# ---------------------------------------------------------------------------
# Centralized Value Network
# ---------------------------------------------------------------------------

class CentralizedCritic(nn.Module):
    """Lightweight V(s_global) estimator for TD-error calculations."""

    def __init__(self, global_obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs).squeeze(-1)


# ---------------------------------------------------------------------------
# Mediator
# ---------------------------------------------------------------------------

class CoffeeShopMediator:
    def __init__(
            self,
            global_obs_dim:  int,
            buffer_capacity: int   = 10_000,
            gamma:           float = 0.99,
            epsilon_td:      float = 0.05,
            synergy_alpha:   float = 0.3,
            critic_lr:       float = 3e-4,
            device:          str   = "cpu",
    ):
        self.gamma         = gamma
        self.epsilon_td    = epsilon_td
        self.synergy_alpha = synergy_alpha
        self.device        = torch.device(device)

        # ── Centralized critic & optimizer ────────────────────────────────
        self.critic     = CentralizedCritic(global_obs_dim).to(self.device)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ── Shared prioritized buffer ─────────────────────────────────────
        self._buffer: deque[ScoredMemory] = deque(maxlen=buffer_capacity)
        self._value_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

        # ── Global Gossip & Skepticism Tracking ───────────────────────────
        self._reward_window: deque[float] = deque(maxlen=100)
        self._env_reward_windows: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))

        # FIX (bug 3): _atb_reward is written ONLY in push(); get_gossip_factor
        # reads it without touching it, eliminating the dual-write race.
        self._atb_reward: float = -float("inf")
        self._gossip_alpha      = 0.7

        # ── Population baseline tracker ───────────────────────────────────
        self._agent_baselines: Dict[str, float] = {}
        self._baseline_alpha  = 0.05

        # ── Agent last-seen env (for cross-env reputation boosts) ─────────
        self._agent_last_env: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Gossip Logic — Performance Monitoring
    # ------------------------------------------------------------------

    def get_gossip_factor(self) -> float:
        """
        Returns a multiplier in (0, 1] to reduce agent trust when the group
        is underperforming relative to its all-time-best rolling average.

        FIX (bug 3): _atb_reward is no longer mutated here.  It is updated
        exclusively inside push(), ensuring a single authoritative write path.
        """
        if len(self._reward_window) < 25:
            return 1.0

        current_avg = float(np.mean(self._reward_window))

        if current_avg < 0.7 * self._atb_reward:
            return self._gossip_alpha
        return 1.0

    # ------------------------------------------------------------------
    # Public API — called by SocialActors
    # ------------------------------------------------------------------

    def push(self, transition: Transition) -> None:
        """Score and store a transition.  Sole writer of _atb_reward."""
        self._value_history[transition.agent_id].append(transition.value_est)
        self._agent_last_env[transition.agent_id] = transition.env_id

        total_r = transition.reward + transition.sparse_reward
        self._reward_window.append(total_r)
        self._env_reward_windows[transition.env_id].append(total_r)

        # FIX (bug 3): single authoritative update of G-ATB
        if len(self._reward_window) > 0:
            current_global_avg = float(np.mean(self._reward_window))
            self._atb_reward = max(self._atb_reward, current_global_avg)

        td_error, social_bonus = self._compute_td_and_bonus(transition)
        priority = self._compute_priority(td_error, social_bonus, transition)

        self._buffer.append(ScoredMemory(transition, td_error, social_bonus, priority))

    def broadcast(
            self,
            requesting_agent_id: str,
            openness: float,
            n: int = 16,
    ) -> List[ScoredMemory]:
        """
        Returns high-priority memories gated by an 'Effective Openness' factor
        that combines local agent trust and global Mediator skepticism.

        Staleness and recency penalty are now evaluated relative to the
        *requesting* agent's baseline (fixes bugs 1 and 2).

        Reputation-Based Priority Boost:
        - Memories from the highest-performing env are boosted 1.5× when
          broadcasting to an agent from a lower-performing env.
        """
        effective_openness = openness * self.get_gossip_factor()

        # Per-env averages and top environment
        env_avgs = {
            eid: float(np.mean(w)) if len(w) > 0 else -float("inf")
            for eid, w in self._env_reward_windows.items()
        }
        top_env     = max(env_avgs, key=env_avgs.get) if env_avgs else None
        req_env     = self._agent_last_env.get(requesting_agent_id)
        req_env_avg = env_avgs.get(req_env, -float("inf")) if req_env is not None else -float("inf")
        top_avg     = env_avgs.get(top_env, -float("inf")) if top_env is not None else -float("inf")
        is_underperforming = (
                req_env is not None
                and top_env is not None
                and req_env != top_env
                and req_env_avg < top_avg
        )

        # FIX (bugs 1 & 2): pass the requester's baseline into helpers so that
        # staleness and recency penalty are measured from the receiver's
        # perspective, not the sender's.
        requester_baseline = self._agent_baselines.get(requesting_agent_id, 0.0)

        def adjusted_priority(m: ScoredMemory) -> float:
            p = m.priority - self._requester_recency_penalty(m.transition, requester_baseline)
            if is_underperforming and top_env is not None and m.transition.env_id == top_env:
                p *= 1.5
            return p

        candidates = [
            m for m in self._buffer
            if m.priority >= self.epsilon_td
               and m.transition.agent_id != requesting_agent_id
               and not self._is_stale(m, requester_baseline)
        ]

        candidates.sort(key=adjusted_priority, reverse=True)
        top = candidates[:n]

        rng = np.random.default_rng()
        accepted = [m for m in top if rng.random() < effective_openness]
        return accepted

    def update_critic(self, batch: List[Transition]) -> float:
        """
        Update the centralized critic with a batch of transitions.

        FIX (bug 4): TD targets now use V(s') from a fresh forward pass
        through self.critic rather than re-using the actor's stale V(s)
        estimate.  This requires next_global_obs on each Transition.
        """
        if not batch:
            return 0.0

        g_obs      = torch.stack([t.global_obs      for t in batch]).to(self.device)
        next_g_obs = torch.stack([t.next_global_obs for t in batch]).to(self.device)
        rwds       = torch.tensor([t.reward for t in batch],
                                  dtype=torch.float32, device=self.device)
        dones      = torch.tensor([t.done   for t in batch],
                                  dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # Correct Bellman backup: r + γ · V(s') · (1 - done)
            v_next      = self.critic(next_g_obs)          # V(s')
            td_targets  = rwds + self.gamma * v_next * (1.0 - dones)

        values = self.critic(g_obs)
        loss   = nn.functional.mse_loss(values, td_targets)

        self.critic_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_opt.step()

        self._update_baselines(batch)
        return loss.item()

    # ------------------------------------------------------------------
    # Core Logic — Social Bonus & TD-Error
    # ------------------------------------------------------------------

    def _compute_td_and_bonus(self, t: Transition) -> Tuple[float, float]:
        """Compute (td_error, social_bonus) for a freshly pushed transition."""
        social_bonus = 0.0
        if t.sparse_reward > 0.0:
            social_bonus = self._compute_synergy_bonus(t.agent_id, t.sparse_reward)

        effective_reward = t.reward + social_bonus
        v_next = 0.0 if t.done else t.value_est   # actor estimate; critic not yet updated
        td_target = effective_reward + self.gamma * v_next
        td_error  = abs(td_target - t.value_est)

        return td_error, social_bonus

    def _compute_synergy_bonus(self, triggering_agent: str, sparse_reward: float) -> float:
        """
        Reward the triggering agent for achieving a sparse event when peers
        are performing well relative to one another.

        FIX (bug 5): when peer values are all equal (spread ≤ 1e-8) the
        synergy score is 0.5, not 1.0.  Equal performance is neutral — it
        indicates no differential contribution, so max bonus is unwarranted.
        """
        peer_means: List[float] = []
        for aid, hist in self._value_history.items():
            if aid == triggering_agent or len(hist) == 0:
                continue
            peer_means.append(float(np.mean(hist)))

        if not peer_means:
            return 0.0

        arr    = np.array(peer_means)
        spread = float(arr.max() - arr.min())

        if spread > 1e-8:
            synergy_score = float(np.clip((arr.mean() - arr.min()) / spread, 0.0, 1.0))
        else:
            # All peers are indistinguishable — neutral bonus, not maximum
            synergy_score = 0.5

        return self.synergy_alpha * sparse_reward * synergy_score

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _compute_priority(
            self,
            td_error:     float,
            social_bonus: float,
            t:            Transition,
    ) -> float:
        """
        Composite priority at push() time.  The recency penalty here is
        relative to the *sender's* baseline because we are scoring the
        intrinsic quality of the memory, not its usefulness to a specific
        requester.  Per-requester adjustment is applied in broadcast().
        """
        global_avg = float(np.mean(self._reward_window)) if self._reward_window else 0.0
        merit      = 0.2 if (t.reward + t.sparse_reward) > global_avg else 0.0
        sender_pen = self._sender_recency_penalty(t)
        return td_error + 0.5 * social_bonus + merit - sender_pen

    def _sender_recency_penalty(self, t: Transition) -> float:
        """Penalty relative to the *sender's* baseline — used at push() time."""
        baseline = self._agent_baselines.get(t.agent_id, 0.0)
        return 0.1 * max(0.0, baseline - (t.reward + t.sparse_reward))

    def _requester_recency_penalty(
            self,
            t:                  Transition,
            requester_baseline: float,
    ) -> float:
        """
        FIX (bug 2): penalty relative to the *requester's* baseline so that
        low-value memories are deprioritized from the receiver's perspective.
        """
        return 0.1 * max(0.0, requester_baseline - (t.reward + t.sparse_reward))

    def _is_stale(self, m: ScoredMemory, requester_baseline: float) -> bool:
        """
        FIX (bug 1): compare raw transition signal against the *requester's*
        baseline.  The Mediator-computed social_bonus is excluded because it
        is not part of the original environment signal and was not used when
        building _agent_baselines.
        """
        raw_signal = m.transition.reward + m.transition.sparse_reward
        return raw_signal < requester_baseline

    def _update_baselines(self, batch: List[Transition]) -> None:
        per_agent: Dict[str, List[float]] = defaultdict(list)
        for t in batch:
            per_agent[t.agent_id].append(t.reward + t.sparse_reward)

        for aid, rwds in per_agent.items():
            mean_r   = float(np.mean(rwds))
            current  = self._agent_baselines.get(aid, mean_r)
            self._agent_baselines[aid] = (
                    (1 - self._baseline_alpha) * current + self._baseline_alpha * mean_r
            )