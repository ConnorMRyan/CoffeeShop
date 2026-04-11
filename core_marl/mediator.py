from __future__ import annotations

import copy
import heapq
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import threading
from core_marl.memory import Transition, ScoredMemory, PrioritizedBuffer
from models.common import NatureCNN


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Centralized Value Network
# ---------------------------------------------------------------------------

class CentralizedCritic(nn.Module):
    """
    V(s_global) estimator for TD-error calculations.
    """

    def __init__(
            self,
            global_obs_dim: int,
            hidden:         int              = 1024,
            encoder:        str              = "mlp",
            img_shape:      Optional[Tuple[int, int, int]] = None,
    ):
        super().__init__()

        if encoder == "cnn":
            if img_shape is None:
                raise ValueError("img_shape=(C,H,W) is required when encoder='cnn'")
            C, H, W = img_shape
            self.cnn = NatureCNN(C, H, W)
            cnn_out = self.cnn.output_dim
            self.net = nn.Sequential(
                self.cnn,
                nn.Linear(cnn_out, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(global_obs_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        # global_obs: [B, D_global] or [B, C, H, W]
        return self.net(global_obs).squeeze(-1)   # [B]




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
            critic_lr:       float = 5e-4,
            critic_hidden:   int   = 1024,
            encoder:         str   = "mlp",
            img_shape:       Optional[Tuple[int, int, int]] = None,
            device:          str   = "cpu",
            diversity_threshold: float = 0.15,
            timeout_duration:    int   = 500,
            tau:                 float = 5.0,
            tau_min:             float = 0.5,
            baseline_max_jump:   float = 2.0,
            target_tau:          float = 0.005,
    ):
        self.gamma         = gamma
        self.epsilon_td    = epsilon_td
        self.synergy_alpha = synergy_alpha
        self.tau_initial   = tau
        self.tau_min       = tau_min
        self.target_tau    = target_tau
        self.baseline_max_jump = baseline_max_jump
        self.device        = torch.device(device)

        self.diversity_threshold = diversity_threshold
        self.timeout_duration    = timeout_duration
        self._timeouts: Dict[str, int] = {}

        # ── Centralized critic, frozen target copy, and optimizer ─────────
        self.critic     = CentralizedCritic(
                              global_obs_dim,
                              hidden    = critic_hidden,
                              encoder   = encoder,
                              img_shape = img_shape,
                          ).to(self.device)
        # Target network: a periodically-lagged copy used for stable TD
        # bootstrap targets.  Without it the critic trains against its own
        # moving predictions, causing the "deadly triad" divergence seen as
        # monotonically growing critic loss in the training logs.
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ── Shared prioritized buffer ─────────────────────────────────────
        self._buffer = PrioritizedBuffer(buffer_capacity)
        self._value_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))

        # ── Global Gossip & Skepticism Tracking ───────────────────────────
        self._reward_window: deque[float] = deque(maxlen=100)
        self._env_reward_windows: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))

        self._atb_reward: float = -float("inf")
        self._gossip_alpha      = 0.7

        # ── Population baseline tracker ───────────────────────────────────
        self._agent_baselines: Dict[str, float] = {}
        self._baseline_alpha  = 0.05

        # ── Agent last-seen env (for cross-env reputation boosts) ─────────
        self._agent_last_env: Dict[str, int] = {}

        # ── Diagnostics ───────────────────────────────────────────────────
        self._last_broadcast_stats: Dict[str, Any] = {}
        self._rng = np.random.default_rng()
        self._lock = threading.Lock()

    def get_priority_stats(self) -> Dict[str, float]:
        """Returns statistics of the current priorities in the buffer."""
        with self._lock:
            if not self._buffer:
                return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
            
            priorities = np.array([m.priority for m in self._buffer])
        return {
            "min": float(np.min(priorities)),
            "max": float(np.max(priorities)),
            "mean": float(np.mean(priorities)),
            "median": float(np.median(priorities)),
        }

    # ------------------------------------------------------------------
    # Gossip Logic — Performance Monitoring
    # ------------------------------------------------------------------

    def get_gossip_factor(self) -> float:
        """
        Returns a multiplier in (0, 1] to reduce agent trust when the group
        is underperforming relative to its all-time-best rolling average.
        """
        with self._lock:
            if len(self._reward_window) < 25:
                return 1.0

            current_avg = float(np.mean(self._reward_window))
            atb = self._atb_reward

        # Smooth step function: gossip factor transitions from gossip_alpha to 1.0
        # as current_avg approaches 0.7 * atb.
        threshold = 0.7 * atb
        if current_avg < threshold:
            # If current_avg is much lower than threshold, factor is gossip_alpha.
            # We can use a sigmoid or a simple linear interpolation for a "smooth step".
            # For now, let's just make it a bit more continuous.
            return self._gossip_alpha
        
        return 1.0

    # ------------------------------------------------------------------
    # Public API — called by SocialActors
    # ------------------------------------------------------------------

    def push(self, transition: Transition) -> None:
        """Score and store a transition."""
        with self._lock:
            self._value_history[transition.agent_id].append(transition.value_est)
            self._agent_last_env[transition.agent_id] = transition.env_id

            total_r = transition.reward + transition.sparse_reward
            self._reward_window.append(total_r)
            self._env_reward_windows[transition.env_id].append(total_r)

            if self._reward_window:
                current_global_avg = float(np.mean(self._reward_window))
                self._atb_reward = max(self._atb_reward, current_global_avg)

        td_error, social_bonus = self._compute_td_and_bonus(transition)
        priority = self._compute_priority(td_error, social_bonus, transition)

        with self._lock:
            self._buffer.push(ScoredMemory(transition, td_error, social_bonus, priority))

    def batch_push(self, transitions: List[Transition]) -> None:
        """
        Score and store a batch of transitions with two batched critic forward
        passes instead of 2*N individual ones.
        """
        if not transitions:
            return

        with self._lock:
            for t in transitions:
                self._value_history[t.agent_id].append(t.value_est)
                self._agent_last_env[t.agent_id] = t.env_id
                total_r = t.reward + t.sparse_reward
                self._reward_window.append(total_r)
                self._env_reward_windows[t.env_id].append(total_r)

            if self._reward_window:
                self._atb_reward = max(self._atb_reward, float(np.mean(self._reward_window)))

        g_obs      = torch.stack([t.global_obs      for t in transitions]).to(self.device, non_blocking=True)
        next_g_obs = torch.stack([t.next_global_obs for t in transitions]).to(self.device, non_blocking=True)

        with torch.no_grad():
            v_s_all    = self.critic(g_obs)
            v_next_all = self.critic(next_g_obs)

        v_s_np = v_s_all.cpu().numpy()
        v_next_np = v_next_all.cpu().numpy()
        
        new_memories = []
        for i, t in enumerate(transitions):
            social_bonus = (
                self._compute_synergy_bonus(t.agent_id, t.sparse_reward)
                if t.sparse_reward > 0.0 else 0.0
            )
            v_s    = float(v_s_np[i])
            v_next = 0.0 if t.done else float(v_next_np[i])

            effective_reward = t.reward + t.sparse_reward + social_bonus
            td_error = abs(effective_reward + self.gamma * v_next - v_s)
            priority = self._compute_priority(td_error, social_bonus, t)
            new_memories.append(ScoredMemory(t, td_error, social_bonus, priority))

        with self._lock:
            for sm in new_memories:
                self._buffer.push(sm)

    def broadcast(
            self,
            requesting_agent_id: str,
            openness: float,
            n: int = 16,
    ) -> List[ScoredMemory]:
        """
        Returns high-priority memories gated by an 'Effective Openness' factor
        that combines local agent trust and global Mediator skepticism.
        """
        effective_openness = openness * self.get_gossip_factor()
        if effective_openness < 1e-4:
            return []

        with self._lock:
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

            # pass the requester's baseline into helpers so that
            # staleness and recency penalty are measured from the receiver's
            # perspective, not the sender's.
            requester_baseline = self._agent_baselines.get(requesting_agent_id, 0.0)

            # ── Single-pass scan: build candidates + diagnostics together ────
            n_total    = len(self._buffer)
            n_priority = 0
            n_not_self = 0
            candidates = []
            epsilon_td = self.epsilon_td
            for m in self._buffer:
                if m.priority < epsilon_td:
                    continue
                n_priority += 1
                if m.transition.agent_id == requesting_agent_id:
                    continue
                n_not_self += 1
                if not self._is_stale(m, requester_baseline):
                    candidates.append(m)
            n_fresh = len(candidates)

            self._last_broadcast_stats = {
                "n_total":    n_total,
                "n_priority": n_priority,
                "n_not_self": n_not_self,
                "n_fresh":    n_fresh,
                "baseline":   requester_baseline,
            }

        if not candidates:
            return []

        def adjusted_priority(m: ScoredMemory) -> float:
            p = m.priority - self._requester_recency_penalty(m.transition, requester_baseline)
            if is_underperforming and top_env is not None and m.transition.env_id == top_env:
                p *= 1.5
            return p

        # Use nlargest to avoid sorting the entire candidates list if we only need n
        if len(candidates) > n:
            top = heapq.nlargest(n, candidates, key=adjusted_priority)
        else:
            top = sorted(candidates, key=adjusted_priority, reverse=True)

        rng = self._rng
        accepted = [m for m in top if rng.random() < effective_openness]
        return accepted

    def update_critic(self, batch: List[ScoredMemory]) -> float:
        """
        Update the centralized critic with a batch of ScoredMemory entries.

        TD target: (r + social_bonus) + γ · V(s') · (1 − done)
        Uses a frozen target network for stable TD bootstrap targets.
        """
        if not batch:
            return 0.0

        g_obs      = torch.stack([sm.transition.global_obs      for sm in batch]).to(self.device, non_blocking=True)
        next_g_obs = torch.stack([sm.transition.next_global_obs for sm in batch]).to(self.device, non_blocking=True)
        rwds       = torch.tensor(
            [sm.transition.reward + sm.transition.sparse_reward + sm.social_bonus for sm in batch],
            dtype=torch.float32, device=self.device,
        )
        dones      = torch.tensor(
            [sm.transition.done for sm in batch],
            dtype=torch.float32, device=self.device,
        )

        with torch.no_grad():
            # Bootstrap from the frozen target network, not the live critic.
            v_next     = self.critic_target(next_g_obs)
            td_targets = rwds + self.gamma * v_next * (1.0 - dones)

        values = self.critic(g_obs)
        loss   = nn.functional.mse_loss(values, td_targets)

        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_opt.step()

        # Polyak soft-update: θ_target ← τ·θ + (1-τ)·θ_target
        with torch.no_grad():
            for p, p_tgt in zip(self.critic.parameters(),
                                self.critic_target.parameters()):
                p_tgt.data.mul_(1.0 - self.target_tau).add_(self.target_tau * p.data)

        self._update_baselines([sm.transition for sm in batch])
        return loss.item()

    # ------------------------------------------------------------------
    # Core Logic — Social Bonus & TD-Error
    # ------------------------------------------------------------------

    def _compute_td_and_bonus(self, t: Transition) -> Tuple[float, float]:
        """
        Compute (td_error, social_bonus) for a freshly pushed transition.

        The critic is called under no_grad so push() incurs no backward cost.
        Early in training the critic is randomly initialized, so TD errors will
        be noisy — they become meaningful as update_critic() trains the network.
        """
        social_bonus = 0.0
        if t.sparse_reward > 0.0:
            social_bonus = self._compute_synergy_bonus(t.agent_id, t.sparse_reward)

        with torch.no_grad():
            v_s    = self.critic(t.global_obs.unsqueeze(0).to(self.device)).item()
            v_next = (
                0.0 if t.done
                else self.critic(t.next_global_obs.unsqueeze(0).to(self.device)).item()
            )

        effective_reward = t.reward + t.sparse_reward + social_bonus
        td_target        = effective_reward + self.gamma * v_next
        td_error         = abs(td_target - v_s)

        return td_error, social_bonus

    def _compute_synergy_bonus(self, triggering_agent: str, sparse_reward: float) -> float:
        """
        Reward the triggering agent for achieving a sparse event when peers
        are performing well relative to one another.
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
        Penalty relative to the *requester's* baseline so that
        low-value memories are deprioritized from the receiver's perspective.
        """
        return 0.1 * max(0.0, requester_baseline - (t.reward + t.sparse_reward))

    def _is_stale(self, m: ScoredMemory, requester_baseline: float) -> bool:
        """
        Compare raw transition signal against the *requester's* baseline.
        The Mediator-computed social_bonus is excluded because it is not
        part of the original environment signal.
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

            # Clamped baseline update
            if mean_r > current:
                # 1. Std-based clamping (if enough samples)
                if len(rwds) >= 4:
                    std_r = float(np.std(rwds))
                    mean_r = min(mean_r, current + std_r)
                
                # 2. Hard jump cap (baseline_max_jump)
                # Use abs(current) so baselines near or at zero can still grow.
                jump_cap = max(abs(current) * self.baseline_max_jump, 0.01)
                mean_r = min(mean_r, current + jump_cap)

            self._agent_baselines[aid] = (
                    (1 - self._baseline_alpha) * current + self._baseline_alpha * mean_r
            )

    def get_verifiable_trust(self, td_error: float) -> float:
        """
        Calculate Verifiable Trust (Q) using a dynamically cooled temperature.
        As the population's All-Time Best (ATB) reward climbs, tau shrinks,
        forcing agents to demand higher precision from the Mediator.
        """
        # Scale tau based on how close the current ATB is to the "solved" 1.0 ceiling
        # If ATB is 0.0, tau = tau_initial. If ATB is 1.0+, tau = tau_min.
        with self._lock:
            current_atb = max(0.0, self._atb_reward)
        cooling_factor = np.clip(current_atb, 0.0, 1.0)

        dynamic_tau = self.tau_initial - cooling_factor * (self.tau_initial - self.tau_min)
        dynamic_tau = max(dynamic_tau, 1e-8)

        exponent = np.clip(-td_error / dynamic_tau, -20.0, 0.0)
        return float(np.exp(exponent))

    def step_timeouts(self) -> None:
        """Decrement and clear expired timeouts."""
        with self._lock:
            expired = []
            for aid in list(self._timeouts.keys()):
                self._timeouts[aid] -= 1
                if self._timeouts[aid] <= 0:
                    expired.append(aid)
            for aid in expired:
                del self._timeouts[aid]

    def get_omega_mask(self, agent_id: str) -> float:
        """Returns 0.0 if the agent is in _timeouts, otherwise 1.0."""
        with self._lock:
            return 0.0 if agent_id in self._timeouts else 1.0

    def enforce_diversity(
            self,
            current_diversity: float,
            actors:            Dict[str, Any],
    ) -> Optional[str]:
        """
        Upgraded Sabbatical System: probabilistic lockout of agents who have
        high trust (omega) but low individual diversity contribution.
        """
        self.step_timeouts()

        # 1. Sigmoid-based global trigger
        # Probability of anyone being timed out is high if diversity is low.
        # Prob = sigmoid(10 * (threshold - current_diversity))
        # Clamped to [-5, 5] so extreme diversity values don't cause saturation
        # (without clamping, near-zero diversity gives argument ~+100, locking
        # out agents on every single step).
        arg          = torch.tensor(10.0 * (self.diversity_threshold - current_diversity)).clamp_(-5.0, 5.0)
        trigger_prob = float(torch.sigmoid(arg).item())
        
        if np.random.random() > trigger_prob:
            return None

        # 2. Only sample if no one is currently on timeout (to avoid population-wide lockout)
        with self._lock:
            if self._timeouts:
                return None

        aids = list(actors.keys())
        if len(aids) < 2:
            return None

        # 3. Calculate Agent-Specific Diversity (JS relative to population mean)
        # We need a probe batch for this. We sample from the buffer.
        if len(self._buffer) < 32:
            return None

        memories  = self._buffer.sample(32)
        obs_batch = torch.stack([m.transition.obs for m in memories]).to(self.device)

        agent_probs = {}
        with torch.no_grad():
            for aid, actor in actors.items():
                logits, _ = actor.ac(obs_batch)
                # Clamp logits before softmax so the output remains a valid
                # probability simplex (sum=1).  Clamping after breaks that.
                agent_probs[aid] = torch.softmax(logits.clamp(-50.0, 50.0), dim=-1)

        # Population mean distribution
        all_probs = torch.stack(list(agent_probs.values()))  # [num_agents, batch, action_dim]
        mean_prob = all_probs.mean(dim=0)  # [batch, action_dim]

        def kl_div(p, q):
            return (p * (torch.log(p) - torch.log(q))).sum(dim=-1)

        agent_js = {}
        for aid, p in agent_probs.items():
            m = 0.5 * (p + mean_prob)
            js = 0.5 * kl_div(p, m) + 0.5 * kl_div(mean_prob, m)
            agent_js[aid] = js.mean().item()

        # 4. Compute Sabbatical Probabilities
        # W_i = omega_i / (JS_i + epsilon)
        weights = []
        eps = 1e-6
        for aid in aids:
            omega = float(actors[aid].openness.value)
            js_i  = agent_js[aid]
            weights.append(omega / (js_i + eps))

        weights_arr = np.array(weights)
        if weights_arr.sum() < 1e-9:
            # All weights zero or near-zero, pick uniformly
            probs = np.ones_like(weights_arr) / len(weights_arr)
        else:
            probs = weights_arr / weights_arr.sum()

        # 5. Sample one agent
        target_idx = np.random.choice(len(aids), p=probs)
        target_aid = aids[target_idx]

        with self._lock:
            self._timeouts[target_aid] = self.timeout_duration
        return target_aid