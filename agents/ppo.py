from __future__ import annotations

import math
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from core_marl.memory import Transition, ScoredMemory
from core_marl.mediator import CoffeeShopMediator
from models.common import NatureCNN


# ---------------------------------------------------------------------------
# Actor-Critic Network
# ---------------------------------------------------------------------------

class ActorCriticNet(nn.Module):
    """
    Shared-trunk actor-critic network.
    """

    def __init__(
            self,
            obs_dim:    int,
            action_dim: int,
            hidden:     int                              = 128,
            encoder:    str                              = "mlp",
            img_shape:  Optional[Tuple[int, int, int]]  = None,
    ):
        super().__init__()
        self._encoder_type = encoder
        self._img_shape    = img_shape

        if encoder == "cnn":
            if img_shape is None:
                raise ValueError("img_shape=(C,H,W) is required when encoder='cnn'")
            C, H, W = img_shape
            self.cnn = NatureCNN(C, H, W)
            cnn_out = self.cnn.output_dim
            self.trunk = nn.Sequential(
                self.cnn,
                nn.Linear(cnn_out, hidden),
                nn.LayerNorm(hidden),
                nn.Tanh(),
            )
        else:
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.LayerNorm(hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )

        self.actor_head  = nn.Linear(hidden, action_dim)
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # obs: [B, C, H, W] if CNN, [B, D] if MLP
        h      = self.trunk(obs)                  # [B, hidden]
        logits = self.actor_head(h)               # [B, action_dim]
        value  = self.critic_head(h).squeeze(-1)  # [B]
        return logits, value

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # obs: [1, C, H, W] or [1, D]
        logits, value = self(obs)                 # logits: [1, action_dim], value: [1]
        dist          = Categorical(logits=logits)
        action        = dist.sample()             # [1]
        log_prob      = dist.log_prob(action)     # [1]
        return action, log_prob, value




# ---------------------------------------------------------------------------
# Learned Openness (ω)
# ---------------------------------------------------------------------------

class LearnedOpenness(nn.Module):
    def __init__(self, init_omega: float = 0.5):
        super().__init__()
        # Clamp init_omega to avoid log(0) or division by zero
        init_omega = max(min(init_omega, 0.999), 0.001)
        self._raw = nn.Parameter(torch.tensor(math.log(init_omega / (1 - init_omega))))

    @property
    def value(self) -> float:
        return torch.sigmoid(self._raw).item()

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self._raw)

    def force_reset(self):
        """Reset omega to 0.5 to force social listening."""
        with torch.no_grad():
            self._raw.fill_(0.0)

    def update_from_variance(
            self,
            value_variance:        float,
            critic_loss:           float = 0.0,
            critic_loss_threshold: float = 2.0,
            lr:                    float = 0.01,
            high_var_threshold:    float = 0.05,
    ) -> None:
        """
        Adjust omega based on value variance and mediator critic quality.

        Quality scales the upward nudge:
          critic_loss == 0         → quality 1.0, full upward nudge
          critic_loss >= threshold → quality 0.0, no upward nudge

        Downward nudge is unscaled: omega always closes freely when
        value_variance is low.
        """
        with torch.no_grad():
            quality = max(0.0, 1.0 - critic_loss / max(critic_loss_threshold, 1e-8))
            # Hardening: check for NaN/Inf in quality
            if not math.isfinite(quality):
                quality = 0.0

            if value_variance > high_var_threshold:
                self._raw.add_(lr * quality)
            else:
                self._raw.sub_(lr)
            # clamp: [logit(0.05), logit(0.667)]
            self._raw.clamp_(-2.944, 0.693)


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self):
        self.env_ids:         List[int]          = []
        self.obs:             List[torch.Tensor] = []
        self.global_obs:      List[torch.Tensor] = []
        self.next_global_obs: List[torch.Tensor] = []
        self.actions:         List[torch.Tensor] = []
        self.log_probs:       List[torch.Tensor] = []
        self.rewards:         List[float]        = []
        self.sparse_rewards:  List[float]        = []
        self.values:          List[float]        = []
        self.dones:           List[bool]         = []
        self.truncateds:      List[bool]         = []

        self._pending: Optional[tuple] = None
        self._lock = threading.Lock()

    def record_action(
            self,
            env_id:     int,
            obs:        torch.Tensor,
            global_obs: torch.Tensor,
            action:     torch.Tensor,
            log_prob:   torch.Tensor,
            value:      float,
    ) -> None:
        with self._lock:
            self._pending = (env_id, obs, global_obs, action, log_prob, value)

    def record_outcome(
            self,
            next_global_obs: torch.Tensor,
            reward:          float,
            sparse_reward:   float,
            done:            bool,
            truncated:       bool,
    ) -> None:
        with self._lock:
            if self._pending is None:
                # In a multi-threaded env, this might happen if two threads
                # call record_outcome without their own record_action first.
                return
            
            env_id, obs, global_obs, action, log_prob, value = self._pending
            self._pending = None

            self.env_ids.append(env_id)
            self.obs.append(obs)
            self.global_obs.append(global_obs)
            self.next_global_obs.append(next_global_obs)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.rewards.append(reward)
            self.sparse_rewards.append(sparse_reward)
            self.values.append(value)
            self.dones.append(done)
            self.truncateds.append(truncated)

    def to_transitions(self, agent_id: str) -> List[Transition]:
        with self._lock:
            return [
                Transition(
                    agent_id        = agent_id,
                    env_id          = eid,
                    obs             = o,
                    global_obs      = g,
                    next_global_obs = ng,
                    action          = a,
                    log_prob        = lp,
                    reward          = r,
                    sparse_reward   = sr,
                    value_est       = v,
                    done            = d,
                    truncated       = tr,
                )
                for eid, o, g, ng, a, lp, r, sr, v, d, tr in zip(
                    self.env_ids, self.obs, self.global_obs, self.next_global_obs,
                    self.actions, self.log_probs, self.rewards, self.sparse_rewards,
                    self.values, self.dones, self.truncateds,
                )
            ]

    def __len__(self) -> int:
        with self._lock:
            return len(self.rewards)

    def clear(self):
        with self._lock:
            self.env_ids.clear()
            self.obs.clear()
            self.global_obs.clear()
            self.next_global_obs.clear()
            self.actions.clear()
            self.log_probs.clear()
            self.rewards.clear()
            self.sparse_rewards.clear()
            self.values.clear()
            self.dones.clear()
            self.truncateds.clear()
            self._pending = None


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------

class PPOAgent:
    def __init__(
            self,
            agent_id:        str,
            obs_dim:         int,
            action_dim:      int,
            global_obs_dim:  int,
            mediator:        CoffeeShopMediator,
            gamma:           float = 0.99,
            lam:             float = 0.95,
            clip_eps:        float = 0.2,
            c_vf:            float = 0.5,
            c_ent:           float = 0.05,
            ppo_epochs:      int   = 4,
            mini_batch_size: int   = 64,
            lr:              float = 3e-4,
            push_every:      int   = 128,
            pull_every:      int   = 2,
            hidden:          int   = 128,
            encoder:         str   = "mlp",
            img_shape:       Optional[Tuple[int, int, int]] = None,
            device:          str   = "cpu",
            vanilla:         bool  = False,
    ):
        self.agent_id        = agent_id
        self.gamma           = gamma
        self.lam             = lam
        self.clip_eps        = clip_eps
        self.c_vf            = c_vf
        self.base_c_ent      = c_ent
        self.ppo_epochs      = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.push_every      = push_every
        self.pull_every      = pull_every
        self.device          = torch.device(device)
        self.vanilla         = vanilla

        self.ac       = ActorCriticNet(
            obs_dim, action_dim,
            hidden    = hidden,
            encoder   = encoder,
            img_shape = img_shape,
        ).to(self.device)
        self.openness  = LearnedOpenness(init_omega=0.5)
        # Vanilla mode: optimizer covers only the actor-critic network.
        # Social mode: also includes openness so ω is learned end-to-end.
        opt_params = (
            self.ac.parameters() if vanilla
            else list(self.ac.parameters()) + list(self.openness.parameters())
        )
        self.optimizer = torch.optim.Adam(opt_params, lr=lr, eps=1e-5)

        self.mediator      = mediator
        self.buffer        = RolloutBuffer()
        self._step_count   = 0
        self._update_count = 0

        self._variance_history: deque[float] = deque(maxlen=50)

        # FIX (bug 8d): written by train.py after each mediator critic update.
        # Default 1.0 sits below the 2.0 threshold so early updates behave
        # normally while the critic warms up.
        self._last_critic_loss: float = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(
            self,
            env_id:     int,
            obs:        torch.Tensor,
            global_obs: torch.Tensor,
    ) -> torch.Tensor:
        # Tensors are now moved to device in the training loop for batching performance
        with torch.no_grad():
            action, log_prob, value = self.ac.act(obs)

        self.buffer.record_action(env_id, obs, global_obs, action, log_prob, value.item())
        return action

    def observe_outcome(
            self,
            next_global_obs: torch.Tensor,
            reward:          float,
            sparse_reward:   float,
            done:            bool,
            truncated:       bool,
    ) -> None:
        # Tensors are now moved to device in the training loop for batching performance
        self.buffer.record_outcome(next_global_obs, reward, sparse_reward, done, truncated)

        self._step_count += 1
        if not self.vanilla and self._step_count % self.push_every == 0:
            self._push_to_mediator()

    def update(self, last_obs: Optional[torch.Tensor] = None) -> dict:
        transitions = self.buffer.to_transitions(self.agent_id)
        if not transitions:
            return {}

        obs_t        = torch.stack([t.obs        for t in transitions]).to(self.device)
        actions_t    = torch.stack([t.action     for t in transitions]).to(self.device)
        old_lp_t     = torch.stack([t.log_prob   for t in transitions]).to(self.device)
        values_np    = np.array([t.value_est     for t in transitions])
        rewards_np   = np.array([t.reward        for t in transitions])
        sparse_np    = np.array([t.sparse_reward for t in transitions])
        dones_np     = np.array([t.done      for t in transitions], dtype=bool)
        truncated_np = np.array([t.truncated for t in transitions], dtype=bool)

        last_value = 0.0
        if last_obs is not None:
            with torch.no_grad():
                _, last_val_t = self.ac(last_obs.to(self.device).unsqueeze(0))
                last_value = last_val_t.item()

        advantages, returns = self._compute_gae(
            values_np, rewards_np, dones_np, truncated_np, last_value,
        )
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t    = torch.tensor(returns,    dtype=torch.float32, device=self.device)

        # Hardening: handle NaN/Inf in advantages/returns
        if not torch.isfinite(advantages_t).all():
            advantages_t = torch.nan_to_num(advantages_t, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(returns_t).all():
            returns_t = torch.nan_to_num(returns_t, nan=0.0, posinf=0.0, neginf=0.0)

        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1e-8)

        value_variance = float(np.var(values_np))
        if not math.isfinite(value_variance):
            value_variance = 0.0
        self._variance_history.append(value_variance)

        if self.vanilla:
            # Pure PPO: no social updates, no openness, no peer memories.
            pass
        else:
            no_delivery     = np.sum(sparse_np) <= 0
            genuinely_stuck = value_variance < 1e-4
            if no_delivery and genuinely_stuck:
                self.openness.force_reset()
            else:
                # FIX (bug 8a): pass critic loss so upward nudge is quality-gated.
                self.openness.update_from_variance(
                    value_variance,
                    critic_loss=self._last_critic_loss,
                )

        current_c_ent = self.base_c_ent
        if len(self._variance_history) >= 10:
            median_var = float(np.median(self._variance_history))
            threshold  = max(1e-8, 0.1 * median_var)
            if value_variance < threshold:
                current_c_ent *= 3.0

        peer_memories  = []
        critic_quality = 1.0
        if not self.vanilla:
            if self._update_count % self.pull_every == 0:
                peer_memories = self.mediator.broadcast(self.agent_id, self.openness.value)
            # FIX: Q is now derived from the mediator's global state directly if available,
            # but we fallback to a safe default if not provided to this agent.
            # Using mediator's gossip factor as a proxy for Q if we want local sensitivity.
            critic_quality = self.mediator.get_verifiable_trust(getattr(self, "_last_critic_loss", 1.0))

        n   = len(transitions)
        idx = np.arange(n)
        metrics: dict = {
            "policy_loss": [], "value_loss": [], "entropy": [],
            "grad_norm_actor": [], "grad_norm_critic": []
        }
        if not self.vanilla:
            metrics["bc_loss"]        = []
            metrics["critic_quality"] = []

        for _ in range(self.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.mini_batch_size):
                mb = idx[start : start + self.mini_batch_size]
                if len(mb) == 0:
                    continue

                logits, values = self.ac(obs_t[mb])
                dist           = Categorical(logits=logits)
                new_log_probs  = dist.log_prob(actions_t[mb])
                entropy        = dist.entropy().mean()

                ratio  = torch.exp(new_log_probs - old_lp_t[mb].detach())
                surr1  = ratio * advantages_t[mb]
                surr2  = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_t[mb]
                L_clip = -torch.min(surr1, surr2).mean()
                L_vf   = F.mse_loss(values, returns_t[mb])

                if self.vanilla:
                    L_total = L_clip + self.c_vf * L_vf - current_c_ent * entropy
                    clip_params = list(self.ac.parameters())
                else:
                    L_bc            = self._behavior_cloning_loss(peer_memories)
                    omega_mask      = self.mediator.get_omega_mask(self.agent_id)
                    effective_omega = self.openness() * omega_mask
                    # critic_quality gates mediator influence.
                    L_total = (
                            L_clip
                            + self.c_vf * L_vf
                            - current_c_ent * entropy
                            + effective_omega * critic_quality * L_bc
                    )
                    clip_params = list(self.ac.parameters()) + list(self.openness.parameters())

                self.optimizer.zero_grad(set_to_none=True)
                L_total.backward()

                # Calculate grad norms before clipping
                gn_actor = nn.utils.clip_grad_norm_(self.ac.actor_head.parameters(), max_norm=1e9)
                gn_critic = nn.utils.clip_grad_norm_(self.ac.critic_head.parameters(), max_norm=1e9)
                metrics["grad_norm_actor"].append(gn_actor.item())
                metrics["grad_norm_critic"].append(gn_critic.item())

                nn.utils.clip_grad_norm_(clip_params, max_norm=0.5)
                self.optimizer.step()

                metrics["policy_loss"].append(L_clip.item())
                metrics["value_loss"].append(L_vf.item())
                metrics["entropy"].append(entropy.item())
                if not self.vanilla:
                    metrics["bc_loss"].append(L_bc.item())
                    metrics["critic_quality"].append(critic_quality)

        self.buffer.clear()
        self._update_count += 1
        return (
                {k: float(np.mean(v)) for k, v in metrics.items()}
                | {"omega": self.openness.value, "value_variance": value_variance}
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _behavior_cloning_loss(self, peer_memories: List[ScoredMemory]) -> torch.Tensor:
        if not peer_memories:
            return torch.zeros(1, device=self.device).squeeze()

        priorities   = torch.tensor([m.priority for m in peer_memories], dtype=torch.float32)
        weights      = F.softmax(priorities, dim=0).to(self.device)
        peer_obs     = torch.stack([m.transition.obs    for m in peer_memories]).to(self.device)
        peer_actions = torch.stack([m.transition.action for m in peer_memories]).to(self.device)
        logits, _    = self.ac(peer_obs)
        return -(weights * Categorical(logits=logits).log_prob(peer_actions)).mean()

    def _compute_gae(
            self,
            values:     np.ndarray,
            rewards:    np.ndarray,
            dones:      np.ndarray,
            truncateds: np.ndarray,
            last_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n          = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        gae        = 0.0

        for t in reversed(range(n)):
            if t == n - 1:
                next_val  = last_value
                next_term = float(dones[t])
            else:
                next_val  = values[t + 1]
                next_term = float(dones[t])

            if truncateds[t]:
                next_term = 0.0
                gae       = 0.0

            delta         = rewards[t] + self.gamma * next_val * (1.0 - next_term) - values[t]
            gae           = delta + self.gamma * self.lam * (1.0 - next_term) * gae
            advantages[t] = gae

        return advantages, advantages + values

    def _push_to_mediator(self) -> None:
        self.mediator.batch_push(self.buffer.to_transitions(self.agent_id))