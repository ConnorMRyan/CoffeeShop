"""
CoffeeShop PPO PPOActor  —  agents/ppo.py (Patched for Movement)
============================================================
Patches applied:
  1. Omega Reset: If sparse_rewards == 0, ω is forced to 0.5 (prevents stubbornness).
  2. Entropy Spike: If value_variance is near zero (stalled), c_ent is boosted
     to force thermal agitation and break the "vibration trap."
"""

from __future__ import annotations
import math
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from core_marl.mediator import CoffeeShopMediator, ScoredMemory, Transition

# ---------------------------------------------------------------------------
# Actor-Critic Network
# ---------------------------------------------------------------------------

class ActorCriticNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
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
        h      = self.trunk(obs)
        logits = self.actor_head(h)
        value  = self.critic_head(h).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(obs)
        dist          = Categorical(logits=logits)
        action        = dist.sample()
        log_prob      = dist.log_prob(action)
        return action, log_prob, value

# ---------------------------------------------------------------------------
# Learned Openness (ω)
# ---------------------------------------------------------------------------

class LearnedOpenness(nn.Module):
    def __init__(self, init_omega: float = 0.5):
        super().__init__()
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
            value_variance: float,
            lr: float = 0.01,
            high_var_threshold: float = 0.05,
    ) -> None:
        with torch.no_grad():
            if value_variance > high_var_threshold:
                self._raw.add_(lr)
            else:
                self._raw.sub_(lr)
            self._raw.clamp_(-2.944, 2.944)

# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    def __init__(self):
        self.env_ids: List[int] = []
        self.obs: List[torch.Tensor] = []
        self.global_obs: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.sparse_rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []

    def add(self, env_id, obs, global_obs, action, log_prob, reward, sparse_reward, value, done):
        self.env_ids.append(env_id)
        self.obs.append(obs)
        self.global_obs.append(global_obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.sparse_rewards.append(sparse_reward)
        self.values.append(value)
        self.dones.append(done)

    def to_transitions(self, agent_id: str) -> List[Transition]:
        return [Transition(agent_id=agent_id, env_id=eid, obs=o, global_obs=g, action=a,
                           log_prob=lp, reward=r, sparse_reward=sr, value_est=v, done=d)
                for eid, o, g, a, lp, r, sr, v, d in zip(
                self.env_ids, self.obs, self.global_obs, self.actions, self.log_probs,
                self.rewards, self.sparse_rewards, self.values, self.dones)]

    def clear(self):
        self.__init__()

# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------

class PPOAgent:
    def __init__(self, agent_id: str, obs_dim: int, action_dim: int, global_obs_dim: int,
                 mediator: CoffeeShopMediator, gamma: float = 0.99, lam: float = 0.95,
                 clip_eps: float = 0.2, c_vf: float = 0.5, c_ent: float = 0.05,
                 ppo_epochs: int = 4, mini_batch_size: int = 64, lr: float = 3e-4,
                 push_every: int = 128, pull_every: int = 2, device: str = "cpu"):
        self.agent_id = agent_id
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.c_vf = c_vf
        self.base_c_ent = c_ent # Store the base entropy
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.push_every = push_every
        self.pull_every = pull_every
        self.device = torch.device(device)

        self.ac = ActorCriticNet(obs_dim, action_dim).to(self.device)
        self.openness = LearnedOpenness(init_omega=0.5)
        self.optimizer = torch.optim.Adam(
            list(self.ac.parameters()) + list(self.openness.parameters()),
            lr=lr, eps=1e-5
        )

        self.mediator = mediator
        self.buffer = RolloutBuffer()
        self._step_count = 0
        self._update_count = 0

    def step(self, env_id: int, obs: torch.Tensor, global_obs: torch.Tensor,
             reward: float, sparse_reward: float, done: bool) -> torch.Tensor:
        obs = obs.to(self.device)
        global_obs = global_obs.to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.ac.act(obs)

        self.buffer.add(env_id, obs, global_obs, action, log_prob, reward, sparse_reward, value.item(), done)
        self._step_count += 1
        if self._step_count % self.push_every == 0:
            self._push_to_mediator()
        return action

    def update(self) -> dict:
        transitions = self.buffer.to_transitions(self.agent_id)
        if not transitions: return {}

        obs_t = torch.stack([t.obs for t in transitions]).to(self.device)
        actions_t = torch.stack([t.action for t in transitions]).to(self.device)
        old_lp_t = torch.stack([t.log_prob for t in transitions]).to(self.device)
        values_np = np.array([t.value_est for t in transitions])
        rewards_np = np.array([t.reward for t in transitions])
        sparse_rewards_np = np.array([t.sparse_reward for t in transitions])
        dones_np = np.array([t.done for t in transitions])

        advantages, returns = self._compute_gae(values_np, rewards_np, dones_np)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # ── PATCH 1: SKEPTICISM RESET ─────────────────────────────────────
        # If we haven't found a sparse reward (soup), don't allow "Stubbornness"
        value_variance = float(np.var(values_np))
        if np.sum(sparse_rewards_np) <= 0:
            self.openness.force_reset()
        else:
            self.openness.update_from_variance(value_variance)

        # ── PATCH 2: DYNAMIC ENTROPY SPIKE ────────────────────────────────
        # If value variance is near-zero, we are stalled. Triple the entropy!
        current_c_ent = self.base_c_ent
        if value_variance < 0.001:
            current_c_ent *= 3.0

        peer_memories = []
        if self._update_count % self.pull_every == 0:
            peer_memories = self.mediator.broadcast(self.agent_id, self.openness.value)

        n = len(transitions)
        idx = np.arange(n)
        metrics = {"policy_loss": [], "value_loss": [], "entropy": [], "bc_loss": []}

        for _ in range(self.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.mini_batch_size):
                mb = idx[start : start + self.mini_batch_size]
                if len(mb) == 0: continue
                logits, values = self.ac(obs_t[mb])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions_t[mb])
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - old_lp_t[mb].detach())
                surr1 = ratio * advantages_t[mb]
                surr2 = torch.clamp(ratio, 1.0-self.clip_eps, 1.0+self.clip_eps) * advantages_t[mb]
                L_clip = -torch.min(surr1, surr2).mean()
                L_vf = F.mse_loss(values, returns_t[mb])
                L_bc = self._behavior_cloning_loss(peer_memories)

                omega = self.openness()
                L_total = L_clip + self.c_vf * L_vf - current_c_ent * entropy + omega * L_bc

                self.optimizer.zero_grad()
                L_total.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=0.5)
                self.optimizer.step()

                metrics["policy_loss"].append(L_clip.item()); metrics["value_loss"].append(L_vf.item())
                metrics["entropy"].append(entropy.item()); metrics["bc_loss"].append(L_bc.item())

        self.buffer.clear()
        self._update_count += 1
        return {k: float(np.mean(v)) for k, v in metrics.items()} | {"omega": self.openness.value, "value_variance": value_variance}

    def _behavior_cloning_loss(self, peer_memories: List[ScoredMemory]) -> torch.Tensor:
        if not peer_memories: return torch.tensor(0.0, device=self.device, requires_grad=True)
        priorities = torch.tensor([m.priority for m in peer_memories], dtype=torch.float32)
        weights = F.softmax(priorities, dim=0).to(self.device)
        peer_obs = torch.stack([m.transition.obs for m in peer_memories]).to(self.device)
        peer_actions = torch.stack([m.transition.action for m in peer_memories]).to(self.device)
        logits, _ = self.ac(peer_obs)
        return -(weights * Categorical(logits=logits).log_prob(peer_actions)).sum()

    def _compute_gae(self, values, rewards, dones, last_value=0.0):
        n = len(rewards); advantages = np.zeros(n, dtype=np.float32); gae = 0.0
        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else values[t + 1]
            next_done = 1.0 if (t == n - 1 or dones[t]) else 0.0
            delta = rewards[t] + self.gamma * next_val * (1.0 - next_done) - values[t]
            gae = delta + self.gamma * self.lam * (1.0 - next_done) * gae
            advantages[t] = gae
        return advantages, advantages + values

    def _push_to_mediator(self) -> None:
        for t in self.buffer.to_transitions(self.agent_id): self.mediator.push(t)