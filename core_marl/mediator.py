from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from envs import SocialEnvWrapper
from core_marl.experience_buffer import ExperienceBatch


@dataclass
class MediatorConfig:
    """Configuration for the CoffeeShop CoffeeShopMediator."""
    shared_reward: bool = True
    hidden_size: int = 64
    gamma: float = 0.99
    learning_rate: float = 1e-3


class CentralCriticNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CoffeeShopMediator:
    """Coordinates multi-agent interaction between env and agents.

    Now includes a centralized off-policy critic that evaluates shared
    experiences using TD-error.
    """

    def __init__(self, env: SocialEnvWrapper, config: MediatorConfig | None = None) -> None:
        self.env = env
        self.config = config or MediatorConfig()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.raw_rewards: Dict[str, float] = {}

        # We assume local observation spaces apply to central critic for now
        if hasattr(self.env, "obs_dim"):
            obs_dim = self.env.obs_dim
        elif hasattr(self.env, "obs_space") and hasattr(self.env.obs_space, "shape") and self.env.obs_space.shape is not None:
            obs_dim = int(np.prod(self.env.obs_space.shape))
        else:
            obs_dim = 96 # default overcooked

        self.critic = CentralCriticNetwork(obs_dim, self.config.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)

    @property
    def agent_ids(self) -> List[str]:
        return self.env.agent_ids

    def reset(self, seed: int | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self.env.reset(seed=seed)

    def step(
            self, actions: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        obs, rewards, terminated, truncated, infos = self.env.step(actions)

        # CRITICAL FIX 1: Capture raw rewards before any averaging
        self.raw_rewards = rewards.copy()

        # Inject raw rewards into the infos dict so the buffer can natively store them
        for aid in self.agent_ids:
            if aid not in infos:
                infos[aid] = {}
            infos[aid]["raw_reward"] = self.raw_rewards.get(aid, 0.0)

        if self.config.shared_reward:
            # average reward across agents
            if len(rewards) > 0:
                avg = sum(rewards.values()) / len(rewards)
                rewards = {aid: float(avg) for aid in rewards}

        return obs, rewards, terminated, truncated, infos

    def evaluate_and_prioritize(self, batch: ExperienceBatch) -> Tuple[torch.Tensor, float]:
        """Compute TD-error for a batch of shared experiences, update critic, return priorities."""
        all_obs, all_next_obs, all_rewards, all_dones = [], [], [], []

        for t in range(len(batch.observations)):
            for aid in self.agent_ids:
                if aid in batch.observations[t]:
                    all_obs.append(batch.observations[t][aid])
                    all_next_obs.append(batch.next_observations[t].get(aid, batch.observations[t][aid]))

                    # CRITICAL FIX 2: Extract the raw environment reward, fallback to batch.rewards if missing
                    raw_rew = batch.infos[t].get(aid, {}).get("raw_reward", batch.rewards[t].get(aid, 0.0))
                    all_rewards.append(raw_rew)

                    # CRITICAL FIX 3: Strict mathematical distinction between termination and truncation
                    is_term = batch.terminated[t].get(aid, False)
                    is_trunc = hasattr(batch, 'truncated') and batch.truncated[t].get(aid, False)

                    don = 1.0 if (is_term and not is_trunc) else 0.0
                    all_dones.append(don)

        if len(all_obs) == 0:
            return torch.tensor([], device=self.device), 0.0

        obs_t = torch.as_tensor(np.stack(all_obs), dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(np.stack(all_next_obs), dtype=torch.float32, device=self.device)
        rew_t = torch.as_tensor(all_rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        done_t = torch.as_tensor(all_dones, dtype=torch.float32, device=self.device).unsqueeze(-1)

        with torch.no_grad():
            next_v = self.critic(next_obs_t)
            target = rew_t + self.config.gamma * next_v * (1 - done_t)

        current_v = self.critic(obs_t)

        td_errors = torch.abs(target - current_v).detach()
        loss = nn.functional.mse_loss(current_v, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping to prevent explosion in diverse multi-agent scenarios
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.optimizer.step()

        return td_errors.squeeze(-1), loss.item()

    def close(self) -> None:
        self.env.close()