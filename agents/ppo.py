from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 64
    hidden_size: int = 64

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_size: int):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, obs: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        value = self.critic(obs)
        return dist, value

class RolloutBuffer:
    def __init__(self):
        self.obs = []
        self.acts = []
        self.rewards = []
        self.vals = []
        self.logps = []
        self.terminals = []

    def store(self, obs, act, reward, val, logp, terminal):
        self.obs.append(obs)
        self.acts.append(act)
        self.rewards.append(reward)
        self.vals.append(val)
        self.logps.append(logp)
        self.terminals.append(terminal)

    def get(self, gamma=0.99, lam=0.95, last_val=0.0):
        # Calculate GAE
        # last_val is the value of the final state (bootstrapping for truncations)
        vals = self.vals + [last_val]
        advs = np.zeros(len(self.rewards), dtype=np.float32)
        last_gaelam = 0
        for t in reversed(range(len(self.rewards))):
            nextnonterminal = 1.0 - self.terminals[t]
            delta = self.rewards[t] + gamma * vals[t+1] * nextnonterminal - vals[t]
            advs[t] = last_gaelam = delta + gamma * lam * nextnonterminal * last_gaelam
        
        rets = advs + np.array(self.vals)
        data = dict(
            obs=torch.as_tensor(np.array(self.obs, dtype=np.float32)),
            act=torch.as_tensor(np.array(self.acts, dtype=np.int64)),
            ret=torch.as_tensor(rets, dtype=torch.float32),
            adv=torch.as_tensor(advs, dtype=torch.float32),
            logp=torch.as_tensor(np.array(self.logps, dtype=np.float32)),
        )
        self.clear()
        return data
        
    def clear(self):
        self.obs.clear()
        self.acts.clear()
        self.rewards.clear()
        self.vals.clear()
        self.logps.clear()
        self.terminals.clear()
        
    def __len__(self):
        return len(self.obs)

class PPOAgent:
    """PyTorch PPO agent."""

    def __init__(self, obs_space: Any | None = None, act_space: Any | None = None, config: PPOConfig | None = None) -> None:
        self.obs_space = obs_space
        self.act_space = act_space
        self.config = config or PPOConfig()
        
        # Require valid spaces — fail fast rather than silently building wrong-shaped networks
        if self.obs_space is None or not hasattr(self.obs_space, "shape"):
            raise ValueError(
                "PPOAgent requires obs_space with a .shape attribute. "
                "Pass env.obs_space when constructing the agent."
            )
        if self.act_space is None or not hasattr(self.act_space, "n"):
            raise ValueError(
                "PPOAgent requires act_space with a .n attribute (discrete action count). "
                "Pass env.act_space when constructing the agent."
            )
        obs_dim = int(np.prod(self.obs_space.shape))
        act_dim = self.act_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(obs_dim, act_dim, self.config.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        
        # Buffer per agent_id for shared weight models
        self.buffers: Dict[str, RolloutBuffer] = collections.defaultdict(RolloutBuffer)

    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> Dict[str, Any]:
        """Return actions per agent_id."""
        actions = {}
        for aid, single_obs in obs.items():
            # Ensure we are not creating new tensors on CPU and then moving to GPU in a loop
            # single_obs is typically a numpy array from the wrapper
            o = torch.as_tensor(single_obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                dist, val = self.model(o)
                a = dist.sample() if not deterministic else dist.probs.argmax(dim=-1)
                logp = dist.log_prob(a)
            # Keep logp and val as floats to avoid tensor overhead in the dict if they are single values
            actions[aid] = {"action": a.item(), "val": val.item(), "logp": logp.item()}
        return actions

    def get_action_dist(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Get the action distribution for the given observation."""
        dists = {}
        for aid, single_obs in obs.items():
            o = torch.as_tensor(single_obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                dist, _ = self.model(o)
                dists[aid] = dist.probs.cpu().numpy()
        return dists

    def store_transition(self, aid: str, obs: Any, act: int, reward: float, val: float, logp: float, terminal: bool):
        self.buffers[aid].store(obs, act, reward, val, logp, terminal)

    def behavior_cloning_update(self, aid: str, obs_batch: torch.Tensor, act_batch: torch.Tensor, omega: Any) -> float:
        """Perform auxiliary distillation update."""
        # Assume tensors are already on the correct device and float/long
        
        # Simple BC: maximize log_prob of given actions
        # No optimizer.zero_grad() here because SocialActor might be calling this 
        # and it might want to accumulate gradients if omega is learnable.
        dist, _ = self.model(obs_batch)
        logp = dist.log_prob(act_batch)
        
        # Loss modulated by omega. If omega is a tensor, this maintains the graph.
        loss = -(logp.mean() * omega)
        loss.backward()
        # No optimizer.step() here, SocialActor or train.py handles it.
        
        return loss.item()

    def update(self, last_vals: Dict[str, float] | None = None) -> Dict[str, float]:
        """Perform a training update from accumulated buffers.
        
        last_vals: Map of agent_id to the value of the final observation in the rollout.
        """
        all_obs, all_act, all_ret, all_adv, all_logp = [], [], [], [], []
        
        total_steps = 0
        for aid, buf in self.buffers.items():
            if len(buf) == 0: continue
            
            # Use last_vals for bootstrapping if provided, otherwise 0
            last_val = last_vals.get(aid, 0.0) if last_vals else 0.0
            data = buf.get(self.config.gamma, self.config.lam, last_val=last_val)
            
            # Per-agent advantage normalization to prevent high-reward agents from dominating gradients
            adv = data['adv']
            data['adv'] = (adv - adv.mean()) / (adv.std() + 1e-8)
            
            all_obs.append(data['obs'])
            all_act.append(data['act'])
            all_ret.append(data['ret'])
            all_adv.append(data['adv'])
            all_logp.append(data['logp'])
            total_steps += len(data['obs'])
            
        if total_steps == 0:
            return {"loss_policy": 0.0, "loss_value": 0.0}

        obs = torch.cat(all_obs).to(self.device)
        act = torch.cat(all_act).to(self.device)
        ret = torch.cat(all_ret).to(self.device)
        adv = torch.cat(all_adv).to(self.device)
        logp_old = torch.cat(all_logp).to(self.device)

        # Dataset for minibatch updates
        dataset = torch.utils.data.TensorDataset(obs, act, ret, adv, logp_old)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.config.minibatch_size, shuffle=True)

        avg_pi_loss = 0.0
        avg_v_loss = 0.0
        updates = 0

        for _ in range(self.config.update_epochs):
            for b_obs, b_act, b_ret, b_adv, b_logp in loader:
                dist, val = self.model(b_obs)
                logp = dist.log_prob(b_act)
                ratio = torch.exp(logp - b_logp)
                
                clip_adv = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * b_adv
                loss_pi = -(torch.min(ratio * b_adv, clip_adv)).mean()
                loss_v = ((val.squeeze(-1) - b_ret)**2).mean() * self.config.value_coef
                loss_ent = -dist.entropy().mean() * self.config.entropy_coef
                
                loss = loss_pi + loss_v + loss_ent
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                avg_pi_loss += loss_pi.item()
                avg_v_loss += loss_v.item()
                updates += 1

        return {
            "loss_policy": avg_pi_loss / max(1, updates),
            "loss_value": avg_v_loss / max(1, updates)
        }
