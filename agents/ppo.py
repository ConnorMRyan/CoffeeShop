from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tensordict import TensorDict, MemoryMappedTensor

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
    def __init__(self, capacity: int = 2048, obs_shape: tuple = (96,), act_dim: int = 1):
        self.capacity = capacity
        # Use MemoryMappedTensor for high-throughput zero-copy sharing
        self.data = TensorDict({
            "obs": MemoryMappedTensor.empty((capacity, *obs_shape), dtype=torch.float32),
            "act": MemoryMappedTensor.empty((capacity,), dtype=torch.long),
            "rew": MemoryMappedTensor.empty((capacity,), dtype=torch.float32),
            "val": MemoryMappedTensor.empty((capacity,), dtype=torch.float32),
            "logp": MemoryMappedTensor.empty((capacity,), dtype=torch.float32),
            "term": MemoryMappedTensor.empty((capacity,), dtype=torch.bool),
        }, batch_size=[capacity])
        self.ptr = 0

    def store(self, obs, act, reward, val, logp, terminal):
        if self.ptr >= self.capacity:
            return # Full
        self.data["obs"][self.ptr] = torch.as_tensor(obs)
        self.data["act"][self.ptr] = torch.as_tensor(act)
        self.data["rew"][self.ptr] = torch.as_tensor(reward)
        self.data["val"][self.ptr] = torch.as_tensor(val)
        self.data["logp"][self.ptr] = torch.as_tensor(logp)
        self.data["term"][self.ptr] = torch.as_tensor(terminal)
        self.ptr += 1

    def get(self, gamma=0.99, lam=0.95, last_val=0.0):
        # Slice the valid data
        valid_data = self.data[:self.ptr]
        
        # Compute GAE and returns
        rews = valid_data["rew"].numpy()
        vals = np.append(valid_data["val"].numpy(), last_val)
        terms = valid_data["term"].numpy()
        
        adv = np.zeros_like(rews)
        last_gae = 0
        for t in reversed(range(self.ptr)):
            delta = rews[t] + gamma * vals[t+1] * (1 - terms[t]) - vals[t]
            adv[t] = last_gae = delta + gamma * lam * (1 - terms[t]) * last_gae
            
        ret = adv + vals[:-1]
        
        data = {
            'obs': valid_data["obs"].clone(), # Clone to ensure it's not MM anymore when passed to training
            'act': valid_data["act"].clone(),
            'ret': torch.as_tensor(ret, dtype=torch.float32),
            'adv': torch.as_tensor(adv, dtype=torch.float32),
            'logp': valid_data["logp"].clone()
        }
        self.clear()
        return data

    def clear(self):
        self.ptr = 0

    def __len__(self):
        return self.ptr

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
        obs_shape = obs_space.shape if hasattr(obs_space, "shape") else (96,)
        self.buffers: Dict[str, RolloutBuffer] = {
            aid: RolloutBuffer(capacity=4096, obs_shape=obs_shape) 
            for aid in ["agent_0", "agent_1"]
        }

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
        if aid not in self.buffers:
            # Dynamically add buffer if new agent encountered (unlikely in this repo but good for robustness)
            obs_shape = obs.shape if hasattr(obs, "shape") else (96,)
            self.buffers[aid] = RolloutBuffer(capacity=4096, obs_shape=obs_shape)
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
        all_data = []
        
        total_steps = 0
        for aid, buf in self.buffers.items():
            if len(buf) == 0: continue
            
            # Use last_vals for bootstrapping if provided, otherwise 0
            last_val = last_vals.get(aid, 0.0) if last_vals else 0.0
            data = buf.get(self.config.gamma, self.config.lam, last_val=last_val)
            
            # Per-agent advantage normalization
            adv = data['adv']
            data['adv'] = (adv - adv.mean()) / (adv.std() + 1e-8)
            
            # Convert dict to TensorDict
            td = TensorDict({
                "obs": data['obs'],
                "act": data['act'],
                "ret": data['ret'],
                "adv": data['adv'],
                "logp": data['logp']
            }, batch_size=[len(data['obs'])])
            
            all_data.append(td)
            
            total_steps += len(data['obs'])
            
        if total_steps == 0:
            return {"loss_policy": 0.0, "loss_value": 0.0}

        # tensordict Integration: Single line cat and move to device
        data = torch.cat(all_data, dim=0).to(self.device)

        # Dataset for minibatch updates using TensorDict
        indices = torch.randperm(total_steps)
        
        avg_pi_loss = 0.0
        avg_v_loss = 0.0
        updates = 0

        for _ in range(self.config.update_epochs):
            for i in range(0, total_steps, self.config.minibatch_size):
                idx = indices[i:i+self.config.minibatch_size]
                batch = data[idx]
                
                dist, val = self.model(batch["obs"])
                logp = dist.log_prob(batch["act"])
                ratio = torch.exp(logp - batch["logp"])
                
                clip_adv = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * batch["adv"]
                loss_pi = -(torch.min(ratio * batch["adv"], clip_adv)).mean()
                loss_v = ((val.squeeze(-1) - batch["ret"])**2).mean() * self.config.value_coef
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
