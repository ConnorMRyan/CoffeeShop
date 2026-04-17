from __future__ import annotations
import os
import torch
import numpy as np
from typing import Tuple, Any, Dict, Optional
from utils.checkpointing import Checkpointer
from envs import SocialEnvWrapper

class RandomPartner:
    """A baseline agent that takes random actions."""
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self._rng = np.random.default_rng()

    def act(self, obs: torch.Tensor) -> int:
        return int(self._rng.integers(low=0, high=self.action_dim))

class ActorFromCheckpoint:
    """Loads a policy from a CoffeeShop checkpoint for evaluation."""
    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            ckpt: str,
            key_hints: Tuple[str, ...],
            device: str = "cpu"
    ):
        from agents.ppo import ActorCritic
        if not ckpt:
            raise ValueError("A checkpoint path is required.")

        self.device = torch.device(device)
        # Use weight_only=True for safety
        data = torch.load(ckpt, map_location=self.device, weights_only=True)

        # ActorCritic is fixed to (obs_dim, action_dim, hidden_size)
        # We'll use 64 as default hidden_size to match current models
        self.net = ActorCritic(
            obs_dim    = obs_dim,
            act_dim    = action_dim,
            hidden_size = 64
        ).to(self.device)

        state_dict = None
        for k in key_hints:
            if k in data and isinstance(data[k], dict):
                state_dict = data[k]
                break

        if state_dict is None:
            # Fallback to general 'model' key if found
            if "model" in data:
                state_dict = data["model"]
            else:
                available = [k for k, v in data.items() if isinstance(v, dict)]
                raise KeyError(f"Agent state not found in {ckpt}. Tried {key_hints}. Available: {available}")

        self.net.load_state_dict(state_dict, strict=False)
        self.net.eval()

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> int:
        """Evaluation uses greedy actions."""
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float32)
        
        # Ensure input is [batch, obs_dim]
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
            
        logits, _ = self.net(obs.to(self.device))
        return int(torch.argmax(logits, dim=-1).item())

def run_episode(
        env: SocialEnvWrapper,
        actors: Dict[str, Any],
        horizon: int = 1000,
        render: bool = False
) -> Dict[str, Any]:
    """
    Runs a single evaluation episode for an N-agent environment.

    Args:
        env: Instantiated SocialEnvWrapper.
        actors: Dict mapping agent_id (e.g., "agent_0") to an actor object.
        horizon: Maximum step limit.
        render: Whether to call env.render().

    Returns:
        Dict containing total_team_score, agent_rewards, and custom metrics.
    """
    obs_dict, _ = env.reset()
    agent_ids = env.agent_ids

    agent_rewards = {aid: 0.0 for aid in agent_ids}
    total_deliveries = 0.0

    for _ in range(horizon):
        actions = {}
        for aid in agent_ids:
            if aid not in actors:
                raise ValueError(f"Missing evaluation actor for env agent: {aid}")

            # Each agent evaluates its own local observation
            actions[aid] = actors[aid].act(obs_dict[aid])

        n_obs, rewards, terminated, truncated, infos = env.step(actions)

        # Track custom environment-specific metrics (e.g., Overcooked)
        if infos.get("has_delivery", False):
            total_deliveries += 1.0

        # Accumulate sparse rewards dynamically for all agents
        sr = infos.get("sparse_rewards", {})
        for aid in agent_ids:
            agent_rewards[aid] += float(sr.get(aid, 0.0))

        obs_dict = n_obs
        if render:
            env.render()

        if any(terminated.values()) or any(truncated.values()):
            break

    return {
        "total_team_score": sum(agent_rewards.values()),
        "agent_rewards": agent_rewards,
        "total_deliveries": total_deliveries
    }
