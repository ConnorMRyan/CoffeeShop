from __future__ import annotations

import numpy as np
import torch

from core_marl import CoffeeShopMediator
from agents.ppo import PPOAgent as PPOActor
from envs.aisaac.wrapper import AIsaacWrapper


def test_stub_training_loop_runs_200_steps():
    # Set seeds for determinism of the stub randomness (best-effort)
    np.random.seed(0)
    torch.manual_seed(0)

    # --- Build stub env ---
    env = AIsaacWrapper(num_agents=2, use_stub=True)

    # --- Build mediator ---
    mediator = CoffeeShopMediator(
        global_obs_dim=env.global_obs_dim,
        gamma=0.99,
        epsilon_td=0.05,
        synergy_alpha=0.3,
        device="cpu",
    )

    # --- Build two actors ---
    actors = {
        aid: PPOActor(
            agent_id=aid,
            obs_space=env.obs_dim,
            act_space=env.action_dim,
            global_obs_dim=env.global_obs_dim,
            mediator=mediator,
            gamma=0.99,
            lam=0.95,
            clip_eps=0.2,
            c_vf=0.5,
            c_ent=0.01,
            ppo_epochs=2,          # keep test fast
            mini_batch_size=32,
            lr=1e-3,               # slightly higher LR for faster signal
            push_every=32,         # push frequently so buffer fills within 200 steps
            device="cpu",
        )
        for aid in env.agent_ids
    }

    obs_dict, infos = env.reset()

    # Storage for at least one update's metrics
    got_metrics = False
    some_metrics = None

    total_steps = 200
    for t in range(total_steps):
        # Global obs for centralized critic
        global_obs = env.get_global_obs(obs_dict)

        # Actions per actor
        actions = {}
        for aid, actor in actors.items():
            # env_id=0 for simple stub
            action = actor.act(0, obs_dict[aid], global_obs)
            actions[aid] = action

        # Env step
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        sparse_rewards = infos.get("sparse_rewards", {aid: 0.0 for aid in env.agent_ids})
        next_global_obs = env.get_global_obs(next_obs)

        # Actor steps (store transitions and occasional push)
        for aid, actor in actors.items():
            done = terminated.get(aid, False)
            trunc = truncated.get(aid, False)
            actor.observe_outcome(
                next_global_obs=next_global_obs,
                reward=rewards[aid],
                sparse_reward=sparse_rewards[aid],
                done=done,
                truncated=trunc,
            )

        obs_dict = next_obs

        # Simple episodic reset if any agent done (to avoid degenerate rollouts)
        if any(terminated.values()) or any(truncated.values()):
            obs_dict, infos = env.reset()

        # Trigger PPO updates a couple of times within 200 steps
        if (t + 1) % 100 == 0:
            for aid, actor in actors.items():
                m = actor.update()
                if m:
                    some_metrics = m
                    got_metrics = True

        # Occasionally update mediator critic from recent transitions
        if (t + 1) % 64 == 0:
            # Pull last up-to-64 pushed ScoredMemory entries across buffer
            recent = list(mediator._buffer)[-64:]
            mediator.update_critic(recent)

    # --- Assertions ---
    assert got_metrics and isinstance(some_metrics, dict), "Expected at least one PPO update metrics dict."
    for key in ("policy_loss", "value_loss", "entropy", "bc_loss"):
        assert key in some_metrics, f"Missing metric key: {key}"

    # Openness in [0, 1]
    for actor in actors.values():
        assert 0.0 <= actor.openness.value <= 1.0, "Openness ω must be within [0, 1]"

    # Mediator buffer should have received some transitions
    assert len(mediator._buffer) > 0, "Mediator buffer should not be empty after 200 steps"

    env.close()
