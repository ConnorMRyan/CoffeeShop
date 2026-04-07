# CoffeeShop AI Agent Guidelines

## Architecture Overview
CoffeeShop is a modular multi-agent reinforcement learning (MARL) framework implementing Asynchronous Social Experience Sharing. Core components:
- **SocialEnvWrapper** (`envs/base.py`): Abstract Base Class for environment wrappers; `reset()` and `step(actions)` return dicts keyed by `agent_id`. Subclasses must implement `get_global_obs()`.
- **CoffeeShopMediator** (`core_marl/mediator.py`): Off-policy centralized critic. Manages peer experience sharing via `PrioritizedBuffer`.
- **PrioritizedBuffer** (`core_marl/memory.py`): Stores `Transition` and `ScoredMemory` objects. Decoupled from mediator neural logic to handle sampling and priority weighting.
- **PPOAgent** (`agents/ppo.py`): Fully implemented on-policy actor with ActorCriticNet (shared trunk + actor/critic heads), RolloutBuffer, GAE, PPO clip loss, and auxiliary behavior cloning (BC) loss.
- **LearnedOpenness (ω)**: A per-agent `nn.Parameter` (logit-space) that controls the BC loss weight. Clamped to [0.05, 0.75].
- **SocialActor** (`core_marl/social_actor.py`): Thin wrapper binding an agent to a global ID; delegates `act()`.
- **VectorSocialRunner** (`utils/factory.py`): Manages N parallel `SocialEnvWrapper` instances. Global agent IDs are `env{e}_agent_{i}`.
- **Metrics & Diversity** (`utils/metrics.py`): Handles population-wide logging, including the **Population Diversity Signal** (mean pairwise Jensen-Shannon divergence).

Data flows: Env → VectorSocialRunner → PPOAgents (act/observe) → CoffeeShopMediator (push/broadcast) → BC loss.

## Key Patterns
- **Multi-agent dicts**: All env/agent interactions use `{agent_id: value}` dicts. Example: `actions = {"env0_agent_0": 2, "env0_agent_1": 0}`.
- **Hierarchical Configuration**: Uses `OmegaConf` for YAML configs. Configs are split into `agent`, `env`, `mediator`, and `trainer` (see `configs/`).
- **PPOAgent API (act/observe split)**: Call `actor.act(env_id, obs, global_obs)` before the env step, then `actor.observe_outcome(next_global_obs, reward, sparse_reward, done, truncated)` after.
- **Transition / RolloutBuffer**: `RolloutBuffer` uses a pending-step mechanism to ensure rewards are correctly paired with the causal actions.
- **Logging**: Supports `TensorBoard` and `Weights & Biases` (`WandbWriter` in `utils/__init__.py`).

## Workflows
- **Training**: `python scripts/train.py env=overcooked env.layout=cramped_room trainer.total_steps=2000000`. CLI overrides use dot-notation for nested YAML fields.
- **Cross-play Evaluation**: `python scripts/eval.py --ckpt_a <path> --ckpt_b <path> --episodes 5`. Uses `ActorFromCheckpoint` and `run_episode` from `utils/evaluation.py`.
- **Playback**: `python scripts/playback.py <checkpoint.pt> --output out.gif --layout cramped_room`. Renders saved checkpoints to GIF.
- **Adding Envs**: Implement `SocialEnvWrapper` subclass in `envs/{name}/wrapper.py`; register in `make_env` (`utils/factory.py`).
- **Implementing Agents**: Extend `agents/{agent}.py`; wire to PyTorch models; use `RolloutBuffer` / `ExperienceBuffer`.
- **Checkpointing**: `Checkpointer(dirpath, run_id).save(state_dict)` saves under `checkpoints/{run_id}/`. Keys include `env{e}_agent_{i}`, `mediator`, and `step`.

## Conventions
- **Agent IDs**: Local wrappers use `["agent_0", "agent_1"]`; `VectorSocialRunner` promotes these to global IDs `env{e}_agent_{i}`.
- **Reward Shaping**: `team_reward / N_agents + shaped_reward * shaping_factor`.
- **Termination vs Truncation**: Strictly distinguished for GAE bootstrapping. `truncated=True` triggers V(s') bootstrapping; `terminated=True` does not.
- **Population Diversity**: Calculated using JS divergence on a shared probe batch sampled from the `Mediator`.

Reference: `core_marl/`, `utils/factory.py`, `utils/evaluation.py`, `agents/ppo.py`, `scripts/train.py`, `configs/`, `ARCHITECTURE.md`.
