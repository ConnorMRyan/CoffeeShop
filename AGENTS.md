# CoffeeShop AI Agent Guidelines

## Architecture Overview
CoffeeShop is a modular multi-agent reinforcement learning (MARL) skeleton. Core components:
- **SocialEnvWrapper**: Abstract base for env wrappers; `reset()` and `step(actions)` return dicts keyed by `agent_id` (e.g., `{"agent_0": obs, "agent_1": obs}`).
- **Mediator**: Wraps env, handles reward shaping (e.g., shared reward averaging in `mediator.py`).
- **SocialActor**: Binds an agent to an ID; delegates `act()` to underlying agent.
- **ExperienceBuffer**: Collects transitions as `ExperienceBatch` (lists of per-agent dicts).

Data flows: Env → Mediator → SocialActors (for actions) → ExperienceBuffer.

## Key Patterns
- **Multi-agent dicts**: All env/agent interactions use `{agent_id: value}` dicts. Example: `actions = {"agent_0": "interact", "agent_1": "up"}`.
- **Lazy imports**: Optional deps (e.g., `overcooked-ai`) imported only when needed; raise `ImportError` if missing.
- **Stub agents**: `PPOAgent.act()` returns `{aid: None}`; implement with models/optimizers for real learning.
- **Config loading**: Use Hydra for configuration management (`conf/` directory). Override via CLI: `python scripts/train.py env=overcooked agent=ppo`.
- **Logging**: Uses Hydra's output directory for logs and `Polars` for parquet metrics.
- **MARL Tensors**: Uses `tensordict` and `einops` for efficient multi-agent data handling.

## Workflows
- **Training**: Run `python scripts/train.py env=overcooked run.steps=1000`. This uses Hydra to load defaults from `conf/`.
- **Distributed Training**: Run `torchrun --nproc_per_node=X scripts/train.py run.steps=1000`.
- **Evaluation**: Use `scripts/evaluate.py` for simple stubs or `coffeeshop/eval.py` (if restored/modernized) for checkpoint evaluation.
- **Adding envs**: Implement `SocialEnvWrapper` subclass in `envs/{env_name}/wrapper.py`; update `make_env()` in `scripts/train.py` and `utils/factory.py`.
- **Implementing agents**: Extend `agents/{agent}.py` models; use `TensorDict` for rollout data and `einops` for dimension manipulation.
- **Checkpointing**: Use `torch.save` for state_dicts.

## Conventions
- **Agent IDs**: Consistent across components (e.g., `["agent_0", "agent_1"]`).
- **Reward shaping**: Managed by `CoffeeShopMediator`.
- **Metrics**: Aggregated via `utils/metrics.py` and exported to parquet.

Reference: `core_marl/`, `envs/`, `agents/ppo.py`, `conf/`, `scripts/train.py`.
