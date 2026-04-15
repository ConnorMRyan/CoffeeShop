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
- **Config loading**: Use `omegaconf` for YAML configs (agent params, env params, run settings).
- **Logging**: Prefer `rich` for console output; fallback to standard logging.

## Workflows
- **Training**: Run `python scripts/train.py --env overcooked --layout cramped_room --agent ppo --steps 1000`. Loops: `mediator.reset()`, `agent.act(obs)`, `mediator.step(actions)`, collect metrics.
- **Evaluation**: Similar to training but without updates; use `scripts/evaluate.py`.
- **Adding envs**: Implement `SocialEnvWrapper` subclass in `envs/{env_name}/wrapper.py`; add to `make_env()` in `scripts/train.py`.
- **Implementing agents**: Extend `agents/{agent}.py` stubs; wire to PyTorch models, use `ExperienceBuffer.export()` for batches.
- **Checkpointing**: Use `Checkpointer.save/load()` for PyTorch state_dicts; requires `torch` installed.

## Conventions
- **Agent IDs**: Hardcoded in wrappers (e.g., `["agent_0", "agent_1"]` in Overcooked); consistent across components.
- **Reward shaping**: Modify `Mediator.step()` for social rewards (e.g., averaging).
- **Metrics**: Use `Metrics.update({"reward": sum(rewards.values())})`; log means periodically.
- **Diversity Monitoring**: Population diversity is measured using Jensen-Shannon Divergence on a shared probe batch. Logged as `population_diversity`.
- **Dependencies**: Core in `requirements.txt`; optional envs commented out; install PyTorch explicitly for CUDA/CPU.

Reference: `core_marl/`, `envs/overcooked/wrapper.py`, `agents/ppo.py`, `configs/`.</content>
<parameter name="filePath">C:\Users\mynam\PycharmProjects\CoffeeShop\AGENTS.md
