# CoffeeShop AI Agent Guidelines

## Architecture Overview

CoffeeShop is a modular multi-agent reinforcement learning (MARL) framework. Core components:

- **SocialEnvWrapper**: Abstract base for env wrappers. `reset()` and `step(actions)` return dicts keyed by `agent_id` (e.g., `{"agent_0": obs, "agent_1": obs}`).
- **CoffeeShopMediator**: Centralized experience broker. Manages the prioritized buffer, per-agent baseline filtering, Gossip Factor (η), and reward shaping. Lives in `core_marl/mediator.py`.
- **SocialActor**: Binds an agent to an ID; delegates `act()` to the underlying agent policy.
- **ExperienceBuffer**: Collects transitions as `ExperienceBatch` (lists of per-agent dicts). See `core_marl/experience_buffer.py`.

Data flow: `Env → Mediator → SocialActors (act) → ExperienceBuffer → Mediator (share) → SocialActors (update)`

---

## Key Patterns

- **Multi-agent dicts**: All env/agent interactions use `{agent_id: value}` dicts. Example: `actions = {"agent_0": "interact", "agent_1": "up"}`. Agent IDs must be consistent across all components.
- **Lazy imports**: Optional deps (e.g., `overcooked-ai`) are imported only when needed. Raise `ImportError` with an install hint if missing.
- **Stub agents**: `PPOAgent.act()` returns `{aid: None}` by default. Implement with real models/optimizers before running non-smoke experiments.
- **Config loading**: Hydra manages all configuration (`conf/` directory). OmegaConf is the config backend. Override via CLI: `python scripts/train.py env=overcooked agent=ppo`.
- **Logging**: Hydra writes logs to its output directory. Metrics are exported to Parquet via `Polars`.
- **MARL tensors**: Use `tensordict` for rollout data and `einops` for dimension manipulation throughout.

---

## Mediator Internals

If modifying `core_marl/`, be aware of these mechanisms:

- **Priority buffer**: Fixed-capacity min-heap retaining highest-priority trajectories across population history. Backed by `core_marl/experience_buffer.py`.
- **Per-agent baseline filtering**: At query time, trajectories with reward below the requesting agent's current rolling average are excluded. This prevents expert agents regressing on novice data.
- **Learned Openness (ω)**: Each agent holds a logit-space parameter ω ∈ [0.05, 0.75] gating receptivity to Mediator-provided trajectories. Driven by internal value-network variance — do not confuse with a behavioural cloning weight.
- **Gossip Factor (η)**: When population rolling average drops below 70% of All-Time Best, η scales down effective ω for all agents, suppressing sharing during collective regression.

---

## Workflows

### Training
```bash
# Single process
python scripts/train.py env=overcooked agent=ppo run.steps=1000

# Distributed (Linux/macOS only)
torchrun --nproc_per_node=2 scripts/train.py env=overcooked agent=ppo run.steps=1000
```

### Evaluation
```bash
python scripts/evaluate.py --help   # canonical evaluation entry point
```

### Testing
Always run the test suite after making changes:
```bash
# With uv (canonical)
uv run pytest tests/

# With pip
pytest tests/
```

Key tests:
- `tests/test_invariants.py` — property-based buffer integrity tests (Hypothesis)
- `tests/test_experience_buffer.py` — experience collection and sampling logic

---

## Adding Environments

1. Implement `SocialEnvWrapper` subclass in `envs/{env_name}/wrapper.py`.
2. Ensure `reset()` and `step()` return dicts keyed by `agent_id`.
3. Register via `make_env()` in `utils/factory.py` — this is the single canonical registration point.

## Implementing Agents

1. Extend `agents/{agent}.py`.
2. Implement `act()` and `update()`.
3. Use `TensorDict` for rollout data and `einops` for tensor manipulation.
4. Use `ExperienceBuffer.export()` for batching transitions.

## Checkpointing

Use `utils/checkpointing.py` for all checkpoint save/load operations — do not call `torch.save` directly. This ensures consistent state_dict handling across distributed runs.

---

## Dependency Management

`uv` is the canonical package manager. `uv.lock` is the authoritative lockfile. `requirements.txt` is provided for legacy pip compatibility only.

```bash
uv sync              # install core deps
uv sync --all-extras # include all optional envs (Overcooked, Crafter, NetHack, etc.)
```

---

## Reference

| Concern | Location |
|---|---|
| Mediator & buffer logic | `core_marl/` |
| Agent policies | `agents/` |
| Environment wrappers | `envs/` |
| Config schemas | `conf/` |
| Training entry point | `scripts/train.py` |
| Evaluation entry point | `scripts/evaluate.py` |
| Env/agent registration | `utils/factory.py` |
| Metrics & JS divergence | `utils/metrics.py` |
| Checkpointing | `utils/checkpointing.py` |