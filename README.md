# CoffeeShop — Multi-Agent Reinforcement Learning Skeleton

CoffeeShop is a modular, environment-agnostic MARL research skeleton. It is designed to implement **Asynchronous Social Experience Sharing**, a novel architecture that combines the stability of on-policy PPO with the sample efficiency of off-policy centralized critics.

> 📖 **Read the full theoretical abstract and architectural breakdown in [ARCHITECTURE.md](ARCHITECTURE.md).**

### Features
- A unified multi-agent environment interface (`SocialEnvWrapper`)
- Reference baselines (`PPOAgent`, `SACAgent`) as placeholders
- Core components for social mediation and experience collection (`Mediator`, `SocialActor`, `ExperienceBuffer`)
- Environment wrappers for Overcooked-AI, Crafter, NetHack, and a custom AIsaac env (all optional deps)
- Config, utils, scripts, and minimal runnable stubs

## Repository Structure
```
CoffeeShop/
├── agents/                  # Baseline agents (stubs)
├── configs/                 # YAML configs
├── core_marl/               # CoffeeShop core components
├── envs/                    # Env-agnostic wrappers (optional deps)
├── scripts/                 # Train/Eval stubs
├── utils/                   # Logging, metrics, checkpointing
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation
Create and activate a virtual environment (recommended), then install core dependencies. Optional packages for specific environments are commented out in `requirements.txt`.

```bash
pip install -r requirements.txt
# Then pick your torch build explicitly (CPU or CUDA) per the comments in the file.
```

Optional environments (enable as needed):
- Overcooked-AI: `overcooked-ai`, `pygame`
- Crafter: `crafter`
- NetHack Learning Environment: `nle` (may require build tools)
- AIsaac (custom): add your private package URL

## Quickstart
The included scripts are minimal stubs that exercise the architecture without performing real learning.

```bash
# Example (Overcooked layout; requires overcooked-ai installed)
python scripts/train.py --env overcooked --layout cramped_room --agent ppo --steps 1000

python scripts/evaluate.py --env overcooked --layout cramped_room --episodes 3
```

If optional dependencies are not installed, the wrappers will raise a clear `ImportError`.

## Configuration Files
- `configs/env_config.yaml` — choose environment and parameters
- `configs/agent_config.yaml` — baseline agent and hyperparameters
- `configs/run_config.yaml` — run settings (steps, logging, tracking)

These are templates. If you prefer Hydra/OmegaConf, enable `hydra-core` and integrate config loading.

## Core Concepts
- `SocialEnvWrapper`: unified multi-agent API: `reset() -> (obs, infos)` and `step(actions) -> (obs, rewards, terminated, truncated, infos)`.
- `Mediator`: coordinates env-agent interactions and can apply social reward shaping.
- `SocialActor`: binds an `agent` to an `id`; delegates action selection.
- `ExperienceBuffer`: simple rollout buffer for multi-agent transitions.

## Notes
- The agents are placeholders that return no-op actions; connect your models/optimizers and gym spaces to make them functional.
- Wrappers for Crafter/NLE/AIsaac are stubs until those dependencies are installed and integrated; Overcooked wrapper demonstrates the intended flow.

## Development
- Code style: follow local patterns and docstrings.
- Logging prefers `rich` if available.
- Checkpointing uses PyTorch if installed.

Contributions welcome. Tailor `requirements.txt` to your CUDA/CPU setup for PyTorch.
