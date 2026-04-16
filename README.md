# ☕ CoffeeShop: Asynchronous Social Experience Sharing for MARL

CoffeeShop is a modular Multi-Agent Reinforcement Learning (MARL) framework that implements **Asynchronous Social Experience Sharing**. It allows decentralized agents to learn independently while sharing high-value transitions through a centralized Mediator.

Designed for research and rapid experimentation, CoffeeShop provides a unified API for environments, hierarchical YAML configuration, and advanced telemetry for monitoring population metrics.

> **Paper:** [CoffeeShop: Asynchronous Social Experience Sharing via Off-Policy Mediation in Multi-Agent Reinforcement Learning](./CoffeeShop_Refactored.md)

---

## ✨ Key Features

- **Asynchronous Social Experience Sharing:** Agents push high-reward transitions to a central `PrioritizedBuffer`. A `CoffeeShopMediator` evaluates and broadcasts these trajectories back to the population.
- **Dynamic Openness (ω):** Agents learn a continuous parameter (ω ∈ [0.05, 0.75]) that gates receptivity to Mediator-provided trajectories, scaled by internal value-network uncertainty.
- **Hierarchical Configuration:** Experiments are entirely config-driven via [Hydra](https://hydra.cc/) (which uses [OmegaConf](https://omegaconf.readthedocs.io/) as its config backend). Mix and match `agent`, `env`, `mediator`, and `trainer` parameters without touching Python code.
- **Strict API Contracts:** `SocialEnvWrapper` enforces a standardized dictionary-based API across different environments (Overcooked, Crafter, etc.).
- **Telemetry:** Built-in support for TensorBoard and Weights & Biases, including population-level metrics such as **Population Diversity** (measured via Jensen-Shannon Divergence).

---

## 🛠️ Stack & Requirements

- **Language:** Python 3.10+ (required for `tensordict` compatibility)
- **Package Manager:** [uv](https://github.com/astral-sh/uv) (recommended for fast, reproducible environment setup)
- **Deep Learning:** [PyTorch](https://pytorch.org/) 2.2.0
- **High Performance:** [Tensordict](https://github.com/pytorch/tensordict), [Einops](https://einops.rocks/), [Polars](https://pola.rs/)
- **RL Ecosystem:** [Gymnasium](https://gymnasium.farama.org/), [Shimmy](https://shimmy.farama.org/)
- **Configuration:** [Hydra](https://hydra.cc/) + [OmegaConf](https://omegaconf.readthedocs.io/)
- **Logging & UI:** [Rich](https://github.com/Textualize/rich), [Loguru](https://github.com/Delgan/loguru), [tqdm](https://github.com/tqdm/tqdm)

> **Note:** Due to compatibility issues between PyTorch 2.2.0 and NumPy 2.0, this project requires **NumPy < 2.0** (specifically `1.26.4`).

---

## 🚀 Quick Start

### 1. Installation (Recommended: uv)

`uv` is the canonical package manager for this project. `requirements.txt` is provided for legacy compatibility only; `uv.lock` is the authoritative lockfile.

```bash
# Install uv
# Linux/macOS
curl -LsSf https://astral-sh.uv.run/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral-sh.uv.run/install.ps1 | iex"

# Clone the repository
git clone https://github.com/ConnorMRyan/CoffeeShop.git
cd CoffeeShop

# Create virtualenv and sync dependencies
uv sync
```

To run commands within the managed environment:

```bash
# Run training
uv run python scripts/train.py env=overcooked agent=ppo

# Run tests
uv run pytest tests/
```

### 2. Legacy Installation (pip)

If you prefer `pip`, ensure you are using **Python 3.10+**:

```bash
# Create and activate your own venv first, then:
pip install -r requirements.txt
pip install -e .
```

To install all environment dependencies (Overcooked, Crafter, NetHack):

```bash
# With uv
uv sync --all-extras

# With pip
pip install ".[all]"
```

> **Note:** Some optional environments (like `crafter`) may face encoding issues during build on Windows. If `uv sync --all-extras` fails, install core dependencies first and then add specific extras as needed.

### 3. Training a Population

The primary entry point is `scripts/train.py`, orchestrated via Hydra.

```bash
# Standard single-process run
python scripts/train.py env=overcooked agent=ppo run_id=experiment_1

# Overriding environment and steps
python scripts/train.py env=overcooked agent=ppo run.steps=50000 env.layout_name=cramped_room

# Distributed training via torchrun (Linux/macOS, 2 processes)
torchrun --nproc_per_node=2 scripts/train.py env=overcooked agent=ppo run_id=dist_run run.dist_backend=gloo
```

---

## 📂 Project Structure

```text
CoffeeShop/
├── agents/              # RL Algorithms (PPO optimized with TensorDict)
├── conf/                # Hydra YAML configurations (agent, env, mediator, trainer)
├── core_marl/           # The Engine: Mediator, buffers, and social actors
├── envs/                # Environment wrappers (Overcooked, Crafter, NetHack, etc.)
├── models/              # Neural network architectures (MLP, CNN encoders)
├── outputs/             # Default Hydra output directory for logs and parquet metrics
├── scripts/             # Main training and evaluation scripts
├── tests/               # Core unit and integration tests (pytest + Hypothesis)
├── utils/               # Shared utilities (metrics, diversity analysis, factories)
├── pyproject.toml       # Package metadata and locked dependencies
├── uv.lock              # Authoritative lockfile for reproducible environments
└── requirements.txt     # Legacy pip dependencies
```

---

## 📜 Scripts & Entry Points

### Main Entry Points
- `scripts/train.py`: Unified training loop supporting single-process and distributed (DDP) execution via Hydra.
- `scripts/evaluate.py`: Standalone evaluation utility. Run `python scripts/evaluate.py --help` for available flags.

### Legacy Components
Legacy orchestration layers (`coffeeshop/` and redundant `configs/`) have been removed in favour of the modernized `scripts/` and `conf/` workflow to ensure architectural consistency.

---

## 🧪 Testing

Run the test suite using `pytest`:

```bash
pytest tests/
```

Key tests:
- `tests/test_invariants.py`: Property-based testing for buffer integrity using Hypothesis.
- `tests/test_experience_buffer.py`: Core logic for experience collection and sampling.

---

## ⚙️ Environment Variables

- `WANDB_API_KEY`: Set your API key for Weights & Biases logging.
- `CUDA_VISIBLE_DEVICES`: (Optional) Specify which GPUs to use for training.

---

## 🤝 Extending CoffeeShop

### Adding an Environment
1. Implement a `SocialEnvWrapper` in `envs/{env_name}/wrapper.py`.
2. Ensure observations/actions are dictionaries keyed by `agent_id`.
3. Register the environment in `utils/factory.py`.

### Adding an Agent
1. Extend the stubs in `agents/`.
2. Implement the `act()` and `update()` methods.
3. Use `ExperienceBuffer.export()` for batching.

---

## 📄 Citation

If you use CoffeeShop in your research, please cite:

```bibtex
@misc{ryan2025coffeeshop,
  title   = {CoffeeShop: Asynchronous Social Experience Sharing via Off-Policy Mediation in Multi-Agent Reinforcement Learning},
  author  = {Ryan, Connor},
  year    = {2025},
  url     = {https://github.com/ConnorMRyan/CoffeeShop}
}
```

---

## ⚖️ License

This project is licensed under the **Apache License 2.0**. See [LICENSE](./LICENSE) for details.