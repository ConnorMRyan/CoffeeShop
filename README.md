# ☕ CoffeeShop: Asynchronous Social Experience Sharing for MARL

CoffeeShop is a modular, Multi-Agent Reinforcement Learning (MARL) framework that implements **Asynchronous Social Experience Sharing**. It allows decentralized agents to learn independently while sharing high-value transitions through a centralized Mediator.

Designed for research and rapid experimentation, CoffeeShop provides a unified API for environments, hierarchical YAML configuration, and advanced telemetry for monitoring population metrics.

---

## ✨ Key Features

- **Asynchronous Social Experience Sharing:** Agents push high-reward transitions to a central `PrioritizedBuffer`. A `CoffeeShopMediator` evaluates and broadcasts these memories back to the population.
- **Dynamic Openness (ω):** Agents learn a continuous parameter ($\omega \in [0.05, 0.75]$) to dynamically weight how much they trust and clone behaviors broadcasted by the Mediator.
- **Hierarchical Configuration:** Experiments are entirely config-driven via `OmegaConf`. Mix and match `agent`, `env`, `mediator`, and `trainer` parameters without touching Python code.
- **Strict API Contracts:** `SocialEnvWrapper` enforces a standardized dictionary-based API across different environments (Overcooked, Crafter, etc.).
- **Enterprise Telemetry:** Built-in support for TensorBoard and Weights & Biases, including advanced metrics like **Population Diversity** (measured via Jensen-Shannon Divergence).

---

## 🛠️ Stack & Requirements

- **Language:** Python 3.10+ (Recommended for `tensordict` compatibility)
- **Package Manager:** [uv](https://github.com/astral-sh/uv) (Highly recommended for fast environment setup)
- **Deep Learning:** [PyTorch](https://pytorch.org/) 2.2.0
- **High Performance:** [Tensordict](https://github.com/pytorch/tensordict), [Einops](https://einops.rocks/), [Polars](https://pola.rs/)
- **RL Ecosystem:** [Gymnasium](https://gymnasium.farama.org/), [Shimmy](https://shimmy.farama.org/)
- **Configuration:** [Hydra](https://hydra.cc/) (formerly OmegaConf)
- **Logging & UI:** [Rich](https://github.com/Textualize/rich), [Loguru](https://github.com/Delgan/loguru), [tqdm](https://github.com/tqdm/tqdm)

> **Note:** Due to compatibility issues between PyTorch and NumPy 2.0, this project requires **NumPy < 2.0** (specifically `1.26.4`).

---

## 🚀 Quick Start

### 1. Installation (Recommended: uv)

CoffeeShop uses `uv` for lightning-fast, reproducible environments.

```bash
# Install uv if you haven't already
powershell -c "irm https://astral-sh.uv.run/install.ps1 | iex" # Windows
# curl -LsSf https://astral-sh.uv.run/install.sh | sh         # Linux/macOS

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

> **Note:** Some optional environments (like `crafter`) may face encoding issues during build on Windows. If `uv sync --all-extras` fails, install core dependencies first and then specific extras as needed.

### 3. Training a Population

The primary entry point is `scripts/train.py` (orchestrated via Hydra).

```bash
# Standard single-process run
python scripts/train.py env=overcooked agent=ppo run_id=experiment_1

# Overriding environment and steps
python scripts/train.py env=overcooked agent=ppo run.steps=50000 env.layout_name=cramped_room

# Distributed training via torchrun (2 GPUs/Processes)
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
├── uv.lock              # Modern lockfile for reproducible environments
└── requirements.txt     # Version-locked core dependencies (legacy)
```

---

## 📜 Scripts & Entry Points

### Main Entry Points
- `scripts/train.py`: Unified training loop supporting single-process and distributed (DDP) execution via Hydra.
- `scripts/evaluate.py`: Standalone evaluation utility.

### Legacy Components
Legacy orchestration layers (`coffeeshop/` and redundant `configs/`) have been removed in favor of the modernized `scripts/` and `conf/` workflow to ensure architectural consistency.

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

## ⚖️ License

TODO: Add license information (e.g., MIT, Apache 2.0).