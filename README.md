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

- **Language:** Python 3.9+
- **Deep Learning:** [PyTorch](https://pytorch.org/) 2.2.0
- **High Performance:** [Tensordict](https://github.com/pytorch/tensordict), [Einops](https://einops.rocks/), [Polars](https://pola.rs/)
- **RL Ecosystem:** [Gymnasium](https://gymnasium.farama.org/), [Shimmy](https://shimmy.farama.org/)
- **Configuration:** [Hydra](https://hydra.cc/) (formerly OmegaConf)
- **Logging & UI:** [Rich](https://github.com/Textualize/rich), [Loguru](https://github.com/Delgan/loguru), [tqdm](https://github.com/tqdm/tqdm)

> **Note:** Due to compatibility issues between PyTorch and NumPy 2.0, this project requires **NumPy < 2.0** (specifically `1.26.4`).

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ConnorMRyan/CoffeeShop.git
cd CoffeeShop
pip install -r requirements.txt
```

To install all environment dependencies (Overcooked, Crafter, NetHack):

```bash
pip install ".[all]"
```

### 2. Training a Population

The primary entry point is `scripts/train.py` (orchestrated via Hydra).

```bash
# Standard single-process run
python scripts/train.py env=overcooked agent=ppo run_id=experiment_1

# Overriding environment and steps
python scripts/train.py env=overcooked agent=ppo run.steps=50000 env.layout_name=cramped_room

# Distributed training via torchrun (2 GPUs/Processes)
torchrun --nproc_per_node=2 scripts/train.py env=overcooked agent=ppo run_id=dist_run dist=true

# Distributed sweep using Hydra's multirun
torchrun --nproc_per_node=2 scripts/train.py --multirun env=overcooked agent=ppo run.seed=42,43,44 dist=true
```

### 3. Evaluation & Playback

```bash
# Evaluate cross-play between checkpoints
python coffeeshop/eval.py --env overcooked --layout cramped_room \
    --ckpt_a checkpoints/run_1/model.pt \
    --ckpt_b checkpoints/run_2/model.pt

# Generate behavior GIFs
python coffeeshop/playback.py checkpoints/run_1/model.pt --env overcooked --output demo.gif
```

---

## 📂 Project Structure

```text
CoffeeShop/
├── agents/              # RL Algorithms (PPO optimized with TensorDict)
├── coffeeshop/          # High-level execution layer (Main entry points)
├── conf/                # Hydra YAML configurations (agent, env, mediator, trainer)
├── core_marl/           # The Engine: Mediator, buffers, and social actors
├── envs/                # Environment wrappers (Overcooked, Crafter, NetHack, etc.)
├── models/              # Neural network architectures (MLP, CNN encoders)
├── outputs/             # Default Hydra output directory for logs and parquet metrics
├── scripts/             # Main training and evaluation scripts
├── tests/               # Core unit and integration tests (pytest + Hypothesis)
├── utils/               # Shared utilities (metrics, diversity analysis, factories)
├── pyproject.toml       # Package metadata and locked dependencies
└── requirements.txt     # Version-locked core dependencies
```

---

## 📜 Scripts & Entry Points

CoffeeShop provides several ways to run the code:

### Command Line Interfaces (CLIs)
When installed as a package (`pip install -e .`), the following commands are available:
- `coffeeshop-train`: Maps to `coffeeshop.train:main`
- `coffeeshop-eval`: Maps to `coffeeshop.eval:main`
- `coffeeshop-evaluate`: Maps to `coffeeshop.evaluate:evaluate`
- `coffeeshop-playback`: Maps to `coffeeshop.playback:main`

### Lightweight Scripts
- `scripts/train.py`: A simplified training loop for quick debugging without full config hierarchy.
- `scripts/evaluate.py`: Standalone evaluation utility.

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