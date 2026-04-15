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
- **Deep Learning:** [PyTorch](https://pytorch.org/) 2.2+
- **RL Ecosystem:** [Gymnasium](https://gymnasium.farama.org/), [Shimmy](https://shimmy.farama.org/)
- **Configuration:** [OmegaConf](https://omegaconf.readthedocs.io/)
- **Logging & UI:** [Rich](https://github.com/Textualize/rich), [Loguru](https://github.com/Delgan/loguru), [tqdm](https://github.com/tqdm/tqdm)
- **Experiment Tracking:** [WandB](https://wandb.ai/), [TensorBoard](https://www.tensorflow.org/tensorboard)

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

The primary entry point is `coffeeshop/train.py`. You can use shorthand CLI overrides:

```bash
# Train with default config (Overcooked)
python coffeeshop/train.py

# Override environment and steps
python coffeeshop/train.py env=crafter trainer.total_steps=5000000 trainer.device=cuda

# Run a vanilla PPO baseline (disables social features)
python coffeeshop/train.py --baseline
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
├── agents/              # RL Algorithms (PPO, SAC stubs)
├── coffeeshop/          # High-level execution layer (Main entry points)
│   ├── train.py         # Main training orchestrator
│   ├── eval.py          # Evaluation loops
│   └── playback.py      # Rendering and GIF generation
├── configs/             # OmegaConf YAML files (agent, env, mediator, trainer)
├── core_marl/           # The Engine: Mediator, buffers, and social actors
├── envs/                # Environment wrappers (Overcooked, Crafter, NetHack, etc.)
├── models/              # Neural network architectures (MLP, CNN encoders)
├── scripts/             # Lightweight/standalone utility scripts
├── test/                # Core unit and integration tests (pytest)
├── utils/               # Shared utilities (metrics, checkpointing, factories)
├── pyproject.toml       # Package metadata and CLI tool definitions
└── requirements.txt     # Core dependencies
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
pytest test/
```

Key tests:
- `test/test_smoke.py`: Verifies a short training run (1k steps).
- `test/test_experience_buffer.py`: Tests for prioritized and shared buffers.
- `test/test_mediator_math.py`: Validates social trust and TD-error logic.

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