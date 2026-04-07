Here is a comprehensive, professional `README.md` drafted to reflect the enterprise-grade architecture and robust refactoring we just completed for the CoffeeShop framework.

***

# ☕ CoffeeShop: Asynchronous Social Experience Sharing for MARL

CoffeeShop is a highly modular, enterprise-grade Multi-Agent Reinforcement Learning (MARL) framework. It introduces the concept of **Asynchronous Social Experience Sharing**, allowing independently acting, decentralized agents to securely share transition data through a centralized, off-policy Mediator.

Designed for robust research and rapid experimentation, CoffeeShop abstracts away boilerplate infrastructure, featuring strict environment contracts, hierarchical YAML configuration, and unified telemetry.

---

## ✨ Key Features

* **Asynchronous Social Experience Sharing:** Agents push high-value transitions to a central `PrioritizedBuffer`. A `CoffeeShopMediator` evaluates these memories and broadcasts them back to the population, gating low-quality or stale data.
* **Dynamic Openness (ω):** Agents utilize a learned, continuous parameter ($\omega \in [0.05, 0.75]$) to dynamically weight how much they trust and clone the behaviors broadcasted by the Mediator.
* **Hierarchical Configuration:** Powered by `OmegaConf`, experiments are entirely config-driven. Mix and match `agent`, `env`, `mediator`, and `trainer` parameters without touching Python code.
* **Strict API Contracts:** The `SocialEnvWrapper` enforces a standardized dictionary-based API across vastly different environments (from grid-worlds to complex 3D simulators).
* **Enterprise Telemetry:** Built-in graceful degradation for TensorBoard and Weights & Biases logging, including advanced metrics like **Population Diversity** (tracked via Jensen-Shannon Divergence).

---

## 🏗️ Architecture Overview

The CoffeeShop architecture separates the RL logic from the communication layer, ensuring that agents can learn independently while benefiting from the collective experience of the swarm.

1. **The Swarm (Local Policies):** Decentralized agents (e.g., On-policy PPO) interact with their local environment instances. They maintain their own specific goals and observation spaces.
2. **The Mediator (Centralized Critic):** A global, off-policy observer. It maintains a `PrioritizedBuffer` of shared experiences.
3. **The Feedback Loop:** * Agents securely push high-reward transitions to the Mediator.
    * The Mediator calculates a "Social Bonus" using its global value function.
    * High-quality experiences are broadcast back to the agents, who incorporate them via an auxiliary Behavioral Cloning (BC) loss.

---

## 📂 Repository Structure

```text
CoffeeShop/
├── core_marl/           # The Brain: Communication and Experience Sharing
│   ├── mediator.py          # Centralized critic and broadcasting logic
│   ├── experience_buffer.py # Prioritized replay buffer for social memories
│   └── social_actor.py      # Bindings between global IDs and local policies
├── agents/              # The Local Policies: RL Algorithms
│   ├── ppo.py               # Flagship PPO with auxiliary BC loss and learned openness (ω)
│   └── sac.py               # Continuous control off-policy baseline
├── envs/                # The Domains: Strict API Contracts
│   ├── base.py              # SocialEnvWrapper (Abstract Base Class)
│   ├── overcooked/          # 2D Cooperative Gridworld
│   ├── crafter/             # Open-world Survival
│   └── aisaac/              # Complex reinforcement learning swarm
├── utils/               # The Infrastructure: Factories and Telemetry
│   ├── factory.py           # Centralized instantiation (make_env, make_actors)
│   ├── evaluation.py        # N-agent dynamic evaluation loops
│   ├── metrics.py           # System-wide calculations (e.g., JS Divergence)
│   └── checkpointing.py     # Robust state-dict loading and saving
├── configs/             # The State Management: OmegaConf YAMLs
│   ├── agent/               
│   ├── env/                 
│   ├── mediator/            
│   └── trainer/             
└── scripts/             # The Execution Layer: Entry Points
    ├── train.py             # Main orchestration loop
    ├── eval.py              # Cross-play and baseline benchmarking
    └── playback.py          # Visual debugging and GIF rendering
```

---

## 🚀 Quick Start

### Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/ConnorMRyan/CoffeeShop.git
cd CoffeeShop
pip install -r requirements.txt
```
*(Note: TensorBoard and Weights & Biases are optional dependencies. The framework will gracefully default to standard console logging if they are not installed).*

### 1. Training a Population
CoffeeShop uses a clean, dot-notation CLI driven by `OmegaConf`. You can override any nested parameter directly from the command line.

```bash
# Train on the default environment (Overcooked Cramped Room)
python scripts/train.py

# Train on Crafter with specific overrides
python scripts/train.py env=crafter trainer.total_steps=5000000 trainer.device=cuda
```

### 2. Evaluating Checkpoints
The evaluation utility dynamically scales to handle N-agent environments.

```bash
# Evaluate cross-play between two different checkpoints
python scripts/eval.py --env overcooked --layout cramped_room \
    --ckpt_a checkpoints/run_1/checkpoint_100k.pt \
    --ckpt_b checkpoints/run_2/checkpoint_100k.pt \
    --episodes 10
```

### 3. Rendering Playback
Generate `.gif` files to visually inspect policy behavior and coordination.

```bash
python scripts/playback.py checkpoints/run_1/checkpoint_100k.pt \
    --env overcooked \
    --layout cramped_room \
    --output syngergy_demo.gif
```

---

## 🛠️ Extending CoffeeShop

### Adding a New Environment
CoffeeShop can support any environment. Simply subclass `SocialEnvWrapper` in `envs/base.py` and implement the required abstract methods (`reset`, `step`, `get_global_obs`). Ensure that all observations, actions, and rewards use dictionaries keyed by `agent_id` (e.g., `{"agent_0": 1.5}`). Register your new wrapper in `utils/factory.py:make_env()`.

### Tracking New Metrics
All system-wide metrics are tracked in `utils/metrics.py`. The orchestration loop in `scripts/train.py` automatically pulls these and broadcasts them to both TensorBoard and WandB simultaneously.