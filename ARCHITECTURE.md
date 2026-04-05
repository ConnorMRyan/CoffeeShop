# CoffeeShop MARL: Asynchronous Social Experience Sharing via Off-Policy Mediation

**Author:** Connor Ryan

## Abstract
While on-policy algorithms like Proximal Policy Optimization (PPO) offer robust stability in reinforcement learning, they inherently struggle with sample inefficiency and local optima stagnation, particularly in procedurally generated environments with sparse, highly complex reward triggers. Conversely, off-policy algorithms exhibit high sample efficiency but suffer from training instability.

We propose a novel hybrid framework termed **Social Reinforcement Learning** via a Mediator Architecture (the "CoffeeShop" model). In this setup, a population of decentralized, on-policy PPO Actor networks explore independent environment instances. At defined step intervals, agents asynchronously push their trajectory data to a centralized, off-policy Mediator Critic. The Mediator evaluates these episodic memories, calculating the Temporal Difference (TD) error to identify highly novel or extreme-reward events.

To bypass the mathematical constraints of on-policy learning, the Mediator distills these high-priority experiences back to the population via an auxiliary behavior cloning loss. The integration of foreign experiences is dynamically gated by a learned "Openness" threshold ($\omega$), preventing network corruption from sub-optimal peers, while obsolete strategies are purged via a population-relative recency decay ($\gamma_{time}$). This architecture preserves the stability of PPO while drastically accelerating the discovery of global optima through mediated, crowdsourced experience sharing.

---

## Core Architecture

The repository is built around the concept of **Social Agency and Selective Integration**. Unlike standard centralized-critic models where actors are forcibly overwritten, CoffeeShop is a Peer-to-Peer network with a Mediator.

1. **The Actors (PPO):** Independent agents training locally, providing stable gradient updates and mechanical foundations.
2. **The Mediator (Off-Policy Critic):** A centralized buffer and evaluator that learns the global $Q$-value of all shared experiences.
3. **The Meeting (Auxiliary Distillation):** The Mediator brokers high-value memories to actors. Actors dynamically decide whether to integrate these memories based on their internal confidence.

### Key Mechanisms

* **Dynamically Learned Openness ($\omega$):** Agents do not blindly accept shared memories. $\omega$ is a learned parameter tied to the agent's internal Value network variance. High internal uncertainty spikes $\omega$ (making the agent receptive to help), while high confidence drops $\omega$ (making the agent stubborn and protecting its weights).
* **Population-Relative Recency Decay ($\gamma_{time}$):** Memories are not decayed purely by step-count, but by the moving average of the population's baseline capability. As the society of agents improves, beginner-level "success" stories are purged to prevent regressive training.
* **Mediator Priority Threshold ($\epsilon_{TD}$):** A baseline filter ensuring the Mediator only passes along memories that significantly challenge an agent's current worldview.

---

## Benchmarks & Environments

This architecture is designed to solve the bottlenecks of high-variance, combinatorially complex, and procedurally generated domains. The repository abstracts the environment layer to benchmark the CoffeeShop model cleanly against standard baselines across both recognized academic standards and custom complex environments.

* **AIsaac:** A custom procedurally generated environment training an agent to navigate the combinatorial item synergies and brutal mechanics of *The Binding of Isaac*.
* **Crafter:** DeepMind's 2D open-world survival benchmark testing long, sparse-reward tech trees.
* **NLE (NetHack Learning Environment):** A highly stochastic roguelike domain serving as a standardized RL benchmark.
* **Overcooked-AI:** A classic MARL benchmark for multi-agent coordination and strategy adaptation.

---

## Repository Structure

```text
CoffeeShop/
├── agents/                           # Standard baselines (for ablation studies)
│   ├── __init__.py
│   ├── ppo.py
│   └── sac.py
├── configs/                          # YAML configuration management
│   ├── agent_config.yaml
│   ├── env_config.yaml
│   └── run_config.yaml
├── core_marl/                        # Novel CoffeeShop architecture
│   ├── __init__.py
│   ├── experience_buffer.py          # Centralized prioritized replay
│   ├── mediator.py                   # Off-policy TD-error evaluation
│   └── social_actor.py               # PPO Actor with learned openness (\omega)
├── envs/                             # Environment agnosticism layer
│   ├── __init__.py
│   ├── aisaac/                       # The Binding of Isaac integration
│   │   ├── __init__.py
│   │   └── wrapper.py
│   ├── crafter/                      # Crafter benchmark
│   │   ├── __init__.py
│   │   └── wrapper.py
│   ├── nethack/                      # NetHack Learning Environment
│   │   ├── __init__.py
│   │   └── wrapper.py
│   └── overcooked/                   # Overcooked-AI
│       ├── __init__.py
│       └── wrapper.py
├── scripts/                          # Execution scripts
│   ├── evaluate.py
│   └── train.py
├── utils/                            # Experiment tracking & tooling
│   ├── __init__.py
│   ├── checkpointing.py
│   ├── logging.py
│   └── metrics.py
├── .gitignore                        
├── README.md                         
└── requirements.txt