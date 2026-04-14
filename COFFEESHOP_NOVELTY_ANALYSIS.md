# CoffeeShop: Asynchronous Social Experience Sharing via Off-Policy Mediation

**Author:** Connor Ryan (Autonomous System Implementation)
**Framework:** CoffeeShop MARL v2.0

## Abstract
Standard on-policy reinforcement learning algorithms like Proximal Policy Optimization (PPO) provide high stability but suffer from extreme sample inefficiency and local optima stagnation in complex, sparse-reward, and procedurally generated environments. We present **CoffeeShop**, a novel multi-agent reinforcement learning (MARL) framework that bridges the gap between on-policy stability and off-policy discovery through a mechanism termed **Asynchronous Social Experience Sharing (ASES)**. By utilizing a centralized **Mediator** to broker high-value experiences across a decentralized population of actors, CoffeeShop enables the discovery of global optima while preserving actor stability via adaptive cognitive gating (Learned Openness, $\omega$).

---

## 1. Introduction: The Exploration-Stability Dilemma
In high-variance domains such as *The Binding of Isaac* (AIsaac) or survival benchmarks (Crafter), agents often fail to encounter the rare trajectories required to learn complex strategies. Forcibly introducing off-policy data into an on-policy gradient update typically leads to network collapse or catastrophic forgetting. CoffeeShop addresses this by treating other agents in the population not as "opponents" or "partners" in a single environment, but as **social peers** whose successful "memories" can be selectively integrated.

---

## 2. The Mediator Architecture: The Marketplace of Trajectories
The **CoffeeShop Mediator** acts as a centralized off-policy critic and experience broker. Unlike a standard replay buffer, it implements several novel mechanisms:

### 2.1 Top-K Priority Buffer (`PrioritizedBuffer`)
Instead of a First-In-First-Out (FIFO) queue, the Mediator maintains a fixed-capacity **Min-Heap**. This ensures that the buffer always contains the highest-priority trajectories across the entire population's history. As agents improve, lower-value "early success" stories are naturally purged, maintaining a frontier of expert-level demonstrations.

### 2.2 Synergy Bonus ($\alpha_{syn}$)
The Mediator calculates a **Synergy Bonus** when an agent triggers a sparse reward event. This bonus is scaled by the performance spread of its peers:
$$ \mathcal{B}_{syn} = \alpha_{syn} \cdot R_{sparse} \cdot \frac{\text{mean}(\text{peers}) - \text{min}(\text{peers})}{\text{max}(\text{peers}) - \text{min}(\text{peers})} $$
This rewards "frontier discovery" that significantly outperforms the current social average, incentivizing collaborative exploration of the state-action space.

---

## 3. Adaptive Integration: The Social Contract
CoffeeShop's most critical innovation is how actors decide which experiences to "listen" to.

### 3.1 Learned Openness ($\omega$)
Each agent maintains a personal "Openness" parameter $\omega \in [0.05, 0.75]$. This is a logit-space parameter that adjusts based on two factors:
1.  **Internal Uncertainty**: High value-network variance ($\sigma^2_{value}$) suggests the agent is lost, triggering an increase in $\omega$ to seek social guidance.
2.  **Mediator Quality**: If the Mediator's own critic loss is high (unreliable signal), the agent reduces its "upward nudge" to $\omega$, protecting itself from poor advice.

### 3.2 Gossip Factor ($\eta$)
The framework implements a global "Skepticism" layer called the **Gossip Factor**. When the population's current rolling average reward drops below **70% of the All-Time Best (ATB)**, the Gossip Factor restricts the effective openness of all agents:
$$ \Omega_{effective} = \omega \cdot \eta $$
This prevents the "Blind leading the Blind" scenario where agents share regressive behaviors during a collective performance slump.

---

## 4. Population Dynamics & Diversity
### 4.1 Population-Relative Recency Decay
Memories in CoffeeShop do not decay purely by time. Instead, the Mediator uses **Per-Agent Baselines**. When an agent requests memories, the Mediator filters out transitions where the reward signal is lower than the *receiver's* current capability. This ensures that "expert" agents are not distracted by "novice" data, even if that data was high-priority for the sender at the time of collection.

### 4.2 Jensen-Shannon Strategy Probe
To prevent the population from collapsing into a single, potentially sub-optimal strategy, CoffeeShop monitors **Population Diversity**. Using a shared probe batch sampled from the Mediator, the framework calculates the mean pairwise **Jensen-Shannon (JS) Divergence** of all agent policies. The Mediator can then "enforce diversity" by penalizing agents that mirror a single dominant strategy too closely.

---

## 5. Benchmark Performance: AIsaac and Beyond
The CoffeeShop architecture is specifically tailored for:
-   **AIsaac**: Navigating combinatorial item synergies where a single discovery (e.g., Brimstone + Homing) changes the entire value landscape.
-   **Crafter**: Long-horizon tech tree progression where sparse rewards (e.g., crafting a Diamond Pickaxe) are rare and difficult to replicate.
-   **Overcooked-AI**: Coordinating adaptive strategies with varying peer capability levels.

## 6. Conclusion
CoffeeShop represents a shift from "Multi-Agent Learning" to **"Social Reinforcement Learning."** By decoupling exploration (decentralized actors) from evaluation (centralized mediator) and gating the flow of information through learned cognitive thresholds, the framework achieves a state-of-the-art balance between sample efficiency and training stability.
