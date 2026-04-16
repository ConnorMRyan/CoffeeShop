# CoffeeShop: Asynchronous Social Experience Sharing via Off-Policy Mediation in Multi-Agent Reinforcement Learning

**Connor Ryan**
*CoffeeShop MARL v2.0*

---

## Abstract

On-policy algorithms such as Proximal Policy Optimization (PPO) offer training stability but are sample-inefficient in sparse-reward and procedurally generated environments. We present CoffeeShop, a multi-agent reinforcement learning (MARL) framework that combines the stability of on-policy gradient updates with selective off-policy experience sharing across a population of agents. The architecture draws on established ideas from Distributed Prioritized Experience Replay (APEX), Population-Based Training (PBT), and V-trace, integrating them with two targeted contributions: (1) a per-agent Learned Openness parameter ω that dynamically gates receptivity to shared experiences based on internal uncertainty, and (2) a population-level Gossip Factor η that suppresses inter-agent sharing during collective performance regression. We describe the architecture, motivate each design choice against prior work, and outline benchmark evaluation plans on AIsaac, Crafter, and Overcooked-AI.

---

## 1. Introduction: The Exploration-Stability Dilemma

In high-variance domains such as *The Binding of Isaac* (AIsaac) or long-horizon survival benchmarks (Crafter), agents frequently fail to encounter the rare trajectories necessary to learn complex strategies. Standard on-policy methods like PPO (Schulman et al., 2017) address the off-policy data problem by discarding experience after each gradient update, which preserves policy consistency at the cost of sample efficiency.

Several prior frameworks have addressed this tradeoff. APEX (Horgan et al., 2018) decouples acting from learning via a centralized prioritized replay buffer fed by distributed actors. IMPALA (Espeholt et al., 2018) similarly uses asynchronous actors with a centralized learner and corrects for off-policy data via the V-trace importance sampling estimator. Population-Based Training (PBT; Jaderberg et al., 2017) maintains an explicit population of agents and periodically copies weights and hyperparameters from higher-performing members to lower-performing ones.

CoffeeShop synthesizes these approaches for the specific case of sparse-reward MARL. Rather than wholesale policy replacement (PBT) or unconstrained off-policy gradients (APEX/IMPALA), it introduces a selective, agent-controlled receptivity mechanism. The core research question is whether dynamically gating inter-agent experience sharing — conditioned on both receiver uncertainty and sender quality — yields better sample efficiency than fixed-rate sharing, without the training instability that unconstrained off-policy data introduces into on-policy agents.

---

## 2. Related Work

### 2.1 Distributed Off-Policy RL

APEX (Horgan et al., 2018) is the most direct architectural precedent. It uses a centralized replay buffer with priority scheduling (Schaul et al., 2015) fed by many distributed actors. The CoffeeShop Mediator is structurally similar, with the key difference that the Mediator also acts as a per-receiver filter, rejecting transitions whose reward signal falls below the requesting agent's current capability baseline. This prevents expert agents from regressing on novice experience, a scenario not addressed by APEX's global priority ordering.

### 2.2 Asynchronous Actor-Learner Architectures

IMPALA (Espeholt et al., 2018) and SEED RL (Espeholt et al., 2019) both demonstrate scalable asynchronous training where actors generate trajectories that are consumed by a central learner. CoffeeShop differs in scope: it targets a population of on-policy agents that selectively absorb peer experience rather than a single learner consuming all actor data. The Learned Openness parameter ω (Section 3.1) serves a similar corrective function to V-trace importance weights, but operates at the trajectory-selection level rather than the gradient level.

### 2.3 Population-Based Training

PBT (Jaderberg et al., 2017) is the closest conceptual precedent to CoffeeShop's population dynamics. PBT copies weights and hyperparameters from the top-performing fraction of a population to the bottom-performing fraction at fixed intervals. CoffeeShop differs in two respects: (a) experience sharing is continuous and demand-driven rather than periodic and wholesale, and (b) the receiver agent controls its own receptivity rather than having weights overwritten by a scheduler.

### 2.4 Diversity in MARL

Policy diversity maintenance via Jensen-Shannon divergence has been explored in Quality-Diversity (QD) RL methods (Pugh et al., 2016; Cully & Demiris, 2017) and in explicit diversity-regularized MARL (Pacchiano et al., 2020). CoffeeShop's diversity probe (Section 4.2) applies a similar principle at the mediator level, penalizing population collapse without requiring explicit behavioral descriptors.

### 2.5 Social Learning in RL

The term 'Social Reinforcement Learning' has been used in prior work (e.g., Böhmer et al., 2020; Ndousse et al., 2021) to describe settings where agents learn by observing peers. CoffeeShop builds on this theme but frames the interaction at the trajectory level rather than through direct behavioral imitation or communication channels.

---

## 3. The Mediator Architecture

The CoffeeShop Mediator is a centralized experience broker. Unlike a standard FIFO replay buffer, it implements priority-based retention and per-receiver filtering.

### 3.1 Top-K Priority Buffer

Following Schaul et al. (2015) and the APEX architecture, the Mediator maintains a fixed-capacity min-heap indexed by trajectory priority. This ensures the buffer retains the highest-value experiences across population history, automatically evicting lower-priority entries as new data arrives. Unlike APEX's global priority ordering, the Mediator applies per-agent baseline filtering at query time: trajectories with reward signals below the requesting agent's current rolling average are excluded from the returned set. This is the primary novel element of the buffer design.

### 3.2 Synergy Bonus (α_syn)

The Mediator assigns a synergy bonus to sparse reward events, scaled by the performance spread of the population at the time of discovery:

$$\mathcal{B}_{syn} = \alpha_{syn} \cdot R_{sparse} \cdot \frac{\text{mean(peers)} - \text{min(peers)}}{\text{max(peers)} - \text{min(peers)}}$$

This formula is a normalized advantage scaled by a sparse reward signal. Normalized advantages are a standard technique in policy gradient methods (Schulman et al., 2017). The contribution here is applying this normalization to the priority assignment of shared trajectories rather than to gradient updates, with the intent of surfacing 'frontier discoveries' — transitions that represent a significant performance leap relative to the population.

---

## 4. Adaptive Integration: The Social Contract

The framework's central claim is that agent-controlled receptivity to shared experience outperforms fixed-rate sharing. Two mechanisms implement this.

### 4.1 Learned Openness (ω)

Each agent maintains a logit-space parameter ω ∈ [0.05, 0.75] representing its receptivity to Mediator-provided experience. ω is adjusted by two signals:

- **Internal Uncertainty:** High variance in the agent's value network (σ²_value) increases ω, causing the agent to seek external guidance when its own value estimates are unreliable. This is functionally similar to the uncertainty-driven exploration bonus in model-based RL methods (e.g., RMAX, Brafman & Tennenholtz, 2002), but applied to social receptivity rather than environmental exploration.

- **Mediator Quality:** If the Mediator's own critic loss is elevated, the upward adjustment to ω is dampened. This is analogous to the V-trace importance sampling clipping in IMPALA, which limits the influence of highly off-policy data. The difference is that CoffeeShop applies this clipping at the agent's decision to request data, rather than at the gradient computation step.

The primary distinction from V-trace is architectural: rather than correcting for off-policy gradients after the fact, ω gates whether off-policy data enters the agent's update at all. Whether this selection-based approach outperforms importance sampling correction is an empirical question that the benchmark evaluation (Section 6) is designed to address.

### 4.2 Gossip Factor (η)

When the population's rolling average reward drops below 70% of the All-Time Best (ATB), the Gossip Factor reduces effective openness for all agents:

$$\Omega_{effective} = \omega \cdot \eta$$

This is a performance-conditioned regularizer on the social learning rate. It bears conceptual similarity to adaptive learning rate schedules and the entropy annealing strategies used in PPO. The specific threshold of 70% ATB is a design choice that requires empirical validation; it is not derived from first principles.

---

## 5. Population Dynamics and Diversity

### 5.1 Per-Agent Baseline Filtering

As described in Section 3.1, the Mediator filters shared experience by receiver capability. This is motivated by curriculum learning principles (Bengio et al., 2009): presenting an agent with transitions that are too simple relative to its current capability provides no training signal. The per-agent baseline implements a dynamic difficulty threshold without requiring explicit curriculum design.

### 5.2 Jensen-Shannon Strategy Probe

To prevent the population from collapsing onto a single dominant strategy, CoffeeShop periodically computes the mean pairwise Jensen-Shannon divergence of all agent policies over a shared probe batch drawn from the Mediator. Agents whose policies fall too close to the population mode receive a diversity penalty. This approach is consistent with Quality-Diversity methods (Pugh et al., 2016) but operates as an online regularizer rather than an archive-based selection mechanism. The computational overhead of pairwise JS divergence scales quadratically with population size, which limits scalability and is noted as a limitation.

---

## 6. Benchmark Evaluation Plan

The CoffeeShop architecture is targeted at the following benchmark environments:

- **AIsaac:** Navigating combinatorial item synergies where a single discovery (e.g., Brimstone + Homing) changes the entire value landscape, creating sudden sparse reward events that require rapid policy revision.
- **Crafter:** Long-horizon tech tree progression where sparse rewards (e.g., crafting a Diamond Pickaxe) are rare and require sustained multi-step planning.
- **Overcooked-AI:** Coordinating adaptive strategies with varying peer capability levels, testing the Gossip Factor and Learned Openness mechanisms in a cooperative setting.

Planned ablations include: (a) fixed ω vs. learned ω, (b) global priority vs. per-agent baseline filtering, (c) with/without Gossip Factor, and (d) CoffeeShop vs. PBT and APEX baselines on sample efficiency and final performance. Results will be reported as mean ± standard deviation over five seeds.

---

## 7. Conclusion

CoffeeShop proposes a selective social experience sharing mechanism for multi-agent RL in sparse-reward environments. The architecture synthesizes established components — prioritized replay (Schaul et al., 2015), distributed actor-learner design (APEX, IMPALA), and population diversity maintenance (PBT, QD-RL) — and contributes two targeted mechanisms: Learned Openness (ω) as a demand-driven, uncertainty-conditioned receptivity gate, and the Gossip Factor (η) as a population-level sharing suppressor during collective regression. The primary empirical claim — that selection-based off-policy gating outperforms fixed-rate sharing and importance-sampling correction in sparse-reward domains — remains to be validated by the benchmark study outlined in Section 6.

---

## References

Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. *ICML*.

Böhmer, W., Kurin, V., & Whiteson, S. (2020). Deep coordination graphs. *ICML*.

Brafman, R. I., & Tennenholtz, M. (2002). R-MAX: A general polynomial time algorithm for near-optimal reinforcement learning. *JMLR*, 3, 213–231.

Cully, A., & Demiris, Y. (2017). Quality and diversity optimization: A unifying modular framework. *IEEE TEVC*.

Espeholt, L., et al. (2018). IMPALA: Scalable distributed deep-RL with importance weighted actor-learner architectures. *ICML*.

Espeholt, L., et al. (2019). SEED RL: Scalable and efficient deep-RL with accelerated central inference. *arXiv:1910.06591*.

Horgan, D., et al. (2018). Distributed prioritized experience replay (APEX). *ICLR*.

Jaderberg, M., et al. (2017). Population based training of neural networks. *arXiv:1711.09846*.

Ndousse, K., et al. (2021). Emergent social learning via multi-agent reinforcement learning. *ICML*.

Pacchiano, A., et al. (2020). On optimism in model-based reinforcement learning. *arXiv*.

Pugh, J. K., Soros, L. B., & Stanley, K. O. (2016). Quality diversity: A new frontier for evolutionary computation. *Frontiers in Robotics and AI*.

Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). Prioritized experience replay. *ICLR*.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.