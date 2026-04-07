# CoffeeShop: Architecture vs. Implementation Gap Analysis

Comparing `ARCHITECTURE.md` promises against `core_marl/` and `agents/ppo.py`.
Gaps are ordered roughly by theoretical impact.

---

## 1. Population-Relative Recency Decay (γ_time) — Not Implemented

**What the paper says:**
> "Obsolete strategies are purged via a population-relative recency decay (γ_time). Memories are not decayed purely by step-count, but by the moving average of the population's baseline capability. As the society of agents improves, beginner-level success stories are purged to prevent regressive training."

**What exists:**
The buffer is a FIFO `deque(maxlen=10_000)` — it evicts the *oldest* entry by insertion order, not the lowest-quality one. There is no γ_time variable anywhere in the code. `Transition.timestamp` is stored (`mediator.py:69`) but is never read.

The `_is_stale` check (`mediator.py:368`) is a binary threshold against the requester's current baseline — this is a staleness *filter*, not a decay. The recency penalty (`_requester_recency_penalty`, `mediator.py:357`) is a linear push-down, not a time-based exponential.

**Improvement ideas:**
- Replace FIFO eviction with priority-based eviction: track a per-entry `priority` score and evict the lowest-priority entry instead of the oldest.
- Implement γ_time as a time-decay weight: `priority_effective = priority * exp(-λ * (now - transition.timestamp))`. The `timestamp` field already exists on `Transition`.
- Use `_agent_baselines` as the "society baseline" denominator: a memory should have its effective priority multiplied by `max(0, signal / population_baseline)`, shrinking toward zero as the population improves beyond that memory's signal.

---

## 2. TD Error Uses Actor's V(s), Not the Centralized Critic — Architectural Mismatch

**What the paper says:**
> "The Mediator evaluates these episodic memories, calculating the Temporal Difference error to identify highly novel or extreme-reward events."

The whole point of a centralized off-policy critic is to compute a richer TD signal from the global state.

**What exists:**
`_compute_td_and_bonus()` (`mediator.py:289`) computes:
```python
v_next = 0.0 if t.done else t.value_est   # actor estimate; critic not yet updated
td_error = abs((reward + social_bonus) + gamma * v_next - t.value_est)
```
This is `|r + γ·V_actor(s) − V_actor(s)|`, which degenerates to `|r + (γ−1)·V_actor(s)|`. It uses the *actor's local* V(s) for both the target and the baseline — the centralized critic plays no role in priority scoring at push time.

**Improvement ideas:**
- Call `self.critic(t.global_obs.unsqueeze(0))` inside `_compute_td_and_bonus()` to get `V_critic(s_global)` and use that as the baseline. Use `self.critic(t.next_global_obs.unsqueeze(0))` for `V(s')`.
- Cache the critic's `V(s)` estimate at push time and update priorities in the buffer retroactively after each `update_critic()` call (a common technique in PER implementations).
- In the short term, even using `t.value_est` as the baseline but replacing `v_next` with a critic forward pass would be a meaningful improvement.

---

## 3. ExperienceBuffer Is Dead Code

**What the paper implies:**
`core_marl/experience_buffer.py` is listed as "Centralized prioritized replay" in `ARCHITECTURE.md`.

**What exists:**
`ExperienceBuffer` is never instantiated anywhere in the codebase. The Mediator uses its own `deque[ScoredMemory]`, and `PPOAgent` has a separate `RolloutBuffer`. `ExperienceBuffer` and `ExperienceBatch` have zero callers.

**Improvement ideas:**
- Either delete `ExperienceBuffer` to reduce confusion, or integrate it as the shared buffer backing `CoffeeShopMediator` (replacing the raw `deque[ScoredMemory]`). The latter would unify the two buffer concepts the architecture intended.

---

## 4. SocialActor (core_marl) Is Never Used

**What the paper implies:**
`core_marl/social_actor.py` is the architectural abstraction that binds any policy to an agent ID.

**What exists:**
`scripts/train.py` imports PPOAgent directly with an alias:
```python
from agents.ppo import PPOAgent as SocialActor
```
`core_marl/social_actor.py` is never instantiated. Its `act()` signature (`obs: Dict[str, Any]`) is also incompatible with the actual `PPOAgent.act(env_id, obs, global_obs)` call site.

**Improvement ideas:**
- Update `SocialActor` to match the `PPOAgent` interface (or define an abstract base class / protocol), then wire `train.py` to instantiate `SocialActor(agent=PPOAgent(...), config=SocialActorConfig(id=aid))` so the abstraction is actually load-bearing.
- Alternatively, delete `core_marl/social_actor.py` if the intent is to keep PPOAgent as the direct actor — the dead wrapper just adds confusion.

---

## 5. Learned Openness (ω) — Dual-Update Mechanism Is Inconsistent

**What the paper says:**
> "ω is a learned parameter tied to the agent's internal Value network variance."

"Learned" implies gradient-based optimization.

**What exists:**
`LearnedOpenness` is in the optimizer (`ppo.py:277`) so it *is* differentiated. But `update_from_variance()` (`ppo.py:113`) also manually calls `_raw.add_(lr)` / `_raw.sub_(lr)` with a fixed step every PPO update. These two mechanisms fight each other: the gradient learns one thing, and the heuristic immediately overwrites it.

Additionally, `force_reset()` hard-sets `_raw = 0.0` whenever the agent is simultaneously starved of sparse rewards and has near-zero value variance, which overrides everything the gradient learned.

**Improvement ideas:**
- Remove `update_from_variance()` and let gradient descent alone drive ω. To preserve the "high variance = more open" inductive bias, add a regularization term to the loss: e.g. `L_omega_reg = lambda * (omega - sigmoid(value_variance / threshold))^2`.
- If the heuristic update is kept, at minimum stop including `openness.parameters()` in the Adam optimizer — it creates a confusing hybrid that the paper does not describe.
- Document which update path is authoritative and test that ω actually learns a meaningful signal (e.g., log `omega` vs `value_variance` scatter plots).

---

## 6. Gossip Factor Is Binary, Not Continuous

**What the paper implies:**
A dynamic trust multiplier that smoothly reduces peer influence when the group underperforms.

**What exists:**
`get_gossip_factor()` (`mediator.py:151`) returns either `1.0` or `_gossip_alpha` (hardcoded `0.7`) — a hard switch triggered when `current_avg < 0.7 * atb_reward`. There is no interpolation between the two states.

The 0.7 multiplier for both the threshold (`0.7 * atb_reward`) and the gossip factor (`_gossip_alpha = 0.7`) are coincidental magic numbers with no architectural justification.

**Improvement ideas:**
- Replace the binary with a continuous decay: `gossip_factor = clip(current_avg / atb_reward, _gossip_alpha, 1.0)`. This scales trust proportionally to how far performance has fallen.
- Separate the detection threshold from the penalty magnitude so they can be tuned independently.
- Consider per-env gossip factors (the `_env_reward_windows` dict already exists but is only used for reputation boosts, not gossip gating).

---

## 7. Buffer Is Not Truly Prioritized — FIFO Eviction Discards High-Value Memories

**What the paper says:**
> "Centralized prioritized replay buffer."

**What exists:**
`deque(maxlen=10_000)` evicts from the *left* (oldest entry) when full. A transition with priority 100 pushed 11,000 steps ago will be dropped before a priority-0 transition pushed one step ago.

**Improvement ideas:**
- Use a min-heap or sorted structure keyed by priority so the lowest-priority memory is evicted when capacity is reached.
- A simple approximation: periodically (e.g., every 1,000 pushes) compact the buffer by sorting and truncating the bottom half by priority.

---

## 8. Social Bonus Only Fires on Sparse Reward Events

**What exists:**
```python
# mediator.py:292
if t.sparse_reward > 0.0:
    social_bonus = self._compute_synergy_bonus(...)
```
Dense-reward steps contribute zero social bonus regardless of team synergy.

**What the paper implies:**
Social bonus should reflect any peer contribution to team outcomes, not just delivery events.

**Improvement ideas:**
- Compute a scaled synergy bonus for all steps: `social_bonus = synergy_alpha * t.reward * synergy_score` (clamped to prevent over-inflation on large shaped rewards).
- Keep the existing sparse-event path as a larger bonus trigger, and add a smaller continuous bonus for dense steps.

---

## 9. Critic TD Target in `update_critic` Excludes Social Bonus

**What exists:**
```python
# mediator.py:272
td_targets = rwds + self.gamma * v_next * (1.0 - dones)
```
`rwds` is `t.reward` — it does not include the social bonus that was computed at push time and credited to the agent.

**Why it matters:**
The centralized critic is supposed to model the *social* value of global states. If the social bonus is not in the training target, the critic learns to predict raw env reward and the social bonus becomes a ghost signal that the critic never accounts for.

**Improvement ideas:**
- Add social_bonus to the TD target: `rwds = t.reward + scored_memory.social_bonus`.
- This requires storing `ScoredMemory` in the batch passed to `update_critic()` rather than raw `Transition` objects, or adding a `social_bonus` field to `Transition`.

---

## 10. `broadcast()` Creates a New RNG Instance on Every Call

**What exists:**
```python
# mediator.py:247
rng = np.random.default_rng()
```
Called inside `broadcast()`, which runs every `pull_every` PPO updates. Each call creates a new unseeded RNG, making experiments non-reproducible even with a global seed.

**Improvement ideas:**
- Initialize `self._rng = np.random.default_rng(seed)` in `__init__` and use `self._rng` in `broadcast()`.
- Thread the seed through `CoffeeShopMediator.__init__` so experiments are fully reproducible.

---

## Summary Table

| # | Area | Gap Severity | Effort to Fix |
|---|------|-------------|---------------|
| 1 | γ_time / population-relative decay | High — core architectural claim unimplemented | Medium |
| 2 | TD error uses actor V, not centralized critic | High — defeats the purpose of the centralized critic at push time | Medium |
| 3 | ExperienceBuffer is dead code | Medium — misleads readers; wastes complexity | Low (delete or integrate) |
| 4 | SocialActor never used | Medium — abstraction layer is bypassed | Low |
| 5 | ω dual-update is inconsistent | Medium — gradient and heuristic fight each other | Low |
| 6 | Gossip factor is binary, not continuous | Low-Medium — blunt signal | Low |
| 7 | Buffer evicts by age, not priority | Medium — defeats "prioritized" replay claim | Medium |
| 8 | Social bonus gated on sparse reward only | Low — misses dense-reward synergy | Low |
| 9 | Critic TD target excludes social bonus | Medium — critic doesn't model social value | Low |
| 10 | New RNG per `broadcast()` call | Low — reproducibility only | Trivial |
