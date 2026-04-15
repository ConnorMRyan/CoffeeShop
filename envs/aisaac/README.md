# AIsaac Environment (Stub)

## Overview
AIsaac is a custom, procedurally generated multi-agent reinforcement learning environment designed to train swarms of agents to navigate the combinatorial item synergies and brutal mechanics inspired by *The Binding of Isaac*.

## Current Status
The environment is currently in a **stub** state. The implementation in `envs/aisaac/wrapper.py` provides the required `SocialEnvWrapper` API but does not yet contain the core game logic or procedural generation engine.

## Target API
AIsaac is designed for "True Multi-Agent" interaction from the start:
- **Observation Space:** High-dimensional vector representing local entity states, room layout, and current item synergies.
- **Action Space:** Discrete movement and combat actions.
- **Global Observation:** A concatenated view of all agent states, used by the `CoffeeShopMediator`.

## Integration
To use the full AIsaac environment, the `aisaac_env` package must be installed and wired into the `AIsaacWrapper`.

```python
# envs/aisaac/wrapper.py integration point
try:
    import aisaac_env
except ImportError:
    aisaac_env = None
```

## Future Work
- [ ] Procedural room generation engine integration.
- [ ] Item synergy graph implementation.
- [ ] Multi-agent swarm coordination tasks.
