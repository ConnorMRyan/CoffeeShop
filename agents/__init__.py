"""Standard baseline agents for CoffeeShop MARL.

This package provides reference implementations (e.g., PPO, SAC) that can be
used directly or serve as baselines against the novel CoffeeShop architecture
in `core_marl`.
"""

from .ppo import PPOAgent  # noqa: F401
from .sac import SACAgent  # noqa: F401
