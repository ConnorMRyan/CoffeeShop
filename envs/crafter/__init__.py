"""Crafter environment wrapper package for CoffeeShop.

Provides a thin adapter to normalize Crafter to the `SocialEnvWrapper` API.
Implementation is optional until Crafter is enabled in requirements.
"""

from .wrapper import CrafterSocialWrapper  # noqa: F401
from .wrapper import CrafterWrapper        # noqa: F401  (backward-compat alias)
