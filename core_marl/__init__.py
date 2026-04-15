"""CoffeeShop core multi-agent RL architecture.

This package contains the novel CoffeeShop components: mediator, social actors,
experience buffer, and associated glue to interface with env wrappers.
"""

from .mediator import Mediator as CoffeeShopMediator

__all__ = ["CoffeeShopMediator"]
