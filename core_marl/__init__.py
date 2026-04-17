"""CoffeeShop core multi-agent RL architecture.

This package contains the novel CoffeeShop components: mediator, social actors,
experience buffer, and associated glue to interface with env wrappers.
"""

from .mediator import CoffeeShopMediator
from .social_actor import SocialActor, SocialActorConfig
from .experience_buffer import ExperienceBuffer, SharedExperienceBuffer

__all__ = [
    "CoffeeShopMediator",
    "SocialActor",
    "SocialActorConfig",
    "ExperienceBuffer",
    "SharedExperienceBuffer"
]
