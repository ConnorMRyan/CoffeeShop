from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SocialActorConfig:
    """Configuration for a SocialActor."""
    id: str


class SocialActor:
    """Represents a single agent participating in social (multi-agent) interaction.

    This is a lightweight abstraction that can be backed by any policy (baseline
    or CoffeeShop-specific). It delegates action selection to a provided agent.
    """

    def __init__(self, agent: Any, config: SocialActorConfig) -> None:
        self.agent = agent
        self.config = config

    @property
    def id(self) -> str:
        return self.config.id

    def act(self, obs: Dict[str, Any], deterministic: bool = False) -> Any:
        """Select an action for this actor given its own observation."""
        # Delegate to underlying `agent` which may be shared among actors.
        # For simple baselines, `agent.act` can accept multi-agent obs; here we
        # just pass a single-agent dict keyed by our id for consistency.
        return self.agent.act({self.id: obs}, deterministic=deterministic).get(self.id)
