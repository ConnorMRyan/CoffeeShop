from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Metrics:
    """Lightweight metrics accumulator.

    Example:
        m = Metrics()
        m.update({'reward': 1.0})
        mean = m.mean()
    """

    history: Dict[str, List[float]] = field(default_factory=dict)

    def update(self, values: Dict[str, float]) -> None:
        for k, v in values.items():
            self.history.setdefault(k, []).append(float(v))

    def mean(self) -> Dict[str, float]:
        return {k: (sum(vs) / len(vs) if vs else 0.0) for k, vs in self.history.items()}

    def clear(self) -> None:
        self.history.clear()
