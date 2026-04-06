"""Utilities stubs for CoffeeShop: minimal logger and metrics.

This module intentionally provides a very small surface so scripts/train.py
can run in environments without optional logging libraries.
"""
from __future__ import annotations

from typing import Dict, List


# ----------------------------------------------------------------------------
# Minimal logger
# ----------------------------------------------------------------------------
class _SimpleLogger:
    def info(self, data: Dict) -> None:
        """Prints structured dicts as `[STEP N] key=val ...`.

        If `step` is not provided, prints `-`.
        Non-step keys are printed in insertion order as `key=value`.
        """
        if not isinstance(data, dict):
            print(str(data))
            return
        step = data.get("step", "-")
        rest = " ".join(f"{k}={v}" for k, v in data.items() if k != "step")
        prefix = f"[STEP {step}]" if step is not None else "[STEP -]"
        print(f"{prefix} {rest}".strip())


def get_logger() -> _SimpleLogger:
    return _SimpleLogger()


# ----------------------------------------------------------------------------
# Minimal metrics
# ----------------------------------------------------------------------------
class Metrics:
    """Tracks lists of scalar metrics and computes means."""

    def __init__(self) -> None:
        self._hist: Dict[str, List[float]] = {}

    def update(self, data: Dict[str, float]) -> None:
        for k, v in data.items():
            self._hist.setdefault(k, []).append(float(v))

    def mean(self) -> Dict[str, float]:
        return {k: (sum(vs) / len(vs) if vs else 0.0) for k, vs in self._hist.items()}

    def clear(self) -> None:
        self._hist.clear()
