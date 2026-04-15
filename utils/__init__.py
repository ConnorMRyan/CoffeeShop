"""Utilities for CoffeeShop: logger, metrics, and TensorBoard writer.

This module intentionally provides a very small surface so coffeeshop/train.py
can run in environments without optional logging libraries.
"""
from __future__ import annotations
from .metrics import Metrics
import os
from typing import Dict, List, Optional


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
# TensorBoard writer (optional dependency)
# ----------------------------------------------------------------------------
class TBWriter:
    """Thin wrapper around torch.utils.tensorboard.SummaryWriter.

    Degrades gracefully to a no-op if TensorBoard is not installed, so the
    rest of the codebase never needs to guard TB calls with try/except.

    Usage:
        tb = TBWriter(log_dir="checkpoints/my_run/tb")
        tb.add_scalar("train/reward", 1.23, step=100)
        tb.add_scalars({"loss/ppo": 0.5, "loss/vf": 0.1}, step=100)
        tb.close()
    """

    def __init__(self, log_dir: str) -> None:
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            self._writer = SummaryWriter(log_dir=log_dir)
            parent = os.path.dirname(os.path.abspath(log_dir))
            print(f"[TBWriter] Logging to {log_dir}")
            print(f"[TBWriter] Visualize with: tensorboard --logdir {parent}")
        except ImportError:
            print(
                "[TBWriter] TensorBoard not installed — console-only logging.\n"
                "           Install with: pip install tensorboard"
            )
    def flush(self) -> None:
        """Force write pending events to disk (critical for WSL/Network drives)."""
        if self._writer is not None:
            self._writer.flush()
            
    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, float(value), global_step=step)

    def add_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """Write every key/value pair in `metrics` as a separate scalar."""
        if self._writer is None:
            return
        for tag, value in metrics.items():
            self._writer.add_scalar(tag, float(value), global_step=step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


# ----------------------------------------------------------------------------
# Weights & Biases writer (optional dependency)
# ----------------------------------------------------------------------------
class WandbWriter:
    """Thin wrapper around wandb.

    Degrades gracefully to a no-op if wandb is not installed or not used.
    """

    def __init__(self, config: Dict, project: str, entity: Optional[str] = None, run_id: Optional[str] = None, use_wandb: bool = False) -> None:
        self._active = False
        if not use_wandb:
            return

        try:
            import wandb
            wandb.init(
                project=project,
                entity=entity,
                config=config,
                id=run_id,
                resume="allow"
            )
            self._active = True
            print(f"[WandbWriter] Logging to project: {project}")
        except ImportError:
            print("[WandbWriter] wandb not installed — skipping W&B logging.")

    def log(self, metrics: Dict[str, float], step: int) -> None:
        if self._active:
            import wandb
            wandb.log(metrics, step=step)

    def close(self) -> None:
        if self._active:
            import wandb
            wandb.finish()
