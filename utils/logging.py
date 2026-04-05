from __future__ import annotations

import logging
from typing import Optional

try:  # Prefer rich logging if available
    from rich.logging import RichHandler  # type: ignore
except Exception:  # pragma: no cover
    RichHandler = None  # type: ignore


_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str = "CoffeeShop", level: int = logging.INFO) -> logging.Logger:
    """Get a configured logger.

    - Uses RichHandler if available for nice console formatting.
    - Idempotent: calling multiple times returns the same configured logger.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        if RichHandler is not None:
            handler = RichHandler(rich_tracebacks=True, show_time=True, show_level=True, show_path=False)
            formatter = logging.Formatter("%(message)s")
        else:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    _LOGGERS[name] = logger
    return logger
