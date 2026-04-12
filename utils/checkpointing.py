

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch  # type: ignore
except Exception:           # pragma: no cover
    torch = None            # type: ignore


def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


@dataclass
class Checkpointer:
    """Save and load training checkpoints, one subfolder per run.

    Parameters
    ----------
    dirpath:
        Root directory under which per-run subfolders are created.
    run_id:
        Unique identifier for this training session.  Defaults to a UTC
        timestamp so separate runs never overwrite each other.
    filename:
        Default filename used when save()/load() are called without an
        explicit filename argument.
    run_args:
        A dictionary or OmegaConf object containing the run arguments to be
        stored in the checkpoint.
    """

    dirpath:  str = "checkpoints"
    run_id:   str = field(default_factory=_utc_timestamp)
    filename: str = "latest.pt"
    run_args: Optional[Dict[str, Any]] = None

    @property
    def run_dir(self) -> Path:
        """Absolute path to this run's subdirectory."""
        return Path(self.dirpath) / self.run_id

    def _ensure_dir(self) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def save(self, state: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Serialize `state` to disk and return the absolute path written."""
        if torch is None:
            raise RuntimeError("Checkpointing requires PyTorch.")
        self._ensure_dir()
        dest = self.run_dir / (filename or self.filename)
        
        # We create a new dict for the actual on-disk format
        to_save: Dict[str, Any] = {}
        
        # 1. Process the incoming state (models, buffers, etc.)
        for k, v in state.items():
            if hasattr(v, "state_dict") and callable(v.state_dict):
                to_save[k] = v.state_dict()
            else:
                to_save[k] = v
        
        # 2. Include run_args in the checkpoint if available
        if self.run_args is not None:
            to_save["run_args"] = self.run_args
        elif "run_args" in state:
            to_save["run_args"] = state["run_args"]
            
        torch.save(to_save, str(dest))
        return str(dest)

    def load(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """Load a checkpoint.

        If `filename` is an absolute path it is used directly, allowing
        eval.py and playback.py to load from any run's subfolder by passing
        the full path returned by save().
        """
        if torch is None:
            raise RuntimeError("Checkpointing requires PyTorch.")
        raw = Path(filename) if filename else None
        if raw is not None and raw.exists():
            # Path exists as given (absolute or relative to cwd) — use directly
            f = raw
        elif filename and os.path.isabs(filename):
            f = Path(filename)
        else:
            f = self.run_dir / (filename or self.filename)
        if not f.exists():
            raise FileNotFoundError(f"Checkpoint not found: {f}")
        return torch.load(str(f), map_location="cpu")