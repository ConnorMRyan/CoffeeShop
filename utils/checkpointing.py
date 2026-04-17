from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


@dataclass
class Checkpointer:
    """Simple checkpoint utility.

    Saves/loads dictionaries of states. If PyTorch is available and a value is a
    torch.nn.Module or optimizer, it will use their `state_dict()` / `load_state_dict()`.
    Otherwise stores raw Python objects via `torch.save` if torch exists, or raises.
    """

    dirpath: str = "checkpoints"
    filename: str = "latest.pt"

    def _ensure_dir(self) -> Path:
        p = Path(self.dirpath)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save(self, state: Dict[str, Any], filename: Optional[str] = None) -> str:
        if torch is None:
            raise RuntimeError("Checkpointing requires PyTorch; please install torch.")
        self._ensure_dir()
        f = str(Path(self.dirpath) / (filename or self.filename))
        to_save: Dict[str, Any] = {}
        for k, v in state.items():
            if hasattr(v, "state_dict") and callable(getattr(v, "state_dict")):
                to_save[k] = v.state_dict()
            else:
                to_save[k] = v
        torch.save(to_save, f)
        return f

    def load(self, filename: Optional[str] = None) -> Dict[str, Any]:
        if torch is None:
            raise RuntimeError("Checkpointing requires PyTorch; please install torch.")
        f = str(Path(self.dirpath) / (filename or self.filename))
        if not os.path.exists(f):
            raise FileNotFoundError(f"Checkpoint not found: {f}")
        data: Dict[str, Any] = torch.load(f, map_location="cpu", weights_only=True)
        return data
