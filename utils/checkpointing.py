from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore


@dataclass
class Checkpointer:
    """Checkpoint utility with optional GCS sync.

    Local saves are always written first.  When ``gcs_bucket`` is set the
    file is immediately uploaded to GCS so it survives instance preemption.
    On resume, if no local checkpoint is found the latest one is pulled from
    GCS automatically.

    GCS auth is handled by the environment (ADC, a mounted service-account
    JSON pointed to by GOOGLE_APPLICATION_CREDENTIALS, or the VM's attached
    service account — whichever is present).  ``google-cloud-storage`` must
    be installed; if it is not, GCS operations are silently skipped.

    Parameters
    ----------
    dirpath:    Local directory for checkpoint files.
    filename:   Default filename used when no explicit name is supplied.
    gcs_bucket: GCS bucket name (e.g. ``"my-training-bucket"``).
                Set to ``None`` (default) to disable cloud sync.
    gcs_prefix: Path prefix inside the bucket (e.g. ``"runs/nethack-ppo"``).
    """

    dirpath:    str           = "checkpoints"
    filename:   str           = "latest.pt"
    gcs_bucket: Optional[str] = None
    gcs_prefix: str           = "checkpoints"

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_dir(self) -> Path:
        p = Path(self.dirpath)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _gcs_client(self):
        """Return a GCS client, or None if the library is not installed."""
        try:
            from google.cloud import storage as gcs  # type: ignore
            return gcs.Client()
        except ImportError:
            return None

    def _blob_name(self, filename: str) -> str:
        return f"{self.gcs_prefix.rstrip('/')}/{filename}"

    def _gcs_upload(self, local_path: str, filename: str) -> None:
        """Upload *local_path* to GCS as *filename* under the configured prefix."""
        client = self._gcs_client()
        if client is None or not self.gcs_bucket:
            return
        try:
            bucket = client.bucket(self.gcs_bucket)
            bucket.blob(self._blob_name(filename)).upload_from_filename(local_path)
        except Exception as exc:  # pragma: no cover
            print(f"[Checkpointer] GCS upload failed (non-fatal): {exc}")

    def _gcs_download(self, filename: str, local_path: str) -> bool:
        """Download *filename* from GCS to *local_path*.  Returns True on success."""
        client = self._gcs_client()
        if client is None or not self.gcs_bucket:
            return False
        try:
            bucket = client.bucket(self.gcs_bucket)
            blob = bucket.blob(self._blob_name(filename))
            if not blob.exists():
                return False
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(local_path)
            return True
        except Exception as exc:  # pragma: no cover
            print(f"[Checkpointer] GCS download failed (non-fatal): {exc}")
            return False

    def _latest_gcs_filename(self) -> Optional[str]:
        """Return the filename of the highest-numbered checkpoint in GCS, or None."""
        client = self._gcs_client()
        if client is None or not self.gcs_bucket:
            return None
        try:
            bucket = client.bucket(self.gcs_bucket)
            prefix = f"{self.gcs_prefix.rstrip('/')}/checkpoint_"
            blobs = sorted(bucket.list_blobs(prefix=prefix), key=lambda b: b.name)
            if not blobs:
                return None
            # Filenames are checkpoint_<step>.pt — lexicographic sort is correct
            # because step numbers are zero-padded... actually they're not, so we
            # parse them to be safe.
            def _step(blob) -> int:
                try:
                    return int(Path(blob.name).stem.split("_")[-1])
                except ValueError:
                    return -1
            return Path(max(blobs, key=_step).name).name
        except Exception as exc:  # pragma: no cover
            print(f"[Checkpointer] GCS listing failed (non-fatal): {exc}")
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    def save(self, state: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Save *state* locally and, if GCS is configured, upload to GCS.

        Values that have a ``state_dict()`` method (nn.Module, Optimizer) are
        serialised via that method; everything else is stored as-is.
        """
        if torch is None:
            raise RuntimeError("Checkpointing requires PyTorch; please install torch.")
        self._ensure_dir()
        fname = filename or self.filename
        local_path = str(Path(self.dirpath) / fname)
        to_save: Dict[str, Any] = {
            k: v.state_dict() if hasattr(v, "state_dict") and callable(v.state_dict) else v
            for k, v in state.items()
        }
        torch.save(to_save, local_path)
        self._gcs_upload(local_path, fname)
        return local_path

    def load(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """Load a checkpoint.

        Tries the local directory first.  If the file does not exist locally
        and GCS is configured, downloads it from GCS before loading.

        weights_only=False is required to deserialise optimizer state dicts,
        which contain Python longs (Adam's step counter).  Only load
        checkpoints from trusted sources.
        """
        if torch is None:
            raise RuntimeError("Checkpointing requires PyTorch; please install torch.")
        fname = filename or self.filename
        local_path = str(Path(self.dirpath) / fname)
        if not os.path.exists(local_path):
            downloaded = self._gcs_download(fname, local_path)
            if not downloaded:
                raise FileNotFoundError(
                    f"Checkpoint not found locally or in GCS: {fname}"
                )
        return torch.load(local_path, map_location="cpu", weights_only=False)

    def latest(self) -> Optional[str]:
        """Return the filename of the highest-numbered local checkpoint.

        If the local directory is empty *and* GCS is configured, downloads
        the latest checkpoint from GCS and returns its filename so the caller
        can pass it straight to ``load()``.
        """
        p = Path(self.dirpath)
        if p.exists():
            pts = sorted(p.glob("checkpoint_*.pt"))
            if pts:
                return pts[-1].name

        # Nothing local — try GCS
        gcs_fname = self._latest_gcs_filename()
        if gcs_fname is None:
            return None
        local_path = str(Path(self.dirpath) / gcs_fname)
        print(f"[Checkpointer] Pulling latest checkpoint from GCS: {gcs_fname}")
        self._gcs_download(gcs_fname, local_path)
        return gcs_fname if os.path.exists(local_path) else None
