from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List
import polars as pl
import numpy as np

@dataclass
class Metrics:
    """Lightweight metrics accumulator using Polars for high-performance aggregation.
    """

    history: Dict[str, List[float]] = field(default_factory=dict)

    def update(self, values: Dict[str, float]) -> None:
        for k, v in values.items():
            self.history.setdefault(k, []).append(float(v))

    def mean(self) -> Dict[str, float]:
        if not self.history:
            return {}
        
        results = {}
        for k, v in self.history.items():
            if not v:
                continue
            # Use Polars for faster mean calculation on individual series
            # Or just use numpy for simplicity if series are small
            results[k] = float(np.mean(v))
        return results

    def clear(self) -> None:
        self.history.clear()

    def _to_df(self) -> pl.DataFrame:
        """Convert current history to a Polars DataFrame, padding ragged columns with NaN."""
        if not self.history:
            return pl.DataFrame()
        max_len = max(len(v) for v in self.history.values())
        padded = {
            k: v + [float("nan")] * (max_len - len(v))
            for k, v in self.history.items()
        }
        return pl.DataFrame(padded)

    def flush(self, output_path: str) -> None:
        """Append current history to a parquet file on disk, then clear in-memory state.

        Appending is done by reading the existing file (if any), concatenating with the
        current window, and rewriting. The diagonal concat handles the case where a
        metric first appears mid-run (missing columns are filled with null).

        Call this periodically (e.g. every 10 000 steps) so that:
          - In-memory accumulation stays bounded (no RAM growth over 1B steps).
          - A crash doesn't lose the full training history.
        """
        if not self.history:
            return
        new_df = self._to_df()
        if os.path.exists(output_path):
            existing = pl.read_parquet(output_path)
            combined = pl.concat([existing, new_df], how="diagonal_relaxed")
            combined.write_parquet(output_path)
        else:
            new_df.write_parquet(output_path)
        self.history.clear()

    def report_final(self, output_path: str = "metrics.parquet") -> None:
        """Flush any remaining in-memory metrics to disk. Alias for flush()."""
        self.flush(output_path)
