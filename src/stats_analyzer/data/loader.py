from __future__ import annotations

from pathlib import Path
from typing import Any

from stats_analyzer.config.settings import AnalyzerSettings


class DataLoader:
    """Read tabular datasets from disk."""

    def __init__(self, settings: AnalyzerSettings | None = None) -> None:
        self.settings = settings or AnalyzerSettings()

    def load(self, path: Path) -> Any:
        import pandas as pd

        if not path.exists():
            raise FileNotFoundError(f"Input dataset not found: {path}")

        suffix = path.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(path)
        elif suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(path)
        elif suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported dataset extension: {suffix}")

        return frame

