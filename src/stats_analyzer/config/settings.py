from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class AnalyzerSettings:
    missing_policy: Literal["allow", "drop", "error"] = "allow"
    min_group_size: int = 3
    min_total_n: int = 10
    min_n_per_factor: int = 2
    min_n_per_group: int = 3
    default_alpha: float = 0.05
    default_validation_alpha: float = 0.05
    default_assumption_alpha: float = 0.05
    covariance_type: str = "HC3"
    validation_mode: Literal["univariate", "joint"] = "univariate"
    categorical_validation_mode: Literal["rules", "joint", "both"] = "both"
    figure_dpi: int = 150
    figure_format: str = "png"

    @classmethod
    def from_yaml(cls, path: Path | None) -> "AnalyzerSettings":
        if path is None:
            return cls()
        if not path.exists():
            raise FileNotFoundError(f"Settings file not found: {path}")

        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        return cls(**payload)
