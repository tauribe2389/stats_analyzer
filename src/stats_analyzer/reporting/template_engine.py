from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TEMPLATE: dict[str, Any] = {
    "title": "Statistical Analysis Report",
    "subtitle": "ANOVA / ANCOVA workflow output",
    "sections": [
        "executive_summary",
        "data_quality",
        "validation",
        "modeling",
        "assumptions",
        "tables",
        "figures",
    ],
    "include_flag_table": True,
    "include_assumption_notes": True,
    "tables": {
        "include_defaults": True,
        "defaults": [
            "flags",
            "fit_statistics",
            "parameter_estimates",
            "adjusted_means",
            "assumption_metrics",
        ],
        "custom": [],
        "default_formats": {},
        "preview_rows": 20,
        "pdf_max_rows": 20,
        "pdf_max_columns": 8,
        "pdf_overflow_mode": "auto",
        "pdf_overflow_placement": "inline",
        "pdf_truncate_chars": 80,
        "pdf_min_col_width": 52,
        "pdf_min_col_width_truncated": 32,
        "pdf_max_col_width": 240,
        "pdf_split_key_columns": 1,
        "pdf_split_orientation": "auto",
    },
}


class TemplateEngine:
    """Load and merge user template settings with defaults."""

    def load(self, template_path: Path | None) -> dict[str, Any]:
        merged = deepcopy(DEFAULT_TEMPLATE)
        if template_path is None:
            return merged
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        with template_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        self._deep_merge(merged, payload)
        return merged

    def _deep_merge(self, base: dict[str, Any], updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
