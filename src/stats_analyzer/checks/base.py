from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from stats_analyzer.core.models import ModelSpec, ValidationResult


class BaseCheck(ABC):
    name = "base-check"
    assumptions: list[str] = []

    @abstractmethod
    def run(self, dataset: Any, model_spec: ModelSpec) -> ValidationResult:
        raise NotImplementedError

    def describe(self) -> str:
        lines = [f"Check: {self.name}", "Assumptions:"]
        lines.extend(f"- {entry}" for entry in self.assumptions)
        return "\n".join(lines)

