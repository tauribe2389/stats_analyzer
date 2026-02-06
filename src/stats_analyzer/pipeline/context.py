from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from stats_analyzer.core.models import (
    AnalysisRequest,
    FigureArtifact,
    Flag,
    ModelResult,
    ModelSpec,
    TableArtifact,
    ValidationResult,
)


@dataclass
class PipelineContext:
    request: AnalysisRequest
    dataset: Any | None = None
    model_spec: ModelSpec | None = None
    split_datasets: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    validations: dict[str, ValidationResult] = field(default_factory=dict)
    model_result: ModelResult | None = None
    figures: list[FigureArtifact] = field(default_factory=list)
    tables: list[TableArtifact] = field(default_factory=list)
    flags: list[Flag] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def add_validation(self, key: str, result: ValidationResult) -> None:
        self.validations[key] = result
        self.flags.extend(result.flags)

    def add_model_result(self, result: ModelResult) -> None:
        self.model_result = result
        self.flags.extend(result.flags)

    def add_figure(self, artifact: FigureArtifact) -> None:
        self.figures.append(artifact)

    def add_table(self, artifact: TableArtifact) -> None:
        self.tables.append(artifact)
