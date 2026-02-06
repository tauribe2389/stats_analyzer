from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Severity = Literal["INFO", "WARN", "ERROR"]
AnalysisType = Literal["anova", "ancova"]
ValidationMode = Literal["univariate", "joint"]
CategoricalValidationMode = Literal["rules", "joint", "both"]


@dataclass
class Flag:
    code: str
    message: str
    severity: Severity = "WARN"
    stage: str = "unknown"
    variables: list[str] = field(default_factory=list)
    recommendation: str | None = None


@dataclass
class ValidationResult:
    name: str
    passed: bool
    summary: str = ""
    flags: list[Flag] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)


@dataclass
class ModelSpec:
    analysis_type: AnalysisType
    response: str
    primary_factor: str
    covariates: list[str] = field(default_factory=list)
    group_variables: list[str] = field(default_factory=list)
    interaction_terms: list[str] = field(default_factory=list)
    alpha: float = 0.05
    validation_alpha: float = 0.05
    assumption_alpha: float = 0.05
    min_total_n: int = 10
    min_n_per_factor: int = 2
    min_n_per_group: int = 3
    covariance_type: str = "HC3"
    validation_mode: ValidationMode = "univariate"
    categorical_validation_mode: CategoricalValidationMode = "both"


@dataclass
class AnalysisRequest:
    input_path: Path
    output_dir: Path
    response: str
    primary_factor: str
    covariates: list[str] = field(default_factory=list)
    group_variables: list[str] = field(default_factory=list)
    candidate_id_variables: list[str] = field(default_factory=list)
    analysis_type: AnalysisType | None = None
    template_path: Path | None = None
    table_config_path: Path | None = None
    run_plots: bool = True
    run_tables: bool = True
    run_report: bool = True
    alpha: float = 0.05
    validation_alpha: float | None = None
    assumption_alpha: float | None = None
    min_total_n: int = 10
    min_n_per_factor: int = 2
    min_n_per_group: int = 3
    covariance_type: str = "HC3"
    validation_mode: ValidationMode = "univariate"
    categorical_validation_mode: CategoricalValidationMode = "both"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class FigureArtifact:
    figure_id: str
    path: Path
    title: str
    caption: str
    section: str
    tags: list[str] = field(default_factory=list)


@dataclass
class TableArtifact:
    table_id: str
    path: Path
    title: str
    source: str
    columns: list[str] = field(default_factory=list)
    row_count: int = 0
    preview_rows: list[dict[str, Any]] = field(default_factory=list)
    include_in_pdf: bool = True


@dataclass
class ModelResult:
    analysis_type: AnalysisType
    formula: str
    fit_statistics: dict[str, float] = field(default_factory=dict)
    parameter_table: list[dict[str, Any]] = field(default_factory=list)
    adjusted_means: list[dict[str, Any]] = field(default_factory=list)
    assumptions_passed: bool = True
    flags: list[Flag] = field(default_factory=list)
    raw_result: Any = None


@dataclass
class RunResult:
    request: AnalysisRequest
    health: ValidationResult | None = None
    categorical_validation: ValidationResult | None = None
    covariate_validation: ValidationResult | None = None
    id_variable_validation: ValidationResult | None = None
    assumption_validation: ValidationResult | None = None
    model: ModelResult | None = None
    figures: list[FigureArtifact] = field(default_factory=list)
    tables: list[TableArtifact] = field(default_factory=list)
    flags: list[Flag] = field(default_factory=list)
    report_path: Path | None = None
