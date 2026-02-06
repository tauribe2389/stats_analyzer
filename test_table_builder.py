from __future__ import annotations

from pathlib import Path

import pandas as pd

from stats_analyzer.core.models import AnalysisRequest, ModelResult, RunResult, ValidationResult
from stats_analyzer.pipeline.orchestrator import PipelineOrchestrator
from stats_analyzer.reporting.table_builder import TableBuilder


def _base_run_result() -> RunResult:
    request = AnalysisRequest(
        input_path=Path("input.csv"),
        output_dir=Path("out"),
        response="y",
        primary_factor="arm",
    )
    model = ModelResult(
        analysis_type="anova",
        formula="y ~ C(arm)",
        fit_statistics={"r_squared": 0.5},
        parameter_table=[{"term": "Intercept", "Coef.": 1.23}],
        adjusted_means=[{"arm": "A", "adjusted_mean": 10.0}],
    )
    assumptions = ValidationResult(
        name="model-assumptions",
        passed=True,
        metrics={"durbin_watson": 2.1},
    )
    covariate_validation = ValidationResult(
        name="covariate-validation",
        passed=True,
        metrics={
            "covariate_count": 2,
            "covariate_metrics": {
                "age": {
                    "missing_count": 1,
                    "missing_rate": 0.1,
                }
            },
        },
    )
    return RunResult(
        request=request,
        model=model,
        assumption_validation=assumptions,
        covariate_validation=covariate_validation,
    )


def test_table_builder_writes_default_tables(tmp_path: Path) -> None:
    dataset = pd.DataFrame({"subject_id": ["S1", "S2"], "arm": ["A", "B"], "y": [10.0, 12.0]})
    run_result = _base_run_result()
    template = {
        "tables": {
            "include_defaults": True,
            "defaults": ["fit_statistics", "adjusted_means"],
            "default_formats": {"fit_statistics": {"value": "decimal:3"}},
            "custom": [],
            "preview_rows": 5,
        }
    }

    artifacts, flags = TableBuilder().build(dataset, run_result, template, tmp_path)

    assert len(flags) == 0
    ids = {artifact.table_id for artifact in artifacts}
    assert "fit_statistics" in ids
    assert "adjusted_means" in ids
    for artifact in artifacts:
        assert artifact.path.exists()
        assert artifact.path.suffix == ".csv"
    fit_table = next(item for item in artifacts if item.table_id == "fit_statistics")
    assert fit_table.preview_rows[0]["value"] == "0.500"


def test_table_builder_supports_custom_dataset_table(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        {
            "subject_id": ["S1", "S2", "S3"],
            "arm": ["A", "B", "A"],
            "y": [10.0, 14.0, 11.5],
            "rate": [0.1234, 0.4567, 0.7812],
        }
    )
    run_result = _base_run_result()
    template = {
        "tables": {
            "include_defaults": False,
            "custom": [
                {
                    "id": "high_response",
                    "source": "dataset",
                    "columns": ["subject_id", "arm", "y", "rate"],
                    "query": "y >= 11",
                    "sort_by": ["y"],
                    "ascending": False,
                    "limit": 2,
                    "rename": {"subject_id": "Subject ID", "y": "Outcome", "rate": "Response Rate"},
                    "format": {"Outcome": "decimal:1", "Response Rate": "percent:1"},
                    "include_in_pdf": True,
                }
            ],
            "preview_rows": 10,
        }
    }

    artifacts, flags = TableBuilder().build(dataset, run_result, template, tmp_path)

    assert len(flags) == 0
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.table_id == "high_response"
    assert artifact.row_count == 2
    assert artifact.columns == ["Subject ID", "arm", "Outcome", "Response Rate"]
    assert artifact.preview_rows[0]["Outcome"] == "14.0"
    assert artifact.preview_rows[0]["Response Rate"] == "45.7%"
    assert artifact.path.exists()


def test_orchestrator_generates_tables_when_enabled(tmp_path: Path) -> None:
    rows = []
    for i in range(1, 41):
        rows.append(
            {
                "subject_id": f"S{i:03d}",
                "treatment": "A" if i % 2 == 0 else "B",
                "site": "North" if i % 3 == 0 else "South",
                "age": 45 + (i % 7),
                "baseline": 90 + (i % 10),
                "outcome": 50 + (i % 2) * 3 + (i % 5),
            }
        )
    dataset = pd.DataFrame(rows)
    csv_path = tmp_path / "input.csv"
    dataset.to_csv(csv_path, index=False)

    request = AnalysisRequest(
        input_path=csv_path,
        output_dir=tmp_path / "output",
        response="outcome",
        primary_factor="treatment",
        covariates=["age", "baseline"],
        group_variables=["site"],
        run_plots=False,
        run_tables=True,
        run_report=False,
    )

    result = PipelineOrchestrator().run(request)

    assert len(result.tables) > 0
    assert any(item.table_id == "fit_statistics" for item in result.tables)
    assert all(item.path.exists() for item in result.tables)


def test_table_builder_warns_on_invalid_format_rules(tmp_path: Path) -> None:
    dataset = pd.DataFrame({"subject_id": ["S1"], "y": [10.0]})
    run_result = _base_run_result()
    template = {
        "tables": {
            "include_defaults": False,
            "custom": [
                {
                    "id": "bad_format",
                    "source": "dataset",
                    "columns": ["subject_id", "y"],
                    "format": {
                        "missing_col": "decimal:2",
                        "y": {"type": "unknown_type"},
                    },
                }
            ],
        }
    }

    artifacts, flags = TableBuilder().build(dataset, run_result, template, tmp_path)
    codes = {flag.code for flag in flags}

    assert len(artifacts) == 1
    assert "table_format_column_missing" in codes
    assert "table_format_type_unknown" in codes


def test_table_builder_supports_covariate_metrics_source(tmp_path: Path) -> None:
    dataset = pd.DataFrame({"subject_id": ["S1"], "y": [10.0]})
    run_result = _base_run_result()
    template = {
        "tables": {
            "include_defaults": False,
            "custom": [
                {
                    "id": "covariate_metrics_snapshot",
                    "source": "covariate_metrics",
                    "columns": ["metric", "value"],
                    "include_in_pdf": False,
                }
            ],
        }
    }

    artifacts, flags = TableBuilder().build(dataset, run_result, template, tmp_path)

    assert len(flags) == 0
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.table_id == "covariate_metrics_snapshot"
    metric_names = {str(row["metric"]) for row in artifact.preview_rows}
    assert "covariate_count" in metric_names
    assert "covariate_metrics.age.missing_count" in metric_names
