from __future__ import annotations

import numpy as np
import pandas as pd

from stats_analyzer.checks.assumptions import AssumptionChecker
from stats_analyzer.checks.health import DataHealthChecker
from stats_analyzer.core.models import ModelSpec
from stats_analyzer.modeling.ancova import AncovaModelRunner
from stats_analyzer.validation.categorical import CategoricalValidator
from stats_analyzer.validation.covariate import CovariateValidator


def test_health_checker_enforces_min_n_per_group() -> None:
    dataset = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "arm": ["A", "A", "B", "B"],
            "site": ["North", "North", "South", "East"],
            "x": [0.1, 0.2, 0.3, 0.4],
        }
    )
    spec = ModelSpec(
        analysis_type="ancova",
        response="y",
        primary_factor="arm",
        covariates=["x"],
        group_variables=["site"],
        min_total_n=4,
        min_n_per_group=2,
    )

    result = DataHealthChecker().run(dataset, spec)
    codes = {flag.code for flag in result.flags}

    assert result.passed is False
    assert "insufficient_n_per_group" in codes


def test_categorical_validator_runs_joint_model_mode() -> None:
    dataset = pd.DataFrame(
        {
            "y": [10.0, 11.0, 12.0, 13.0, 9.5, 10.5, 11.5, 12.5],
            "arm": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "site": ["North", "South", "North", "South", "North", "South", "North", "South"],
            "x": [1.2, 1.3, 1.4, 1.5, 1.0, 1.1, 1.2, 1.3],
        }
    )
    spec = ModelSpec(
        analysis_type="ancova",
        response="y",
        primary_factor="arm",
        covariates=["x"],
        group_variables=["site"],
        min_total_n=8,
        min_n_per_factor=2,
        min_n_per_group=2,
        categorical_validation_mode="joint",
    )

    result = CategoricalValidator().run(dataset, spec)

    assert "joint_formula" in result.metrics
    assert "joint_design_rank" in result.metrics
    assert result.metrics["categorical_validation_mode"] == "joint"


def test_covariate_validator_reports_joint_metrics() -> None:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(80):
        arm = "A" if i % 2 == 0 else "B"
        site = "North" if i % 3 == 0 else "South"
        x = rng.normal(50, 10)
        y = 10 + (1.0 if arm == "B" else 0.0) + (0.8 if site == "North" else -0.6) + 0.5 * x + rng.normal(0, 1)
        rows.append({"y": y, "arm": arm, "site": site, "x": x})
    dataset = pd.DataFrame(rows)

    spec = ModelSpec(
        analysis_type="ancova",
        response="y",
        primary_factor="arm",
        covariates=["x"],
        group_variables=["site"],
        min_total_n=30,
        min_n_per_factor=2,
        min_n_per_group=2,
        validation_mode="joint",
    )

    result = CovariateValidator().run(dataset, spec)
    covariate_metric = result.metrics["covariate_metrics"]["x"]

    assert result.metrics["covariate_count"] == 1
    assert result.metrics["covariates_numeric_count"] == 1
    assert covariate_metric["present"] is True
    assert covariate_metric["numeric_castable"] is True
    assert covariate_metric["missing_count"] == 0
    assert covariate_metric["non_missing_count"] == int(dataset.shape[0])
    assert "joint_formula" in result.metrics
    assert "joint_covariate_pvalues" in result.metrics
    assert "x" in result.metrics["joint_covariate_pvalues"]
    assert "joint_durbin_watson" in result.metrics or "covariate_joint_assumption_checks_failed" in {
        flag.code for flag in result.flags
    }


def test_assumption_checker_reports_metrics_for_fitted_model() -> None:
    rng = np.random.default_rng(7)
    rows = []
    for i in range(90):
        arm = "A" if i % 2 == 0 else "B"
        site = "North" if i % 3 == 0 else "South"
        x = rng.normal(0, 1)
        y = 5 + (1.2 if arm == "B" else 0.0) + (0.5 if site == "North" else -0.5) + 2.0 * x + rng.normal(0, 0.6)
        rows.append({"y": y, "arm": arm, "site": site, "x": x})
    dataset = pd.DataFrame(rows)

    spec = ModelSpec(
        analysis_type="ancova",
        response="y",
        primary_factor="arm",
        covariates=["x"],
        group_variables=["site"],
        min_total_n=30,
        min_n_per_factor=2,
        min_n_per_group=2,
    )
    model_result = AncovaModelRunner().run(dataset, spec)
    assumption_result = AssumptionChecker().run(dataset, spec, model_result)
    codes = {flag.code for flag in assumption_result.flags}

    assert "jarque_bera_pvalue" in assumption_result.metrics or "normality_check_failed" in codes
    assert "breusch_pagan_lm_pvalue" in assumption_result.metrics or "homoscedasticity_check_failed" in codes
    assert "durbin_watson" in assumption_result.metrics or "independence_check_failed" in codes
