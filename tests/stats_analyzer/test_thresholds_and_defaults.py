from __future__ import annotations

from pathlib import Path

from stats_analyzer.core.models import AnalysisRequest, ModelSpec
from stats_analyzer.modeling.adjusted_means import AdjustedMeansCalculator
from stats_analyzer.modeling.model_selector import ModelSelector


def test_model_selector_uses_validation_alpha_fallback() -> None:
    request = AnalysisRequest(
        input_path=Path("in.csv"),
        output_dir=Path("out"),
        response="y",
        primary_factor="arm",
        covariates=["x1"],
        alpha=0.01,
    )

    spec = ModelSelector().select(request)
    assert spec.validation_alpha == 0.01
    assert spec.assumption_alpha == 0.01
    assert spec.covariance_type == "HC3"
    assert spec.min_n_per_group == 3
    assert spec.categorical_validation_mode == "both"


def test_adjusted_mean_extrapolation_flags_detect_out_of_range_and_unobserved_stratum() -> None:
    import pandas as pd

    dataset = pd.DataFrame(
        {
            "arm": ["A", "A", "B", "B"],
            "site": ["X", "X", "Y", "Y"],
            "age": [10, 12, 11, 13],
            "y": [1, 2, 3, 4],
        }
    )
    spec = ModelSpec(
        analysis_type="ancova",
        response="y",
        primary_factor="arm",
        covariates=["age"],
        group_variables=["site"],
    )

    rows = [
        {"arm": "A", "site": "X", "age": 11},
        {"arm": "B", "site": "X", "age": 99},
    ]
    flags = AdjustedMeansCalculator().extrapolation_flags(dataset, spec, rows)
    codes = {flag.code for flag in flags}

    assert "adjusted_mean_covariate_extrapolation" in codes
    assert "adjusted_mean_unobserved_stratum" in codes
