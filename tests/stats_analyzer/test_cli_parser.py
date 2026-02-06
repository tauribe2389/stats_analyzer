from __future__ import annotations

from stats_analyzer.cli.main import build_parser


def test_parse_run_all_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-all",
            "sample.csv",
            "--response",
            "outcome",
            "--primary-factor",
            "treatment",
            "--covariates",
            "age",
            "baseline",
            "--group-vars",
            "site",
            "--id-vars",
            "subject_id",
        ]
    )

    assert args.command == "run-all"
    assert args.input == "sample.csv"
    assert args.response == "outcome"
    assert args.primary_factor == "treatment"
    assert args.covariates == ["age", "baseline"]
    assert args.group_vars == ["site"]
    assert args.id_vars == ["subject_id"]


def test_parse_explain_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "validate-categorical",
            "sample.csv",
            "--response",
            "outcome",
            "--primary-factor",
            "arm",
            "--explain",
        ]
    )

    assert args.command == "validate-categorical"
    assert args.explain is True


def test_parser_defaults_include_thresholds_and_covariance() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run-model",
            "sample.csv",
            "--response",
            "outcome",
            "--primary-factor",
            "arm",
        ]
    )

    assert args.alpha == 0.05
    assert args.validation_alpha is None
    assert args.assumption_alpha is None
    assert args.min_total_n == 10
    assert args.min_n_per_factor == 2
    assert args.min_n_per_group == 3
    assert args.cov_type == "HC3"
    assert args.validation_mode == "univariate"
    assert args.categorical_validation_mode == "both"
    assert args.table_config is None
    assert args.no_tables is False
