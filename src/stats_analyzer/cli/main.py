from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from stats_analyzer.checks.assumptions import AssumptionChecker
from stats_analyzer.checks.health import DataHealthChecker
from stats_analyzer.cli.help_text import EXPLANATIONS
from stats_analyzer.core.models import AnalysisRequest
from stats_analyzer.data.loader import DataLoader
from stats_analyzer.modeling.adjusted_means import AdjustedMeansCalculator
from stats_analyzer.modeling.ancova import AncovaModelRunner
from stats_analyzer.modeling.anova import AnovaModelRunner
from stats_analyzer.modeling.model_selector import ModelSelector
from stats_analyzer.pipeline.orchestrator import PipelineOrchestrator
from stats_analyzer.plotting.comparison import ComparisonPlotter
from stats_analyzer.plotting.diagnostic import DiagnosticPlotter
from stats_analyzer.validation.categorical import CategoricalValidator
from stats_analyzer.validation.covariate import CovariateValidator
from stats_analyzer.validation.identify import IdentifyVariableValidator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stats-analyzer",
        description="CLI-first statistics analyzer scaffold.",
    )
    subparsers = parser.add_subparsers(dest="command")

    ingest = subparsers.add_parser("ingest", help="Read a dataset and report dimensions.")
    ingest.add_argument("input", type=str, help="Path to input dataset.")
    ingest.add_argument("--explain", action="store_true", help="Show assumptions and method context.")

    health = subparsers.add_parser("check-health", help="Run data health checks.")
    _add_analysis_arguments(health, require_model_fields=True)

    categorical = subparsers.add_parser(
        "validate-categorical",
        help="Validate primary categorical factor and level distribution.",
    )
    _add_analysis_arguments(categorical, require_model_fields=True)

    covariates = subparsers.add_parser(
        "validate-covariates",
        help="Validate covariates for ANCOVA readiness.",
    )
    _add_analysis_arguments(covariates, require_model_fields=True)

    identify = subparsers.add_parser(
        "validate-id-vars",
        help="Validate identify-variable presence and uniqueness.",
    )
    _add_analysis_arguments(identify, require_model_fields=True)

    run_model = subparsers.add_parser(
        "run-model",
        help="Fit ANOVA/ANCOVA and print fit summary.",
    )
    _add_analysis_arguments(run_model, require_model_fields=True)

    plot = subparsers.add_parser(
        "plot",
        help="Generate diagnostic and comparison plots from fitted model.",
    )
    _add_analysis_arguments(plot, require_model_fields=True)

    report = subparsers.add_parser(
        "report",
        help="Generate PDF report from full pipeline run.",
    )
    _add_analysis_arguments(report, require_model_fields=True)

    run_all = subparsers.add_parser("run-all", help="Execute full analysis pipeline.")
    _add_analysis_arguments(run_all, require_model_fields=True)

    return parser


def _add_analysis_arguments(parser: argparse.ArgumentParser, require_model_fields: bool) -> None:
    parser.add_argument("input", type=str, help="Path to input dataset.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/stats_analyzer",
        help="Directory for figures and report artifacts.",
    )
    parser.add_argument(
        "--response",
        type=str,
        required=require_model_fields,
        help="Response variable for ANOVA/ANCOVA.",
    )
    parser.add_argument(
        "--primary-factor",
        type=str,
        required=require_model_fields,
        help="Primary categorical factor.",
    )
    parser.add_argument(
        "--covariates",
        nargs="*",
        default=[],
        help="Covariate columns used for ANCOVA.",
    )
    parser.add_argument(
        "--group-vars",
        nargs="*",
        default=[],
        help="Grouping variables to split or stratify data.",
    )
    parser.add_argument(
        "--id-vars",
        nargs="*",
        default=[],
        help="Candidate identify variables to validate.",
    )
    parser.add_argument(
        "--analysis-type",
        choices=["auto", "anova", "ancova"],
        default="auto",
        help="Model type. 'auto' picks ANCOVA when covariates are present, otherwise ANOVA.",
    )
    parser.add_argument(
        "--template",
        type=str,
        help="Path to custom report template YAML.",
    )
    parser.add_argument(
        "--table-config",
        type=str,
        help="Path to table specification YAML. Defaults to --template when omitted.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for diagnostics and flags.",
    )
    parser.add_argument(
        "--validation-alpha",
        type=float,
        default=None,
        help="Significance threshold for validation models. Defaults to --alpha.",
    )
    parser.add_argument(
        "--assumption-alpha",
        type=float,
        default=None,
        help="Significance threshold for model assumption checks. Defaults to --alpha.",
    )
    parser.add_argument(
        "--min-total-n",
        type=int,
        default=10,
        help="Minimum total sample size required before model fitting.",
    )
    parser.add_argument(
        "--min-n-per-factor",
        type=int,
        default=2,
        help="Minimum observations required per level of the primary factor.",
    )
    parser.add_argument(
        "--min-n-per-group",
        type=int,
        default=3,
        help="Minimum observations required per group stratum from --group-vars.",
    )
    parser.add_argument(
        "--cov-type",
        type=str,
        default="HC3",
        help="Covariance estimator for ANCOVA (e.g., HC3, HC1, nonrobust).",
    )
    parser.add_argument(
        "--validation-mode",
        choices=["univariate", "joint"],
        default="univariate",
        help="Validation style: univariate checks or joint covariate model screening.",
    )
    parser.add_argument(
        "--categorical-validation-mode",
        choices=["rules", "joint", "both"],
        default="both",
        help="Categorical validation style: threshold rules, joint model, or both.",
    )
    parser.add_argument(
        "--assumption-scope",
        choices=["global", "group", "both"],
        default="global",
        help=(
            "Assumption diagnostic scope: global model residuals, per-group residuals, "
            "or both."
        ),
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Disable table artifact generation.",
    )
    parser.add_argument("--explain", action="store_true", help="Show assumptions and method context.")


def _build_request(
    args: argparse.Namespace,
    *,
    run_plots: bool = True,
    run_tables: bool = True,
    run_report: bool = True,
) -> AnalysisRequest:
    analysis_type = None if args.analysis_type == "auto" else args.analysis_type
    template_path = Path(args.template) if getattr(args, "template", None) else None
    table_config_path = Path(args.table_config) if getattr(args, "table_config", None) else None
    return AnalysisRequest(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        response=args.response,
        primary_factor=args.primary_factor,
        covariates=args.covariates or [],
        group_variables=args.group_vars or [],
        candidate_id_variables=args.id_vars or [],
        analysis_type=analysis_type,
        template_path=template_path,
        table_config_path=table_config_path,
        run_plots=run_plots,
        run_tables=run_tables,
        run_report=run_report,
        alpha=args.alpha,
        validation_alpha=args.validation_alpha,
        assumption_alpha=args.assumption_alpha,
        min_total_n=args.min_total_n,
        min_n_per_factor=args.min_n_per_factor,
        min_n_per_group=args.min_n_per_group,
        covariance_type=args.cov_type,
        validation_mode=args.validation_mode,
        categorical_validation_mode=args.categorical_validation_mode,
        assumption_scope=args.assumption_scope,
    )


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _print_explain(command: str | None) -> bool:
    if not command:
        return False
    explanation = EXPLANATIONS.get(command)
    if explanation is None:
        return False
    print(explanation)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    if args.explain:
        _print_explain(args.command)
        return 0

    loader = DataLoader()

    if args.command == "ingest":
        dataset = loader.load(Path(args.input))
        _print_json(
            {
                "rows": int(dataset.shape[0]),
                "columns": int(dataset.shape[1]),
                "column_names": list(dataset.columns),
            }
        )
        return 0

    request = _build_request(
        args,
        run_plots=args.command in {"plot", "report", "run-all"},
        run_tables=args.command in {"report", "run-all"} and not args.no_tables,
        run_report=args.command in {"report", "run-all"},
    )

    dataset = loader.load(request.input_path)
    selector = ModelSelector()
    model_spec = selector.select(request)

    if args.command == "check-health":
        result = DataHealthChecker().run(dataset, model_spec)
        _print_json(
            {
                "passed": result.passed,
                "summary": result.summary,
                "flags": [asdict(flag) for flag in result.flags],
                "metrics": result.metrics,
            }
        )
        return 0

    if args.command == "validate-categorical":
        result = CategoricalValidator().run(dataset, model_spec)
        _print_json(
            {
                "passed": result.passed,
                "summary": result.summary,
                "flags": [asdict(flag) for flag in result.flags],
                "metrics": result.metrics,
            }
        )
        return 0

    if args.command == "validate-covariates":
        result = CovariateValidator().run(dataset, model_spec)
        _print_json(
            {
                "passed": result.passed,
                "summary": result.summary,
                "flags": [asdict(flag) for flag in result.flags],
                "metrics": result.metrics,
            }
        )
        return 0

    if args.command == "validate-id-vars":
        result = IdentifyVariableValidator().run(dataset, request.candidate_id_variables)
        _print_json(
            {
                "passed": result.passed,
                "summary": result.summary,
                "flags": [asdict(flag) for flag in result.flags],
                "metrics": result.metrics,
            }
        )
        return 0

    if args.command in {"run-model", "plot"}:
        runner = AncovaModelRunner() if model_spec.analysis_type == "ancova" else AnovaModelRunner()
        model_result = runner.run(dataset, model_spec)
        model_result.adjusted_means = AdjustedMeansCalculator().run(dataset, model_spec, model_result)
        model_result.flags.extend(
            AdjustedMeansCalculator().extrapolation_flags(
                dataset,
                model_spec,
                model_result.adjusted_means,
            )
        )
        assumption_validation = AssumptionChecker().run(dataset, model_spec, model_result)
        model_result.flags.extend(assumption_validation.flags)
        model_result.assumptions_passed = assumption_validation.passed and model_result.assumptions_passed

        figures = []
        if args.command == "plot":
            plots_dir = request.output_dir / "figures"
            figures.extend(DiagnosticPlotter().run(model_result, plots_dir))
            figures.extend(ComparisonPlotter().run(dataset, model_spec, model_result, plots_dir))

        _print_json(
            {
                "analysis_type": model_result.analysis_type,
                "formula": model_result.formula,
                "fit_statistics": model_result.fit_statistics,
                "flags": [asdict(flag) for flag in model_result.flags],
                "adjusted_means_count": len(model_result.adjusted_means),
                "assumptions_passed": model_result.assumptions_passed,
                "assumption_metrics": assumption_validation.metrics,
                "figures": [str(figure.path) for figure in figures],
            }
        )
        return 0

    if args.command == "report":
        orchestrator = PipelineOrchestrator()
        run_result = orchestrator.run(request)
        if run_result.report_path:
            _print_json(
                {
                    "report_path": str(run_result.report_path),
                    "flag_count": len(run_result.flags),
                    "table_count": len(run_result.tables),
                    "tables": [str(table.path) for table in run_result.tables],
                }
            )
            return 0
        _print_json(
            {
                "report_path": None,
                "flag_count": len(run_result.flags),
                "table_count": len(run_result.tables),
                "tables": [str(table.path) for table in run_result.tables],
            }
        )
        return 1

    if args.command == "run-all":
        orchestrator = PipelineOrchestrator()
        run_result = orchestrator.run(request)
        _print_json(
            {
                "analysis_type": run_result.model.analysis_type if run_result.model else None,
                "report_path": str(run_result.report_path) if run_result.report_path else None,
                "figure_count": len(run_result.figures),
                "table_count": len(run_result.tables),
                "flag_count": len(run_result.flags),
            }
        )
        if request.run_report and run_result.report_path is None:
            return 1
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
