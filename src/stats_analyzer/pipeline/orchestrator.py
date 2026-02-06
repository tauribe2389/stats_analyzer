from __future__ import annotations

from pathlib import Path
from typing import Iterable

from stats_analyzer.checks.assumptions import AssumptionChecker
from stats_analyzer.checks.health import DataHealthChecker
from stats_analyzer.core.models import AnalysisRequest, Flag, RunResult, ValidationResult
from stats_analyzer.data.loader import DataLoader
from stats_analyzer.data.splitter import DataSplitter
from stats_analyzer.modeling.adjusted_means import AdjustedMeansCalculator
from stats_analyzer.modeling.ancova import AncovaModelRunner
from stats_analyzer.modeling.anova import AnovaModelRunner
from stats_analyzer.modeling.model_selector import ModelSelector
from stats_analyzer.pipeline.context import PipelineContext
from stats_analyzer.plotting.comparison import ComparisonPlotter
from stats_analyzer.plotting.diagnostic import DiagnosticPlotter
from stats_analyzer.reporting.pdf_builder import PdfReportBuilder
from stats_analyzer.reporting.table_builder import TableBuilder
from stats_analyzer.reporting.template_engine import TemplateEngine
from stats_analyzer.validation.categorical import CategoricalValidator
from stats_analyzer.validation.covariate import CovariateValidator
from stats_analyzer.validation.identify import IdentifyVariableValidator


class PipelineOrchestrator:
    """Run end-to-end analysis from ingest to report generation."""

    def __init__(
        self,
        loader: DataLoader | None = None,
        splitter: DataSplitter | None = None,
        health_checker: DataHealthChecker | None = None,
        categorical_validator: CategoricalValidator | None = None,
        covariate_validator: CovariateValidator | None = None,
        id_validator: IdentifyVariableValidator | None = None,
        selector: ModelSelector | None = None,
        anova_runner: AnovaModelRunner | None = None,
        ancova_runner: AncovaModelRunner | None = None,
        adjusted_means: AdjustedMeansCalculator | None = None,
        assumption_checker: AssumptionChecker | None = None,
        diagnostic_plotter: DiagnosticPlotter | None = None,
        comparison_plotter: ComparisonPlotter | None = None,
        template_engine: TemplateEngine | None = None,
        table_builder: TableBuilder | None = None,
        pdf_builder: PdfReportBuilder | None = None,
    ) -> None:
        self.loader = loader or DataLoader()
        self.splitter = splitter or DataSplitter()
        self.health_checker = health_checker or DataHealthChecker()
        self.categorical_validator = categorical_validator or CategoricalValidator()
        self.covariate_validator = covariate_validator or CovariateValidator()
        self.id_validator = id_validator or IdentifyVariableValidator()
        self.selector = selector or ModelSelector()
        self.anova_runner = anova_runner or AnovaModelRunner()
        self.ancova_runner = ancova_runner or AncovaModelRunner()
        self.adjusted_means = adjusted_means or AdjustedMeansCalculator()
        self.assumption_checker = assumption_checker or AssumptionChecker()
        self.diagnostic_plotter = diagnostic_plotter or DiagnosticPlotter()
        self.comparison_plotter = comparison_plotter or ComparisonPlotter()
        self.template_engine = template_engine or TemplateEngine()
        self.table_builder = table_builder or TableBuilder()
        self.pdf_builder = pdf_builder or PdfReportBuilder()

    def run(self, request: AnalysisRequest) -> RunResult:
        context = PipelineContext(request=request)
        request.output_dir.mkdir(parents=True, exist_ok=True)

        dataset = self.loader.load(request.input_path)
        context.dataset = dataset
        context.split_datasets = self.splitter.split(dataset, request.group_variables)

        model_spec = self.selector.select(request)
        context.model_spec = model_spec

        health = self.health_checker.run(dataset, model_spec)
        context.add_validation("health", health)

        categorical_validation = self.categorical_validator.run(dataset, model_spec)
        context.add_validation("categorical", categorical_validation)

        covariate_validation = self.covariate_validator.run(dataset, model_spec)
        context.add_validation("covariate", covariate_validation)

        id_validation = self.id_validator.run(dataset, request.candidate_id_variables)
        context.add_validation("identify_variables", id_validation)

        table_template = (
            self.template_engine.load(request.table_config_path or request.template_path)
            if request.run_tables
            else {}
        )
        report_template = self.template_engine.load(request.template_path) if request.run_report else {}

        if self._has_blocking_errors(context.validations.values()):
            assumption_validation = ValidationResult(
                name="model-assumptions",
                passed=False,
                summary="Model not fit because blocking validation errors were found.",
                flags=[],
            )
            context.add_validation("assumptions", assumption_validation)
            partial_result = RunResult(
                request=request,
                health=health,
                categorical_validation=categorical_validation,
                covariate_validation=covariate_validation,
                id_variable_validation=id_validation,
                assumption_validation=assumption_validation,
                model=None,
                figures=[],
                tables=[],
                flags=context.flags,
            )
            if request.run_tables:
                tables, table_flags = self.table_builder.build(
                    dataset=dataset,
                    run_result=partial_result,
                    template=table_template,
                    output_dir=request.output_dir,
                )
                partial_result.tables = tables
                partial_result.flags.extend(table_flags)
            return RunResult(
                request=request,
                health=health,
                categorical_validation=categorical_validation,
                covariate_validation=covariate_validation,
                id_variable_validation=id_validation,
                assumption_validation=assumption_validation,
                model=None,
                figures=[],
                tables=partial_result.tables,
                flags=partial_result.flags,
            )

        runner = self.ancova_runner if model_spec.analysis_type == "ancova" else self.anova_runner
        model_result = runner.run(dataset, model_spec)
        model_result.adjusted_means = self.adjusted_means.run(dataset, model_spec, model_result)
        model_result.flags.extend(
            self.adjusted_means.extrapolation_flags(
                dataset,
                model_spec,
                model_result.adjusted_means,
            )
        )
        context.add_model_result(model_result)

        assumption_validation = self.assumption_checker.run(dataset, model_spec, model_result)
        model_result.flags.extend(assumption_validation.flags)
        model_result.assumptions_passed = assumption_validation.passed and model_result.assumptions_passed
        context.add_validation("assumptions", assumption_validation)

        if request.run_plots:
            plots_dir = request.output_dir / "figures"
            for artifact in self.diagnostic_plotter.run(model_result, plots_dir):
                context.add_figure(artifact)
            for artifact in self.comparison_plotter.run(dataset, model_spec, model_result, plots_dir):
                context.add_figure(artifact)

        tables = []
        if request.run_tables:
            table_input = RunResult(
                request=request,
                health=health,
                categorical_validation=categorical_validation,
                covariate_validation=covariate_validation,
                id_variable_validation=id_validation,
                assumption_validation=assumption_validation,
                model=model_result,
                figures=context.figures,
                flags=context.flags,
            )
            tables, table_flags = self.table_builder.build(
                dataset=dataset,
                run_result=table_input,
                template=table_template,
                output_dir=request.output_dir,
            )
            for artifact in tables:
                context.add_table(artifact)
            context.flags.extend(table_flags)

        report_path: Path | None = None
        if request.run_report:
            report_path = request.output_dir / "analysis_report.pdf"
            report_input = RunResult(
                request=request,
                health=health,
                categorical_validation=categorical_validation,
                covariate_validation=covariate_validation,
                id_variable_validation=id_validation,
                assumption_validation=assumption_validation,
                model=model_result,
                figures=context.figures,
                tables=context.tables,
                flags=context.flags,
            )
            try:
                report_path = self.pdf_builder.build(report_input, report_template, report_path)
            except ModuleNotFoundError as exc:
                context.flags.append(
                    Flag(
                        code="report_dependency_missing",
                        message=f"Missing dependency for report generation: {exc.name}",
                        severity="ERROR",
                        stage="reporting",
                        recommendation="Install reportlab to enable PDF output.",
                    )
                )
                report_path = None

        return RunResult(
            request=request,
            health=health,
            categorical_validation=categorical_validation,
            covariate_validation=covariate_validation,
            id_variable_validation=id_validation,
            assumption_validation=assumption_validation,
            model=model_result,
            figures=context.figures,
            tables=context.tables,
            flags=context.flags,
            report_path=report_path,
        )

    def _has_blocking_errors(self, validations: Iterable[ValidationResult]) -> bool:
        return any(
            flag.severity == "ERROR"
            for validation in validations
            for flag in validation.flags
        )
