from __future__ import annotations

from stats_analyzer.core.models import AnalysisRequest, ModelSpec


class ModelSelector:
    """Choose ANOVA vs ANCOVA and compile a model specification."""

    def select(self, request: AnalysisRequest) -> ModelSpec:
        analysis_type = request.analysis_type
        if analysis_type is None:
            analysis_type = "ancova" if request.covariates else "anova"

        validation_alpha = request.validation_alpha if request.validation_alpha is not None else request.alpha
        assumption_alpha = request.assumption_alpha if request.assumption_alpha is not None else request.alpha

        return ModelSpec(
            analysis_type=analysis_type,
            response=request.response,
            primary_factor=request.primary_factor,
            covariates=request.covariates,
            group_variables=request.group_variables,
            alpha=request.alpha,
            validation_alpha=validation_alpha,
            assumption_alpha=assumption_alpha,
            min_total_n=request.min_total_n,
            min_n_per_factor=request.min_n_per_factor,
            min_n_per_group=request.min_n_per_group,
            covariance_type=request.covariance_type,
            validation_mode=request.validation_mode,
            categorical_validation_mode=request.categorical_validation_mode,
        )
