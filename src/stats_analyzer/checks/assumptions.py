from __future__ import annotations

import math
from typing import Any

from stats_analyzer.core.models import Flag, ModelResult, ModelSpec, ValidationResult


class AssumptionChecker:
    name = "model-assumptions"

    def _append_slope_flags(self, model_result: ModelResult) -> list[Flag]:
        flags: list[Flag] = []
        for model_flag in model_result.flags:
            if model_flag.code != "slope_non_homogeneity":
                continue
            flags.append(
                Flag(
                    code="ancova_slope_non_homogeneity",
                    message="ANCOVA slope homogeneity appears violated based on interaction diagnostics.",
                    severity="ERROR",
                    stage="assumptions",
                    variables=model_flag.variables,
                    recommendation="Model interactions explicitly or consider stratified analyses.",
                )
            )
        return flags

    def run(self, dataset: Any, model_spec: ModelSpec, model_result: ModelResult | None) -> ValidationResult:
        flags: list[Flag] = []
        metrics: dict[str, Any] = {
            "observation_count": int(len(dataset)),
            "assumption_alpha": model_spec.assumption_alpha,
        }
        assumptions = [
            "Residuals are approximately normal.",
            "Residual variance is approximately homogeneous.",
            "Residuals are independent (design-based and Durbin-Watson screened).",
            "No extreme influence points dominate the fit.",
        ]
        if model_spec.analysis_type == "ancova":
            assumptions.append("Covariate relationships are approximately linear.")
            assumptions.append("Covariate slopes are homogeneous across treatment levels.")

        if model_result is None:
            flags.append(
                Flag(
                    code="model_missing",
                    message="Assumption checks skipped because no fitted model was provided.",
                    severity="WARN",
                    stage="assumptions",
                    recommendation="Run the modeling step before assumptions validation.",
                )
            )
            return ValidationResult(
                name=self.name,
                passed=False,
                summary="No model available for assumption checks.",
                flags=flags,
                assumptions=assumptions,
            )

        if model_result.raw_result is None:
            flags.append(
                Flag(
                    code="raw_model_missing",
                    message="Assumption checks skipped because fitted model object is unavailable.",
                    severity="WARN",
                    stage="assumptions",
                )
            )
            return ValidationResult(
                name=self.name,
                passed=False,
                summary="Assumption checks were partially skipped.",
                flags=flags,
                assumptions=assumptions,
                metrics=metrics,
            )

        model = model_result.raw_result
        residuals = model.resid
        exog = model.model.exog

        try:
            from statsmodels.stats.stattools import jarque_bera

            jb_stat, jb_pvalue, skewness, kurtosis = jarque_bera(residuals)
            metrics["jarque_bera_stat"] = float(jb_stat)
            metrics["jarque_bera_pvalue"] = float(jb_pvalue)
            metrics["residual_skew"] = float(skewness)
            metrics["residual_kurtosis"] = float(kurtosis)
            if float(jb_pvalue) < model_spec.assumption_alpha:
                flags.append(
                    Flag(
                        code="residual_non_normality",
                        message=(
                            "Residual normality check failed (Jarque-Bera "
                            f"p={float(jb_pvalue):.4g}, alpha={model_spec.assumption_alpha:.4g})."
                        ),
                        severity="WARN",
                        stage="assumptions",
                        recommendation="Inspect QQ plot and consider transformations or robust inference.",
                    )
                )
        except Exception as exc:
            flags.append(
                Flag(
                    code="normality_check_failed",
                    message=f"Residual normality check failed to execute: {exc}",
                    severity="WARN",
                    stage="assumptions",
                )
            )

        try:
            from statsmodels.stats.diagnostic import het_breuschpagan

            lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(residuals, exog)
            metrics["breusch_pagan_lm_stat"] = float(lm_stat)
            metrics["breusch_pagan_lm_pvalue"] = float(lm_pvalue)
            metrics["breusch_pagan_f_stat"] = float(f_stat)
            metrics["breusch_pagan_f_pvalue"] = float(f_pvalue)
            if float(lm_pvalue) < model_spec.assumption_alpha:
                flags.append(
                    Flag(
                        code="heteroscedasticity_detected",
                        message=(
                            "Breusch-Pagan detected heteroscedasticity "
                            f"(p={float(lm_pvalue):.4g}, alpha={model_spec.assumption_alpha:.4g})."
                        ),
                        severity="WARN",
                        stage="assumptions",
                        recommendation="Use robust covariance and review variance structure by group.",
                    )
                )
        except Exception as exc:
            flags.append(
                Flag(
                    code="homoscedasticity_check_failed",
                    message=f"Homoscedasticity check failed to execute: {exc}",
                    severity="WARN",
                    stage="assumptions",
                )
            )

        if model_spec.covariates:
            try:
                from statsmodels.stats.diagnostic import linear_reset

                reset_result = linear_reset(model, power=2, use_f=True)
                reset_stat = float(getattr(reset_result, "fvalue", reset_result.statistic))
                reset_pvalue = float(reset_result.pvalue)
                metrics["ramsey_reset_stat"] = reset_stat
                metrics["ramsey_reset_pvalue"] = reset_pvalue
                if reset_pvalue < model_spec.assumption_alpha:
                    flags.append(
                        Flag(
                            code="linearity_violation_reset",
                            message=(
                                "Ramsey RESET suggests linearity misspecification "
                                f"(p={reset_pvalue:.4g}, alpha={model_spec.assumption_alpha:.4g})."
                            ),
                            severity="WARN",
                            stage="assumptions",
                            recommendation=(
                                "Check transformations, interactions, or nonlinear terms for covariates."
                            ),
                        )
                    )
            except Exception as exc:
                flags.append(
                    Flag(
                        code="linearity_check_failed",
                        message=f"Linearity check failed to execute: {exc}",
                        severity="WARN",
                        stage="assumptions",
                    )
                )

        try:
            from statsmodels.stats.stattools import durbin_watson

            dw_stat = float(durbin_watson(residuals))
            metrics["durbin_watson"] = dw_stat
            if dw_stat < 1.5 or dw_stat > 2.5:
                flags.append(
                    Flag(
                        code="residual_autocorrelation_signal",
                        message=(
                            f"Durbin-Watson statistic is {dw_stat:.4g}, outside heuristic [1.5, 2.5]."
                        ),
                        severity="WARN",
                        stage="assumptions",
                        recommendation="Review observation ordering and dependence structure.",
                    )
                )
        except Exception as exc:
            flags.append(
                Flag(
                    code="independence_check_failed",
                    message=f"Independence check failed to execute: {exc}",
                    severity="WARN",
                    stage="assumptions",
                )
            )

        try:
            influence = model.get_influence()
            cooks_distance = influence.cooks_distance[0]
            threshold = 4.0 / max(int(len(cooks_distance)), 1)
            high_count = int((cooks_distance > threshold).sum())
            metrics["cooks_distance_threshold"] = threshold
            metrics["high_cooks_distance_count"] = high_count
            if high_count > 0:
                flags.append(
                    Flag(
                        code="influential_observations_detected",
                        message=(
                            f"{high_count} observations exceed Cook's distance threshold {threshold:.4g}."
                        ),
                        severity="WARN",
                        stage="assumptions",
                        recommendation="Inspect influence diagnostics and assess sensitivity to outliers.",
                    )
                )
        except Exception as exc:
            flags.append(
                Flag(
                    code="influence_check_failed",
                    message=f"Influence diagnostics failed to execute: {exc}",
                    severity="WARN",
                    stage="assumptions",
                )
            )

        condition_number = float(getattr(model, "condition_number", float("nan")))
        if not math.isnan(condition_number):
            metrics["condition_number"] = condition_number
            if condition_number > 1_000:
                flags.append(
                    Flag(
                        code="high_condition_number",
                        message=f"Model condition number is high ({condition_number:.5g}).",
                        severity="WARN",
                        stage="assumptions",
                        recommendation="Inspect collinearity and sparse design cells.",
                    )
                )

        if model_spec.analysis_type == "ancova":
            flags.extend(self._append_slope_flags(model_result))

        passed = all(flag.severity != "ERROR" for flag in flags)
        summary = "Assumption checks completed."
        return ValidationResult(
            name=self.name,
            passed=passed,
            summary=summary,
            flags=flags,
            assumptions=assumptions,
            metrics=metrics,
        )
