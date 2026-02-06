from __future__ import annotations

from math import isfinite
from typing import Any

from stats_analyzer.core.models import Flag, ModelSpec, ValidationResult


class CovariateValidator:
    name = "covariate-validation"
    assumptions = [
        "Covariates are numeric or can be cast to numeric values.",
        "Covariates have enough variation for model adjustment.",
    ]

    def _safe_finite_float(self, value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if isfinite(numeric) else None

    def _run_joint_validation(
        self,
        dataset: Any,
        model_spec: ModelSpec,
    ) -> tuple[list[Flag], dict[str, Any]]:
        import numpy as np
        import statsmodels.formula.api as smf

        metrics: dict[str, Any] = {}
        if not model_spec.covariates:
            return [], metrics

        flags: list[Flag] = []
        terms = [f"C({model_spec.primary_factor})", *model_spec.covariates]
        terms.extend(f"C({variable})" for variable in model_spec.group_variables)
        formula = f"{model_spec.response} ~ " + " + ".join(terms)
        metrics["joint_formula"] = formula

        try:
            model = smf.ols(formula=formula, data=dataset).fit()
        except Exception as exc:
            flags.append(
                Flag(
                    code="covariate_joint_validation_failed",
                    message=f"Joint validation model failed: {exc}",
                    severity="WARN",
                    stage="covariate-validation",
                    recommendation="Review covariate inputs and missingness handling.",
                )
            )
            return flags, metrics

        exog = model.model.exog
        metrics["joint_design_rank"] = int(np.linalg.matrix_rank(exog))
        metrics["joint_design_columns"] = int(exog.shape[1])
        metrics["joint_condition_number"] = float(model.condition_number)
        metrics["joint_nobs"] = int(model.nobs)
        metrics["joint_df_resid"] = float(model.df_resid)

        covariate_pvalues: dict[str, float] = {}
        for covariate in model_spec.covariates:
            p_value = model.pvalues.get(covariate)
            if p_value is None:
                continue
            p_value_float = self._safe_finite_float(p_value)
            if p_value_float is None:
                continue
            covariate_pvalues[covariate] = p_value_float
            if p_value_float > model_spec.validation_alpha:
                flags.append(
                    Flag(
                        code="covariate_joint_weak_signal",
                        message=(
                            f"Covariate '{covariate}' is not significant in joint validation model "
                            f"(p={p_value_float:.4g}, alpha={model_spec.validation_alpha:.4g})."
                        ),
                        severity="WARN",
                        stage="covariate-validation",
                        variables=[covariate],
                        recommendation=(
                            "Consider whether to retain this covariate based on domain context, "
                            "not p-value alone."
                        ),
                    )
                )
        metrics["joint_covariate_pvalues"] = covariate_pvalues
        metrics["joint_validation_alpha"] = model_spec.validation_alpha

        try:
            from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
            from statsmodels.stats.stattools import durbin_watson, jarque_bera

            jb_stat, jb_pvalue, skewness, kurtosis = jarque_bera(model.resid)
            jb_stat_float = self._safe_finite_float(jb_stat)
            jb_pvalue_float = self._safe_finite_float(jb_pvalue)
            skewness_float = self._safe_finite_float(skewness)
            kurtosis_float = self._safe_finite_float(kurtosis)
            if jb_stat_float is not None:
                metrics["joint_jarque_bera_stat"] = jb_stat_float
            if jb_pvalue_float is not None:
                metrics["joint_jarque_bera_pvalue"] = jb_pvalue_float
            if skewness_float is not None:
                metrics["joint_residual_skew"] = skewness_float
            if kurtosis_float is not None:
                metrics["joint_residual_kurtosis"] = kurtosis_float
            if jb_pvalue_float is not None and jb_pvalue_float < model_spec.validation_alpha:
                flags.append(
                    Flag(
                        code="covariate_joint_residual_non_normality",
                        message=(
                            "Joint covariate validation model residual normality check failed "
                            f"(p={jb_pvalue_float:.4g}, alpha={model_spec.validation_alpha:.4g})."
                        ),
                        severity="WARN",
                        stage="covariate-validation",
                        recommendation="Inspect residual diagnostics for validation model robustness.",
                    )
                )

            lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(model.resid, model.model.exog)
            lm_stat_float = self._safe_finite_float(lm_stat)
            lm_pvalue_float = self._safe_finite_float(lm_pvalue)
            f_stat_float = self._safe_finite_float(f_stat)
            f_pvalue_float = self._safe_finite_float(f_pvalue)
            if lm_stat_float is not None:
                metrics["joint_breusch_pagan_lm_stat"] = lm_stat_float
            if lm_pvalue_float is not None:
                metrics["joint_breusch_pagan_lm_pvalue"] = lm_pvalue_float
            if f_stat_float is not None:
                metrics["joint_breusch_pagan_f_stat"] = f_stat_float
            if f_pvalue_float is not None:
                metrics["joint_breusch_pagan_f_pvalue"] = f_pvalue_float
            if lm_pvalue_float is not None and lm_pvalue_float < model_spec.validation_alpha:
                flags.append(
                    Flag(
                        code="covariate_joint_heteroscedasticity",
                        message=(
                            "Joint covariate validation model suggests heteroscedasticity "
                            f"(p={lm_pvalue_float:.4g}, alpha={model_spec.validation_alpha:.4g})."
                        ),
                        severity="WARN",
                        stage="covariate-validation",
                        recommendation="Use robust inference and inspect variance patterns.",
                    )
                )

            reset_result = linear_reset(model, power=2, use_f=True)
            reset_stat = self._safe_finite_float(getattr(reset_result, "fvalue", None))
            reset_pvalue = self._safe_finite_float(reset_result.pvalue)
            if reset_stat is not None:
                metrics["joint_ramsey_reset_stat"] = reset_stat
            if reset_pvalue is not None:
                metrics["joint_ramsey_reset_pvalue"] = reset_pvalue
            if reset_pvalue is not None and reset_pvalue < model_spec.validation_alpha:
                flags.append(
                    Flag(
                        code="covariate_joint_linearity_violation",
                        message=(
                            "Joint covariate validation model shows potential linearity misspecification "
                            f"(RESET p={reset_pvalue:.4g}, alpha={model_spec.validation_alpha:.4g})."
                        ),
                        severity="WARN",
                        stage="covariate-validation",
                        recommendation="Consider nonlinear terms or transformations for covariates.",
                    )
                )

            dw_stat = self._safe_finite_float(durbin_watson(model.resid))
            if dw_stat is not None:
                metrics["joint_durbin_watson"] = dw_stat
            if dw_stat is not None and (dw_stat < 1.5 or dw_stat > 2.5):
                flags.append(
                    Flag(
                        code="covariate_joint_autocorrelation_signal",
                        message=(
                            f"Joint covariate validation model Durbin-Watson is {dw_stat:.4g}, "
                            "outside [1.5, 2.5]."
                        ),
                        severity="WARN",
                        stage="covariate-validation",
                        recommendation="Review dependence structure and ordering assumptions.",
                    )
                )
        except Exception as exc:
            flags.append(
                Flag(
                    code="covariate_joint_assumption_checks_failed",
                    message=f"Joint covariate assumption checks failed to execute: {exc}",
                    severity="WARN",
                    stage="covariate-validation",
                )
            )
        return flags, metrics

    def run(self, dataset: Any, model_spec: ModelSpec) -> ValidationResult:
        flags: list[Flag] = []
        row_count = int(len(dataset))
        metrics: dict[str, Any] = {
            "covariate_count": len(model_spec.covariates),
            "validation_mode": model_spec.validation_mode,
            "validation_alpha": model_spec.validation_alpha,
            "dataset_rows": row_count,
        }
        covariate_metrics: dict[str, dict[str, Any]] = {}
        missing_covariates: list[str] = []
        non_numeric_covariates: list[str] = []
        zero_variance_covariates: list[str] = []
        covariates_with_missing_values: list[str] = []
        present_covariates: list[str] = []
        numeric_covariates: list[str] = []

        for covariate in model_spec.covariates:
            metric_row: dict[str, Any] = {"present": False, "numeric_castable": False}
            if covariate not in dataset.columns:
                flags.append(
                    Flag(
                        code="covariate_missing",
                        message=f"Covariate '{covariate}' is not in the dataset.",
                        severity="ERROR",
                        stage="covariate-validation",
                        variables=[covariate],
                    )
                )
                missing_covariates.append(covariate)
                covariate_metrics[covariate] = metric_row
                continue

            series = dataset[covariate]
            metric_row["present"] = True
            metric_row["dtype"] = str(series.dtype)
            present_covariates.append(covariate)
            try:
                numeric_series = series.astype(float)
            except (TypeError, ValueError):
                flags.append(
                    Flag(
                        code="covariate_not_numeric",
                        message=f"Covariate '{covariate}' cannot be interpreted as numeric.",
                        severity="ERROR",
                        stage="covariate-validation",
                        variables=[covariate],
                        recommendation="Clean or recode this covariate before ANCOVA.",
                    )
                )
                non_numeric_covariates.append(covariate)
                covariate_metrics[covariate] = metric_row
                continue

            metric_row["numeric_castable"] = True
            numeric_covariates.append(covariate)

            non_missing = int(numeric_series.notna().sum())
            missing = int(numeric_series.isna().sum())
            variance = self._safe_finite_float(numeric_series.var(skipna=True))
            mean = self._safe_finite_float(numeric_series.mean(skipna=True))
            min_value = self._safe_finite_float(numeric_series.min(skipna=True))
            max_value = self._safe_finite_float(numeric_series.max(skipna=True))
            metric_row["non_missing_count"] = non_missing
            metric_row["missing_count"] = missing
            metric_row["missing_rate"] = (missing / row_count) if row_count > 0 else None
            metric_row["variance"] = variance
            metric_row["mean"] = mean
            metric_row["min"] = min_value
            metric_row["max"] = max_value

            zero_variance = variance is not None and variance == 0
            metric_row["zero_variance"] = zero_variance
            if zero_variance:
                flags.append(
                    Flag(
                        code="covariate_zero_variance",
                        message=f"Covariate '{covariate}' has zero variance.",
                        severity="WARN",
                        stage="covariate-validation",
                        variables=[covariate],
                    )
                )
                zero_variance_covariates.append(covariate)
            if missing > 0:
                flags.append(
                    Flag(
                        code="covariate_missing_values",
                        message=f"Covariate '{covariate}' has {missing} missing values.",
                        severity="WARN",
                        stage="covariate-validation",
                        variables=[covariate],
                    )
                )
                covariates_with_missing_values.append(covariate)

            covariate_metrics[covariate] = metric_row

        metrics["covariates_present_count"] = len(present_covariates)
        metrics["covariates_numeric_count"] = len(numeric_covariates)
        metrics["covariates_missing_columns_count"] = len(missing_covariates)
        metrics["covariates_non_numeric_count"] = len(non_numeric_covariates)
        metrics["covariates_zero_variance_count"] = len(zero_variance_covariates)
        metrics["covariates_with_missing_values_count"] = len(covariates_with_missing_values)
        metrics["covariates_missing_columns"] = missing_covariates
        metrics["covariates_non_numeric"] = non_numeric_covariates
        metrics["covariates_zero_variance"] = zero_variance_covariates
        metrics["covariates_with_missing_values"] = covariates_with_missing_values
        metrics["covariate_metrics"] = covariate_metrics

        if model_spec.validation_mode == "joint":
            joint_flags, joint_metrics = self._run_joint_validation(dataset, model_spec)
            flags.extend(joint_flags)
            metrics.update(joint_metrics)

        passed = all(flag.severity != "ERROR" for flag in flags)
        return ValidationResult(
            name=self.name,
            passed=passed,
            summary="Covariate validation completed.",
            flags=flags,
            metrics=metrics,
            assumptions=self.assumptions,
        )
