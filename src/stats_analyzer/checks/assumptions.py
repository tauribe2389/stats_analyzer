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

    def _group_label(self, group_key: Any, grouping_columns: list[str]) -> str:
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        parts = []
        for idx, column in enumerate(grouping_columns):
            value = key_tuple[idx] if idx < len(key_tuple) else None
            parts.append(f"{column}={value}")
        return ", ".join(parts)

    def _run_global_checks(self, model: Any, model_spec: ModelSpec) -> tuple[list[Flag], dict[str, Any]]:
        flags: list[Flag] = []
        metrics: dict[str, Any] = {}
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

        return flags, metrics

    def _run_group_checks(
        self,
        dataset: Any,
        model: Any,
        model_spec: ModelSpec,
    ) -> tuple[list[Flag], dict[str, Any]]:
        import numpy as np

        flags: list[Flag] = []
        grouping_columns: list[str] = []
        for column in [model_spec.primary_factor, *model_spec.group_variables]:
            if column in grouping_columns:
                continue
            if column in dataset.columns:
                grouping_columns.append(column)

        metrics: dict[str, Any] = {
            "assumption_grouping_columns": grouping_columns,
            "group_assumption_group_count": 0,
            "group_assumption_metrics": {},
        }

        if not grouping_columns:
            flags.append(
                Flag(
                    code="group_assumptions_no_group_columns",
                    message="Group-scoped assumptions requested, but no grouping columns were available.",
                    severity="WARN",
                    stage="assumptions",
                    recommendation="Set --primary-factor/--group-vars to enable group-level diagnostics.",
                )
            )
            return flags, metrics

        group_frame = None
        data_obj = getattr(model.model, "data", None)
        raw_row_labels = getattr(data_obj, "row_labels", None)
        row_labels = list(raw_row_labels) if raw_row_labels is not None else []
        if len(row_labels) > 0:
            try:
                group_frame = dataset.loc[row_labels, grouping_columns].copy()
            except Exception:
                group_frame = None

        if group_frame is None:
            model_frame = getattr(data_obj, "frame", None)
            if model_frame is not None and all(column in model_frame.columns for column in grouping_columns):
                group_frame = model_frame[grouping_columns].copy()

        if group_frame is None:
            flags.append(
                Flag(
                    code="group_assumptions_alignment_unavailable",
                    message=(
                        "Could not align fitted-model rows to grouping columns for group-level assumptions."
                    ),
                    severity="WARN",
                    stage="assumptions",
                    recommendation="Check index alignment and ensure grouping columns are present in model input.",
                )
            )
            return flags, metrics

        residual_values = np.asarray(model.resid)
        exog = np.asarray(model.model.exog)
        if exog.ndim == 1:
            exog = exog.reshape(-1, 1)

        aligned_len = min(len(group_frame), residual_values.shape[0], exog.shape[0])
        if aligned_len <= 0:
            return flags, metrics
        if aligned_len < len(group_frame) or aligned_len < residual_values.shape[0] or aligned_len < exog.shape[0]:
            flags.append(
                Flag(
                    code="group_assumptions_alignment_trimmed",
                    message=(
                        "Group-level assumptions used trimmed aligned rows due to model/data length mismatch."
                    ),
                    severity="WARN",
                    stage="assumptions",
                )
            )

        group_frame = group_frame.iloc[:aligned_len].reset_index(drop=True)
        residual_values = residual_values[:aligned_len]
        exog = exog[:aligned_len, :]

        try:
            grouped_indices = group_frame.groupby(grouping_columns, dropna=False, sort=False).indices
        except TypeError:
            grouped_indices = group_frame.groupby(grouping_columns, sort=False).indices

        group_metrics: dict[str, dict[str, Any]] = {}
        for group_key, positions in grouped_indices.items():
            position_array = np.asarray(list(positions), dtype=int)
            if position_array.size == 0:
                continue

            label = self._group_label(group_key, grouping_columns)
            subset_residuals = residual_values[position_array]
            subset_exog = exog[position_array, :]

            metric_row: dict[str, Any] = {"n_obs": int(position_array.size)}
            if int(position_array.size) < 8:
                flags.append(
                    Flag(
                        code="group_assumptions_small_sample",
                        message=(
                            f"Group '{label}' has low sample size for stable diagnostics "
                            f"(n={int(position_array.size)})."
                        ),
                        severity="WARN",
                        stage="assumptions",
                        variables=grouping_columns,
                        recommendation="Interpret group-level assumption p-values cautiously for small groups.",
                    )
                )

            try:
                from statsmodels.stats.stattools import jarque_bera

                jb_stat, jb_pvalue, skewness, kurtosis = jarque_bera(subset_residuals)
                metric_row["jarque_bera_stat"] = float(jb_stat)
                metric_row["jarque_bera_pvalue"] = float(jb_pvalue)
                metric_row["residual_skew"] = float(skewness)
                metric_row["residual_kurtosis"] = float(kurtosis)
                if float(jb_pvalue) < model_spec.assumption_alpha:
                    flags.append(
                        Flag(
                            code="group_residual_non_normality",
                            message=(
                                f"Group '{label}' failed residual normality check "
                                f"(p={float(jb_pvalue):.4g}, alpha={model_spec.assumption_alpha:.4g})."
                            ),
                            severity="WARN",
                            stage="assumptions",
                            variables=grouping_columns,
                            recommendation="Inspect within-group residual distributions and outliers.",
                        )
                    )
            except Exception as exc:
                flags.append(
                    Flag(
                        code="group_normality_check_failed",
                        message=f"Group '{label}' normality check failed to execute: {exc}",
                        severity="WARN",
                        stage="assumptions",
                        variables=grouping_columns,
                    )
                )

            try:
                from statsmodels.stats.diagnostic import het_breuschpagan

                if subset_exog.shape[0] > subset_exog.shape[1]:
                    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(subset_residuals, subset_exog)
                    metric_row["breusch_pagan_lm_stat"] = float(lm_stat)
                    metric_row["breusch_pagan_lm_pvalue"] = float(lm_pvalue)
                    metric_row["breusch_pagan_f_stat"] = float(f_stat)
                    metric_row["breusch_pagan_f_pvalue"] = float(f_pvalue)
                    if float(lm_pvalue) < model_spec.assumption_alpha:
                        flags.append(
                            Flag(
                                code="group_heteroscedasticity_detected",
                                message=(
                                    f"Group '{label}' shows heteroscedasticity signal "
                                    f"(p={float(lm_pvalue):.4g}, alpha={model_spec.assumption_alpha:.4g})."
                                ),
                                severity="WARN",
                                stage="assumptions",
                                variables=grouping_columns,
                                recommendation="Inspect within-group variance structure.",
                            )
                        )
                else:
                    metric_row["breusch_pagan_skipped"] = "insufficient_df"
            except Exception as exc:
                flags.append(
                    Flag(
                        code="group_homoscedasticity_check_failed",
                        message=f"Group '{label}' homoscedasticity check failed to execute: {exc}",
                        severity="WARN",
                        stage="assumptions",
                        variables=grouping_columns,
                    )
                )

            try:
                from statsmodels.stats.stattools import durbin_watson

                dw_stat = float(durbin_watson(subset_residuals))
                metric_row["durbin_watson"] = dw_stat
                if dw_stat < 1.5 or dw_stat > 2.5:
                    flags.append(
                        Flag(
                            code="group_residual_autocorrelation_signal",
                            message=(
                                f"Group '{label}' Durbin-Watson is {dw_stat:.4g}, outside [1.5, 2.5]."
                            ),
                            severity="WARN",
                            stage="assumptions",
                            variables=grouping_columns,
                            recommendation="Review within-group ordering/dependence assumptions.",
                        )
                    )
            except Exception as exc:
                flags.append(
                    Flag(
                        code="group_independence_check_failed",
                        message=f"Group '{label}' independence check failed to execute: {exc}",
                        severity="WARN",
                        stage="assumptions",
                        variables=grouping_columns,
                    )
                )

            try:
                condition_number = float(np.linalg.cond(subset_exog))
                if math.isfinite(condition_number):
                    metric_row["condition_number"] = condition_number
                    if condition_number > 1_000:
                        flags.append(
                            Flag(
                                code="group_high_condition_number",
                                message=f"Group '{label}' condition number is high ({condition_number:.5g}).",
                                severity="WARN",
                                stage="assumptions",
                                variables=grouping_columns,
                                recommendation="Inspect sparse rows or collinearity within this group.",
                            )
                        )
            except Exception:
                pass

            group_metrics[label] = metric_row

        metrics["group_assumption_group_count"] = len(group_metrics)
        metrics["group_assumption_metrics"] = group_metrics
        return flags, metrics

    def run(self, dataset: Any, model_spec: ModelSpec, model_result: ModelResult | None) -> ValidationResult:
        flags: list[Flag] = []
        assumption_scope = getattr(model_spec, "assumption_scope", "global")
        metrics: dict[str, Any] = {
            "observation_count": int(len(dataset)),
            "assumption_alpha": model_spec.assumption_alpha,
            "assumption_scope": assumption_scope,
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

        if assumption_scope in {"group", "both"}:
            assumptions.append("Group-level residual diagnostics are evaluated for defined groups.")

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
                metrics=metrics,
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
        run_global = assumption_scope in {"global", "both"}
        run_group = assumption_scope in {"group", "both"}

        if run_global:
            global_flags, global_metrics = self._run_global_checks(model, model_spec)
            flags.extend(global_flags)
            metrics.update(global_metrics)

        if run_group:
            group_flags, group_metrics = self._run_group_checks(dataset, model, model_spec)
            flags.extend(group_flags)
            metrics.update(group_metrics)

        if model_spec.analysis_type == "ancova":
            flags.extend(self._append_slope_flags(model_result))

        passed = all(flag.severity != "ERROR" for flag in flags)
        summary = "Assumption checks completed."
        if assumption_scope == "group":
            summary = "Group-level assumption checks completed."
        elif assumption_scope == "both":
            summary = "Global and group-level assumption checks completed."

        return ValidationResult(
            name=self.name,
            passed=passed,
            summary=summary,
            flags=flags,
            assumptions=assumptions,
            metrics=metrics,
        )
