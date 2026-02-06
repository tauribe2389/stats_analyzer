from __future__ import annotations

from math import isfinite
from typing import Any

from stats_analyzer.core.models import Flag, ModelSpec, ValidationResult


class CategoricalValidator:
    name = "categorical-validation"
    assumptions = [
        "Primary factor has at least two non-empty levels.",
        "Per-level sample sizes are sufficient for stable inference.",
        "Joint categorical validation model is estimable without severe rank deficiency.",
    ]

    def _run_rule_checks(self, dataset: Any, model_spec: ModelSpec) -> tuple[list[Flag], dict[str, Any]]:
        flags: list[Flag] = []
        metrics: dict[str, Any] = {}
        primary_factor = model_spec.primary_factor
        if primary_factor not in dataset.columns:
            flags.append(
                Flag(
                    code="primary_factor_missing",
                    message=f"Primary factor '{primary_factor}' is not in the dataset.",
                    severity="ERROR",
                    stage="categorical-validation",
                    variables=[primary_factor],
                )
            )
            return flags, {"levels": 0}

        series = dataset[primary_factor].dropna()
        level_counts = series.value_counts(dropna=False)
        level_count = int(level_counts.shape[0])
        metrics["levels"] = level_count
        if level_count < 2:
            flags.append(
                Flag(
                    code="insufficient_levels",
                    message=f"Primary factor '{primary_factor}' has fewer than two levels.",
                    severity="ERROR",
                    stage="categorical-validation",
                    variables=[primary_factor],
                    recommendation="Provide at least two levels for ANOVA/ANCOVA.",
                )
            )

        low_n_levels = [
            str(level)
            for level, count in level_counts.items()
            if int(count) < model_spec.min_n_per_factor
        ]
        if low_n_levels:
            flags.append(
                Flag(
                    code="insufficient_n_per_factor_level",
                    message=(
                        f"Some levels in '{primary_factor}' have fewer than "
                        f"{model_spec.min_n_per_factor} observations."
                    ),
                    severity="ERROR",
                    stage="categorical-validation",
                    variables=[primary_factor],
                    recommendation=f"Review sparse levels: {', '.join(low_n_levels)}",
                )
            )

        if model_spec.group_variables:
            cross_terms = [*model_spec.group_variables, primary_factor]
            try:
                cross_counts = dataset.groupby(cross_terms, dropna=False).size()
            except TypeError:
                cross_counts = dataset.groupby(cross_terms).size()

            sparse_strata: list[str] = []
            for key, count in cross_counts.items():
                if int(count) < model_spec.min_n_per_factor:
                    key_tuple = key if isinstance(key, tuple) else (key,)
                    key_str = ", ".join(str(part) for part in key_tuple)
                    sparse_strata.append(f"[{key_str}]={int(count)}")

            metrics["group_factor_strata"] = int(cross_counts.shape[0])
            if sparse_strata:
                preview = ", ".join(sparse_strata[:10])
                if len(sparse_strata) > 10:
                    preview = f"{preview}, ..."
                flags.append(
                    Flag(
                        code="insufficient_n_group_factor_stratum",
                        message=(
                            "Some group/factor strata are below minimum N per factor "
                            f"({model_spec.min_n_per_factor}): {preview}"
                        ),
                        severity="ERROR",
                        stage="categorical-validation",
                        variables=[primary_factor, *model_spec.group_variables],
                        recommendation=(
                            "Collapse sparse strata or increase sample size in underrepresented cells."
                        ),
                    )
                )

        return flags, metrics

    def _categorical_terms(self, model_spec: ModelSpec) -> list[str]:
        terms = [f"C({model_spec.primary_factor})"]
        terms.extend(f"C({group_variable})" for group_variable in model_spec.group_variables)
        terms.extend(
            f"C({model_spec.primary_factor}):C({group_variable})"
            for group_variable in model_spec.group_variables
        )
        return terms

    def _run_joint_model(self, dataset: Any, model_spec: ModelSpec) -> tuple[list[Flag], dict[str, Any]]:
        import numpy as np
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        flags: list[Flag] = []
        metrics: dict[str, Any] = {}

        if model_spec.response not in dataset.columns:
            flags.append(
                Flag(
                    code="categorical_joint_response_missing",
                    message=f"Response '{model_spec.response}' not found; joint categorical model skipped.",
                    severity="ERROR",
                    stage="categorical-validation",
                    variables=[model_spec.response],
                )
            )
            return flags, metrics

        categorical_terms = self._categorical_terms(model_spec)
        rhs_terms = [*categorical_terms, *model_spec.covariates]
        formula = f"{model_spec.response} ~ " + " + ".join(rhs_terms)
        metrics["joint_formula"] = formula

        try:
            model = smf.ols(formula=formula, data=dataset).fit()
        except Exception as exc:
            flags.append(
                Flag(
                    code="categorical_joint_model_failed",
                    message=f"Joint categorical model failed to fit: {exc}",
                    severity="ERROR",
                    stage="categorical-validation",
                    recommendation="Check sparse cells, collinearity, and data types for categorical columns.",
                )
            )
            return flags, metrics

        exog = model.model.exog
        rank = int(np.linalg.matrix_rank(exog))
        n_columns = int(exog.shape[1])
        metrics["joint_design_rank"] = rank
        metrics["joint_design_columns"] = n_columns
        metrics["joint_condition_number"] = float(model.condition_number)
        if rank < n_columns:
            flags.append(
                Flag(
                    code="categorical_joint_rank_deficient",
                    message=(
                        "Joint categorical model is rank-deficient "
                        f"(rank={rank}, columns={n_columns})."
                    ),
                    severity="ERROR",
                    stage="categorical-validation",
                    recommendation="Remove redundant terms or collapse categorical levels.",
                )
            )

        if model.df_resid <= 0:
            flags.append(
                Flag(
                    code="categorical_joint_df_resid_nonpositive",
                    message="Joint categorical model has non-positive residual degrees of freedom.",
                    severity="ERROR",
                    stage="categorical-validation",
                    recommendation="Reduce model complexity or increase sample size.",
                )
            )

        condition_number = float(model.condition_number)
        if isfinite(condition_number) and condition_number > 1_000:
            flags.append(
                Flag(
                    code="categorical_joint_condition_high",
                    message=(
                        f"Joint categorical model condition number is high ({condition_number:.5g})."
                    ),
                    severity="WARN",
                    stage="categorical-validation",
                    recommendation="Inspect collinearity and sparse categorical combinations.",
                )
            )

        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            from statsmodels.stats.stattools import durbin_watson, jarque_bera

            jb_stat, jb_pvalue, _, _ = jarque_bera(model.resid)
            metrics["joint_jarque_bera_stat"] = float(jb_stat)
            metrics["joint_jarque_bera_pvalue"] = float(jb_pvalue)
            if float(jb_pvalue) < model_spec.validation_alpha:
                flags.append(
                    Flag(
                        code="categorical_joint_residual_non_normality",
                        message=(
                            "Joint categorical model residual normality check failed "
                            f"(p={float(jb_pvalue):.4g}, alpha={model_spec.validation_alpha:.4g})."
                        ),
                        severity="WARN",
                        stage="categorical-validation",
                        recommendation="Inspect residual diagnostics before relying on inference.",
                    )
                )

            _, lm_pvalue, _, _ = het_breuschpagan(model.resid, model.model.exog)
            metrics["joint_breusch_pagan_pvalue"] = float(lm_pvalue)
            if float(lm_pvalue) < model_spec.validation_alpha:
                flags.append(
                    Flag(
                        code="categorical_joint_heteroscedasticity",
                        message=(
                            "Joint categorical model suggests heteroscedasticity "
                            f"(p={float(lm_pvalue):.4g}, alpha={model_spec.validation_alpha:.4g})."
                        ),
                        severity="WARN",
                        stage="categorical-validation",
                        recommendation="Use robust inference and inspect variance patterns across cells.",
                    )
                )

            dw_stat = float(durbin_watson(model.resid))
            metrics["joint_durbin_watson"] = dw_stat
            if dw_stat < 1.5 or dw_stat > 2.5:
                flags.append(
                    Flag(
                        code="categorical_joint_autocorrelation_signal",
                        message=(
                            f"Joint categorical model Durbin-Watson is {dw_stat:.4g}, outside [1.5, 2.5]."
                        ),
                        severity="WARN",
                        stage="categorical-validation",
                        recommendation="Review dependence structure and ordering assumptions.",
                    )
                )
        except Exception as exc:
            flags.append(
                Flag(
                    code="categorical_joint_assumption_checks_failed",
                    message=f"Joint categorical assumption checks failed to execute: {exc}",
                    severity="WARN",
                    stage="categorical-validation",
                )
            )

        try:
            anova_table = sm.stats.anova_lm(model, typ=2)
        except Exception as exc:
            flags.append(
                Flag(
                    code="categorical_joint_anova_failed",
                    message=f"Could not compute ANOVA table for joint categorical model: {exc}",
                    severity="WARN",
                    stage="categorical-validation",
                    recommendation="Proceed with caution and inspect model parameter table directly.",
                )
            )
            return flags, metrics

        term_pvalues: dict[str, float] = {}
        for term in categorical_terms:
            if term not in anova_table.index:
                continue
            value = anova_table.loc[term, "PR(>F)"]
            if value is None or not isfinite(float(value)):
                flags.append(
                    Flag(
                        code="categorical_joint_term_unstable",
                        message=f"Term '{term}' has non-finite ANOVA p-value in joint model.",
                        severity="WARN",
                        stage="categorical-validation",
                        variables=[term],
                        recommendation="Inspect sparse cells and model identifiability.",
                    )
                )
                continue
            pvalue = float(value)
            term_pvalues[term] = pvalue
            if pvalue > model_spec.validation_alpha:
                flags.append(
                    Flag(
                        code="categorical_joint_weak_signal",
                        message=(
                            f"Categorical term '{term}' is weak in joint model "
                            f"(p={pvalue:.4g}, alpha={model_spec.validation_alpha:.4g})."
                        ),
                        severity="WARN",
                        stage="categorical-validation",
                        variables=[term],
                        recommendation="Check whether this term is still required by design intent.",
                    )
                )

        metrics["joint_term_pvalues"] = term_pvalues
        metrics["joint_validation_alpha"] = model_spec.validation_alpha
        return flags, metrics

    def run(self, dataset: Any, model_spec: ModelSpec) -> ValidationResult:
        flags: list[Flag] = []
        metrics: dict[str, Any] = {
            "categorical_validation_mode": model_spec.categorical_validation_mode
        }

        if model_spec.categorical_validation_mode in {"rules", "both"}:
            rule_flags, rule_metrics = self._run_rule_checks(dataset, model_spec)
            flags.extend(rule_flags)
            metrics.update(rule_metrics)

        if model_spec.categorical_validation_mode in {"joint", "both"}:
            joint_flags, joint_metrics = self._run_joint_model(dataset, model_spec)
            flags.extend(joint_flags)
            metrics.update(joint_metrics)

        passed = all(flag.severity != "ERROR" for flag in flags)
        return ValidationResult(
            name=self.name,
            passed=passed,
            summary="Categorical validation completed.",
            flags=flags,
            metrics=metrics,
            assumptions=self.assumptions,
        )
