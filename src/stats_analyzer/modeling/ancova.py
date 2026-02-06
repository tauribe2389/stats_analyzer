from __future__ import annotations

from typing import Any

import pandas as pd

from stats_analyzer.core.models import Flag, ModelResult, ModelSpec


class AncovaModelRunner:
    """Fit ANCOVA models and detect potential slope non-homogeneity."""

    def _fit_with_covariance(self, formula: str, dataset: Any, cov_type: str) -> tuple[Any, list[Flag]]:
        import statsmodels.formula.api as smf

        flags: list[Flag] = []
        if cov_type.lower() == "nonrobust":
            return smf.ols(formula=formula, data=dataset).fit(), flags
        try:
            return smf.ols(formula=formula, data=dataset).fit(cov_type=cov_type), flags
        except Exception as exc:
            flags.append(
                Flag(
                    code="invalid_covariance_type",
                    message=f"Covariance type '{cov_type}' failed; model fit reverted to nonrobust.",
                    severity="WARN",
                    stage="modeling",
                    recommendation=f"Use a valid statsmodels covariance type. Original error: {exc}",
                )
            )
            return smf.ols(formula=formula, data=dataset).fit(), flags

    def build_formula(self, model_spec: ModelSpec) -> str:
        terms = [f"C({model_spec.primary_factor})"]
        terms.extend(model_spec.covariates)
        terms.extend(f"C({variable})" for variable in model_spec.group_variables)
        rhs = " + ".join(terms) if terms else "1"
        return f"{model_spec.response} ~ {rhs}"

    def _run_slope_homogeneity_screen(self, dataset: Any, model_spec: ModelSpec) -> list[Flag]:
        if not model_spec.covariates:
            return []

        base_terms = [f"C({model_spec.primary_factor})", *model_spec.covariates]
        base_terms.extend(f"C({variable})" for variable in model_spec.group_variables)
        interaction_terms = [
            f"C({model_spec.primary_factor}):{covariate}" for covariate in model_spec.covariates
        ]
        interaction_formula = (
            f"{model_spec.response} ~ " + " + ".join([*base_terms, *interaction_terms])
        )

        interaction_model, fit_flags = self._fit_with_covariance(
            interaction_formula,
            dataset,
            model_spec.covariance_type,
        )
        p_values = interaction_model.pvalues

        flags: list[Flag] = list(fit_flags)
        for covariate in model_spec.covariates:
            pattern = f"C({model_spec.primary_factor})"
            violated = any(
                pattern in term and covariate in term and float(value) < model_spec.alpha
                for term, value in p_values.items()
            )
            if violated:
                flags.append(
                    Flag(
                        code="slope_non_homogeneity",
                        message=f"Slope homogeneity may be violated for covariate '{covariate}'.",
                        severity="ERROR",
                        stage="modeling",
                        variables=[covariate, model_spec.primary_factor],
                        recommendation="Investigate interaction terms or stratify analysis.",
                    )
                )
        return flags

    def run(self, dataset: Any, model_spec: ModelSpec) -> ModelResult:
        formula = self.build_formula(model_spec)
        model, flags = self._fit_with_covariance(formula, dataset, model_spec.covariance_type)
        parameter_table = (
            model.summary2().tables[1].reset_index().rename(columns={"index": "term"}).to_dict("records")
        )

        fit_stats = {
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "aic": float(model.aic),
            "bic": float(model.bic),
            "n_obs": float(model.nobs),
            "covariance_type": str(model.cov_type),
        }

        flags.extend(self._run_slope_homogeneity_screen(dataset, model_spec))
        return ModelResult(
            analysis_type="ancova",
            formula=formula,
            fit_statistics=fit_stats,
            parameter_table=parameter_table,
            flags=flags,
            raw_result=model,
            adjusted_means=[],
            assumptions_passed=all(flag.severity != "ERROR" for flag in flags),
        )

    def adjusted_means_frame(self, dataset: Any, model_spec: ModelSpec) -> pd.DataFrame:
        means = {covariate: dataset[covariate].mean() for covariate in model_spec.covariates}
        rows = []
        for level in dataset[model_spec.primary_factor].dropna().unique():
            row = {model_spec.primary_factor: level, **means}
            for group_variable in model_spec.group_variables:
                row[group_variable] = dataset[group_variable].mode().iloc[0]
            rows.append(row)
        return pd.DataFrame(rows)
