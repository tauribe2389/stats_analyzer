from __future__ import annotations

from typing import Any

from stats_analyzer.core.models import Flag, ModelResult, ModelSpec


class AnovaModelRunner:
    """Fit ANOVA models using statsmodels."""

    def build_formula(self, model_spec: ModelSpec) -> str:
        terms = [f"C({model_spec.primary_factor})"]
        terms.extend(f"C({variable})" for variable in model_spec.group_variables)
        terms.extend(
            f"C({model_spec.primary_factor}):C({variable})"
            for variable in model_spec.group_variables
        )
        rhs = " + ".join(terms) if terms else "1"
        return f"{model_spec.response} ~ {rhs}"

    def run(self, dataset: Any, model_spec: ModelSpec) -> ModelResult:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf

        formula = self.build_formula(model_spec)
        model = smf.ols(formula=formula, data=dataset).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        parameter_table = (
            model.summary2().tables[1].reset_index().rename(columns={"index": "term"}).to_dict("records")
        )

        fit_stats = {
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "f_statistic": float(model.fvalue) if model.fvalue is not None else float("nan"),
            "f_pvalue": float(model.f_pvalue) if model.f_pvalue is not None else float("nan"),
            "n_obs": float(model.nobs),
        }

        flags: list[Flag] = []
        if model.df_resid <= 0:
            flags.append(
                Flag(
                    code="model_df_resid_nonpositive",
                    message="Model residual degrees of freedom are non-positive.",
                    severity="ERROR",
                    stage="modeling",
                    recommendation="Simplify model terms or increase sample size.",
                )
            )

        return ModelResult(
            analysis_type="anova",
            formula=formula,
            fit_statistics=fit_stats,
            parameter_table=parameter_table,
            flags=flags,
            raw_result=model,
            adjusted_means=[],
            assumptions_passed=all(flag.severity != "ERROR" for flag in flags),
        )

