from __future__ import annotations

from typing import Any

import pandas as pd

from stats_analyzer.core.models import Flag, ModelResult, ModelSpec


class AdjustedMeansCalculator:
    """Compute adjusted means for the primary factor."""

    def run(self, dataset: Any, model_spec: ModelSpec, model_result: ModelResult) -> list[dict[str, Any]]:
        if model_result.raw_result is None:
            return []

        if model_spec.analysis_type == "anova":
            try:
                grouped = dataset.groupby(model_spec.primary_factor, dropna=False)
            except TypeError:
                grouped = dataset.groupby(model_spec.primary_factor)

            summary = grouped[model_spec.response].mean().reset_index().rename(
                columns={model_spec.response: "adjusted_mean"}
            )
            return summary.to_dict("records")

        baseline_values = {
            covariate: float(dataset[covariate].mean()) for covariate in model_spec.covariates
        }
        rows: list[dict[str, Any]] = []
        for level in dataset[model_spec.primary_factor].dropna().unique():
            row = {model_spec.primary_factor: level, **baseline_values}
            for group_variable in model_spec.group_variables:
                mode_value = dataset[group_variable].mode(dropna=True)
                row[group_variable] = mode_value.iloc[0] if not mode_value.empty else None
            rows.append(row)

        frame = pd.DataFrame(rows)
        predictions = model_result.raw_result.predict(frame)
        frame["adjusted_mean"] = predictions
        return frame.to_dict("records")

    def extrapolation_flags(
        self,
        dataset: Any,
        model_spec: ModelSpec,
        adjusted_rows: list[dict[str, Any]],
    ) -> list[Flag]:
        if not adjusted_rows:
            return []

        flags: list[Flag] = []
        categorical_columns = [model_spec.primary_factor, *model_spec.group_variables]

        for idx, row in enumerate(adjusted_rows, start=1):
            for covariate in model_spec.covariates:
                if covariate not in dataset.columns:
                    continue
                value = row.get(covariate)
                if value is None:
                    continue
                observed_min = float(dataset[covariate].min(skipna=True))
                observed_max = float(dataset[covariate].max(skipna=True))
                if float(value) < observed_min or float(value) > observed_max:
                    flags.append(
                        Flag(
                            code="adjusted_mean_covariate_extrapolation",
                            message=(
                                f"Adjusted-mean row {idx} uses covariate '{covariate}' value {value:.5g} "
                                f"outside observed range [{observed_min:.5g}, {observed_max:.5g}]."
                            ),
                            severity="WARN",
                            stage="modeling",
                            variables=[covariate],
                            recommendation="Interpret adjusted means cautiously for extrapolated covariate values.",
                        )
                    )

            if categorical_columns:
                matched = dataset
                for column in categorical_columns:
                    if column not in dataset.columns:
                        continue
                    matched = matched[matched[column] == row.get(column)]
                if matched.empty:
                    row_keys = ", ".join(f"{col}={row.get(col)}" for col in categorical_columns)
                    flags.append(
                        Flag(
                            code="adjusted_mean_unobserved_stratum",
                            message=(
                                "Adjusted-mean row is based on a categorical stratum not present in data: "
                                f"{row_keys}"
                            ),
                            severity="WARN",
                            stage="modeling",
                            variables=categorical_columns,
                            recommendation=(
                                "Confirm this stratification is intended or add data in that stratum."
                            ),
                        )
                    )

        return flags
