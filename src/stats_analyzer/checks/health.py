from __future__ import annotations

from typing import Any

from stats_analyzer.checks.base import BaseCheck
from stats_analyzer.core.models import Flag, ModelSpec, ValidationResult


class DataHealthChecker(BaseCheck):
    name = "data-health"
    assumptions = [
        "Required model variables exist in the input dataset.",
        "Dataset has enough observations to estimate model terms.",
        "Missingness patterns are reviewed before fitting ANOVA/ANCOVA.",
    ]

    def run(self, dataset: Any, model_spec: ModelSpec) -> ValidationResult:
        flags: list[Flag] = []
        required_columns = {
            model_spec.response,
            model_spec.primary_factor,
            *model_spec.covariates,
            *model_spec.group_variables,
        }
        missing_columns = sorted(col for col in required_columns if col not in dataset.columns)
        if missing_columns:
            flags.append(
                Flag(
                    code="missing_columns",
                    message=f"Required columns are missing: {', '.join(missing_columns)}",
                    severity="ERROR",
                    stage="health",
                    variables=missing_columns,
                    recommendation="Update CLI inputs or input dataset columns.",
                )
            )

        row_count = int(len(dataset))
        if row_count == 0:
            flags.append(
                Flag(
                    code="empty_dataset",
                    message="Dataset has zero rows.",
                    severity="ERROR",
                    stage="health",
                    recommendation="Provide a non-empty dataset.",
                )
            )
        elif row_count < model_spec.min_total_n:
            flags.append(
                Flag(
                    code="insufficient_total_n",
                    message=(
                        f"Dataset has {row_count} rows, below configured minimum total N "
                        f"({model_spec.min_total_n})."
                    ),
                    severity="ERROR",
                    stage="health",
                    recommendation="Increase sample size or lower the minimum threshold intentionally.",
                )
            )

        for column in required_columns:
            if column not in dataset.columns:
                continue
            null_count = int(dataset[column].isna().sum())
            if null_count > 0:
                flags.append(
                    Flag(
                        code="missing_values",
                        message=f"Column '{column}' has {null_count} missing values.",
                        severity="WARN",
                        stage="health",
                        variables=[column],
                        recommendation="Set explicit missing value policy before final reporting.",
                    )
                )

        if model_spec.group_variables:
            try:
                group_counts = dataset.groupby(model_spec.group_variables, dropna=False).size()
            except TypeError:
                group_counts = dataset.groupby(model_spec.group_variables).size()

            low_group_rows: list[str] = []
            for group_key, count in group_counts.items():
                if int(count) < model_spec.min_n_per_group:
                    key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
                    key_str = ", ".join(str(part) for part in key_tuple)
                    low_group_rows.append(f"[{key_str}]={int(count)}")

            if low_group_rows:
                preview = ", ".join(low_group_rows[:10])
                if len(low_group_rows) > 10:
                    preview = f"{preview}, ..."
                flags.append(
                    Flag(
                        code="insufficient_n_per_group",
                        message=(
                            "Some group strata are below minimum N "
                            f"({model_spec.min_n_per_group}): {preview}"
                        ),
                        severity="ERROR",
                        stage="health",
                        variables=model_spec.group_variables,
                        recommendation=(
                            "Increase sample size in sparse groups or reduce grouping granularity."
                        ),
                    )
                )

        passed = all(flag.severity != "ERROR" for flag in flags)
        summary = "Health checks completed."
        return ValidationResult(
            name=self.name,
            passed=passed,
            summary=summary,
            flags=flags,
            metrics={
                "row_count": row_count,
                "column_count": int(len(dataset.columns)),
                "group_variable_count": len(model_spec.group_variables),
            },
            assumptions=self.assumptions,
        )
