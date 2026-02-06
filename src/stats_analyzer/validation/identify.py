from __future__ import annotations

from typing import Any

from stats_analyzer.core.models import Flag, ValidationResult


class IdentifyVariableValidator:
    name = "identify-variable-validation"
    assumptions = [
        "Candidate identify variables exist in the dataset.",
        "Identify variables should uniquely identify records when expected.",
    ]

    def run(self, dataset: Any, candidate_id_variables: list[str]) -> ValidationResult:
        flags: list[Flag] = []
        for variable in candidate_id_variables:
            if variable not in dataset.columns:
                flags.append(
                    Flag(
                        code="id_variable_missing",
                        message=f"Identify variable '{variable}' is not in the dataset.",
                        severity="ERROR",
                        stage="identify-variable-validation",
                        variables=[variable],
                    )
                )
                continue

            duplicated_rows = int(dataset.duplicated(subset=[variable]).sum())
            if duplicated_rows > 0:
                flags.append(
                    Flag(
                        code="id_variable_not_unique",
                        message=f"Identify variable '{variable}' has {duplicated_rows} duplicate rows.",
                        severity="WARN",
                        stage="identify-variable-validation",
                        variables=[variable],
                        recommendation="Decide whether duplicates are acceptable for this analysis.",
                    )
                )

        passed = all(flag.severity != "ERROR" for flag in flags)
        return ValidationResult(
            name=self.name,
            passed=passed,
            summary="Identify variable validation completed.",
            flags=flags,
            assumptions=self.assumptions,
            metrics={"candidate_id_variable_count": len(candidate_id_variables)},
        )

