from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from stats_analyzer.core.models import Flag, RunResult, TableArtifact


class TableBuilder:
    """Create CSV table artifacts from result data and user-defined specs."""

    DEFAULT_TABLE_IDS = [
        "flags",
        "fit_statistics",
        "parameter_estimates",
        "adjusted_means",
        "assumption_metrics",
    ]

    def build(
        self,
        dataset: Any,
        run_result: RunResult,
        template: dict[str, Any],
        output_dir: Path,
    ) -> tuple[list[TableArtifact], list[Flag]]:
        table_config = template.get("tables", {}) if isinstance(template, dict) else {}
        include_defaults = bool(table_config.get("include_defaults", True))
        default_ids = table_config.get("defaults", self.DEFAULT_TABLE_IDS)
        default_formats = table_config.get("default_formats", {})
        custom_specs = table_config.get("custom", [])
        preview_rows = int(table_config.get("preview_rows", 20))

        tables_dir = output_dir / "tables"
        tables_dir.mkdir(parents=True, exist_ok=True)

        artifacts: list[TableArtifact] = []
        flags: list[Flag] = []

        if include_defaults:
            for table_id in default_ids:
                frame, title, source = self._default_table(table_id, dataset, run_result)
                if frame is None:
                    continue
                format_spec = self._resolve_format_spec(default_formats, table_id)
                formatted_frame, format_flags = self._apply_column_formats(
                    frame=frame,
                    format_spec=format_spec,
                    table_id=self._normalize_id(table_id),
                )
                flags.extend(format_flags)
                artifacts.append(
                    self._write_table(
                        frame=formatted_frame,
                        table_id=self._normalize_id(table_id),
                        title=title,
                        source=source,
                        tables_dir=tables_dir,
                        preview_rows=preview_rows,
                        include_in_pdf=True,
                    )
                )

        for spec in custom_specs:
            artifact, table_flags = self._build_custom_table(
                spec=spec,
                dataset=dataset,
                run_result=run_result,
                tables_dir=tables_dir,
                preview_rows=preview_rows,
            )
            if artifact is not None:
                artifacts.append(artifact)
            flags.extend(table_flags)

        return artifacts, flags

    def _normalize_id(self, value: str) -> str:
        normalized = value.strip().lower().replace(" ", "_")
        return "".join(ch for ch in normalized if ch.isalnum() or ch == "_")

    def _default_table(
        self,
        table_id: str,
        dataset: Any,
        run_result: RunResult,
    ) -> tuple[pd.DataFrame | None, str, str]:
        table_key = self._normalize_id(table_id)
        if table_key == "flags":
            frame = pd.DataFrame([asdict(flag) for flag in run_result.flags])
            return frame, "Flags", "flags"

        if table_key == "fit_statistics":
            if run_result.model is None:
                return None, "", ""
            rows = [{"metric": key, "value": value} for key, value in run_result.model.fit_statistics.items()]
            frame = pd.DataFrame(rows)
            return frame, "Model Fit Statistics", "model.fit_statistics"

        if table_key == "parameter_estimates":
            if run_result.model is None:
                return None, "", ""
            frame = pd.DataFrame(run_result.model.parameter_table)
            return frame, "Model Parameter Estimates", "model.parameter_table"

        if table_key == "adjusted_means":
            if run_result.model is None:
                return None, "", ""
            frame = pd.DataFrame(run_result.model.adjusted_means)
            return frame, "Adjusted Means", "model.adjusted_means"

        if table_key == "assumption_metrics":
            if run_result.assumption_validation is None:
                return None, "", ""
            rows = [
                {"metric": key, "value": value}
                for key, value in run_result.assumption_validation.metrics.items()
            ]
            frame = pd.DataFrame(rows)
            return frame, "Assumption Metrics", "assumptions.metrics"

        if table_key == "dataset_preview":
            frame = pd.DataFrame(dataset).head(25)
            return frame, "Dataset Preview", "dataset"

        return None, "", ""

    def _source_frame(
        self,
        source: str,
        dataset: Any,
        run_result: RunResult,
    ) -> pd.DataFrame:
        normalized = self._normalize_id(source)
        if normalized in {"dataset", "original_dataset", "raw_dataset"}:
            return pd.DataFrame(dataset)
        if normalized in {"flags", "all_flags"}:
            return pd.DataFrame([asdict(flag) for flag in run_result.flags])
        if normalized in {"adjusted_means", "model_adjusted_means"}:
            rows = run_result.model.adjusted_means if run_result.model else []
            return pd.DataFrame(rows)
        if normalized in {"parameter_estimates", "model_parameter_table"}:
            rows = run_result.model.parameter_table if run_result.model else []
            return pd.DataFrame(rows)
        if normalized in {"fit_statistics", "model_fit_statistics"}:
            rows = [
                {"metric": key, "value": value}
                for key, value in (run_result.model.fit_statistics.items() if run_result.model else [])
            ]
            return pd.DataFrame(rows)
        if normalized in {"assumption_metrics", "assumptions"}:
            rows = self._metrics_rows(
                run_result.assumption_validation.metrics if run_result.assumption_validation else {}
            )
            return pd.DataFrame(rows)
        if normalized in {"health_metrics", "health_validation_metrics"}:
            rows = self._metrics_rows(run_result.health.metrics if run_result.health else {})
            return pd.DataFrame(rows)
        if normalized in {"categorical_metrics", "categorical_validation_metrics"}:
            rows = self._metrics_rows(
                run_result.categorical_validation.metrics if run_result.categorical_validation else {}
            )
            return pd.DataFrame(rows)
        if normalized in {"covariate_metrics", "covariate_validation_metrics"}:
            rows = self._metrics_rows(
                run_result.covariate_validation.metrics if run_result.covariate_validation else {}
            )
            return pd.DataFrame(rows)
        if normalized in {
            "identify_variable_metrics",
            "identify_variables_metrics",
            "id_variable_metrics",
            "id_vars_metrics",
        }:
            rows = self._metrics_rows(
                run_result.id_variable_validation.metrics if run_result.id_variable_validation else {}
            )
            return pd.DataFrame(rows)
        if normalized in {"validation_metrics", "all_validation_metrics"}:
            rows = self._validation_metric_rows(run_result)
            return pd.DataFrame(rows)
        if normalized in {"figures", "figure_artifacts"}:
            rows = [
                {
                    "figure_id": item.figure_id,
                    "path": str(item.path),
                    "title": item.title,
                    "section": item.section,
                    "tags": ",".join(item.tags),
                }
                for item in run_result.figures
            ]
            return pd.DataFrame(rows)
        if normalized in {"validations", "validation_summary"}:
            validations = [
                ("health", run_result.health),
                ("categorical", run_result.categorical_validation),
                ("covariate", run_result.covariate_validation),
                ("identify_variables", run_result.id_variable_validation),
                ("assumptions", run_result.assumption_validation),
            ]
            rows = []
            for key, result in validations:
                if result is None:
                    continue
                rows.append(
                    {
                        "validation": key,
                        "passed": result.passed,
                        "summary": result.summary,
                        "flag_count": len(result.flags),
                    }
                )
            return pd.DataFrame(rows)

        raise KeyError(f"Unknown table source: {source}")

    def _metrics_rows(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        def _flatten(metric_key: str, value: Any) -> None:
            if isinstance(value, dict):
                if not value:
                    rows.append({"metric": metric_key, "value": "{}"})
                    return
                for nested_key, nested_value in value.items():
                    _flatten(f"{metric_key}.{nested_key}", nested_value)
                return

            if isinstance(value, list):
                if value and all(not isinstance(item, (dict, list, tuple, set)) for item in value):
                    rows.append(
                        {"metric": metric_key, "value": ", ".join(str(item) for item in value)}
                    )
                else:
                    rows.append({"metric": metric_key, "value": str(value)})
                return

            rows.append({"metric": metric_key, "value": value})

        for key, value in metrics.items():
            _flatten(str(key), value)
        return rows

    def _validation_metric_rows(self, run_result: RunResult) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        validations = [
            ("health", run_result.health),
            ("categorical", run_result.categorical_validation),
            ("covariate", run_result.covariate_validation),
            ("identify_variables", run_result.id_variable_validation),
            ("assumptions", run_result.assumption_validation),
        ]
        for validation_name, validation_result in validations:
            if validation_result is None:
                continue
            for item in self._metrics_rows(validation_result.metrics):
                rows.append(
                    {
                        "validation": validation_name,
                        "metric": item["metric"],
                        "value": item["value"],
                    }
                )
        return rows

    def _build_custom_table(
        self,
        spec: dict[str, Any],
        dataset: Any,
        run_result: RunResult,
        tables_dir: Path,
        preview_rows: int,
    ) -> tuple[TableArtifact | None, list[Flag]]:
        flags: list[Flag] = []
        if not isinstance(spec, dict):
            flags.append(
                Flag(
                    code="custom_table_spec_invalid",
                    message="Custom table specification must be a dictionary.",
                    severity="WARN",
                    stage="tables",
                )
            )
            return None, flags

        table_id = self._normalize_id(str(spec.get("id", "")))
        if not table_id:
            flags.append(
                Flag(
                    code="custom_table_missing_id",
                    message="Custom table spec is missing a valid 'id' field.",
                    severity="WARN",
                    stage="tables",
                )
            )
            return None, flags

        source = str(spec.get("source", ""))
        if not source:
            flags.append(
                Flag(
                    code="custom_table_missing_source",
                    message=f"Custom table '{table_id}' is missing source.",
                    severity="WARN",
                    stage="tables",
                )
            )
            return None, flags

        try:
            frame = self._source_frame(source, dataset, run_result)
        except Exception as exc:
            flags.append(
                Flag(
                    code="custom_table_source_error",
                    message=f"Custom table '{table_id}' source error: {exc}",
                    severity="WARN",
                    stage="tables",
                )
            )
            return None, flags

        frame = frame.copy()
        query_expr = spec.get("query")
        if isinstance(query_expr, str) and query_expr.strip():
            try:
                frame = frame.query(query_expr)
            except Exception as exc:
                flags.append(
                    Flag(
                        code="custom_table_query_error",
                        message=f"Custom table '{table_id}' query failed: {exc}",
                        severity="WARN",
                        stage="tables",
                    )
                )

        columns = spec.get("columns")
        if isinstance(columns, list) and columns:
            missing_cols = [col for col in columns if col not in frame.columns]
            if missing_cols:
                flags.append(
                    Flag(
                        code="custom_table_missing_columns",
                        message=(
                            f"Custom table '{table_id}' missing columns: "
                            f"{', '.join(str(col) for col in missing_cols)}"
                        ),
                        severity="WARN",
                        stage="tables",
                    )
                )
            selected = [col for col in columns if col in frame.columns]
            if selected:
                frame = frame[selected]

        rename_map = spec.get("rename")
        if isinstance(rename_map, dict) and rename_map:
            frame = frame.rename(columns=rename_map)

        sort_by = spec.get("sort_by")
        if isinstance(sort_by, str):
            sort_by = [sort_by]
        if isinstance(sort_by, list) and sort_by:
            valid_sort_columns = [col for col in sort_by if col in frame.columns]
            if valid_sort_columns:
                ascending = spec.get("ascending", True)
                frame = frame.sort_values(by=valid_sort_columns, ascending=ascending)

        limit = spec.get("limit")
        if isinstance(limit, int) and limit > 0:
            frame = frame.head(limit)

        format_spec = spec.get("format")
        frame, format_flags = self._apply_column_formats(frame, format_spec, table_id)
        flags.extend(format_flags)

        include_in_pdf = bool(spec.get("include_in_pdf", True))
        title = str(spec.get("title", table_id.replace("_", " ").title()))
        artifact = self._write_table(
            frame=frame,
            table_id=table_id,
            title=title,
            source=source,
            tables_dir=tables_dir,
            preview_rows=preview_rows,
            include_in_pdf=include_in_pdf,
        )
        return artifact, flags

    def _resolve_format_spec(self, default_formats: Any, table_id: str) -> dict[str, Any] | None:
        if not isinstance(default_formats, dict):
            return None
        normalized_id = self._normalize_id(table_id)
        if table_id in default_formats and isinstance(default_formats[table_id], dict):
            return default_formats[table_id]
        if normalized_id in default_formats and isinstance(default_formats[normalized_id], dict):
            return default_formats[normalized_id]
        return None

    def _parse_format_rule(self, rule: Any) -> tuple[str | None, dict[str, Any]]:
        if isinstance(rule, str):
            raw = rule.strip()
            if not raw:
                return None, {}
            if ":" in raw:
                prefix, suffix = raw.split(":", 1)
                prefix = prefix.strip().lower()
                suffix = suffix.strip()
                if suffix.isdigit():
                    return prefix, {"decimals": int(suffix)}
                return prefix, {}
            return raw.lower(), {}
        if isinstance(rule, dict):
            kind = str(rule.get("type", "")).strip().lower()
            options = dict(rule)
            options.pop("type", None)
            return (kind if kind else None), options
        return None, {}

    def _format_number(self, value: Any, kind: str, options: dict[str, Any]) -> str:
        null_value = str(options.get("null", ""))
        if pd.isna(value):
            return null_value
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)

        decimals = int(options.get("decimals", 2))
        thousands = bool(options.get("thousands", False))
        prefix = str(options.get("prefix", ""))
        suffix = str(options.get("suffix", ""))

        if kind == "decimal":
            pattern = f"{{:{',' if thousands else ''}.{decimals}f}}"
            return f"{prefix}{pattern.format(number)}{suffix}"

        if kind == "percent":
            multiply_by_100 = bool(options.get("multiply_by_100", True))
            scaled = number * 100.0 if multiply_by_100 else number
            pattern = f"{{:{',' if thousands else ''}.{decimals}f}}"
            percent_suffix = suffix if suffix else "%"
            return f"{prefix}{pattern.format(scaled)}{percent_suffix}"

        if kind == "integer":
            rounded = int(round(number))
            pattern = "{:,d}" if thousands else "{:d}"
            return f"{prefix}{pattern.format(rounded)}{suffix}"

        if kind == "scientific":
            pattern = f"{{:.{decimals}e}}"
            return f"{prefix}{pattern.format(number)}{suffix}"

        if kind == "string":
            return str(value)

        pattern = options.get("pattern")
        if isinstance(pattern, str) and pattern:
            try:
                return pattern.format(number)
            except Exception:
                return str(value)
        return str(value)

    def _apply_column_formats(
        self,
        frame: pd.DataFrame,
        format_spec: Any,
        table_id: str,
    ) -> tuple[pd.DataFrame, list[Flag]]:
        if not isinstance(format_spec, dict) or not format_spec:
            return frame, []

        flags: list[Flag] = []
        output = frame.copy()
        for column_name, rule in format_spec.items():
            if column_name not in output.columns:
                flags.append(
                    Flag(
                        code="table_format_column_missing",
                        message=(
                            f"Table '{table_id}' format rule references missing column '{column_name}'."
                        ),
                        severity="WARN",
                        stage="tables",
                    )
                )
                continue

            kind, options = self._parse_format_rule(rule)
            if kind is None:
                flags.append(
                    Flag(
                        code="table_format_rule_invalid",
                        message=f"Table '{table_id}' has invalid format rule for column '{column_name}'.",
                        severity="WARN",
                        stage="tables",
                    )
                )
                continue

            supported = {"decimal", "percent", "integer", "scientific", "string", "pattern"}
            if kind not in supported:
                flags.append(
                    Flag(
                        code="table_format_type_unknown",
                        message=(
                            f"Table '{table_id}' has unknown format type '{kind}' "
                            f"for column '{column_name}'."
                        ),
                        severity="WARN",
                        stage="tables",
                    )
                )
                continue

            normalized_kind = kind if kind != "pattern" else "string"
            output[column_name] = output[column_name].map(
                lambda value, k=kind, opt=options: self._format_number(value, k, opt)
                if k != "string"
                else (str(value) if not pd.isna(value) else str(opt.get("null", "")))
            )
            if normalized_kind == "string" and "pattern" in options:
                output[column_name] = output[column_name].map(
                    lambda value, opt=options: opt["pattern"].format(value)
                )
        return output, flags

    def _write_table(
        self,
        frame: pd.DataFrame,
        table_id: str,
        title: str,
        source: str,
        tables_dir: Path,
        preview_rows: int,
        include_in_pdf: bool,
    ) -> TableArtifact:
        safe_frame = frame if frame is not None else pd.DataFrame()
        csv_path = tables_dir / f"{table_id}.csv"
        safe_frame.to_csv(csv_path, index=False)
        preview = safe_frame.head(preview_rows).to_dict("records")
        return TableArtifact(
            table_id=table_id,
            path=csv_path,
            title=title,
            source=source,
            columns=[str(col) for col in safe_frame.columns],
            row_count=int(len(safe_frame)),
            preview_rows=preview,
            include_in_pdf=include_in_pdf,
        )
