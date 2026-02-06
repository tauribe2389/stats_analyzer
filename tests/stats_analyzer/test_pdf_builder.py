from __future__ import annotations

from stats_analyzer.reporting.pdf_builder import PdfReportBuilder


def test_pdf_builder_formats_metric_values_by_type() -> None:
    builder = PdfReportBuilder()

    assert builder._format_metric_value(0.1234567) == "0.12346"
    assert builder._format_metric_value(42) == "42"
    assert builder._format_metric_value("HC3") == "HC3"
    assert builder._format_metric_value(True) == "True"


def test_pdf_builder_plans_landscape_for_wide_table_in_auto_mode() -> None:
    builder = PdfReportBuilder()
    headers = [f"col_{idx}" for idx in range(1, 9)]
    rows = [["x" * 30 for _ in headers] for _ in range(3)]

    plan = builder._plan_table_segments(
        headers=headers,
        rows=rows,
        portrait_width=220.0,
        landscape_width=420.0,
        overflow_mode="auto",
        truncate_chars=80,
        min_col_width=52.0,
        min_col_width_truncated=32.0,
        max_col_width=240.0,
        split_key_columns=1,
        split_orientation="auto",
    )

    assert plan["strategy"] == "landscape"
    assert plan["overflowed"] is True
    assert len(plan["segments"]) == 1
    assert plan["segments"][0]["orientation"] == "landscape"


def test_pdf_builder_plans_split_when_split_mode_selected() -> None:
    builder = PdfReportBuilder()
    headers = [f"col_{idx}" for idx in range(1, 9)]
    rows = [["y" * 25 for _ in headers] for _ in range(2)]

    plan = builder._plan_table_segments(
        headers=headers,
        rows=rows,
        portrait_width=220.0,
        landscape_width=420.0,
        overflow_mode="split",
        truncate_chars=80,
        min_col_width=52.0,
        min_col_width_truncated=32.0,
        max_col_width=240.0,
        split_key_columns=1,
        split_orientation="portrait",
    )

    assert plan["strategy"] == "split"
    assert len(plan["segments"]) > 1
    assert all(segment["orientation"] == "portrait" for segment in plan["segments"])
    assert all(segment["headers"][0] == headers[0] for segment in plan["segments"])


def test_pdf_builder_uses_portrait_truncation_before_overflow_modes() -> None:
    builder = PdfReportBuilder()
    headers = ["very_long_header_one", "very_long_header_two", "very_long_header_three"]
    rows = [["z" * 40, "k" * 40, "m" * 40]]

    plan = builder._plan_table_segments(
        headers=headers,
        rows=rows,
        portrait_width=120.0,
        landscape_width=420.0,
        overflow_mode="auto",
        truncate_chars=20,
        min_col_width=52.0,
        min_col_width_truncated=32.0,
        max_col_width=240.0,
        split_key_columns=1,
        split_orientation="auto",
    )

    assert plan["strategy"] == "portrait_truncated"
    assert plan["overflowed"] is False
    assert plan["segments"][0]["orientation"] == "portrait"
