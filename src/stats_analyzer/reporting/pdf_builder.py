from __future__ import annotations

from pathlib import Path
from typing import Any

from stats_analyzer.core.models import RunResult


class PdfReportBuilder:
    """Create a PDF report using ReportLab from a run result and template."""

    def build(self, result: RunResult, template: dict[str, Any], output_path: Path) -> Path:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        title = template.get("title", "Statistical Analysis Report")
        subtitle = template.get("subtitle", "")
        story.append(Paragraph(title, styles["Title"]))
        if subtitle:
            story.append(Paragraph(subtitle, styles["Normal"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Executive Summary", styles["Heading2"]))
        story.append(
            Paragraph(
                f"Input: {result.request.input_path} | Analysis: {result.model.analysis_type if result.model else 'n/a'}",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 10))

        if template.get("include_flag_table", True):
            story.append(Paragraph("Flags", styles["Heading2"]))
            table_data = [["Severity", "Code", "Stage", "Message"]]
            for flag in result.flags:
                table_data.append([flag.severity, flag.code, flag.stage, flag.message])
            if len(table_data) == 1:
                table_data.append(["INFO", "none", "n/a", "No flags raised."])

            table = Table(table_data, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 12))

        if result.model:
            story.append(Paragraph("Model Summary", styles["Heading2"]))
            story.append(Paragraph(f"Formula: {result.model.formula}", styles["Code"]))
            for metric_name, metric_value in result.model.fit_statistics.items():
                story.append(Paragraph(f"{metric_name}: {metric_value:.5g}", styles["Normal"]))
            story.append(Spacer(1, 12))

            if result.model.adjusted_means:
                story.append(Paragraph("Adjusted Means", styles["Heading2"]))
                headers = list(result.model.adjusted_means[0].keys())
                rows = [[str(row.get(col, "")) for col in headers] for row in result.model.adjusted_means]
                table = Table([headers, *rows], repeatRows=1)
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 12))

        table_config = template.get("tables", {})
        max_rows = int(table_config.get("pdf_max_rows", 20))
        max_columns = int(table_config.get("pdf_max_columns", 8))
        include_tables_section = "tables" in template.get("sections", []) or bool(result.tables)
        if include_tables_section and result.tables:
            story.append(Paragraph("Tables", styles["Heading2"]))
            for table_artifact in result.tables:
                if not table_artifact.include_in_pdf:
                    continue

                story.append(Paragraph(table_artifact.title, styles["Heading3"]))
                story.append(
                    Paragraph(
                        f"Source: {table_artifact.source} | Rows: {table_artifact.row_count} | File: {table_artifact.path}",
                        styles["Normal"],
                    )
                )

                headers = table_artifact.columns[:max_columns]
                preview = table_artifact.preview_rows[:max_rows]
                if not headers:
                    story.append(Paragraph("No columns available for this table.", styles["Normal"]))
                    story.append(Spacer(1, 8))
                    continue

                table_rows = [[str(row.get(col, "")) for col in headers] for row in preview]
                if not table_rows:
                    table_rows = [["" for _ in headers]]

                table = Table([headers, *table_rows], repeatRows=1)
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ]
                    )
                )
                story.append(table)

                if table_artifact.row_count > max_rows:
                    story.append(
                        Paragraph(
                            f"Showing first {max_rows} rows of {table_artifact.row_count}. Full table saved as CSV.",
                            styles["Italic"],
                        )
                    )
                if len(table_artifact.columns) > max_columns:
                    story.append(
                        Paragraph(
                            f"Showing first {max_columns} columns of {len(table_artifact.columns)}.",
                            styles["Italic"],
                        )
                    )
                story.append(Spacer(1, 10))

        if result.figures:
            story.append(Paragraph("Figures", styles["Heading2"]))
            for figure in result.figures:
                if figure.path.exists():
                    story.append(Paragraph(figure.title, styles["Heading3"]))
                    story.append(Image(str(figure.path), width=450, height=260))
                    story.append(Paragraph(figure.caption, styles["Normal"]))
                    story.append(Spacer(1, 10))

        doc.build(story)
        return output_path
