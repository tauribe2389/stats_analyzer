from __future__ import annotations

import math
from numbers import Real
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from stats_analyzer.core.models import RunResult


class PdfReportBuilder:
    """Create a PDF report using ReportLab from a run result and template."""

    def _format_metric_value(self, metric_value: Any) -> str:
        if isinstance(metric_value, bool):
            return str(metric_value)
        if isinstance(metric_value, Real):
            numeric = float(metric_value)
            if math.isfinite(numeric):
                return f"{numeric:.5g}"
            return str(numeric)
        return str(metric_value)

    def _normalize_choice(self, value: Any, allowed: set[str], default: str) -> str:
        normalized = str(value).strip().lower() if value is not None else ""
        return normalized if normalized in allowed else default

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        if max_chars <= 3:
            return text[:max_chars]
        return f"{text[: max_chars - 3]}..."

    def _truncate_cells(
        self,
        headers: list[str],
        rows: list[list[str]],
        max_chars: int,
    ) -> tuple[list[str], list[list[str]]]:
        clipped_headers = [self._truncate_text(str(header), max_chars) for header in headers]
        clipped_rows = [
            [self._truncate_text(str(cell), max_chars) for cell in row]
            for row in rows
        ]
        return clipped_headers, clipped_rows

    def _fit_table_widths(
        self,
        headers: list[str],
        rows: list[list[str]],
        available_width: float,
        min_col_width: float,
        max_col_width: float,
        char_width: float = 4.8,
        padding: float = 10.0,
    ) -> dict[str, Any]:
        column_count = len(headers)
        if column_count == 0:
            return {"fits": True, "widths": [], "total": 0.0}

        max_lengths: list[int] = []
        for col_idx in range(column_count):
            column_max = len(str(headers[col_idx]))
            for row in rows:
                if col_idx >= len(row):
                    continue
                column_max = max(column_max, len(str(row[col_idx])))
            max_lengths.append(column_max)

        desired = [
            max(min_col_width, min(max_col_width, (length * char_width) + padding))
            for length in max_lengths
        ]
        desired_total = float(sum(desired))
        if desired_total <= available_width + 1e-6:
            return {"fits": True, "widths": desired, "total": desired_total}

        minimum_total = float(min_col_width * column_count)
        if minimum_total > available_width + 1e-6:
            return {"fits": False, "widths": [min_col_width] * column_count, "total": minimum_total}

        shrink_ratio = (available_width - minimum_total) / (desired_total - minimum_total)
        widths = [
            min_col_width + ((width - min_col_width) * shrink_ratio)
            for width in desired
        ]
        total = float(sum(widths))
        if widths:
            widths[-1] += available_width - total
        return {"fits": True, "widths": widths, "total": float(sum(widths))}

    def _force_fit_widths(self, column_count: int, available_width: float) -> list[float]:
        if column_count <= 0:
            return []
        even_width = available_width / float(column_count)
        return [even_width] * column_count

    def _split_table_segments(
        self,
        headers: list[str],
        rows: list[list[str]],
        *,
        available_width: float,
        orientation: str,
        split_key_columns: int,
        min_col_width: float,
        max_col_width: float,
    ) -> list[dict[str, Any]]:
        column_count = len(headers)
        if column_count == 0:
            return []

        max_cols_per_chunk = max(1, int(available_width // max(min_col_width, 1.0)))
        key_count = max(0, min(split_key_columns, max(0, column_count - 1)))
        if max_cols_per_chunk <= key_count:
            key_count = max(0, max_cols_per_chunk - 1)
        non_key_chunk_size = max(1, max_cols_per_chunk - key_count)

        if key_count >= column_count:
            fit = self._fit_table_widths(
                headers=headers,
                rows=rows,
                available_width=available_width,
                min_col_width=min_col_width,
                max_col_width=max_col_width,
            )
            widths = fit["widths"] if fit["fits"] else self._force_fit_widths(column_count, available_width)
            return [
                {
                    "orientation": orientation,
                    "headers": headers,
                    "rows": rows,
                    "col_widths": widths,
                    "column_indices": list(range(column_count)),
                }
            ]

        segments: list[dict[str, Any]] = []
        key_indices = list(range(key_count))
        non_key_indices = list(range(key_count, column_count))
        for offset in range(0, len(non_key_indices), non_key_chunk_size):
            chunk_indices = non_key_indices[offset: offset + non_key_chunk_size]
            column_indices = [*key_indices, *chunk_indices]
            segment_headers = [headers[idx] for idx in column_indices]
            segment_rows = [
                [str(row[idx]) if idx < len(row) else "" for idx in column_indices]
                for row in rows
            ]
            fit = self._fit_table_widths(
                headers=segment_headers,
                rows=segment_rows,
                available_width=available_width,
                min_col_width=min_col_width,
                max_col_width=max_col_width,
            )
            widths = (
                fit["widths"]
                if fit["fits"]
                else self._force_fit_widths(len(segment_headers), available_width)
            )
            segments.append(
                {
                    "orientation": orientation,
                    "headers": segment_headers,
                    "rows": segment_rows,
                    "col_widths": widths,
                    "column_indices": column_indices,
                }
            )
        return segments

    def _plan_table_segments(
        self,
        headers: list[str],
        rows: list[list[str]],
        *,
        portrait_width: float,
        landscape_width: float,
        overflow_mode: str,
        truncate_chars: int,
        min_col_width: float,
        min_col_width_truncated: float,
        max_col_width: float,
        split_key_columns: int,
        split_orientation: str,
    ) -> dict[str, Any]:
        mode = self._normalize_choice(
            overflow_mode,
            {"auto", "portrait", "landscape", "split"},
            "auto",
        )
        split_mode = self._normalize_choice(
            split_orientation,
            {"auto", "portrait", "landscape"},
            "auto",
        )

        direct_fit = self._fit_table_widths(
            headers=headers,
            rows=rows,
            available_width=portrait_width,
            min_col_width=min_col_width,
            max_col_width=max_col_width,
        )
        if direct_fit["fits"]:
            return {
                "overflowed": False,
                "strategy": "portrait",
                "segments": [
                    {
                        "orientation": "portrait",
                        "headers": headers,
                        "rows": rows,
                        "col_widths": direct_fit["widths"],
                        "column_indices": list(range(len(headers))),
                    }
                ],
                "notes": [],
            }

        truncated_headers, truncated_rows = self._truncate_cells(
            headers=headers,
            rows=rows,
            max_chars=max(4, int(truncate_chars)),
        )
        truncated_fit = self._fit_table_widths(
            headers=truncated_headers,
            rows=truncated_rows,
            available_width=portrait_width,
            min_col_width=min_col_width_truncated,
            max_col_width=max_col_width,
        )
        if truncated_fit["fits"]:
            return {
                "overflowed": False,
                "strategy": "portrait_truncated",
                "segments": [
                    {
                        "orientation": "portrait",
                        "headers": truncated_headers,
                        "rows": truncated_rows,
                        "col_widths": truncated_fit["widths"],
                        "column_indices": list(range(len(truncated_headers))),
                    }
                ],
                "notes": [
                    "Long values were truncated in the PDF preview to fit page width.",
                ],
            }

        if mode in {"auto", "landscape"}:
            landscape_fit = self._fit_table_widths(
                headers=truncated_headers,
                rows=truncated_rows,
                available_width=landscape_width,
                min_col_width=min_col_width_truncated,
                max_col_width=max_col_width,
            )
            if landscape_fit["fits"]:
                return {
                    "overflowed": True,
                    "strategy": "landscape",
                    "segments": [
                        {
                            "orientation": "landscape",
                            "headers": truncated_headers,
                            "rows": truncated_rows,
                            "col_widths": landscape_fit["widths"],
                            "column_indices": list(range(len(truncated_headers))),
                        }
                    ],
                    "notes": [
                        "Rendered on a landscape page because portrait width was insufficient.",
                        "Long values were truncated in the PDF preview to fit page width.",
                    ],
                }

        if mode in {"auto", "split"}:
            if split_mode == "landscape":
                split_target = "landscape"
            elif split_mode == "portrait":
                split_target = "portrait"
            else:
                split_target = "landscape" if landscape_width > portrait_width else "portrait"
            split_width = landscape_width if split_target == "landscape" else portrait_width
            segments = self._split_table_segments(
                headers=truncated_headers,
                rows=truncated_rows,
                available_width=split_width,
                orientation=split_target,
                split_key_columns=max(0, int(split_key_columns)),
                min_col_width=min_col_width_truncated,
                max_col_width=max_col_width,
            )
            return {
                "overflowed": True,
                "strategy": "split",
                "segments": segments,
                "notes": [
                    "Table was split into column groups because it exceeded page width.",
                    "Long values were truncated in the PDF preview to fit page width.",
                ],
            }

        forced_widths = self._force_fit_widths(len(truncated_headers), portrait_width)
        return {
            "overflowed": True,
            "strategy": "portrait_forced",
            "segments": [
                {
                    "orientation": "portrait",
                    "headers": truncated_headers,
                    "rows": truncated_rows,
                    "col_widths": forced_widths,
                    "column_indices": list(range(len(truncated_headers))),
                }
            ],
            "notes": [
                "Table width exceeded configured layout limits; using aggressive compression in PDF preview.",
                "Long values were truncated in the PDF preview to fit page width.",
            ],
        }

    def _switch_orientation(
        self,
        story: list[Any],
        current_orientation: str,
        target_orientation: str,
        NextPageTemplate: Any,
        PageBreak: Any,
    ) -> str:
        if current_orientation == target_orientation:
            return current_orientation
        story.append(NextPageTemplate(target_orientation))
        story.append(PageBreak())
        return target_orientation

    def _render_table_plan(
        self,
        *,
        story: list[Any],
        current_orientation: str,
        title: str,
        source_line: str,
        column_count: int,
        plan: dict[str, Any],
        notes: list[str],
        styles: Any,
        header_style: Any,
        cell_style: Any,
        Table: Any,
        TableStyle: Any,
        Paragraph: Any,
        Spacer: Any,
        colors: Any,
        NextPageTemplate: Any,
        PageBreak: Any,
    ) -> str:
        segments = plan.get("segments", [])
        segment_count = len(segments)
        for idx, segment in enumerate(segments, start=1):
            target_orientation = str(segment.get("orientation", "portrait"))
            current_orientation = self._switch_orientation(
                story=story,
                current_orientation=current_orientation,
                target_orientation=target_orientation,
                NextPageTemplate=NextPageTemplate,
                PageBreak=PageBreak,
            )

            part_title = title if segment_count == 1 else f"{title} (part {idx}/{segment_count})"
            story.append(Paragraph(escape(part_title), styles["Heading3"]))
            story.append(Paragraph(escape(source_line), styles["Normal"]))

            segment_headers = [str(item) for item in segment.get("headers", [])]
            segment_rows = [
                [str(cell) for cell in row]
                for row in segment.get("rows", [])
            ]
            if not segment_rows:
                segment_rows = [["" for _ in segment_headers]]

            table_data = [[Paragraph(escape(item), header_style) for item in segment_headers]]
            for row in segment_rows:
                table_data.append([Paragraph(escape(cell), cell_style) for cell in row])

            table = Table(
                table_data,
                repeatRows=1,
                colWidths=segment.get("col_widths"),
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 2),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ]
                )
            )
            story.append(table)

            if segment_count > 1:
                indices = [int(i) for i in segment.get("column_indices", [])]
                if indices:
                    start_col = min(indices) + 1
                    end_col = max(indices) + 1
                    story.append(
                        Paragraph(
                            f"Columns {start_col}-{end_col} of {column_count}.",
                            styles["Italic"],
                        )
                    )

            if idx == segment_count:
                for note in notes:
                    story.append(Paragraph(escape(note), styles["Italic"]))
            story.append(Spacer(1, 10))
        return current_orientation

    def build(self, result: RunResult, template: dict[str, Any], output_path: Path) -> Path:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import (
            BaseDocTemplate,
            Frame,
            Image,
            NextPageTemplate,
            PageBreak,
            PageTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc = BaseDocTemplate(str(output_path), pagesize=letter)
        portrait_size = letter
        landscape_size = landscape(letter)
        portrait_width = portrait_size[0] - doc.leftMargin - doc.rightMargin
        portrait_height = portrait_size[1] - doc.topMargin - doc.bottomMargin
        landscape_width = landscape_size[0] - doc.leftMargin - doc.rightMargin
        landscape_height = landscape_size[1] - doc.topMargin - doc.bottomMargin
        doc.addPageTemplates(
            [
                PageTemplate(
                    id="portrait",
                    pagesize=portrait_size,
                    frames=[
                        Frame(
                            doc.leftMargin,
                            doc.bottomMargin,
                            portrait_width,
                            portrait_height,
                            id="portrait_frame",
                        )
                    ],
                ),
                PageTemplate(
                    id="landscape",
                    pagesize=landscape_size,
                    frames=[
                        Frame(
                            doc.leftMargin,
                            doc.bottomMargin,
                            landscape_width,
                            landscape_height,
                            id="landscape_frame",
                        )
                    ],
                ),
            ]
        )
        styles = getSampleStyleSheet()
        table_header_style = ParagraphStyle(
            "StatsTableHeader",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=9,
            wordWrap="CJK",
        )
        table_cell_style = ParagraphStyle(
            "StatsTableCell",
            parent=styles["BodyText"],
            fontSize=8,
            leading=9,
            wordWrap="CJK",
        )
        story: list[Any] = []
        current_orientation = "portrait"

        title = template.get("title", "Statistical Analysis Report")
        subtitle = template.get("subtitle", "")
        story.append(Paragraph(escape(str(title)), styles["Title"]))
        if subtitle:
            story.append(Paragraph(escape(str(subtitle)), styles["Normal"]))
        story.append(Spacer(1, 12))

        story.append(Paragraph("Executive Summary", styles["Heading2"]))
        story.append(
            Paragraph(
                escape(
                    f"Input: {result.request.input_path} | "
                    f"Analysis: {result.model.analysis_type if result.model else 'n/a'}"
                ),
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 10))

        if template.get("include_flag_table", True):
            story.append(Paragraph("Flags", styles["Heading2"]))
            flag_headers = ["Severity", "Code", "Stage", "Message"]
            flag_rows = [
                [str(flag.severity), str(flag.code), str(flag.stage), str(flag.message)]
                for flag in result.flags
            ]
            if not flag_rows:
                flag_rows = [["INFO", "none", "n/a", "No flags raised."]]
            flag_fit = self._fit_table_widths(
                headers=flag_headers,
                rows=flag_rows,
                available_width=portrait_width,
                min_col_width=55.0,
                max_col_width=300.0,
            )
            flag_widths = (
                flag_fit["widths"]
                if flag_fit["fits"]
                else self._force_fit_widths(len(flag_headers), portrait_width)
            )
            flag_table_data = [[Paragraph(escape(item), table_header_style) for item in flag_headers]]
            for row in flag_rows:
                flag_table_data.append([Paragraph(escape(item), table_cell_style) for item in row])

            table = Table(flag_table_data, repeatRows=1, colWidths=flag_widths)
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
            story.append(Paragraph(escape(f"Formula: {result.model.formula}"), styles["Code"]))
            for metric_name, metric_value in result.model.fit_statistics.items():
                story.append(
                    Paragraph(
                        escape(f"{metric_name}: {self._format_metric_value(metric_value)}"),
                        styles["Normal"],
                    )
                )
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
        overflow_mode = self._normalize_choice(
            table_config.get("pdf_overflow_mode", "auto"),
            {"auto", "portrait", "landscape", "split"},
            "auto",
        )
        overflow_placement = self._normalize_choice(
            table_config.get("pdf_overflow_placement", "inline"),
            {"inline", "appendix"},
            "inline",
        )
        truncate_chars = max(4, int(table_config.get("pdf_truncate_chars", 80)))
        min_col_width = max(10.0, float(table_config.get("pdf_min_col_width", 52)))
        min_col_width_truncated = max(
            8.0,
            float(table_config.get("pdf_min_col_width_truncated", 32)),
        )
        max_col_width = max(min_col_width_truncated, float(table_config.get("pdf_max_col_width", 240)))
        split_key_columns = max(0, int(table_config.get("pdf_split_key_columns", 1)))
        split_orientation = self._normalize_choice(
            table_config.get("pdf_split_orientation", "auto"),
            {"auto", "portrait", "landscape"},
            "auto",
        )
        appendix_entries: list[dict[str, Any]] = []

        include_tables_section = "tables" in template.get("sections", []) or bool(result.tables)
        if include_tables_section and result.tables:
            story.append(Paragraph("Tables", styles["Heading2"]))
            for table_artifact in result.tables:
                if not table_artifact.include_in_pdf:
                    continue

                headers = table_artifact.columns[:max_columns]
                preview = table_artifact.preview_rows[:max_rows]
                if not headers:
                    story.append(Paragraph(escape(str(table_artifact.title)), styles["Heading3"]))
                    story.append(Paragraph("No columns available for this table.", styles["Normal"]))
                    story.append(Spacer(1, 8))
                    continue

                table_rows = [[str(row.get(col, "")) for col in headers] for row in preview]
                if not table_rows:
                    table_rows = [["" for _ in headers]]

                base_notes: list[str] = []
                if table_artifact.row_count > max_rows:
                    base_notes.append(
                        f"Showing first {max_rows} rows of {table_artifact.row_count}. Full table saved as CSV."
                    )
                if len(table_artifact.columns) > max_columns:
                    base_notes.append(
                        f"Showing first {max_columns} columns of {len(table_artifact.columns)}."
                    )

                plan = self._plan_table_segments(
                    headers=[str(item) for item in headers],
                    rows=table_rows,
                    portrait_width=portrait_width,
                    landscape_width=landscape_width,
                    overflow_mode=overflow_mode,
                    truncate_chars=truncate_chars,
                    min_col_width=min_col_width,
                    min_col_width_truncated=min_col_width_truncated,
                    max_col_width=max_col_width,
                    split_key_columns=split_key_columns,
                    split_orientation=split_orientation,
                )
                source_line = (
                    f"Source: {table_artifact.source} | Rows: {table_artifact.row_count} | File: {table_artifact.path}"
                )
                combined_notes = [*base_notes, *plan.get("notes", [])]

                if overflow_placement == "appendix" and bool(plan.get("overflowed")):
                    story.append(Paragraph(escape(str(table_artifact.title)), styles["Heading3"]))
                    story.append(Paragraph(escape(source_line), styles["Normal"]))
                    story.append(
                        Paragraph(
                            "This table exceeded page width and is shown in the Wide Tables Appendix.",
                            styles["Italic"],
                        )
                    )
                    story.append(Spacer(1, 10))
                    appendix_entries.append(
                        {
                            "title": str(table_artifact.title),
                            "source_line": source_line,
                            "column_count": len(headers),
                            "plan": plan,
                            "notes": combined_notes,
                        }
                    )
                    continue

                current_orientation = self._render_table_plan(
                    story=story,
                    current_orientation=current_orientation,
                    title=str(table_artifact.title),
                    source_line=source_line,
                    column_count=len(headers),
                    plan=plan,
                    notes=combined_notes,
                    styles=styles,
                    header_style=table_header_style,
                    cell_style=table_cell_style,
                    Table=Table,
                    TableStyle=TableStyle,
                    Paragraph=Paragraph,
                    Spacer=Spacer,
                    colors=colors,
                    NextPageTemplate=NextPageTemplate,
                    PageBreak=PageBreak,
                )

        current_orientation = self._switch_orientation(
            story=story,
            current_orientation=current_orientation,
            target_orientation="portrait",
            NextPageTemplate=NextPageTemplate,
            PageBreak=PageBreak,
        )
        if result.figures:
            story.append(Paragraph("Figures", styles["Heading2"]))
            for figure in result.figures:
                if figure.path.exists():
                    story.append(Paragraph(escape(str(figure.title)), styles["Heading3"]))
                    story.append(Image(str(figure.path), width=450, height=260))
                    story.append(Paragraph(escape(str(figure.caption)), styles["Normal"]))
                    story.append(Spacer(1, 10))

        if appendix_entries:
            current_orientation = self._switch_orientation(
                story=story,
                current_orientation=current_orientation,
                target_orientation="portrait",
                NextPageTemplate=NextPageTemplate,
                PageBreak=PageBreak,
            )
            story.append(Paragraph("Wide Tables Appendix", styles["Heading2"]))
            for entry in appendix_entries:
                current_orientation = self._render_table_plan(
                    story=story,
                    current_orientation=current_orientation,
                    title=entry["title"],
                    source_line=entry["source_line"],
                    column_count=int(entry["column_count"]),
                    plan=entry["plan"],
                    notes=entry["notes"],
                    styles=styles,
                    header_style=table_header_style,
                    cell_style=table_cell_style,
                    Table=Table,
                    TableStyle=TableStyle,
                    Paragraph=Paragraph,
                    Spacer=Spacer,
                    colors=colors,
                    NextPageTemplate=NextPageTemplate,
                    PageBreak=PageBreak,
                )

        doc.build(story)
        return output_path
