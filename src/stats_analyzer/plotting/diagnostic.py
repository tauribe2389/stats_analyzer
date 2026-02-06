from __future__ import annotations

from pathlib import Path
from typing import Any

from stats_analyzer.core.models import FigureArtifact, ModelResult


class DiagnosticPlotter:
    """Generate model diagnostic plots."""

    def run(self, model_result: ModelResult, output_dir: Path) -> list[FigureArtifact]:
        if model_result.raw_result is None:
            return []

        import matplotlib.pyplot as plt

        output_dir.mkdir(parents=True, exist_ok=True)
        figure_path = output_dir / "diagnostic_residuals.png"

        fitted = model_result.raw_result.fittedvalues
        residuals = model_result.raw_result.resid

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(fitted, residuals, alpha=0.7)
        ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax.set_title("Residuals vs Fitted")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
        fig.tight_layout()
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)

        return [
            FigureArtifact(
                figure_id="diagnostic_residuals",
                path=figure_path,
                title="Residuals vs Fitted",
                caption="Diagnostic residual plot for model quality review.",
                section="assumptions",
                tags=["diagnostic", "residuals"],
            )
        ]

