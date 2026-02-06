from __future__ import annotations

from pathlib import Path
from typing import Any

from stats_analyzer.core.models import FigureArtifact, ModelResult, ModelSpec


class ComparisonPlotter:
    """Plot observed values vs model-adjusted predictions."""

    def run(
        self,
        dataset: Any,
        model_spec: ModelSpec,
        model_result: ModelResult,
        output_dir: Path,
    ) -> list[FigureArtifact]:
        if model_result.raw_result is None:
            return []

        import matplotlib.pyplot as plt

        output_dir.mkdir(parents=True, exist_ok=True)
        figure_path = output_dir / "comparison_observed_vs_model.png"

        observed = dataset[model_spec.response]
        predicted = model_result.raw_result.predict(dataset)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(observed, predicted, alpha=0.7)
        min_axis = min(float(observed.min()), float(predicted.min()))
        max_axis = max(float(observed.max()), float(predicted.max()))
        ax.plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--", color="black", linewidth=1)
        ax.set_title("Observed vs Model-Predicted")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        fig.tight_layout()
        fig.savefig(figure_path, dpi=150)
        plt.close(fig)

        return [
            FigureArtifact(
                figure_id="comparison_observed_vs_model",
                path=figure_path,
                title="Observed vs Model-Predicted",
                caption="Comparison of raw response values against model predictions.",
                section="figures",
                tags=["comparison", "model-fit"],
            )
        ]

