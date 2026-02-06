from __future__ import annotations

from dataclasses import dataclass, field

from stats_analyzer.core.models import FigureArtifact


@dataclass
class FigureRegistry:
    figures: list[FigureArtifact] = field(default_factory=list)

    def register(self, figure: FigureArtifact) -> None:
        self.figures.append(figure)

    def all(self) -> list[FigureArtifact]:
        return list(self.figures)
