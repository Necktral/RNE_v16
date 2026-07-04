"""Runtime lineage T4.

Adaptador compatible hacia lineage legacy.
"""

from __future__ import annotations

from dataclasses import dataclass

from .lineage import LineageState


@dataclass
class LineageRuntime:
    state: LineageState

    def consistency_score(self) -> float:
        return self.state.consistency_score()

    def record_divergence(self, divergence_id: str, description: str) -> None:
        self.state.record_divergence(divergence_id=divergence_id, description=description)
