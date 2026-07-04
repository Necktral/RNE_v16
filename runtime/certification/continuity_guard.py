"""Guardia de continuidad identitaria entre episodios."""

from __future__ import annotations

from typing import Any, Dict

from runtime.storage import EpisodeCertificateRecord


class ContinuityGuard:
    def __init__(self, *, continuity_alert_threshold: float = 0.35):
        self.continuity_alert_threshold = continuity_alert_threshold

    def score(
        self,
        *,
        previous_certificate: EpisodeCertificateRecord | None,
        current_episode: Dict[str, Any],
        fallback_continuity: float | None = None,
    ) -> float:
        if fallback_continuity is not None:
            return max(0.0, min(1.0, float(fallback_continuity)))
        if previous_certificate is None:
            return 1.0

        previous_sequence = previous_certificate.metadata.get("reasoning_sequence", [])
        current_sequence = current_episode.get("result", {}).get("reasoning_sequence", [])
        sequence_score = self._sequence_score(previous_sequence, current_sequence)

        # Variable principal del escenario (no hardcodea "temperature").
        scenario_metadata = current_episode.get("scenario_metadata", {}) or {}
        main_var = scenario_metadata.get("main_variable", "temperature")
        current_value = (
            current_episode.get("result", {}).get("updated_world", {}).get(main_var)
        )
        prev_meta = previous_certificate.metadata
        prev_main_var = prev_meta.get("world_main_variable", "temperature")
        if prev_main_var == main_var:
            previous_value = prev_meta.get(
                "world_main_variable_value", prev_meta.get("world_temperature")
            )
        else:
            previous_value = None  # transición cross-variable: no comparable
        variable_score = self._variable_stability_score(previous_value, current_value)
        score = (0.6 * sequence_score) + (0.4 * variable_score)
        return max(0.0, min(1.0, score))

    def has_alert(self, continuity_score: float) -> bool:
        return float(continuity_score) < self.continuity_alert_threshold

    def _sequence_score(self, previous: Any, current: Any) -> float:
        prev = list(previous or [])
        curr = list(current or [])
        if not prev and not curr:
            return 1.0
        if not prev or not curr:
            return 0.0
        matches = sum(1 for i in range(min(len(prev), len(curr))) if prev[i] == curr[i])
        return matches / max(len(prev), len(curr))

    def _variable_stability_score(self, previous_value: Any, current_value: Any) -> float:
        """Estabilidad de la variable principal: 1 - |Δ| (variable-agnóstica)."""
        if not isinstance(previous_value, (int, float)) or not isinstance(
            current_value, (int, float)
        ):
            return 0.5
        delta = abs(float(current_value) - float(previous_value))
        return max(0.0, 1.0 - min(1.0, delta))
