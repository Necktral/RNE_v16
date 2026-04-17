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

        previous_temp = previous_certificate.metadata.get("world_temperature")
        current_temp = (
            current_episode.get("result", {}).get("updated_world", {}).get("temperature")
        )
        temp_score = self._temperature_score(previous_temp, current_temp)
        score = (0.6 * sequence_score) + (0.4 * temp_score)
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

    def _temperature_score(self, previous_temp: Any, current_temp: Any) -> float:
        if not isinstance(previous_temp, (int, float)) or not isinstance(
            current_temp, (int, float)
        ):
            return 0.5
        delta = abs(float(current_temp) - float(previous_temp))
        return max(0.0, 1.0 - min(1.0, delta))
