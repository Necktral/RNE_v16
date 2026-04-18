"""Adaptador legacy T4 -> Corte T5."""

from __future__ import annotations

from .court_runtime import ConstitutionalCourtRuntime, CourtEpisodeResult


class ConstitutionalTrajectoryRuntime(ConstitutionalCourtRuntime):
    """Alias de compatibilidad para importadores T4."""


T4EpisodeResult = CourtEpisodeResult

