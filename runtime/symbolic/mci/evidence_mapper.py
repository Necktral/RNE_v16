"""Puente conservador entre referencias externas y evidencias del MCI."""

from __future__ import annotations

from collections.abc import Iterable

from .causal_learning import TransitionEvidence


def map_external_evidence_refs(
    references: Iterable[str],
    recent_evidence: Iterable[TransitionEvidence],
) -> tuple[str, ...]:
    """Traduce solo referencias inequívocas; nunca fabrica evidencia."""

    lookup: dict[str, set[str]] = {}
    for evidence in recent_evidence:
        aliases = {
            evidence.evidence_id,
            evidence.replay_unit_id,
            evidence.decision_trace_sha256 or "",
        }
        for alias in aliases - {""}:
            lookup.setdefault(alias, set()).add(evidence.evidence_id)
    mapped: set[str] = set()
    for reference in references:
        matches = lookup.get(str(reference), set())
        if len(matches) == 1:
            mapped.update(matches)
    return tuple(sorted(mapped))
