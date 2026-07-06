"""Condensador micro/meso/macro para MFM_lite."""

from __future__ import annotations

from typing import Any, Dict, Iterable


class MFMCondenser:
    # Claves del contexto que NO deben guardarse en memoria: son ellas mismas memorias
    # recuperadas / trazas, y anidarlas hace que cada memoria contenga las anteriores →
    # crecimiento EXPONENCIAL del payload (episode.closed llegó a 677 MB, doblando por episodio).
    _RECURSIVE_CONTEXT_KEYS = (
        "retrieved_memory",
        "memory_hits",
        "memory_rag_attestation",
        "overlay_directives",
        "inherited_rules",
        "belief_state",
    )

    def _prune_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pruned = dict(context or {})
        for key in self._RECURSIVE_CONTEXT_KEYS:
            pruned.pop(key, None)
        return pruned

    def micro(self, *, episode_result: Dict[str, Any], certificate) -> Dict[str, Any]:
        episode = episode_result.get("episode", {})
        result = {
            "episode_id": episode.get("episode_id"),
            "result": episode.get("result", {}),
            # Contexto PODADO: firma de situación (observación, fórmula, intervención) sin
            # las memorias recuperadas — evita la recursión que reventaba el payload.
            "context": self._prune_context(episode.get("context", {})),
            "ioc_proxy": certificate.ioc_proxy,
            "continuity_score": certificate.continuity_score,
        }
        scenario_metadata = episode.get("scenario_metadata")
        if scenario_metadata:
            result["scenario_metadata"] = scenario_metadata
        return result

    def meso(self, *, episode_result: Dict[str, Any], certificate) -> Dict[str, Any]:
        episode = episode_result.get("episode", {})
        result = episode.get("result", {})
        context = episode.get("context", {})
        pattern_key = f"{context.get('formula','NA')}"
        meso_result = {
            "pattern_key": pattern_key,
            "reasoning_sequence": result.get("reasoning_sequence", []),
            "relation_kind": result.get("relation_kind"),
            "temperature": result.get("updated_world", {}).get("temperature"),
            "ioc_proxy": certificate.ioc_proxy,
        }
        scenario_metadata = episode.get("scenario_metadata")
        if scenario_metadata:
            meso_result["scenario_metadata"] = scenario_metadata
        return meso_result

    def macro(
        self,
        *,
        pattern_key: str,
        records: Iterable[Dict[str, Any]],
        mean_ioc: float,
        std_ioc: float,
    ) -> Dict[str, Any]:
        return {
            "pattern_key": pattern_key,
            "support_count": len(list(records)),
            "mean_ioc": mean_ioc,
            "std_ioc": std_ioc,
        }
