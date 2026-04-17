"""Condensador micro/meso/macro para MFM_lite."""

from __future__ import annotations

from typing import Any, Dict, Iterable


class MFMCondenser:
    def micro(self, *, episode_result: Dict[str, Any], certificate) -> Dict[str, Any]:
        episode = episode_result.get("episode", {})
        result = {
            "episode_id": episode.get("episode_id"),
            "result": episode.get("result", {}),
            "context": episode.get("context", {}),
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
