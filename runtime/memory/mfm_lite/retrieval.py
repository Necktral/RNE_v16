"""Retrieval estructural desde memoria multiescala."""

from __future__ import annotations

from typing import Any, Dict, List

from runtime.storage import StorageFacade

# Penalty applied to cross-scenario memories in analogical mode
_CROSS_SCENARIO_PENALTY = 0.5


class MemoryRetrieval:
    def __init__(self, *, storage: StorageFacade):
        self.storage = storage

    def retrieve(
        self,
        *,
        run_id: str,
        query: Dict[str, Any],
        scales: list[str] | None = None,
        limit: int = 5,
        scenario_name: str | None = None,
        scenario_version: str | None = None,
        scenario_filter_mode: str = "strict_same_scenario",
    ) -> List[Dict[str, Any]]:
        """Retrieve memory records scored by structural overlap.

        Args:
            run_id: Run ID to scope retrieval.
            query: Query dict for overlap scoring.
            scales: Memory scales to search.
            limit: Max results to return.
            scenario_name: Filter memories by scenario name.
            scenario_version: Optionally filter by scenario version as well.
            scenario_filter_mode: 'strict_same_scenario' discards cross-scenario
                memories. 'cross_scenario_analogical' (alias 'analogical') keeps
                them with a penalty.

        Returns:
            List of scored memory dicts with retrieval_metrics.
        """
        # Normalize alias
        if scenario_filter_mode == "analogical":
            scenario_filter_mode = "cross_scenario_analogical"
        candidates = self.storage.retrieve_memory_records(
            run_id=run_id,
            scales=scales or ["macro", "meso", "micro"],
            limit=max(20, limit * 4),
        )

        same_scenario_count = 0
        cross_scenario_count = 0
        penalty_applied = False

        scored = []
        for item in candidates:
            structure = item.structure_json or {}
            meta = item.metadata or {}
            score = self._score(query=query, structure=structure)

            # Scenario filtering
            is_cross_scenario = False
            if scenario_name is not None:
                item_scenario = (
                    meta.get("scenario_metadata", {}).get("scenario_name")
                    or meta.get("scenario_name")
                )
                is_same_scenario = item_scenario == scenario_name
                if scenario_version is not None and is_same_scenario:
                    item_version = meta.get("scenario_metadata", {}).get("scenario_version")
                    is_same_scenario = item_version == scenario_version

                if is_same_scenario:
                    same_scenario_count += 1
                else:
                    cross_scenario_count += 1
                    is_cross_scenario = True
                    if scenario_filter_mode == "strict_same_scenario":
                        continue  # Discard cross-scenario memory
                    # Analogical mode: penalize but keep
                    score *= _CROSS_SCENARIO_PENALTY
                    penalty_applied = True

            scored.append((score, item, is_cross_scenario))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        out = []
        for score, item, is_cross in scored[:limit]:
            entry: Dict[str, Any] = {
                "memory_id": item.memory_id,
                "scale": item.scale,
                "score": score,
                "structure": item.structure_json,
                "ioc_proxy": item.ioc_proxy,
                "support_count": item.support_count,
            }
            if is_cross:
                entry["analogical_source"] = True
            out.append(entry)

        # Attach retrieval metrics to each result for observability
        retrieval_metrics = {
            "retrieved_same_scenario_count": same_scenario_count,
            "retrieved_cross_scenario_count": cross_scenario_count,
            "scenario_filter_mode": scenario_filter_mode,
            "cross_scenario_penalty_applied": penalty_applied,
        }
        for entry in out:
            entry["retrieval_metrics"] = retrieval_metrics

        return out

    def _score(self, *, query: Dict[str, Any], structure: Dict[str, Any]) -> float:
        query_tokens = {str(v) for v in query.values() if v is not None}
        if not query_tokens:
            return 0.0
        structure_tokens = set()
        for value in structure.values():
            if isinstance(value, list):
                structure_tokens.update(str(item) for item in value)
            else:
                structure_tokens.add(str(value))
        if not structure_tokens:
            return 0.0
        overlap = len(query_tokens & structure_tokens) / len(query_tokens | structure_tokens)
        return float(overlap)
