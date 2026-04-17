"""Retrieval estructural desde memoria multiescala."""

from __future__ import annotations

from typing import Any, Dict, List

from runtime.storage import StorageFacade


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
    ) -> List[Dict[str, Any]]:
        candidates = self.storage.retrieve_memory_records(
            run_id=run_id,
            scales=scales or ["macro", "meso", "micro"],
            limit=max(20, limit * 4),
        )
        scored = []
        for item in candidates:
            structure = item.structure_json or {}
            score = self._score(query=query, structure=structure)
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        out = []
        for score, item in scored[:limit]:
            out.append(
                {
                    "memory_id": item.memory_id,
                    "scale": item.scale,
                    "score": score,
                    "structure": item.structure_json,
                    "ioc_proxy": item.ioc_proxy,
                    "support_count": item.support_count,
                }
            )
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
