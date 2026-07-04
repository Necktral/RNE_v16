"""Guardia de memoria entre escalas para evitar contaminación estructural."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from .contracts import CrossScaleMemoryReport


class CrossScaleMemoryGuard:
    """Filtra qué información puede cruzar entre escalas."""

    ALLOWED_INVARIANTS = {
        "world_level",
        "viability_margin",
        "regime_category",
        "heterogeneity_score",
        "scale_success_history",
        "cost_history",
        "episode_signature",
        "prior_scale_verdict",
    }

    PROHIBITED_DIRECT_TRANSFER = {
        "cell_states",
        "local_indices",
        "neighbors",
        "dense_spatial_trajectory",
        "raw_cell_tensor",
        "raw_grid",
    }

    def sanitize_for_cross_scale(
        self,
        *,
        source_scale_id: str,
        target_scale_id: str,
        payload: Dict[str, Any],
    ) -> CrossScaleMemoryReport:
        blocked_fields: List[str] = []
        sanitized: Dict[str, Any] = {}

        for key, value in payload.items():
            if key in self.PROHIBITED_DIRECT_TRANSFER:
                blocked_fields.append(key)
                continue
            if key in self.ALLOWED_INVARIANTS:
                sanitized[key] = value
                continue

            # Permitir subconjunto conocido dentro de metadata.
            if key == "metadata" and isinstance(value, dict):
                subset = {
                    sub_key: sub_val
                    for sub_key, sub_val in value.items()
                    if sub_key in self.ALLOWED_INVARIANTS
                }
                if subset:
                    sanitized[key] = subset
                blocked_fields.extend(
                    [f"metadata.{sub_key}" for sub_key in value.keys() if sub_key not in subset]
                )
                continue

            blocked_fields.append(key)

        contamination_detected = len(blocked_fields) > 0
        total = max(len(payload), 1)
        contamination_rate = len(blocked_fields) / total

        sanitized["source_scale_id"] = source_scale_id
        sanitized["target_scale_id"] = target_scale_id

        return CrossScaleMemoryReport(
            contamination_detected=contamination_detected,
            blocked_fields_count=len(blocked_fields),
            allowed_fields_count=len(sanitized),
            cross_scale_memory_contamination_rate=contamination_rate,
            sanitized_payload=sanitized,
            blocked_fields=blocked_fields,
        )

    def compute_contamination_rate(self, reports: Iterable[CrossScaleMemoryReport]) -> float:
        collected = list(reports)
        if not collected:
            return 0.0
        contaminated = sum(1 for item in collected if item.contamination_detected)
        return contaminated / len(collected)

    def assert_transfer_safe(
        self,
        *,
        source_scale_id: str,
        target_scale_id: str,
        payload: Dict[str, Any],
    ) -> Tuple[bool, CrossScaleMemoryReport]:
        report = self.sanitize_for_cross_scale(
            source_scale_id=source_scale_id,
            target_scale_id=target_scale_id,
            payload=payload,
        )
        is_safe = not report.contamination_detected
        return is_safe, report
