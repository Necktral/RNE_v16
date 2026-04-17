"""Reglas de promoción macro para MFM_lite."""

from __future__ import annotations

from statistics import mean, pstdev
from typing import Any, Dict, List

from runtime.storage import StorageFacade


class MacroPromotion:
    def __init__(self, *, storage: StorageFacade):
        self.storage = storage

    def should_promote(
        self,
        *,
        run_id: str,
        pattern_key: str,
        continuity_alert: bool,
    ) -> Dict[str, Any]:
        if continuity_alert:
            return {"promote": False, "reason": "continuity_alert"}
        meso = self.storage.retrieve_memory_records(
            run_id=run_id, scales=["meso"], limit=500
        )
        pattern_records: List[Any] = [
            item
            for item in meso
            if item.structure_json.get("pattern_key") == pattern_key
            and item.ioc_proxy is not None
        ]
        if len(pattern_records) < 3:
            return {"promote": False, "reason": "insufficient_support", "support": len(pattern_records)}
        ioc_values = [float(item.ioc_proxy) for item in pattern_records]
        mean_ioc = mean(ioc_values)
        std_ioc = pstdev(ioc_values) if len(ioc_values) > 1 else 0.0
        promote = bool(mean_ioc >= 0.72 and std_ioc <= 0.08)
        return {
            "promote": promote,
            "reason": "eligible" if promote else "threshold_not_met",
            "support": len(pattern_records),
            "mean_ioc": mean_ioc,
            "std_ioc": std_ioc,
            "records": pattern_records,
        }
