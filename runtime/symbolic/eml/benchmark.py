"""Benchmark comparativo baseline vs shadow para EML."""

from __future__ import annotations

from typing import Any, Dict, List

from .advisory import evaluate_advisory_promotion


def compare_baseline_vs_shadow(
    *,
    baseline_result: Dict[str, Any],
    shadow_result: Dict[str, Any],
) -> Dict[str, Any]:
    baseline = baseline_result["bench_run"]
    shadow = shadow_result["bench_run"]
    no_regression = bool(
        shadow["closure_rate"] >= baseline["closure_rate"]
        and shadow["continuity_mean"] >= baseline["continuity_mean"]
        and shadow["collapse_count"] <= baseline["collapse_count"]
    )
    return {
        "no_regression": no_regression,
        "baseline": baseline,
        "shadow": shadow,
    }


def collect_eml_candidates_from_events(storage, *, run_id: str) -> List[Dict[str, Any]]:
    events = storage.list_events(run_id=run_id, limit=2000)
    out: List[Dict[str, Any]] = []
    for item in events:
        if item.event_type != "eml.candidate.generated":
            continue
        payload = item.payload or {}
        candidate = payload.get("candidate")
        if isinstance(candidate, dict):
            out.append(candidate)
    return out


def advisory_from_run(storage, *, run_id: str, continuity_alert_threshold: float = 0.35):
    assessments = storage.list_reality_assessments(run_id=run_id, limit=500)
    continuity_alert = any(a.continuity_score < continuity_alert_threshold for a in assessments)
    candidates = collect_eml_candidates_from_events(storage, run_id=run_id)
    return evaluate_advisory_promotion(
        candidates=candidates,
        continuity_alert=continuity_alert,
    )
