"""Familia EML-SR experimental para descubrimiento simbólico en shadow mode."""

from __future__ import annotations

from typing import Any, Dict, List

from runtime.symbolic.eml import EMLRunner

FAMILY_ID = "EML_SR"


def _dataset_from_state(state: Dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(state.get("eml_dataset"), list):
        rows = []
        for item in state["eml_dataset"]:
            if isinstance(item, dict):
                rows.append(item)
        return rows

    observation = state.get("observation")
    counterfactual = state.get("counterfactual")
    factual = state.get("updated_world")
    if isinstance(observation, dict):
        x = float(observation.get("temperature", 0.0))
        y = float((factual or {}).get("temperature", x))
        cf = float((counterfactual or {}).get("temperature", y))
        return [
            {"x": x, "cf": cf, "y": y},
            {"x": x + 0.01, "cf": cf, "y": y},
            {"x": x - 0.01, "cf": cf, "y": y},
        ]
    return [{"x": 0.0, "cf": 0.0, "y": 0.0}]


def execute(state: Dict[str, Any]) -> Dict[str, Any]:
    mode = state.get("eml_mode", "disabled")
    run_id = state.get("run_id", "no-run")
    episode_id = state.get("episode_id", "no-episode")
    if mode != "shadow":
        return {
            "family": FAMILY_ID,
            "status": "idle",
            "state_delta": {"eml_shadow_active": False},
            "confidence": 0.0,
            "cost": 0.0,
            "candidate_count": 0,
            "recommended_next_family": "PROB",
        }

    dataset = _dataset_from_state(state)
    runner = EMLRunner(storage=state.get("storage"))
    result = runner.run_shadow(run_id=str(run_id), episode_id=str(episode_id), rows=dataset)
    top: List[dict[str, Any]] = result["run"].get("top_candidates", [])
    confidence = float(top[0]["composite_score"]) if top else 0.0
    return {
        "family": FAMILY_ID,
        "status": "ok",
        "state_delta": {
            "eml_shadow_active": True,
            "eml_top_candidates": top,
            "eml_run_id": result["run"]["eml_run_id"],
        },
        "confidence": confidence,
        "cost": min(5.0, 0.5 + (0.01 * float(result["run"]["candidate_count"]))),
        "candidate_count": result["run"]["candidate_count"],
        "recommended_next_family": "PROB",
    }

