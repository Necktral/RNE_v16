"""Familia EML-SR experimental para descubrimiento simbólico en shadow mode.

OFF (byte-idéntico): dataset sintético (x ± 0.01) — señal pobre para el regresor.
DEEP (opt-in, RNFE_EML_SR_DEEP / RNFE_REASONING_DEEP): construye un **barrido
dosis-respuesta real** desde el modelo de efectos declarado de la firma causal
(y ≈ x + Δ), dando al regresor simbólico una ley genuina que descubrir en vez de
ruido. Sigue siendo shadow (eml_mode == 'shadow'). Determinista.
"""

from __future__ import annotations

from typing import Any, Dict, List

from runtime.symbolic.eml import EMLRunner
from runtime.reasoning.families import core_inference as ci

FAMILY_ID = "EML_SR"


def _effect_model_sweep(state: Dict[str, Any]) -> list[dict[str, Any]]:
    """Barrido determinista x→y usando el modelo de efectos (y = x + Δ_correctivo,
    cf = x + Δ_neutro). Ley lineal real y descubrible."""
    model = ci._effect_model(state)
    mv = ci.main_variable(state)
    obs = state.get("observation") if isinstance(state.get("observation"), dict) else {}
    x0 = obs.get(mv)
    if not model or not isinstance(x0, (int, float)):
        return []
    best_delta = min(model.values())      # intervención más reductora
    neutral_delta = max(model.values())   # intervención más neutra
    rows: list[dict[str, Any]] = []
    for k in range(6):
        x = min(1.0, max(0.0, float(x0) - 0.10 + k * 0.04))
        y = min(1.0, max(0.0, x + best_delta))
        cf = min(1.0, max(0.0, x + neutral_delta))
        rows.append({"x": round(x, 4), "cf": round(cf, 4), "y": round(y, 4)})
    return rows


def _dataset_from_state(state: Dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(state.get("eml_dataset"), list):
        rows = []
        for item in state["eml_dataset"]:
            if isinstance(item, dict):
                rows.append(item)
        return rows

    if ci.family_deep_enabled(FAMILY_ID):
        sweep = _effect_model_sweep(state)
        if sweep:
            return sweep

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
        "artifacts": result.get("artifacts", {}),
    }
