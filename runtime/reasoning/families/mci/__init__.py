"""Familia MCI: planificación SMT, auto-modelo y revisión causal integrada."""

from __future__ import annotations

from typing import Any

from z3 import Z3Exception

from runtime.symbolic.mci.runtime import MCIRuntime

FAMILY_ID = "MCI"


def execute(state: dict[str, Any]) -> dict[str, Any]:
    budget = ((state.get("_meta") or {}).get("budget") or {})
    if float(budget.get("cost_budget", 1.5)) < 1.5:
        return {
            "family": FAMILY_ID,
            "status": "skip",
            "state_delta": {},
            "confidence": 0.0,
            "cost": 0.0,
            "failure_mode": "msrc_budget_unavailable",
        }
    runtime = state.get("_mci_runtime")
    if not isinstance(runtime, MCIRuntime):
        return {
            "family": FAMILY_ID,
            "status": "skip",
            "state_delta": {},
            "confidence": 0.0,
            "cost": 0.0,
            "failure_mode": "mci_runtime_unavailable",
        }
    observation = state.get("observation") or {}
    world_state = {
        key: value
        for key, value in observation.items()
        if key not in {"propositions", "alarm", "level"}
    }
    try:
        result = runtime.propose(
            state=world_state,
            external_input=float(state.get("_mci_external_input", 0.0)),
            logical_time=int(state.get("_preaction_logical_time", 0)),
            regime=str(state.get("regime_hint") or state.get("scenario") or "unknown"),
            replay_unit_id=str(state.get("_replay_unit_id") or ""),
        )
    except (KeyError, RuntimeError, TypeError, ValueError, Z3Exception):
        return {
            "family": FAMILY_ID,
            "status": "skip",
            "state_delta": {},
            "confidence": 0.0,
            "cost": 0.0,
            "failure_mode": "mci_degraded_to_certified_baseline",
        }
    return {
        "family": FAMILY_ID,
        "status": "ok",
        "state_delta": result,
        "confidence": float(result["mci_self_model"]["success_probability"]),
        "cost": 1.5,
    }
