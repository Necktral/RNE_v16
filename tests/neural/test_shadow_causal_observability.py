import json
from pathlib import Path

from runtime.neural.observability import (
    SHADOW_OBSERVATION_SCHEMA_VERSION,
    build_shadow_observation,
    shadow_pair_id,
    validate_shadow_observation,
)


def _row(*, resource_pressure: float, candidate_hash: str | None) -> dict:
    organ = {
        "organ": "N1",
        "candidate_hash": candidate_hash,
        "candidate": {"status": "abstained"} if candidate_hash else None,
        "abstained": bool(candidate_hash),
        "fallback_reason": None,
        "cost": {"cpu_ms": 2.0},
    }
    return {
        "step_index": 1,
        "decision": {
            "action": "act",
            "external_input": 0.125,
            "mode": "normal",
            "priority": 1.0,
            "reason": "test",
            "scenario": "thermal_homeostasis",
        },
        "vital_signs": {
            "certified": True,
            "cognitive_quality": 0.75,
            "identity_continuity": 1.0,
            "mode": "normal",
            "risk_score": 0.1,
            "viability_margin": 0.8,
            "resource_pressure": resource_pressure,
        },
        "episode_result": {
            "episode": {
                "scenario": "thermal_homeostasis",
                "context": {"intervention": "activate_cooling"},
                "result": {
                    "alarm_transition": "stable",
                    "counterfactual_delta": 0.1,
                    "factual_delta": 0.2,
                    "intervention_effect": 0.3,
                    "reasoning_sequence": ["CAU"],
                    "relation_kind": "causal",
                    "updated_world": {"temperature": 0.5},
                },
            },
            "certification": {"verdict": "certified"},
            "neural_symbiosis_trace": {
                "trace_group_id": "trace-test",
                "resource_state": {"cpu_pressure": resource_pressure},
                "organs": [organ],
            },
        },
    }


def test_pair_id_is_deterministic_and_versioned() -> None:
    values = dict(seed=990001, step_index=1, scenario_id="thermal_homeostasis", external_input_hash="a" * 64)
    assert shadow_pair_id(**values) == shadow_pair_id(**values)
    assert len(shadow_pair_id(**values)) == 64


def test_shadow_candidate_differs_but_authoritative_behavior_is_identical() -> None:
    spans = [{"name": "life_step_total", "duration_ns": 100, "end_ns": 100}]
    off = build_shadow_observation(
        run_id="diagnostic-off", lane="off", seed=990001,
        row=_row(resource_pressure=0.1, candidate_hash=None), spans=spans,
    )
    shadow = build_shadow_observation(
        run_id="diagnostic-shadow", lane="shadow", seed=990001,
        row=_row(resource_pressure=0.8, candidate_hash="b" * 64), spans=spans,
    )
    assert off["pair_id"] == shadow["pair_id"]
    assert off["external_input_hash"] == shadow["external_input_hash"]
    assert off["candidate_output_hash"] != shadow["candidate_output_hash"]
    assert off["effective_output_hash"] == shadow["effective_output_hash"]
    assert off["authoritative_decision_hash"] == shadow["authoritative_decision_hash"]
    assert off["action_hash"] == shadow["action_hash"]
    assert off["outcome_hash"] == shadow["outcome_hash"]
    assert off["behavior_hash"] == shadow["behavior_hash"]
    assert off["resource_cost_hash"] != shadow["resource_cost_hash"]
    assert shadow["decision_influence"] == "none"
    assert shadow["candidate_applied_count"] == 0
    assert shadow["admission_status"] == "not_evaluated"
    assert shadow["admission_reason"] == "shadow_authority_mode"


def test_schema_rejects_unknown_version_and_secret_fields() -> None:
    schema_path = Path("runtime/neural/schemas/shadow_observation_v1.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    record = build_shadow_observation(
        run_id="diagnostic", lane="shadow", seed=990001,
        row=_row(resource_pressure=0.2, candidate_hash="c" * 64), spans=[],
    )
    assert record["schema_version"] == SHADOW_OBSERVATION_SCHEMA_VERSION
    assert schema["properties"]["schema_version"]["const"] == record["schema_version"]
    validate_shadow_observation(record)
    invalid = {**record, "schema_version": "neural.shadow_observation.v999"}
    try:
        validate_shadow_observation(invalid)
    except ValueError as exc:
        assert str(exc) == "shadow_observation_schema_unknown"
    else:
        raise AssertionError("unknown schema accepted")
    secret = {**record, "postgres_dsn": "postgresql://secret"}
    try:
        validate_shadow_observation(secret)
    except ValueError as exc:
        assert str(exc).startswith("shadow_observation_fields_unknown")
    else:
        raise AssertionError("secret field accepted")
