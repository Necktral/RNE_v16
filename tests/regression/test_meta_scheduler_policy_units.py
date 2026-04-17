from runtime.reasoning.scheduler_meta.budgeting import compute_budget
from runtime.reasoning.scheduler_meta.context_features import extract_context_features
from runtime.reasoning.scheduler_meta.fallbacks import (
    confidence_from_step,
    cost_from_step,
    should_early_stop,
)
from runtime.reasoning.scheduler_meta.policy import select_sequence


def test_context_features_extracts_bounded_values():
    features = extract_context_features(
        {
            "uncertainty": 1.2,
            "contradiction_signal": -1.0,
            "continuity_recent": 2.0,
            "edge_pressure": 0.8,
            "counterfactual_gap": -0.9,
        }
    )
    assert 0.0 <= features["uncertainty"] <= 1.0
    assert 0.0 <= features["contradiction_signal"] <= 1.0
    assert 0.0 <= features["continuity_recent"] <= 1.0
    assert features["causal_risk"] == 0.9


def test_budgeting_respects_limits():
    budget = compute_budget(
        {
            "uncertainty": 0.9,
            "contradiction_signal": 0.8,
            "continuity_recent": 0.3,
            "edge_pressure": 0.1,
            "causal_risk": 0.9,
        }
    )
    assert 4 <= budget["max_steps"] <= 10
    assert 0.0 <= budget["risk_budget"] <= 1.0


def test_policy_selects_adversarial_guard_when_contradiction_is_high():
    features = {
        "uncertainty": 0.7,
        "contradiction_signal": 0.8,
        "continuity_recent": 0.8,
        "edge_pressure": 0.2,
        "causal_risk": 0.6,
    }
    sequence, _, _ = select_sequence(features=features, budget={"max_steps": 8.0})
    assert "dia_adv" in sequence
    assert "fal_guard" in sequence


def test_fallback_primitives_are_deterministic():
    features = {
        "uncertainty": 0.4,
        "contradiction_signal": 0.2,
        "continuity_recent": 0.9,
        "edge_pressure": 0.1,
        "causal_risk": 0.3,
    }
    result = {"status": "ok", "confidence": 0.8, "cost": 0.7}
    assert confidence_from_step(result, features=features) == 0.8
    assert cost_from_step(result) == 0.7
    assert (
        should_early_stop(
            step_result=result,
            state={},
            features=features,
            step_index=0,
            max_steps=6,
        )
        is False
    )
