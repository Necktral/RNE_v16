from runtime.reasoning.scheduler_meta.budgeting import compute_budget
from runtime.reasoning.scheduler_meta.context_features import extract_context_features
from runtime.reasoning.scheduler_meta.fallbacks import (
    confidence_from_step,
    cost_from_step,
    should_early_stop,
)
from runtime.reasoning.scheduler_meta.policy import (
    resolve_regime_labels,
    select_sequence,
    validate_reasoning_sequence,
)


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


def test_vram_favorable_inherits_floor_from_cognitive_regime():
    labels = resolve_regime_labels(
        regime_hint="vram_favorable",
        features={
            "vram_favorable_signal": 0.9,
            "heterogeneity_signal": 0.52,
            "world_level_signal": 0.70,
            "viability_edge_signal": 0.20,
        },
    )
    assert labels["primary_regime_label"] == "vram_favorable"
    assert labels["cognitive_regime_label"] == "heterogeneous_elevated"
    assert labels["floor_regime_label"] == "heterogeneous_elevated"


def test_validate_reasoning_sequence_corrige_patologia_v1_que_desplaza_ded():
    validation = validate_reasoning_sequence(
        proposed_sequence=["abd", "heur", "ana", "cau", "ctf", "prob"],
        allowed_families=["abd", "ana", "cau", "ctf", "ded", "prob", "heur"],
        requested_max_steps=6,
        primary_regime_label="heterogeneous_elevated",
        cognitive_regime_label="heterogeneous_elevated",
        floor_regime_label="heterogeneous_elevated",
        mandatory_family_floor=["abd", "ana", "cau", "ctf"],
        default_overlays=["heur"],
        admitted_overlays=["heur"],
        scores={"heur": 0.9, "abd": 1.0, "ana": 1.0, "cau": 1.0, "ctf": 1.0, "ded": 1.0, "prob": 1.0},
        allow_experimental=False,
        features={},
        enforce_overlay_anchors=True,
    )
    assert validation["proposed_passed"] is False
    assert validation["validated_passed"] is True
    assert validation["autocorrected"] is True
    assert validation["fallback_used"] is False
    assert validation["missing_core"] == ["DED"]
    assert validation["validated_sequence"] == ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]


def test_validate_reasoning_sequence_activa_fallback_si_allowed_families_rompen_cierre():
    validation = validate_reasoning_sequence(
        proposed_sequence=["abd", "heur", "ana", "cau", "ctf"],
        allowed_families=["abd", "ana", "cau", "ctf", "ded", "heur"],
        requested_max_steps=6,
        primary_regime_label="heterogeneous_warning",
        cognitive_regime_label="heterogeneous_warning",
        floor_regime_label="heterogeneous_warning",
        mandatory_family_floor=["abd", "ana", "cau", "ctf", "ded"],
        default_overlays=["heur", "fal_guard"],
        admitted_overlays=["heur"],
        scores={"heur": 0.9, "fal_guard": 0.8, "abd": 1.0, "ana": 1.0, "cau": 1.0, "ctf": 1.0, "ded": 1.0},
        allow_experimental=False,
        features={},
        enforce_overlay_anchors=True,
    )
    assert validation["proposed_passed"] is False
    assert validation["fallback_used"] is True
    assert validation["validated_passed"] is False
    assert validation["fallback_profile_name"] == "core_plus_guard"


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
