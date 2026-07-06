from runtime.reasoning.scheduler_meta.budgeting import compute_budget
from runtime.reasoning.scheduler_meta.context_features import extract_context_features
from runtime.reasoning.scheduler_meta.degradation import build_degradation_plan
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


def test_context_features_extracts_hardware_and_gpu_opportunity():
    features = extract_context_features(
        {
            "cpu_load": 0.30,
            "memory_load": 0.40,
            "gpu_available": True,
            "gpu_load": 0.10,
            "vram_snapshot": {
                "vram_headroom": 0.82,
                "vram_pressure": 0.18,
                "vram_opportunity_score": 0.86,
            },
        }
    )
    assert features["hardware_pressure_signal"] == 0.40
    assert features["gpu_acceleration_signal"] >= 0.70
    assert features["vram_favorable_signal"] >= 0.70


def test_context_features_lift_risk_from_attestations():
    features = extract_context_features(
        {
            "uncertainty": 0.1,
            "contradiction_signal": 0.0,
            "counterfactual_gap": 0.05,
            "causal_attestation": {
                "validation_status": "fail",
                "degradation_level": "relation_mismatch",
            },
            "memory_rag_attestation": {
                "validation_status": "warn",
                "retrieval_purity": 0.60,
            },
        }
    )
    assert features["causal_risk"] >= 0.85
    assert features["contradiction_signal"] >= 0.75
    assert features["fragility_risk_signal"] >= 0.75


def test_degradation_plan_combines_causal_memory_hardware_and_autonomy():
    plan = build_degradation_plan(
        features={
            "hardware_pressure_signal": 0.88,
            "gpu_acceleration_signal": 0.20,
        },
        sequence_validation={"validated_passed": True},
        causal_attestation={"validation_status": "fail"},
        memory_attestation={"validation_status": "warn", "retrieval_purity": 0.62},
        autonomy_policy={
            "requested_mode": "unlimited",
            "active_mode": "bounded",
            "policy_authorized": False,
        },
    )
    assert plan["schema"] == "degradation_plan.v1"
    assert plan["level"] == "causal_recovery"
    assert plan["severity"] >= 0.90
    assert "block_actuation_until_causal_revalidated" in plan["actions"]
    assert "degrade_autonomy_scope" in plan["actions"]
    assert plan["budget_multiplier"] <= 0.50


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


def test_budgeting_uses_gpu_margin_and_hardware_pressure():
    gpu_budget = compute_budget(
        {
            "uncertainty": 0.2,
            "contradiction_signal": 0.0,
            "causal_risk": 0.1,
            "edge_pressure": 0.1,
            "hardware_pressure_signal": 0.2,
            "gpu_acceleration_signal": 0.85,
        }
    )
    constrained_budget = compute_budget(
        {
            "uncertainty": 0.2,
            "contradiction_signal": 0.0,
            "causal_risk": 0.1,
            "edge_pressure": 0.1,
            "hardware_pressure_signal": 0.90,
            "gpu_acceleration_signal": 0.85,
        }
    )
    assert gpu_budget["max_steps"] == 7.0
    assert constrained_budget["max_steps"] == 4.0


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


def test_policy_uses_gpu_opportunity_for_shadow_induction():
    features = {
        "uncertainty": 0.3,
        "contradiction_signal": 0.1,
        "continuity_recent": 0.8,
        "edge_pressure": 0.1,
        "causal_risk": 0.2,
        "pattern_without_structure_signal": 0.55,
        "gpu_acceleration_signal": 0.88,
        "hardware_pressure_signal": 0.20,
    }
    sequence, _, _, meta = select_sequence(
        features=features,
        budget={"max_steps": 7.0},
        mode="adaptive",
        return_metadata=True,
    )
    assert "ind" in sequence
    assert "IND" in meta["validated_sequence"]


def test_policy_degrades_gpu_shadow_when_hardware_is_constrained():
    features = {
        "uncertainty": 0.3,
        "contradiction_signal": 0.1,
        "continuity_recent": 0.8,
        "edge_pressure": 0.1,
        "causal_risk": 0.2,
        "pattern_without_structure_signal": 0.55,
        "gpu_acceleration_signal": 0.88,
        "hardware_pressure_signal": 0.90,
    }
    sequence, _, _, meta = select_sequence(
        features=features,
        budget={"max_steps": 7.0},
        mode="adaptive",
        return_metadata=True,
    )
    assert "ind" not in sequence
    assert "IND" not in meta["validated_sequence"]


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
