from __future__ import annotations

from pathlib import Path

import pytest

from runtime.reasoning.scheduler_meta.family_profiles import (
    EXT_OPEN_THINKER_ADMISSION,
    family_admission_record,
    lab_only_profiles,
    nominal_profiles,
    profile_uses_external_reasoner,
    validate_external_reasoner_admission,
)
from runtime.reasoning.scheduler_meta.policy import select_sequence


def _select(profile: str) -> list[str]:
    sequence, _, _ = select_sequence(
        features={},
        budget={"max_steps": 6},
        mode="fixed",
        profile_name=profile,
        allow_experimental=False,
    )
    return [family.upper() for family in sequence]


def test_core_only_does_not_activate_external_reasoner() -> None:
    assert not profile_uses_external_reasoner("core_only")
    assert "EXT_OPEN_THINKER" not in _select("core_only")


def test_ext_open_thinker_not_in_nominal_profiles() -> None:
    for profile in nominal_profiles().values():
        assert "ext_open_thinker" not in profile.optional_families


def test_legacy_scheduler_external_profiles_are_lab_only_and_blocked() -> None:
    lab_profiles = lab_only_profiles()
    assert "core_plus_external_reasoner" in lab_profiles
    assert "core_plus_external_reasoner_guarded" in lab_profiles
    # Los perfiles lab-only que USAN el razonador externo deben estar bloqueados en
    # nominal. (Otros perfiles lab-only existen por razones no externas: p.ej.
    # full_family_deep_v1, core_plus_imagination — no aplican a esta garantía.)
    external_lab_profiles = [
        name for name in lab_profiles if profile_uses_external_reasoner(name)
    ]
    assert external_lab_profiles  # al menos los legacy externos
    for profile_name in external_lab_profiles:
        with pytest.raises(ValueError, match="external_reasoner_profile_not_nominal"):
            _select(profile_name)


def test_admission_record_reflects_conditional_lab_status() -> None:
    record = family_admission_record("ext_open_thinker")
    assert record == EXT_OPEN_THINKER_ADMISSION
    assert record is not None
    assert record.stratum == "external_experimental"
    assert record.nominal_status == "conditional_lab"
    assert record.allowed_in_nominal_runtime is False
    assert record.validated_regimes == ["causal_counterfactual_conflict"]
    assert record.forbidden_default_regimes == [
        "viability_edge",
        "heterogeneous_warning",
        "homogeneous_safe",
    ]
    assert record.activation_policy == "ExternalReasonerGate v1"
    assert record.guard_required is True
    assert record.schema_required is True
    assert record.fallback_required is True
    assert record.evidence_status == "conflict_resolver_repetible"


def test_admission_record_references_existing_evidence_artifacts() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    for artifact_path in EXT_OPEN_THINKER_ADMISSION.evidence_artifacts.values():
        assert (repo_root / artifact_path).exists()
    assert (
        repo_root
        / "data/benchmarks/external_reasoner_gain/conflict-repeatability-gated-v1-4x4/evidence_manifest.json"
    ).exists()


def test_admission_accepts_only_gated_profile_in_validated_regime() -> None:
    decision = validate_external_reasoner_admission(
        profile_name="core_plus_external_reasoner_gated_v1",
        regime="causal_counterfactual_conflict",
        gate_present=True,
        guard_present=True,
        schema_present=True,
        fallback_present=True,
    )
    assert decision.allowed is True
    assert decision.reason == "external_reasoner_admitted_conditional_lab"


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"gate_present": False}, "external_reasoner_gate_required"),
        ({"guard_present": False}, "external_reasoner_guard_required"),
        ({"schema_present": False}, "external_reasoner_schema_required"),
        ({"fallback_present": False}, "external_reasoner_fallback_required"),
    ],
)
def test_admission_requires_gate_guard_schema_and_fallback(kwargs: dict, reason: str) -> None:
    params = {
        "profile_name": "core_plus_external_reasoner_gated_v1",
        "regime": "causal_counterfactual_conflict",
        "gate_present": True,
        "guard_present": True,
        "schema_present": True,
        "fallback_present": True,
    }
    params.update(kwargs)
    decision = validate_external_reasoner_admission(**params)
    assert decision.allowed is False
    assert decision.reason == reason


def test_admission_rejects_unvalidated_regime() -> None:
    decision = validate_external_reasoner_admission(
        profile_name="core_plus_external_reasoner_gated_v1",
        regime="viability_edge",
        gate_present=True,
        guard_present=True,
        schema_present=True,
        fallback_present=True,
    )
    assert decision.allowed is False
    assert decision.reason == "external_reasoner_regime_not_validated"


def test_admission_rejects_general_external_profile() -> None:
    decision = validate_external_reasoner_admission(
        profile_name="core_plus_external_reasoner",
        regime="causal_counterfactual_conflict",
        gate_present=True,
        guard_present=True,
        schema_present=True,
        fallback_present=True,
    )
    assert decision.allowed is False
    assert decision.reason == "external_reasoner_profile_not_admitted"
