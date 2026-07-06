"""Tests del razonador externo admitido en runtime (Bloque C)."""

from __future__ import annotations

import pytest

from runtime.reasoning.scheduler_meta.policy import select_sequence
from runtime.reasoning.scheduler_meta.family_profiles import PROFILES, lab_only_profiles
from runtime.reasoning.external_models.guard import guard_external_choice, GuardDecision


def _select(profile: str, *, regime_hint: str | None = None) -> list[str]:
    seq, _, _ = select_sequence(
        features={"contradiction_signal": 0.9, "causal_risk": 0.9},
        budget={"max_steps": 8},
        mode="fixed",
        profile_name=profile,
        regime_hint=regime_hint,
        allow_experimental=True,
    )
    return list(seq)


def test_gated_profile_registered_and_lab_only():
    assert "core_plus_external_reasoner_gated_v1" in PROFILES
    assert "core_plus_external_reasoner_gated_v1" in lab_only_profiles()


def test_flag_off_still_raises_for_gated_profile(monkeypatch):
    monkeypatch.delenv("RNFE_EXTERNAL_REASONER_RUNTIME", raising=False)
    with pytest.raises(ValueError, match="external_reasoner_profile_not_nominal"):
        _select("core_plus_external_reasoner_gated_v1", regime_hint="causal_counterfactual_conflict")


def test_flag_on_validated_regime_schedules_external(monkeypatch):
    monkeypatch.setenv("RNFE_EXTERNAL_REASONER_RUNTIME", "1")
    seq = _select("core_plus_external_reasoner_gated_v1", regime_hint="causal_counterfactual_conflict")
    assert "ext_open_thinker" in seq
    # El núcleo canónico se preserva y PROB cierra.
    assert seq[-1] == "prob"


def test_flag_on_wrong_regime_degrades_without_crash(monkeypatch):
    monkeypatch.setenv("RNFE_EXTERNAL_REASONER_RUNTIME", "1")
    # Régimen no validado -> degrada (no agenda ext) y NO lanza excepción.
    seq = _select("core_plus_external_reasoner_gated_v1", regime_hint="homogeneous_safe")
    assert "ext_open_thinker" not in seq
    assert seq[-1] == "prob"


def test_flag_on_non_admitted_external_profile_still_raises(monkeypatch):
    monkeypatch.setenv("RNFE_EXTERNAL_REASONER_RUNTIME", "1")
    # Perfil externo legacy NO admitido -> rechazo duro aún con el flag on.
    with pytest.raises(ValueError, match="external_reasoner_profile_not_nominal"):
        _select("core_plus_external_reasoner", regime_hint="causal_counterfactual_conflict")


def test_runtime_guard_rejects_regressions():
    core = {"viability_margin": 0.5, "intervention_precision": 0.8, "closure_stable": True}
    worse = {"viability_margin": 0.3, "intervention_precision": 0.8, "closure_stable": True}
    d = guard_external_choice(
        allowed_interventions=["activate_cooling", "deactivate_cooling"],
        core_intervention="activate_cooling",
        recommended_intervention="deactivate_cooling",
        core_metrics=core,
        candidate_metrics=worse,
    )
    assert isinstance(d, GuardDecision)
    assert d.accepted is False
    assert d.reason == "viability_regression"
