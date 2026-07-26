from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from runtime.symbolic.mci import (
    CausalOverlay,
    MCIRuntime,
    OverlayTranslator,
    resource_with_energy_spec,
    thermal_battery_spec,
    thermal_battery_to_resource_energy,
)
from runtime.symbolic.mci.provider_adapter import adapt_transferred_hypothesis
from runtime.symbolic.mci.causal_learning import TransitionEvidence
from runtime.symbolic.mci.compiler import TransitionCompiler


def _source_overlay() -> CausalOverlay:
    source = thermal_battery_spec()
    return CausalOverlay(
        overlay_id="source-overlay",
        version=1,
        base_spec_sha256=source.sha256,
        parameter_updates={"cooling_effect": 0.07},
        precondition_updates={
            "thermal_battery/cooling": "battery_level > 0.3"
        },
        edge_updates=(),
        evidence={"source": "autonomous"},
        train_mae_before=0.1,
        train_mae_after=0.0,
        holdout_mae_before=0.1,
        holdout_mae_after=0.0,
    )


def test_transfer_package_translation_and_adapter_are_typed() -> None:
    source, target = thermal_battery_spec(), resource_with_energy_spec()
    mapping = thermal_battery_to_resource_energy(source, target)
    translator = OverlayTranslator()
    package = translator.build_package(
        _source_overlay(), mapping, evidence_refs=("source/e1",)
    )
    hypotheses = translator.translate_overlay(
        _source_overlay(), mapping, package, logical_time=9
    )
    condition = next(item for item in hypotheses if item.kind == "precondition")
    parameter = next(item for item in hypotheses if item.kind == "parameter")
    assert condition.target_id == "resource_energy/production"
    assert condition.expression == "energy_level > 0.3"
    assert parameter.proposed_value == pytest.approx(0.08)
    adapted = adapt_transferred_hypothesis(condition)
    assert adapted.provider == "structural-transfer"
    assert adapted.model_ref == mapping.morphism_id
    assert adapted.evidence_refs == ()
    assert adapted.submitted_evidence_refs == ("source/e1",)
    with pytest.raises(FrozenInstanceError):
        package.confidence = 0.1  # type: ignore[misc]
    assert package.sha256 == package.sha256


def test_transferred_belief_is_revisable_until_empirical_evaluation() -> None:
    source, target = thermal_battery_spec(), resource_with_energy_spec()
    mapping = thermal_battery_to_resource_energy(source, target)
    translator = OverlayTranslator()
    overlay = _source_overlay()
    package = translator.build_package(overlay, mapping)
    hypotheses = translator.translate_overlay(
        overlay, mapping, package, logical_time=1
    )
    runtime = MCIRuntime(target)
    runtime.ingest_transferred_hypotheses(hypotheses)
    snapshot = runtime.transfer_ledger.snapshot()
    assert snapshot
    assert {item["status"] for item in snapshot} == {"REVISABLE"}
    assert {item["belief_origin"] for item in snapshot} == {
        "structural-transfer"
    }


def test_package_rejects_wrong_overlay() -> None:
    source, target = thermal_battery_spec(), resource_with_energy_spec()
    mapping = thermal_battery_to_resource_energy(source, target)
    translator = OverlayTranslator()
    overlay = _source_overlay()
    package = translator.build_package(overlay, mapping)
    other = CausalOverlay(
        **{**overlay.to_dict(), "overlay_id": "other"}  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="overlay_mismatch"):
        translator.translate_overlay(other, mapping, package, logical_time=1)


def test_transferred_hypothesis_promotes_only_after_destination_validation() -> None:
    source = thermal_battery_spec()
    target = resource_with_energy_spec()
    from runtime.symbolic.mci import resource_with_energy_oracle_spec

    actual = resource_with_energy_oracle_spec()
    mapping = thermal_battery_to_resource_energy(source, target)
    translator = OverlayTranslator()
    source_overlay = _source_overlay()
    package = translator.build_package(source_overlay, mapping)
    transferred = [
        item
        for item in translator.translate_overlay(
            source_overlay, mapping, package, logical_time=23
        )
        if item.kind == "precondition"
    ]
    runtime = MCIRuntime(target)
    for episode in range(11):
        energy = 0.1 if episode % 2 == 0 else 0.8
        state = {
            "stock_level": 0.4,
            "energy_level": energy,
            "production_active": False,
            "scarcity_alert": False,
        }
        runtime.learner.observe(
            TransitionEvidence(
                state=state,
                action="start_production",
                external_input=0.02,
                observed=TransitionCompiler(actual).execute(
                    state, action="start_production", external_input=0.02
                ),
                predicted=TransitionCompiler(target).execute(
                    state, action="start_production", external_input=0.02
                ),
                replay_unit_id=f"target/{episode}",
                logical_time=episode * 2 + 2,
            )
        )
    runtime.ingest_transferred_hypotheses(transferred)
    state = {
        "stock_level": 0.4,
        "energy_level": 0.8,
        "production_active": False,
        "scarcity_alert": False,
    }
    promoted = runtime.learner.observe(
        TransitionEvidence(
            state=state,
            action="start_production",
            external_input=0.02,
            observed=TransitionCompiler(actual).execute(
                state, action="start_production", external_input=0.02
            ),
            predicted=TransitionCompiler(target).execute(
                state, action="start_production", external_input=0.02
            ),
            replay_unit_id="target/11",
            logical_time=24,
        )
    )
    evaluations = runtime.learner.last_neural_evaluations()
    assert promoted is not None
    assert promoted.evidence["source"] == "transferred"
    assert evaluations[0].status == "promoted"
