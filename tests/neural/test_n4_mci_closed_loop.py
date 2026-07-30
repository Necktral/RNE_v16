from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from runtime.neural import (
    NeuralInferenceRequest,
    NeuralMode,
    NeuralModelManifest,
    NeuralRuntime,
)
from runtime.neural.calibration import N4ProviderCalibration
from runtime.neural.hypothesis_generator import StructuralHypothesisGenerator
from runtime.neural.integration import MCIToN4GraphBuilder
from runtime.neural.organs import N4CausalRankingBackend
from runtime.neural.training import (
    N4TrainingSample,
    score_n4_validity,
    train_n4_ranking,
)
from runtime.symbolic.mci.causal_learning import TransitionEvidence
from runtime.symbolic.mci.compiler import TransitionCompiler
from runtime.symbolic.mci.contracts import NeuralHypothesisEvaluation
from runtime.symbolic.mci import NeuralHypothesis
from runtime.symbolic.mci.hypothesis_ledger import HypothesisLedger
from runtime.symbolic.mci.specs import thermal_battery_spec, with_overlay


def _manifest(*, trained: bool = False) -> NeuralModelManifest:
    return NeuralModelManifest(
        organ="N4",
        capability="causal_hypothesis_ranking",
        model_id="test-n4",
        version="1",
        backend="python",
        artifact_path="test/n4.json",
        artifact_sha256=hashlib.sha256(b"n4").hexdigest(),
        trained=trained,
        training_provenance=({"dataset": "heldout"} if trained else {}),
    )


def _evidence(count: int = 16):
    base = thermal_battery_spec()
    actual = with_overlay(
        base, {}, {"thermal_battery/cooling": "battery_level > 0.3"}
    )
    rows = []
    for index in range(count):
        battery = 0.1 if index % 2 == 0 else 0.8
        state = {
            "temperature": 0.9,
            "battery_level": battery,
            "cooling_active": False,
            "alarm": True,
        }
        rows.append(
            TransitionEvidence(
                state=state,
                action="activate_cooling",
                external_input=0.0,
                predicted=TransitionCompiler(base).execute(
                    state, action="activate_cooling", external_input=0.0
                ),
                observed=TransitionCompiler(actual).execute(
                    state, action="activate_cooling", external_input=0.0
                ),
                logical_time=2 * index + 2,
            )
        )
    return base, tuple(rows)


def test_graph_and_candidate_generation_are_deterministic():
    spec, rows = _evidence()
    builder = MCIToN4GraphBuilder()
    kwargs = {
        "spec": spec,
        "evidence": rows,
        "state": rows[-1].state,
        "candidate_action": "activate_cooling",
        "logical_time": 35,
    }
    first = builder.build(**kwargs)
    second = builder.build(**kwargs)
    assert first.graph_sha256 == second.graph_sha256
    generator = StructuralHypothesisGenerator(beam_width=64)
    candidates = generator.generate(spec=spec, evidence=rows)
    assert candidates == generator.generate(spec=spec, evidence=rows)
    assert any(
        item.kind == "precondition"
        and item.target_id == "thermal_battery/cooling"
        and "battery_level" in str(item.expression)
        for item in candidates
    )
    with pytest.raises(ValueError, match="future_evidence"):
        builder.build(**{**kwargs, "logical_time": rows[-1].logical_time})


def test_hybrid_generator_marks_change_boundary_source():
    spec, _ = _evidence()
    actual = with_overlay(
        spec, {}, {"thermal_battery/cooling": "battery_level > 0.3"}
    )
    rows = []
    for index, battery in enumerate(
        (0.28, 0.32, 0.29, 0.31, 0.295, 0.305, 0.299, 0.301)
    ):
        state = {
            "temperature": 0.9,
            "battery_level": battery,
            "cooling_active": False,
            "alarm": True,
        }
        rows.append(
            TransitionEvidence(
                state=state,
                action="activate_cooling",
                external_input=0.0,
                predicted=TransitionCompiler(spec).execute(
                    state, action="activate_cooling", external_input=0.0
                ),
                observed=TransitionCompiler(actual).execute(
                    state, action="activate_cooling", external_input=0.0
                ),
                logical_time=index + 1,
            )
        )
    candidates = StructuralHypothesisGenerator(beam_width=64).generate(
        spec=spec, evidence=rows
    )
    assert any(
        item.expression == "battery_level > 0.3"
        and item.source == "change_boundary"
        for item in candidates
    )


def test_runtime_is_fail_open_and_resource_guarded():
    class Broken:
        def infer(self, request):
            raise OSError("unavailable")

    runtime = NeuralRuntime(
        backend=Broken(), manifest=_manifest(), mode=NeuralMode.EXPERIMENTAL
    )
    result = runtime.infer(
        NeuralInferenceRequest(
            inference_id="i-1",
            run_id="r-1",
            payload={},
            logical_time=1,
        )
    )
    assert result.fallback_used
    assert result.fallback_reason == "backend_failure:OSError"


def test_ranking_is_deterministic_and_proposal_only():
    backend = N4CausalRankingBackend()
    request = NeuralInferenceRequest(
        inference_id="i",
        run_id="r",
        logical_time=1,
        payload={
            "candidates": [
                {
                    "hypothesis_id": "weak",
                    "empirical_gain": 0.1,
                    "holdout_support": 0.1,
                    "coverage": 1.0,
                    "simplicity": 1.0,
                    "stability": 1.0,
                    "invariant_safety": 1.0,
                },
                {
                    "hypothesis_id": "strong",
                    "empirical_gain": 0.9,
                    "holdout_support": 0.8,
                    "coverage": 1.0,
                    "simplicity": 1.0,
                    "stability": 1.0,
                    "invariant_safety": 1.0,
                },
            ]
        },
    )
    output = backend.infer(request)
    assert output.candidate_output["rankings"][0]["hypothesis_id"] == "strong"
    assert "action" not in output.candidate_output


def test_calibration_updates_only_final_evaluations():
    calibration = N4ProviderCalibration()
    calibration.register("h", "thermal|precondition|cooling")
    pending = NeuralHypothesisEvaluation(
        "h", "pending", "insufficient_evidence", None, None, None, None, None
    )
    assert calibration.update((pending,)) == ()
    promoted = NeuralHypothesisEvaluation(
        "h", "promoted", "empirical_gates_passed", 1.0, 0.0, 1.0, 0.0, "o"
    )
    report = calibration.update((promoted,))[0]
    assert report.alpha == 2.0
    assert report.posterior_success > 0.5


def test_trained_manifest_requires_provenance():
    with pytest.raises(ValueError, match="requires_provenance"):
        NeuralModelManifest(
            organ="N4",
            capability="rank",
            model_id="bad",
            version="1",
            backend="python",
            artifact_path="n4.json",
            artifact_sha256="0" * 64,
            trained=True,
        )


def test_offline_training_exports_deterministic_runtime_artifact(tmp_path: Path):
    samples = tuple(
        N4TrainingSample(
            features=(
                float(index % 2),
                float(index % 2),
                1.0,
                1.0,
                0.9,
                1.0,
            ),
            valid=bool(index % 2),
            mae_gain=float(index % 2),
            invariant_risk=0.0,
            scenario=f"scenario-{index % 3}",
            seed=index % 3,
            logical_time=index,
        )
        for index in range(60)
    )
    artifact = train_n4_ranking(
        samples,
        artifact_path=tmp_path / "n4.json",
        epochs=12,
        patience=4,
    )
    assert artifact["schema"] == "n4-ranking-artifact.v1"
    assert len(artifact["ranking_weights"]) == 6
    assert (tmp_path / "n4.json").read_bytes().endswith(b"\n")
    features = samples[0].features
    expected = score_n4_validity(features, artifact)
    backend = N4CausalRankingBackend(artifact)
    output = backend.infer(
        NeuralInferenceRequest(
            inference_id="parity",
            run_id="parity",
            logical_time=1,
            payload={
                "candidates": [
                    {
                        "hypothesis_id": "candidate",
                        **dict(zip(artifact["feature_names"], features)),
                    }
                ]
            },
        )
    )
    observed = output.candidate_output["rankings"][0]["probability_valid"]
    assert abs(expected - observed) < 1e-8


def test_hypothesis_ledger_is_idempotent_and_rejects_collision():
    ledger = HypothesisLedger()
    hypothesis = NeuralHypothesis(
        hypothesis_id="h-1",
        kind="parameter",
        target_id="cooling_effect",
        expression=None,
        proposed_value=0.14,
        confidence=0.8,
        evidence_refs=(),
        provider="n4",
        model_ref="n4-v1",
        logical_time=1,
    )
    assert ledger.submit(hypothesis)
    assert ledger.submit(hypothesis)
    ledger.apply(
        NeuralHypothesisEvaluation(
            "h-1", "rejected", "empirical_gates_failed", 1.0, 1.0, 1.0, 1.0, None
        )
    )
    assert not ledger.submit(hypothesis)
    collision = NeuralHypothesis(
        **{**hypothesis.to_dict(), "proposed_value": 0.2}
    )
    with pytest.raises(ValueError, match="Colisión"):
        ledger.submit(collision)
