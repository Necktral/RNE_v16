from __future__ import annotations

import copy
import hashlib
import json

import pytest

from runtime.neural.integration.p1_n4 import (
    FEATURE_NAMES,
    N4PreactionArtifactV2,
    N4PreactionInterventionSet,
    PREACTION_SCHEMA_VERSION,
    causal_signature_prior_evidence,
    evaluate_preaction_scores,
    lagged_evidence_from_memory,
    load_preaction_artifact_v2,
    score_preaction_interventions,
)
from runtime.world.thermal_scenario import ThermalScenario


def _request(**overrides):
    values = {
        "scenario_id": "thermal@1",
        "main_variable": "temperature",
        "optimization_direction": "minimize",
        "observation": {"temperature": 10.0, "alarm": True},
        "interventions": ("cool", "idle"),
        "canonical_intervention": "idle",
        "prior_evidence": {
            "cool": {"expected_delta": 0.0, "confidence": 0.1, "evidence_ref": "prior:cool"},
            "idle": {"expected_delta": -1.0, "confidence": 1.0, "evidence_ref": "prior:idle"},
        },
        "lagged_evidence": {
            "cool": {"mean_delta": -2.0, "confidence": 1.0, "sample_count": 8},
        },
        "n3_signals": {},
    }
    values.update(overrides)
    return N4PreactionInterventionSet(**values)


def _artifact(request):
    return N4PreactionArtifactV2(
        model_id="n4-v2-test",
        coefficients={name: (1.0 if name in {"prior_delta", "lagged_delta"} else 0.0) for name in FEATURE_NAMES},
        pair_bias={f"{request.scenario_id}::{action}": 0.0 for action in request.interventions},
        feature_ranges={
            f"{request.scenario_id}::{action}": {name: [-100.0, 100.0] for name in FEATURE_NAMES}
            for action in request.interventions
        },
        calibration_half_width=0.25,
        confidence=0.8,
        training_provenance={
            "trajectory_counts": {"train": 24, "validation": 6, "evaluation": 12},
            "split_disjoint": True,
        },
        artifact_sha256="a" * 64,
    )


def _score(request):
    return score_preaction_interventions(request, artifact=_artifact(request))


def test_scores_every_ordered_intervention_and_never_acquires_authority() -> None:
    request = _request(
        interventions=("cool", "idle", "unknown"),
        prior_evidence={
            "cool": {"expected_delta": -2.0, "confidence": 1.0},
            "idle": {"expected_delta": 0.0, "confidence": 0.5},
        },
        lagged_evidence={},
    )

    candidate = _score(request)
    payload = candidate.to_dict()

    assert [row["intervention"] for row in candidate.scores] == ["cool", "idle", "unknown"]
    assert [row["status"] for row in candidate.scores] == ["scored", "scored", "scored"]
    assert candidate.scores[1]["predicted_delta"] == 0.0
    assert payload["authority"] == {
        "authority_effect": "none",
        "proposal_only": True,
        "may_choose_intervention": False,
        "may_authorize_action": False,
        "may_mutate_graph": False,
    }
    assert payload["decision_influence"] == "none"


def test_hidden_outcome_changes_do_not_change_candidate_or_hash() -> None:
    request = _request()
    candidate_before = _score(request)
    frozen_payload = copy.deepcopy(candidate_before.to_dict())
    frozen_hash = candidate_before.candidate_hash

    first = evaluate_preaction_scores(
        candidate_before,
        outcomes={"cool": {"temperature": 8.0}, "idle": {"temperature": 9.0}},
        observed_value=10.0,
    )
    second = evaluate_preaction_scores(
        candidate_before,
        outcomes={"cool": {"temperature": 12.0}, "idle": {"temperature": 7.0}},
        observed_value=10.0,
    )

    assert first["candidate_hash"] == second["candidate_hash"] == frozen_hash
    assert first["candidate_hash_preserved"] is True
    assert second["candidate_hash_preserved"] is True
    assert candidate_before.to_dict() == frozen_payload
    assert candidate_before.candidate_hash == frozen_hash
    assert first["top1_correct"] is True
    assert second["top1_correct"] is False


def test_prior_and_lagged_memory_use_only_preaction_or_completed_evidence() -> None:
    scenario = ThermalScenario()
    prior = causal_signature_prior_evidence(
        scenario.causal_signature,
        interventions=scenario.config.interventions,
    )
    lagged = lagged_evidence_from_memory(
        [
            {
                "memory_id": "old-episode",
                "structure": {
                    "context": {"intervention": "activate_cooling"},
                    "result": {"factual_delta": -0.08},
                },
            },
            {
                "memory_id": "ignored",
                "structure": {"context": {"intervention": "unknown"}},
            },
        ],
        interventions=scenario.config.interventions,
    )

    assert prior["activate_cooling"]["expected_delta"] < 0.0
    assert prior["deactivate_cooling"]["expected_delta"] == 0.0
    assert lagged["activate_cooling"]["mean_delta"] == pytest.approx(-0.08)
    assert "deactivate_cooling" not in lagged


def test_evaluator_reports_accuracy_ranking_and_regret_against_both_baselines() -> None:
    candidate = _score(_request())

    result = evaluate_preaction_scores(
        candidate,
        outcomes={"cool": {"state": {"temperature": 8.0}}, "idle": 9.0},
        observed_value=10.0,
    )

    assert result["ground_truth_top_interventions"] == ["cool"]
    assert result["predicted_top_interventions"] == ["cool"]
    assert result["prior_top_interventions"] == ["idle"]
    assert result["top1_correct"] is True
    assert result["pairwise_ranking_accuracy"] == 1.0
    assert result["pairwise_comparisons"] == 1
    assert result["canonical_regret"] == 1.0
    assert result["n4_regret"] == 0.0
    assert result["prior_regret"] == 1.0
    assert result["regret_delta_vs_canonical"] == 1.0
    assert result["regret_delta_vs_prior"] == 1.0
    assert result["mae_next_value"] == 0.0
    assert result["mae_delta"] == 0.0


def test_top1_accepts_ground_truth_ties_and_uses_stable_action_order() -> None:
    request = _request(
            canonical_intervention="cool",
            prior_evidence={
                "cool": {"expected_delta": -1.0, "confidence": 1.0},
                "idle": {"expected_delta": -1.0, "confidence": 1.0},
            },
            lagged_evidence={},
        )
    candidate = _score(request)

    result = evaluate_preaction_scores(
        candidate,
        outcomes={"cool": 9.0, "idle": 9.0 + 5e-10},
        observed_value=10.0,
    )

    assert result["ground_truth_top_interventions"] == ["cool", "idle"]
    assert result["predicted_top_interventions"] == ["cool", "idle"]
    assert result["shadow_intervention"] == "cool"
    assert result["top1_correct"] is True
    assert result["pairwise_ranking_accuracy"] is None


def test_maximize_direction_uses_higher_values_for_ranking_and_regret() -> None:
    request = _request(
            main_variable="stock",
            optimization_direction="maximize",
            observation={"stock": 2.0},
            canonical_intervention="idle",
            prior_evidence={
                "cool": {"expected_delta": 2.0, "confidence": 1.0},
                "idle": {"expected_delta": 0.0, "confidence": 1.0},
            },
            lagged_evidence={},
        )
    candidate = _score(request)

    result = evaluate_preaction_scores(
        candidate, outcomes={"cool": 4.0, "idle": 2.0}, observed_value=2.0
    )

    assert result["predicted_top_interventions"] == ["cool"]
    assert result["top1_correct"] is True
    assert result["canonical_regret"] == 2.0
    assert result["regret_delta_vs_canonical"] == 2.0


@pytest.mark.parametrize(
    "leak",
    [
        {"causal_attestation": {}},
        {"current_transition": {"temperature": 8.0}},
        {"relation_kind": "support"},
        {"committed_intervention": "cool"},
        {"nested": {"outcomes": {"cool": 8.0}}},
    ],
)
def test_outcome_leak_guard_rejects_forbidden_observation_fields(leak) -> None:
    with pytest.raises(ValueError, match="n4_preaction_outcome_leak"):
        _request(observation={"temperature": 10.0, **leak})


def test_mapping_contract_rejects_extra_post_action_fields() -> None:
    payload = _request().to_dict()
    payload["causal_attestation"] = {"factual_delta": -2.0}

    with pytest.raises(ValueError, match="n4_preaction_field_forbidden"):
        N4PreactionInterventionSet.from_mapping(payload)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"interventions": ("cool", "cool")}, "ordered_unique"),
        ({"canonical_intervention": "forbidden"}, "not_allowed"),
        ({"observation": {"temperature": float("nan")}}, "must_be_finite"),
        ({"optimization_direction": "target_band"}, "unsupported"),
        (
            {"prior_evidence": {"forbidden": {"expected_delta": 1.0}}},
            "unknown_intervention",
        ),
    ],
)
def test_preaction_contract_fails_closed(overrides, error) -> None:
    with pytest.raises(ValueError, match=error):
        _request(**overrides)


def test_evaluation_rejects_outcomes_for_actions_outside_candidate() -> None:
    candidate = _score(_request())

    with pytest.raises(ValueError, match="unknown_intervention"):
        evaluate_preaction_scores(
            candidate,
            outcomes={"cool": 8.0, "idle": 9.0, "injected": -100.0},
            observed_value=10.0,
        )


def test_from_mapping_roundtrip_preserves_input_hash() -> None:
    request = _request()
    restored = N4PreactionInterventionSet.from_mapping(request.to_dict())

    assert restored.schema_version == PREACTION_SCHEMA_VERSION
    assert restored.to_dict() == request.to_dict()
    assert restored.input_hash == request.input_hash


def test_missing_artifact_abstains_all_interventions() -> None:
    candidate = score_preaction_interventions(_request())

    assert candidate.execution_class == "abstained"
    assert all(row["status"] == "abstained" for row in candidate.scores)
    assert {row["abstention_reason"] for row in candidate.scores} == {"artifact_missing"}


def test_loader_binds_manifest_to_artifact_hash_and_rejects_tampering(tmp_path) -> None:
    artifact = _artifact(_request())
    payload = {
        "schema_version": artifact.schema_version,
        "backend": artifact.backend,
        "model_id": artifact.model_id,
        "feature_names": list(artifact.feature_names),
        "coefficients": dict(artifact.coefficients),
        "pair_bias": dict(artifact.pair_bias),
        "feature_ranges": {
            pair: {name: list(bounds) for name, bounds in ranges.items()}
            for pair, ranges in artifact.feature_ranges.items()
        },
        "calibration_half_width": artifact.calibration_half_width,
        "confidence": artifact.confidence,
        "training_provenance": dict(artifact.training_provenance),
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_bytes(raw)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "n4-preaction-manifest-v1",
                "backend": "rnfe-n4-preaction-linear-v2",
                "artifact_path": "artifact.json",
                "artifact_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    )

    loaded = load_preaction_artifact_v2(manifest_path)
    assert loaded.model_id == artifact.model_id
    artifact_path.write_text("{}")
    with pytest.raises(ValueError, match="hash_mismatch"):
        load_preaction_artifact_v2(manifest_path)


def test_ood_request_abstains_without_falling_back_to_prior() -> None:
    request = _request()
    artifact = _artifact(request)
    narrow = N4PreactionArtifactV2(
        model_id=artifact.model_id,
        coefficients=artifact.coefficients,
        pair_bias=artifact.pair_bias,
        feature_ranges={
            pair: {
                name: ([0.0, 1.0] if name == "observed_value" else bounds)
                for name, bounds in ranges.items()
            }
            for pair, ranges in artifact.feature_ranges.items()
        },
        calibration_half_width=artifact.calibration_half_width,
        confidence=artifact.confidence,
        training_provenance=artifact.training_provenance,
        artifact_sha256=artifact.artifact_sha256,
    )

    candidate = score_preaction_interventions(request, artifact=narrow)

    assert all(row["status"] == "abstained" for row in candidate.scores)
    assert {row["abstention_reason"] for row in candidate.scores} == {
        "ood:observed_value"
    }


def test_evaluator_rejects_outcome_substitution_against_oracle_seal() -> None:
    candidate = _score(_request())
    snapshot = "b" * 64
    sealed_rows = [
        {
            "intervention": "cool",
            "value": 8.0,
            "delta": -2.0,
            "state": {"temperature": 8.0},
        },
        {
            "intervention": "idle",
            "value": 9.0,
            "delta": -1.0,
            "state": {"temperature": 9.0},
        },
    ]
    seal = hashlib.sha256(
        json.dumps(
            {
                "snapshot_sha256": snapshot,
                "outcomes": sealed_rows,
                "best_actions": ["cool"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="outcome_seal_mismatch"):
        evaluate_preaction_scores(
            candidate,
            outcomes={
                "cool": {"state": {"temperature": 12.0}},
                "idle": {"state": {"temperature": 9.0}},
            },
            observed_value=10.0,
            oracle_snapshot_sha256=snapshot,
            outcome_set_sha256=seal,
        )
