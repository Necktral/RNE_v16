from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, is_dataclass
import math

import pytest

import runtime.experiments.cognitive_gain.contracts as contracts
from runtime.experiments.cognitive_gain.contracts import (
    CausalEpisodeEvidence,
    CausalInfluenceReceipt,
    CausalLedgerEvent,
    CognitiveAdmissionDecision,
    CognitiveAdmissionVerdict,
    CognitiveInfluenceProposal,
    InfluenceType,
    LedgerEventType,
    LessonStatus,
    LessonCandidate,
    ObservedEffect,
    ProducerType,
    RetrievalDecision,
    ValidatedLesson,
    VerificationRecord,
    VerificationVerdict,
)


H = "a" * 64


def evidence(**overrides):
    values = {
        "evidence_id": "evidence-1",
        "experiment_id": "experiment-1",
        "arm_id": "shadow",
        "chain_id": "chain-1",
        "scenario_id": "scenario-1",
        "scenario_version": "v1",
        "causal_signature_hash": H,
        "episode_id": "episode-1",
        "paired_run_id": "pair-1",
        "seed": 1,
        "state_before_hash": H,
        "available_interventions": ["hold", "cool"],
        "baseline_intervention": "hold",
        "committed_intervention": "hold",
        "state_after_hash": H,
        "reward": 1.0,
        "severity": 0.2,
        "regret": 0.0,
        "outcome_hash": H,
    }
    values.update(overrides)
    return CausalEpisodeEvidence(**values)


def test_contract_is_frozen_slotted_and_defensively_copies_collections():
    actions = ["hold", "cool"]
    item = evidence(available_interventions=actions)
    assert is_dataclass(item)
    assert item.__slots__
    assert item.available_interventions == ("hold", "cool")
    actions.append("heat")
    assert item.available_interventions == ("hold", "cool")
    with pytest.raises(FrozenInstanceError):
        item.reward = 2.0


@pytest.mark.parametrize("field_name", ["evidence_id", "experiment_id", "chain_id"])
def test_empty_ids_are_rejected(field_name):
    with pytest.raises(ValueError, match="required"):
        evidence(**{field_name: "  "})


def test_malformed_hash_is_rejected():
    with pytest.raises(ValueError, match="sha256"):
        evidence(outcome_hash="ABC")


def test_unknown_enum_values_are_rejected():
    with pytest.raises(ValueError, match="ProducerType"):
        LessonCandidate(
            candidate_id="candidate-1",
            producer_type="unknown",
            producer_version="v1",
            source_evidence_ids=("evidence-1",),
            applicability_conditions_hash=H,
            preferred_intervention="hold",
            avoided_intervention=None,
            proposition_hash=H,
            supporting_evidence_hash=H,
            expected_effect=1.0,
            refutation_conditions_hash=H,
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_contract_numbers_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        evidence(reward=value)


def test_admission_name_is_collision_safe():
    assert not hasattr(contracts, "AdmissionDecision")
    decision = CognitiveAdmissionDecision(
        decision_id="decision-1",
        proposal_id="proposal-1",
        verdict=CognitiveAdmissionVerdict.REJECTED,
        checks_executed=("scope",),
        reasons=("passive_only",),
        authorized_budget=0.0,
    )
    assert isinstance(decision, CognitiveAdmissionDecision)


def test_passive_contracts_expose_no_action_application_methods():
    forbidden = {"apply", "select", "commit", "execute", "replace", "substitute"}
    contract_types = [
        value
        for value in vars(contracts).values()
        if isinstance(value, type) and is_dataclass(value) and value.__module__ == contracts.__name__
    ]
    for kind in contract_types:
        method_names = {name.lower() for name, value in vars(kind).items() if callable(value)}
        assert not (method_names & forbidden), kind.__name__


def test_all_public_contracts_are_frozen_and_slotted():
    for name in contracts.__all__:
        kind = getattr(contracts, name)
        if isinstance(kind, type) and is_dataclass(kind):
            assert kind.__dataclass_params__.frozen, name
            assert "__slots__" in vars(kind), name


def test_wall_clock_time_is_excluded_from_causal_event_hash():
    common = dict(
        event_id="event-1",
        event_type=LedgerEventType.EXPERIENCE_OBSERVED,
        experiment_id="experiment-1",
        arm_id="shadow",
        chain_id="chain-1",
        event_sequence=0,
        previous_event_hash=contracts.GENESIS_PREVIOUS_EVENT_HASH,
        logical_time=0,
        payload_contract_type="CausalEpisodeEvidence",
        payload_hash=H,
    )
    first = CausalLedgerEvent(wall_clock_time="2026-08-03T00:00:00Z", **common)
    second = CausalLedgerEvent(wall_clock_time="2030-01-01T00:00:00Z", **common)
    assert first.event_hash == second.event_hash


def test_derived_hash_ignores_supplied_hash_value():
    item = evidence(evidence_hash="f" * 64)
    assert item.evidence_hash != "f" * 64
    assert len(item.evidence_hash) == 64


def test_proposal_is_data_only_and_immutable():
    hashes = [H]
    proposal = CognitiveInfluenceProposal(
        proposal_id="proposal-1",
        influence_type=InfluenceType.VALIDATED_LESSON,
        source_artifact_id="lesson-1",
        original_intervention="hold",
        proposed_intervention="cool",
        evidence_hashes=hashes,
        expected_effect=0.4,
        requested_budget=0.1,
    )
    hashes.append("b" * 64)
    assert proposal.evidence_hashes == (H,)
    assert set(field.name for field in fields(proposal)) >= {"proposal_id", "proposal_hash"}


def test_remaining_required_contracts_construct_and_hash():
    verification = VerificationRecord(
        verification_id="verification-1",
        candidate_id="candidate-1",
        verifier_id="verifier-1",
        verifier_version="v1",
        verification_method="paired-counterfactual",
        evidence_checked_ids=("evidence-1",),
        counterfactuals_checked_hash=H,
        verdict=VerificationVerdict.VALID,
        scope_adjustments_hash=H,
        rejection_reasons=(),
    )
    lesson = ValidatedLesson(
        lesson_id="lesson-1",
        candidate_id="candidate-1",
        verification_id=verification.verification_id,
        lesson_version="v1",
        status=LessonStatus.ACTIVE,
        applicability_scope_hash=H,
        causal_signature_hash=H,
        preferred_intervention="cool",
        avoided_intervention="heat",
        expected_effect=0.5,
        confidence_bound=0.8,
        authorized_consumers=("shadow-observer",),
        maximum_influence_budget=0.1,
        expires_at_logical_time=10,
        revocation_conditions_hash=H,
    )
    retrieval = RetrievalDecision(
        retrieval_id="retrieval-1",
        episode_id="episode-1",
        query_context_hash=H,
        eligible_lesson_ids=(lesson.lesson_id,),
        retrieved_lesson_ids=(lesson.lesson_id,),
        rejected_lesson_ids=(),
        scope_checks_hash=H,
        version_checks_hash=H,
        retrieval_budget=0.1,
    )
    receipt = CausalInfluenceReceipt(
        receipt_id="receipt-1",
        experiment_id="experiment-1",
        arm_id="shadow",
        chain_id="chain-1",
        episode_id="episode-1",
        source_artifact_id=lesson.lesson_id,
        retrieval_id=retrieval.retrieval_id,
        proposal_id="proposal-1",
        cognitive_admission_decision_id="decision-1",
        baseline_intervention="hold",
        proposed_intervention="hold",
        committed_intervention="hold",
        action_changed=False,
        budget_consumed=0.0,
        reversible=True,
        observed_effect_id="effect-1",
    )
    effect = ObservedEffect(
        effect_id="effect-1",
        receipt_id=receipt.receipt_id,
        episode_id="episode-1",
        paired_run_id="pair-1",
        state_after_hash=H,
        reward=1.0,
        severity=0.1,
        regret=0.0,
        constraint_violations=(),
        resource_cost_hash=H,
        latency=0.01,
    )
    for derived_hash in (
        verification.verification_hash,
        lesson.lesson_hash,
        retrieval.retrieval_hash,
        receipt.receipt_hash,
        effect.effect_hash,
    ):
        assert len(derived_hash) == 64
