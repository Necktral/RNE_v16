from __future__ import annotations

from copy import deepcopy

import pytest

from runtime.neural.integration.adapters import N2Adapter
from runtime.neural.integration.p1_n2 import (
    N2GroundTruthScore,
    N2RevisionRequest,
    N2ShadowRevisionRecord,
    build_n2_revision_request,
    cap_n2_revision_confidence,
    n2_revision_eligibility,
    score_n2_ground_truth,
    select_n2_revision_action,
)


def _candidate(
    *,
    ded: bool = True,
    lotf: bool = True,
    nesy: bool = False,
    verified: bool = False,
    dissonance: tuple[str, ...] = ("numeric_unsupport",),
) -> dict:
    return {
        "verified": verified,
        "verification": {"DED": ded, "LOT-F": lotf, "NESY": nesy},
        "nesy": {"state_delta": {"nesy_dissonance": list(dissonance)}},
    }


def _request() -> N2RevisionRequest:
    return N2RevisionRequest(
        initial_candidate_hash="a" * 64,
        initial_input_hash="b" * 64,
        base_intervention="KEEP",
        candidate_intervention="COOL",
        selection_source="ABD",
        trigger_codes=("numeric_unsupport",),
    )


def test_revision_is_eligible_only_for_nesy_rejection_after_symbolic_acceptance() -> None:
    eligible, triggers, reason = n2_revision_eligibility(_candidate())
    assert eligible is True
    assert triggers == ("numeric_unsupport",)
    assert reason == "eligible_nesy_rejection"

    assert n2_revision_eligibility(_candidate(ded=False))[2] == "symbolic_boundary_rejected"
    assert n2_revision_eligibility(_candidate(lotf=False))[2] == "symbolic_boundary_rejected"
    assert n2_revision_eligibility(
        _candidate(nesy=True, verified=True, dissonance=())
    )[2] == "initial_candidate_accepted"
    assert n2_revision_eligibility(None)[2] == "candidate_unavailable"


def test_action_selection_prefers_existing_reasoning_and_never_repeats_base() -> None:
    action, source = select_n2_revision_action(
        reasoning_state={
            "abd_top_intervention": "KEEP",
            "opt_intervention": "COOL",
            "ctf_checked": {"resim": {"best_alternative": "WAIT"}},
        },
        allowed_interventions=("KEEP", "COOL", "WAIT"),
        base_intervention="KEEP",
    )
    assert (action, source) == ("COOL", "OPT")

    fallback = select_n2_revision_action(
        reasoning_state={"abd_top_intervention": "UNKNOWN"},
        allowed_interventions=("KEEP", "WAIT"),
        base_intervention="KEEP",
    )
    assert fallback == ("WAIT", "ALLOWED_ALTERNATIVE")

    assert select_n2_revision_action(
        reasoning_state={},
        allowed_interventions=("KEEP",),
        base_intervention="KEEP",
    ) == (None, "NO_ALTERNATIVE")


def test_request_is_exactly_one_shadow_attempt_and_has_no_authority() -> None:
    request = build_n2_revision_request(
        candidate=_candidate(dissonance=("symbolic_numeric_action_mismatch",)),
        initial_candidate_hash="f" * 64,
        initial_input_hash="e" * 64,
        reasoning_state={"abd_top_intervention": "COOL"},
        allowed_interventions=("KEEP", "COOL"),
        base_intervention="KEEP",
    )
    assert request is not None
    payload = request.to_dict()
    assert payload["attempt_index"] == payload["max_attempts"] == 1
    assert payload["required_family"] == "IND"
    assert payload["mode"] == "shadow_counterfactual"
    assert payload["authority_effect"] == "none"
    assert payload["decision_influence"] == "none"

    with pytest.raises(ValueError, match="exactly_one_attempt"):
        N2RevisionRequest(
            initial_candidate_hash="f" * 64,
            initial_input_hash="e" * 64,
            base_intervention="KEEP",
            candidate_intervention="COOL",
            selection_source="ABD",
            trigger_codes=("numeric_unsupport",),
            max_attempts=2,
        )


def test_request_builder_fails_closed_for_symbolic_rejection_or_no_alternative() -> None:
    assert build_n2_revision_request(
        candidate=_candidate(ded=False),
        initial_candidate_hash="f" * 64,
        initial_input_hash="e" * 64,
        reasoning_state={"abd_top_intervention": "COOL"},
        allowed_interventions=("KEEP", "COOL"),
        base_intervention="KEEP",
    ) is None
    assert build_n2_revision_request(
        candidate=_candidate(),
        initial_candidate_hash="f" * 64,
        initial_input_hash="e" * 64,
        reasoning_state={},
        allowed_interventions=("KEEP",),
        base_intervention="KEEP",
    ) is None


@pytest.mark.parametrize(
    ("initial", "revised", "expected"),
    [(0.9, 0.95, 0.72), (0.5, 0.9, 0.4), (-1.0, 0.8, 0.0), (float("nan"), 0.8, 0.0)],
)
def test_revision_confidence_is_penalized_and_bounded(
    initial: float, revised: float, expected: float
) -> None:
    assert cap_n2_revision_confidence(
        initial_confidence=initial,
        revised_confidence=revised,
    ) == pytest.approx(expected)


def test_ground_truth_score_detects_valid_correction_without_oracle_leakage() -> None:
    action_values = {"KEEP": 0.8, "COOL": 0.3, "WAIT": 0.9}
    snapshot = deepcopy(action_values)
    score = score_n2_ground_truth(
        action_values=action_values,
        optimization_direction="minimize",
        base_intervention="KEEP",
        retry_intervention="COOL",
        initial_verified=False,
        retry_verified=True,
    )
    assert action_values == snapshot
    assert score.scored is True
    assert score.best_actions == ("COOL",)
    assert score.initial_false_rejection is False
    assert score.valid_correction is True
    assert score.retry_false_accept is False
    assert score.final_false_rejection is False


def test_ground_truth_score_handles_ties_and_false_rejection() -> None:
    score = score_n2_ground_truth(
        action_values={"KEEP": 1.0, "COOL": 1.0, "WAIT": 0.5},
        optimization_direction="maximize",
        base_intervention="KEEP",
        retry_intervention="COOL",
        initial_verified=False,
        retry_verified=True,
    )
    assert score.best_actions == ("COOL", "KEEP")
    assert score.initial_false_rejection is True
    assert score.valid_correction is True
    assert score.final_false_rejection is False


def test_ground_truth_score_marks_missing_or_invalid_oracle_unscored() -> None:
    missing = score_n2_ground_truth(
        action_values={"KEEP": 1.0},
        optimization_direction="minimize",
        base_intervention="KEEP",
        retry_intervention="COOL",
        initial_verified=False,
        retry_verified=False,
    )
    assert missing == N2GroundTruthScore(
        scored=False,
        unscored_reason="required_action_value_missing",
    )
    invalid = score_n2_ground_truth(
        action_values={"KEEP": 1.0, "COOL": 0.0},
        optimization_direction="sideways",
        base_intervention="KEEP",
        retry_intervention="COOL",
        initial_verified=False,
        retry_verified=False,
    )
    assert invalid.unscored_reason == "optimization_direction_invalid"


def test_shadow_record_rejects_false_acceptance_contract_and_serializes_no_authority() -> None:
    with pytest.raises(ValueError, match="accepted_without_all_verifiers"):
        N2ShadowRevisionRecord(
            request=_request(),
            status="accepted",
            initial_confidence=0.7,
            revised_confidence=0.8,
            effective_confidence=0.56,
            revised_verification={"DED": True, "LOT-F": True, "NESY": False},
        )
    with pytest.raises(ValueError, match="effective_confidence_exceeds_cap"):
        N2ShadowRevisionRecord(
            request=_request(),
            status="rejected",
            initial_confidence=0.5,
            revised_confidence=0.8,
            effective_confidence=0.5,
        )
    record = N2ShadowRevisionRecord(
        request=_request(),
        status="accepted",
        initial_confidence=0.7,
        revised_confidence=0.8,
        effective_confidence=0.56,
        revised_verification={"DED": True, "LOT-F": True, "NESY": True},
        revised_candidate_hash="b" * 64,
    )
    payload = record.to_dict()
    assert payload["attempt_count"] == 1
    assert payload["authority_effect"] == payload["decision_influence"] == "none"


def test_n2_adapter_pure_verify_preserves_state_and_existing_candidate_shape() -> None:
    state = {
        "ded_validated": True,
        "ded_conclusion": "safe",
        "intervention": "COOL",
        "abd_top_intervention": "COOL",
        "cau_link": {"helps_goal": True},
        "prob_posterior": {"point": 0.8},
    }
    snapshot = deepcopy(state)
    output = N2Adapter.verify(
        reasoning_state=state,
        lotf_valid=True,
        formula="SAFE",
    )
    assert state == snapshot
    assert output.candidate_output["verified"] is True
    assert output.candidate_output["verification"] == {
        "DED": True,
        "LOT-F": True,
        "NESY": True,
    }
    assert output.candidate_output["authority"] == "DED+LOT-F+NESY"
