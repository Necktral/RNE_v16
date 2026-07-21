"""Pure contracts and policies for the bounded N2 shadow revision experiment.

This module deliberately does not orchestrate a retry.  It only decides whether
one is eligible, selects a non-authoritative alternative, caps its confidence,
and scores the resulting evidence against an oracle that is supplied after the
reasoning pass.  Keeping these operations pure prevents the evaluation oracle
from leaking into the candidate-generation path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .contracts import canonical_sha256


N2_REVISION_REQUEST_SCHEMA_VERSION = "n2-shadow-revision-request-v1"
N2_REVISION_RECORD_SCHEMA_VERSION = "n2-shadow-revision-record-v1"
N2_GROUND_TRUTH_SCHEMA_VERSION = "n2-ground-truth-score-v1"

_REVISION_STATUSES = frozenset({"accepted", "rejected", "abstained", "error"})


def _finite_probability(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(max(number, 0.0), 1.0)


def _string_tuple(values: Sequence[Any] | None) -> tuple[str, ...]:
    output: list[str] = []
    for value in values or ():
        item = str(value).strip()
        if item and item not in output:
            output.append(item)
    return tuple(output)


@dataclass(frozen=True, slots=True)
class N2RevisionRequest:
    """One and only one counterfactual revision requested by an N2 rejection."""

    initial_candidate_hash: str
    initial_input_hash: str
    base_intervention: str
    candidate_intervention: str
    selection_source: str
    trigger_codes: tuple[str, ...]
    required_family: str = "IND"
    attempt_index: int = 1
    max_attempts: int = 1

    def __post_init__(self) -> None:
        if not self.initial_candidate_hash.strip():
            raise ValueError("n2_revision_initial_candidate_hash_required")
        if not self.initial_input_hash.strip():
            raise ValueError("n2_revision_initial_input_hash_required")
        if not self.base_intervention.strip():
            raise ValueError("n2_revision_base_intervention_required")
        if not self.candidate_intervention.strip():
            raise ValueError("n2_revision_candidate_intervention_required")
        if self.base_intervention == self.candidate_intervention:
            raise ValueError("n2_revision_candidate_must_differ_from_base")
        if not self.selection_source.strip():
            raise ValueError("n2_revision_selection_source_required")
        if not self.trigger_codes:
            raise ValueError("n2_revision_trigger_codes_required")
        if self.required_family.upper() != "IND":
            raise ValueError("n2_revision_required_family_must_be_ind")
        if self.attempt_index != 1 or self.max_attempts != 1:
            raise ValueError("n2_revision_exactly_one_attempt_required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": N2_REVISION_REQUEST_SCHEMA_VERSION,
            "initial_candidate_hash": self.initial_candidate_hash,
            "initial_input_hash": self.initial_input_hash,
            "base_binding_hash": canonical_sha256(
                {
                    "candidate_hash": self.initial_candidate_hash,
                    "input_hash": self.initial_input_hash,
                    "base_intervention": self.base_intervention,
                }
            ),
            "base_intervention": self.base_intervention,
            "candidate_intervention": self.candidate_intervention,
            "selection_source": self.selection_source,
            "trigger_codes": list(self.trigger_codes),
            "required_family": self.required_family.upper(),
            "attempt_index": self.attempt_index,
            "max_attempts": self.max_attempts,
            "mode": "shadow_counterfactual",
            "authority_effect": "none",
            "decision_influence": "none",
        }


@dataclass(frozen=True, slots=True)
class N2GroundTruthScore:
    """Action-grounded score computed without exposing the oracle to N2."""

    scored: bool
    best_actions: tuple[str, ...] = ()
    initial_optimal: bool | None = None
    retry_optimal: bool | None = None
    initial_false_rejection: bool | None = None
    valid_correction: bool | None = None
    retry_false_accept: bool | None = None
    final_false_rejection: bool | None = None
    unscored_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": N2_GROUND_TRUTH_SCHEMA_VERSION,
            "scored": self.scored,
            "best_actions": list(self.best_actions),
            "initial_optimal": self.initial_optimal,
            "retry_optimal": self.retry_optimal,
            "initial_false_rejection": self.initial_false_rejection,
            "valid_correction": self.valid_correction,
            "retry_false_accept": self.retry_false_accept,
            "final_false_rejection": self.final_false_rejection,
            "unscored_reason": self.unscored_reason,
        }


@dataclass(frozen=True, slots=True)
class N2ShadowRevisionRecord:
    """Durable evidence for a retry that has no live authority."""

    request: N2RevisionRequest
    status: str
    initial_confidence: float
    revised_confidence: float
    effective_confidence: float
    revised_verification: Mapping[str, bool] = field(default_factory=dict)
    revised_candidate_hash: str | None = None
    ground_truth: N2GroundTruthScore | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _REVISION_STATUSES:
            raise ValueError("n2_revision_status_invalid")
        for name in ("initial_confidence", "revised_confidence", "effective_confidence"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"n2_revision_{name}_invalid")
        allowed_confidence = cap_n2_revision_confidence(
            initial_confidence=self.initial_confidence,
            revised_confidence=self.revised_confidence,
        )
        if self.effective_confidence > allowed_confidence + 1e-12:
            raise ValueError("n2_revision_effective_confidence_exceeds_cap")
        if self.status == "accepted" and not all(
            bool(self.revised_verification.get(authority))
            for authority in ("DED", "LOT-F", "NESY")
        ):
            raise ValueError("n2_revision_accepted_without_all_verifiers")
        if self.status == "accepted" and not self.revised_candidate_hash:
            raise ValueError("n2_revision_accepted_candidate_hash_required")
        if self.status in {"abstained", "error"} and not self.failure_reason:
            raise ValueError("n2_revision_failure_reason_required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": N2_REVISION_RECORD_SCHEMA_VERSION,
            "request": self.request.to_dict(),
            "status": self.status,
            "attempt_count": 1,
            "initial_confidence": self.initial_confidence,
            "revised_confidence": self.revised_confidence,
            "effective_confidence": self.effective_confidence,
            "revised_verification": dict(self.revised_verification),
            "revised_candidate_hash": self.revised_candidate_hash,
            "ground_truth": self.ground_truth.to_dict() if self.ground_truth else None,
            "failure_reason": self.failure_reason,
            "mode": "shadow_counterfactual",
            "authority_effect": "none",
            "decision_influence": "none",
        }


def n2_revision_eligibility(candidate: Mapping[str, Any] | None) -> tuple[bool, tuple[str, ...], str]:
    """Return eligibility, trigger codes, and a stable terminal reason.

    Only a semantic NESY rejection may request another reasoning pass.  A DED or
    LOT-F rejection is a hard symbolic boundary and must abstain instead of
    laundering an invalid proposition through another family.
    """

    if not isinstance(candidate, Mapping):
        return False, (), "candidate_unavailable"
    if bool(candidate.get("verified")):
        return False, (), "initial_candidate_accepted"
    verification = candidate.get("verification")
    verification = verification if isinstance(verification, Mapping) else {}
    if not bool(verification.get("DED")) or not bool(verification.get("LOT-F")):
        return False, (), "symbolic_boundary_rejected"
    if bool(verification.get("NESY")):
        return False, (), "verification_contract_inconsistent"
    nesy = candidate.get("nesy")
    nesy = nesy if isinstance(nesy, Mapping) else {}
    delta = nesy.get("state_delta")
    delta = delta if isinstance(delta, Mapping) else {}
    triggers = _string_tuple(delta.get("nesy_dissonance"))
    if not triggers:
        triggers = ("nesy_rejected_without_dissonance_code",)
    return True, triggers, "eligible_nesy_rejection"


def select_n2_revision_action(
    *,
    reasoning_state: Mapping[str, Any],
    allowed_interventions: Sequence[str],
    base_intervention: str,
) -> tuple[str | None, str]:
    """Select one alternative without consulting outcome ground truth.

    Existing reasoning suggestions win.  The final fallback is the first allowed
    alternative in scenario order; it is experimental exploration, not an oracle
    choice, and its source remains explicit in the record.
    """

    allowed = _string_tuple(allowed_interventions)
    allowed_set = set(allowed)
    ctf = reasoning_state.get("ctf_checked")
    ctf = ctf if isinstance(ctf, Mapping) else {}
    resim = ctf.get("resim")
    resim = resim if isinstance(resim, Mapping) else {}
    proposed = (
        (reasoning_state.get("abd_top_intervention"), "ABD"),
        (reasoning_state.get("opt_intervention"), "OPT"),
        (resim.get("best_alternative"), "CTF_RESIM"),
        (reasoning_state.get("ind_best_intervention"), "IND"),
    )
    for raw_action, source in proposed:
        action = str(raw_action or "").strip()
        if action and action != base_intervention and action in allowed_set:
            return action, source
    for action in allowed:
        if action != base_intervention:
            return action, "ALLOWED_ALTERNATIVE"
    return None, "NO_ALTERNATIVE"


def build_n2_revision_request(
    *,
    candidate: Mapping[str, Any] | None,
    initial_candidate_hash: str,
    initial_input_hash: str,
    reasoning_state: Mapping[str, Any],
    allowed_interventions: Sequence[str],
    base_intervention: str,
) -> N2RevisionRequest | None:
    """Build the sole retry request, or ``None`` for every fail-closed case."""

    eligible, trigger_codes, _reason = n2_revision_eligibility(candidate)
    if not eligible:
        return None
    action, source = select_n2_revision_action(
        reasoning_state=reasoning_state,
        allowed_interventions=allowed_interventions,
        base_intervention=base_intervention,
    )
    if action is None:
        return None
    return N2RevisionRequest(
        initial_candidate_hash=initial_candidate_hash,
        initial_input_hash=initial_input_hash,
        base_intervention=base_intervention,
        candidate_intervention=action,
        selection_source=source,
        trigger_codes=trigger_codes,
    )


def cap_n2_revision_confidence(
    *,
    initial_confidence: Any,
    revised_confidence: Any,
    penalty: float = 0.8,
    ceiling: float = 0.75,
) -> float:
    """Apply a conservative confidence penalty to a repaired candidate."""

    if not math.isfinite(penalty) or penalty < 0.0 or penalty > 1.0:
        raise ValueError("n2_revision_penalty_out_of_range")
    if not math.isfinite(ceiling) or ceiling < 0.0 or ceiling > 1.0:
        raise ValueError("n2_revision_ceiling_out_of_range")
    return min(
        _finite_probability(revised_confidence),
        _finite_probability(initial_confidence) * penalty,
        ceiling,
    )


def score_n2_ground_truth(
    *,
    action_values: Mapping[str, Any],
    optimization_direction: str,
    base_intervention: str,
    retry_intervention: str,
    initial_verified: bool,
    retry_verified: bool,
    epsilon: float = 1e-9,
) -> N2GroundTruthScore:
    """Score initial/retry predictions against post-hoc action values.

    The caller must compute ``action_values`` from the same pre-action state and
    must not include this score in either reasoning context.
    """

    if optimization_direction not in {"minimize", "maximize"}:
        return N2GroundTruthScore(scored=False, unscored_reason="optimization_direction_invalid")
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("n2_ground_truth_epsilon_invalid")
    values: dict[str, float] = {}
    for raw_action, raw_value in action_values.items():
        action = str(raw_action).strip()
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if action and math.isfinite(value):
            values[action] = value
    if base_intervention not in values or retry_intervention not in values:
        return N2GroundTruthScore(scored=False, unscored_reason="required_action_value_missing")
    target = min(values.values()) if optimization_direction == "minimize" else max(values.values())
    best_actions = tuple(
        sorted(action for action, value in values.items() if abs(value - target) <= epsilon)
    )
    initial_optimal = base_intervention in best_actions
    retry_optimal = retry_intervention in best_actions
    initial_false_rejection = (not initial_verified) and initial_optimal
    valid_correction = (not initial_verified) and retry_verified and retry_optimal
    retry_false_accept = retry_verified and not retry_optimal
    final_false_rejection = retry_optimal and not retry_verified
    return N2GroundTruthScore(
        scored=True,
        best_actions=best_actions,
        initial_optimal=initial_optimal,
        retry_optimal=retry_optimal,
        initial_false_rejection=initial_false_rejection,
        valid_correction=valid_correction,
        retry_false_accept=retry_false_accept,
        final_false_rejection=final_false_rejection,
    )
