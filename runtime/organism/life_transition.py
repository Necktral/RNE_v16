"""Cadena criptográfica de transiciones vitales Ω_t → Ω_t+1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

from .dynamic_state import OrganismDynamicState, canonical_hash


LIFE_TRANSITION_SCHEMA_VERSION = "organism-life-transition-v1"
DYNAMIC_CHAIN_CHECKPOINT_VERSION = "organism-dynamic-chain-checkpoint-v1"


@dataclass(frozen=True, slots=True)
class LifeTransition:
    transition_id: str
    transition_index: int
    previous_transition_id: str | None
    previous_transition_hash: str
    transition_hash: str
    state_before: OrganismDynamicState
    action_proposals: tuple[Mapping[str, Any], ...]
    authoritative_decision: Mapping[str, Any]
    committed_intervention: Mapping[str, Any]
    external_input: Any
    factual_outcome: Mapping[str, Any]
    counterfactual_evidence: Mapping[str, Any]
    certificate: Mapping[str, Any]
    reward: Mapping[str, Any]
    memory_delta: Mapping[str, Any]
    neural_state_delta: Mapping[str, Any]
    policy_delta: Mapping[str, Any]
    resource_delta: Mapping[str, Any]
    viability_delta: Mapping[str, Any]
    regime_before: Mapping[str, Any]
    regime_after: Mapping[str, Any]
    rollback_refuge_result: Mapping[str, Any]
    state_after: OrganismDynamicState
    trace_group_id: str
    status: str
    reason_code: str | None = None
    schema_version: str = LIFE_TRANSITION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DynamicLifeChain:
    """Cadena acotada por organismo/linaje; un nuevo run_id no rompe continuidad."""

    def __init__(self, *, organism_id: str, lineage_id: str, run_id: str, max_history: int = 128):
        self.organism_id = organism_id
        self.lineage_id = lineage_id
        self.run_id = run_id
        self.max_history = max(1, int(max_history))
        self.transition_index = 0
        self.head_transition_id: str | None = None
        self.head_transition_hash = canonical_hash(
            {"genesis": True, "organism_id": organism_id, "lineage_id": lineage_id}
        )
        self.last_state: OrganismDynamicState | None = None
        self.transitions: list[LifeTransition] = []
        self.chain_epoch = 1
        self.restore_reason: str | None = None

    def commit(
        self,
        *,
        state_before: OrganismDynamicState,
        state_after: OrganismDynamicState,
        trace_group_id: str,
        action_proposals: tuple[Mapping[str, Any], ...],
        authoritative_decision: Mapping[str, Any],
        committed_intervention: Mapping[str, Any],
        external_input: Any,
        factual_outcome: Mapping[str, Any],
        counterfactual_evidence: Mapping[str, Any],
        certificate: Mapping[str, Any],
        reward: Mapping[str, Any],
        memory_delta: Mapping[str, Any],
        neural_state_delta: Mapping[str, Any],
        policy_delta: Mapping[str, Any],
        resource_delta: Mapping[str, Any],
        viability_delta: Mapping[str, Any],
        regime_before: Mapping[str, Any],
        regime_after: Mapping[str, Any],
        rollback_refuge_result: Mapping[str, Any],
        storage: Any,
    ) -> LifeTransition:
        self._validate_states(state_before, state_after)
        index = self.transition_index + 1
        identity_payload = {
            "organism_id": self.organism_id,
            "lineage_id": self.lineage_id,
            "transition_index": index,
            "previous_transition_hash": self.head_transition_hash,
            "state_before_hash": state_before.state_hash,
            "state_after_hash": state_after.state_hash,
            "trace_group_id": trace_group_id,
        }
        transition_id = f"life-{canonical_hash(identity_payload)[:32]}"
        payload = {
            "schema_version": LIFE_TRANSITION_SCHEMA_VERSION,
            "transition_id": transition_id,
            "transition_index": index,
            "previous_transition_id": self.head_transition_id,
            "previous_transition_hash": self.head_transition_hash,
            "state_before": state_before,
            "action_proposals": action_proposals,
            "authoritative_decision": dict(authoritative_decision),
            "committed_intervention": dict(committed_intervention),
            "external_input": external_input,
            "factual_outcome": dict(factual_outcome),
            "counterfactual_evidence": dict(counterfactual_evidence),
            "certificate": dict(certificate),
            "reward": dict(reward),
            "memory_delta": dict(memory_delta),
            "neural_state_delta": dict(neural_state_delta),
            "policy_delta": dict(policy_delta),
            "resource_delta": dict(resource_delta),
            "viability_delta": dict(viability_delta),
            "regime_before": dict(regime_before),
            "regime_after": dict(regime_after),
            "rollback_refuge_result": dict(rollback_refuge_result),
            "state_after": state_after,
            "trace_group_id": trace_group_id,
            "status": "committed",
            "reason_code": None,
        }
        hash_payload = {
            key: (value.to_dict() if isinstance(value, OrganismDynamicState) else value)
            for key, value in payload.items()
        }
        transition = LifeTransition(**payload, transition_hash=canonical_hash(hash_payload))
        try:
            persisted_event = storage.append_event(
                event_type="organism.life_transition.committed",
                run_id=self.run_id,
                source="scenario_episode_runner",
                payload=transition.to_dict(),
            )
            if persisted_event is None:
                raise RuntimeError("life_transition_persistence_unconfirmed")
        except Exception:
            incomplete_payload = {
                **hash_payload,
                "status": "incomplete",
                "reason_code": "life_transition_persistence_failed",
            }
            return replace(
                transition,
                transition_hash=canonical_hash(incomplete_payload),
                status="incomplete",
                reason_code="life_transition_persistence_failed",
            )
        self.transition_index = index
        self.head_transition_id = transition.transition_id
        self.head_transition_hash = transition.transition_hash
        self.last_state = state_after
        self.transitions.append(transition)
        if len(self.transitions) > self.max_history:
            del self.transitions[: len(self.transitions) - self.max_history]
        return transition

    def _validate_states(
        self, state_before: OrganismDynamicState, state_after: OrganismDynamicState
    ) -> None:
        for state in (state_before, state_after):
            if state.organism_id != self.organism_id:
                raise ValueError("dynamic_chain_organism_mismatch")
            if state.lineage_id != self.lineage_id:
                raise ValueError("dynamic_chain_lineage_mismatch")
        if state_after.previous_state_hash != state_before.state_hash:
            raise ValueError("dynamic_chain_state_link_mismatch")
        if self.last_state is not None and state_before.previous_state_hash != self.last_state.state_hash:
            raise ValueError("dynamic_chain_previous_state_mismatch")

    def export_checkpoint(self) -> dict[str, Any]:
        return {
            "schema_version": DYNAMIC_CHAIN_CHECKPOINT_VERSION,
            "organism_id": self.organism_id,
            "lineage_id": self.lineage_id,
            "last_run_id": self.run_id,
            "transition_index": self.transition_index,
            "last_transition_id": self.head_transition_id,
            "last_transition_hash": self.head_transition_hash,
            "last_dynamic_state": self.last_state.to_dict() if self.last_state else None,
            "chain_epoch": self.chain_epoch,
            "restore_reason": self.restore_reason,
            "recent_transitions": [item.to_dict() for item in self.transitions],
            "contract_versions": {
                "dynamic_state": "organism-dynamic-state-v1",
                "life_transition": LIFE_TRANSITION_SCHEMA_VERSION,
            },
        }

    def restore_checkpoint(self, raw: Mapping[str, Any] | None) -> dict[str, Any]:
        data = dict(raw or {})
        if data.get("schema_version") != DYNAMIC_CHAIN_CHECKPOINT_VERSION:
            self.chain_epoch += 1
            self.restore_reason = "legacy_checkpoint_without_dynamic_chain"
            return {"restored": False, "reason": self.restore_reason, "chain_epoch": self.chain_epoch}
        if str(data.get("organism_id")) != self.organism_id:
            raise ValueError("dynamic_chain_checkpoint_organism_mismatch")
        if str(data.get("lineage_id")) != self.lineage_id:
            raise ValueError("dynamic_chain_checkpoint_lineage_mismatch")
        state_raw = data.get("last_dynamic_state")
        state = OrganismDynamicState.from_dict(state_raw) if isinstance(state_raw, Mapping) else None
        self.transition_index = int(data.get("transition_index", 0))
        self.head_transition_id = data.get("last_transition_id")
        self.head_transition_hash = str(data.get("last_transition_hash") or "")
        if not self.head_transition_hash:
            raise ValueError("dynamic_chain_checkpoint_head_missing")
        self.last_state = state
        self.chain_epoch = int(data.get("chain_epoch", 1))
        self.restore_reason = None
        return {"restored": True, "reason": None, "chain_epoch": self.chain_epoch}
