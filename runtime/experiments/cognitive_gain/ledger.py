"""Pure append-only causal ledger used to validate ASCG invariants."""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from .canonical import canonical_json, canonical_sha256
from .contracts import (
    GENESIS_PREVIOUS_EVENT_HASH,
    CausalLedgerEvent,
    ExperimentalQuarantineRecord,
    LedgerChainState,
    LedgerWriteResult,
    LedgerWriteStatus,
    QuarantineReason,
)


ChainKey = tuple[str, str, str]


@dataclass(slots=True)
class _StoredEntry:
    event: CausalLedgerEvent
    payload_json: str


@dataclass(slots=True)
class _MutableChain:
    head_sequence: int = -1
    head_event_hash: str = GENESIS_PREVIOUS_EVENT_HASH
    head_logical_time: int = -1
    quarantined: bool = False
    quarantine_id: str | None = None
    events_by_id: dict[str, _StoredEntry] = field(default_factory=dict)
    events_by_sequence: dict[int, _StoredEntry] = field(default_factory=dict)
    conflicts: list[ExperimentalQuarantineRecord] = field(default_factory=list)


def chain_key(event: CausalLedgerEvent) -> ChainKey:
    return (event.experiment_id, event.arm_id, event.chain_id)


def empty_chain_state(experiment_id: str, arm_id: str, chain_id: str) -> LedgerChainState:
    return LedgerChainState(
        experiment_id=experiment_id,
        arm_id=arm_id,
        chain_id=chain_id,
        head_sequence=-1,
        head_event_hash=GENESIS_PREVIOUS_EVENT_HASH,
        head_logical_time=-1,
    )


def quarantine_for_divergence(
    *,
    incoming: CausalLedgerEvent,
    stored: CausalLedgerEvent,
) -> ExperimentalQuarantineRecord:
    evidence = {
        "experiment_id": incoming.experiment_id,
        "arm_id": incoming.arm_id,
        "chain_id": incoming.chain_id,
        "event_id": incoming.event_id,
        "stored_payload_hash": stored.payload_hash,
        "incoming_payload_hash": incoming.payload_hash,
        "stored_event_hash": stored.event_hash,
        "incoming_event_hash": incoming.event_hash,
    }
    evidence_hash = canonical_sha256(evidence)
    quarantine_id = "cgq-" + canonical_sha256(
        {
            "chain": list(chain_key(incoming)),
            "event_id": incoming.event_id,
            "stored_payload_hash": stored.payload_hash,
            "incoming_payload_hash": incoming.payload_hash,
        }
    )
    return ExperimentalQuarantineRecord(
        quarantine_id=quarantine_id,
        experiment_id=incoming.experiment_id,
        arm_id=incoming.arm_id,
        chain_id=incoming.chain_id,
        reason=QuarantineReason.DIVERGENT_EVENT_ID,
        conflicting_event_id=incoming.event_id,
        stored_payload_hash=stored.payload_hash,
        incoming_payload_hash=incoming.payload_hash,
        observed_sequence=stored.event_sequence,
        evidence_hash=evidence_hash,
        logical_time=max(stored.logical_time, incoming.logical_time),
    )


class InMemoryCausalLedger:
    """Thread-safe reference implementation with no external side effects."""

    def __init__(self) -> None:
        self._chains: dict[ChainKey, _MutableChain] = {}
        self._lock = RLock()

    def append(self, event: CausalLedgerEvent, payload: Any) -> LedgerWriteResult:
        if not isinstance(event, CausalLedgerEvent):
            return LedgerWriteResult(
                status=LedgerWriteStatus.REJECTED_INVALID_EVENT,
                stored_event=None,
                chain_state=None,
                quarantine_record=None,
                detail="event_contract_invalid",
            )
        try:
            payload_json = canonical_json(payload)
            payload_hash = canonical_sha256(payload)
        except ValueError as exc:
            return LedgerWriteResult(
                status=LedgerWriteStatus.REJECTED_INVALID_EVENT,
                stored_event=None,
                chain_state=None,
                quarantine_record=None,
                detail=str(exc),
            )
        if payload_hash != event.payload_hash:
            return LedgerWriteResult(
                status=LedgerWriteStatus.REJECTED_INVALID_EVENT,
                stored_event=None,
                chain_state=self.state(*chain_key(event)),
                quarantine_record=None,
                detail="payload_hash_mismatch",
            )

        key = chain_key(event)
        with self._lock:
            chain = self._chains.get(key)
            state = self._state_from(key, chain)
            if chain is not None and chain.quarantined:
                return LedgerWriteResult(
                    status=LedgerWriteStatus.CHAIN_ALREADY_QUARANTINED,
                    stored_event=None,
                    chain_state=state,
                    quarantine_record=chain.conflicts[-1] if chain.conflicts else None,
                )

            existing = chain.events_by_id.get(event.event_id) if chain is not None else None
            if existing is not None:
                if existing.event.payload_hash == event.payload_hash:
                    return LedgerWriteResult(
                        status=LedgerWriteStatus.IDEMPOTENT_REPLAY,
                        stored_event=existing.event,
                        chain_state=state,
                        quarantine_record=None,
                    )
                record = quarantine_for_divergence(incoming=event, stored=existing.event)
                chain.quarantined = True
                chain.quarantine_id = record.quarantine_id
                chain.conflicts.append(record)
                return LedgerWriteResult(
                    status=LedgerWriteStatus.DIVERGENCE_QUARANTINED,
                    stored_event=existing.event,
                    chain_state=self._state_from(key, chain),
                    quarantine_record=record,
                )

            head_sequence = chain.head_sequence if chain is not None else -1
            head_logical_time = chain.head_logical_time if chain is not None else -1
            head_event_hash = chain.head_event_hash if chain is not None else GENESIS_PREVIOUS_EVENT_HASH
            if event.event_sequence != head_sequence + 1 or event.logical_time <= head_logical_time:
                return LedgerWriteResult(
                    status=LedgerWriteStatus.REJECTED_INVALID_SEQUENCE,
                    stored_event=None,
                    chain_state=state,
                    quarantine_record=None,
                )
            if event.previous_event_hash != head_event_hash:
                return LedgerWriteResult(
                    status=LedgerWriteStatus.REJECTED_INVALID_PREVIOUS_HASH,
                    stored_event=None,
                    chain_state=state,
                    quarantine_record=None,
                )

            if chain is None:
                chain = _MutableChain()
                self._chains[key] = chain
            entry = _StoredEntry(event=event, payload_json=payload_json)
            chain.events_by_id[event.event_id] = entry
            chain.events_by_sequence[event.event_sequence] = entry
            chain.head_sequence = event.event_sequence
            chain.head_event_hash = event.event_hash
            chain.head_logical_time = event.logical_time
            return LedgerWriteResult(
                status=LedgerWriteStatus.APPENDED,
                stored_event=event,
                chain_state=self._state_from(key, chain),
                quarantine_record=None,
            )

    def state(self, experiment_id: str, arm_id: str, chain_id: str) -> LedgerChainState:
        key = (experiment_id, arm_id, chain_id)
        with self._lock:
            return self._state_from(key, self._chains.get(key))

    def get_event(
        self,
        experiment_id: str,
        arm_id: str,
        chain_id: str,
        event_id: str,
    ) -> CausalLedgerEvent | None:
        with self._lock:
            chain = self._chains.get((experiment_id, arm_id, chain_id))
            entry = chain.events_by_id.get(event_id) if chain is not None else None
            return entry.event if entry is not None else None

    def get_payload_json(
        self,
        experiment_id: str,
        arm_id: str,
        chain_id: str,
        event_id: str,
    ) -> str | None:
        with self._lock:
            chain = self._chains.get((experiment_id, arm_id, chain_id))
            entry = chain.events_by_id.get(event_id) if chain is not None else None
            return entry.payload_json if entry is not None else None

    def conflicts(
        self,
        experiment_id: str,
        arm_id: str,
        chain_id: str,
    ) -> tuple[ExperimentalQuarantineRecord, ...]:
        with self._lock:
            chain = self._chains.get((experiment_id, arm_id, chain_id))
            return tuple(chain.conflicts) if chain is not None else ()

    @staticmethod
    def _state_from(key: ChainKey, chain: _MutableChain | None) -> LedgerChainState:
        if chain is None:
            return empty_chain_state(*key)
        return LedgerChainState(
            experiment_id=key[0],
            arm_id=key[1],
            chain_id=key[2],
            head_sequence=chain.head_sequence,
            head_event_hash=chain.head_event_hash,
            head_logical_time=chain.head_logical_time,
            quarantined=chain.quarantined,
            quarantine_id=chain.quarantine_id,
        )


__all__ = [
    "InMemoryCausalLedger",
    "chain_key",
    "empty_chain_state",
    "quarantine_for_divergence",
]
