from __future__ import annotations

from runtime.experiments.cognitive_gain.contracts import LedgerWriteStatus
from runtime.experiments.cognitive_gain.ledger import InMemoryCausalLedger


def test_genesis_and_second_event_form_hash_chain(event_factory):
    ledger = InMemoryCausalLedger()
    first_payload = {"episode": 0}
    first = event_factory(first_payload)
    result = ledger.append(first, first_payload)
    assert result.status is LedgerWriteStatus.APPENDED
    assert result.chain_state.head_sequence == 0

    second_payload = {"episode": 1}
    second = event_factory(
        second_payload,
        event_id="event-1",
        sequence=1,
        previous_hash=first.event_hash,
        logical_time=1,
    )
    result = ledger.append(second, second_payload)
    assert result.status is LedgerWriteStatus.APPENDED
    assert result.chain_state.head_event_hash == second.event_hash


def test_sequence_jump_and_repeated_sequence_are_rejected(event_factory):
    ledger = InMemoryCausalLedger()
    first = event_factory({"v": 0})
    assert ledger.append(first, {"v": 0}).status is LedgerWriteStatus.APPENDED
    jump = event_factory(
        {"v": 2}, event_id="jump", sequence=2,
        previous_hash=first.event_hash, logical_time=2,
    )
    repeat = event_factory(
        {"v": 1}, event_id="repeat", sequence=0,
        logical_time=0,
    )
    assert ledger.append(jump, {"v": 2}).status is LedgerWriteStatus.REJECTED_INVALID_SEQUENCE
    assert ledger.append(repeat, {"v": 1}).status is LedgerWriteStatus.REJECTED_INVALID_SEQUENCE


def test_bad_previous_hash_is_rejected(event_factory):
    ledger = InMemoryCausalLedger()
    first = event_factory({"v": 0})
    ledger.append(first, {"v": 0})
    bad = event_factory(
        {"v": 1}, event_id="event-1", sequence=1,
        previous_hash="b" * 64, logical_time=1,
    )
    assert ledger.append(bad, {"v": 1}).status is LedgerWriteStatus.REJECTED_INVALID_PREVIOUS_HASH


def test_same_event_and_payload_is_idempotent(event_factory):
    ledger = InMemoryCausalLedger()
    event = event_factory({"v": 0})
    ledger.append(event, {"v": 0})
    replay = ledger.append(event, {"v": 0})
    assert replay.status is LedgerWriteStatus.IDEMPOTENT_REPLAY
    assert replay.chain_state.head_sequence == 0
    assert ledger.conflicts("experiment-a", "shadow", "chain-a") == ()


def test_divergence_is_recorded_and_chain_fails_closed(event_factory):
    ledger = InMemoryCausalLedger()
    stored = event_factory({"v": 0})
    ledger.append(stored, {"v": 0})
    divergent = event_factory({"v": 9})
    result = ledger.append(divergent, {"v": 9})
    assert result.status is LedgerWriteStatus.DIVERGENCE_QUARANTINED
    assert result.chain_state.quarantined
    assert result.quarantine_record.stored_payload_hash == stored.payload_hash
    assert result.quarantine_record.incoming_payload_hash == divergent.payload_hash
    assert ledger.get_event("experiment-a", "shadow", "chain-a", "event-0") == stored
    assert len(ledger.conflicts("experiment-a", "shadow", "chain-a")) == 1

    later = event_factory(
        {"v": 1}, event_id="event-1", sequence=1,
        previous_hash=stored.event_hash, logical_time=1,
    )
    closed = ledger.append(later, {"v": 1})
    assert closed.status is LedgerWriteStatus.CHAIN_ALREADY_QUARANTINED
    assert closed.quarantine_record == result.quarantine_record


def test_payload_hash_mismatch_is_invalid(event_factory):
    ledger = InMemoryCausalLedger()
    event = event_factory({"v": 0})
    result = ledger.append(event, {"v": 1})
    assert result.status is LedgerWriteStatus.REJECTED_INVALID_EVENT
    assert ledger.state("experiment-a", "shadow", "chain-a").head_sequence == -1


def test_experiment_arm_and_chain_are_independent(event_factory):
    ledger = InMemoryCausalLedger()
    dimensions = [
        ("experiment-a", "shadow", "chain-a"),
        ("experiment-b", "shadow", "chain-a"),
        ("experiment-a", "off", "chain-a"),
        ("experiment-a", "shadow", "chain-b"),
    ]
    for index, (experiment, arm, chain) in enumerate(dimensions):
        payload = {"index": index}
        event = event_factory(
            payload,
            event_id=f"event-{index}",
            experiment_id=experiment,
            arm_id=arm,
            chain_id=chain,
        )
        assert ledger.append(event, payload).status is LedgerWriteStatus.APPENDED
    for dimension in dimensions:
        assert ledger.state(*dimension).head_sequence == 0


def test_wall_clock_does_not_order_events(event_factory):
    ledger = InMemoryCausalLedger()
    first = event_factory({"v": 0}, wall_clock_time="2099-01-01T00:00:00Z")
    ledger.append(first, {"v": 0})
    second = event_factory(
        {"v": 1}, event_id="event-1", sequence=1,
        previous_hash=first.event_hash, logical_time=1,
        wall_clock_time="2000-01-01T00:00:00Z",
    )
    assert ledger.append(second, {"v": 1}).status is LedgerWriteStatus.APPENDED
