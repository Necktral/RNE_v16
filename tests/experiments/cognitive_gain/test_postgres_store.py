from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from runtime.experiments.cognitive_gain.contracts import LedgerWriteStatus
from runtime.experiments.cognitive_gain.postgres_store import PostgresCausalLedgerStore


pytestmark = pytest.mark.requires_postgres


def test_round_trip_persists_event_and_header(isolated_postgres_store, event_factory):
    store, _, _ = isolated_postgres_store
    payload = {"episode": 0, "semantic": "causal"}
    event = event_factory(payload)
    result = store.append(event, payload)
    assert result.status is LedgerWriteStatus.APPENDED
    assert store.get_event("experiment-a", "shadow", "chain-a", "event-0") == event
    state = store.get_chain_state("experiment-a", "shadow", "chain-a")
    assert (state.head_sequence, state.head_event_hash, state.head_logical_time) == (0, event.event_hash, 0)


def test_idempotent_replay_across_two_connections(isolated_postgres_store, event_factory):
    store, dsn, schema = isolated_postgres_store
    payload = {"episode": 0}
    event = event_factory(payload)
    second = PostgresCausalLedgerStore(dsn=dsn, schema=schema)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda backend: backend.append(event, payload), (store, second)))
    assert sorted(result.status.value for result in results) == sorted(
        [LedgerWriteStatus.APPENDED.value, LedgerWriteStatus.IDEMPOTENT_REPLAY.value]
    )
    assert store.get_chain_state("experiment-a", "shadow", "chain-a").head_sequence == 0


def test_divergence_is_durable_and_closes_chain(isolated_postgres_store, event_factory):
    store, dsn, schema = isolated_postgres_store
    original = event_factory({"value": 1})
    assert store.append(original, {"value": 1}).status is LedgerWriteStatus.APPENDED
    divergent = event_factory({"value": 2})
    result = PostgresCausalLedgerStore(dsn=dsn, schema=schema).append(divergent, {"value": 2})
    assert result.status is LedgerWriteStatus.DIVERGENCE_QUARANTINED
    assert result.chain_state.quarantined

    reopened = PostgresCausalLedgerStore(dsn=dsn, schema=schema)
    state = reopened.get_chain_state("experiment-a", "shadow", "chain-a")
    conflicts = reopened.conflicts("experiment-a", "shadow", "chain-a")
    assert state.quarantined and state.quarantine_id == result.quarantine_record.quarantine_id
    assert len(conflicts) == 1
    assert conflicts[0].stored_payload_hash == original.payload_hash
    assert conflicts[0].incoming_payload_hash == divergent.payload_hash
    later = event_factory(
        {"value": 3}, event_id="event-1", sequence=1,
        previous_hash=original.event_hash, logical_time=1,
    )
    assert reopened.append(later, {"value": 3}).status is LedgerWriteStatus.CHAIN_ALREADY_QUARANTINED


def test_divergent_same_id_is_serialized_across_connections(isolated_postgres_store, event_factory):
    store, dsn, schema = isolated_postgres_store
    stores = (store, PostgresCausalLedgerStore(dsn=dsn, schema=schema))
    events = (event_factory({"winner": "a"}), event_factory({"winner": "b"}))
    payloads = ({"winner": "a"}, {"winner": "b"})
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(backend.append, event, payload) for backend, event, payload in zip(stores, events, payloads)]
        results = [future.result() for future in futures]
    assert {result.status for result in results} == {
        LedgerWriteStatus.APPENDED,
        LedgerWriteStatus.DIVERGENCE_QUARANTINED,
    }
    assert store.get_chain_state("experiment-a", "shadow", "chain-a").quarantined
    assert len(store.conflicts("experiment-a", "shadow", "chain-a")) == 1


def test_same_sequence_different_ids_has_single_winner(isolated_postgres_store, event_factory):
    store, dsn, schema = isolated_postgres_store
    stores = (store, PostgresCausalLedgerStore(dsn=dsn, schema=schema))
    entries = (
        (event_factory({"id": "a"}, event_id="event-a"), {"id": "a"}),
        (event_factory({"id": "b"}, event_id="event-b"), {"id": "b"}),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result() for future in [
            pool.submit(stores[index].append, event, payload)
            for index, (event, payload) in enumerate(entries)
        ]]
    assert sorted(result.status.value for result in results) == sorted(
        [LedgerWriteStatus.APPENDED.value, LedgerWriteStatus.REJECTED_INVALID_SEQUENCE.value]
    )
    assert store.get_chain_state("experiment-a", "shadow", "chain-a").head_sequence == 0


def test_different_chains_advance_in_parallel(isolated_postgres_store, event_factory):
    store, dsn, schema = isolated_postgres_store
    stores = (store, PostgresCausalLedgerStore(dsn=dsn, schema=schema))
    entries = tuple(
        (
            event_factory({"chain": name}, event_id=f"event-{name}", chain_id=name),
            {"chain": name},
        )
        for name in ("chain-a", "chain-b")
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [future.result() for future in [
            pool.submit(stores[index].append, event, payload)
            for index, (event, payload) in enumerate(entries)
        ]]
    assert all(result.status is LedgerWriteStatus.APPENDED for result in results)


def test_constraints_reject_direct_duplicates(isolated_postgres_store, event_factory):
    import psycopg
    from psycopg import sql

    store, dsn, schema = isolated_postgres_store
    payload = {"episode": 0}
    event = event_factory(payload)
    store.append(event, payload)
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(schema)))
        with pytest.raises(psycopg.errors.UniqueViolation):
            cur.execute(
                """
                INSERT INTO rnfe_cg_ledger_events
                    (schema_version, experiment_id, arm_id, chain_id, event_id,
                     event_type, event_sequence, previous_event_hash, logical_time,
                     wall_clock_time, payload_contract_type, payload_hash, event_hash,
                     payload_jsonb)
                SELECT schema_version, experiment_id, arm_id, chain_id, event_id,
                       event_type, event_sequence, previous_event_hash, logical_time,
                       wall_clock_time, payload_contract_type, payload_hash, event_hash,
                       payload_jsonb
                FROM rnfe_cg_ledger_events
                WHERE event_id = %s
                """,
                (event.event_id,),
            )


def test_failed_insert_rolls_back_new_header(isolated_postgres_store, event_factory):
    import psycopg
    from psycopg import sql

    store, dsn, schema = isolated_postgres_store
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(schema)))
        cur.execute("""
            CREATE FUNCTION reject_cg_event() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN RAISE EXCEPTION 'forced test rollback'; END $$
        """)
        cur.execute("""
            CREATE TRIGGER reject_cg_event_trigger BEFORE INSERT ON rnfe_cg_ledger_events
            FOR EACH ROW EXECUTE FUNCTION reject_cg_event()
        """)
    payload = {"rollback": True}
    event = event_factory(payload, experiment_id="rollback-experiment")
    result = store.append(event, payload)
    assert result.status is LedgerWriteStatus.REJECTED_INVALID_EVENT
    assert store.get_chain_state("rollback-experiment", "shadow", "chain-a").head_sequence == -1


def test_random_schema_is_isolated_and_removed(event_factory):
    import os
    import psycopg
    from psycopg import sql

    dsn = os.environ.get("RNFE_CG_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("RNFE_CG_POSTGRES_TEST_DSN is not set")
    schema = f"rnfe_cg_cleanup_{uuid4().hex}"
    store = PostgresCausalLedgerStore(dsn=dsn, schema=schema)
    store.initialize(create_schema=True)
    try:
        event = event_factory({"cleanup": True})
        assert store.append(event, {"cleanup": True}).status is LedgerWriteStatus.APPENDED
        with psycopg.connect(dsn) as conn:
            assert conn.execute("SELECT to_regnamespace(%s)", (schema,)).fetchone()[0] == schema
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
    with psycopg.connect(dsn) as conn:
        assert conn.execute("SELECT to_regnamespace(%s)", (schema,)).fetchone()[0] is None
