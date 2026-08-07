from __future__ import annotations

import os
from uuid import uuid4

import pytest

from runtime.experiments.cognitive_gain.contracts import (
    GENESIS_PREVIOUS_EVENT_HASH,
    CausalLedgerEvent,
    LedgerEventType,
)


# The repository-wide pytest guard predates the isolated cognitive-gain DSN and
# checks this compatibility name.  The store and fixtures still read only the
# dedicated RNFE_CG_POSTGRES_TEST_DSN value.
if os.environ.get("RNFE_RUN_PG_TESTS") == "1" and os.environ.get("RNFE_CG_POSTGRES_TEST_DSN"):
    os.environ.setdefault("RNFE_POSTGRES_DSN", os.environ["RNFE_CG_POSTGRES_TEST_DSN"])


@pytest.fixture
def event_factory():
    def make(
        payload,
        *,
        event_id="event-0",
        experiment_id="experiment-a",
        arm_id="shadow",
        chain_id="chain-a",
        sequence=0,
        previous_hash=GENESIS_PREVIOUS_EVENT_HASH,
        logical_time=0,
        wall_clock_time="2026-08-03T00:00:00Z",
    ):
        return CausalLedgerEvent.create(
            payload=payload,
            event_id=event_id,
            event_type=LedgerEventType.EXPERIENCE_OBSERVED,
            experiment_id=experiment_id,
            arm_id=arm_id,
            chain_id=chain_id,
            event_sequence=sequence,
            previous_event_hash=previous_hash,
            logical_time=logical_time,
            wall_clock_time=wall_clock_time,
            payload_contract_type="CausalEpisodeEvidence",
        )

    return make


@pytest.fixture
def isolated_postgres_store():
    dsn = os.environ.get("RNFE_CG_POSTGRES_TEST_DSN")
    if not dsn:
        pytest.skip("RNFE_CG_POSTGRES_TEST_DSN is not set")

    import psycopg
    from psycopg import sql

    from runtime.experiments.cognitive_gain.postgres_store import PostgresCausalLedgerStore

    schema = f"rnfe_cg_test_{uuid4().hex}"
    store = PostgresCausalLedgerStore(dsn=dsn, schema=schema)
    store.initialize(create_schema=True)
    try:
        yield store, dsn, schema
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
