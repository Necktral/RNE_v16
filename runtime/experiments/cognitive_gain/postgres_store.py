"""Dedicated PostgreSQL store for the isolated ASCG causal ledger."""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .canonical import canonical_json, canonical_sha256
from .contracts import (
    GENESIS_PREVIOUS_EVENT_HASH,
    CausalLedgerEvent,
    ExperimentalQuarantineRecord,
    LedgerChainState,
    LedgerEventType,
    LedgerWriteResult,
    LedgerWriteStatus,
    QuarantineReason,
    contract_to_payload,
)
from .ledger import chain_key, empty_chain_state, quarantine_for_divergence


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _RejectedTransaction(Exception):
    def __init__(self, result: LedgerWriteResult):
        super().__init__(result.status.value)
        self.result = result


class PostgresCausalLedgerStore:
    """Transactional ledger isolated from RNFE's canonical ``ledger_events``."""

    def __init__(self, *, dsn: str, schema: str) -> None:
        if not isinstance(dsn, str) or not dsn.strip():
            raise ValueError("dsn_required")
        if not isinstance(schema, str) or not _IDENTIFIER_RE.fullmatch(schema):
            raise ValueError("schema_identifier_invalid")
        self._dsn = dsn
        self.schema = schema
        self._schema_path = Path(__file__).with_name("schema.sql")

    def initialize(self, *, create_schema: bool = False) -> None:
        """Explicitly create/apply the experimental schema. Never runs on import."""

        ddl = self._schema_path.read_text(encoding="utf-8")
        with self._connect() as conn, conn.cursor() as cur:
            if create_schema:
                cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema)))
            self._set_search_path(cur)
            cur.execute(ddl)

    def append(self, event: CausalLedgerEvent, payload: Any) -> LedgerWriteResult:
        if not isinstance(event, CausalLedgerEvent):
            return self._invalid("event_contract_invalid")
        try:
            payload_json = canonical_json(payload)
            payload_hash = canonical_sha256(payload)
        except ValueError as exc:
            return self._invalid(str(exc))
        if payload_hash != event.payload_hash:
            return self._invalid("payload_hash_mismatch")

        try:
            with self._connect() as conn, conn.cursor() as cur:
                self._set_search_path(cur)
                cur.execute(
                    """
                    INSERT INTO rnfe_cg_ledger_chains
                        (experiment_id, arm_id, chain_id)
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    chain_key(event),
                )
                cur.execute(
                    """
                    SELECT experiment_id, arm_id, chain_id, head_sequence,
                           head_event_hash, head_logical_time, quarantined,
                           quarantine_id
                    FROM rnfe_cg_ledger_chains
                    WHERE experiment_id = %s AND arm_id = %s AND chain_id = %s
                    FOR UPDATE
                    """,
                    chain_key(event),
                )
                head_row = cur.fetchone()
                if head_row is None:
                    raise RuntimeError("chain_header_missing_after_insert")
                state = self._state_from_row(head_row)
                if state.quarantined:
                    return LedgerWriteResult(
                        status=LedgerWriteStatus.CHAIN_ALREADY_QUARANTINED,
                        stored_event=None,
                        chain_state=state,
                        quarantine_record=self._get_quarantine_cursor(cur, state.quarantine_id),
                    )

                cur.execute(
                    """
                    SELECT schema_version, event_id, event_type, experiment_id,
                           arm_id, chain_id, event_sequence, previous_event_hash,
                           logical_time, wall_clock_time, payload_contract_type,
                           payload_hash, event_hash
                    FROM rnfe_cg_ledger_events
                    WHERE experiment_id = %s AND arm_id = %s AND chain_id = %s
                      AND event_id = %s
                    """,
                    (*chain_key(event), event.event_id),
                )
                existing_row = cur.fetchone()
                if existing_row is not None:
                    stored = self._event_from_row(existing_row)
                    if stored.payload_hash == event.payload_hash:
                        return LedgerWriteResult(
                            status=LedgerWriteStatus.IDEMPOTENT_REPLAY,
                            stored_event=stored,
                            chain_state=state,
                            quarantine_record=None,
                        )
                    record = quarantine_for_divergence(incoming=event, stored=stored)
                    cur.execute(
                        """
                        INSERT INTO rnfe_cg_ledger_conflicts
                            (quarantine_id, experiment_id, arm_id, chain_id,
                             reason, conflicting_event_id, stored_payload_hash,
                             incoming_payload_hash, observed_sequence, evidence_hash,
                             logical_time, quarantine_hash, record_jsonb)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (
                            record.quarantine_id,
                            record.experiment_id,
                            record.arm_id,
                            record.chain_id,
                            record.reason.value,
                            record.conflicting_event_id,
                            record.stored_payload_hash,
                            record.incoming_payload_hash,
                            record.observed_sequence,
                            record.evidence_hash,
                            record.logical_time,
                            record.quarantine_hash,
                            Jsonb(contract_to_payload(record)),
                        ),
                    )
                    cur.execute(
                        """
                        UPDATE rnfe_cg_ledger_chains
                        SET quarantined = TRUE, quarantine_id = %s, updated_at = now()
                        WHERE experiment_id = %s AND arm_id = %s AND chain_id = %s
                        """,
                        (record.quarantine_id, *chain_key(event)),
                    )
                    quarantined_state = LedgerChainState(
                        experiment_id=state.experiment_id,
                        arm_id=state.arm_id,
                        chain_id=state.chain_id,
                        head_sequence=state.head_sequence,
                        head_event_hash=state.head_event_hash,
                        head_logical_time=state.head_logical_time,
                        quarantined=True,
                        quarantine_id=record.quarantine_id,
                    )
                    return LedgerWriteResult(
                        status=LedgerWriteStatus.DIVERGENCE_QUARANTINED,
                        stored_event=stored,
                        chain_state=quarantined_state,
                        quarantine_record=record,
                    )

                if event.event_sequence != state.head_sequence + 1 or event.logical_time <= state.head_logical_time:
                    raise _RejectedTransaction(
                        LedgerWriteResult(
                            status=LedgerWriteStatus.REJECTED_INVALID_SEQUENCE,
                            stored_event=None,
                            chain_state=state,
                            quarantine_record=None,
                        )
                    )
                if event.previous_event_hash != state.head_event_hash:
                    raise _RejectedTransaction(
                        LedgerWriteResult(
                            status=LedgerWriteStatus.REJECTED_INVALID_PREVIOUS_HASH,
                            stored_event=None,
                            chain_state=state,
                            quarantine_record=None,
                        )
                    )

                cur.execute(
                    """
                    INSERT INTO rnfe_cg_ledger_events
                        (schema_version, experiment_id, arm_id, chain_id, event_id,
                         event_type, event_sequence, previous_event_hash, logical_time,
                         wall_clock_time, payload_contract_type, payload_hash,
                         event_hash, payload_jsonb)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        event.schema_version,
                        event.experiment_id,
                        event.arm_id,
                        event.chain_id,
                        event.event_id,
                        event.event_type.value,
                        event.event_sequence,
                        event.previous_event_hash,
                        event.logical_time,
                        event.wall_clock_time,
                        event.payload_contract_type,
                        event.payload_hash,
                        event.event_hash,
                        Jsonb(json.loads(payload_json)),
                    ),
                )
                cur.execute(
                    """
                    UPDATE rnfe_cg_ledger_chains
                    SET head_sequence = %s, head_event_hash = %s,
                        head_logical_time = %s, updated_at = now()
                    WHERE experiment_id = %s AND arm_id = %s AND chain_id = %s
                    """,
                    (
                        event.event_sequence,
                        event.event_hash,
                        event.logical_time,
                        *chain_key(event),
                    ),
                )
                appended_state = LedgerChainState(
                    experiment_id=event.experiment_id,
                    arm_id=event.arm_id,
                    chain_id=event.chain_id,
                    head_sequence=event.event_sequence,
                    head_event_hash=event.event_hash,
                    head_logical_time=event.logical_time,
                )
                return LedgerWriteResult(
                    status=LedgerWriteStatus.APPENDED,
                    stored_event=event,
                    chain_state=appended_state,
                    quarantine_record=None,
                )
        except _RejectedTransaction as rejected:
            return rejected.result
        except Exception as exc:
            return self._invalid(f"persistence_error:{type(exc).__name__}")

    def get_chain_state(self, experiment_id: str, arm_id: str, chain_id: str) -> LedgerChainState:
        with self._connect() as conn, conn.cursor() as cur:
            self._set_search_path(cur)
            cur.execute(
                """
                SELECT experiment_id, arm_id, chain_id, head_sequence,
                       head_event_hash, head_logical_time, quarantined, quarantine_id
                FROM rnfe_cg_ledger_chains
                WHERE experiment_id = %s AND arm_id = %s AND chain_id = %s
                """,
                (experiment_id, arm_id, chain_id),
            )
            row = cur.fetchone()
        return self._state_from_row(row) if row else empty_chain_state(experiment_id, arm_id, chain_id)

    def get_event(
        self,
        experiment_id: str,
        arm_id: str,
        chain_id: str,
        event_id: str,
    ) -> CausalLedgerEvent | None:
        with self._connect() as conn, conn.cursor() as cur:
            self._set_search_path(cur)
            cur.execute(
                """
                SELECT schema_version, event_id, event_type, experiment_id,
                       arm_id, chain_id, event_sequence, previous_event_hash,
                       logical_time, wall_clock_time, payload_contract_type,
                       payload_hash, event_hash
                FROM rnfe_cg_ledger_events
                WHERE experiment_id = %s AND arm_id = %s AND chain_id = %s
                  AND event_id = %s
                """,
                (experiment_id, arm_id, chain_id, event_id),
            )
            row = cur.fetchone()
        return self._event_from_row(row) if row else None

    def conflicts(
        self,
        experiment_id: str,
        arm_id: str,
        chain_id: str,
    ) -> tuple[ExperimentalQuarantineRecord, ...]:
        with self._connect() as conn, conn.cursor() as cur:
            self._set_search_path(cur)
            cur.execute(
                """
                SELECT quarantine_id, experiment_id, arm_id, chain_id, reason,
                       conflicting_event_id, stored_payload_hash, incoming_payload_hash,
                       observed_sequence, evidence_hash, logical_time, quarantine_hash
                FROM rnfe_cg_ledger_conflicts
                WHERE experiment_id = %s AND arm_id = %s AND chain_id = %s
                ORDER BY created_at, quarantine_id
                """,
                (experiment_id, arm_id, chain_id),
            )
            rows = cur.fetchall()
        return tuple(self._quarantine_from_row(row) for row in rows)

    def _connect(self):
        return psycopg.connect(self._dsn, row_factory=dict_row)

    def _set_search_path(self, cur) -> None:
        cur.execute(sql.SQL("SET LOCAL search_path TO {}").format(sql.Identifier(self.schema)))

    @staticmethod
    def _invalid(detail: str) -> LedgerWriteResult:
        return LedgerWriteResult(
            status=LedgerWriteStatus.REJECTED_INVALID_EVENT,
            stored_event=None,
            chain_state=None,
            quarantine_record=None,
            detail=detail,
        )

    @staticmethod
    def _state_from_row(row: dict[str, Any]) -> LedgerChainState:
        return LedgerChainState(
            experiment_id=row["experiment_id"],
            arm_id=row["arm_id"],
            chain_id=row["chain_id"],
            head_sequence=int(row["head_sequence"]),
            head_event_hash=row["head_event_hash"],
            head_logical_time=int(row["head_logical_time"]),
            quarantined=bool(row["quarantined"]),
            quarantine_id=row["quarantine_id"],
        )

    @staticmethod
    def _event_from_row(row: dict[str, Any]) -> CausalLedgerEvent:
        event = CausalLedgerEvent(
            schema_version=row["schema_version"],
            event_id=row["event_id"],
            event_type=LedgerEventType(row["event_type"]),
            experiment_id=row["experiment_id"],
            arm_id=row["arm_id"],
            chain_id=row["chain_id"],
            event_sequence=int(row["event_sequence"]),
            previous_event_hash=row["previous_event_hash"],
            logical_time=int(row["logical_time"]),
            wall_clock_time=row["wall_clock_time"],
            payload_contract_type=row["payload_contract_type"],
            payload_hash=row["payload_hash"],
        )
        if event.event_hash != row["event_hash"]:
            raise ValueError("persisted_event_hash_mismatch")
        return event

    @staticmethod
    def _quarantine_from_row(row: dict[str, Any]) -> ExperimentalQuarantineRecord:
        record = ExperimentalQuarantineRecord(
            quarantine_id=row["quarantine_id"],
            experiment_id=row["experiment_id"],
            arm_id=row["arm_id"],
            chain_id=row["chain_id"],
            reason=QuarantineReason(row["reason"]),
            conflicting_event_id=row["conflicting_event_id"],
            stored_payload_hash=row["stored_payload_hash"],
            incoming_payload_hash=row["incoming_payload_hash"],
            observed_sequence=int(row["observed_sequence"]),
            evidence_hash=row["evidence_hash"],
            logical_time=int(row["logical_time"]),
        )
        if record.quarantine_hash != row["quarantine_hash"]:
            raise ValueError("persisted_quarantine_hash_mismatch")
        return record

    def _get_quarantine_cursor(self, cur, quarantine_id: str | None) -> ExperimentalQuarantineRecord | None:
        if quarantine_id is None:
            return None
        cur.execute(
            """
            SELECT quarantine_id, experiment_id, arm_id, chain_id, reason,
                   conflicting_event_id, stored_payload_hash, incoming_payload_hash,
                   observed_sequence, evidence_hash, logical_time, quarantine_hash
            FROM rnfe_cg_ledger_conflicts
            WHERE quarantine_id = %s
            """,
            (quarantine_id,),
        )
        row = cur.fetchone()
        return self._quarantine_from_row(row) if row else None


__all__ = ["PostgresCausalLedgerStore"]
