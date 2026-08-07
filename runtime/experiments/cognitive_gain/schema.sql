CREATE TABLE IF NOT EXISTS rnfe_cg_ledger_chains (
    experiment_id TEXT NOT NULL,
    arm_id TEXT NOT NULL,
    chain_id TEXT NOT NULL,
    head_sequence BIGINT NOT NULL DEFAULT -1,
    head_event_hash TEXT NOT NULL DEFAULT 'urn:rnfe:cg:genesis:v1',
    head_logical_time BIGINT NOT NULL DEFAULT -1,
    quarantined BOOLEAN NOT NULL DEFAULT FALSE,
    quarantine_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (experiment_id, arm_id, chain_id),
    CHECK (head_sequence >= -1),
    CHECK (head_logical_time >= -1),
    CHECK (quarantined = (quarantine_id IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS rnfe_cg_ledger_events (
    schema_version TEXT NOT NULL,
    experiment_id TEXT NOT NULL,
    arm_id TEXT NOT NULL,
    chain_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    event_sequence BIGINT NOT NULL,
    previous_event_hash TEXT NOT NULL,
    logical_time BIGINT NOT NULL,
    wall_clock_time TEXT NOT NULL,
    payload_contract_type TEXT NOT NULL,
    payload_hash CHAR(64) NOT NULL,
    event_hash CHAR(64) NOT NULL,
    payload_jsonb JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (experiment_id, arm_id, chain_id, event_id),
    UNIQUE (experiment_id, arm_id, chain_id, event_sequence),
    FOREIGN KEY (experiment_id, arm_id, chain_id)
        REFERENCES rnfe_cg_ledger_chains (experiment_id, arm_id, chain_id)
        ON DELETE RESTRICT,
    CHECK (event_sequence >= 0),
    CHECK (logical_time >= 0),
    CHECK (payload_hash ~ '^[0-9a-f]{64}$'),
    CHECK (event_hash ~ '^[0-9a-f]{64}$')
);

CREATE TABLE IF NOT EXISTS rnfe_cg_ledger_conflicts (
    quarantine_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    arm_id TEXT NOT NULL,
    chain_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    conflicting_event_id TEXT NOT NULL,
    stored_payload_hash CHAR(64) NOT NULL,
    incoming_payload_hash CHAR(64) NOT NULL,
    observed_sequence BIGINT NOT NULL,
    evidence_hash CHAR(64) NOT NULL,
    logical_time BIGINT NOT NULL,
    quarantine_hash CHAR(64) NOT NULL,
    record_jsonb JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (
        experiment_id,
        arm_id,
        chain_id,
        conflicting_event_id,
        stored_payload_hash,
        incoming_payload_hash
    ),
    FOREIGN KEY (experiment_id, arm_id, chain_id)
        REFERENCES rnfe_cg_ledger_chains (experiment_id, arm_id, chain_id)
        ON DELETE RESTRICT,
    CHECK (stored_payload_hash ~ '^[0-9a-f]{64}$'),
    CHECK (incoming_payload_hash ~ '^[0-9a-f]{64}$'),
    CHECK (evidence_hash ~ '^[0-9a-f]{64}$'),
    CHECK (quarantine_hash ~ '^[0-9a-f]{64}$')
);
