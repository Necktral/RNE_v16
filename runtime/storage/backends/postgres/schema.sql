CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    status TEXT,
    started_at TEXT,
    ended_at TEXT,
    context_json JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS ledger_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT,
    event_type TEXT NOT NULL,
    payload_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    event_ts TEXT NOT NULL,
    source TEXT,
    legacy_db_path TEXT,
    legacy_event_id BIGINT,
    payload_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (legacy_db_path, legacy_event_id)
);

CREATE TABLE IF NOT EXISTS telemetry_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    run_id TEXT,
    metrics_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    snapshot_ts TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS reasoning_traces (
    trace_id TEXT PRIMARY KEY,
    run_id TEXT,
    step_index INTEGER NOT NULL,
    family TEXT NOT NULL,
    status TEXT NOT NULL,
    detail_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    trace_ts TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    metadata_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    timestamp TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT,
    kind TEXT NOT NULL,
    rel_path TEXT NOT NULL,
    abs_path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    mime_type TEXT,
    metadata_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ledger_events_run_id_ts
ON ledger_events(run_id, event_ts);

CREATE INDEX IF NOT EXISTS idx_ledger_events_type_ts
ON ledger_events(event_type, event_ts);

CREATE INDEX IF NOT EXISTS idx_telemetry_snapshots_run_id_ts
ON telemetry_snapshots(run_id, snapshot_ts);

CREATE INDEX IF NOT EXISTS idx_reasoning_traces_run_step
ON reasoning_traces(run_id, step_index);

CREATE INDEX IF NOT EXISTS idx_artifacts_run_kind_ts
ON artifacts(run_id, kind, created_at);
