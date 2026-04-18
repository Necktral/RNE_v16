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

CREATE TABLE IF NOT EXISTS reality_assessments (
    assessment_id TEXT PRIMARY KEY,
    run_id TEXT,
    bench_run_id TEXT,
    episode_id TEXT NOT NULL,
    closure_passed BOOLEAN NOT NULL,
    continuity_score DOUBLE PRECISION NOT NULL,
    trace_integrity BOOLEAN NOT NULL,
    collapse_detected BOOLEAN NOT NULL,
    details_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reality_bench_runs (
    bench_run_id TEXT PRIMARY KEY,
    run_id TEXT,
    total_episodes INTEGER NOT NULL,
    closure_rate DOUBLE PRECISION NOT NULL,
    continuity_mean DOUBLE PRECISION NOT NULL,
    collapse_count INTEGER NOT NULL,
    gate_profile TEXT NOT NULL,
    passed BOOLEAN NOT NULL,
    summary_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episode_certificates (
    certificate_id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    trace_id TEXT NOT NULL,
    smg_artifacts_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    lotf_artifacts_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    world_artifacts_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    continuity_score DOUBLE PRECISION NOT NULL,
    ioc_proxy DOUBLE PRECISION NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL,
    verdict TEXT NOT NULL,
    rollback_ready BOOLEAN NOT NULL,
    promotion_candidate BOOLEAN NOT NULL,
    metadata_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS promotion_decisions (
    decision_id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    certificate_id TEXT NOT NULL,
    verdict TEXT NOT NULL,
    reason TEXT NOT NULL,
    rollback_ready BOOLEAN NOT NULL,
    metadata_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_records (
    memory_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    scale TEXT NOT NULL,
    structure_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    ttl_seconds INTEGER,
    no_interference BOOLEAN NOT NULL,
    certificate_id TEXT,
    ioc_proxy DOUBLE PRECISION,
    support_count INTEGER NOT NULL DEFAULT 0,
    metadata_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transfer_assessments (
    assessment_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    source_scenario TEXT NOT NULL,
    target_scenario TEXT NOT NULL,
    compatibility_class TEXT NOT NULL,
    transfer_verdict TEXT NOT NULL,
    memory_purity_score DOUBLE PRECISION NOT NULL,
    transition_stability_score DOUBLE PRECISION NOT NULL,
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

CREATE INDEX IF NOT EXISTS idx_reality_assessments_run_bench_ts
ON reality_assessments(run_id, bench_run_id, created_at);

CREATE INDEX IF NOT EXISTS idx_reality_bench_runs_run_ts
ON reality_bench_runs(run_id, created_at);

CREATE INDEX IF NOT EXISTS idx_episode_certificates_run_ts
ON episode_certificates(run_id, created_at);

CREATE INDEX IF NOT EXISTS idx_promotion_decisions_run_ts
ON promotion_decisions(run_id, created_at);

CREATE INDEX IF NOT EXISTS idx_memory_records_run_scale_ts
ON memory_records(run_id, scale, created_at);

CREATE INDEX IF NOT EXISTS idx_transfer_assessments_run_ts
ON transfer_assessments(run_id, created_at);

CREATE TABLE IF NOT EXISTS organism_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    trajectory_id TEXT NOT NULL,
    regime TEXT NOT NULL,
    snapshot_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trajectory_windows (
    window_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    trajectory_id TEXT NOT NULL,
    start_episode INTEGER NOT NULL,
    end_episode INTEGER NOT NULL,
    snapshots_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    digest_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trajectory_flow_reports (
    report_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    trajectory_id TEXT NOT NULL,
    window_id TEXT NOT NULL,
    flow_validity BOOLEAN NOT NULL,
    erosion DOUBLE PRECISION NOT NULL,
    phase_drift DOUBLE PRECISION NOT NULL,
    rollback_obligation BOOLEAN NOT NULL,
    report_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS renormalization_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    trajectory_id TEXT NOT NULL,
    source_regime TEXT NOT NULL,
    target_regime TEXT NOT NULL,
    residual_error DOUBLE PRECISION NOT NULL,
    transport_uncertainty DOUBLE PRECISION NOT NULL,
    expected_recovery_cost DOUBLE PRECISION NOT NULL,
    map_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS constitutional_risk_states (
    state_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    trajectory_id TEXT NOT NULL,
    scope_type TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    risk_score DOUBLE PRECISION NOT NULL,
    risk_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    prev_state_id TEXT,
    step_index INTEGER NOT NULL DEFAULT 0,
    metadata_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TEXT NOT NULL
);
ALTER TABLE constitutional_risk_states
ADD COLUMN IF NOT EXISTS prev_state_id TEXT;
ALTER TABLE constitutional_risk_states
ADD COLUMN IF NOT EXISTS step_index INTEGER NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS failure_atlas_events (
    event_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    trajectory_id TEXT NOT NULL,
    scope_type TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    failure_class TEXT NOT NULL,
    severity TEXT NOT NULL,
    reversible BOOLEAN NOT NULL,
    recovery_protocol TEXT NOT NULL,
    signature_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata_jsonb JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_organism_snapshots_run_traj_ts
ON organism_snapshots(run_id, trajectory_id, created_at);

CREATE INDEX IF NOT EXISTS idx_trajectory_windows_run_traj_ts
ON trajectory_windows(run_id, trajectory_id, created_at);

CREATE INDEX IF NOT EXISTS idx_trajectory_flow_reports_run_traj_ts
ON trajectory_flow_reports(run_id, trajectory_id, created_at);

CREATE INDEX IF NOT EXISTS idx_renorm_events_run_traj_ts
ON renormalization_events(run_id, trajectory_id, created_at);

CREATE INDEX IF NOT EXISTS idx_risk_states_run_traj_scope
ON constitutional_risk_states(run_id, trajectory_id, scope_type, scope_key, step_index, created_at);

CREATE INDEX IF NOT EXISTS idx_failure_atlas_run_traj_scope
ON failure_atlas_events(run_id, trajectory_id, scope_type, scope_key, created_at);
