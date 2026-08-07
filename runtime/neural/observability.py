"""Health y buffer acotado para trazas neuronales no silenciosas."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import wraps
import hashlib
import json
import os
from threading import RLock
import time
from typing import Any, Callable, Iterator, Mapping, TypeVar
from uuid import uuid4


SHADOW_OBSERVATION_SCHEMA_VERSION = "neural.shadow_observation.v1"
SHADOW_SPAN_SCHEMA_VERSION = "neural.shadow_latency_span.v1"
_TRUE = {"1", "true", "yes", "on"}
_F = TypeVar("_F", bound=Callable[..., Any])
_SHADOW_OBSERVATION_FIELDS = {
    "schema_version", "run_id", "lane", "seed", "step_index", "scenario_id",
    "external_input_hash", "pair_id", "authority_mode", "decision_influence",
    "admission_status", "admission_reason", "candidate_output_hash",
    "candidate_output_hashes_by_organ", "effective_output_hash",
    "authoritative_decision_hash", "action_hash", "outcome_hash", "behavior_hash",
    "resource_cost_hash", "candidate_count", "candidate_applied_count",
    "fallback_used", "certification_status", "timestamp_monotonic_ns", "trace_parent",
}


@dataclass(frozen=True, slots=True)
class BufferedTraceEvent:
    event_type: str
    payload: Mapping[str, Any]
    run_id: str | None
    source: str
    timestamp: str


@dataclass(frozen=True, slots=True)
class TraceHealthSnapshot:
    storage_configured: bool
    degraded: bool
    persistence_failures: int
    consecutive_failures: int
    pending_events: int
    dropped_events: int
    recovered_events: int
    last_error: str | None
    last_failure_at: str | None
    last_recovery_at: str | None


class TracePersistenceMonitor:
    def __init__(self, *, storage_configured: bool, max_buffered_events: int = 128):
        if max_buffered_events <= 0:
            raise ValueError("trace_buffer_size_must_be_positive")
        self.storage_configured = storage_configured
        self.max_buffered_events = int(max_buffered_events)
        self._pending: deque[BufferedTraceEvent] = deque()
        self._persistence_failures = 0
        self._consecutive_failures = 0
        self._dropped_events = 0
        self._recovered_events = 0
        self._last_error: str | None = None
        self._last_failure_at: str | None = None
        self._last_recovery_at: str | None = None
        self._lock = RLock()

    def new_event(
        self,
        *,
        event_type: str,
        payload: Mapping[str, Any],
        run_id: str | None,
        source: str = "runtime.neural",
    ) -> BufferedTraceEvent:
        return BufferedTraceEvent(
            event_type=event_type,
            payload=dict(payload),
            run_id=run_id,
            source=source,
            timestamp=_utc_now_iso(),
        )

    def record_failure(self, event: BufferedTraceEvent, error: BaseException) -> None:
        with self._lock:
            self._record_error(error)
            if len(self._pending) >= self.max_buffered_events:
                self._pending.popleft()
                self._dropped_events += 1
            self._pending.append(event)

    def record_flush_failure(self, error: BaseException) -> None:
        with self._lock:
            self._record_error(error)

    def pending(self) -> tuple[BufferedTraceEvent, ...]:
        with self._lock:
            return tuple(self._pending)

    def mark_flushed(self, count: int) -> None:
        with self._lock:
            actual = min(max(int(count), 0), len(self._pending))
            for _ in range(actual):
                self._pending.popleft()
            self._recovered_events += actual
            if not self._pending:
                self._consecutive_failures = 0
                self._last_recovery_at = _utc_now_iso()

    def snapshot(self) -> TraceHealthSnapshot:
        with self._lock:
            return TraceHealthSnapshot(
                storage_configured=self.storage_configured,
                degraded=bool(self._pending or self._consecutive_failures),
                persistence_failures=self._persistence_failures,
                consecutive_failures=self._consecutive_failures,
                pending_events=len(self._pending),
                dropped_events=self._dropped_events,
                recovered_events=self._recovered_events,
                last_error=self._last_error,
                last_failure_at=self._last_failure_at,
                last_recovery_at=self._last_recovery_at,
            )

    def _record_error(self, error: BaseException) -> None:
        self._persistence_failures += 1
        self._consecutive_failures += 1
        self._last_error = f"{error.__class__.__name__}:{str(error)[:200]}"
        self._last_failure_at = _utc_now_iso()


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def shadow_observability_enabled() -> bool:
    """The diagnostic plane is opt-in and has no authority side effects."""

    return os.environ.get("RNFE_SHADOW_CAUSAL_OBSERVABILITY", "").strip().lower() in _TRUE


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def shadow_pair_id(*, seed: int, step_index: int, scenario_id: str, external_input_hash: str) -> str:
    return canonical_sha256(
        {
            "schema_version": SHADOW_OBSERVATION_SCHEMA_VERSION,
            "seed": int(seed),
            "step_index": int(step_index),
            "scenario_id": str(scenario_id),
            "external_input_hash": str(external_input_hash),
        }
    )


@dataclass(slots=True)
class ShadowLatencySpan:
    name: str
    start_ns: int
    end_ns: int
    duration_ns: int
    parent_span_id: str | None
    exclusive_duration_ns: int
    call_count: int
    status: str
    error_code: str | None
    span_id: str
    trace_parent: str
    schema_version: str = SHADOW_SPAN_SCHEMA_VERSION
    _child_duration_ns: int = field(default=0, repr=False)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("_child_duration_ns", None)
        return data


class ShadowSpanCollector:
    """Per-step nested spans with explicit exclusive durations."""

    def __init__(self, *, trace_parent: str):
        self.trace_parent = str(trace_parent)
        self._stack: list[ShadowLatencySpan] = []
        self._spans: list[ShadowLatencySpan] = []
        self._lock = RLock()

    def begin(self, name: str) -> ShadowLatencySpan:
        now = time.monotonic_ns()
        with self._lock:
            span = ShadowLatencySpan(
                name=str(name),
                start_ns=now,
                end_ns=now,
                duration_ns=0,
                parent_span_id=self._stack[-1].span_id if self._stack else None,
                exclusive_duration_ns=0,
                call_count=1,
                status="running",
                error_code=None,
                span_id=f"span-{uuid4().hex}",
                trace_parent=self.trace_parent,
            )
            self._stack.append(span)
            return span

    def end(self, span: ShadowLatencySpan, *, error: BaseException | None = None) -> None:
        now = time.monotonic_ns()
        with self._lock:
            if span.status != "running":
                return
            span.end_ns = max(now, span.start_ns)
            span.duration_ns = span.end_ns - span.start_ns
            span.exclusive_duration_ns = max(0, span.duration_ns - span._child_duration_ns)
            span.status = "error" if error is not None else "completed"
            span.error_code = error.__class__.__name__ if error is not None else None
            if self._stack and self._stack[-1] is span:
                self._stack.pop()
            elif span in self._stack:
                self._stack.remove(span)
            if self._stack:
                self._stack[-1]._child_duration_ns += span.duration_ns
            self._spans.append(span)

    def finish_open(self) -> None:
        while self._stack:
            self.end(self._stack[-1], error=RuntimeError("observability_span_unclosed"))

    def spans(self) -> list[dict[str, Any]]:
        with self._lock:
            return [span.to_dict() for span in self._spans]


_ACTIVE_COLLECTOR: ContextVar[ShadowSpanCollector | None] = ContextVar(
    "rnfe_shadow_span_collector", default=None
)


@contextmanager
def shadow_observation_scope(collector: ShadowSpanCollector) -> Iterator[ShadowSpanCollector]:
    if not shadow_observability_enabled():
        yield collector
        return
    token = _ACTIVE_COLLECTOR.set(collector)
    try:
        yield collector
    finally:
        collector.finish_open()
        _ACTIVE_COLLECTOR.reset(token)


def start_shadow_span(name: str) -> ShadowLatencySpan | None:
    collector = _ACTIVE_COLLECTOR.get()
    if collector is None:
        return None
    try:
        return collector.begin(name)
    except Exception:
        return None


def finish_shadow_span(span: ShadowLatencySpan | None, error: BaseException | None = None) -> None:
    if span is None:
        return
    collector = _ACTIVE_COLLECTOR.get()
    if collector is None:
        return
    try:
        collector.end(span, error=error)
    except Exception:
        return


def observed_span(name: str) -> Callable[[_F], _F]:
    def decorate(function: _F) -> _F:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            span = start_shadow_span(name)
            try:
                result = function(*args, **kwargs)
            except BaseException as exc:
                finish_shadow_span(span, exc)
                raise
            finish_shadow_span(span)
            return result

        return wrapped  # type: ignore[return-value]

    return decorate


def build_shadow_observation(
    *, run_id: str, lane: str, seed: int, row: Mapping[str, Any], spans: list[dict[str, Any]]
) -> dict[str, Any]:
    """Project causal behavior separately from operational resource cost."""

    decision = dict(row.get("decision") or {})
    episode_result = dict(row.get("episode_result") or {})
    episode = dict(episode_result.get("episode") or {})
    context = dict(episode.get("context") or {})
    outcome = dict(episode.get("result") or {})
    certification = dict(episode_result.get("certification") or {})
    trace = dict(episode_result.get("neural_symbiosis_trace") or {})
    organs = list(trace.get("organs") or ())
    vital_signs = dict(row.get("vital_signs") or {})
    external_input_hash = canonical_sha256(decision.get("external_input"))
    scenario_id = str(decision.get("scenario") or episode.get("scenario") or "")
    step_index = int(row.get("step_index") or 0)
    candidate_hashes = {
        str(item.get("organ")): item.get("candidate_hash")
        for item in organs
        if item.get("candidate_hash")
    }
    authoritative_decision = {
        key: decision.get(key)
        for key in ("action", "external_input", "mode", "priority", "reason", "scenario")
    }
    action = {"action": decision.get("action"), "intervention": context.get("intervention")}
    canonical_outcome = {
        key: outcome.get(key)
        for key in (
            "alarm_transition", "counterfactual_delta", "factual_delta",
            "intervention_effect", "reasoning_sequence", "relation_kind", "updated_world",
        )
    }
    behavior = {
        "decision": authoritative_decision,
        "action": action,
        "outcome": canonical_outcome,
        "abstention": decision.get("action") in {"wait", "sleep", "quarantine"},
        "functional_state": {
            key: vital_signs.get(key)
            for key in ("certified", "cognitive_quality", "identity_continuity", "mode", "risk_score", "viability_margin")
        },
    }
    total_duration_ns = sum(
        int(item.get("duration_ns") or 0) for item in spans if item.get("name") == "life_step_total"
    )
    resource_cost = {
        "resource_pressure": vital_signs.get("resource_pressure"),
        "resource_state": trace.get("resource_state") or {},
        "duration_ns": total_duration_ns,
        "candidate_count": len(candidate_hashes),
        "organ_costs": {str(item.get("organ")): item.get("cost") or {} for item in organs},
    }
    applied_count = sum(
        bool((item.get("candidate") or {}).get("applied"))
        for item in organs
        if isinstance(item.get("candidate"), Mapping)
    )
    authority_mode = "off" if lane == "off" else "shadow"
    effective_output = {"decision": authoritative_decision, "action": action, "outcome": canonical_outcome}
    return {
        "schema_version": SHADOW_OBSERVATION_SCHEMA_VERSION,
        "run_id": str(run_id), "lane": str(lane), "seed": int(seed),
        "step_index": step_index, "scenario_id": scenario_id,
        "external_input_hash": external_input_hash,
        "pair_id": shadow_pair_id(seed=seed, step_index=step_index, scenario_id=scenario_id, external_input_hash=external_input_hash),
        "authority_mode": authority_mode,
        "decision_influence": "none",
        "admission_status": "not_evaluated" if authority_mode == "shadow" else "unavailable",
        "admission_reason": "shadow_authority_mode" if authority_mode == "shadow" else "neural_mode_off",
        "candidate_output_hash": canonical_sha256(candidate_hashes),
        "candidate_output_hashes_by_organ": candidate_hashes,
        "effective_output_hash": canonical_sha256(effective_output),
        "authoritative_decision_hash": canonical_sha256(authoritative_decision),
        "action_hash": canonical_sha256(action),
        "outcome_hash": canonical_sha256(canonical_outcome),
        "behavior_hash": canonical_sha256(behavior),
        "resource_cost_hash": canonical_sha256(resource_cost),
        "candidate_count": len(candidate_hashes),
        "candidate_applied_count": int(applied_count),
        "fallback_used": any(bool(item.get("fallback_reason")) for item in organs),
        "certification_status": certification.get("verdict"),
        "timestamp_monotonic_ns": max((int(item.get("end_ns") or 0) for item in spans), default=time.monotonic_ns()),
        "trace_parent": trace.get("trace_group_id") or f"shadow-step-{seed}-{step_index}",
    }


def latency_attribution(spans: list[Mapping[str, Any]]) -> dict[str, Any]:
    totals: dict[str, int] = {}
    total_ns = 0
    for span in spans:
        exclusive = max(0, int(span.get("exclusive_duration_ns") or 0))
        name = str(span.get("name") or "unknown")
        totals[name] = totals.get(name, 0) + exclusive
        if name == "life_step_total":
            total_ns += int(span.get("duration_ns") or 0)
    attributed = sum(value for name, value in totals.items() if name != "life_step_total")
    remainder = max(0, total_ns - attributed)
    return {
        "schema_version": "neural.shadow_latency_attribution.v1",
        "life_step_total_ns": total_ns,
        "exclusive_by_name_ns": totals,
        "attributed_ns": attributed,
        "uninstrumented_remainder_ns": remainder,
        "coverage": attributed / total_ns if total_ns else 0.0,
        "reconciled": attributed + remainder == total_ns,
    }


def validate_shadow_observation(record: Mapping[str, Any]) -> None:
    """Minimal fail-closed contract validation without an optional JSON-schema runtime."""

    if record.get("schema_version") != SHADOW_OBSERVATION_SCHEMA_VERSION:
        raise ValueError("shadow_observation_schema_unknown")
    unknown = set(record) - _SHADOW_OBSERVATION_FIELDS
    missing = _SHADOW_OBSERVATION_FIELDS - set(record)
    if unknown:
        raise ValueError(f"shadow_observation_fields_unknown:{','.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"shadow_observation_fields_missing:{','.join(sorted(missing))}")
    if record.get("lane") not in {"off", "shadow"}:
        raise ValueError("shadow_observation_lane_invalid")
    if record.get("decision_influence") != "none":
        raise ValueError("shadow_observation_authority_violation")
    if int(record.get("candidate_applied_count") or 0) != 0:
        raise ValueError("shadow_observation_candidate_applied")
    serialized = json.dumps(dict(record), sort_keys=True).lower()
    if any(token in serialized for token in ("postgresql://", "authorization:", "api_key", "password=")):
        raise ValueError("shadow_observation_secret_detected")
