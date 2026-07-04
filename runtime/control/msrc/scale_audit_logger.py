"""Logger auditable para decisiones y transiciones MSRC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from runtime.storage.records import utc_now_iso

from .contracts import ScaleDecisionRecord, ScaleTransitionRecord


class ScaleAuditLogger:
    """Emite eventos canónicos y traza JSONL para reconstrucción offline."""

    def __init__(self, *, storage, output_dir: Optional[Path] = None):
        self.storage = storage
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._decisions_file = self.output_dir / "scale_decisions.jsonl"
            self._events_file = self.output_dir / "scale_events.jsonl"
        else:
            self._decisions_file = None
            self._events_file = None

    def log_decision(self, record: ScaleDecisionRecord) -> None:
        payload = record.to_dict()
        self._emit_event(
            event_type="msrc.decision",
            run_id=record.run_id,
            payload=payload,
        )
        self._append_jsonl(self._decisions_file, payload)

    def log_transition(self, record: ScaleTransitionRecord) -> None:
        payload = record.to_dict()
        event_type = "msrc.transition"
        if record.rollback_applied:
            event_type = "msrc.rollback"
        self._emit_event(
            event_type=event_type,
            run_id=record.run_id,
            payload=payload,
        )
        self._append_jsonl(self._events_file, {"event_type": event_type, **payload})

    def log_probe_started(self, *, run_id: str, payload: Dict[str, Any]) -> None:
        self._emit_event("msrc.probe.started", run_id=run_id, payload=payload)
        self._append_jsonl(self._events_file, {"event_type": "msrc.probe.started", **payload})

    def log_probe_completed(self, *, run_id: str, payload: Dict[str, Any]) -> None:
        self._emit_event("msrc.probe.completed", run_id=run_id, payload=payload)
        self._append_jsonl(self._events_file, {"event_type": "msrc.probe.completed", **payload})

    def log_probe_committed(self, *, run_id: str, payload: Dict[str, Any]) -> None:
        self._emit_event("msrc.probe.committed", run_id=run_id, payload=payload)
        self._append_jsonl(self._events_file, {"event_type": "msrc.probe.committed", **payload})

    def log_probe_discarded(self, *, run_id: str, payload: Dict[str, Any]) -> None:
        self._emit_event("msrc.probe.discarded", run_id=run_id, payload=payload)
        self._append_jsonl(self._events_file, {"event_type": "msrc.probe.discarded", **payload})

    def log_regret(self, *, run_id: str, payload: Dict[str, Any]) -> None:
        self._emit_event("msrc.regret", run_id=run_id, payload=payload)
        self._append_jsonl(self._events_file, {"event_type": "msrc.regret", **payload})

    def log_oscillation(self, *, run_id: str, payload: Dict[str, Any]) -> None:
        self._emit_event("msrc.oscillation", run_id=run_id, payload=payload)
        self._append_jsonl(self._events_file, {"event_type": "msrc.oscillation", **payload})

    def _emit_event(self, event_type: str, *, run_id: str, payload: Dict[str, Any]) -> None:
        stamped = {
            "timestamp": utc_now_iso(),
            **payload,
        }
        self.storage.append_event(
            event_type=event_type,
            run_id=run_id,
            source="msrc",
            payload=stamped,
        )

    def _append_jsonl(self, path: Optional[Path], payload: Dict[str, Any]) -> None:
        if path is None:
            return
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")
