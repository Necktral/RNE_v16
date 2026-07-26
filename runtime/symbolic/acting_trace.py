"""Colector append-only para decisión preacción y outcome certificado."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .schemas import (
    ActingDecisionTrace,
    ActingOutcomeLink,
    CausalGuardReport,
    CoreReport,
    OptimizationDecisionReport,
    append_canonical_jsonl,
    sealed_sha256,
)


class ActingTraceCollector:
    def __init__(
        self,
        *,
        replay_unit_id: str,
        preaction_logical_time: int,
        trace_dir: str | Path,
    ):
        self.replay_unit_id = replay_unit_id
        self.preaction_logical_time = int(preaction_logical_time)
        self.trace_dir = Path(trace_dir)
        self.trace_status = "ok"
        self._core_report: CoreReport | None = None
        self._opt_report: OptimizationDecisionReport | None = None
        self._guard_report: CausalGuardReport | None = None
        self._baseline_action: str | None = None
        self._mci_evidence: Mapping[str, Any] | None = None
        self._trace: ActingDecisionTrace | None = None

    @property
    def trace(self) -> ActingDecisionTrace | None:
        return self._trace

    @property
    def paths(self) -> dict[str, str]:
        return {
            "constraints": str(self.trace_dir / "constraints.jsonl"),
            "core_report": str(self.trace_dir / "core_report.jsonl"),
            "decision_trace": str(self.trace_dir / "acting_decision_trace.jsonl"),
            "outcome_link": str(self.trace_dir / "acting_outcome_link.jsonl"),
        }

    def set_ded_report(self, report: CoreReport | None) -> None:
        self._core_report = report

    def set_opt_report(self, report: OptimizationDecisionReport | None) -> None:
        self._opt_report = report

    def set_guard_report(self, report: CausalGuardReport) -> None:
        self._guard_report = report

    def set_baseline_action(self, action: str) -> None:
        self._baseline_action = str(action)

    def set_mci_evidence(self, evidence: Mapping[str, Any] | None) -> None:
        self._mci_evidence = evidence

    def _append(self, filename: str, value: object) -> None:
        try:
            append_canonical_jsonl(self.trace_dir / filename, value)
        except OSError:
            self.trace_status = "persistence_degraded"

    def mark_persistence_degraded(self) -> None:
        self.trace_status = "persistence_degraded"

    def seal(
        self,
        *,
        committed_action: str,
        governance_verdict: Mapping[str, Any],
    ) -> ActingDecisionTrace:
        if self._trace is not None:
            raise RuntimeError("ActingDecisionTrace ya fue sellada")
        if self._guard_report is None or self._baseline_action is None:
            raise RuntimeError("La traza carece de guarda o acción baseline")
        payload = {
            "replay_unit_id": self.replay_unit_id,
            "preaction_logical_time": self.preaction_logical_time,
            "core_report": self._core_report,
            "opt_report": self._opt_report,
            "guard_report": self._guard_report,
            "baseline_action": self._baseline_action,
            "committed_action": str(committed_action),
            "governance_verdict": dict(governance_verdict),
        }
        if self._mci_evidence is not None:
            payload["mci_evidence"] = self._mci_evidence
        self._trace = ActingDecisionTrace(
            **payload,
            sealed_hash=sealed_sha256(payload),
        )
        self._append("acting_decision_trace.jsonl", self._trace)
        return self._trace

    def link_outcome(
        self,
        *,
        outcome_observed: Mapping[str, Any],
        prediction_error: float | None,
        utility: float | None,
        certification_ref: str | None,
    ) -> ActingOutcomeLink:
        if self._trace is None:
            raise RuntimeError("No se puede enlazar outcome antes de sellar")
        link = ActingOutcomeLink(
            decision_trace_sha256=self._trace.sealed_hash,
            outcome_observed=outcome_observed,
            prediction_error=prediction_error,
            utility=utility,
            certification_ref=certification_ref,
        )
        self._append("acting_outcome_link.jsonl", link)
        return link
