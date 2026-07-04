"""Politica de activacion restringida para razonadores externos."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ExternalReasonerGateDecision:
    considered: bool
    called: bool
    reason: str
    skip_reason: str | None = None
    expected_failure_mode: str | None = None
    trigger_signals: dict[str, bool] = field(default_factory=dict)


@dataclass(frozen=True)
class ExternalReasonerGateInput:
    regime: str
    core_intervention: str
    counterfactual_intervention: str | None = None
    causal_recommended_intervention: str | None = None
    counterfactual_recommended_intervention: str | None = None
    core_confidence_proxy: float | None = None
    core_metrics: Mapping[str, Any] = field(default_factory=dict)
    history: Mapping[str, Any] = field(default_factory=dict)


class ExternalReasonerGate:
    """Gate v1: solo activa el modelo externo ante conflicto causal/contrafactual."""

    def __init__(
        self,
        *,
        low_precision_threshold: float = 0.0,
        low_confidence_threshold: float = 0.55,
        history_correction_threshold: float = 0.20,
    ):
        self.low_precision_threshold = low_precision_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.history_correction_threshold = history_correction_threshold

    def evaluate(self, gate_input: ExternalReasonerGateInput) -> ExternalReasonerGateDecision:
        regime = (gate_input.regime or "").strip().lower()
        metrics = gate_input.core_metrics or {}
        intervention_precision = _as_float(metrics.get("intervention_precision"), 0.0)
        viability_margin = _as_float(metrics.get("viability_margin"), 0.0)
        closure_stable = bool(metrics.get("closure_stable"))

        explicit_conflict_regime = regime == "causal_counterfactual_conflict"
        structural_conflict = self._has_structural_conflict(gate_input)
        core_risky = (
            intervention_precision <= self.low_precision_threshold
            or viability_margin < 0.0
            or not closure_stable
            or (
                gate_input.core_confidence_proxy is not None
                and _as_float(gate_input.core_confidence_proxy, 1.0) < self.low_confidence_threshold
            )
        )
        history_supports_call = self._history_supports_call(regime, gate_input.history)

        trigger_signals = {
            "explicit_conflict_regime": explicit_conflict_regime,
            "structural_conflict": structural_conflict,
            "core_risky": core_risky,
            "history_supports_call": history_supports_call,
        }
        if explicit_conflict_regime:
            return ExternalReasonerGateDecision(
                considered=True,
                called=True,
                reason="causal_counterfactual_conflict",
                expected_failure_mode="core_counterfactual_mismatch",
                trigger_signals=trigger_signals,
            )
        if structural_conflict:
            return ExternalReasonerGateDecision(
                considered=True,
                called=True,
                reason="cau_ctf_intervention_conflict",
                expected_failure_mode="causal_counterfactual_disagreement",
                trigger_signals=trigger_signals,
            )
        if core_risky and history_supports_call:
            return ExternalReasonerGateDecision(
                considered=True,
                called=True,
                reason="core_risky_with_positive_history",
                expected_failure_mode="known_core_failure_regime",
                trigger_signals=trigger_signals,
            )
        if core_risky:
            return ExternalReasonerGateDecision(
                considered=True,
                called=False,
                reason="skip",
                skip_reason="core_risky_but_no_conflict_evidence",
                expected_failure_mode="unproven_external_gain",
                trigger_signals=trigger_signals,
            )
        return ExternalReasonerGateDecision(
            considered=True,
            called=False,
            reason="skip",
            skip_reason="core_stable_no_conflict",
            expected_failure_mode=None,
            trigger_signals=trigger_signals,
        )

    def _has_structural_conflict(self, gate_input: ExternalReasonerGateInput) -> bool:
        cau = (gate_input.causal_recommended_intervention or gate_input.core_intervention or "").strip()
        ctf = (
            gate_input.counterfactual_recommended_intervention
            or gate_input.counterfactual_intervention
            or ""
        ).strip()
        return bool(cau and ctf and cau != ctf)

    def _history_supports_call(self, regime: str, history: Mapping[str, Any]) -> bool:
        if not history:
            return False
        regime_history = history.get(regime) if isinstance(history, Mapping) else None
        if not isinstance(regime_history, Mapping):
            return False
        corrected_rate = _as_float(regime_history.get("external_reasoner_corrected_core_failure_rate"), 0.0)
        return corrected_rate >= self.history_correction_threshold
