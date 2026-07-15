"""Agente interoceptivo: integra viabilidad y señales físicas medidas."""

from __future__ import annotations

import math
from typing import Any, Mapping

from runtime.neural.integration.contracts import AuthorityEffect, SymbiosisIdentity

from .contracts import AgentFinding, AgentReport, AgentRole, AgentState, FindingSeverity


_PRESSURE_AXES = ("cpu_pressure", "memory_pressure", "thermal_pressure", "vram_pressure")


class InteroceptiveHomeostaticAgent:
    """Observa el estado interno sin sustituir MSRC ni el kernel de viabilidad."""

    agent_id = "agent-interoceptive-homeostatic-v1"

    def assess(
        self,
        *,
        identity: SymbiosisIdentity,
        viability: Mapping[str, Any],
        resources: Mapping[str, Any],
        measurement_status: Mapping[str, str],
        trace_health: Mapping[str, Any],
    ) -> AgentReport:
        statuses = {str(key): str(value) for key, value in measurement_status.items()}
        measured_pressures = {
            axis: _number(resources.get(axis))
            for axis in _PRESSURE_AXES
            if statuses.get(axis) == "measured" and _number(resources.get(axis)) is not None
        }
        missing_axes = tuple(
            axis for axis in _PRESSURE_AXES if statuses.get(axis) not in {"measured", "not_applicable"}
        )
        viability_margin = _number(viability.get("viability_margin"))
        distance_to_edge = _number(viability.get("distance_to_edge"))
        is_viable = viability.get("is_viable") if isinstance(viability.get("is_viable"), bool) else None
        rollback_required = bool(viability.get("rollback_required"))
        budget_available = bool(resources.get("msrc_budget_available", True))
        budget_measured = statuses.get("msrc_budget_available") == "measured"
        peak_pressure = max(measured_pressures.values(), default=None)
        persistence_degraded = bool(trace_health.get("degraded")) or bool(
            int(trace_health.get("pending_events", 0) or 0)
            or int(trace_health.get("dropped_events", 0) or 0)
        )

        findings: list[AgentFinding] = []
        if rollback_required:
            findings.append(
                AgentFinding(
                    "interoception_rollback_required",
                    FindingSeverity.CRITICAL,
                    "El kernel de viabilidad solicita refugio o rollback.",
                )
            )
        if budget_measured and not budget_available:
            findings.append(
                AgentFinding(
                    "interoception_msrc_budget_closed",
                    FindingSeverity.CRITICAL,
                    "MSRC reporta presupuesto no disponible.",
                )
            )
        if peak_pressure is not None and peak_pressure >= 0.9:
            findings.append(
                AgentFinding(
                    "interoception_physical_pressure_high",
                    FindingSeverity.WARNING,
                    "Una presión física medida supera 0.90.",
                )
            )
        if is_viable is False or (viability_margin is not None and viability_margin < 0.35):
            findings.append(
                AgentFinding(
                    "interoception_viability_stressed",
                    FindingSeverity.WARNING,
                    "La viabilidad medida está fuera del sobre conservador.",
                )
            )
        if missing_axes or viability_margin is None or distance_to_edge is None:
            findings.append(
                AgentFinding(
                    "interoception_measurement_incomplete",
                    FindingSeverity.WARNING,
                    "Faltan ejes internos; los defaults no se aceptan como mediciones.",
                    evidence_refs=missing_axes,
                )
            )
        if persistence_degraded:
            findings.append(
                AgentFinding(
                    "interoception_persistence_degraded",
                    FindingSeverity.WARNING,
                    "El estado interno observado aún no es completamente durable.",
                )
            )

        emergency = rollback_required or (budget_measured and not budget_available)
        stressed = (
            is_viable is False
            or (viability_margin is not None and viability_margin < 0.35)
            or (peak_pressure is not None and peak_pressure >= 0.9)
        )
        incomplete = bool(missing_axes or viability_margin is None or distance_to_edge is None)
        if emergency:
            state, internal_class, proposal = (
                AgentState.BLOCKED,
                "homeostatic_emergency",
                "request_kernel_refuge_and_defer_optional_neural_work",
            )
        elif stressed:
            state, internal_class, proposal = (
                AgentState.DEGRADED,
                "homeostatic_stress",
                "reduce_optional_load_and_request_viability_reassessment",
            )
        elif incomplete or persistence_degraded:
            state, internal_class, proposal = (
                AgentState.DEGRADED,
                "partially_measured",
                "measure_missing_internal_axes_before_adaptation",
            )
        else:
            state, internal_class, proposal = (
                AgentState.OBSERVED,
                "homeostatic_envelope_observed",
                "retain_current_homeostatic_envelope",
            )

        return AgentReport.create(
            agent_id=self.agent_id,
            role=AgentRole.INTEROCEPTIVE_HOMEOSTATIC,
            identity=identity,
            state=state,
            authority_effect=AuthorityEffect.NONE,
            metrics={
                "viability_margin": viability_margin,
                "distance_to_edge": distance_to_edge,
                "is_viable": is_viable,
                "rollback_required": rollback_required,
                "measured_pressure_count": len(measured_pressures),
                "peak_measured_pressure": peak_pressure,
                "msrc_budget_available": budget_available if budget_measured else None,
                "persistence_degraded": persistence_degraded,
            },
            findings=findings,
            outputs={
                "evidence_pipeline": ["measure", "classify", "analyze", "deliberate"],
                "stages": {
                    "measure": {
                        "viability": dict(viability),
                        "measured_pressures": measured_pressures,
                        "measurement_status": statuses,
                        "missing_axes": list(missing_axes),
                    },
                    "classify": {"interoceptive_class": internal_class},
                    "analyze": {
                        "homeostatic_emergency": emergency,
                        "homeostatic_stress": stressed,
                        "measurement_complete": not incomplete,
                    },
                    "deliberate": {
                        "proposal": proposal,
                        "resource_mutation_authorized": False,
                        "rollback_authorized": False,
                        "actuation_authorized": False,
                    },
                },
                "msrc_authority_preserved": True,
                "viability_kernel_authority_preserved": True,
                "decision_influence": "none",
            },
        )


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
