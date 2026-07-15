"""Agente pedagógico: mide lección 7B → conducta → resultado."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from runtime.neural.integration.contracts import (
    AuthorityEffect,
    SymbiosisIdentity,
    canonical_sha256,
)

from .contracts import AgentFinding, AgentReport, AgentRole, AgentState, FindingSeverity


class PedagogicalTeacherAgent:
    """Evalúa eficacia docente; el 7B nunca obtiene autoridad por escribir una lección."""

    agent_id = "agent-pedagogical-teacher-v1"

    def assess(
        self,
        *,
        identity: SymbiosisIdentity,
        lessons: Sequence[Mapping[str, Any]],
        outcome: Mapping[str, Any],
        certificate: Mapping[str, Any],
        reward: Mapping[str, Any],
    ) -> AgentReport:
        lesson_rows = [dict(item) for item in lessons]
        experience = _mapping(outcome.get("experience"))
        bias = _mapping(outcome.get("experience_bias"))
        situation = str(experience.get("situation_key") or "")
        current_severity = _finite_unit(experience.get("severity"))
        matching = [
            lesson for lesson in lesson_rows
            if situation and str(lesson.get("situation_key") or "") == situation
        ]
        applied = [lesson for lesson in matching if _lesson_applied(lesson, bias)]
        comparisons = []
        for lesson in applied:
            origin_severity = _finite_unit(lesson.get("from_severity"))
            delta = (
                round(origin_severity - current_severity, 6)
                if origin_severity is not None and current_severity is not None
                else None
            )
            comparisons.append(
                {
                    "lesson_ref": str(lesson.get("lesson_id") or canonical_sha256(lesson)),
                    "origin_severity": origin_severity,
                    "current_severity": current_severity,
                    "severity_reduction": delta,
                    "improved": delta > 0.0 if delta is not None else None,
                }
            )

        findings: list[AgentFinding] = []
        if not lesson_rows:
            pedagogical_class = "teacher_inactive"
            report_state = AgentState.ABSTAINED
            proposal = "no_teacher_lesson_to_evaluate"
        elif not matching:
            pedagogical_class = "lesson_not_applicable"
            report_state = AgentState.ABSTAINED
            proposal = "retain_for_matching_situation"
        elif not applied:
            pedagogical_class = "lesson_not_applied"
            report_state = AgentState.DEGRADED
            proposal = "trace_why_lesson_did_not_change_decision"
            findings.append(
                AgentFinding(
                    "teacher_lesson_not_applied",
                    FindingSeverity.WARNING,
                    "Existe una lección para la situación, pero no se vincula al sesgo aplicado.",
                    evidence_refs=tuple(_lesson_ref(item) for item in matching),
                )
            )
        elif any(item["improved"] is False for item in comparisons):
            pedagogical_class = "applied_without_improvement"
            report_state = AgentState.DEGRADED
            proposal = "quarantine_lesson_from_curriculum"
            findings.append(
                AgentFinding(
                    "teacher_lesson_failed_outcome_test",
                    FindingSeverity.WARNING,
                    "La lección aplicada no redujo la severidad frente al golpe origen.",
                    evidence_refs=tuple(item["lesson_ref"] for item in comparisons),
                )
            )
        elif comparisons and all(item["improved"] is True for item in comparisons):
            pedagogical_class = "applied_improved_single_observation"
            report_state = AgentState.OBSERVED
            proposal = "accumulate_repeated_outcomes_before_curriculum_promotion"
        else:
            pedagogical_class = "applied_outcome_unmeasured"
            report_state = AgentState.DEGRADED
            proposal = "require_outcome_measurement"

        return AgentReport.create(
            agent_id=self.agent_id,
            role=AgentRole.PEDAGOGICAL_TEACHER,
            identity=identity,
            state=report_state,
            authority_effect=AuthorityEffect.NONE,
            metrics={
                "lesson_count": len(lesson_rows),
                "matching_lesson_count": len(matching),
                "applied_lesson_count": len(applied),
                "improved_observation_count": sum(
                    item["improved"] is True for item in comparisons
                ),
                "failed_observation_count": sum(
                    item["improved"] is False for item in comparisons
                ),
                "reward": _finite_number(reward.get("reward")),
            },
            findings=findings,
            outputs={
                "evidence_pipeline": ["measure", "classify", "analyze", "deliberate"],
                "stages": {
                    "measure": {
                        "situation_key": situation or None,
                        "experience_bias": bias,
                        "certificate_verdict": certificate.get("verdict"),
                        "current_severity": current_severity,
                    },
                    "classify": {"pedagogical_class": pedagogical_class},
                    "analyze": {
                        "comparisons": comparisons,
                        "causal_effect_proven": False,
                        "single_observation_only": bool(comparisons),
                    },
                    "deliberate": {
                        "proposal": proposal,
                        "teacher_authority": "none",
                        "curriculum_promotion_authorized": False,
                    },
                },
                "teacher_roles_separated": ["tier_3_reasoner", "post_experience_teacher"],
                "decision_influence": "none",
            },
        )


def _lesson_applied(lesson: Mapping[str, Any], bias: Mapping[str, Any]) -> bool:
    avoided = str(bias.get("avoided") or "")
    chosen = str(bias.get("chose") or "")
    return bool(
        (avoided and avoided == str(lesson.get("avoid") or ""))
        or (chosen and chosen == str(lesson.get("prefer") or ""))
    )


def _lesson_ref(lesson: Mapping[str, Any]) -> str:
    return str(lesson.get("lesson_id") or canonical_sha256(lesson))


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _finite_unit(value: Any) -> float | None:
    number = _finite_number(value)
    if number is None or not 0.0 <= number <= 1.0:
        return None
    return number


def _finite_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) else None
