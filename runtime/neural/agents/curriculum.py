"""Evaluación curricular y de eficiencia comparativa de docentes."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

from runtime.neural.integration.contracts import AuthorityEffect, SymbiosisIdentity

from .contracts import AgentFinding, AgentReport, AgentRole, AgentState, FindingSeverity


class CurriculumLearningAgent:
    """Contrasta 7B/Codex; una lección aislada nunca se promueve como aprendizaje."""

    agent_id = "agent-curriculum-learning-v1"

    def assess(
        self,
        *,
        identity: SymbiosisIdentity,
        lessons: Sequence[Mapping[str, Any]],
        pedagogical_report: AgentReport,
    ) -> AgentReport:
        comparisons = list(
            pedagogical_report.outputs.get("stages", {}).get("analyze", {}).get("comparisons", [])
        )
        lesson_by_ref = {
            str(item.get("lesson_id") or ""): dict(item) for item in lessons
            if item.get("lesson_id")
        }
        source_counts: Counter[str] = Counter()
        measured = []
        for row in comparisons:
            lesson = lesson_by_ref.get(str(row.get("lesson_ref") or ""), {})
            source = str(lesson.get("teacher_source") or "local_7b")
            source_counts[source] += 1
            latency = _finite(lesson.get("teacher_latency_s"))
            reduction = _finite(row.get("severity_reduction"))
            measured.append(
                {
                    "lesson_ref": row.get("lesson_ref"),
                    "teacher_source": source,
                    "severity_reduction": reduction,
                    "teacher_latency_s": latency,
                    "benefit_per_second": (
                        round(reduction / latency, 6)
                        if reduction is not None and latency is not None and latency > 0.0
                        else None
                    ),
                    "evaluation_pair_id": lesson.get("evaluation_pair_id"),
                    "evaluation_variant": lesson.get("evaluation_variant"),
                    "raw_semantic_valid": lesson.get("teacher_raw_semantic_valid"),
                    "teacher_repairs": list(lesson.get("teacher_repairs") or ()),
                }
            )
        paired_ids: dict[str, set[str]] = {}
        for row in measured:
            pair_id = str(row.get("evaluation_pair_id") or "")
            variant = str(row.get("evaluation_variant") or "")
            if pair_id and variant:
                paired_ids.setdefault(pair_id, set()).add(variant)
        comparable_pairs = sum(len(variants) >= 2 for variants in paired_ids.values())
        invalid_teacher_outputs = sum(row.get("raw_semantic_valid") is False for row in measured)
        findings = []
        if not lessons:
            state, cls, proposal = AgentState.ABSTAINED, "curriculum_inactive", "collect_teacher_trials"
        elif not comparisons:
            state, cls, proposal = AgentState.DEGRADED, "outcome_unlinked", "link_lesson_to_outcome"
        elif invalid_teacher_outputs:
            state, cls, proposal = AgentState.DEGRADED, "teacher_output_semantically_repaired", "retain_repaired_lesson_as_unproven_proposal"
            findings.append(AgentFinding(
                "curriculum_teacher_semantic_failure",
                FindingSeverity.WARNING,
                "El docente produjo una lección estructurada que falló validación semántica.",
            ))
        elif comparable_pairs == 0:
            state, cls, proposal = AgentState.OBSERVED, "efficiency_unmeasured_unpaired", "run_paired_seeded_teacher_trials"
            findings.append(AgentFinding(
                "curriculum_teacher_comparison_missing",
                FindingSeverity.INFO,
                "No hay control pareado para comparar local_7b con codex_frontier.",
            ))
        else:
            state, cls, proposal = AgentState.OBSERVED, "paired_evidence_candidate", "accumulate_repetitions_before_promotion"
        return AgentReport.create(
            agent_id=self.agent_id,
            role=AgentRole.CURRICULUM_LEARNING,
            identity=identity,
            state=state,
            authority_effect=AuthorityEffect.NONE,
            metrics={
                "lesson_count": len(lessons),
                "outcome_comparison_count": len(comparisons),
                "comparable_pair_count": comparable_pairs,
                "invalid_teacher_output_count": invalid_teacher_outputs,
                "teacher_source_counts": dict(source_counts),
            },
            findings=findings,
            outputs={
                "evidence_pipeline": ["measure", "classify", "analyze", "deliberate"],
                "stages": {
                    "measure": {"teacher_trials": measured},
                    "classify": {"curriculum_class": cls},
                    "analyze": {
                        "teacher_efficiency_ranked": comparable_pairs > 0,
                        "local_7b_assumed_efficient": False,
                        "codex_teacher_protocol_supported": True,
                    },
                    "deliberate": {
                        "proposal": proposal,
                        "curriculum_promotion_authorized": False,
                        "model_training_authorized": False,
                    },
                },
                "required_variants": ["no_teacher", "local_7b", "codex_frontier"],
                "decision_influence": "none",
            },
        )


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None
