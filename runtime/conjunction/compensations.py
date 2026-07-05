"""Operational compensation matrix."""

from __future__ import annotations

from .contracts import CompensationAction, OperationContext, ValidationFinding


_MATRIX = {
    "missing_evidence": (
        "expand_retrieval",
        {
            "recover_more_evidence": True,
            "try_exact_search": True,
            "reject_if_still_missing": True,
        },
    ),
    "contradictory_evidence": (
        "mark_conflict",
        {
            "prioritize_canonical_sources": True,
            "do_not_merge_incompatible_facts": True,
        },
    ),
    "unsupported_causal_assumption": (
        "downgrade_causal_claim",
        {
            "mark_as_hypothesis": True,
            "require_causal_evidence_before_strong_recommendation": True,
        },
    ),
    "compute_tier_exceeds_policy": (
        "degrade_compute_tier",
        {"use_cheapest_allowed_tier": True},
    ),
    "external_not_allowed": (
        "degrade_to_local",
        {"disable_external_reasoner": True, "advisory_only": True},
    ),
    "resource_pressure_limit": (
        "resource_conservation",
        {"reduce_context": True, "use_cache": True, "avoid_large_models": True},
    ),
    "high_risk_action": (
        "block_high_risk_action",
        {"automatic_execution": False},
    ),
    "action_not_permitted": (
        "block_action",
        {"automatic_execution": False},
    ),
    "critical_action_without_policy": (
        "require_agent_policy",
        {"automatic_execution": False},
    ),
    "agent_action_forbidden": (
        "stop_agent",
        {"automatic_execution": False},
    ),
    "agent_tool_forbidden": (
        "stop_agent",
        {"automatic_execution": False, "disable_tool": True},
    ),
    "agent_budget_exhausted": (
        "stop_agent",
        {"automatic_execution": False, "stop_condition": "budget_exhausted"},
    ),
    "agent_step_limit_exceeded": (
        "stop_agent",
        {"automatic_execution": False, "stop_condition": "step_limit_exceeded"},
    ),
    "human_approval_required": (
        "require_human_approval",
        {"automatic_execution": False, "request_human_approval": True},
    ),
    "autonomy_degraded_by_policy": (
        "degrade_autonomy_scope",
        {"active_mode": "bounded", "automatic_execution": True},
    ),
    "unbounded_autonomy_without_agent_policy": (
        "degrade_autonomy_scope",
        {"active_mode": "bounded", "automatic_execution": False},
    ),
    "unbounded_autonomy_not_authorized": (
        "degrade_autonomy_scope",
        {"active_mode": "bounded", "automatic_execution": False},
    ),
    "critical_action_without_validated_plan": (
        "require_plan_validation",
        {"automatic_execution": False, "create_plan_first": True},
    ),
    "critical_action_without_rollback_plan": (
        "require_rollback_plan",
        {"automatic_execution": False, "checkpoint_before_mutation": True},
    ),
}


class CompensationMatrix:
    """Map validation failures to operational correction behavior."""

    def compensate(
        self,
        *,
        context: OperationContext,
        findings: tuple[ValidationFinding, ...],
    ) -> tuple[CompensationAction, ...]:
        actions: list[CompensationAction] = []
        for finding in findings:
            if finding.status == "pass":
                continue
            code, directive = _MATRIX.get(
                finding.code,
                ("reduce_scope", {"reduce_scope": True, "avoid_automatic_execution": True}),
            )
            status = "blocked" if self._blocks(context=context, finding=finding) else "applied"
            directive = {
                **directive,
                "finding_code": finding.code,
                "finding_details": dict(finding.details),
            }
            actions.append(
                CompensationAction(
                    code=code,
                    status=status,
                    reason=finding.message,
                    directive=dict(directive),
                )
            )
        return tuple(actions)

    @staticmethod
    def _blocks(*, context: OperationContext, finding: ValidationFinding) -> bool:
        if finding.status == "fail" and context.is_critical_action():
            return True
        return finding.code in {
            "contradictory_evidence",
            "high_risk_action",
            "action_not_permitted",
            "agent_action_forbidden",
            "agent_tool_forbidden",
            "agent_budget_exhausted",
            "agent_step_limit_exceeded",
            "human_approval_required",
        }
