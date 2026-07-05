"""Validators for operational conjunction."""

from __future__ import annotations

from .contracts import ComputeRoute, OperationContext, ValidationFinding, tier_rank


class OperationalValidatorStack:
    """Run schema, evidence, causal, constraint, risk, and agent validators."""

    def validate(
        self,
        *,
        context: OperationContext,
        route: ComputeRoute,
    ) -> tuple[ValidationFinding, ...]:
        findings: list[ValidationFinding] = []
        findings.extend(self._schema(context=context, route=route))
        findings.extend(self._evidence(context=context))
        findings.extend(self._causal(context=context))
        findings.extend(self._constraints(context=context, route=route))
        findings.extend(self._risk(context=context))
        findings.extend(self._agent_execution(context=context))
        if not findings:
            findings.append(
                ValidationFinding(
                    code="validation_passed",
                    status="pass",
                    validator="stack",
                    message="all validators passed",
                )
            )
        return tuple(findings)

    def _schema(
        self,
        *,
        context: OperationContext,
        route: ComputeRoute,
    ) -> list[ValidationFinding]:
        out: list[ValidationFinding] = []
        if not context.operation_id or not context.user_intent:
            out.append(
                ValidationFinding(
                    code="invalid_operation_contract",
                    status="fail",
                    validator="schema",
                    message="operation_id and user_intent are required",
                )
            )
        if not route.selected_compute_tier or not route.selected_reasoning_path:
            out.append(
                ValidationFinding(
                    code="invalid_route_contract",
                    status="fail",
                    validator="schema",
                    message="selected compute tier and reasoning path are required",
                )
            )
        return out

    def _evidence(self, *, context: OperationContext) -> list[ValidationFinding]:
        out: list[ValidationFinding] = []
        evidence_kinds = context.evidence_kinds()
        for kind in context.required_evidence:
            if kind not in evidence_kinds:
                out.append(
                    ValidationFinding(
                        code="missing_evidence",
                        status="fail" if context.is_critical_action() else "warn",
                        validator="evidence",
                        message=f"required evidence missing: {kind}",
                        details={"missing_kind": kind},
                    )
                )

        conflicts = [
            item for item in context.available_evidence
            if item.kind == "conflict" or bool(item.payload.get("conflict"))
        ]
        if conflicts:
            out.append(
                ValidationFinding(
                    code="contradictory_evidence",
                    status="fail",
                    validator="evidence",
                    message="contradictory evidence must not be fused",
                    details={"conflict_count": len(conflicts)},
                )
            )
        return out

    def _causal(self, *, context: OperationContext) -> list[ValidationFinding]:
        out: list[ValidationFinding] = []
        for assumption in context.causal_assumptions:
            if assumption.supported:
                continue
            status = "fail" if context.task_type in {"causal_decision", "self_modification"} else "warn"
            out.append(
                ValidationFinding(
                    code="unsupported_causal_assumption",
                    status=status,
                    validator="causal",
                    message="causal assumption lacks support and must be downgraded",
                    details={"assumption": assumption.name},
                )
            )
        return out

    def _constraints(
        self,
        *,
        context: OperationContext,
        route: ComputeRoute,
    ) -> list[ValidationFinding]:
        out: list[ValidationFinding] = []
        if tier_rank(route.selected_compute_tier) > tier_rank(context.constraints.max_compute_tier):
            out.append(
                ValidationFinding(
                    code="compute_tier_exceeds_policy",
                    status="fail",
                    validator="constraints",
                    message="selected compute tier exceeds policy",
                    details={
                        "selected": route.selected_compute_tier,
                        "max": context.constraints.max_compute_tier,
                    },
                )
            )
        if route.selected_compute_tier == "tier_3_external" and not context.constraints.allow_external:
            out.append(
                ValidationFinding(
                    code="external_not_allowed",
                    status="fail",
                    validator="constraints",
                    message="external compute was selected while policy forbids it",
                )
            )
        if context.resource_pressure >= context.constraints.resource_pressure_limit:
            out.append(
                ValidationFinding(
                    code="resource_pressure_limit",
                    status="warn" if route.selected_compute_tier == "tier_0_deterministic" else "fail",
                    validator="constraints",
                    message="resource pressure requires cheap/degraded route",
                    details={"resource_pressure": context.resource_pressure},
                )
            )
        return out

    def _risk(self, *, context: OperationContext) -> list[ValidationFinding]:
        out: list[ValidationFinding] = []
        if context.risk_score >= 0.80 and context.requested_action not in {
            "observe",
            "sleep",
            "rollback",
            "quarantine",
            "shutdown",
        }:
            out.append(
                ValidationFinding(
                    code="high_risk_action",
                    status="fail",
                    validator="risk",
                    message="high risk action cannot proceed automatically",
                    details={"risk_score": context.risk_score},
                )
            )
        if context.requested_action not in context.constraints.permitted_actions:
            out.append(
                ValidationFinding(
                    code="action_not_permitted",
                    status="fail",
                    validator="risk",
                    message="requested action is outside operational permissions",
                    details={"requested_action": context.requested_action},
                )
            )
        return out

    def _agent_execution(self, *, context: OperationContext) -> list[ValidationFinding]:
        out: list[ValidationFinding] = []
        policy = context.agent_policy
        if policy is None:
            if context.is_critical_action():
                out.append(
                    ValidationFinding(
                        code="critical_action_without_policy",
                        status="fail",
                        validator="agent_policy",
                        message="critical action requires an agent policy",
                    )
                )
            return out

        if not policy.allows(context.requested_action):
            out.append(
                ValidationFinding(
                    code="agent_action_forbidden",
                    status="fail",
                    validator="agent_policy",
                    message="agent policy forbids requested action",
                    details={"role": policy.role, "requested_action": context.requested_action},
                )
            )
        requested_tool = context.metadata.get("requested_tool")
        if requested_tool and not policy.allows_tool(str(requested_tool)):
            out.append(
                ValidationFinding(
                    code="agent_tool_forbidden",
                    status="fail",
                    validator="agent_policy",
                    message="agent policy forbids requested tool",
                    details={"role": policy.role, "requested_tool": requested_tool},
                )
            )
        if policy.budget_units <= 0:
            out.append(
                ValidationFinding(
                    code="agent_budget_exhausted",
                    status="fail",
                    validator="agent_policy",
                    message="agent policy budget is exhausted",
                    details={"role": policy.role, "budget_units": policy.budget_units},
                )
            )
        step_count = int(context.metadata.get("agent_step_count", 0) or 0)
        if step_count > policy.max_steps:
            out.append(
                ValidationFinding(
                    code="agent_step_limit_exceeded",
                    status="fail",
                    validator="agent_policy",
                    message="agent exceeded maximum policy steps",
                    details={
                        "role": policy.role,
                        "agent_step_count": step_count,
                        "max_steps": policy.max_steps,
                        "stop_conditions": policy.stop_conditions,
                    },
                )
            )
        if (
            context.requested_action in policy.human_approval_required_actions
            and not context.has_evidence("human_approval")
        ):
            out.append(
                ValidationFinding(
                    code="human_approval_required",
                    status="fail",
                    validator="agent_policy",
                    message="requested action requires human approval evidence",
                    details={"role": policy.role, "requested_action": context.requested_action},
                )
            )
        if (
            context.is_critical_action()
            and policy.requires_plan_for_critical
            and context.requested_action not in {"rollback", "shutdown"}
        ):
            if not context.has_evidence("validated_plan"):
                out.append(
                    ValidationFinding(
                        code="critical_action_without_validated_plan",
                        status="fail",
                        validator="agent_policy",
                        message="critical action requires a validated plan evidence item",
                        details={"role": policy.role, "requested_action": context.requested_action},
                    )
                )
        if context.is_critical_action() and policy.rollback_required_for_critical:
            if not context.has_evidence("rollback_plan") and context.requested_action not in {
                "rollback",
                "shutdown",
            }:
                out.append(
                    ValidationFinding(
                        code="critical_action_without_rollback_plan",
                        status="fail",
                        validator="agent_policy",
                        message="critical action requires a rollback plan",
                        details={"role": policy.role, "requested_action": context.requested_action},
                    )
                )
        return out
