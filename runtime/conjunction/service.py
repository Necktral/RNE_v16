"""Operational conjunction service.

This service is the coordination brain requested by the "conjunction +
compensations" design: it does not replace the organism runtime, it governs
which route may act and records why.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

from runtime.memory.mfm_lite.retrieval import MemoryRetrieval
from runtime.storage.records import utc_now_iso
from runtime.world.registry import get_scenario

from .compensations import CompensationMatrix
from .contracts import (
    AgentPolicy,
    CausalAssumption,
    CompensationStatus,
    ComputeTier,
    ConfidenceState,
    EvidenceItem,
    FinalDecision,
    OperationalConjunctionResult,
    OperationalConstraints,
    OperationContext,
    TaskType,
    ValidationFinding,
    ValidationStatus,
)
from .router import ComputeRouter
from .validators import OperationalValidatorStack


class OperationalConjunctionLayer:
    """Coordinate evidence, causality, compute, validators, and compensations."""

    def __init__(self, *, storage=None):
        self.storage = storage
        self.router = ComputeRouter()
        self.validators = OperationalValidatorStack()
        self.compensations = CompensationMatrix()

    def evaluate(self, context: OperationContext) -> OperationalConjunctionResult:
        route = self.router.route(context)
        findings = self.validators.validate(context=context, route=route)
        actions = self.compensations.compensate(context=context, findings=findings)
        validation_status = self._validation_status(findings)
        compensation_status = self._compensation_status(actions, validation_status)
        confidence_state = self._confidence_state(context=context, findings=findings)
        final_decision = self._final_decision(
            context=context,
            findings=findings,
            compensation_status=compensation_status,
        )
        permissions = {
            "requested_action": context.requested_action,
            "is_critical": context.is_critical_action(),
            "agent_role": context.agent_policy.role if context.agent_policy else None,
            "agent_policy_allows": (
                context.agent_policy.allows(context.requested_action)
                if context.agent_policy else not context.is_critical_action()
            ),
            "automatic_execution_allowed": final_decision in {
                "allow",
                "allow_with_compensation",
                "degrade",
                "observe",
            },
        }
        trace = (
            *route.trace,
            {
                "stage": "validation",
                "status": validation_status,
                "findings": [item.to_dict() for item in findings],
            },
            {
                "stage": "compensation",
                "status": compensation_status,
                "actions": [item.to_dict() for item in actions],
            },
            {
                "stage": "final_decision",
                "decision": final_decision,
                "confidence_state": confidence_state,
            },
        )
        result = OperationalConjunctionResult(
            operation_id=context.operation_id,
            selected_compute_tier=route.selected_compute_tier,
            selected_reasoning_path=route.selected_reasoning_path,
            validation_status=validation_status,
            compensation_status=compensation_status,
            confidence_state=confidence_state,
            execution_permissions=permissions,
            final_decision=final_decision,
            validation_findings=findings,
            compensations=actions,
            trace=trace,
            context_summary=self._context_summary(context),
        )
        self._persist_result(context=context, result=result)
        return result

    def evaluate_life_cycle(
        self,
        *,
        run_id: str,
        scenario: str,
        decision: Any,
        vitals: Any,
        goals: Iterable[Any],
        step_index: int,
        external_input: float | None,
        allow_external_reasoner: bool = False,
        max_compute_tier: ComputeTier = "tier_2_specialized",
    ) -> OperationalConjunctionResult:
        context = self._build_life_context(
            run_id=run_id,
            scenario=scenario,
            decision=decision,
            vitals=vitals,
            goals=goals,
            step_index=step_index,
            external_input=external_input,
            allow_external_reasoner=allow_external_reasoner,
            max_compute_tier=max_compute_tier,
        )
        return self.evaluate(context)

    def _build_life_context(
        self,
        *,
        run_id: str,
        scenario: str,
        decision: Any,
        vitals: Any,
        goals: Iterable[Any],
        step_index: int,
        external_input: float | None,
        allow_external_reasoner: bool,
        max_compute_tier: ComputeTier,
    ) -> OperationContext:
        action = str(getattr(decision, "action", "act"))
        task_type = self._task_type(action)
        required = ["vital_signs", "causal_signature"]
        if action == "self_modify":
            required.extend(["healthy_checkpoint", "validated_plan", "rollback_plan"])
        elif action == "consult_external":
            required.extend(["validated_plan"])
        elif action == "rollback":
            required.append("healthy_checkpoint")

        evidence = self._life_evidence(
            run_id=run_id,
            scenario=scenario,
            vitals=vitals,
            goals=goals,
            action=action,
            step_index=step_index,
            external_input=external_input,
            decision=decision,
        )
        causal = self._causal_assumptions(scenario=scenario, evidence=evidence)
        resource_pressure = float(getattr(vitals, "resource_pressure", 0.0) or 0.0)
        constraints = OperationalConstraints(
            max_compute_tier=(
                "tier_0_deterministic" if resource_pressure >= 0.90 else max_compute_tier
            ),
            allow_external=bool(allow_external_reasoner),
            resource_pressure_limit=0.85,
            permitted_actions=self._permitted_actions(allow_external_reasoner),
        )
        agent_policy = AgentPolicy(
            role="life_kernel",
            allowed_actions=self._permitted_actions(allow_external_reasoner),
            prohibited_actions=() if allow_external_reasoner else ("consult_external",),
            budget_units=1.0,
            requires_plan_for_critical=True,
            rollback_required_for_critical=True,
        )
        complexity = self._complexity_for(action=action, goals=goals)
        uncertainty = self._uncertainty(vitals=vitals, evidence=evidence)
        raw_risk = float(getattr(vitals, "risk_score", 0.0) or 0.0)
        if int(getattr(vitals, "episode_count", 0) or 0) <= 0 and action == "act":
            raw_risk = min(raw_risk, 0.50)
        return OperationContext.create(
            run_id=run_id,
            user_intent="maintain_autonomous_life_cycle",
            task_type=task_type,
            requested_action=action,
            scenario=scenario,
            required_evidence=required,
            available_evidence=evidence,
            causal_assumptions=causal,
            constraints=constraints,
            agent_policy=agent_policy,
            risk_score=raw_risk,
            complexity_score=complexity,
            resource_pressure=resource_pressure,
            uncertainty_score=uncertainty,
            metadata={
                "step_index": step_index,
                "external_input": external_input,
                "decision_id": getattr(decision, "decision_id", None),
                "decision_reason": getattr(decision, "reason", None),
            },
        )

    def _life_evidence(
        self,
        *,
        run_id: str,
        scenario: str,
        vitals: Any,
        goals: Iterable[Any],
        action: str,
        step_index: int,
        external_input: float | None,
        decision: Any,
    ) -> tuple[EvidenceItem, ...]:
        evidence: list[EvidenceItem] = [
            EvidenceItem.create(
                kind="vital_signs",
                source="life.vitals",
                confidence=0.90,
                payload=vitals.to_dict() if hasattr(vitals, "to_dict") else {},
                canonical=True,
            )
        ]
        try:
            signature = get_scenario(scenario).causal_signature
            evidence.append(
                EvidenceItem.create(
                    kind="causal_signature",
                    source="world.registry",
                    confidence=self._signature_confidence(signature),
                    payload={
                        "scenario": signature.scenario_name,
                        "main_variable": signature.main_variable,
                        "edge_count": len(signature.causal_edges),
                        "intervention_count": len(signature.intervention_effects),
                    },
                    canonical=True,
                )
            )
        except Exception as exc:
            evidence.append(
                EvidenceItem.create(
                    kind="conflict",
                    source="world.registry",
                    confidence=0.10,
                    payload={"conflict": True, "error": f"{type(exc).__name__}: {exc}"},
                )
            )

        memory_hits = self._retrieve_life_memory(
            run_id=run_id,
            scenario=scenario,
            action=action,
            external_input=external_input,
        )
        if memory_hits:
            top = memory_hits[0]
            evidence.append(
                EvidenceItem.create(
                    kind="memory_rag",
                    source="memory.mfm_lite",
                    confidence=float(top.get("score", 0.0) or 0.0),
                    payload={"hit_count": len(memory_hits), "top": top},
                )
            )

        checkpoint = self._latest_checkpoint(run_id=run_id)
        if checkpoint is not None:
            evidence.append(
                EvidenceItem.create(
                    kind="healthy_checkpoint",
                    source="life.checkpoint",
                    confidence=0.80 if checkpoint.get("healthy") else 0.45,
                    payload=checkpoint,
                    canonical=bool(checkpoint.get("healthy")),
                )
            )
            evidence.append(
                EvidenceItem.create(
                    kind="rollback_plan",
                    source="life.checkpoint",
                    confidence=0.75,
                    payload={
                        "plan": "restore_latest_healthy_checkpoint",
                        "checkpoint_artifact_id": checkpoint.get("artifact_id"),
                    },
                    canonical=True,
                )
            )

        directives = getattr(decision, "directives", {}) or {}
        if directives.get("validated_plan") is True:
            evidence.append(
                EvidenceItem.create(
                    kind="validated_plan",
                    source="autonomy_supervisor",
                    confidence=0.75,
                    payload={"step_index": step_index, "action": action},
                    canonical=True,
                )
            )
        return tuple(evidence)

    def _retrieve_life_memory(
        self,
        *,
        run_id: str,
        scenario: str,
        action: str,
        external_input: float | None,
    ) -> list[Dict[str, Any]]:
        if self.storage is None:
            return []
        try:
            return MemoryRetrieval(storage=self.storage).retrieve(
                run_id=run_id,
                query={
                    "scenario": scenario,
                    "action": action,
                    "external_input_band": "high" if (external_input or 0.0) >= 0.10 else "normal",
                },
                limit=3,
                scenario_name=scenario,
                scenario_filter_mode="strict_same_scenario",
            )
        except Exception:
            return []

    def _latest_checkpoint(self, *, run_id: str) -> Dict[str, Any] | None:
        if self.storage is None:
            return None
        try:
            from runtime.life.checkpoints import LIFE_CHECKPOINT_KIND

            artifacts = self.storage.list_artifacts(
                run_id=run_id,
                kind=LIFE_CHECKPOINT_KIND,
                limit=1,
            )
        except Exception:
            return None
        if not artifacts:
            return None
        artifact = artifacts[0]
        return {
            "artifact_id": artifact.artifact_id,
            "healthy": bool((artifact.metadata or {}).get("healthy")),
            "created_at": artifact.created_at,
        }

    def _causal_assumptions(
        self,
        *,
        scenario: str,
        evidence: tuple[EvidenceItem, ...],
    ) -> tuple[CausalAssumption, ...]:
        try:
            signature = get_scenario(scenario).causal_signature
        except Exception:
            return (
                CausalAssumption(
                    name="scenario_causal_signature",
                    statement="scenario must expose causal support",
                    supported=False,
                    strength=0.0,
                ),
            )
        evidence_ids = tuple(
            item.evidence_id for item in evidence if item.kind == "causal_signature"
        )
        strength = self._signature_confidence(signature)
        return (
            CausalAssumption(
                name="scenario_causal_graph",
                statement="interventions are evaluated against scenario causal signature",
                supported=bool(signature.causal_edges and signature.intervention_effects),
                strength=strength,
                support_evidence_ids=evidence_ids,
            ),
        )

    @staticmethod
    def _signature_confidence(signature: Any) -> float:
        edges = list(getattr(signature, "causal_edges", ()) or ())
        if not edges:
            return 0.0
        return round(sum(float(getattr(edge, "strength", 0.0)) for edge in edges) / len(edges), 4)

    @staticmethod
    def _task_type(action: str) -> TaskType:
        if action == "self_modify":
            return "self_modification"
        if action == "consult_external":
            return "external_consultation"
        if action in {"rollback", "quarantine", "shutdown", "sleep"}:
            return "maintenance"
        if action in {"act", "explore"}:
            return "life_cycle"
        return "agent_action"

    @staticmethod
    def _permitted_actions(allow_external_reasoner: bool) -> tuple[str, ...]:
        base = (
            "act",
            "observe",
            "explore",
            "sleep",
            "self_modify",
            "rollback",
            "quarantine",
            "shutdown",
        )
        if allow_external_reasoner:
            return (*base, "consult_external")
        return base

    @staticmethod
    def _complexity_for(*, action: str, goals: Iterable[Any]) -> float:
        if action in {"self_modify", "rollback", "consult_external"}:
            return 0.80
        if action == "explore":
            return 0.55
        active_goals = [goal for goal in goals if getattr(goal, "status", "active") == "active"]
        return min(0.65, 0.20 + 0.05 * len(active_goals))

    @staticmethod
    def _uncertainty(*, vitals: Any, evidence: tuple[EvidenceItem, ...]) -> float:
        cognitive_quality = float(getattr(vitals, "cognitive_quality", 0.5) or 0.5)
        memory_bonus = 0.20 if any(item.kind == "memory_rag" for item in evidence) else 0.0
        conflict_penalty = 0.35 if any(item.kind == "conflict" for item in evidence) else 0.0
        return max(0.0, min(1.0, 1.0 - cognitive_quality - memory_bonus + conflict_penalty))

    @staticmethod
    def _validation_status(findings: tuple[ValidationFinding, ...]) -> ValidationStatus:
        if any(item.status == "fail" for item in findings):
            return "fail"
        if any(item.status == "warn" for item in findings):
            return "warn"
        return "pass"

    @staticmethod
    def _compensation_status(
        actions: tuple[Any, ...],
        validation_status: ValidationStatus,
    ) -> CompensationStatus:
        if any(item.status == "blocked" for item in actions):
            return "blocked"
        if actions:
            return "required" if validation_status == "fail" else "applied"
        return "none"

    @staticmethod
    def _confidence_state(
        *,
        context: OperationContext,
        findings: tuple[ValidationFinding, ...],
    ) -> ConfidenceState:
        if any(item.code == "contradictory_evidence" for item in findings):
            return "conflicted"
        if context.uncertainty_score >= 0.70 or any(item.status == "fail" for item in findings):
            return "low"
        if context.uncertainty_score >= 0.40 or any(item.status == "warn" for item in findings):
            return "medium"
        return "high"

    @staticmethod
    def _final_decision(
        *,
        context: OperationContext,
        findings: tuple[ValidationFinding, ...],
        compensation_status: CompensationStatus,
    ) -> FinalDecision:
        if compensation_status == "blocked":
            return "block"
        if any(item.status == "fail" for item in findings):
            return "block" if context.is_critical_action() else "degrade"
        if any(item.status == "warn" for item in findings):
            return "allow_with_compensation"
        return "allow"

    @staticmethod
    def _context_summary(context: OperationContext) -> Dict[str, Any]:
        return {
            "run_id": context.run_id,
            "task_type": context.task_type,
            "requested_action": context.requested_action,
            "scenario": context.scenario,
            "required_evidence": list(context.required_evidence),
            "available_evidence_kinds": sorted(context.evidence_kinds()),
            "risk_score": context.risk_score,
            "complexity_score": context.complexity_score,
            "resource_pressure": context.resource_pressure,
            "uncertainty_score": context.uncertainty_score,
        }

    def _persist_result(
        self,
        *,
        context: OperationContext,
        result: OperationalConjunctionResult,
    ) -> None:
        if self.storage is None:
            return
        try:
            self.storage.append_event(
                event_type="operational.conjunction.evaluated",
                run_id=context.run_id,
                source="operational_conjunction",
                payload={
                    "timestamp": utc_now_iso(),
                    "context": context.to_dict(),
                    "result": result.to_dict(),
                },
            )
        except Exception:
            return
