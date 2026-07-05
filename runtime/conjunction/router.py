"""Compute routing policy for operational conjunction."""

from __future__ import annotations

from .contracts import ComputeRoute, ComputeTier, OperationContext, tier_rank


class ComputeRouter:
    """Select the cheapest compute tier that can satisfy the operation."""

    def route(self, context: OperationContext) -> ComputeRoute:
        trace: list[dict] = [
            {
                "stage": "router.input",
                "task_type": context.task_type,
                "requested_action": context.requested_action,
                "risk_score": context.risk_score,
                "complexity_score": context.complexity_score,
                "uncertainty_score": context.uncertainty_score,
                "resource_pressure": context.resource_pressure,
            }
        ]

        if context.resource_pressure >= context.constraints.resource_pressure_limit:
            return self._route(
                tier="tier_0_deterministic",
                path="deterministic.resource_conservation",
                reason="resource_pressure_exceeds_policy",
                expected_quality=0.45,
                estimated_cost=0.05,
                trace=trace,
            )

        missing_evidence = [
            kind for kind in context.required_evidence if not context.has_evidence(kind)
        ]
        if missing_evidence:
            return self._bounded(
                context,
                preferred="tier_1_local_light",
                path="evidence_recovery_and_validation",
                reason=f"missing_required_evidence:{','.join(missing_evidence)}",
                expected_quality=0.55,
                estimated_cost=0.20,
                trace=trace,
            )

        if context.is_critical_action():
            return self._bounded(
                context,
                preferred="tier_0_deterministic",
                path="policy_gate.validators.rollback_guard",
                reason="critical_action_requires_policy_validation",
                expected_quality=0.75,
                estimated_cost=0.10,
                trace=trace,
            )

        causal_needed = bool(context.causal_assumptions) or context.task_type == "causal_decision"
        unsupported_causal = any(not item.supported for item in context.causal_assumptions)
        if causal_needed and not unsupported_causal:
            return self._bounded(
                context,
                preferred="tier_2_specialized",
                path="causal_signature.reasoning_families.validators",
                reason="causal_support_available",
                expected_quality=0.78,
                estimated_cost=0.45,
                trace=trace,
            )

        if unsupported_causal:
            return self._bounded(
                context,
                preferred="tier_1_local_light",
                path="causal_hypothesis_guard",
                reason="causal_support_missing",
                expected_quality=0.50,
                estimated_cost=0.18,
                trace=trace,
            )

        if (
            context.uncertainty_score >= 0.75
            and context.constraints.allow_external
            and tier_rank(context.constraints.max_compute_tier) >= tier_rank("tier_3_external")
        ):
            return self._route(
                tier="tier_3_external",
                path="external_reasoner.gated_advisory",
                reason="high_uncertainty_external_allowed",
                expected_quality=0.82,
                estimated_cost=0.95,
                trace=trace,
            )

        if context.complexity_score <= 0.25 and context.uncertainty_score <= 0.45:
            return self._route(
                tier="tier_0_deterministic",
                path="deterministic.rules.cache.validators",
                reason="simple_low_uncertainty_operation",
                expected_quality=0.70,
                estimated_cost=0.05,
                trace=trace,
            )

        return self._bounded(
            context,
            preferred="tier_1_local_light",
            path="local_light.reasoning_families.rag_lite",
            reason="default_local_light_route",
            expected_quality=0.65,
            estimated_cost=0.25,
            trace=trace,
        )

    def _bounded(
        self,
        context: OperationContext,
        *,
        preferred: ComputeTier,
        path: str,
        reason: str,
        expected_quality: float,
        estimated_cost: float,
        trace: list[dict],
    ) -> ComputeRoute:
        max_tier = context.constraints.max_compute_tier
        tier = preferred
        if tier_rank(preferred) > tier_rank(max_tier):
            tier = max_tier
            reason = f"{reason};degraded_to:{max_tier}"
            path = f"degraded.{path}"
        return self._route(
            tier=tier,
            path=path,
            reason=reason,
            expected_quality=expected_quality,
            estimated_cost=estimated_cost,
            trace=trace,
        )

    @staticmethod
    def _route(
        *,
        tier: ComputeTier,
        path: str,
        reason: str,
        expected_quality: float,
        estimated_cost: float,
        trace: list[dict],
    ) -> ComputeRoute:
        return ComputeRoute(
            selected_compute_tier=tier,
            selected_reasoning_path=path,
            reason=reason,
            expected_quality=round(float(expected_quality), 4),
            estimated_cost=round(float(estimated_cost), 4),
            trace=tuple(
                [
                    *trace,
                    {
                        "stage": "router.output",
                        "selected_compute_tier": tier,
                        "selected_reasoning_path": path,
                        "reason": reason,
                    },
                ]
            ),
        )
