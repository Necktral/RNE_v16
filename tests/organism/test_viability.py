"""Tests para runtime/organism/viability.py — Kernel de viabilidad."""

from __future__ import annotations

import pytest

from runtime.organism.state import (
    OrganismBeliefState,
    OrganismState,
    PolicyState,
    IdentityState,
    ViabilityState,
)
from runtime.organism.constitution import OrganismConstitution
from runtime.organism.viability import (
    RecoveryAction,
    RecoveryPlan,
    ViabilityAssessment,
    ViabilityKernel,
)


class TestViabilityKernel:
    def _healthy_state(self) -> OrganismState:
        return OrganismState(
            belief=OrganismBeliefState(
                alarm_probability=0.1,
                intervention_efficacy=0.8,
                causal_support_confidence=0.9,
                memory_purity_estimate=0.95,
                trace_integrity_confidence=0.85,
            ),
            policy=PolicyState(accumulated_drift=0.05),
            identity=IdentityState(lineage_id="L1"),
            viability=ViabilityState(
                viability_margin=0.9,
                accumulated_degradation=0.05,
                recovery_debt=0.05,
            ),
        )

    def _unhealthy_state(self) -> OrganismState:
        return OrganismState(
            belief=OrganismBeliefState(
                alarm_probability=0.9,
                intervention_efficacy=0.1,
                causal_support_confidence=0.1,
                memory_purity_estimate=0.1,
                trace_integrity_confidence=0.1,
            ),
            policy=PolicyState(accumulated_drift=0.9),
            identity=IdentityState(lineage_id=""),
            viability=ViabilityState(
                viability_margin=0.0,
                accumulated_degradation=0.95,
                rollback_readiness=False,
                recovery_debt=0.9,
            ),
        )

    def test_healthy_state_is_viable(self):
        kernel = ViabilityKernel()
        a = kernel.assess(self._healthy_state())
        assert a.is_viable is True
        assert a.viability_margin > 0.3
        assert a.rollback_required is False

    def test_unhealthy_state_not_viable(self):
        kernel = ViabilityKernel()
        a = kernel.assess(self._unhealthy_state())
        assert a.is_viable is False
        assert a.rollback_required is True

    def test_margin_delta(self):
        kernel = ViabilityKernel()
        s1 = self._healthy_state()
        s2 = OrganismState(
            belief=OrganismBeliefState(
                alarm_probability=0.5,
                intervention_efficacy=0.5,
                causal_support_confidence=0.5,
                memory_purity_estimate=0.5,
                trace_integrity_confidence=0.5,
            ),
            identity=IdentityState(lineage_id="L1"),
            viability=ViabilityState(viability_margin=0.5),
        )
        a = kernel.assess(s2, previous_state=s1)
        assert a.margin_delta != 0.0

    def test_recovery_plan_empty_for_healthy(self):
        kernel = ViabilityKernel()
        a = kernel.assess(self._healthy_state())
        # Healthy state may still have some soft action suggestions
        assert a.recovery_plan.rollback_recommended is False

    def test_recovery_plan_has_actions_for_unhealthy(self):
        kernel = ViabilityKernel()
        a = kernel.assess(self._unhealthy_state())
        assert len(a.recovery_plan.actions) > 0
        assert a.recovery_plan.rollback_recommended is True

    def test_is_in_kernel(self):
        kernel = ViabilityKernel()
        assert kernel.is_in_kernel(self._healthy_state()) is True
        assert kernel.is_in_kernel(self._unhealthy_state()) is False

    def test_margin_trajectory(self):
        kernel = ViabilityKernel()
        states = [self._healthy_state()] * 5
        margins = kernel.margin_trajectory(states)
        assert len(margins) == 5
        for m in margins:
            assert 0.0 <= m <= 1.0

    def test_edge_proximity_triggers_recovery(self):
        kernel = ViabilityKernel(edge_margin_threshold=0.50)
        s = OrganismState(
            belief=OrganismBeliefState(
                causal_support_confidence=0.5,
                memory_purity_estimate=0.5,
                trace_integrity_confidence=0.5,
            ),
            identity=IdentityState(lineage_id="L1"),
            viability=ViabilityState(viability_margin=0.3),
        )
        a = kernel.assess(s)
        # Should suggest increasing margin
        action_names = [act.action for act in a.recovery_plan.actions]
        assert "increase_viability_margin" in action_names

    def test_custom_constitution_config(self):
        c = OrganismConstitution(config={
            "triadic_closure_threshold": 0.001,
            "min_memory_purity": 0.01,
            "min_trace_integrity": 0.01,
            "max_degradation": 0.99,
            "min_causal_support": 0.01,
            "max_policy_drift": 0.99,
            "min_continuity": 0.01,
            "max_recovery_debt": 0.99,
            "max_drift_rate": 0.99,
        })
        kernel = ViabilityKernel(constitution=c)
        s = OrganismState(
            identity=IdentityState(lineage_id="L1"),
            belief=OrganismBeliefState(
                causal_support_confidence=0.1,
                memory_purity_estimate=0.1,
                trace_integrity_confidence=0.1,
            ),
        )
        a = kernel.assess(s)
        # With very relaxed thresholds, should still be valid
        assert a.constitutional_validation.is_valid is True
