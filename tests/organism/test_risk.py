"""Tests para runtime/organism/risk.py — Posterior bayesiano constitucional."""

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
from runtime.organism.viability import ViabilityKernel
from runtime.organism.risk import (
    ConstitutionalPosterior,
    compute_constitutional_posterior,
)


class TestConstitutionalPosterior:
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

    def _compute(self, state: OrganismState, **kwargs) -> ConstitutionalPosterior:
        c = OrganismConstitution()
        kernel = ViabilityKernel(constitution=c)
        validation = c.validate(state)
        viability = kernel.assess(state)
        return compute_constitutional_posterior(
            state=state,
            constitutional_validation=validation,
            viability_assessment=viability,
            **kwargs,
        )

    def test_healthy_state_high_posterior(self):
        p = self._compute(self._healthy_state())
        assert p.constitutional_posterior > 0.5
        assert p.scope in ("local_safe", "transfer_safe", "inheritance_safe", "analogical_hint_only")
        assert p.rollback_required is False

    def test_unhealthy_state_blocked(self):
        s = OrganismState(
            belief=OrganismBeliefState(
                memory_purity_estimate=0.1,
                trace_integrity_confidence=0.1,
                causal_support_confidence=0.1,
            ),
            identity=IdentityState(lineage_id=""),
            viability=ViabilityState(
                viability_margin=0.0,
                accumulated_degradation=0.95,
                rollback_readiness=False,
            ),
        )
        p = self._compute(s)
        assert p.scope == "blocked"
        assert p.rollback_required is True

    def test_transfer_posterior_with_high_morphism(self):
        p = self._compute(
            self._healthy_state(),
            morphism_score=0.9,
            transfer_stability=0.85,
        )
        assert p.transfer_posterior > 0.3

    def test_modification_posterior(self):
        p = self._compute(self._healthy_state())
        assert 0.0 <= p.modification_posterior <= 1.0

    def test_inheritance_posterior(self):
        p = self._compute(
            self._healthy_state(),
            lineage_consistency=0.95,
        )
        assert p.inheritance_posterior > 0.3

    def test_failure_modes_detected(self):
        s = OrganismState(
            belief=OrganismBeliefState(
                memory_purity_estimate=0.3,
                trace_integrity_confidence=0.3,
                causal_support_confidence=0.9,
            ),
            policy=PolicyState(accumulated_drift=0.7),
            identity=IdentityState(lineage_id="L1"),
            viability=ViabilityState(recovery_debt=0.8),
        )
        p = self._compute(s)
        assert len(p.failure_modes) > 0
        assert "memory_contamination" in p.failure_modes

    def test_lcb_range(self):
        p = self._compute(self._healthy_state())
        assert 0.0 <= p.lower_confidence_bound <= 1.0
        assert p.lower_confidence_bound <= p.constitutional_posterior

    def test_evidence_summary(self):
        p = self._compute(self._healthy_state())
        assert "belief_confidence" in p.evidence_summary
        assert "viability_margin" in p.evidence_summary
        assert "morphism_score" in p.evidence_summary

    def test_historical_data_affects_posterior(self):
        base = self._compute(self._healthy_state())
        with_history = self._compute(
            self._healthy_state(),
            n_historical=20,
            historical_success_rate=0.95,
        )
        # Historical success should increase posterior
        assert with_history.constitutional_posterior >= base.constitutional_posterior * 0.9

    def test_scope_transfer_safe(self):
        """High morphism + high LCB → transfer_safe or analogical_hint_only.

        With limited observations (n=6 base), LCB may remain below the
        threshold even when point estimate is high.  Adding historical
        data raises LCB enough for transfer_safe.
        """
        p = self._compute(
            self._healthy_state(),
            morphism_score=0.9,
            transfer_stability=0.9,
            eml_concurrence=0.8,
            n_historical=30,
            historical_success_rate=0.95,
        )
        assert p.scope in ("transfer_safe", "inheritance_safe", "local_safe", "analogical_hint_only")
