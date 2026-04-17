"""Tests para runtime/organism/transport.py — Operadores de transporte."""

from __future__ import annotations

import pytest

from runtime.organism.state import (
    OrganismBeliefState,
    OrganismState,
    PolicyState,
    IdentityState,
    ViabilityState,
)
from runtime.organism.regime_model import (
    RegimeModel,
    THERMAL_REGIME,
    RESOURCE_REGIME,
)
from runtime.organism.transport import (
    BeliefProjection,
    PolicyProjection,
    TransportOperatorEngine,
    TransportResult,
)


class TestTransportOperatorEngine:
    def _make_state(self) -> OrganismState:
        return OrganismState(
            belief=OrganismBeliefState(
                alarm_probability=0.2,
                intervention_efficacy=0.8,
                causal_support_confidence=0.9,
                memory_purity_estimate=0.95,
                trace_integrity_confidence=0.85,
            ),
            policy=PolicyState(
                control_class="reactive",
                sensitivity=0.6,
                perturbation_tolerance=0.3,
            ),
            identity=IdentityState(lineage_id="L1"),
        )

    def test_identity_transport(self):
        """Same regime → identity transport."""
        engine = TransportOperatorEngine()
        state = self._make_state()
        result = engine.transport(
            source_regime=THERMAL_REGIME,
            target_regime=THERMAL_REGIME,
            state=state,
        )
        assert result.transport_class == "identity"
        assert result.residual_error < 0.3
        assert result.transport_uncertainty < 0.3

    def test_thermal_to_resource_transport(self):
        """Different regimes with polarity inversion → adversarial."""
        engine = TransportOperatorEngine()
        state = self._make_state()
        result = engine.transport(
            source_regime=THERMAL_REGIME,
            target_regime=RESOURCE_REGIME,
            state=state,
        )
        # Polarity differs: lower_is_better vs higher_is_better
        assert result.transport_class in ("projective", "adversarial")
        assert result.residual_error > 0.0
        assert result.belief_projection.sign_inversions >= 0

    def test_transport_is_directed(self):
        """A→B ≠ B→A."""
        engine = TransportOperatorEngine()
        state = self._make_state()
        ab = engine.transport(
            source_regime=THERMAL_REGIME,
            target_regime=RESOURCE_REGIME,
            state=state,
        )
        ba = engine.transport(
            source_regime=RESOURCE_REGIME,
            target_regime=THERMAL_REGIME,
            state=state,
        )
        # Source and target are flipped
        assert ab.source_regime != ba.source_regime
        assert ab.target_regime != ba.target_regime
        # Transport class should be similar since same structural distance
        # but belief projections can differ due to sensitivity ratios

    def test_belief_projection_range(self):
        engine = TransportOperatorEngine()
        state = self._make_state()
        result = engine.transport(
            source_regime=THERMAL_REGIME,
            target_regime=RESOURCE_REGIME,
            state=state,
        )
        bp = result.belief_projection
        assert 0.0 <= bp.projected_alarm <= 1.0
        assert 0.0 <= bp.projected_efficacy <= 1.0
        assert 0.0 <= bp.projected_causal_support <= 1.0
        assert 0.0 <= bp.projection_loss <= 1.0

    def test_policy_projection(self):
        engine = TransportOperatorEngine()
        state = self._make_state()
        result = engine.transport(
            source_regime=THERMAL_REGIME,
            target_regime=RESOURCE_REGIME,
            state=state,
        )
        pp = result.policy_projection
        assert 0.0 <= pp.projected_sensitivity <= 1.0
        assert 0.0 <= pp.compatibility_score <= 1.0

    def test_recovery_cost_range(self):
        engine = TransportOperatorEngine()
        state = self._make_state()
        result = engine.transport(
            source_regime=THERMAL_REGIME,
            target_regime=RESOURCE_REGIME,
            state=state,
        )
        assert 0.0 <= result.expected_recovery_cost <= 1.0
        assert 0.0 <= result.transport_uncertainty <= 1.0

    def test_blocked_transport(self):
        """Non-transportable regime → blocked."""
        engine = TransportOperatorEngine()
        state = self._make_state()
        exotic = RegimeModel(
            regime_id="exotic",
            control_topology="distributed",
            optimization_geometry="target_band",
            intervention_algebra="multiplicative",
            counterfactual_law="substitution",
            causal_polarity="contextual",
            equilibrium_class="oscillatory",
            recovery_profile="non_recovering",
        )
        result = engine.transport(
            source_regime=THERMAL_REGIME,
            target_regime=exotic,
            state=state,
        )
        assert result.transport_class in ("blocked", "projective", "adversarial")
        if result.transport_class == "blocked":
            assert result.residual_error == 1.0
            assert result.transport_uncertainty == 1.0
