"""Tests para runtime/organism/self_modification.py — Auto-modificación certificada."""

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
from runtime.organism.self_modification import (
    SelfModificationPipeline,
    SandboxResult,
    ModificationDecision,
)


class TestSelfModificationPipeline:
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

    def test_generate_mutable_proposal(self):
        pipeline = SelfModificationPipeline()
        proposal = pipeline.generate_proposal(
            target="transport_parameters",
            description="Increase sensitivity scale",
        )
        assert proposal.target == "transport_parameters"
        assert proposal.sandbox_verdict == "pending"
        assert proposal.proposal_id.startswith("mod-")

    def test_generate_immutable_proposal_raises(self):
        pipeline = SelfModificationPipeline()
        with pytest.raises(ValueError, match="constitutionally immutable"):
            pipeline.generate_proposal(
                target="baseline_semantics",
                description="Change baseline",
            )

    def test_constitutional_precheck_passes_for_healthy(self):
        pipeline = SelfModificationPipeline()
        proposal = pipeline.generate_proposal(
            target="transport_parameters",
            description="test",
        )
        assert pipeline.constitutional_precheck(proposal, self._healthy_state()) is True

    def test_constitutional_precheck_fails_for_immutable(self):
        pipeline = SelfModificationPipeline()
        from runtime.organism.state import ModificationProposal
        proposal = ModificationProposal(
            proposal_id="x",
            target="baseline_semantics",
            description="bad",
        )
        assert pipeline.constitutional_precheck(proposal, self._healthy_state()) is False

    def test_sandbox_simulate_healthy(self):
        pipeline = SelfModificationPipeline()
        proposal = pipeline.generate_proposal(
            target="transport_parameters",
            description="test",
        )
        result = pipeline.sandbox_simulate(
            proposal=proposal,
            current_state=self._healthy_state(),
        )
        assert isinstance(result, SandboxResult)
        assert result.verdict in ("accepted", "quarantined", "rejected")
        assert result.risk_level in ("low", "medium", "high", "critical")

    def test_sandbox_with_custom_apply_fn(self):
        pipeline = SelfModificationPipeline()
        proposal = pipeline.generate_proposal(
            target="transport_parameters",
            description="increase sensitivity",
        )

        def apply_fn(state, _proposal):
            """Simulate by returning slightly modified state."""
            return OrganismState(
                belief=state.belief,
                policy=PolicyState(
                    sensitivity=min(1.0, state.policy.sensitivity + 0.1),
                    accumulated_drift=state.policy.accumulated_drift + 0.02,
                ),
                identity=state.identity,
                viability=state.viability,
            )

        result = pipeline.sandbox_simulate(
            proposal=proposal,
            current_state=self._healthy_state(),
            apply_fn=apply_fn,
        )
        assert isinstance(result, SandboxResult)

    def test_full_pipeline_evaluate(self):
        pipeline = SelfModificationPipeline()
        proposal = pipeline.generate_proposal(
            target="selection_policy",
            description="adjust selection weights",
        )
        decision = pipeline.evaluate_proposal(
            proposal=proposal,
            current_state=self._healthy_state(),
        )
        assert isinstance(decision, ModificationDecision)
        assert decision.proposal_id == proposal.proposal_id
        assert decision.verdict in ("accepted", "quarantined", "rejected")
        if decision.verdict == "accepted":
            assert decision.lineage_delta.get("modification_id") == proposal.proposal_id

    def test_pipeline_rejects_during_rollback(self):
        """If organism is in rollback, modifications should be rejected."""
        pipeline = SelfModificationPipeline()
        proposal = pipeline.generate_proposal(
            target="transport_parameters",
            description="test",
        )
        # Create state that will trigger rollback
        bad_state = OrganismState(
            belief=OrganismBeliefState(
                memory_purity_estimate=0.05,
                trace_integrity_confidence=0.05,
                causal_support_confidence=0.05,
            ),
            identity=IdentityState(lineage_id=""),
            viability=ViabilityState(
                accumulated_degradation=0.99,
                rollback_readiness=False,
            ),
        )
        decision = pipeline.evaluate_proposal(
            proposal=proposal,
            current_state=bad_state,
        )
        assert decision.verdict == "rejected"
