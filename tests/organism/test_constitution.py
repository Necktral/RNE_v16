"""Tests para runtime/organism/constitution.py — Constitución del organismo."""

from __future__ import annotations

import pytest

from runtime.organism.state import (
    OrganismBeliefState,
    OrganismState,
    PolicyState,
    IdentityState,
    ViabilityState,
)
from runtime.organism.constitution import (
    ConstitutionalInvariant,
    ConstitutionalValidation,
    OrganismConstitution,
    HARD_INVARIANTS,
    SOFT_INVARIANTS,
    MUTABLE_COMPONENTS,
    IMMUTABLE_COMPONENTS,
)


class TestOrganismConstitution:
    def test_default_constitution(self):
        c = OrganismConstitution()
        assert len(c.hard_invariants) == len(HARD_INVARIANTS)
        assert len(c.soft_invariants) == len(SOFT_INVARIANTS)

    def test_healthy_state_validates(self):
        c = OrganismConstitution()
        s = OrganismState(
            belief=OrganismBeliefState(
                alarm_probability=0.1,
                intervention_efficacy=0.8,
                causal_support_confidence=0.9,
                memory_purity_estimate=0.95,
                trace_integrity_confidence=0.85,
            ),
            identity=IdentityState(lineage_id="L1"),
        )
        v = c.validate(s)
        assert v.is_valid is True
        assert v.verdict == "valid"
        assert v.hard_violation_count == 0

    def test_low_memory_purity_triggers_hard_violation(self):
        c = OrganismConstitution()
        s = OrganismState(
            belief=OrganismBeliefState(
                memory_purity_estimate=0.20,
                causal_support_confidence=0.9,
                trace_integrity_confidence=0.8,
            ),
            identity=IdentityState(lineage_id="L1"),
        )
        v = c.validate(s)
        assert v.is_valid is False
        assert v.hard_violation_count > 0
        names = [vl.invariant_name for vl in v.violations]
        assert "min_memory_purity" in names

    def test_low_trace_integrity_triggers_hard_violation(self):
        c = OrganismConstitution()
        s = OrganismState(
            belief=OrganismBeliefState(
                trace_integrity_confidence=0.10,
                memory_purity_estimate=0.9,
                causal_support_confidence=0.9,
            ),
            identity=IdentityState(lineage_id="L1"),
        )
        v = c.validate(s)
        assert v.is_valid is False
        names = [vl.invariant_name for vl in v.violations]
        assert "min_trace_integrity" in names

    def test_high_degradation_triggers_hard_violation(self):
        c = OrganismConstitution()
        s = OrganismState(
            belief=OrganismBeliefState(
                causal_support_confidence=0.9,
                memory_purity_estimate=0.9,
                trace_integrity_confidence=0.9,
            ),
            viability=ViabilityState(accumulated_degradation=0.90),
            identity=IdentityState(lineage_id="L1"),
        )
        v = c.validate(s)
        assert v.is_valid is False
        names = [vl.invariant_name for vl in v.violations]
        assert "baseline_not_degraded" in names

    def test_no_rollback_triggers_hard_violation(self):
        c = OrganismConstitution()
        s = OrganismState(
            belief=OrganismBeliefState(
                causal_support_confidence=0.9,
                memory_purity_estimate=0.9,
                trace_integrity_confidence=0.9,
            ),
            viability=ViabilityState(rollback_readiness=False),
            identity=IdentityState(lineage_id="L1"),
        )
        v = c.validate(s)
        assert v.is_valid is False
        names = [vl.invariant_name for vl in v.violations]
        assert "rollback_available" in names

    def test_empty_lineage_triggers_hard_violation(self):
        c = OrganismConstitution()
        s = OrganismState(
            belief=OrganismBeliefState(
                causal_support_confidence=0.9,
                memory_purity_estimate=0.9,
                trace_integrity_confidence=0.9,
            ),
            identity=IdentityState(lineage_id=""),
        )
        v = c.validate(s)
        assert v.is_valid is False
        names = [vl.invariant_name for vl in v.violations]
        assert "lineage_coherent" in names

    def test_high_policy_drift_triggers_soft_violation(self):
        c = OrganismConstitution()
        s = OrganismState(
            belief=OrganismBeliefState(
                causal_support_confidence=0.9,
                memory_purity_estimate=0.9,
                trace_integrity_confidence=0.9,
            ),
            policy=PolicyState(accumulated_drift=0.70),
            identity=IdentityState(lineage_id="L1"),
        )
        v = c.validate(s)
        # Still valid (soft violation only)
        assert v.is_valid is True
        assert v.soft_violation_count > 0
        names = [vl.invariant_name for vl in v.violations if vl.severity == "soft"]
        assert "policy_stability" in names or "drift_tolerance" in names

    def test_multiple_hard_violations_yield_rollback(self):
        c = OrganismConstitution()
        s = OrganismState(
            belief=OrganismBeliefState(
                memory_purity_estimate=0.10,
                trace_integrity_confidence=0.10,
                causal_support_confidence=0.05,
            ),
            viability=ViabilityState(
                accumulated_degradation=0.95,
                rollback_readiness=False,
            ),
            identity=IdentityState(lineage_id=""),
        )
        v = c.validate(s)
        assert v.verdict == "rollback"
        assert v.hard_violation_count >= 3

    def test_mutable_immutable(self):
        c = OrganismConstitution()
        assert c.is_mutable("transport_parameters")
        assert c.is_immutable("baseline_semantics")
        assert not c.is_mutable("baseline_semantics")
        assert not c.is_immutable("transport_parameters")

    def test_constitution_hash_deterministic(self):
        c = OrganismConstitution()
        h1 = c.constitution_hash()
        h2 = c.constitution_hash()
        assert h1 == h2
        assert len(h1) == 16


class TestConstitutionalValidation:
    def test_quarantine_verdict(self):
        """One or two hard violations → quarantine."""
        c = OrganismConstitution()
        s = OrganismState(
            belief=OrganismBeliefState(
                memory_purity_estimate=0.10,
                trace_integrity_confidence=0.9,
                causal_support_confidence=0.9,
            ),
            identity=IdentityState(lineage_id="L1"),
        )
        v = c.validate(s)
        # Only memory purity + triadic closure might be violated
        assert v.verdict in ("quarantine", "rollback")
        assert v.is_valid is False
