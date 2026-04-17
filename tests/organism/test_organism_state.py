"""Tests para runtime/organism/state.py — OrganismState y sub-estados."""

from __future__ import annotations

import json

import pytest

from runtime.organism.state import (
    ModificationProposal,
    ModificationState,
    OrganismBeliefState,
    OrganismState,
    PolicyState,
    IdentityState,
    ViabilityState,
    transition_organism_state,
)


# ── OrganismBeliefState ──────────────────────────────────────────────────────

class TestOrganismBeliefState:
    def test_default_values(self):
        b = OrganismBeliefState()
        assert 0.0 <= b.alarm_probability <= 1.0
        assert 0.0 <= b.composite_confidence <= 1.0

    def test_composite_confidence_range(self):
        b = OrganismBeliefState(
            alarm_probability=0.0,
            intervention_efficacy=1.0,
            causal_support_confidence=1.0,
            memory_purity_estimate=1.0,
            trace_integrity_confidence=1.0,
            regime_uncertainty=0.0,
        )
        assert b.composite_confidence > 0.8

    def test_low_confidence(self):
        b = OrganismBeliefState(
            alarm_probability=1.0,
            intervention_efficacy=0.0,
            causal_support_confidence=0.0,
            memory_purity_estimate=0.0,
            trace_integrity_confidence=0.0,
        )
        assert b.composite_confidence < 0.3

    def test_distance_to_self(self):
        b = OrganismBeliefState()
        assert b.distance_to(b) == 0.0

    def test_distance_to_different(self):
        a = OrganismBeliefState(alarm_probability=0.0)
        b = OrganismBeliefState(alarm_probability=1.0)
        assert a.distance_to(b) > 0.0


# ── PolicyState ──────────────────────────────────────────────────────────────

class TestPolicyState:
    def test_default(self):
        p = PolicyState()
        assert p.control_class == "reactive"
        assert 0.0 <= p.stability_score <= 1.0

    def test_high_drift_lowers_stability(self):
        stable = PolicyState(accumulated_drift=0.0)
        drifted = PolicyState(accumulated_drift=0.8)
        assert stable.stability_score > drifted.stability_score


# ── IdentityState ────────────────────────────────────────────────────────────

class TestIdentityState:
    def test_self_distance(self):
        i = IdentityState(
            active_invariants=frozenset({"a", "b"}),
            lineage_id="L1",
            constitution_hash="abc123",
        )
        assert i.identity_distance(i) == 0.0

    def test_different_identity(self):
        a = IdentityState(lineage_id="L1", constitution_hash="aaa")
        b = IdentityState(lineage_id="L2", constitution_hash="bbb")
        assert a.identity_distance(b) > 0.0


# ── ViabilityState ───────────────────────────────────────────────────────────

class TestViabilityState:
    def test_viable_default(self):
        v = ViabilityState()
        assert v.is_viable is True
        assert v.distance_to_edge > 0.0

    def test_not_viable_zero_margin(self):
        v = ViabilityState(viability_margin=0.0)
        assert v.is_viable is False

    def test_not_viable_full_degradation(self):
        v = ViabilityState(accumulated_degradation=1.0)
        assert v.is_viable is False


# ── ModificationState ────────────────────────────────────────────────────────

class TestModificationState:
    def test_empty(self):
        m = ModificationState()
        assert m.pending_count == 0
        assert m.has_accepted is False

    def test_pending_proposal(self):
        p = ModificationProposal(proposal_id="p1", sandbox_verdict="pending")
        m = ModificationState(active_proposals=(p,))
        assert m.pending_count == 1

    def test_accepted_proposal(self):
        p = ModificationProposal(proposal_id="p1", sandbox_verdict="accepted")
        m = ModificationState(active_proposals=(p,))
        assert m.has_accepted is True
        assert m.pending_count == 0


# ── OrganismState ────────────────────────────────────────────────────────────

class TestOrganismState:
    def test_default_is_viable(self):
        s = OrganismState()
        assert s.is_viable is True
        assert 0.0 <= s.composite_health <= 1.0

    def test_serialization_roundtrip(self):
        s = OrganismState(
            state_id="test-1",
            episode_count=5,
            belief=OrganismBeliefState(alarm_probability=0.3),
            identity=IdentityState(
                active_invariants=frozenset({"triadic_closure", "min_memory_purity"}),
                lineage_id="L1",
            ),
        )
        d = s.to_dict()
        s2 = OrganismState.from_dict(d)
        assert s2.state_id == "test-1"
        assert s2.episode_count == 5
        assert s2.belief.alarm_probability == 0.3
        assert s2.identity.lineage_id == "L1"
        assert "triadic_closure" in s2.identity.active_invariants

    def test_from_dict_empty(self):
        s = OrganismState.from_dict({})
        assert s.state_id == ""
        assert s.is_viable

    def test_json_serializable(self):
        s = OrganismState(state_id="s1")
        blob = json.dumps(s.to_dict())
        assert "s1" in blob


# ── transition_organism_state ────────────────────────────────────────────────

class TestTransitionOrganismState:
    def test_basic_transition(self):
        current = OrganismState(state_id="s0", episode_count=0)
        episode_result = {
            "episode": {
                "context": {
                    "observation": {"alarm": False, "temperature": 0.5},
                },
                "result": {
                    "relation_kind": "support",
                },
                "scenario_metadata": {"scenario_name": "thermal"},
            },
            "belief_state": {
                "posterior": {
                    "policy_confidence": 0.7,
                    "causal_support_confidence": 0.9,
                    "memory_purity_confidence": 1.0,
                    "trace_confidence": 0.85,
                },
            },
            "certification": {
                "verdict": "certified",
            },
        }
        new = transition_organism_state(
            current=current,
            episode_result=episode_result,
            regime="thermal",
            new_state_id="s1",
        )
        assert new.state_id == "s1"
        assert new.episode_count == 1
        assert new.active_regime == "thermal"
        assert new.belief.alarm_probability < 0.5
        assert new.belief.intervention_efficacy == 0.7

    def test_transition_increments_episode(self):
        s = OrganismState(episode_count=10)
        result = {
            "episode": {"context": {"observation": {}}, "result": {}},
            "certification": {},
        }
        s2 = transition_organism_state(current=s, episode_result=result)
        assert s2.episode_count == 11

    def test_certification_affects_viability(self):
        base = OrganismState()
        certified = transition_organism_state(
            current=base,
            episode_result={
                "episode": {"context": {"observation": {}}, "result": {}},
                "certification": {"verdict": "certified"},
            },
        )
        rejected = transition_organism_state(
            current=base,
            episode_result={
                "episode": {"context": {"observation": {}}, "result": {}},
                "certification": {"verdict": "rejected"},
            },
        )
        # Certified should improve viability vs rejected
        assert certified.viability.viability_margin > rejected.viability.viability_margin
