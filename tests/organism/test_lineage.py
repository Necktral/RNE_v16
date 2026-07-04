"""Tests para runtime/organism/lineage.py — Lineage constitucional."""

from __future__ import annotations

import pytest

from runtime.organism.state import IdentityState
from runtime.organism.constitution import OrganismConstitution
from runtime.organism.lineage import (
    LineageEntry,
    LineageState,
    InheritanceRule,
    DEFAULT_INHERITANCE_RULES,
)


class TestLineageState:
    def test_initial_state(self):
        ls = LineageState()
        assert ls.lineage_id == "genesis"
        assert ls.generation == 0
        assert ls.has_diverged is False
        assert ls.consistency_score() == 1.0

    def test_record_genesis(self):
        ls = LineageState(lineage_id="org-001")
        c = OrganismConstitution()
        ls.record_genesis(c, timestamp="2026-01-01T00:00:00Z")
        assert len(ls.history) == 1
        assert ls.history[0].entry_type == "genesis"
        assert ls.current_constitution_hash != ""
        assert ls.parent_constitution_hash == ls.current_constitution_hash

    def test_record_modification(self):
        ls = LineageState(lineage_id="org-001")
        c = OrganismConstitution()
        ls.record_genesis(c)
        ls.record_modification(
            modification_id="mod-001",
            description="Adjusted transport params",
            posterior=0.85,
        )
        assert ls.generation == 1
        assert "mod-001" in ls.accepted_modifications
        assert len(ls.history) == 2

    def test_record_rollback(self):
        ls = LineageState()
        ls.record_rollback(
            rollback_id="rb-001",
            description="Rollback after degradation",
        )
        assert "rb-001" in ls.rollback_ancestry
        assert len(ls.history) == 1

    def test_record_divergence(self):
        ls = LineageState()
        ls.record_divergence(
            divergence_id="div-001",
            description="Diverged from baseline",
        )
        assert ls.has_diverged is True
        assert "div-001" in ls.divergence_points

    def test_consistency_score_decreases_with_rollbacks(self):
        ls = LineageState()
        c = OrganismConstitution()
        ls.record_genesis(c)
        ls.record_modification(modification_id="m1", description="ok", posterior=0.8)
        score_before = ls.consistency_score()
        ls.record_rollback(rollback_id="rb1", description="fail")
        score_after = ls.consistency_score()
        assert score_after < score_before

    def test_consistency_score_decreases_with_divergences(self):
        ls = LineageState()
        c = OrganismConstitution()
        ls.record_genesis(c)
        ls.record_modification(modification_id="m1", description="ok", posterior=0.8)
        score_before = ls.consistency_score()
        ls.record_divergence(divergence_id="d1", description="diverge")
        score_after = ls.consistency_score()
        assert score_after < score_before

    def test_inheritance_eligibility_all_pass(self):
        ls = LineageState()
        eligible, failed = ls.check_inheritance_eligibility(
            is_certified_safe=True,
            is_constitution_consistent=True,
            is_baseline_preserved=True,
            is_contamination_free=True,
        )
        assert eligible is True
        assert failed == []

    def test_inheritance_eligibility_fails(self):
        ls = LineageState()
        eligible, failed = ls.check_inheritance_eligibility(
            is_certified_safe=False,
            is_constitution_consistent=True,
            is_baseline_preserved=True,
            is_contamination_free=False,
        )
        assert eligible is False
        assert "certified_safe" in failed
        assert "no_contamination" in failed

    def test_to_identity_state(self):
        ls = LineageState(lineage_id="org-123")
        c = OrganismConstitution()
        ls.record_genesis(c)
        identity = ls.to_identity_state(c)
        assert isinstance(identity, IdentityState)
        assert identity.lineage_id == "org-123"
        assert identity.constitution_hash == ls.current_constitution_hash
        assert len(identity.active_invariants) > 0

    def test_to_dict(self):
        ls = LineageState(lineage_id="org-001")
        c = OrganismConstitution()
        ls.record_genesis(c)
        ls.record_modification(modification_id="m1", description="test", posterior=0.9)
        d = ls.to_dict()
        assert d["lineage_id"] == "org-001"
        assert d["generation"] == 1
        assert d["consistency_score"] > 0.0
        assert "m1" in d["accepted_modifications"]


class TestInheritanceRules:
    def test_default_rules_exist(self):
        assert len(DEFAULT_INHERITANCE_RULES) == 4
        conditions = {r.condition for r in DEFAULT_INHERITANCE_RULES}
        assert "certified_safe" in conditions
        assert "constitution_consistent" in conditions
        assert "baseline_preserved" in conditions
        assert "no_contamination" in conditions
