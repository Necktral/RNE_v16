from pathlib import Path

import pytest
from z3 import Bool, Not

from runtime.symbolic.acting_trace import ActingTraceCollector
from runtime.symbolic.constraint_registry import ConstraintRegistry, constraint_id
from runtime.symbolic.optimization_tracker import build_optimization_report
from runtime.symbolic.schemas import (
    CausalGuardReport,
    ConstraintRecord,
    canonical_json,
    sealed_sha256,
)
from runtime.symbolic.tracked_solver import TrackedSolver


def test_canonical_json_and_hash_ignore_mapping_insertion_order():
    left = {"b": {"y": 2, "x": 1}, "a": [3, 4]}
    right = {"a": [3, 4], "b": {"x": 1, "y": 2}}
    assert canonical_json(left) == canonical_json(right)
    assert sealed_sha256(left) == sealed_sha256(right)


def test_schema_deep_freezes_nested_collections():
    provenance = {"nested": {"values": [1, 2]}}
    record = ConstraintRecord(
        constraint_id="c",
        expression="A",
        hard_or_revisable="hard",
        provenance=provenance,
        parent_ids=["p"],
        logical_time=1,
        replay_unit_id="unit",
    )
    provenance["nested"]["values"].append(3)
    assert record.to_dict()["provenance"]["nested"]["values"] == [1, 2]
    with pytest.raises(TypeError):
        record.provenance["new"] = "value"


def test_registry_is_idempotent_and_rejects_semantic_collision():
    registry = ConstraintRegistry(replay_unit_id="thermal/a", logical_time=1)
    identifier = constraint_id(
        replay_unit_id="thermal/a",
        logical_time=1,
        kind="formula",
        source="scenario/main",
        rule_id="r/1",
    )
    record = ConstraintRecord(
        constraint_id=identifier,
        expression="A",
        hard_or_revisable="hard",
        provenance={},
        parent_ids=(),
        logical_time=1,
        replay_unit_id="thermal/a",
    )
    registry.register(record)
    registry.register(record)
    assert len(registry.records) == 1
    with pytest.raises(ValueError, match="Colisión"):
        registry.register(
            ConstraintRecord(
                **{**record.to_dict(), "expression": "B"},
            )
        )


def test_tracked_solver_reports_sat_and_unsat_cores():
    sat_registry = ConstraintRegistry(replay_unit_id="sat", logical_time=1)
    sat_solver = TrackedSolver(
        registry=sat_registry, replay_unit_id="sat", logical_time=1
    )
    a = Bool("A")
    sat_solver.add_formula(a, source="scenario", rule_id="main")
    status, _, core, report = sat_solver.check()
    assert status == "SAT"
    assert core == ()
    assert report.core_ids == ()
    assert len(report.all_constraint_ids) == 1

    unsat_registry = ConstraintRegistry(replay_unit_id="unsat", logical_time=1)
    unsat_solver = TrackedSolver(
        registry=unsat_registry, replay_unit_id="unsat", logical_time=1
    )
    unsat_solver.add_formula(a, source="scenario", rule_id="main")
    unsat_solver.add_assumption(a, False, source="observation", rule_id="A")
    status, _, core, report = unsat_solver.check()
    assert status == "UNSAT"
    assert set(core) == set(report.all_constraint_ids)


def test_optimization_adapter_preserves_candidates_and_tie_breaks():
    opt_choice = {
        "status": "ok",
        "x0": 0.8,
        "optimization_direction": "minimize",
        "intervention": "cool",
        "steps": 1,
        "projected": 0.7,
        "objective": 0.75,
        "effort_cost": 0.05,
        "alternatives": [
            {
                "intervention": "cool",
                "steps": 1,
                "projected": 0.7,
                "objective": 0.75,
            }
        ],
    }
    report = build_optimization_report({}, {"opt_choice": opt_choice}, "unit", 1)
    assert report.winner == report.candidates[0]
    assert report.tie_breaks == ("objective", "steps", "intervention")


def test_collector_seals_once_links_outcome_and_degrades_on_io_failure(tmp_path: Path):
    collector = ActingTraceCollector(
        replay_unit_id="unit",
        preaction_logical_time=1,
        trace_dir=tmp_path / "ok",
    )
    collector.set_baseline_action("idle")
    collector.set_guard_report(
        CausalGuardReport(False, None, None, 0.0, "no_conflict", {})
    )
    trace = collector.seal(
        committed_action="idle",
        governance_verdict={"verdict": "admitted"},
    )
    link = collector.link_outcome(
        outcome_observed={"x": 1},
        prediction_error=0.0,
        utility=1.0,
        certification_ref="sha256:abc",
    )
    assert link.decision_trace_sha256 == trace.sealed_hash
    with pytest.raises(RuntimeError, match="sellada"):
        collector.seal(
            committed_action="idle",
            governance_verdict={"verdict": "admitted"},
        )

    blocked = tmp_path / "not-a-directory"
    blocked.write_text("x", encoding="utf-8")
    degraded = ActingTraceCollector(
        replay_unit_id="unit",
        preaction_logical_time=1,
        trace_dir=blocked,
    )
    degraded.set_baseline_action("idle")
    degraded.set_guard_report(
        CausalGuardReport(False, None, None, 0.0, "no_conflict", {})
    )
    degraded.seal(
        committed_action="idle",
        governance_verdict={"verdict": "admitted"},
    )
    assert degraded.trace_status == "persistence_degraded"
