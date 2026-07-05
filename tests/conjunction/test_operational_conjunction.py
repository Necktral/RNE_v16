from pathlib import Path

from runtime.conjunction import (
    AgentPolicy,
    CausalAssumption,
    EvidenceItem,
    OperationalConjunctionLayer,
    OperationalConstraints,
    OperationContext,
)
from runtime.life import LifeKernel, LifeKernelConfig, VitalSignsSnapshot
from runtime.storage import StorageConfig, StorageFactory


def _storage(tmp_path: Path):
    return StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(tmp_path / "conjunction.db"),
            postgres_dsn=None,
            artifact_root=tmp_path / "artifacts",
            prefer_postgres_reads=False,
            strict_dual_write=False,
        )
    )


def _vitals(**overrides):
    payload = {
        "run_id": "conjunction-life",
        "episode_count": 3,
        "mode": "normal",
        "viability_margin": 0.80,
        "continuity_score": 0.90,
        "ioc_proxy": 0.80,
        "risk_score": 0.20,
        "memory_purity": 0.95,
        "cognitive_quality": 0.78,
        "resource_pressure": 0.10,
        "recovery_debt": 0.0,
        "accumulated_drift": 0.0,
        "reversible": True,
        "identity_continuity": 0.90,
        "certified": True,
    }
    payload.update(overrides)
    return VitalSignsSnapshot(**payload)


def test_simple_operation_uses_deterministic_tier():
    context = OperationContext.create(
        run_id="op-simple",
        user_intent="answer_simple_fact",
        task_type="life_cycle",
        requested_action="act",
        required_evidence=["vital_signs"],
        available_evidence=[
            EvidenceItem.create(
                kind="vital_signs",
                source="test",
                confidence=1.0,
                canonical=True,
            )
        ],
        complexity_score=0.10,
        uncertainty_score=0.10,
    )

    result = OperationalConjunctionLayer().evaluate(context)

    assert result.final_decision == "allow"
    assert result.selected_compute_tier == "tier_0_deterministic"
    assert result.execution_permissions["automatic_execution_allowed"] is True
    assert any(item["stage"] == "router.output" for item in result.to_dict()["trace"])


def test_missing_evidence_activates_compensation_without_external_oracle():
    context = OperationContext.create(
        run_id="op-missing-evidence",
        user_intent="answer_with_evidence",
        task_type="life_cycle",
        requested_action="act",
        required_evidence=["memory_rag"],
        available_evidence=[],
        uncertainty_score=0.65,
        constraints=OperationalConstraints(allow_external=False),
    )

    result = OperationalConjunctionLayer().evaluate(context)

    assert result.final_decision == "allow_with_compensation"
    assert result.selected_compute_tier != "tier_3_external"
    assert result.compensation_status == "applied"
    assert any(item.code == "expand_retrieval" for item in result.compensations)


def test_contradictory_evidence_blocks_fusion():
    context = OperationContext.create(
        run_id="op-conflict",
        user_intent="merge_conflicting_facts",
        task_type="life_cycle",
        requested_action="act",
        available_evidence=[
            EvidenceItem.create(
                kind="conflict",
                source="test",
                confidence=0.9,
                payload={"conflict": True, "left": "A", "right": "not-A"},
            )
        ],
    )

    result = OperationalConjunctionLayer().evaluate(context)

    assert result.final_decision == "block"
    assert result.confidence_state == "conflicted"
    assert any(item.code == "mark_conflict" for item in result.compensations)


def test_unsupported_causal_claim_is_not_allowed_as_strong_causality():
    context = OperationContext.create(
        run_id="op-causal",
        user_intent="make_causal_recommendation",
        task_type="causal_decision",
        requested_action="act",
        causal_assumptions=[
            CausalAssumption(
                name="x_causes_y",
                statement="x causes y",
                supported=False,
                strength=0.0,
            )
        ],
    )

    result = OperationalConjunctionLayer().evaluate(context)

    assert result.final_decision == "degrade"
    assert any(item.code == "downgrade_causal_claim" for item in result.compensations)
    assert any(
        item.code == "unsupported_causal_assumption"
        for item in result.validation_findings
    )


def test_critical_agent_action_requires_validated_plan_and_rollback():
    context = OperationContext.create(
        run_id="op-critical",
        user_intent="self_modify_policy",
        task_type="self_modification",
        requested_action="self_modify",
        required_evidence=["validated_plan", "rollback_plan"],
        available_evidence=[],
        constraints=OperationalConstraints(
            permitted_actions=("self_modify",),
            critical_actions=("self_modify",),
        ),
        agent_policy=AgentPolicy(
            role="life_kernel",
            allowed_actions=("self_modify",),
            requires_plan_for_critical=True,
            rollback_required_for_critical=True,
        ),
    )

    result = OperationalConjunctionLayer().evaluate(context)

    assert result.final_decision == "block"
    assert result.execution_permissions["automatic_execution_allowed"] is False
    assert any(item.code == "require_plan_validation" for item in result.compensations)
    assert any(item.code == "require_rollback_plan" for item in result.compensations)


def test_life_kernel_persists_operational_conjunction_trace(tmp_path: Path):
    storage = _storage(tmp_path)
    kernel = LifeKernel(
        config=LifeKernelConfig(
            run_id="life-conjunction",
            scenarios=("thermal_homeostasis",),
            restore=False,
            enable_msrc=False,
        ),
        storage=storage,
    )

    result = kernel.step(external_input=0.05)

    assert result.operational["final_decision"] == "allow"
    assert result.episode_result is not None
    assert "operational_conjunction" in result.episode_result
    events = storage.list_events(
        run_id="life-conjunction",
        event_types=["operational.conjunction.evaluated"],
        limit=5,
    )
    assert events


def test_life_kernel_blocks_unproven_self_modification(tmp_path: Path):
    storage = _storage(tmp_path)
    kernel = LifeKernel(
        config=LifeKernelConfig(
            run_id="life-conjunction-block",
            scenarios=("thermal_homeostasis",),
            restore=False,
            enable_msrc=False,
        ),
        storage=storage,
    )
    kernel.last_vitals = _vitals(memory_purity=0.20)

    result = kernel.step(external_input=0.05)

    assert result.operational["final_decision"] == "block"
    assert result.decision.action == "act"
    assert result.decision.mode == "recovery"
    assert result.decision.directives["blocked_action"] == "self_modify"
