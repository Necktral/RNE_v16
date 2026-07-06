from pathlib import Path

from runtime.conjunction import (
    AgentPolicy,
    AutonomyPolicy,
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


def test_missing_memory_rag_executes_verified_recovery(tmp_path: Path):
    storage = _storage(tmp_path)
    storage.write_memory_record(
        run_id="op-memory-recovery",
        episode_id="episode-1",
        scale="micro",
        structure_json={
            "scenario": "thermal_homeostasis",
            "action": "act",
            "external_input_band": "normal",
        },
        metadata={"scenario_name": "thermal_homeostasis"},
        support_count=2,
    )
    storage.append_event(
        event_type="life.step.completed",
        run_id="op-memory-recovery",
        source="life_kernel",
        payload={"scenario": "thermal_homeostasis", "action": "act"},
    )
    context = OperationContext.create(
        run_id="op-memory-recovery",
        user_intent="answer_with_recovered_memory",
        task_type="life_cycle",
        requested_action="act",
        scenario="thermal_homeostasis",
        required_evidence=["memory_rag"],
        available_evidence=[],
        metadata={"external_input": 0.05},
    )

    result = OperationalConjunctionLayer(storage=storage).evaluate(context)

    assert result.final_decision == "allow_with_compensation"
    assert result.validation_status == "pass"
    assert result.compensation_status == "applied"
    assert result.context_summary["available_evidence_kinds"] == ["memory_rag"]
    assert any(
        item["stage"] == "compensation.execution"
        and item["added_evidence_kinds"] == ["memory_rag"]
        for item in result.to_dict()["trace"]
    )
    events = storage.list_events(
        run_id="op-memory-recovery",
        event_types=["operational.compensation.executed"],
        limit=5,
    )
    assert events
    evaluated = storage.list_events(
        run_id="op-memory-recovery",
        event_types=["operational.conjunction.evaluated"],
        limit=1,
    )
    memory_evidence = [
        item
        for item in evaluated[0].payload["context"]["available_evidence"]
        if item["kind"] == "memory_rag"
    ]
    assert memory_evidence[0]["payload"]["rag_attestation"]["validation_status"] == "pass"
    assert memory_evidence[0]["payload"]["rag_attestation"]["retrieval_purity"] == 1.0


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


def test_human_approval_required_blocks_critical_agent_action():
    context = OperationContext.create(
        run_id="op-human-approval",
        user_intent="self_modify_policy",
        task_type="self_modification",
        requested_action="self_modify",
        required_evidence=["validated_plan", "rollback_plan", "human_approval"],
        available_evidence=[
            EvidenceItem.create(kind="validated_plan", source="test", confidence=1.0),
            EvidenceItem.create(kind="rollback_plan", source="test", confidence=1.0),
        ],
        constraints=OperationalConstraints(
            permitted_actions=("self_modify",),
            critical_actions=("self_modify",),
        ),
        agent_policy=AgentPolicy(
            role="life_kernel",
            allowed_actions=("self_modify",),
            human_approval_required_actions=("self_modify",),
        ),
    )

    result = OperationalConjunctionLayer().evaluate(context)

    assert result.final_decision == "block"
    assert result.execution_permissions["automatic_execution_allowed"] is False
    assert any(item.code == "require_human_approval" for item in result.compensations)


def test_governed_unbounded_autonomy_is_authorized_by_policy():
    autonomy = AutonomyPolicy.resolve(
        requested_mode="unlimited",
        risk_score=0.20,
        resource_pressure=0.10,
        memory_purity=0.95,
        agent_policy_present=True,
    )
    context = OperationContext.create(
        run_id="op-autonomy-unbounded",
        user_intent="maintain_unbounded_policy_runtime",
        task_type="life_cycle",
        requested_action="act",
        available_evidence=[
            EvidenceItem.create(kind="vital_signs", source="test", confidence=1.0)
        ],
        agent_policy=AgentPolicy(
            role="life_kernel",
            allowed_actions=("act",),
            max_steps=0,
        ),
        autonomy_policy=autonomy,
        metadata={"agent_step_count": 100},
    )

    result = OperationalConjunctionLayer().evaluate(context)

    assert result.final_decision == "allow"
    assert result.execution_permissions["autonomy_policy"]["active_mode"] == "governed_unbounded"
    assert result.execution_permissions["autonomy_policy"]["policy_authorized"] is True
    assert not any(item.code == "agent_step_limit_exceeded" for item in result.validation_findings)


def test_unbounded_autonomy_degrades_under_resource_pressure():
    autonomy = AutonomyPolicy.resolve(
        requested_mode="unlimited",
        risk_score=0.20,
        resource_pressure=0.92,
        memory_purity=0.95,
        agent_policy_present=True,
    )
    context = OperationContext.create(
        run_id="op-autonomy-degraded",
        user_intent="maintain_unbounded_policy_runtime",
        task_type="life_cycle",
        requested_action="act",
        available_evidence=[
            EvidenceItem.create(kind="vital_signs", source="test", confidence=1.0)
        ],
        agent_policy=AgentPolicy(role="life_kernel", allowed_actions=("act",)),
        autonomy_policy=autonomy,
        resource_pressure=0.92,
    )

    result = OperationalConjunctionLayer().evaluate(context)

    assert result.final_decision == "allow_with_compensation"
    assert result.context_summary["autonomy_policy"]["active_mode"] == "bounded"
    assert any(item.code == "degrade_autonomy_scope" for item in result.compensations)
    assert any(item.code == "autonomy_degraded_by_policy" for item in result.validation_findings)


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


def test_life_kernel_can_disable_operational_conjunction(tmp_path: Path):
    storage = _storage(tmp_path)
    kernel = LifeKernel(
        config=LifeKernelConfig(
            run_id="life-conjunction-disabled",
            scenarios=("thermal_homeostasis",),
            restore=False,
            enable_msrc=False,
            enable_operational_conjunction=False,
        ),
        storage=storage,
    )

    result = kernel.step(external_input=0.05)

    assert result.operational == {}
    assert kernel.last_operational is None
    assert result.episode_result is not None
    assert "operational_conjunction" not in result.episode_result
    events = storage.list_events(
        run_id="life-conjunction-disabled",
        event_types=["operational.conjunction.evaluated"],
        limit=5,
    )
    assert events == []


def test_life_kernel_propagates_governed_unbounded_autonomy_policy(tmp_path: Path):
    storage = _storage(tmp_path)
    kernel = LifeKernel(
        config=LifeKernelConfig(
            run_id="life-conjunction-unbounded",
            scenarios=("thermal_homeostasis",),
            restore=False,
            enable_msrc=False,
            autonomy_policy="unlimited",
        ),
        storage=storage,
    )

    result = kernel.step(external_input=0.05)

    policy = result.operational["context_summary"]["autonomy_policy"]
    assert policy["requested_mode"] == "unlimited"
    assert policy["active_mode"] == "governed_unbounded"
    assert policy["policy_authorized"] is True
