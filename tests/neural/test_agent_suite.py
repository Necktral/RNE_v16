from __future__ import annotations

from pathlib import Path

from runtime.neural.agents import (
    CORE_AGENT_ROLES,
    AgentRole,
    AgentState,
    CurriculumLearningAgent,
    DevelopmentLineageAgent,
    HorizontalCreativityAgent,
    InteroceptiveHomeostaticAgent,
    MetacognitiveEpistemicAgent,
    MemoryConsolidationAgent,
    ModelDataImmuneAgent,
    MetabolicBudgetAgent,
    PedagogicalTeacherAgent,
    SensorimotorWorldModelAgent,
    SocialExocortexAgent,
    NeuralOrchestrationAgent,
    SpecializedAgentBundle,
)
from runtime.neural.connectome import ConnectomeRuntime
from runtime.neural.integration import (
    AuthorityEffect,
    ConsumerReceipt,
    ConsumerVerdictClass,
    OrganTrace,
    SymbiosisIdentity,
    SymbioticNeuralCoordinator,
)
from runtime.neural.integration.contracts import canonical_sha256
from runtime.storage import StorageConfig, StorageFactory
from runtime.world.scenario_runner import ScenarioEpisodeRunner


def _identity(episode: str = "episode-agents") -> SymbiosisIdentity:
    return SymbiosisIdentity(
        trace_group_id=f"trace-{episode}",
        organism_id="organism-agents",
        lineage_id="lineage-agents",
        run_id="run-agents",
        episode_id=episode,
        scenario_id="scenario@1",
        decision_id="decision-agents",
    )


def _organ(identity: SymbiosisIdentity, *, candidate_hash: str | None = None) -> OrganTrace:
    candidate = {"proposal": "IND", "authority_effect": "none"}
    return OrganTrace(
        identity=identity,
        organ="N1",
        capability="family_routing",
        requested_mode="shadow",
        effective_mode="shadow",
        authority_ceiling="bounded_proposal",
        input_hash="input",
        candidate_hash=candidate_hash or canonical_sha256(candidate),
        consumer="scheduler comparison",
        consumer_verdict="accepted",
        latency_ms=0.1,
        confidence=0.8,
        uncertainty=0.2,
        candidate=candidate,
    )


def _receipt(
    identity: SymbiosisIdentity,
    organ: OrganTrace,
    verdict: ConsumerVerdictClass = ConsumerVerdictClass.ACCEPTED,
) -> ConsumerReceipt:
    return ConsumerReceipt(
        receipt_id="receipt-agent-1",
        identity=identity,
        organ=organ.organ,
        candidate_hash=str(organ.candidate_hash),
        consumer_id="scheduler_comparison",
        consumer_contract_version="scheduler-comparison-v1",
        consumer_input_hash="consumer-input",
        consumer_output_hash="consumer-output",
        verdict_class=verdict,
        verdict_detail=verdict.value,
        evidence_refs=("scheduler",),
        authority_effect=AuthorityEffect.EVIDENCE_ONLY,
        persisted=True,
    )


def _storage(tmp_path: Path, name: str = "agents"):
    return StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(tmp_path / f"{name}.db"),
            postgres_dsn=None,
            artifact_root=tmp_path / "artifacts",
            prefer_postgres_reads=False,
            strict_dual_write=False,
        )
    )


def test_five_agent_cycle_is_deterministic_bounded_and_integrated() -> None:
    identity = _identity()
    organ = _organ(identity)
    receipt = _receipt(identity, organ)
    connectome = ConnectomeRuntime()
    activity = connectome.observe(
        identity=identity, organs=(organ,), receipts=(receipt,), mode="shadow"
    ).to_dict()
    orchestrator = NeuralOrchestrationAgent(connectome=connectome)

    first = orchestrator.run_cycle(
        identity=identity,
        organs=(organ,),
        receipts=(receipt,),
        connectome_activity=activity,
    )
    second = orchestrator.run_cycle(
        identity=identity,
        organs=(organ,),
        receipts=(receipt,),
        connectome_activity=activity,
    )

    assert first.cycle_hash == second.cycle_hash
    assert first.experimental is True
    assert {report.role for report in first.reports} == set(CORE_AGENT_ROLES)
    assert all(report.experimental is True for report in first.reports)
    assert all(report.authority_effect.value in {"none", "evidence_only"} for report in first.reports)
    assert first.blocked is False
    latent = first.by_role(AgentRole.LATENT_COMMUNICATION)
    assert latent.state is AgentState.OBSERVED
    modulation = latent.outputs["modulations"][0]
    assert latent.outputs["evidence_pipeline"] == [
        "measure", "classify", "analyze", "deliberate"
    ]
    assert latent.outputs["stages"]["measure"][0]["status"] == "measured"
    assert latent.outputs["stages"]["classify"][0]["status"] == "informative"
    assert latent.outputs["stages"]["analyze"][0]["benefit_measured"] is False
    assert latent.outputs["stages"]["deliberate"][0]["verdict"] == "propose"
    assert latent.outputs["gain_bounds_kind"] == "safety_envelope_not_setpoint"
    assert latent.outputs["setpoint_status"] == "unlearned"
    assert 1.0 < modulation["proposed_gain"] <= 1.25
    assert modulation["apply_authorized"] is False
    synergy = first.by_role(AgentRole.SYMBIOSIS_SYNERGY)
    assert synergy.outputs["synergy_state"] == "integrated"
    assert synergy.metrics["connectivity_coverage"] == 1.0


def test_neutral_receipts_do_not_fabricate_latent_gain_or_plasticity() -> None:
    identity = _identity("neutral")
    organ = _organ(identity)
    receipt = _receipt(identity, organ, ConsumerVerdictClass.COMPARED)
    connectome = ConnectomeRuntime()
    activity = connectome.observe(
        identity=identity, organs=(organ,), receipts=(receipt,), mode="shadow"
    )
    cycle = NeuralOrchestrationAgent(connectome=connectome).run_cycle(
        identity=identity,
        organs=(organ,),
        receipts=(receipt,),
        connectome_activity=activity.to_dict(),
    )

    assert activity.plasticity_proposals == ()
    latent = cycle.by_role(AgentRole.LATENT_COMMUNICATION)
    assert latent.state is AgentState.ABSTAINED
    assert latent.outputs["modulations"] == []
    assert latent.outputs["stages"]["classify"][0]["status"] == "non_informative"
    assert latent.outputs["stages"]["deliberate"][0]["verdict"] == "abstain"
    assert any(
        finding.code == "latent_consumer_evidence_non_informative"
        for finding in latent.findings
    )


def test_adversarial_agent_quarantines_tampered_candidate_fail_closed() -> None:
    identity = _identity("tampered")
    organ = _organ(identity, candidate_hash="tampered")
    receipt = _receipt(identity, organ)
    connectome = ConnectomeRuntime()
    activity = connectome.observe(
        identity=identity, organs=(organ,), receipts=(receipt,), mode="shadow"
    ).to_dict()
    cycle = NeuralOrchestrationAgent(connectome=connectome).run_cycle(
        identity=identity,
        organs=(organ,),
        receipts=(receipt,),
        connectome_activity=activity,
    )

    assert cycle.blocked is True
    adversarial = cycle.by_role(AgentRole.ADVERSARIAL)
    assert adversarial.state is AgentState.BLOCKED
    assert adversarial.outputs["quarantined_organs"] == ["N1"]
    assert cycle.by_role(AgentRole.LATENT_COMMUNICATION).outputs["modulations"] == []
    assert cycle.by_role(AgentRole.SYMBIOSIS_SYNERGY).outputs["synergy_state"] == "blocked"


def test_coordinator_exposes_agent_cycle_in_certification(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RNFE_NEURAL_MODE", "shadow")
    storage = _storage(tmp_path)
    coordinator = SymbioticNeuralCoordinator(storage=storage)
    identity = _identity("coordinator")
    coordinator.begin_episode(
        identity=identity,
        observation={"value": 0.8},
        formula="value > 0.5",
        proposition="value high",
        memory_hits=[],
        scenario_metadata={"main_variable": "value"},
        causal_attestation={"main_variable": "value", "factual_delta": -0.1},
        resources={},
    )
    block = coordinator.certification_block(identity.episode_id)

    assert block["neural_agents"]["schema_version"] == "rnfe-neural-agent-cycle-v1"
    assert len(block["neural_agents"]["reports"]) == 5
    assert block["neural_agents"]["cycle_hash"] == coordinator.agent_cycle(
        identity.episode_id
    ).cycle_hash
    storage.close()


def test_n4_trace_is_bound_to_committed_intervention(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RNFE_NEURAL_MODE", "shadow")
    storage = _storage(tmp_path, "n4-committed")
    result = ScenarioEpisodeRunner(
        storage=storage,
        run_id="run-n4-committed",
        scenario="thermal_homeostasis",
    ).run_episode(external_input=0.08)

    committed = result["episode"]["context"]["intervention"]
    n4 = next(
        entry for entry in result["neural_symbiosis_trace"]["organs"]
        if entry["organ"] == "N4"
    )
    comparison = n4["candidate"]["canonical_comparison"]
    receipt = next(
        entry for entry in result["neural_symbiosis_trace"]["consumer_receipts"]
        if entry["organ"] == "N4" and entry["consumer_id"] == "canonical_causal_comparator"
    )
    assert comparison["temporal_binding"] == "committed_action"
    assert comparison["committed_intervention"] == committed
    assert "committed_action" in receipt["evidence_refs"]
    extensions = result["neural_symbiosis"]["neural_agent_extensions"]
    assert extensions["schema_version"] == "rnfe-neural-agent-extensions-v1"
    extension_by_role = {item["role"]: item for item in extensions["reports"]}
    assert set(extension_by_role) == {
        "development_lineage",
        "horizontal_creativity",
        "interoceptive_homeostatic",
        "memory_consolidation",
        "metabolic_budget",
        "metacognitive_epistemic",
        "model_data_immune",
        "sensorimotor_world_model",
        "social_exocortex",
    }
    assert extension_by_role["metacognitive_epistemic"]["authority_effect"] == "none"
    assert extension_by_role["interoceptive_homeostatic"]["authority_effect"] == "none"
    final_extensions = result["neural_symbiosis_trace"]["neural_agent_extensions"]
    final_by_role = {item["role"]: item for item in final_extensions["reports"]}
    assert "pedagogical_teacher" in final_by_role
    assert "curriculum_learning" in final_by_role
    assert (
        final_by_role["pedagogical_teacher"]["outputs"]["stages"]["classify"]
        ["pedagogical_class"]
        == "teacher_inactive"
    )
    storage.close()


def test_interoceptive_agent_distinguishes_measured_state_from_defaults() -> None:
    identity = _identity("interoception")
    agent = InteroceptiveHomeostaticAgent()
    stable = agent.assess(
        identity=identity,
        viability={
            "is_viable": True,
            "viability_margin": 0.72,
            "distance_to_edge": 0.72,
            "rollback_required": False,
        },
        resources={
            "cpu_pressure": 0.2,
            "memory_pressure": 0.3,
            "thermal_pressure": 0.1,
            "vram_pressure": 0.4,
            "msrc_budget_available": True,
        },
        measurement_status={
            "cpu_pressure": "measured",
            "memory_pressure": "measured",
            "thermal_pressure": "measured",
            "vram_pressure": "measured",
            "msrc_budget_available": "measured",
        },
        trace_health={"degraded": False, "pending_events": 0, "dropped_events": 0},
    )
    assert stable.state is AgentState.OBSERVED
    assert stable.metrics["measured_pressure_count"] == 4
    assert stable.outputs["stages"]["classify"]["interoceptive_class"] == (
        "homeostatic_envelope_observed"
    )
    assert stable.outputs["stages"]["deliberate"]["actuation_authorized"] is False

    defaulted = agent.assess(
        identity=identity,
        viability={},
        resources={
            "cpu_pressure": 0.0,
            "memory_pressure": 0.0,
            "thermal_pressure": 0.0,
            "vram_pressure": 1.0,
            "msrc_budget_available": True,
        },
        measurement_status={axis: "defaulted" for axis in (
            "cpu_pressure",
            "memory_pressure",
            "thermal_pressure",
            "vram_pressure",
            "msrc_budget_available",
        )},
        trace_health={},
    )
    assert defaulted.state is AgentState.DEGRADED
    assert defaulted.metrics["measured_pressure_count"] == 0
    assert defaulted.metrics["peak_measured_pressure"] is None
    assert any(
        finding.code == "interoception_measurement_incomplete"
        for finding in defaulted.findings
    )


def test_epistemic_agent_measures_without_claiming_pre_outcome_gain() -> None:
    identity = _identity("epistemic")
    organ = _organ(identity)
    receipt = _receipt(identity, organ)
    reasoning = {
        "sequence": ["ABD", "CAU", "CTF", "DED", "PROB"],
        "state": {
            "prob_point": 0.72,
            "prob_lcb": 0.51,
            "cau_link": {"helps_goal": True},
            "ctf_checked": {"supports_choice": True},
        },
        "sequence_validation": {"validated_passed": True},
    }
    report = MetacognitiveEpistemicAgent().assess(
        identity=identity,
        reasoning=reasoning,
        organs=(organ,),
        receipts=(receipt,),
    )
    bundle = SpecializedAgentBundle.create(identity=identity, reports=(report,))

    assert report.state is AgentState.OBSERVED
    assert report.outputs["stages"]["classify"]["epistemic_class"] == "measured_consistent"
    assert report.outputs["stages"]["analyze"]["committed_certainty"] == 0.51
    assert report.outputs["stages"]["analyze"]["epistemic_gain"] is None
    assert report.outputs["stages"]["analyze"]["gain_status"] == "unmeasured_pre_outcome"
    assert report.outputs["stages"]["deliberate"]["scheduler_authority_preserved"] is True
    assert bundle.by_role(AgentRole.METACOGNITIVE_EPISTEMIC) == report


def test_epistemic_agent_abstains_when_prob_measurement_is_missing() -> None:
    identity = _identity("epistemic-missing")
    organ = _organ(identity)
    report = MetacognitiveEpistemicAgent().assess(
        identity=identity,
        reasoning={
            "sequence": ["ABD", "CAU", "CTF"],
            "state": {
                "cau_link": {"helps_goal": True},
                "ctf_checked": {"supports_choice": False},
            },
            "sequence_validation": {"validated_passed": True},
        },
        organs=(organ,),
        receipts=(),
    )

    assert report.state is AgentState.ABSTAINED
    assert report.outputs["stages"]["classify"]["epistemic_class"] == "unmeasured"
    assert report.outputs["stages"]["deliberate"]["escalation_targets"] == ["PROB"]
    assert any(finding.code == "epistemic_calibration_missing" for finding in report.findings)


def test_memory_agent_audits_provenance_without_writing_memory() -> None:
    identity = _identity("memory-clean")
    report = MemoryConsolidationAgent().assess(
        identity=identity,
        memory_hits=(
            {
                "episode_id": "past-1",
                "scenario_name": "scenario",
                "metadata": {"scenario_name": "scenario"},
            },
        ),
        organs=(),
        receipts=(),
    )

    assert report.state is AgentState.OBSERVED
    assert report.outputs["stages"]["classify"]["memory_class"] == "traceable_candidate"
    assert report.outputs["stages"]["deliberate"]["writes_memory"] is False
    assert report.outputs["stages"]["deliberate"]["promotion_authorized"] is False
    assert report.outputs["mfm_authority_preserved"] is True


def test_memory_agent_quarantines_untraceable_or_cross_scenario_hits() -> None:
    identity = _identity("memory-degraded")
    report = MemoryConsolidationAgent().assess(
        identity=identity,
        memory_hits=(
            {"payload": "sin procedencia"},
            {"episode_id": "other-1", "scenario_name": "other-world"},
        ),
        organs=(),
        receipts=(),
    )

    assert report.state is AgentState.DEGRADED
    assert report.outputs["stages"]["classify"]["memory_class"] == "provenance_degraded"
    assert (
        report.outputs["stages"]["deliberate"]["proposal"]
        == "quarantine_untraceable_or_unattested_memory"
    )
    codes = {finding.code for finding in report.findings}
    assert "memory_provenance_missing" in codes
    assert "memory_cross_scenario_requires_attestation" in codes


def test_pedagogical_agent_links_applied_lesson_to_measured_improvement() -> None:
    identity = _identity("teacher-improved")
    lesson = {
        "lesson_id": "lesson-1",
        "situation_key": "situation-1",
        "avoid": "bad-action",
        "prefer": "better-action",
        "from_severity": 0.8,
    }
    report = PedagogicalTeacherAgent().assess(
        identity=identity,
        lessons=(lesson,),
        outcome={
            "experience": {"situation_key": "situation-1", "severity": 0.3},
            "experience_bias": {"avoided": "bad-action", "chose": "better-action"},
        },
        certificate={"verdict": "certified"},
        reward={"reward": 0.4},
    )

    assert report.state is AgentState.OBSERVED
    classification = report.outputs["stages"]["classify"]["pedagogical_class"]
    assert classification == "applied_improved_single_observation"
    comparison = report.outputs["stages"]["analyze"]["comparisons"][0]
    assert comparison["severity_reduction"] == 0.5
    assert comparison["improved"] is True
    assert report.outputs["stages"]["analyze"]["causal_effect_proven"] is False


def test_model_data_immune_admits_clean_observation_without_training_authority() -> None:
    identity = _identity("immune-clean")
    organ = _organ(identity)
    report = ModelDataImmuneAgent().assess(
        identity=identity,
        organs=(organ,),
        receipts=(_receipt(identity, organ),),
        trace_health={"pending_events": 0, "dropped_events": 0, "degraded": False},
    )

    assert report.state is AgentState.OBSERVED
    assert report.outputs["stages"]["classify"]["immune_class"] == "clean_observation"
    assert report.outputs["stages"]["analyze"]["training_evidence_eligible"] is True
    assert report.outputs["stages"]["deliberate"]["training_authorized"] is False


def test_model_data_immune_quarantines_incomplete_artifact_binding() -> None:
    identity = _identity("immune-binding")
    organ = _organ(identity)
    organ.manifest_sha256 = "manifest-without-artifact"
    report = ModelDataImmuneAgent().assess(
        identity=identity,
        organs=(organ,),
        receipts=(),
        trace_health={"pending_events": 0, "dropped_events": 0, "degraded": False},
    )

    assert report.state is AgentState.BLOCKED
    assert report.outputs["stages"]["classify"]["quarantined_organs"] == ["N1"]
    assert report.outputs["stages"]["analyze"]["training_evidence_eligible"] is False
    assert any(
        finding.code == "immune_incomplete_artifact_binding"
        for finding in report.findings
    )


def test_curriculum_requires_paired_7b_codex_trials_before_ranking() -> None:
    identity = _identity("curriculum")
    lesson = {
        "lesson_id": "lesson-local",
        "teacher_source": "local_7b",
        "teacher_latency_s": 2.0,
    }
    pedagogy = PedagogicalTeacherAgent().assess(
        identity=identity,
        lessons=({**lesson, "situation_key": "s", "avoid": "bad", "prefer": "good", "from_severity": 0.8},),
        outcome={"experience": {"situation_key": "s", "severity": 0.4}, "experience_bias": {"chose": "good"}},
        certificate={"verdict": "certified"},
        reward={"reward": 0.2},
    )
    report = CurriculumLearningAgent().assess(identity=identity, lessons=(lesson,), pedagogical_report=pedagogy)

    assert report.outputs["stages"]["classify"]["curriculum_class"] == "efficiency_unmeasured_unpaired"
    assert report.outputs["stages"]["analyze"]["teacher_efficiency_ranked"] is False
    assert report.outputs["stages"]["analyze"]["local_7b_assumed_efficient"] is False
    assert report.outputs["required_variants"] == ["no_teacher", "local_7b", "codex_frontier"]


def test_sensorimotor_requires_committed_n4_and_causal_attestation() -> None:
    identity = _identity("sensorimotor")
    report = SensorimotorWorldModelAgent().assess(identity=identity, observation={"x": 1}, causal_attestation={}, organs=())
    assert report.state is AgentState.DEGRADED
    assert report.outputs["stages"]["classify"]["sensorimotor_class"] == "open_loop"
    assert report.outputs["stages"]["deliberate"]["actuation_authorized"] is False


def test_metabolic_agent_blocks_when_msrc_budget_is_unavailable() -> None:
    report = MetabolicBudgetAgent().assess(identity=_identity("metabolic"), resources={"msrc_budget_available": False})
    assert report.state is AgentState.BLOCKED
    assert report.outputs["n0_resource_authority_preserved"] is True
    assert report.outputs["stages"]["deliberate"]["budget_mutation_authorized"] is False


def test_development_agent_blocks_irreversible_n6_proposal() -> None:
    identity = _identity("development")
    organ = _organ(identity)
    organ.organ = "N6"
    organ.candidate = {"mutation_type": "parameter_bound", "lineage_id": identity.lineage_id}
    report = DevelopmentLineageAgent().assess(identity=identity, viability={}, organs=(organ,))
    assert report.state is AgentState.BLOCKED
    assert report.outputs["stages"]["deliberate"]["mutation_authorized"] is False


def test_horizontal_creativity_measures_breadth_not_mamba_transport() -> None:
    report = HorizontalCreativityAgent().assess(
        identity=_identity("creativity"),
        reasoning={"sequence": ["ABD", "CAU", "CTF"], "state": {"abd_hypothesis": {}, "cau_link": {}, "ctf_checked": {}}},
        memory_hits=(),
    )
    assert report.state is AgentState.OBSERVED
    assert report.outputs["horizontal_not_latent_transport"] is True
    assert report.outputs["stages"]["analyze"]["mamba2_transport_required"] is False


def test_social_exocortex_quarantines_untraceable_external_evidence() -> None:
    report = SocialExocortexAgent().assess(
        identity=_identity("social"),
        scenario_metadata={"external_evidence": [{"text": "rumor"}]},
    )
    assert report.state is AgentState.DEGRADED
    assert report.outputs["stages"]["classify"]["social_class"] == "external_evidence_untraceable"
    assert report.outputs["stages"]["deliberate"]["external_write_authorized"] is False


def test_pedagogical_agent_quarantines_lesson_without_improvement() -> None:
    identity = _identity("teacher-failed")
    report = PedagogicalTeacherAgent().assess(
        identity=identity,
        lessons=(
            {
                "lesson_id": "lesson-2",
                "situation_key": "situation-2",
                "avoid": "bad-action",
                "prefer": "other-action",
                "from_severity": 0.6,
            },
        ),
        outcome={
            "experience": {"situation_key": "situation-2", "severity": 0.7},
            "experience_bias": {"avoided": "bad-action", "chose": "other-action"},
        },
        certificate={"verdict": "rejected"},
        reward={"reward": -0.2},
    )

    assert report.state is AgentState.DEGRADED
    assert (
        report.outputs["stages"]["deliberate"]["proposal"]
        == "quarantine_lesson_from_curriculum"
    )
    assert any(
        finding.code == "teacher_lesson_failed_outcome_test"
        for finding in report.findings
    )
