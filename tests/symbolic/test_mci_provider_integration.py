from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from runtime.storage import StorageConfig, StorageFactory
from runtime.symbolic.mci import (
    EdgeMapping,
    ExternalHypothesis,
    HypothesisMappingRegistry,
    MCIRuntime,
    NeuralHypothesisEvaluation,
)
from runtime.symbolic.mci.causal_learning import TransitionEvidence
from runtime.symbolic.mci.compiler import TransitionCompiler
from runtime.symbolic.mci.feedback import relay_feedback
from runtime.symbolic.mci.provider_adapter import adapt_hypotheses
from runtime.symbolic.mci.specs import thermal_spec
from runtime.world import ScenarioEpisodeRunner


class DeterministicProvider:
    def __init__(self, proposals: Sequence[ExternalHypothesis]) -> None:
        self.proposals = tuple(proposals)
        self.contexts: list[Mapping[str, Any]] = []
        self.feedback: list[NeuralHypothesisEvaluation] = []

    def infer_hypotheses(
        self, context: Mapping[str, Any]
    ) -> Sequence[ExternalHypothesis]:
        self.contexts.append(context)
        proposals, self.proposals = self.proposals, ()
        return proposals

    def receive_feedback(
        self, evaluations: Sequence[NeuralHypothesisEvaluation]
    ) -> None:
        self.feedback.extend(evaluations)


def _thermal_evidence(
    base_effect: float,
    actual_effect: float,
    episode: int,
) -> TransitionEvidence:
    base = thermal_spec(cooling_effect=base_effect)
    actual = thermal_spec(cooling_effect=actual_effect)
    state = {"temperature": 0.9, "cooling_active": False, "alarm": True}
    return TransitionEvidence(
        state=state,
        action="activate_cooling",
        external_input=0.0,
        observed=TransitionCompiler(actual).execute(
            state, action="activate_cooling", external_input=0.0
        ),
        predicted=TransitionCompiler(base).execute(
            state, action="activate_cooling", external_input=0.0
        ),
        replay_unit_id=f"provider/ep-{episode}",
        logical_time=episode * 2 + 2,
    )


def _storage(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(tmp_path / "provider.db"),
            postgres_dsn=None,
            artifact_root=tmp_path / "artifacts",
            prefer_postgres_reads=True,
            strict_dual_write=False,
        )
    )


def test_edge_mapping_promotes_a_parameter_without_changing_ir_hash():
    base = thermal_spec(cooling_effect=0.07)
    runtime = MCIRuntime(base)
    for episode in range(11):
        runtime.learner.observe(_thermal_evidence(0.07, 0.14, episode))
    registry = HypothesisMappingRegistry(
        (
            EdgeMapping(
                "cooling_active",
                "temperature",
                "parameter",
                "cooling_effect",
            ),
        )
    )
    proposal = ExternalHypothesis(
        kind="edge",
        source="cooling_active",
        target="temperature",
        proposed_value=0.14,
        confidence=0.9,
        provider="synthetic",
        model_ref="deterministic-v1",
    )
    hypothesis = adapt_hypotheses(
        (proposal,),
        registry=registry,
        spec=base,
        recent_evidence=runtime.learner.get_recent_evidence(),
        logical_time=23,
    )[0]
    runtime.ingest_neural_hypotheses([hypothesis])
    overlay = runtime.learner.observe(_thermal_evidence(0.07, 0.14, 11))
    evaluation = runtime.learner.last_neural_evaluations()[0]

    assert base.sha256 == thermal_spec(cooling_effect=0.07).sha256
    assert hypothesis.kind == "parameter"
    assert hypothesis.target_id == "cooling_effect"
    assert overlay is not None
    assert overlay.parameter_updates["cooling_effect"] == 0.14
    assert overlay.evidence["source"] == "neural"
    assert evaluation.status == "promoted"


def test_unmapped_edge_is_discarded_and_evidence_mapping_is_conservative():
    spec = thermal_spec()
    evidence = _thermal_evidence(0.07, 0.14, 1)
    proposals = (
        ExternalHypothesis(
            kind="edge",
            source="unknown",
            target="temperature",
            proposed_value=1.0,
            evidence_refs=("invented",),
        ),
        ExternalHypothesis(
            kind="parameter",
            target_id="cooling_effect",
            proposed_value=0.14,
            evidence_refs=(evidence.replay_unit_id, "invented"),
        ),
    )
    adapted = adapt_hypotheses(
        proposals,
        registry=HypothesisMappingRegistry(),
        spec=spec,
        recent_evidence=(evidence,),
        logical_time=3,
    )
    repeated = adapt_hypotheses(
        proposals,
        registry=HypothesisMappingRegistry(),
        spec=spec,
        recent_evidence=(evidence,),
        logical_time=3,
    )
    assert len(adapted) == 1
    assert adapted[0].evidence_refs == (evidence.evidence_id,)
    assert [item.to_dict() for item in adapted] == [
        item.to_dict() for item in repeated
    ]


def test_feedback_relay_uses_the_typed_contract():
    provider = DeterministicProvider(())
    evaluations = relay_feedback(
        provider,
        (
            {
                "hypothesis_id": "h-1",
                "status": "rejected",
                "reason": "empirical_gates_failed",
                "train_mae_before": 0.2,
                "train_mae_after": 0.2,
                "holdout_mae_before": 0.2,
                "holdout_mae_after": 0.2,
                "promoted_overlay_id": None,
            },
        ),
    )
    assert provider.feedback == list(evaluations)
    assert provider.feedback[0].hypothesis_id == "h-1"


def test_runner_invokes_provider_only_with_active_mci(tmp_path: Path):
    proposal = ExternalHypothesis(
        kind="parameter",
        target_id="cooling_effect",
        proposed_value=0.14,
        confidence=0.9,
        provider="synthetic",
        model_ref="deterministic-v1",
    )
    active_provider = DeterministicProvider((proposal,))
    active = ScenarioEpisodeRunner(
        storage=_storage(tmp_path / "active"),
        run_id="provider-active",
        scenario="thermal_homeostasis",
        family_profile="mci_integrated_v1",
    )
    active.set_hypothesis_provider(active_provider)
    result = active.run_episode(
        replay_unit_id="provider-active/ep-1",
        trace_dir=tmp_path / "active-traces",
    )

    inactive_provider = DeterministicProvider((proposal,))
    inactive = ScenarioEpisodeRunner(
        storage=_storage(tmp_path / "inactive"),
        run_id="provider-inactive",
        scenario="thermal_homeostasis",
        family_profile="core_plus_opt",
    )
    inactive.set_hypothesis_provider(inactive_provider)
    inactive_result = inactive.run_episode(
        replay_unit_id="provider-inactive/ep-1",
        trace_dir=tmp_path / "inactive-traces",
    )

    assert len(active_provider.contexts) == 1
    assert active_provider.contexts[0]["candidate_action"] in {
        "activate_cooling",
        "deactivate_cooling",
    }
    assert result["mci_outcome"]["neural_hypotheses_evaluated"][0][
        "status"
    ] == "pending"
    assert active_provider.feedback[0].status == "pending"
    assert inactive_provider.contexts == []
    assert "mci_outcome" not in inactive_result


def test_n4_closed_loop_profile_seals_versioned_n4_evidence(tmp_path: Path):
    runner = ScenarioEpisodeRunner(
        storage=_storage(tmp_path / "n4"),
        run_id="n4-closed-loop",
        scenario="thermal_with_battery",
        family_profile="mci_n4_closed_loop_v1",
    )
    result = runner.run_episode(
        replay_unit_id="n4-closed-loop/ep-1",
        trace_dir=tmp_path / "traces",
    )
    decision_path = Path(result["acting_trace"]["paths"]["decision_trace"])
    import json

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    evidence = decision["mci_evidence"]["n4_evidence"]
    assert evidence["schema"] == "n4_evidence.v1"
    assert evidence["model_ref"] == "n4-reference-ranking-v1"
    assert result["mci_outcome"]["status"] == "observed"
