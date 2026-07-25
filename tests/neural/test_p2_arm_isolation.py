"""W0-R2 characterization: deterministic unit snapshots and arm isolation."""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import FrozenInstanceError

import pytest

from runtime.memory.embeddings import reset_embedder
from runtime.memory.embeddings.llama_cpp_embedder import LlamaCppEmbedder
from runtime.memory.mfm_lite.retrieval import MemoryRetrieval
from runtime.neural.integration.p2_arm_isolation import (
    BackendOutputSnapshot,
    capture_unit_state,
    deserialize_snapshot,
    dispose_arm_context,
    instantiate_arm_context,
    serialize_snapshot,
    verify_arm_prestate,
)
from runtime.neural.integration.p2_n3_decision import (
    P2CandidatePool,
    P2N3DecisionEvaluator,
    P2PreActionSnapshot,
)
from runtime.neural.technology_backends import Mamba2TemporalTorchBackend
from runtime.storage.records import MemoryRecord
from runtime.world.deferred_load_scenario import DeferredLoadScenario
from runtime.world.grid_thermal_scenario import GridThermalScenario
from runtime.world.resource_scenario import ResourceScenario
from runtime.world.thermal_scenario import ThermalScenario


SCENARIOS = (
    ThermalScenario(initial_temperature=0.73, cooling_effect=0.09),
    ResourceScenario(initial_stock=0.17, production_rate=0.11),
    DeferredLoadScenario(initial_load=0.66, boost_debt=0.12),
    GridThermalScenario(
        initial_temperature=0.71,
        topology="hotspot_center",
        topology_params={"hotspot_delta": 0.12},
    ),
)


def _record(memory_id: str, *, created_at: str, scale: str = "micro") -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        run_id="run",
        episode_id=f"episode-{memory_id}",
        scale=scale,
        structure_json={
            "proposition": "TEMP_HIGH",
            "intervention": "activate_cooling",
        },
        ttl_seconds=None,
        certificate_id=f"cert-{memory_id}",
        ioc_proxy=0.5,
        support_count=2,
        created_at=created_at,
        metadata={
            "scenario_metadata": {
                "scenario_name": "thermal_homeostasis",
                "scenario_version": "1.0",
            }
        },
    )


def _capture(
    scenario=None,
    *,
    seed: int = 7,
    storage_records=None,
    retrieval_configuration=None,
    canonical_scored_pool=None,
    reference_state=None,
    trained_state=None,
    reference_deriver=None,
    trained_deriver=None,
):
    scenario = scenario or ThermalScenario(initial_temperature=0.73)
    return capture_unit_state(
        unit_id=f"{scenario.config.name}:{seed}:3",
        scenario=scenario,
        seed=seed,
        episode_index=3,
        external_input=0.02,
        storage_records=storage_records
        if storage_records is not None
        else [
            _record("first", created_at="2026-01-02T00:00:00+00:00"),
            _record(
                "second",
                created_at="2026-01-01T00:00:00+00:00",
                scale="macro",
            ),
        ],
        retrieval_configuration=retrieval_configuration
        or {"limit": 2, "embedding_mode": "off"},
        canonical_scored_pool=canonical_scored_pool
        or {
            "scoring_version": "mfm-canonical-structural-v1",
            "candidates": [
                {"memory_id": "first", "canonical_score": 1.0},
                {"memory_id": "second", "canonical_score": 0.5},
            ],
        },
        reference_state=reference_state
        or {
            "schema_version": "n3-temporal-checkpoint-v1",
            "entries": [],
        },
        trained_state=trained_state
        or {
            "schema_version": "mamba2-temporal-history-v1",
            "entries": [],
        },
        reference_deriver=reference_deriver or (lambda: {"status": "ok", "risk": 0.2}),
        trained_deriver=trained_deriver
        or (lambda: {"status": "ok", "risk": 0.3}),
        environment={},
    )


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_snapshot_round_trip_preserves_all_four_scenarios(scenario):
    snapshot = _capture(scenario)
    restored = deserialize_snapshot(serialize_snapshot(snapshot))
    context = instantiate_arm_context(restored, "canonical")

    assert restored == snapshot
    assert verify_arm_prestate(restored, context) is True
    assert context.scenario.observe().state == scenario.observe().state
    assert serialize_snapshot(restored) == serialize_snapshot(snapshot)


def test_snapshot_hash_changes_for_every_declared_causal_surface():
    base = _capture()
    assert _capture().snapshot_hash == base.snapshot_hash
    variants = [
        _capture(ThermalScenario(initial_temperature=0.74)),
        _capture(
            storage_records=[
                _record("changed", created_at="2026-01-02T00:00:00+00:00")
            ]
        ),
        _capture(
            storage_records=list(
                reversed(
                    [
                        _record("first", created_at="2026-01-02T00:00:00+00:00"),
                        _record("second", created_at="2026-01-01T00:00:00+00:00"),
                    ]
                )
            )
        ),
        _capture(reference_state={"entries": [{"state_key": ["a", "b", "c"]}]}),
        _capture(trained_state={"entries": [{"state_key": ["a", "b", "c"]}]}),
        _capture(seed=8),
        _capture(retrieval_configuration={"limit": 8, "embedding_mode": "off"}),
        _capture(
            canonical_scored_pool={
                "scoring_version": "mfm-canonical-structural-v1",
                "candidates": [{"memory_id": "second", "canonical_score": 0.5}],
            }
        ),
        _capture(reference_deriver=lambda: {"status": "ok", "risk": 0.4}),
    ]
    changed_actions = ThermalScenario(initial_temperature=0.73)
    changed_actions.config.interventions = [
        "deactivate_cooling",
        "activate_cooling",
    ]
    variants.append(_capture(changed_actions))
    changed_cell = GridThermalScenario(initial_temperature=0.71)
    changed_cell._grid.cells[0].temperature += 0.01
    variants.append(_capture(changed_cell))

    assert len({item.snapshot_hash for item in [base, *variants]}) == len(variants) + 1


def test_snapshot_and_backend_outputs_are_deeply_immutable():
    snapshot = _capture()
    original = serialize_snapshot(snapshot)
    payload = snapshot.payload
    payload["scenario"]["state"]["temperature"] = -1
    payload["storage_records"][0]["metadata"]["mutated"] = True

    outputs = BackendOutputSnapshot.from_dict(snapshot.payload["backend_outputs"])
    reference = outputs.reference_output
    reference["risk"] = 99

    with pytest.raises(FrozenInstanceError):
        snapshot.seed = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        outputs.sha256 = "bad"  # type: ignore[misc]
    assert serialize_snapshot(snapshot) == original
    assert outputs.reference_output["risk"] == 0.2


def test_contexts_have_no_mutable_aliases_with_each_other_or_sources():
    source = ThermalScenario(initial_temperature=0.73)
    source_records = [_record("first", created_at="2026-01-02T00:00:00+00:00")]
    snapshot = _capture(source, storage_records=source_records)
    a = instantiate_arm_context(snapshot, "A")
    b = instantiate_arm_context(snapshot, "B")

    a.scenario.factual_transition(
        intervention="activate_cooling", external_input=0.1
    )
    a_record = a.storage.retrieve_memory_records(
        run_id="run", scales=["micro"], limit=1
    )[0]
    a_record.structure_json["mutated"] = True
    a.python_rng.random()

    assert b.scenario.observe().state == source.observe().state
    assert "mutated" not in b.storage.retrieve_memory_records(
        run_id="run", scales=["micro"], limit=1
    )[0].structure_json
    assert "mutated" not in source_records[0].structure_json
    assert verify_arm_prestate(snapshot, b) is True
    with pytest.raises(RuntimeError, match="read_only"):
        a.storage.write_memory_record(a_record)


def test_reference_and_trained_are_derived_once_per_unit_not_per_arm():
    calls = {"reference": 0, "trained": 0}

    def reference():
        calls["reference"] += 1
        return {"status": "ok"}

    def trained():
        calls["trained"] += 1
        return {"status": "ok"}

    snapshot = _capture(
        reference_deriver=reference,
        trained_deriver=trained,
    )
    contexts = [
        instantiate_arm_context(snapshot, arm)
        for arm in ("canonical", "n3-reference", "n3-trained")
    ]
    assert calls == {"reference": 1, "trained": 1}
    assert len({item.backend_outputs.sha256 for item in contexts}) == 1


def test_unit_prestate_is_frozen_before_stateful_backend_derivation():
    reference_state = {"entries": []}
    trained_state = {"entries": []}
    before = random.getstate()
    expected_rng = random.Random()
    expected_rng.setstate(before)

    def reference():
        reference_state["entries"].append({"advanced": True})
        random.random()
        return {"status": "ok", "producer": "reference"}

    def trained():
        trained_state["entries"].append({"advanced": True})
        random.random()
        return {"status": "ok", "producer": "trained"}

    try:
        snapshot = _capture(
            reference_state=reference_state,
            trained_state=trained_state,
            reference_deriver=reference,
            trained_deriver=trained,
        )
    finally:
        random.setstate(before)

    assert reference_state["entries"] == [{"advanced": True}]
    assert trained_state["entries"] == [{"advanced": True}]
    assert snapshot.payload["reference_state"]["entries"] == []
    assert snapshot.payload["trained_state"]["entries"] == []
    context = instantiate_arm_context(snapshot, "canonical")
    assert context.python_rng.random() == expected_rng.random()


def _decision_fixture(snapshot):
    payload = snapshot.payload
    scenario = instantiate_arm_context(snapshot, "fixture").scenario
    observation = scenario.observe()
    preaction = P2PreActionSnapshot.build(
        scenario=scenario.config.name,
        seed=snapshot.seed,
        episode_index=snapshot.episode_index,
        observation={
            **observation.state,
            "alarm": observation.alarm,
            "propositions": observation.propositions,
        },
        allowed_interventions=scenario.config.interventions,
        external_input=payload["external_input"],
    )
    pool = P2CandidatePool.freeze(
        (
            {
                "memory_id": "micro",
                "scale": "micro",
                "score": 1.0,
                "structure": {"intervention": "activate_cooling"},
            },
            {
                "memory_id": "macro",
                "scale": "macro",
                "score": 1.0,
                "structure": {"intervention": "deactivate_cooling"},
            },
        )
    )
    return preaction, pool


def _run_order(snapshot, order):
    preaction, pool = _decision_fixture(snapshot)
    evaluator = P2N3DecisionEvaluator(campaign_id="w0-r2-test")
    results = {}
    signals = {
        "canonical": None,
        "n3-reference": {"micro": 1.0, "macro": 0.0},
        "n3-trained": {"micro": 0.0, "macro": 1.0},
    }
    for arm in order:
        context = instantiate_arm_context(snapshot, arm)
        receipt = evaluator.decide(
            scenario=context.scenario,
            snapshot=preaction,
            pool=pool,
            arm_id=arm,
            scale_signals=signals[arm],
        )
        assert verify_arm_prestate(snapshot, context) is True
        results[arm] = receipt.to_dict()
    return results


def test_arm_order_and_repeated_arm_are_invariant():
    snapshot = _capture()
    orders = [
        ("canonical", "n3-reference", "n3-trained"),
        ("n3-trained", "canonical", "n3-reference"),
        ("n3-reference", "n3-trained", "canonical"),
    ]
    runs = [_run_order(snapshot, order) for order in orders]

    for arm in orders[0]:
        assert runs[0][arm] == runs[1][arm] == runs[2][arm]
    first = _run_order(snapshot, ("canonical",))["canonical"]
    second = _run_order(snapshot, ("canonical",))["canonical"]
    assert first == second


def test_embedding_cache_warm_and_cold_are_equivalent(monkeypatch):
    monkeypatch.setenv("RNFE_MEMORY_EMBEDDINGS", "hashed")
    monkeypatch.setenv("RNFE_MEMORY_EMBEDDINGS_WEIGHT", "0.4")
    snapshot = _capture()
    cold_context = instantiate_arm_context(snapshot, "cold")
    reset_embedder()
    cold = MemoryRetrieval(storage=cold_context.storage).retrieve(
        run_id="run",
        query={"proposition": "TEMP_HIGH"},
        limit=2,
        candidate_pool_size=20,
    )
    warm_context = instantiate_arm_context(snapshot, "warm")
    warm_retrieval = MemoryRetrieval(storage=warm_context.storage)
    warm_retrieval.retrieve(
        run_id="run",
        query={"proposition": "TEMP_HIGH"},
        limit=2,
        candidate_pool_size=20,
    )
    warm = warm_retrieval.retrieve(
        run_id="run",
        query={"proposition": "TEMP_HIGH"},
        limit=2,
        candidate_pool_size=20,
    )
    assert cold == warm
    preaction, _ = _decision_fixture(snapshot)
    evaluator = P2N3DecisionEvaluator(campaign_id="w0-r2-cache-test")
    cold_receipt = evaluator.decide(
        scenario=cold_context.scenario,
        snapshot=preaction,
        pool=P2CandidatePool.freeze(cold),
        arm_id="canonical",
    )
    warm_receipt = evaluator.decide(
        scenario=warm_context.scenario,
        snapshot=preaction,
        pool=P2CandidatePool.freeze(warm),
        arm_id="canonical",
    )
    assert cold_receipt.chosen_intervention == warm_receipt.chosen_intervention
    reset_embedder()


def test_llama_provider_internal_cache_is_content_equivalent(monkeypatch):
    calls = {"count": 0}
    provider = LlamaCppEmbedder(cache_size=2)
    monkeypatch.setattr(provider, "available", lambda: True)

    def deterministic_run(text):
        calls["count"] += 1
        return [float(len(text)), 1.0]

    monkeypatch.setattr(provider, "_run", deterministic_run)
    cold = provider.embed("same preaction input")
    warm = provider.embed("same preaction input")

    assert cold == warm == [20.0, 1.0]
    assert calls["count"] == 1


def test_rng_instances_are_isolated():
    snapshot = _capture()
    a = instantiate_arm_context(snapshot, "A")
    b = instantiate_arm_context(snapshot, "B")
    expected = instantiate_arm_context(snapshot, "expected")

    a.python_rng.random()
    assert b.python_rng.random() == expected.python_rng.random()
    if a.numpy_rng is not None:
        a.numpy_rng.random_sample()
        assert b.numpy_rng.random_sample() == expected.numpy_rng.random_sample()
    if a.torch_cpu_rng is not None:
        import torch

        torch.rand(1, generator=a.torch_cpu_rng)
        assert torch.equal(
            torch.rand(1, generator=b.torch_cpu_rng),
            torch.rand(1, generator=expected.torch_cpu_rng),
        )


def test_oracle_is_closed_until_decision_seal_and_prestate_has_no_oracle_data():
    snapshot = _capture()
    context = instantiate_arm_context(snapshot, "canonical")
    text = serialize_snapshot(snapshot).decode()
    assert all(
        key not in text
        for key in (
            '"outcomes"',
            '"optimal_utility"',
            '"regret"',
            '"oracle_action"',
        )
    )
    with pytest.raises(RuntimeError, match="oracle_access_before"):
        context.scenario.simulate_counterfactual(
            intervention="activate_cooling", external_input=0.01
        )
    context.scenario.seal_preaction_decision("a" * 64)
    assert context.scenario.simulate_counterfactual(
        intervention="activate_cooling", external_input=0.01
    )


def test_environment_drift_blocks_context_instantiation(monkeypatch):
    monkeypatch.setenv("RNFE_MEMORY_EMBEDDINGS", "off")
    snapshot = capture_unit_state(
        unit_id="environment:7:3",
        scenario=ThermalScenario(),
        seed=7,
        episode_index=3,
        external_input=0.0,
        storage_records=[],
        retrieval_configuration={"limit": 1},
        canonical_scored_pool={"candidates": []},
        reference_state={"entries": []},
        trained_state={"entries": []},
        reference_deriver=lambda: {"status": "ok"},
        trained_deriver=lambda: {"status": "ok"},
    )
    monkeypatch.setenv("RNFE_MEMORY_EMBEDDINGS", "hashed")
    with pytest.raises(ValueError, match="environment_drift"):
        instantiate_arm_context(snapshot, "canonical")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        _capture(retrieval_configuration={"weight": value})
    with pytest.raises(ValueError, match="finite"):
        _capture(ThermalScenario(initial_temperature=value))


def test_unsupported_objects_and_non_string_mapping_keys_are_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        _capture(retrieval_configuration={"bad": object()})
    with pytest.raises(ValueError, match="keys"):
        _capture(retrieval_configuration={1: "bad"})


def test_trained_history_round_trip_and_compatibility_guards():
    source = Mamba2TemporalTorchBackend()
    source.model_id = "model"
    source.artifact_hash = "artifact"
    source.input_size = 2
    source.history_size = 3
    source.histories[("org", "scenario", "lineage")] = deque(
        [(0.1, 0.2), (0.3, 0.4)], maxlen=3
    )
    payload = source.export_state()

    restored = Mamba2TemporalTorchBackend()
    restored.model_id = "model"
    restored.artifact_hash = "artifact"
    restored.input_size = 2
    restored.history_size = 3
    assert restored.restore_state(payload) == 1
    assert restored.export_state() == payload

    incompatible = Mamba2TemporalTorchBackend()
    with pytest.raises(ValueError, match="backend_mismatch"):
        incompatible.restore_state(payload)


def test_disposed_context_rejects_further_use():
    context = instantiate_arm_context(_capture(), "canonical")
    dispose_arm_context(context)
    dispose_arm_context(context)
    with pytest.raises(RuntimeError, match="disposed"):
        context.scenario.observe()
    with pytest.raises(RuntimeError, match="disposed"):
        context.storage.retrieve_memory_records(run_id="run", limit=1)
    with pytest.raises(RuntimeError, match="disposed"):
        context.canonical_pool
