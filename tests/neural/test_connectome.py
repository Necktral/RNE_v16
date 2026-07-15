from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from runtime.neural.connectome import (
    CONNECTOME_ACTIVITY_SCHEMA_VERSION,
    CONNECTOME_SCHEMA_VERSION,
    AuthorityEffect as ConnectomeAuthority,
    ConnectomeEdge,
    ConnectomeEdgeType,
    ConnectomeNode,
    ConnectomeNodeType,
    ConnectomeRuntime,
    ConnectomeTopology,
    canonical_connectome,
)
from runtime.neural.integration import (
    AuthorityEffect,
    ConsumerReceipt,
    ConsumerVerdictClass,
    OrganTrace,
    SymbiosisIdentity,
    SymbioticNeuralCoordinator,
)
from runtime.storage import StorageConfig, StorageFactory


def _identity(episode: str = "episode-connectome") -> SymbiosisIdentity:
    return SymbiosisIdentity(
        trace_group_id=f"trace-{episode}",
        organism_id="organism-connectome",
        lineage_id="lineage-connectome",
        run_id="run-connectome",
        episode_id=episode,
        scenario_id="thermal@1",
        decision_id="decision-connectome",
    )


def _organ(identity: SymbiosisIdentity, organ: str = "N1") -> OrganTrace:
    return OrganTrace(
        identity=identity,
        organ=organ,
        capability="test",
        requested_mode="shadow",
        effective_mode="shadow",
        authority_ceiling="shadow",
        input_hash="input",
        candidate_hash="candidate",
        consumer="test",
        consumer_verdict="observed",
        latency_ms=0.1,
        candidate={"proposal": True},
    )


def _receipt(
    identity: SymbiosisIdentity,
    index: int,
    *,
    verdict: ConsumerVerdictClass = ConsumerVerdictClass.COMPARED,
) -> ConsumerReceipt:
    return ConsumerReceipt(
        receipt_id=f"receipt-{index}",
        identity=identity,
        organ="N1",
        candidate_hash="candidate",
        consumer_id="scheduler_comparison",
        consumer_contract_version="scheduler-comparison-v1",
        consumer_input_hash=f"input-{index}",
        consumer_output_hash=f"output-{index}",
        verdict_class=verdict,
        verdict_detail=verdict.value,
        evidence_refs=(f"decision-{index}",),
        authority_effect=AuthorityEffect.EVIDENCE_ONLY,
        persisted=True,
    )


def test_canonical_connectome_is_typed_deterministic_and_non_authoritative() -> None:
    first = canonical_connectome()
    second = canonical_connectome()
    assert first.schema_version == CONNECTOME_SCHEMA_VERSION
    assert first.topology_hash == second.topology_hash
    assert {node.node_id for node in first.nodes}.issuperset(
        {"N0", "N1", "N2", "N3", "N4", "N5", "N6", "MFM", "SMG"}
    )
    node_types = {node.node_id: node.node_type for node in first.nodes}
    for edge in first.edges:
        if node_types[edge.source] is ConnectomeNodeType.NEURAL_ORGAN:
            assert edge.authority_ceiling is not ConnectomeAuthority.AUTHORITATIVE
    assert all(edge.source != edge.target for edge in first.edges)
    assert all(
        any(
            edge.target == organ
            and edge.target_port == "feedback"
            and edge.edge_type is ConnectomeEdgeType.CONSUMER_FEEDBACK
            for edge in first.edges
        )
        for organ in {"N1", "N2", "N3", "N4", "N5", "N6"}
    )


def test_connectome_validation_fails_closed() -> None:
    source = ConnectomeNode(
        "N1", ConnectomeNodeType.NEURAL_ORGAN, "Codex", ConnectomeAuthority.NONE,
        input_ports=("gate",), output_ports=("candidate",),
    )
    target = ConnectomeNode(
        "scheduler", ConnectomeNodeType.REASONING_AUTHORITY, "RNFE",
        ConnectomeAuthority.AUTHORITATIVE, input_ports=("proposal",), output_ports=("verdict",),
    )
    edge = ConnectomeEdge(
        "N1->scheduler", "N1", "scheduler", ConnectomeEdgeType.PROPOSAL,
        "candidate", "proposal", ConnectomeAuthority.EVIDENCE_ONLY, "test",
    )
    with pytest.raises(ValueError, match="duplicate_node"):
        ConnectomeTopology.create(nodes=(source, source), edges=())
    with pytest.raises(ValueError, match="dangling_edge"):
        ConnectomeTopology.create(nodes=(source,), edges=(edge,))
    with pytest.raises(ValueError, match="neural_authority_forbidden"):
        ConnectomeTopology.create(
            nodes=(source, target),
            edges=(replace(edge, authority_ceiling=ConnectomeAuthority.AUTHORITATIVE),),
        )


def test_activity_requires_evidence_and_plasticity_never_applies() -> None:
    identity = _identity()
    runtime = ConnectomeRuntime()
    organ = _organ(identity)

    off = runtime.observe(identity=identity, organs=(organ,), receipts=(), mode="off")
    assert off.schema_version == CONNECTOME_ACTIVITY_SCHEMA_VERSION
    assert off.active_nodes == ()
    assert off.active_connections == ()
    assert off.graph_mutated is False

    neutral_receipt = _receipt(identity, 1)
    neutral = runtime.observe(
        identity=identity, organs=(organ,), receipts=(neutral_receipt,), mode="shadow"
    )
    edge = next(item for item in neutral.active_connections if item.edge_id.startswith("N1->scheduler"))
    assert edge.signal_state == "non_informative"
    assert edge.authority_effect.value == "evidence_only"
    feedback = next(
        item for item in neutral.active_connections
        if item.edge_id == "scheduler->N1:feedback"
    )
    assert feedback.signal_state == "non_informative"
    assert feedback.receipt_ids == (neutral_receipt.receipt_id,)
    assert neutral.plasticity_proposals == ()

    first_receipt = _receipt(identity, 2, verdict=ConsumerVerdictClass.ACCEPTED)
    first = runtime.observe(
        identity=identity, organs=(organ,), receipts=(first_receipt,), mode="shadow"
    )
    assert first.plasticity_proposals[0].eligible is False
    assert first.plasticity_proposals[0].proposed_delta == 0.0

    repeated = runtime.observe(
        identity=identity, organs=(organ,), receipts=(first_receipt,), mode="shadow"
    )
    assert repeated.plasticity_proposals[0].observation_count == 1

    third = runtime.observe(
        identity=identity,
        organs=(organ,),
        receipts=(
            _receipt(identity, 3, verdict=ConsumerVerdictClass.ACCEPTED),
            _receipt(identity, 4, verdict=ConsumerVerdictClass.ACCEPTED),
        ),
        mode="shadow",
    )
    proposal = third.plasticity_proposals[0]
    assert proposal.observation_count == 3
    assert proposal.eligible is True
    assert 0.0 < proposal.proposed_delta <= 0.05
    assert proposal.apply_authorized is False
    assert proposal.authority_effect.value == "none"
    assert third.graph_mutated is False

    checkpoint = runtime.export_state()
    restored = ConnectomeRuntime()
    assert restored.restore_state(checkpoint) == 3
    replayed = restored.observe(
        identity=identity,
        organs=(organ,),
        receipts=(
            _receipt(identity, 2, verdict=ConsumerVerdictClass.ACCEPTED),
            _receipt(identity, 3, verdict=ConsumerVerdictClass.ACCEPTED),
            _receipt(identity, 4, verdict=ConsumerVerdictClass.ACCEPTED),
        ),
        mode="shadow",
    )
    assert replayed.plasticity_proposals[0].observation_count == 3
    with pytest.raises(ValueError, match="topology_mismatch"):
        restored.restore_state({**checkpoint, "topology_hash": "foreign"})


def test_activity_rejects_foreign_or_authoritative_evidence() -> None:
    identity = _identity()
    organ = _organ(identity)
    receipt = _receipt(identity, 1)
    runtime = ConnectomeRuntime()
    with pytest.raises(ValueError, match="receipt_identity_mismatch"):
        runtime.observe(
            identity=identity,
            organs=(organ,),
            receipts=(replace(receipt, identity=_identity("foreign")),),
            mode="shadow",
        )
    with pytest.raises(ValueError, match="candidate_hash_mismatch"):
        runtime.observe(
            identity=identity,
            organs=(organ,),
            receipts=(replace(receipt, candidate_hash="foreign"),),
            mode="shadow",
        )
    with pytest.raises(ValueError, match="authority_forbidden"):
        runtime.observe(
            identity=identity,
            organs=(organ,),
            receipts=(replace(receipt, authority_effect=AuthorityEffect.AUTHORITATIVE),),
            mode="shadow",
        )
    with pytest.raises(ValueError, match="consumer_unknown"):
        runtime.observe(
            identity=identity,
            organs=(organ,),
            receipts=(replace(receipt, consumer_id="invented"),),
            mode="shadow",
        )
    with pytest.raises(ValueError, match="mode_invalid"):
        runtime.observe(
            identity=identity, organs=(organ,), receipts=(receipt,), mode="authoritative"
        )


def _storage(tmp_path: Path):
    return StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(tmp_path / "connectome.db"),
            postgres_dsn=None,
            artifact_root=tmp_path / "artifacts",
            prefer_postgres_reads=False,
            strict_dual_write=False,
        )
    )


def test_coordinator_exposes_real_connectome_activity(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RNFE_NEURAL_MODE", "shadow")
    storage = _storage(tmp_path)
    coordinator = SymbioticNeuralCoordinator(storage=storage)
    identity = _identity("episode-live-connectome")
    coordinator.begin_episode(
        identity=identity,
        observation={"temperature": 0.8},
        formula="temperature > 0.5",
        proposition="temperature high",
        memory_hits=[],
        scenario_metadata={"main_variable": "temperature"},
        causal_attestation={
            "main_variable": "temperature",
            "factual_delta": -0.2,
            "counterfactual_delta": 0.1,
        },
        resources={},
    )
    block = coordinator.certification_block(identity.episode_id)
    topology = coordinator.connectome_topology()
    activity = block["connectome_activity"]
    assert topology["topology_hash"] == activity["topology_hash"]
    assert activity["graph_mutated"] is False
    assert activity["authority_effect"] == "none"
    assert {
        "MSRC", "N0", "N1", "N3", "N4", "N5", "StorageFacade"
    }.issubset(activity["active_nodes"])
    active = {item["edge_id"]: item for item in activity["active_connections"]}
    assert active["MSRC->N0:resources"]["signal_state"] == "available"
    assert active["StorageFacade->N0:persistence"]["signal_state"] == "durable"
    assert active["life-chain->N3:feedback"]["receipt_ids"]
    assert any(
        item["edge_id"] == "N3->life-chain:consumption"
        for item in activity["active_connections"]
    )
    storage.close()
