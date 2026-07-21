"""Conectoma tipado del organismo neural-simbólico.

La topología describe conexiones existentes; no inventa autoridad ni ejecuta
acciones. La actividad funcional se deriva exclusivamente de candidatos y recibos
de consumo ya observados. La plasticidad sólo puede emitir propuestas auditables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .contracts import NeuralMode


CONNECTOME_SCHEMA_VERSION = "rnfe-connectome-v1"
CONNECTOME_ACTIVITY_SCHEMA_VERSION = "rnfe-connectome-activity-v1"
CONNECTOME_PLASTICITY_SCHEMA_VERSION = "rnfe-connectome-plasticity-v1"
CONNECTOME_CHECKPOINT_SCHEMA_VERSION = "rnfe-connectome-checkpoint-v1"


class AuthorityEffect(str, Enum):
    """Techo conectómico; refleja valores públicos sin importar integration."""

    NONE = "none"
    EVIDENCE_ONLY = "evidence_only"
    BOUNDED_PROPOSAL = "bounded_proposal"
    AUTHORITATIVE = "authoritative"


class _Identity(Protocol):
    def to_dict(self) -> dict[str, Any]: ...


def _canonical_sha256(value: Any) -> str:
    # Diferido: evita el ciclo connectome -> integration.__init__ -> coordinator.
    from .integration.contracts import canonical_sha256

    return canonical_sha256(value)


class ConnectomeNodeType(str, Enum):
    RUNTIME = "runtime"
    NEURAL_ORGAN = "neural_organ"
    REASONING_AUTHORITY = "reasoning_authority"
    MEMORY_SUBSTRATE = "memory_substrate"
    WORLD_SUBSTRATE = "world_substrate"
    GOVERNANCE_AUTHORITY = "governance_authority"
    RESOURCE_GOVERNOR = "resource_governor"
    EXPERIENCE_SUBSTRATE = "experience_substrate"


class ConnectomeEdgeType(str, Enum):
    RESOURCE_GATING = "resource_gating"
    PROPOSAL = "proposal"
    VERIFICATION = "verification"
    CONTEXT = "context"
    CAUSAL_COMPARISON = "causal_comparison"
    MEMORY_CANDIDATE = "memory_candidate"
    TEMPORAL_FEEDBACK = "temporal_feedback"
    CERTIFICATION_OBSERVATION = "certification_observation"
    EVOLUTION_EVIDENCE = "evolution_evidence"
    PERSISTENCE = "persistence"
    CONSUMER_FEEDBACK = "consumer_feedback"


@dataclass(frozen=True, slots=True)
class ConnectomeNode:
    node_id: str
    node_type: ConnectomeNodeType
    owner: str
    authority_effect: AuthorityEffect
    input_ports: tuple[str, ...] = ()
    output_ports: tuple[str, ...] = ()
    schema_version: str = CONNECTOME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.node_id or not self.owner:
            raise ValueError("connectome_node_identity_required")
        if self.schema_version != CONNECTOME_SCHEMA_VERSION:
            raise ValueError("connectome_node_schema_mismatch")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["node_type"] = self.node_type.value
        data["authority_effect"] = self.authority_effect.value
        return data


@dataclass(frozen=True, slots=True)
class ConnectomeEdge:
    edge_id: str
    source: str
    target: str
    edge_type: ConnectomeEdgeType
    source_port: str
    target_port: str
    authority_ceiling: AuthorityEffect
    provenance: str
    plastic: bool = False
    schema_version: str = CONNECTOME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        required = (
            self.edge_id,
            self.source,
            self.target,
            self.source_port,
            self.target_port,
            self.provenance,
        )
        if not all(str(item).strip() for item in required):
            raise ValueError("connectome_edge_identity_required")
        if self.schema_version != CONNECTOME_SCHEMA_VERSION:
            raise ValueError("connectome_edge_schema_mismatch")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["edge_type"] = self.edge_type.value
        data["authority_ceiling"] = self.authority_ceiling.value
        return data


@dataclass(frozen=True, slots=True)
class ConnectomeTopology:
    nodes: tuple[ConnectomeNode, ...]
    edges: tuple[ConnectomeEdge, ...]
    topology_hash: str
    schema_version: str = CONNECTOME_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        nodes: Iterable[ConnectomeNode],
        edges: Iterable[ConnectomeEdge],
    ) -> "ConnectomeTopology":
        ordered_nodes = tuple(sorted(nodes, key=lambda item: item.node_id))
        ordered_edges = tuple(sorted(edges, key=lambda item: item.edge_id))
        node_ids = [item.node_id for item in ordered_nodes]
        edge_ids = [item.edge_id for item in ordered_edges]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("connectome_duplicate_node")
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("connectome_duplicate_edge")
        known = set(node_ids)
        for edge in ordered_edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError("connectome_dangling_edge")
            source = next(item for item in ordered_nodes if item.node_id == edge.source)
            target = next(item for item in ordered_nodes if item.node_id == edge.target)
            if edge.source_port not in source.output_ports:
                raise ValueError("connectome_unknown_source_port")
            if edge.target_port not in target.input_ports:
                raise ValueError("connectome_unknown_target_port")
            if (
                source.node_type is ConnectomeNodeType.NEURAL_ORGAN
                and edge.authority_ceiling is AuthorityEffect.AUTHORITATIVE
            ):
                raise ValueError("connectome_neural_authority_forbidden")
        payload = {
            "schema_version": CONNECTOME_SCHEMA_VERSION,
            "nodes": [item.to_dict() for item in ordered_nodes],
            "edges": [item.to_dict() for item in ordered_edges],
        }
        return cls(
            nodes=ordered_nodes,
            edges=ordered_edges,
            topology_hash=_canonical_sha256(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "topology_hash": self.topology_hash,
            "nodes": [item.to_dict() for item in self.nodes],
            "edges": [item.to_dict() for item in self.edges],
        }


@dataclass(frozen=True, slots=True)
class ActiveConnection:
    edge_id: str
    signal_state: str
    evidence_refs: tuple[str, ...]
    receipt_ids: tuple[str, ...]
    authority_effect: AuthorityEffect

    def to_dict(self) -> dict[str, Any]:
        return {
            "edge_id": self.edge_id,
            "signal_state": self.signal_state,
            "evidence_refs": list(self.evidence_refs),
            "receipt_ids": list(self.receipt_ids),
            "authority_effect": self.authority_effect.value,
        }


@dataclass(frozen=True, slots=True)
class PlasticityProposal:
    edge_id: str
    proposed_delta: float
    observation_count: int
    positive_count: int
    negative_count: int
    confidence: float
    eligible: bool
    apply_authorized: bool = False
    authority_effect: AuthorityEffect = AuthorityEffect.NONE
    schema_version: str = CONNECTOME_PLASTICITY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["authority_effect"] = self.authority_effect.value
        return data


@dataclass(frozen=True, slots=True)
class ConnectomeActivitySnapshot:
    identity: _Identity
    topology_hash: str
    active_nodes: tuple[str, ...]
    active_connections: tuple[ActiveConnection, ...]
    plasticity_proposals: tuple[PlasticityProposal, ...]
    snapshot_hash: str
    mode: str
    authority_effect: AuthorityEffect = AuthorityEffect.NONE
    graph_mutated: bool = False
    schema_version: str = CONNECTOME_ACTIVITY_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            **self.identity.to_dict(),
            "topology_hash": self.topology_hash,
            "active_nodes": list(self.active_nodes),
            "active_connections": [item.to_dict() for item in self.active_connections],
            "plasticity_proposals": [item.to_dict() for item in self.plasticity_proposals],
            "snapshot_hash": self.snapshot_hash,
            "mode": self.mode,
            "authority_effect": self.authority_effect.value,
            "graph_mutated": self.graph_mutated,
        }


_CONSUMER_TARGETS: Mapping[tuple[str, str], tuple[str, ...]] = {
    ("N1", "scheduler_comparison"): ("scheduler",),
    ("N1", "delayed_outcome_observer"): ("experience",),
    ("N2", "ded_verifier"): ("DED",),
    ("N2", "lotf_verifier"): ("LOT-F",),
    ("N2", "nesy_verifier"): ("NESY",),
    ("N3", "next_episode_state"): ("life-chain",),
    ("N3", "checkpoint_continuity"): ("MFM",),
    ("N4", "canonical_causal_comparator"): ("CAU", "CTF", "C-GWM"),
    ("N4", "certification_metadata"): ("certification",),
    ("N5", "smg_write_result"): ("SMG",),
    ("N5", "mfm_candidate_gate"): ("MFM",),
    ("N6", "sandbox"): ("autoevolution",),
    ("N6", "certification"): ("certification",),
    ("N6", "autoevolution_evidence_observer"): ("autoevolution",),
}


class ConnectomeRuntime:
    """Observador funcional con memoria plástica acotada y no aplicable."""

    def __init__(self, topology: ConnectomeTopology | None = None) -> None:
        self.topology = topology or canonical_connectome()
        self._observations: dict[str, list[bool]] = {}
        self._observed_receipts: set[str] = set()
        self._receipt_order: list[str] = []

    def observe(
        self,
        *,
        identity: _Identity,
        organs: Sequence[Any],
        receipts: Sequence[Any],
        mode: NeuralMode | str,
        resource_state: Mapping[str, Any] | None = None,
        persistence_state: Mapping[str, Any] | None = None,
    ) -> ConnectomeActivitySnapshot:
        try:
            mode_value = NeuralMode(mode).value
        except ValueError as exc:
            raise ValueError("connectome_mode_invalid") from exc
        if mode_value == NeuralMode.OFF.value:
            active_nodes: tuple[str, ...] = ()
            active_connections: tuple[ActiveConnection, ...] = ()
            proposals: tuple[PlasticityProposal, ...] = ()
        else:
            organ_map = self._validate_observation(identity, organs, receipts)
            candidates = {
                item.organ for item in organs
                if item.candidate_hash and item.effective_mode != NeuralMode.OFF.value
            }
            receipt_groups: dict[tuple[str, str], list[Any]] = {}
            for receipt in receipts:
                targets = _CONSUMER_TARGETS.get((receipt.organ, receipt.consumer_id), ())
                for target in targets:
                    receipt_groups.setdefault((receipt.organ, target), []).append(receipt)
            connections: list[ActiveConnection] = []
            nodes = {"MSRC", "N0", "StorageFacade"}
            normalized_resources = dict(resource_state or {})
            normalized_persistence = dict(persistence_state or {})
            pressures = tuple(
                float(normalized_resources.get(name, 0.0) or 0.0)
                for name in ("cpu_pressure", "memory_pressure", "thermal_pressure")
            )
            budget_available = bool(
                normalized_resources.get("msrc_budget_available", True)
            )
            resource_signal = (
                "blocked"
                if not budget_available
                else "constrained"
                if max(pressures, default=0.0) >= 0.8
                else "available"
            )
            connections.append(
                ActiveConnection(
                    edge_id="MSRC->N0:resources",
                    signal_state=resource_signal,
                    evidence_refs=(f"resource_state:{_canonical_sha256(normalized_resources)}",),
                    receipt_ids=(),
                    authority_effect=AuthorityEffect.NONE,
                )
            )
            storage_configured = bool(
                normalized_persistence.get("storage_configured", False)
            )
            persistence_degraded = bool(
                normalized_persistence.get("degraded", False)
                or int(normalized_persistence.get("pending_events", 0) or 0) > 0
            )
            persistence_signal = (
                "unavailable"
                if not storage_configured
                else "degraded"
                if persistence_degraded
                else "durable"
            )
            connections.append(
                ActiveConnection(
                    edge_id="StorageFacade->N0:persistence",
                    signal_state=persistence_signal,
                    evidence_refs=(
                        f"persistence_state:{_canonical_sha256(normalized_persistence)}",
                    ),
                    receipt_ids=(),
                    authority_effect=AuthorityEffect.NONE,
                )
            )
            for organ in sorted(candidates):
                edge_id = f"N0->{organ}:resource_gating"
                organ_trace = organ_map[organ]
                signal_state = (
                    "fallback"
                    if organ_trace.fallback_reason
                    else "observed_shadow"
                    if organ_trace.effective_mode == "shadow"
                    else "gated"
                )
                connections.append(
                    ActiveConnection(
                        edge_id,
                        signal_state,
                        (organ_trace.fallback_reason or f"{organ}:{signal_state}",),
                        (),
                        AuthorityEffect.NONE,
                    )
                )
                nodes.add(organ)
            for (organ, target), group in sorted(receipt_groups.items()):
                edge_id = f"{organ}->{target}:consumption"
                verdicts = {
                    getattr(item.verdict_class, "value", str(item.verdict_class))
                    for item in group
                }
                semantic_signals = tuple(_plasticity_signal(value) for value in verdicts)
                negative = any(value is False for value in semantic_signals)
                positive = any(value is True for value in semantic_signals)
                connections.append(
                    ActiveConnection(
                        edge_id=edge_id,
                        signal_state=(
                            "rejected"
                            if negative
                            else "accepted"
                            if positive
                            else "non_informative"
                        ),
                        evidence_refs=tuple(sorted({ref for item in group for ref in item.evidence_refs})),
                        receipt_ids=tuple(sorted(item.receipt_id for item in group)),
                        authority_effect=max(
                            (item.authority_effect for item in group),
                            key=lambda effect: list(AuthorityEffect).index(effect),
                        ),
                    )
                )
                connections.append(
                    ActiveConnection(
                        edge_id=f"{target}->{organ}:feedback",
                        signal_state=(
                            "rejected"
                            if negative
                            else "accepted"
                            if positive
                            else "non_informative"
                        ),
                        evidence_refs=tuple(
                            sorted({ref for item in group for ref in item.evidence_refs})
                        ),
                        receipt_ids=tuple(sorted(item.receipt_id for item in group)),
                        authority_effect=AuthorityEffect.EVIDENCE_ONLY,
                    )
                )
                nodes.update((organ, target))
                fresh = [
                    item for item in group
                    if f"{edge_id}\x1f{item.receipt_id}" not in self._observed_receipts
                ]
                if fresh:
                    observations = self._observations.setdefault(edge_id, [])
                    for item in fresh:
                        verdict = getattr(
                            item.verdict_class, "value", str(item.verdict_class)
                        )
                        signal = _plasticity_signal(verdict)
                        if signal is not None:
                            observations.append(signal)
                    if observations:
                        self._observations[edge_id] = observations[-128:]
                    else:
                        self._observations.pop(edge_id, None)
                    for item in fresh:
                        observation_id = f"{edge_id}\x1f{item.receipt_id}"
                        if observation_id not in self._observed_receipts:
                            self._observed_receipts.add(observation_id)
                            self._receipt_order.append(observation_id)
                    if len(self._receipt_order) > 2048:
                        expired = self._receipt_order[:-2048]
                        self._receipt_order = self._receipt_order[-2048:]
                        self._observed_receipts.difference_update(expired)
            active_nodes = tuple(sorted(nodes))
            active_connections = tuple(sorted(connections, key=lambda item: item.edge_id))
            proposals = tuple(
                self._proposal(edge_id, observations)
                for edge_id, observations in sorted(self._observations.items())
            )
        payload = {
            "schema_version": CONNECTOME_ACTIVITY_SCHEMA_VERSION,
            **identity.to_dict(),
            "topology_hash": self.topology.topology_hash,
            "active_nodes": list(active_nodes),
            "active_connections": [item.to_dict() for item in active_connections],
            "plasticity_proposals": [item.to_dict() for item in proposals],
            "mode": mode_value,
            "authority_effect": AuthorityEffect.NONE.value,
            "graph_mutated": False,
        }
        return ConnectomeActivitySnapshot(
            identity=identity,
            topology_hash=self.topology.topology_hash,
            active_nodes=active_nodes,
            active_connections=active_connections,
            plasticity_proposals=proposals,
            snapshot_hash=_canonical_sha256(payload),
            mode=mode_value,
        )

    def _validate_observation(
        self,
        identity: _Identity,
        organs: Sequence[Any],
        receipts: Sequence[Any],
    ) -> dict[str, Any]:
        organ_map: dict[str, Any] = {}
        topology_nodes = {node.node_id for node in self.topology.nodes}
        for organ in organs:
            if organ.identity != identity:
                raise ValueError("connectome_organ_identity_mismatch")
            if organ.organ in organ_map:
                raise ValueError("connectome_duplicate_organ_trace")
            if organ.organ not in topology_nodes or not str(organ.organ).startswith("N"):
                raise ValueError("connectome_organ_unknown")
            organ_map[organ.organ] = organ
        for receipt in receipts:
            if receipt.identity != identity:
                raise ValueError("connectome_receipt_identity_mismatch")
            organ = organ_map.get(receipt.organ)
            if organ is None:
                raise ValueError("connectome_receipt_organ_missing")
            if not organ.candidate_hash or receipt.candidate_hash != organ.candidate_hash:
                raise ValueError("connectome_receipt_candidate_hash_mismatch")
            if (receipt.organ, receipt.consumer_id) not in _CONSUMER_TARGETS:
                raise ValueError("connectome_receipt_consumer_unknown")
            authority = getattr(receipt.authority_effect, "value", receipt.authority_effect)
            if authority not in {
                AuthorityEffect.NONE.value,
                AuthorityEffect.EVIDENCE_ONLY.value,
            }:
                raise ValueError("connectome_receipt_authority_forbidden")
        return organ_map

    def export_state(self) -> dict[str, Any]:
        return {
            "schema_version": CONNECTOME_CHECKPOINT_SCHEMA_VERSION,
            "topology_hash": self.topology.topology_hash,
            "observations": {
                edge_id: list(values[-128:])
                for edge_id, values in sorted(self._observations.items())
            },
            "observed_receipts": list(self._receipt_order[-2048:]),
        }

    def restore_state(self, raw: Mapping[str, Any] | None) -> int:
        payload = dict(raw or {})
        if not payload:
            return 0
        if payload.get("schema_version") != CONNECTOME_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("connectome_checkpoint_schema_mismatch")
        if payload.get("topology_hash") != self.topology.topology_hash:
            raise ValueError("connectome_checkpoint_topology_mismatch")
        plastic_edges = {edge.edge_id for edge in self.topology.edges if edge.plastic}
        restored: dict[str, list[bool]] = {}
        for edge_id, values in dict(payload.get("observations") or {}).items():
            if edge_id not in plastic_edges:
                raise ValueError("connectome_checkpoint_edge_unknown")
            if not isinstance(values, list) or any(type(item) is not bool for item in values):
                raise ValueError("connectome_checkpoint_observation_invalid")
            restored[edge_id] = values[-128:]
        receipts = list(payload.get("observed_receipts") or ())
        if any(not isinstance(item, str) or not item for item in receipts):
            raise ValueError("connectome_checkpoint_receipt_invalid")
        self._observations = restored
        self._receipt_order = receipts[-2048:]
        self._observed_receipts = set(self._receipt_order)
        return sum(len(values) for values in restored.values())

    @staticmethod
    def _proposal(edge_id: str, observations: Sequence[bool]) -> PlasticityProposal:
        total = len(observations)
        positive = sum(observations)
        negative = total - positive
        eligible = total >= 3
        delta = max(-0.05, min(0.05, (positive - negative) / max(total, 1) * 0.01))
        return PlasticityProposal(
            edge_id=edge_id,
            proposed_delta=round(delta, 6) if eligible else 0.0,
            observation_count=total,
            positive_count=positive,
            negative_count=negative,
            confidence=round(total / (total + 5.0), 6),
            eligible=eligible,
        )


def _node(
    node_id: str,
    node_type: ConnectomeNodeType,
    owner: str,
    *,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    authority: AuthorityEffect = AuthorityEffect.NONE,
) -> ConnectomeNode:
    return ConnectomeNode(node_id, node_type, owner, authority, inputs, outputs)


def canonical_connectome() -> ConnectomeTopology:
    """Topología declarada de las fronteras ya ejecutables en RNFE."""

    nodes = [
        _node("N0", ConnectomeNodeType.RUNTIME, "Codex", inputs=("resources", "persistence"), outputs=("gate",)),
        *[
            _node(f"N{i}", ConnectomeNodeType.NEURAL_ORGAN, "Codex", inputs=("gate", "feedback"), outputs=("candidate",))
            for i in range(1, 7)
        ],
        _node("scheduler", ConnectomeNodeType.REASONING_AUTHORITY, "RNFE", inputs=("proposal",), outputs=("verdict",), authority=AuthorityEffect.AUTHORITATIVE),
        _node("DED", ConnectomeNodeType.REASONING_AUTHORITY, "RNFE", inputs=("candidate",), outputs=("verdict",), authority=AuthorityEffect.AUTHORITATIVE),
        _node("LOT-F", ConnectomeNodeType.REASONING_AUTHORITY, "RNFE", inputs=("candidate",), outputs=("verdict",), authority=AuthorityEffect.AUTHORITATIVE),
        _node("NESY", ConnectomeNodeType.REASONING_AUTHORITY, "RNFE", inputs=("candidate",), outputs=("verdict",), authority=AuthorityEffect.AUTHORITATIVE),
        _node("CAU", ConnectomeNodeType.REASONING_AUTHORITY, "RNFE", inputs=("candidate",), outputs=("verdict",), authority=AuthorityEffect.AUTHORITATIVE),
        _node("CTF", ConnectomeNodeType.REASONING_AUTHORITY, "RNFE", inputs=("candidate",), outputs=("verdict",), authority=AuthorityEffect.AUTHORITATIVE),
        _node("C-GWM", ConnectomeNodeType.REASONING_AUTHORITY, "RNFE", inputs=("candidate",), outputs=("verdict",), authority=AuthorityEffect.AUTHORITATIVE),
        _node("MFM", ConnectomeNodeType.MEMORY_SUBSTRATE, "RNFE", inputs=("candidate",), outputs=("context",), authority=AuthorityEffect.AUTHORITATIVE),
        _node("SMG", ConnectomeNodeType.MEMORY_SUBSTRATE, "RNFE", inputs=("candidate",), outputs=("context",), authority=AuthorityEffect.AUTHORITATIVE),
        _node("life-chain", ConnectomeNodeType.WORLD_SUBSTRATE, "RNFE", inputs=("candidate",), outputs=("context",), authority=AuthorityEffect.AUTHORITATIVE),
        _node("certification", ConnectomeNodeType.GOVERNANCE_AUTHORITY, "RNFE", inputs=("observation",), outputs=("verdict",), authority=AuthorityEffect.AUTHORITATIVE),
        _node("autoevolution", ConnectomeNodeType.GOVERNANCE_AUTHORITY, "RNFE", inputs=("proposal",), outputs=("verdict",), authority=AuthorityEffect.AUTHORITATIVE),
        _node("experience", ConnectomeNodeType.EXPERIENCE_SUBSTRATE, "RNFE", inputs=("observation",), outputs=("context",), authority=AuthorityEffect.EVIDENCE_ONLY),
        _node("MSRC", ConnectomeNodeType.RESOURCE_GOVERNOR, "Fable/Opus", inputs=("telemetry",), outputs=("resources",), authority=AuthorityEffect.AUTHORITATIVE),
        _node("StorageFacade", ConnectomeNodeType.GOVERNANCE_AUTHORITY, "Fable/Opus", inputs=("event",), outputs=("persistence",), authority=AuthorityEffect.AUTHORITATIVE),
    ]
    edges: list[ConnectomeEdge] = []
    for organ in (f"N{i}" for i in range(1, 7)):
        edges.append(ConnectomeEdge(f"N0->{organ}:resource_gating", "N0", organ, ConnectomeEdgeType.RESOURCE_GATING, "gate", "gate", AuthorityEffect.NONE, "NeuralRuntime.infer_reference"))
    consumer_edges = {
        ("N1", "scheduler"): ConnectomeEdgeType.PROPOSAL,
        ("N1", "experience"): ConnectomeEdgeType.TEMPORAL_FEEDBACK,
        ("N2", "DED"): ConnectomeEdgeType.VERIFICATION,
        ("N2", "LOT-F"): ConnectomeEdgeType.VERIFICATION,
        ("N2", "NESY"): ConnectomeEdgeType.VERIFICATION,
        ("N3", "life-chain"): ConnectomeEdgeType.TEMPORAL_FEEDBACK,
        ("N3", "MFM"): ConnectomeEdgeType.TEMPORAL_FEEDBACK,
        ("N4", "CAU"): ConnectomeEdgeType.CAUSAL_COMPARISON,
        ("N4", "CTF"): ConnectomeEdgeType.CAUSAL_COMPARISON,
        ("N4", "C-GWM"): ConnectomeEdgeType.CAUSAL_COMPARISON,
        ("N4", "certification"): ConnectomeEdgeType.CERTIFICATION_OBSERVATION,
        ("N5", "SMG"): ConnectomeEdgeType.MEMORY_CANDIDATE,
        ("N5", "MFM"): ConnectomeEdgeType.MEMORY_CANDIDATE,
        ("N6", "autoevolution"): ConnectomeEdgeType.EVOLUTION_EVIDENCE,
        ("N6", "certification"): ConnectomeEdgeType.CERTIFICATION_OBSERVATION,
    }
    target_ports = {
        "scheduler": "proposal", "experience": "observation", "DED": "candidate",
        "LOT-F": "candidate", "NESY": "candidate", "life-chain": "candidate",
        "MFM": "candidate", "CAU": "candidate", "CTF": "candidate", "C-GWM": "candidate",
        "certification": "observation", "SMG": "candidate", "autoevolution": "proposal",
    }
    for (source, target), edge_type in consumer_edges.items():
        edges.append(ConnectomeEdge(f"{source}->{target}:consumption", source, target, edge_type, "candidate", target_ports[target], AuthorityEffect.EVIDENCE_ONLY, "validated ConsumerReceipt", plastic=True))
        target_node = next(item for item in nodes if item.node_id == target)
        feedback_port = target_node.output_ports[0]
        edges.append(
            ConnectomeEdge(
                f"{target}->{source}:feedback",
                target,
                source,
                ConnectomeEdgeType.CONSUMER_FEEDBACK,
                feedback_port,
                "feedback",
                AuthorityEffect.EVIDENCE_ONLY,
                "validated ConsumerReceipt feedback",
            )
        )
    edges.extend((
        ConnectomeEdge("MSRC->N0:resources", "MSRC", "N0", ConnectomeEdgeType.RESOURCE_GATING, "resources", "resources", AuthorityEffect.NONE, "NeuralRuntimeConfig resource policy"),
        ConnectomeEdge("StorageFacade->N0:persistence", "StorageFacade", "N0", ConnectomeEdgeType.PERSISTENCE, "persistence", "persistence", AuthorityEffect.NONE, "NeuralRuntime bounded observability"),
    ))
    return ConnectomeTopology.create(nodes=nodes, edges=edges)


def _plasticity_signal(verdict: str) -> bool | None:
    """Mapeo semántico explícito; neutral/ausente jamás cuenta como éxito."""

    if verdict == "accepted":
        return True
    if verdict in {"rejected", "invalid", "persistence_degraded"}:
        return False
    if verdict in {"observed", "compared", "abstained", "unavailable"}:
        return None
    raise ValueError(f"connectome_verdict_unknown:{verdict}")
