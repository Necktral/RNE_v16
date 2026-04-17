"""Semantic Memory Graph mínimo para RNFE."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal
from uuid import uuid4

from runtime.storage import get_storage
from runtime.storage.records import utc_now_iso


RelationKind = Literal["support", "contradiction"]


@dataclass(slots=True)
class Observation:
    observation_id: str
    payload: Dict[str, Any]
    timestamp: str = field(default_factory=utc_now_iso)


@dataclass(slots=True)
class SignNode:
    sign_id: str
    proposition: str
    observation_id: str
    timestamp: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SignRelation:
    relation_id: str
    source_sign_id: str
    target_sign_id: str
    kind: RelationKind
    timestamp: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SMGMin:
    def __init__(self, *, storage=None, run_id: str | None = None):
        self.storage = storage or get_storage()
        self.run_id = run_id
        self.observations: Dict[str, Observation] = {}
        self.signs: Dict[str, SignNode] = {}
        self.relations: Dict[str, SignRelation] = {}

    def add_observation(self, payload: Dict[str, Any]) -> Observation:
        observation = Observation(observation_id=str(uuid4()), payload=dict(payload))
        self.observations[observation.observation_id] = observation
        self.storage.append_event(
            event_type="smg.observation_added",
            payload={
                "observation_id": observation.observation_id,
                "payload": observation.payload,
                "timestamp": observation.timestamp,
            },
            run_id=self.run_id,
            source="smg_min",
        )
        return observation

    def create_sign(
        self, *, proposition: str, observation_id: str, metadata: Dict[str, Any] | None = None
    ) -> SignNode:
        if observation_id not in self.observations:
            raise KeyError(f"Observación inexistente: {observation_id}")
        sign = SignNode(
            sign_id=str(uuid4()),
            proposition=proposition,
            observation_id=observation_id,
            metadata=dict(metadata or {}),
        )
        self.signs[sign.sign_id] = sign
        self.storage.append_event(
            event_type="smg.sign_created",
            payload={
                "sign_id": sign.sign_id,
                "proposition": sign.proposition,
                "observation_id": sign.observation_id,
                "metadata": sign.metadata,
                "timestamp": sign.timestamp,
            },
            run_id=self.run_id,
            source="smg_min",
        )
        return sign

    def link_signs(
        self,
        *,
        source_sign_id: str,
        target_sign_id: str,
        kind: RelationKind,
        metadata: Dict[str, Any] | None = None,
    ) -> SignRelation:
        if kind not in {"support", "contradiction"}:
            raise ValueError(f"Tipo de relación no soportado: {kind}")
        if source_sign_id not in self.signs or target_sign_id not in self.signs:
            raise KeyError("Signo fuente o destino inexistente")
        relation = SignRelation(
            relation_id=str(uuid4()),
            source_sign_id=source_sign_id,
            target_sign_id=target_sign_id,
            kind=kind,
            metadata=dict(metadata or {}),
        )
        self.relations[relation.relation_id] = relation
        self.storage.append_event(
            event_type="smg.relation_created",
            payload={
                "relation_id": relation.relation_id,
                "source_sign_id": relation.source_sign_id,
                "target_sign_id": relation.target_sign_id,
                "kind": relation.kind,
                "metadata": relation.metadata,
                "timestamp": relation.timestamp,
            },
            run_id=self.run_id,
            source="smg_min",
        )
        return relation

    def snapshot(self) -> Dict[str, Any]:
        return {
            "observations": [asdict(obs) for obs in self.observations.values()],
            "signs": [asdict(sign) for sign in self.signs.values()],
            "relations": [asdict(rel) for rel in self.relations.values()],
        }
