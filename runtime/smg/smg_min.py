"""Semantic Memory Graph mínimo para RNFE."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal
from uuid import uuid4

from runtime.storage import get_storage
from runtime.storage.records import utc_now_iso


# ── Vocabulario de relaciones semióticas (SSOT) ──────────────────────────────
#
# El contraste factual/contrafactual tiene TRES resultados, no dos. Los dos primeros son
# MEDICIONES (el contrafactual discriminó); el tercero es una DECLARACIÓN de que no hubo
# nada que medir.
#
# `no_discriminating_evidence` NO es un tercer color decorativo: es la única forma que
# tiene el grafo de registrar un enlace cuyo soporte causal NO SE MIDIÓ (ambas acciones
# dejaban al organismo del mismo lado de su objetivo). Sin él, el SMG estaba OBLIGADO a
# afirmar soporte o contradicción incluso cuando no tenía evidencia de ninguno de los dos.
#
# Vive acá —y no en `runtime/world/scenario.py`— porque el SMG es el módulo hoja que
# registra estas relaciones: `runtime.world` y `runtime.reality` pueden importarlo sin
# ciclos, y al revés no (world/__init__ arrastra el runner, que importa reality).

SUPPORT = "support"
CONTRADICTION = "contradiction"
NO_DISCRIMINATING_EVIDENCE = "no_discriminating_evidence"

RelationKind = Literal["support", "contradiction", "no_discriminating_evidence"]

#: Todas las relaciones que el grafo acepta.
VALID_RELATION_KINDS = frozenset({SUPPORT, CONTRADICTION, NO_DISCRIMINATING_EVIDENCE})

#: Relaciones que constituyen evidencia causal MEDIDA (las únicas puntuables).
MEASURED_RELATION_KINDS = frozenset({SUPPORT, CONTRADICTION})


def is_measured_relation(relation_kind: str | None) -> bool:
    """True si ``relation_kind`` es evidencia causal medida (no una abstención).

    El eje causal sólo puede puntuarse cuando el contrafactual DISCRIMINÓ. Todo lo demás
    —``no_discriminating_evidence``, ``None``, un valor desconocido— es un NO MEDIDO y
    debe declararse como tal, nunca rellenarse con un número.
    """
    return relation_kind in MEASURED_RELATION_KINDS


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
        if kind not in VALID_RELATION_KINDS:
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
