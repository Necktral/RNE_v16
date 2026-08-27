"""SMG mínimo: signos persistentes y relaciones básicas."""

from .smg_min import (
    CONTRADICTION,
    MEASURED_RELATION_KINDS,
    NO_DISCRIMINATING_EVIDENCE,
    Observation,
    RelationKind,
    SignNode,
    SignRelation,
    SMGMin,
    SUPPORT,
    VALID_RELATION_KINDS,
    is_measured_relation,
)

__all__ = [
    "CONTRADICTION",
    "MEASURED_RELATION_KINDS",
    "NO_DISCRIMINATING_EVIDENCE",
    "Observation",
    "RelationKind",
    "SignNode",
    "SignRelation",
    "SMGMin",
    "SUPPORT",
    "VALID_RELATION_KINDS",
    "is_measured_relation",
]
