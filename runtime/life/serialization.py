"""Serializacion segura de identidad viva para checkpoints."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from runtime.organism.constitution import OrganismConstitution
from runtime.organism.lineage import (
    DEFAULT_INHERITANCE_RULES,
    InheritanceRule,
    LineageEntry,
    LineageState,
)
from runtime.organism.state import OrganismState


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return jsonable(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return jsonable(asdict(value))
    return str(value)


def organism_to_payload(state: OrganismState) -> Dict[str, Any]:
    return jsonable(state.to_dict())


def organism_from_payload(payload: Dict[str, Any] | None) -> OrganismState:
    return OrganismState.from_dict(dict(payload or {}))


def lineage_to_payload(lineage: LineageState) -> Dict[str, Any]:
    return {
        "lineage_id": lineage.lineage_id,
        "parent_constitution_hash": lineage.parent_constitution_hash,
        "current_constitution_hash": lineage.current_constitution_hash,
        "history": [jsonable(entry) for entry in lineage.history],
        "accepted_modifications": list(lineage.accepted_modifications),
        "inherited_certificates": list(lineage.inherited_certificates),
        "inherited_transport_operators": list(lineage.inherited_transport_operators),
        "forbidden_mutations": sorted(lineage.forbidden_mutations),
        "rollback_ancestry": list(lineage.rollback_ancestry),
        "divergence_points": list(lineage.divergence_points),
        "inheritance_rules": [jsonable(rule) for rule in lineage.inheritance_rules],
    }


def lineage_from_payload(payload: Dict[str, Any] | None) -> LineageState:
    data = dict(payload or {})
    rules = []
    for item in data.get("inheritance_rules") or []:
        if isinstance(item, dict):
            try:
                rules.append(
                    InheritanceRule(
                        name=str(item.get("name", "")),
                        condition=item.get("condition", "certified_safe"),
                        description=str(item.get("description", "")),
                    )
                )
            except Exception:
                continue
    history = []
    for item in data.get("history") or []:
        if not isinstance(item, dict):
            continue
        try:
            history.append(
                LineageEntry(
                    entry_id=str(item.get("entry_id", "")),
                    entry_type=item.get("entry_type", "genesis"),
                    description=str(item.get("description", "")),
                    state_hash=str(item.get("state_hash", "")),
                    constitution_hash=str(item.get("constitution_hash", "")),
                    posterior=float(item.get("posterior", 0.0)),
                    timestamp=str(item.get("timestamp", "")),
                )
            )
        except Exception:
            continue

    lineage = LineageState(
        lineage_id=str(data.get("lineage_id") or "genesis"),
        parent_constitution_hash=str(data.get("parent_constitution_hash") or ""),
        current_constitution_hash=str(data.get("current_constitution_hash") or ""),
        history=history,
        accepted_modifications=list(data.get("accepted_modifications") or []),
        inherited_certificates=list(data.get("inherited_certificates") or []),
        inherited_transport_operators=list(data.get("inherited_transport_operators") or []),
        forbidden_mutations=frozenset(data.get("forbidden_mutations") or []),
        rollback_ancestry=list(data.get("rollback_ancestry") or []),
        divergence_points=list(data.get("divergence_points") or []),
        inheritance_rules=tuple(rules or DEFAULT_INHERITANCE_RULES),
    )
    if not lineage.current_constitution_hash:
        constitution = OrganismConstitution()
        lineage.current_constitution_hash = constitution.constitution_hash()
        lineage.parent_constitution_hash = lineage.current_constitution_hash
    return lineage
