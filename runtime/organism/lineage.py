"""Lineage y herencia constitucional del organismo.

Convierte continuidad temporal en continuidad generacional controlada.

Solo se hereda:
  - lo certificado como seguro
  - lo consistente con la constitución vigente
  - lo que no degrada baseline
  - lo que no introduce contaminación estructural

Registra:
  - parent constitution hash
  - organism lineage id
  - accepted modifications
  - inherited certificates
  - inherited transport operators
  - forbidden mutations
  - rollback ancestry
  - divergence points
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Literal, Tuple

from .state import OrganismState, IdentityState
from .constitution import OrganismConstitution


@dataclass(frozen=True)
class LineageEntry:
    """Entrada en el registro de lineage.

    Attributes:
        entry_id: Identificador único.
        entry_type: Tipo de entrada.
        description: Descripción del evento.
        state_hash: Hash del estado al momento del evento.
        constitution_hash: Hash de la constitución vigente.
        posterior: Posterior constitucional del momento.
        timestamp: Timestamp del evento.
    """

    entry_id: str
    entry_type: Literal[
        "genesis",
        "modification_accepted",
        "modification_rejected",
        "transport_inherited",
        "rollback",
        "divergence",
        "constitution_update",
    ]
    description: str
    state_hash: str = ""
    constitution_hash: str = ""
    posterior: float = 0.0
    timestamp: str = ""


@dataclass(frozen=True)
class InheritanceRule:
    """Regla de herencia constitucional.

    Attributes:
        name: Nombre de la regla.
        condition: Condición que debe cumplirse para heredar.
        description: Descripción.
    """

    name: str
    condition: Literal[
        "certified_safe",
        "constitution_consistent",
        "baseline_preserved",
        "no_contamination",
    ]
    description: str


# Default inheritance rules
DEFAULT_INHERITANCE_RULES: Tuple[InheritanceRule, ...] = (
    InheritanceRule(
        "certified_safe",
        "certified_safe",
        "Only inherit what has been certified as safe by the constitutional posterior",
    ),
    InheritanceRule(
        "constitution_consistent",
        "constitution_consistent",
        "Only inherit what is consistent with the current constitution",
    ),
    InheritanceRule(
        "baseline_preserved",
        "baseline_preserved",
        "Only inherit what does not degrade the baseline",
    ),
    InheritanceRule(
        "no_contamination",
        "no_contamination",
        "Only inherit what does not introduce structural contamination",
    ),
)


@dataclass
class LineageState:
    """Estado de lineage del organismo.

    Attributes:
        lineage_id: Identificador único de lineage.
        parent_constitution_hash: Hash de la constitución padre.
        current_constitution_hash: Hash de la constitución actual.
        history: Historial de entradas de lineage.
        accepted_modifications: IDs de modificaciones aceptadas.
        inherited_certificates: IDs de certificados heredados.
        inherited_transport_operators: IDs de operadores de transporte heredados.
        forbidden_mutations: Componentes que no pueden mutar.
        rollback_ancestry: IDs de rollbacks en la historia.
        divergence_points: Puntos de divergencia.
        inheritance_rules: Reglas de herencia vigentes.
    """

    lineage_id: str = "genesis"
    parent_constitution_hash: str = ""
    current_constitution_hash: str = ""
    history: List[LineageEntry] = field(default_factory=list)
    accepted_modifications: List[str] = field(default_factory=list)
    inherited_certificates: List[str] = field(default_factory=list)
    inherited_transport_operators: List[str] = field(default_factory=list)
    forbidden_mutations: FrozenSet[str] = frozenset()
    rollback_ancestry: List[str] = field(default_factory=list)
    divergence_points: List[str] = field(default_factory=list)
    inheritance_rules: Tuple[InheritanceRule, ...] = DEFAULT_INHERITANCE_RULES

    @property
    def generation(self) -> int:
        """Número de generación (modificaciones aceptadas)."""
        return len(self.accepted_modifications)

    @property
    def has_diverged(self) -> bool:
        """True si ha divergido del lineage padre."""
        return len(self.divergence_points) > 0

    def record_genesis(self, constitution: OrganismConstitution, timestamp: str = "") -> None:
        """Registra el evento de genesis."""
        c_hash = constitution.constitution_hash()
        self.current_constitution_hash = c_hash
        self.parent_constitution_hash = c_hash
        self.history.append(LineageEntry(
            entry_id=f"gen-{self.lineage_id}",
            entry_type="genesis",
            description="Organism genesis",
            constitution_hash=c_hash,
            timestamp=timestamp,
        ))

    def record_modification(
        self,
        *,
        modification_id: str,
        description: str,
        posterior: float,
        state_hash: str = "",
        timestamp: str = "",
    ) -> None:
        """Registra una modificación aceptada."""
        self.accepted_modifications.append(modification_id)
        self.history.append(LineageEntry(
            entry_id=modification_id,
            entry_type="modification_accepted",
            description=description,
            state_hash=state_hash,
            posterior=posterior,
            constitution_hash=self.current_constitution_hash,
            timestamp=timestamp,
        ))

    def record_rollback(
        self,
        *,
        rollback_id: str,
        description: str,
        timestamp: str = "",
    ) -> None:
        """Registra un rollback."""
        self.rollback_ancestry.append(rollback_id)
        self.history.append(LineageEntry(
            entry_id=rollback_id,
            entry_type="rollback",
            description=description,
            constitution_hash=self.current_constitution_hash,
            timestamp=timestamp,
        ))

    def record_divergence(
        self,
        *,
        divergence_id: str,
        description: str,
        timestamp: str = "",
    ) -> None:
        """Registra un punto de divergencia."""
        self.divergence_points.append(divergence_id)
        self.history.append(LineageEntry(
            entry_id=divergence_id,
            entry_type="divergence",
            description=description,
            constitution_hash=self.current_constitution_hash,
            timestamp=timestamp,
        ))

    def check_inheritance_eligibility(
        self,
        *,
        is_certified_safe: bool,
        is_constitution_consistent: bool,
        is_baseline_preserved: bool,
        is_contamination_free: bool,
    ) -> Tuple[bool, List[str]]:
        """Verifica si algo es elegible para herencia.

        Returns:
            (eligible, failed_rules)
        """
        checks = {
            "certified_safe": is_certified_safe,
            "constitution_consistent": is_constitution_consistent,
            "baseline_preserved": is_baseline_preserved,
            "no_contamination": is_contamination_free,
        }

        failed = []
        for rule in self.inheritance_rules:
            if not checks.get(rule.condition, False):
                failed.append(rule.name)

        return len(failed) == 0, failed

    def consistency_score(self) -> float:
        """Score de consistencia del lineage [0, 1].

        Basado en ratio de modificaciones aceptadas vs rollbacks,
        y ausencia de divergencias.
        """
        total = len(self.accepted_modifications) + len(self.rollback_ancestry)
        if total == 0:
            return 1.0

        acceptance_ratio = len(self.accepted_modifications) / total
        divergence_penalty = min(0.3, len(self.divergence_points) * 0.10)
        rollback_penalty = min(0.3, len(self.rollback_ancestry) * 0.05)

        return max(0.0, min(1.0, acceptance_ratio - divergence_penalty - rollback_penalty))

    def to_identity_state(self, constitution: OrganismConstitution) -> IdentityState:
        """Convierte lineage en IdentityState para el OrganismState."""
        return IdentityState(
            active_invariants=frozenset(i.name for i in constitution.hard_invariants),
            lineage_id=self.lineage_id,
            constitution_hash=self.current_constitution_hash,
            baseline_anchor="baseline_fixed",
            inheritable_memory_scope="local",
            min_continuity_threshold=0.60,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "lineage_id": self.lineage_id,
            "parent_constitution_hash": self.parent_constitution_hash,
            "current_constitution_hash": self.current_constitution_hash,
            "generation": self.generation,
            "accepted_modifications": self.accepted_modifications,
            "inherited_certificates": self.inherited_certificates,
            "inherited_transport_operators": self.inherited_transport_operators,
            "forbidden_mutations": sorted(self.forbidden_mutations),
            "rollback_ancestry": self.rollback_ancestry,
            "divergence_points": self.divergence_points,
            "history_length": len(self.history),
            "consistency_score": self.consistency_score(),
        }
