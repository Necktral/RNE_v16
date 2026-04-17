"""Estado constitucional del organismo RNFE.

El OrganismState es la entidad persistente que representa al organismo
por encima de cualquier episodio o escenario individual.  Cada episodio
ya no es una unidad autosuficiente sino una transición:

    (x_t, r_t, o_t, u_t) → x_{t+1}

donde x_t = OrganismState, r_t = régimen activo, o_t = observación,
u_t = política/intervención aplicada.

Sub-estados:
  - BeliefState:       creencias internas del organismo
  - PolicyState:       política de control vigente
  - IdentityState:     firma de identidad e invariantes
  - ViabilityState:    distancia al borde de inviabilidad
  - ModificationState: propuestas de cambio pendientes
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, FrozenSet, List, Literal, Sequence, Tuple


# ── Sub-state: Belief ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OrganismBeliefState:
    """Creencias internas del organismo sobre su régimen actual.

    Attributes:
        alarm_probability: P(alarma) ∈ [0,1].
        intervention_efficacy: Confianza en eficacia de la intervención actual.
        causal_support_confidence: Confianza en el soporte causal observado.
        memory_purity_estimate: Pureza estimada de la memoria activa.
        trace_integrity_confidence: Confianza en la integridad de la traza.
        regime_uncertainty: Incertidumbre total sobre el régimen actual.
    """

    alarm_probability: float = 0.1
    intervention_efficacy: float = 0.5
    causal_support_confidence: float = 0.5
    memory_purity_estimate: float = 1.0
    trace_integrity_confidence: float = 0.8
    regime_uncertainty: float = 0.3

    @property
    def composite_confidence(self) -> float:
        """Confianza compuesta [0, 1]."""
        return min(1.0, max(0.0, (
            0.20 * self.intervention_efficacy
            + 0.25 * self.causal_support_confidence
            + 0.15 * self.memory_purity_estimate
            + 0.20 * self.trace_integrity_confidence
            + 0.20 * (1.0 - self.alarm_probability)
        )))

    def distance_to(self, other: OrganismBeliefState) -> float:
        """L1 normalizada entre dos belief states."""
        components = [
            abs(self.alarm_probability - other.alarm_probability),
            abs(self.intervention_efficacy - other.intervention_efficacy),
            abs(self.causal_support_confidence - other.causal_support_confidence),
            abs(self.memory_purity_estimate - other.memory_purity_estimate),
            abs(self.trace_integrity_confidence - other.trace_integrity_confidence),
            abs(self.regime_uncertainty - other.regime_uncertainty),
        ]
        return sum(components) / len(components)


# ── Sub-state: Policy ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PolicyState:
    """Estado de la política de control vigente.

    Attributes:
        control_class: Clase de control activa ('reactive', 'adaptive', 'anticipatory').
        sensitivity: Sensibilidad a perturbaciones [0, 1].
        perturbation_tolerance: Tolerancia a perturbación antes de cambiar política.
        recovery_capacity: Capacidad de recuperación estimada [0, 1].
        accumulated_drift: Drift acumulado de política desde baseline.
    """

    control_class: Literal["reactive", "adaptive", "anticipatory"] = "reactive"
    sensitivity: float = 0.5
    perturbation_tolerance: float = 0.3
    recovery_capacity: float = 0.8
    accumulated_drift: float = 0.0

    @property
    def stability_score(self) -> float:
        """Score de estabilidad de la política [0, 1]."""
        return max(0.0, min(1.0,
            0.40 * self.recovery_capacity
            + 0.30 * (1.0 - self.accumulated_drift)
            + 0.30 * self.perturbation_tolerance
        ))


# ── Sub-state: Identity ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class IdentityState:
    """Firma de identidad del organismo.

    Attributes:
        active_invariants: Conjunto de invariantes vigentes (por nombre).
        lineage_id: Identificador de lineage actual.
        constitution_hash: Hash del contenido constitucional vigente.
        baseline_anchor: Anchor de lineage al baseline.
        inheritable_memory_scope: Alcance de memoria heredable.
        min_continuity_threshold: Umbral mínimo de continuidad identitaria.
    """

    active_invariants: FrozenSet[str] = frozenset()
    lineage_id: str = "genesis"
    constitution_hash: str = ""
    baseline_anchor: str = "baseline_fixed"
    inheritable_memory_scope: Literal["local", "compatible", "analogical"] = "local"
    min_continuity_threshold: float = 0.60

    def identity_distance(self, other: IdentityState) -> float:
        """Distancia identitaria entre dos estados."""
        # Invariant drift
        if not self.active_invariants and not other.active_invariants:
            inv_sim = 1.0
        elif not self.active_invariants or not other.active_invariants:
            inv_sim = 0.0
        else:
            union = self.active_invariants | other.active_invariants
            inter = self.active_invariants & other.active_invariants
            inv_sim = len(inter) / len(union)

        hash_same = 1.0 if self.constitution_hash == other.constitution_hash else 0.0
        anchor_same = 1.0 if self.baseline_anchor == other.baseline_anchor else 0.0
        scope_same = 1.0 if self.inheritable_memory_scope == other.inheritable_memory_scope else 0.5

        return max(0.0, 1.0 - (
            0.30 * inv_sim
            + 0.30 * hash_same
            + 0.20 * anchor_same
            + 0.20 * scope_same
        ))


# ── Sub-state: Viability ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ViabilityState:
    """Qué tan cerca está el organismo del borde de inviabilidad.

    Attributes:
        viability_margin: Margen de viabilidad [0, 1].  0 = al borde.
        reserve_stability: Estabilidad de reserva [0, 1].
        accumulated_degradation: Degradación acumulada [0, 1].
        rollback_readiness: Si el organismo puede hacer rollback.
        recovery_debt: Deuda de recuperación pendiente [0, 1].
    """

    viability_margin: float = 1.0
    reserve_stability: float = 0.8
    accumulated_degradation: float = 0.0
    rollback_readiness: bool = True
    recovery_debt: float = 0.0

    @property
    def is_viable(self) -> bool:
        """True si el organismo está en región viable."""
        return self.viability_margin > 0.0 and self.accumulated_degradation < 1.0

    @property
    def distance_to_edge(self) -> float:
        """Distancia al borde de inviabilidad [0, 1]."""
        return max(0.0, self.viability_margin * (1.0 - self.accumulated_degradation))


# ── Sub-state: Modification ──────────────────────────────────────────────────

@dataclass(frozen=True)
class ModificationProposal:
    """Propuesta de auto-modificación.

    Attributes:
        proposal_id: Identificador único.
        target: Qué componente se modifica.
        description: Descripción del cambio.
        risk_posterior: Riesgo estimado de la modificación.
        sandbox_verdict: Resultado de sandbox (None si no evaluado).
        rollback_plan: Plan de rollback.
    """

    proposal_id: str = ""
    target: str = ""
    description: str = ""
    risk_posterior: float = 0.5
    sandbox_verdict: Literal["pending", "accepted", "quarantined", "rejected"] = "pending"
    rollback_plan: str = "revert_to_previous_state"


@dataclass(frozen=True)
class ModificationState:
    """Estado de propuestas de modificación pendientes.

    Attributes:
        active_proposals: Propuestas activas.
        pending_count: Número de propuestas pendientes.
        lineage_delta_pending: Si hay cambios de lineage pendientes.
    """

    active_proposals: Tuple[ModificationProposal, ...] = ()
    lineage_delta_pending: bool = False

    @property
    def pending_count(self) -> int:
        return sum(1 for p in self.active_proposals if p.sandbox_verdict == "pending")

    @property
    def has_accepted(self) -> bool:
        return any(p.sandbox_verdict == "accepted" for p in self.active_proposals)


# ── Full OrganismState ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class OrganismState:
    """Estado constitucional completo del organismo.

    Reúne los cinco sub-estados bajo una entidad única y persistente.
    Cada episodio produce una transición x_t → x_{t+1}.
    """

    state_id: str = ""
    timestamp: str = ""
    active_regime: str = "unknown"
    episode_count: int = 0
    belief: OrganismBeliefState = field(default_factory=OrganismBeliefState)
    policy: PolicyState = field(default_factory=PolicyState)
    identity: IdentityState = field(default_factory=IdentityState)
    viability: ViabilityState = field(default_factory=ViabilityState)
    modification: ModificationState = field(default_factory=ModificationState)

    @property
    def is_viable(self) -> bool:
        return self.viability.is_viable

    @property
    def composite_health(self) -> float:
        """Score de salud compuesto del organismo [0, 1]."""
        return min(1.0, max(0.0, (
            0.25 * self.belief.composite_confidence
            + 0.20 * self.policy.stability_score
            + 0.25 * self.viability.distance_to_edge
            + 0.15 * (1.0 - self.identity.identity_distance(IdentityState()))
            + 0.15 * (1.0 if not self.modification.lineage_delta_pending else 0.5)
        )))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert frozensets to sorted lists for JSON serialization
        if "identity" in d and "active_invariants" in d["identity"]:
            d["identity"]["active_invariants"] = sorted(d["identity"]["active_invariants"])
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OrganismState:
        """Reconstruye OrganismState desde diccionario."""
        if not data:
            return cls()
        return cls(
            state_id=data.get("state_id", ""),
            timestamp=data.get("timestamp", ""),
            active_regime=data.get("active_regime", "unknown"),
            episode_count=data.get("episode_count", 0),
            belief=OrganismBeliefState(**data["belief"]) if "belief" in data else OrganismBeliefState(),
            policy=PolicyState(**data["policy"]) if "policy" in data else PolicyState(),
            identity=IdentityState(
                active_invariants=frozenset(data.get("identity", {}).get("active_invariants", [])),
                lineage_id=data.get("identity", {}).get("lineage_id", "genesis"),
                constitution_hash=data.get("identity", {}).get("constitution_hash", ""),
                baseline_anchor=data.get("identity", {}).get("baseline_anchor", "baseline_fixed"),
                inheritable_memory_scope=data.get("identity", {}).get("inheritable_memory_scope", "local"),
                min_continuity_threshold=data.get("identity", {}).get("min_continuity_threshold", 0.60),
            ) if "identity" in data else IdentityState(),
            viability=ViabilityState(**data["viability"]) if "viability" in data else ViabilityState(),
            modification=ModificationState(
                active_proposals=tuple(
                    ModificationProposal(**p)
                    for p in data.get("modification", {}).get("active_proposals", [])
                ),
                lineage_delta_pending=data.get("modification", {}).get("lineage_delta_pending", False),
            ) if "modification" in data else ModificationState(),
        )


# ── Transition builder ───────────────────────────────────────────────────────

def transition_organism_state(
    *,
    current: OrganismState,
    episode_result: Dict[str, Any],
    regime: str = "unknown",
    new_state_id: str = "",
    timestamp: str = "",
) -> OrganismState:
    """Transiciona el estado del organismo tras un episodio.

    Actualiza belief, policy y viability a partir de los resultados
    del episodio, manteniendo identity y modification sin cambios
    (esos se actualizan por canales separados).

    Args:
        current: Estado actual del organismo.
        episode_result: Resultado completo del episodio.
        regime: Régimen activo.
        new_state_id: ID del nuevo estado.
        timestamp: Timestamp de la transición.

    Returns:
        Nuevo OrganismState.
    """
    episode = episode_result.get("episode", {})
    result = episode.get("result", {})
    context = episode.get("context", {})
    observation = context.get("observation", {})
    cert = episode_result.get("certification", {})

    # Update belief
    alarm = observation.get("alarm", False)
    relation_kind = result.get("relation_kind", "unknown")
    belief_data = episode_result.get("belief_state", {})
    posterior_data = belief_data.get("posterior", {}) if belief_data else {}

    new_belief = OrganismBeliefState(
        alarm_probability=0.9 if alarm else 0.1,
        intervention_efficacy=float(posterior_data.get("policy_confidence", current.belief.intervention_efficacy)),
        causal_support_confidence=float(posterior_data.get(
            "causal_support_confidence",
            0.9 if relation_kind == "support" else (0.2 if relation_kind == "contradiction" else 0.5),
        )),
        memory_purity_estimate=float(posterior_data.get("memory_purity_confidence", current.belief.memory_purity_estimate)),
        trace_integrity_confidence=float(posterior_data.get("trace_confidence", current.belief.trace_integrity_confidence)),
        regime_uncertainty=max(0.0, current.belief.regime_uncertainty * 0.95),  # Slight decay
    )

    # Update policy
    drift_delta = abs(new_belief.intervention_efficacy - current.belief.intervention_efficacy)
    new_drift = min(1.0, current.policy.accumulated_drift + drift_delta * 0.1)
    new_policy = PolicyState(
        control_class=current.policy.control_class,
        sensitivity=current.policy.sensitivity,
        perturbation_tolerance=current.policy.perturbation_tolerance,
        recovery_capacity=max(0.0, current.policy.recovery_capacity - drift_delta * 0.05),
        accumulated_drift=round(new_drift, 4),
    )

    # Update viability
    is_certified = cert.get("verdict") == "certified"
    margin_delta = 0.01 if is_certified else -0.03
    deg_delta = 0.0 if is_certified else 0.02
    new_viability = ViabilityState(
        viability_margin=max(0.0, min(1.0, current.viability.viability_margin + margin_delta)),
        reserve_stability=max(0.0, min(1.0,
            current.viability.reserve_stability + (0.01 if is_certified else -0.02),
        )),
        accumulated_degradation=max(0.0, min(1.0,
            current.viability.accumulated_degradation + deg_delta,
        )),
        rollback_readiness=current.viability.rollback_readiness,
        recovery_debt=max(0.0, current.viability.recovery_debt + (
            -0.01 if is_certified else 0.02
        )),
    )

    return OrganismState(
        state_id=new_state_id,
        timestamp=timestamp,
        active_regime=regime,
        episode_count=current.episode_count + 1,
        belief=new_belief,
        policy=new_policy,
        identity=current.identity,
        viability=new_viability,
        modification=current.modification,
    )
