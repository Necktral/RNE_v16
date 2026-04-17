"""Constitución operativa del organismo RNFE.

Define la ley viva del organismo en código.  Contiene:
- Invariantes duros: no pueden violarse sin cuarentena o rollback
- Invariantes blandos: pueden relajarse bajo sandbox
- Reglas de mutación: qué puede cambiar el organismo
- Reglas de no-mutación: qué requiere aprobación superior

La constitución convierte a PromotionGate en una parte de la corte,
no en toda la corte.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Literal, Tuple

from .state import OrganismState


# ── Invariant types ──────────────────────────────────────────────────────────

InvariantSeverity = Literal["hard", "soft"]

@dataclass(frozen=True)
class ConstitutionalInvariant:
    """Invariante constitucional del organismo.

    Attributes:
        name: Nombre único del invariante.
        severity: 'hard' (cuarentena/rollback si violado) o 'soft' (relajable en sandbox).
        description: Descripción del invariante.
        check_fn_name: Nombre de la función de chequeo.
    """

    name: str
    severity: InvariantSeverity
    description: str
    check_fn_name: str = ""


@dataclass(frozen=True)
class InvariantViolation:
    """Violación de un invariante detectada."""

    invariant_name: str
    severity: InvariantSeverity
    evidence_value: float
    threshold: float
    description: str


@dataclass(frozen=True)
class ConstitutionalValidation:
    """Resultado de validación constitucional.

    Attributes:
        is_valid: True si no hay violaciones hard.
        verdict: 'valid', 'quarantine', 'rollback'.
        violations: Lista de violaciones detectadas.
        hard_violation_count: Número de violaciones hard.
        soft_violation_count: Número de violaciones soft.
        margin_to_threshold: Margen mínimo sobre cualquier invariante hard.
    """

    is_valid: bool
    verdict: Literal["valid", "quarantine", "rollback"]
    violations: Tuple[InvariantViolation, ...]
    hard_violation_count: int
    soft_violation_count: int
    margin_to_threshold: float


# ── Hard invariant checks ────────────────────────────────────────────────────

def _check_triadic_closure(state: OrganismState, config: Dict[str, float]) -> InvariantViolation | None:
    """Cierre triádico válido: causal support + trace integrity + memory purity."""
    threshold = config.get("triadic_closure_threshold", 0.50)
    value = (
        state.belief.causal_support_confidence
        * state.belief.trace_integrity_confidence
        * state.belief.memory_purity_estimate
    )
    if value < threshold:
        return InvariantViolation(
            invariant_name="triadic_closure",
            severity="hard",
            evidence_value=round(value, 4),
            threshold=threshold,
            description=f"Triadic closure product {value:.4f} < {threshold}",
        )
    return None


def _check_memory_purity(state: OrganismState, config: Dict[str, float]) -> InvariantViolation | None:
    threshold = config.get("min_memory_purity", 0.40)
    if state.belief.memory_purity_estimate < threshold:
        return InvariantViolation(
            invariant_name="min_memory_purity",
            severity="hard",
            evidence_value=round(state.belief.memory_purity_estimate, 4),
            threshold=threshold,
            description=f"Memory purity {state.belief.memory_purity_estimate:.4f} < {threshold}",
        )
    return None


def _check_trace_integrity(state: OrganismState, config: Dict[str, float]) -> InvariantViolation | None:
    threshold = config.get("min_trace_integrity", 0.30)
    if state.belief.trace_integrity_confidence < threshold:
        return InvariantViolation(
            invariant_name="min_trace_integrity",
            severity="hard",
            evidence_value=round(state.belief.trace_integrity_confidence, 4),
            threshold=threshold,
            description=f"Trace integrity {state.belief.trace_integrity_confidence:.4f} < {threshold}",
        )
    return None


def _check_baseline_integrity(state: OrganismState, config: Dict[str, float]) -> InvariantViolation | None:
    """Baseline no degradado."""
    max_degradation = config.get("max_degradation", 0.80)
    if state.viability.accumulated_degradation >= max_degradation:
        return InvariantViolation(
            invariant_name="baseline_not_degraded",
            severity="hard",
            evidence_value=round(state.viability.accumulated_degradation, 4),
            threshold=max_degradation,
            description=f"Accumulated degradation {state.viability.accumulated_degradation:.4f} >= {max_degradation}",
        )
    return None


def _check_rollback_available(state: OrganismState, config: Dict[str, float]) -> InvariantViolation | None:
    if not state.viability.rollback_readiness:
        return InvariantViolation(
            invariant_name="rollback_available",
            severity="hard",
            evidence_value=0.0,
            threshold=1.0,
            description="Rollback not available",
        )
    return None


def _check_coherence(state: OrganismState, config: Dict[str, float]) -> InvariantViolation | None:
    """Coherencia factual/contrafactual: causal_support > threshold."""
    threshold = config.get("min_causal_support", 0.20)
    if state.belief.causal_support_confidence < threshold:
        return InvariantViolation(
            invariant_name="factual_counterfactual_coherence",
            severity="hard",
            evidence_value=round(state.belief.causal_support_confidence, 4),
            threshold=threshold,
            description=f"Causal support {state.belief.causal_support_confidence:.4f} < {threshold}",
        )
    return None


def _check_lineage_coherent(state: OrganismState, config: Dict[str, float]) -> InvariantViolation | None:
    if not state.identity.lineage_id:
        return InvariantViolation(
            invariant_name="lineage_coherent",
            severity="hard",
            evidence_value=0.0,
            threshold=1.0,
            description="Lineage ID is empty",
        )
    return None


# ── Soft invariant checks ────────────────────────────────────────────────────

def _check_policy_stability(state: OrganismState, config: Dict[str, float]) -> InvariantViolation | None:
    threshold = config.get("max_policy_drift", 0.50)
    if state.policy.accumulated_drift > threshold:
        return InvariantViolation(
            invariant_name="policy_stability",
            severity="soft",
            evidence_value=round(state.policy.accumulated_drift, 4),
            threshold=threshold,
            description=f"Policy drift {state.policy.accumulated_drift:.4f} > {threshold}",
        )
    return None


def _check_continuity_minimum(state: OrganismState, config: Dict[str, float]) -> InvariantViolation | None:
    threshold = config.get("min_continuity", 0.40)
    composite = state.belief.composite_confidence
    if composite < threshold:
        return InvariantViolation(
            invariant_name="continuity_minimum",
            severity="soft",
            evidence_value=round(composite, 4),
            threshold=threshold,
            description=f"Composite confidence {composite:.4f} < {threshold}",
        )
    return None


def _check_recovery_cost(state: OrganismState, config: Dict[str, float]) -> InvariantViolation | None:
    threshold = config.get("max_recovery_debt", 0.70)
    if state.viability.recovery_debt > threshold:
        return InvariantViolation(
            invariant_name="max_recovery_cost",
            severity="soft",
            evidence_value=round(state.viability.recovery_debt, 4),
            threshold=threshold,
            description=f"Recovery debt {state.viability.recovery_debt:.4f} > {threshold}",
        )
    return None


def _check_drift_tolerance(state: OrganismState, config: Dict[str, float]) -> InvariantViolation | None:
    threshold = config.get("max_drift_rate", 0.60)
    drift = state.policy.accumulated_drift
    if drift > threshold:
        return InvariantViolation(
            invariant_name="drift_tolerance",
            severity="soft",
            evidence_value=round(drift, 4),
            threshold=threshold,
            description=f"Drift rate {drift:.4f} > {threshold}",
        )
    return None


# ── Hard and soft invariant registries ───────────────────────────────────────

HARD_INVARIANTS: Tuple[ConstitutionalInvariant, ...] = (
    ConstitutionalInvariant("triadic_closure", "hard", "Triadic closure product >= threshold"),
    ConstitutionalInvariant("min_memory_purity", "hard", "Memory purity >= minimum"),
    ConstitutionalInvariant("min_trace_integrity", "hard", "Trace integrity >= minimum"),
    ConstitutionalInvariant("baseline_not_degraded", "hard", "Accumulated degradation < maximum"),
    ConstitutionalInvariant("rollback_available", "hard", "Rollback must be ready"),
    ConstitutionalInvariant("factual_counterfactual_coherence", "hard", "Causal coherence >= threshold"),
    ConstitutionalInvariant("lineage_coherent", "hard", "Lineage ID must be non-empty"),
)

SOFT_INVARIANTS: Tuple[ConstitutionalInvariant, ...] = (
    ConstitutionalInvariant("policy_stability", "soft", "Policy drift within tolerance"),
    ConstitutionalInvariant("continuity_minimum", "soft", "Composite confidence above minimum"),
    ConstitutionalInvariant("max_recovery_cost", "soft", "Recovery debt below maximum"),
    ConstitutionalInvariant("drift_tolerance", "soft", "Drift rate within tolerance"),
)

_HARD_CHECKS = [
    _check_triadic_closure,
    _check_memory_purity,
    _check_trace_integrity,
    _check_baseline_integrity,
    _check_rollback_available,
    _check_coherence,
    _check_lineage_coherent,
]

_SOFT_CHECKS = [
    _check_policy_stability,
    _check_continuity_minimum,
    _check_recovery_cost,
    _check_drift_tolerance,
]


# ── Mutation rules ───────────────────────────────────────────────────────────

MUTABLE_COMPONENTS: FrozenSet[str] = frozenset({
    "transport_parameters",
    "selection_policy",
    "benchmark_policy_experimental",
    "reasoning_activation_policy",
    "memory_scoring_secondary_weights",
    "analogical_lab_parameters",
})

IMMUTABLE_COMPONENTS: FrozenSet[str] = frozenset({
    "baseline_semantics",
    "constitutional_invariants",
    "baseline_fixed",
    "constitutional_purity_minimum",
    "lineage_identity_anchor",
})


# ── Constitution ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OrganismConstitution:
    """Constitución operativa del organismo.

    Define la ley viva del organismo: invariantes, reglas de mutación,
    y configuración de umbrales constitucionales.
    """

    hard_invariants: Tuple[ConstitutionalInvariant, ...] = HARD_INVARIANTS
    soft_invariants: Tuple[ConstitutionalInvariant, ...] = SOFT_INVARIANTS
    mutable_components: FrozenSet[str] = MUTABLE_COMPONENTS
    immutable_components: FrozenSet[str] = IMMUTABLE_COMPONENTS
    config: Dict[str, float] = field(default_factory=lambda: {
        "triadic_closure_threshold": 0.50,
        "min_memory_purity": 0.40,
        "min_trace_integrity": 0.30,
        "max_degradation": 0.80,
        "min_causal_support": 0.20,
        "max_policy_drift": 0.50,
        "min_continuity": 0.40,
        "max_recovery_debt": 0.70,
        "max_drift_rate": 0.60,
    })

    def validate(self, state: OrganismState) -> ConstitutionalValidation:
        """Valida el estado del organismo contra la constitución.

        Returns:
            ConstitutionalValidation con violaciones y verdict.
        """
        violations: List[InvariantViolation] = []

        for check in _HARD_CHECKS:
            v = check(state, self.config)
            if v is not None:
                violations.append(v)

        for check in _SOFT_CHECKS:
            v = check(state, self.config)
            if v is not None:
                violations.append(v)

        hard_count = sum(1 for v in violations if v.severity == "hard")
        soft_count = sum(1 for v in violations if v.severity == "soft")

        if hard_count > 0:
            verdict = "rollback" if hard_count >= 3 else "quarantine"
        else:
            verdict = "valid"

        # Margin to threshold: minimum distance any hard invariant is from its threshold
        margin = 1.0
        for check in _HARD_CHECKS:
            v = check(state, self.config)
            if v is None:
                continue
            # If violated, margin is negative
            margin = min(margin, v.threshold - v.evidence_value if v.severity == "hard" else margin)
        if not violations:
            margin = min(
                state.belief.memory_purity_estimate - self.config.get("min_memory_purity", 0.40),
                state.belief.trace_integrity_confidence - self.config.get("min_trace_integrity", 0.30),
                state.belief.causal_support_confidence - self.config.get("min_causal_support", 0.20),
                1.0 - state.viability.accumulated_degradation / max(self.config.get("max_degradation", 0.80), 0.01),
            )

        return ConstitutionalValidation(
            is_valid=hard_count == 0,
            verdict=verdict,
            violations=tuple(violations),
            hard_violation_count=hard_count,
            soft_violation_count=soft_count,
            margin_to_threshold=round(margin, 4),
        )

    def is_mutable(self, component: str) -> bool:
        """Verifica si un componente puede ser mutado."""
        return component in self.mutable_components

    def is_immutable(self, component: str) -> bool:
        """Verifica si un componente es inmutable."""
        return component in self.immutable_components

    def constitution_hash(self) -> str:
        """Hash SHA256 del contenido constitucional."""
        import hashlib
        import json
        blob = json.dumps({
            "hard": [i.name for i in self.hard_invariants],
            "soft": [i.name for i in self.soft_invariants],
            "mutable": sorted(self.mutable_components),
            "immutable": sorted(self.immutable_components),
            "config": {k: v for k, v in sorted(self.config.items())},
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]
