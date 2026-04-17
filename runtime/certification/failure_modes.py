"""Taxonomía de modos de fallo de transferencia inter-escenario.

Define los modos de fallo que pueden ocurrir durante transferencia:
- memory_contamination: memoria cross-scenario contamina decisiones
- causal_inversion: la polaridad causal se invierte sin compensación
- policy_drift: la política de intervención se desvía sin soporte
- belief_collapse: el estado de creencia pierde coherencia
- trace_discontinuity: la traza de razonamiento se rompe
- morphism_failure: el morfismo dirigido no alcanza umbral mínimo

Cada modo tiene severidad, evidencia requerida y mitigación sugerida.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Sequence

FailureSeverity = Literal["critical", "high", "medium", "low"]

FailureModeName = Literal[
    "memory_contamination",
    "causal_inversion",
    "policy_drift",
    "belief_collapse",
    "trace_discontinuity",
    "morphism_failure",
]


@dataclass(frozen=True)
class TransferFailureMode:
    """Modo de fallo detectado en una transferencia."""

    mode: FailureModeName
    severity: FailureSeverity
    evidence_score: float       # [0, 1] strength of evidence for this failure
    description: str
    mitigation: str


@dataclass(frozen=True)
class FailureModeAssessment:
    """Evaluación completa de modos de fallo para una transferencia."""

    detected_modes: tuple[TransferFailureMode, ...]
    total_risk: float           # Aggregate risk [0, 1]
    critical_count: int
    high_count: int
    has_blocking_failure: bool   # Any critical mode → blocks transfer


# ── Severity weights for risk aggregation ────────────────────────────────────

_SEVERITY_WEIGHTS = {
    "critical": 1.0,
    "high": 0.6,
    "medium": 0.3,
    "low": 0.1,
}


# ── Detection functions ──────────────────────────────────────────────────────

def detect_failure_modes(
    *,
    memory_purity: float,
    morphism_score: float,
    belief_shift_kl: float,
    policy_confidence: float,
    causal_support: float,
    trace_integrity: bool,
    polarity_inversion: bool,
) -> FailureModeAssessment:
    """Detecta modos de fallo de transferencia a partir de las evidencias.

    Args:
        memory_purity: Pureza de memoria [0, 1].
        morphism_score: Score del morfismo dirigido [0, 1].
        belief_shift_kl: KL divergence approx del belief shift [0, 1].
        policy_confidence: Confianza en la política actual [0, 1].
        causal_support: Confianza en soporte causal [0, 1].
        trace_integrity: Si la traza de razonamiento es íntegra.
        polarity_inversion: Si hay inversión de polaridad causal.

    Returns:
        FailureModeAssessment con modos detectados y riesgo total.
    """
    modes: list[TransferFailureMode] = []

    # Memory contamination
    if memory_purity < 0.70:
        severity: FailureSeverity = "critical" if memory_purity < 0.40 else "high"
        modes.append(TransferFailureMode(
            mode="memory_contamination",
            severity=severity,
            evidence_score=round(1.0 - memory_purity, 4),
            description=f"Memory purity {memory_purity:.2f} below safety threshold",
            mitigation="Restrict to strict_same_scenario mode or flush cross-scenario cache",
        ))

    # Causal inversion
    if polarity_inversion and causal_support < 0.60:
        severity = "critical" if causal_support < 0.30 else "high"
        modes.append(TransferFailureMode(
            mode="causal_inversion",
            severity=severity,
            evidence_score=round(1.0 - causal_support, 4),
            description="Causal polarity inverted without compensating support",
            mitigation="Apply polarity correction in transport operator or block transfer",
        ))

    # Policy drift
    if policy_confidence < 0.50:
        severity = "high" if policy_confidence < 0.30 else "medium"
        modes.append(TransferFailureMode(
            mode="policy_drift",
            severity=severity,
            evidence_score=round(1.0 - policy_confidence, 4),
            description=f"Policy confidence {policy_confidence:.2f} indicates drift",
            mitigation="Reset to default policy for target scenario",
        ))

    # Belief collapse
    if belief_shift_kl > 0.40:
        severity = "critical" if belief_shift_kl > 0.60 else "high"
        modes.append(TransferFailureMode(
            mode="belief_collapse",
            severity=severity,
            evidence_score=round(belief_shift_kl, 4),
            description=f"Belief shift KL={belief_shift_kl:.3f} indicates potential collapse",
            mitigation="Increase warmup episodes in target scenario before trusting beliefs",
        ))

    # Trace discontinuity
    if not trace_integrity:
        modes.append(TransferFailureMode(
            mode="trace_discontinuity",
            severity="high",
            evidence_score=1.0,
            description="Reasoning trace integrity broken during transition",
            mitigation="Re-derive trace in target scenario context",
        ))

    # Morphism failure
    if morphism_score < 0.35:
        severity = "critical" if morphism_score < 0.15 else "medium"
        modes.append(TransferFailureMode(
            mode="morphism_failure",
            severity=severity,
            evidence_score=round(1.0 - morphism_score, 4),
            description=f"Morphism score {morphism_score:.3f} below transfer threshold",
            mitigation="Restrict to local certification only",
        ))

    # Aggregate risk
    total_risk = 0.0
    for m in modes:
        total_risk += _SEVERITY_WEIGHTS[m.severity] * m.evidence_score
    total_risk = min(1.0, total_risk / max(len(modes), 1)) if modes else 0.0

    critical_count = sum(1 for m in modes if m.severity == "critical")
    high_count = sum(1 for m in modes if m.severity == "high")
    has_blocking = critical_count > 0

    return FailureModeAssessment(
        detected_modes=tuple(modes),
        total_risk=round(total_risk, 4),
        critical_count=critical_count,
        high_count=high_count,
        has_blocking_failure=has_blocking,
    )
