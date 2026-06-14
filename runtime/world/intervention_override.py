"""Override de intervención determinista guardado (actuación del razonamiento).

En el camino nominal la intervención la elige el núcleo greedy
(``scenario.select_intervention``) y se aplica ANTES de que las familias razonen:
el razonamiento es advisory. En el régimen de conflicto causal-contrafactual el
greedy elige mal (p.ej. ``deactivate_cooling`` con alarma activa) mientras el
contrafactual (``activate_cooling``, ya simulado) triunfa — pero las familias no
pueden corregir la acción ya ejecutada.

Este módulo da a las familias DETERMINISTAS el poder de actuar, como hace el
razonador LLM externo pero en milisegundos y sin red: si (a) hay conflicto
estructural, (b) una familia activa recomienda una intervención concreta distinta
de la greedy, y (c) una GUARDA certifica que la alterna es genuinamente mejor en
la dirección de optimización (no regresa), se adopta la alterna.

Honestidad por construcción:
- Gated por ``RNFE_REASONING_ACTUATES=1`` (sombra OFF por defecto ⇒ camino nominal
  byte-idéntico).
- La guarda usa el RESULTADO SIMULADO (el contrafactual ya calculado): no puede
  disparar salvo que la alterna sea de verdad mejor. No se fabrica ganancia.
- Atribución: sin familia que recomiende la alterna, no hay override (el baseline
  greedy falla de verdad).

Python puro, determinista, sin dependencias.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple


def is_actuation_enabled() -> bool:
    return os.environ.get("RNFE_REASONING_ACTUATES", "0").strip() == "1"


# Familias que producen una intervención CONCRETA (no solo señal de conflicto).
# Orden de prioridad como "driver" del override.
_CONCRETE_RECOMMENDERS: Tuple[Tuple[str, str], ...] = (
    ("opt", "opt_intervention"),
    ("plan", "plan_first_action"),
    ("ind", "ind_best_intervention"),
)


@dataclass(frozen=True)
class OverrideDecision:
    fired: bool
    driver_family: Optional[str] = None
    from_intervention: Optional[str] = None
    to_intervention: Optional[str] = None
    guard_reason: str = ""
    margin_gain: float = 0.0
    conflict: bool = False

    def to_dict(self) -> dict:
        return {
            "fired": self.fired,
            "driver_family": self.driver_family,
            "from_intervention": self.from_intervention,
            "to_intervention": self.to_intervention,
            "guard_reason": self.guard_reason,
            "margin_gain": round(float(self.margin_gain), 6),
            "conflict": self.conflict,
        }


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def detect_structural_conflict(reasoning_state: Mapping[str, Any]) -> bool:
    """Conflicto estructural: el contrafactual o la causa contradicen la acción."""
    ctf = reasoning_state.get("ctf_checked")
    if isinstance(ctf, Mapping) and ctf.get("agreement_with_relation_kind") is False:
        return True
    cau = reasoning_state.get("cau_link")
    if isinstance(cau, Mapping) and cau.get("helps_goal") is False:
        return True
    return False


def family_recommendations(
    reasoning_state: Mapping[str, Any],
    allowed_interventions: Sequence[str],
) -> List[Tuple[str, str]]:
    """(familia, intervención) recomendadas, normalizadas a allowed, en prioridad."""
    allowed = {_norm(i): i for i in allowed_interventions}
    out: List[Tuple[str, str]] = []
    for family, key in _CONCRETE_RECOMMENDERS:
        rec = reasoning_state.get(key)
        norm = _norm(rec)
        if norm and norm in allowed:
            out.append((family, allowed[norm]))
    return out


def _safety_margin(*, value: float, direction: str) -> float:
    """Margen en la dirección de optimización (mayor = mejor)."""
    # minimize ⇒ menor valor es mejor ⇒ margen = −valor; maximize ⇒ margen = valor.
    return -float(value) if direction == "minimize" else float(value)


def guard_candidate(
    *,
    direction: str,
    factual_value: float,
    candidate_value: float,
    epsilon: float = 1e-6,
) -> Tuple[bool, str, float]:
    """Adopta la alterna SOLO si mejora estrictamente en la dirección de optimización."""
    m_factual = _safety_margin(value=factual_value, direction=direction)
    m_candidate = _safety_margin(value=candidate_value, direction=direction)
    gain = m_candidate - m_factual
    if gain <= epsilon:
        return False, "no_improvement", gain
    return True, "guard_passed", gain


def evaluate_override(
    *,
    reasoning_state: Mapping[str, Any],
    allowed_interventions: Sequence[str],
    greedy_intervention: str,
    direction: str,
    factual_value: float,
    simulate_value: Callable[[str], float],
) -> OverrideDecision:
    """Decide si la deliberación debe sobre-escribir la acción greedy.

    El candidato es la PRIMERA intervención concreta recomendada por una familia
    activa que difiere de la greedy. ``simulate_value(intervention) -> float``
    devuelve el valor de la variable principal bajo esa intervención (el runner
    la implementa con ``scenario.factual_transition``). La guarda solo adopta la
    alterna si mejora estrictamente en la dirección de optimización.
    """
    if not detect_structural_conflict(reasoning_state):
        return OverrideDecision(fired=False, guard_reason="no_conflict", conflict=False)

    recs = family_recommendations(reasoning_state, allowed_interventions)
    pick = next(
        ((fam, iv) for fam, iv in recs if _norm(iv) != _norm(greedy_intervention)),
        None,
    )
    if pick is None:
        return OverrideDecision(
            fired=False, guard_reason="no_family_recommendation", conflict=True
        )

    driver, candidate = pick
    candidate_value = simulate_value(candidate)
    ok, reason, gain = guard_candidate(
        direction=direction,
        factual_value=factual_value,
        candidate_value=candidate_value,
    )
    return OverrideDecision(
        fired=ok,
        driver_family=driver,
        from_intervention=greedy_intervention,
        to_intervention=candidate,
        guard_reason=reason,
        margin_gain=gain,
        conflict=True,
    )
