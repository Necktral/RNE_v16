"""Recompensa semi-Markov del razonamiento (canon f2.4 §8).

El canon define la recompensa de una opción de razonamiento como

    r_t = ΔIoC*_t − λ_E·ΔE_t − λ_D·D_t − λ_B·B_safe_t

Aquí materializamos la versión disponible hoy, reutilizando R1 (el certificado
ampliado 𝔠ₜ⁺ ya expone ΔIoC y B_safe) y el coste de cómputo del scheduler como
proxy de energía ΔE:

    r_t = ΔIoC − λ_E·(coste_razonamiento / presupuesto) − λ_B·penalización_B_safe

La disipación física D_t (RQA/telemetría) queda para R4; su peso es 0 por ahora.
Esta recompensa es el escalar de control que un scheduler semi-Markov necesita
para valorar opciones; en R3a se computa, se persiste y se adjunta al episodio
(no cambia aún la selección, que sigue siendo la política determinista por
régimen — eso es R3b). Python puro, sin dependencias.

Pesos por defecto sobre-escribibles por entorno: ``RNFE_REWARD_LAMBDA_ENERGY``,
``RNFE_REWARD_LAMBDA_BSAFE``, ``RNFE_REWARD_LAMBDA_EFFECTIVENESS``, ``RNFE_REWARD_LAMBDA_NU``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _env_float(name: str, default: float) -> float:
    try:
        raw = os.environ.get(name)
        return float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def reasoning_cost_from_trace(trace: Sequence[Mapping[str, Any]]) -> float:
    """Suma del coste por paso del trace de razonamiento (ΔE proxy)."""
    total = 0.0
    for step in trace or ():
        detail = step.get("detail") if isinstance(step, Mapping) else None
        cost = (detail or {}).get("cost") if isinstance(detail, Mapping) else None
        if isinstance(cost, (int, float)):
            total += float(cost)
        else:
            total += 1.0
    return total


def _nu_value(nu: Optional[Any]) -> float:
    """ν ∈ [0,1]: bool `helps_goal` (True→1, False→0) o float clampeado. None→0."""
    if nu is True:
        return 1.0
    if nu is False or nu is None:
        return 0.0
    if isinstance(nu, (int, float)):
        return max(0.0, min(1.0, float(nu)))
    return 0.0


def _b_safe_penalty(b_safe: Optional[Mapping[str, Any]], lambda_bsafe: float) -> float:
    if not b_safe:
        return 0.0
    if b_safe.get("violated"):
        return lambda_bsafe  # barrera violada ⇒ penalización plena
    value = b_safe.get("value")
    if isinstance(value, (int, float)):
        # φ_bar crece al acercarse al límite; normalización suave a ~[0,1].
        return lambda_bsafe * min(1.0, max(0.0, float(value)) / 10.0)
    return 0.0


def compute_episode_reward(
    *,
    delta_ioc: Optional[float],
    reasoning_cost: float,
    cost_budget: Optional[float],
    b_safe: Optional[Mapping[str, Any]] = None,
    lambda_energy: Optional[float] = None,
    lambda_bsafe: Optional[float] = None,
    delta_ioc_star: Optional[float] = None,
    effectiveness: Optional[float] = None,
    lambda_effectiveness: Optional[float] = None,
    nu: Optional[Any] = None,
    lambda_nu: Optional[float] = None,
) -> Dict[str, Any]:
    """r = ΔIoC* − λ_E·(coste/presupuesto) − λ_B·B_safe + λ_V·efectividad + λ_ν·ν (D_t=0).

    El canon define la recompensa sobre ΔIoC* (coherencia/cierre). Pero la
    campaña de conflicto reveló que ΔIoC* es CIEGO a la efectividad del mundo:
    una familia que resuelve el conflicto (IVC-R ×9.3) tiene Δr<0 porque solo
    suma coste sin cambiar el cierre. El término de efectividad ``λ_V·efectividad``
    cierra esa ceguera: ``efectividad`` ∈ [−1,1] es el margen de seguridad del
    resultado factual en la dirección de optimización (positivo = la acción logró
    el objetivo). Desactivado por defecto (λ_V=0 ⇒ recompensa byte-idéntica);
    opt-in vía ``RNFE_REWARD_LAMBDA_EFFECTIVENESS``.

    El término de viabilidad ``λ_ν·ν`` es la CURA probada por el estudio del
    funcional crítico J(h|X): tratar ν = ``cau.helps_goal`` ∈ {0,1} (¿la acción
    factual va en la dirección del objetivo?) como criterio ADITIVO de primera
    clase recupera la familia efectiva con λ_ν≈1.0 — frente a λ_V≈20 que exige la
    efectividad sobre el IoC colapsado (inflación 40× por la anti-correlación
    continuidad↔desviación). ``nu`` acepta bool (helps_goal) o float ∈ [0,1].
    Desactivado por defecto (λ_ν=0 ⇒ byte-idéntico); opt-in ``RNFE_REWARD_LAMBDA_NU``.
    """
    lam_e = _env_float("RNFE_REWARD_LAMBDA_ENERGY", 0.10) if lambda_energy is None else lambda_energy
    lam_b = _env_float("RNFE_REWARD_LAMBDA_BSAFE", 0.50) if lambda_bsafe is None else lambda_bsafe
    lam_v = (
        _env_float("RNFE_REWARD_LAMBDA_EFFECTIVENESS", 0.0)
        if lambda_effectiveness is None
        else lambda_effectiveness
    )
    lam_nu = _env_float("RNFE_REWARD_LAMBDA_NU", 0.0) if lambda_nu is None else lambda_nu

    if isinstance(delta_ioc_star, (int, float)):
        d, delta_used = float(delta_ioc_star), "delta_ioc_star"
    elif isinstance(delta_ioc, (int, float)):
        d, delta_used = float(delta_ioc), "delta_ioc"
    else:
        d, delta_used = 0.0, "none"
    budget = max(1.0, float(cost_budget or 1.0))
    energy_term = lam_e * min(1.0, max(0.0, float(reasoning_cost)) / budget)
    bsafe_penalty = _b_safe_penalty(b_safe, lam_b)
    eff = 0.0 if not isinstance(effectiveness, (int, float)) else max(-1.0, min(1.0, float(effectiveness)))
    effectiveness_term = lam_v * eff
    nu_val = _nu_value(nu)
    nu_term = lam_nu * nu_val
    reward = d - energy_term - bsafe_penalty + effectiveness_term + nu_term
    return {
        "schema": "reasoning_reward.v3",
        "reward": round(reward, 6),
        "delta_ioc": None if delta_ioc is None else round(float(delta_ioc), 6),
        "delta_ioc_star": None if delta_ioc_star is None else round(float(delta_ioc_star), 6),
        "delta_used": delta_used,
        "energy_term": round(energy_term, 6),
        "bsafe_penalty": round(bsafe_penalty, 6),
        "effectiveness": None if effectiveness is None else round(eff, 6),
        "effectiveness_term": round(effectiveness_term, 6),
        "nu": None if nu is None else round(nu_val, 6),
        "nu_term": round(nu_term, 6),
        "reasoning_cost": round(float(reasoning_cost), 4),
        "cost_budget": round(budget, 4),
        "lambda_energy": lam_e,
        "lambda_bsafe": lam_b,
        "lambda_effectiveness": lam_v,
        "lambda_nu": lam_nu,
        "dissipation_term": 0.0,  # D_t (RQA/telemetría) — R4
    }


def summarize_rewards(rewards: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Resumen de una serie de recompensas (para análisis/option-valuation futura)."""
    values: List[float] = [
        float(r["reward"]) for r in rewards if isinstance(r, Mapping) and "reward" in r
    ]
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "cumulative": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": n,
        "mean": round(sum(values) / n, 6),
        "cumulative": round(sum(values), 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }
