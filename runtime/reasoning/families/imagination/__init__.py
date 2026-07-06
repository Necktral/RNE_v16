"""Familia IMAGINATION (A11) — previsión de consecuencia diferida.

OFF (byte-idéntico): idle.
DEEP (opt-in, RNFE_IMAGINATION_DEEP / RNFE_REASONING_DEEP): imagina hacia adelante
sobre un modelo de mundo **estado-dependiente** para detectar consecuencias diferidas
que las familias de effect-model LINEAL (PLAN/EVO/CTF) no ven.

Hallazgo de diseño (2026-07-06):
El effect-model lineal (Δ fijo por intervención) es *incapaz* de miopía/consecuencia
diferida — con Δ constantes la mejor acción a 1 y a H pasos coincide. La previsión sólo
aporta sobre un mundo con **estado**. Por eso A11 imagina sobre un world-model
*keyed por escenario*:
  - térmico (`activate_cooling`/`deactivate_cooling`): estado = (temp, cooling_active).
  - carga diferida (`boost_throughput`/`shed_load`): estado = (load, debt); la deuda
    acumulada rebota la carga hacia la alarma — la trampa que el Δ lineal no ve.

Política de rollout: **repetir la acción** H pasos ("si sostengo esto, ¿dónde termino?"),
correcta para ambas semánticas (modo persistente y acción por paso). Determinista.

La salida es SIEMPRE advisory (nunca canónica/vinculante). Mientras el modelo de deuda
sea una asunción declarada, se marca `imagination_speculative=True` y confianza baja.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from runtime.reasoning.families import core_inference as ci
from runtime.reasoning.families import _deep_common as dc

FAMILY_ID = "IMAGINATION"

_HORIZON = 20
_ASSUMED_DRIFT = 0.04       # entrada externa asumida (default del episode runner); calibrar en Fase 2+
_BOOST_DEBT = 0.08          # modelo de deuda de la imaginación para carga diferida (declarado)
_SHED_DEBT = 0.02


def _idle() -> Dict[str, Any]:
    return {"family": FAMILY_ID, "status": "idle", "state_delta": {}, "confidence": 0.0, "cost": 0.0}


def _idle_signal(mode: str) -> Dict[str, Any]:
    return {
        "family": FAMILY_ID,
        "status": "idle",
        "state_delta": {"imagination_active": False},
        "confidence": 0.2,
        "cost": 0.0,
        "failure_mode": mode,
        "recommended_next_family": "PROB",
    }


_RISK_CEILING = 0.80


def gate(state_delta: Dict[str, Any], *, checkpoint_healthy: bool, risk: float,
         risk_ceiling: float = _RISK_CEILING) -> Dict[str, Any]:
    """Ejecución gated (Fase 3): decide si la previsión de A11 puede SESGAR la intervención.

    La compuerta abre sólo si TODO se cumple (advisory→gated bajo garantías tipo R1):
      - A11 activo,
      - su recomendación discrepa de la elección reactiva,
      - predice un breach diferido para la elección reactiva,
      - hay checkpoint sano (estado restaurable),
      - el riesgo está por debajo del techo.

    Devuelve {override, intervention, reason}. Si no abre, override=False (sólo advisory).
    """
    def _no(reason):
        return {"override": False, "intervention": None, "reason": reason}

    if not state_delta.get("imagination_active"):
        return _no("imagination_inactive")
    if not state_delta.get("imagination_disagrees_with_choice"):
        return _no("no_disagreement")
    if state_delta.get("imagination_chosen_breaches_at") is None:
        return _no("no_predicted_breach")
    if not checkpoint_healthy:
        return _no("no_healthy_checkpoint")
    if risk >= risk_ceiling:
        return _no("risk_too_high")
    return {
        "override": True,
        "intervention": state_delta.get("imagination_recommended_intervention"),
        "reason": "gated_override",
    }


def imagine(
    *,
    interventions: Sequence[str],
    init: Callable[[], Any],
    step: Callable[[Any, str], Any],
    value: Callable[[Any], float],
    breached: Callable[[Any], bool],
    observe: Callable[[Any], float],
    chosen: Optional[str] = None,
    horizon: int = _HORIZON,
) -> Dict[str, Any]:
    """Núcleo puro de imaginación (determinista, agnóstico del mundo).

    Para cada intervención candidata simula "sostenerla" `horizon` pasos sobre un
    modelo de mundo con estado latente opaco. Devuelve la recomendada por previsión,
    el terminal por acción y —si se pasa `chosen`— si esa elección cruza el umbral.
    """
    def rollout(action: str):
        lat = init()
        traj: List[float] = []
        breach_at: Optional[int] = None
        for i in range(horizon):
            lat = step(lat, action)
            traj.append(round(observe(lat), 6))
            if breach_at is None and breached(lat):
                breach_at = i + 1
        return lat, traj, breach_at

    outcomes = {iv: rollout(iv) for iv in interventions}
    # recomendada = menor value(terminal); desempate estable por orden de `interventions`
    recommended = min(interventions, key=lambda iv: value(outcomes[iv][0]))

    result: Dict[str, Any] = {
        "recommended_intervention": recommended,
        "recommended_terminal": round(observe(outcomes[recommended][0]), 6),
        "per_action_terminal": {iv: round(observe(outcomes[iv][0]), 6) for iv in interventions},
    }
    if chosen in outcomes:
        clat, ctraj, cbreach = outcomes[chosen]
        result.update({
            "chosen_intervention": chosen,
            "chosen_terminal": round(observe(clat), 6),
            "chosen_breaches_at": cbreach,
            "chosen_trajectory": ctraj,
            "disagrees_with_choice": chosen != recommended,
        })
    return result


def thermal_world(*, x0: float, cooling: float, drift: float, threshold: Optional[float],
                  direction: str, cooling_active: bool = False) -> Dict[str, Any]:
    """Mundo térmico con estado (temp, cooling_active). Réplica de cgwm_min."""
    def init():
        return (x0, cooling_active)

    def step(lat, iv):
        t, ca = lat
        if iv == "activate_cooling":
            ca = True
        elif iv == "deactivate_cooling":
            ca = False
        return (dc.clamp(t + drift - (cooling if ca else 0.0)), ca)

    def value(lat):
        return lat[0] if direction == "minimize" else (1.0 - lat[0])

    def breached(lat):
        if threshold is None:
            return False
        return (lat[0] >= threshold) if direction == "minimize" else (lat[0] <= threshold)

    return dict(
        interventions=["activate_cooling", "deactivate_cooling"],
        init=init, step=step, value=value, breached=breached, observe=lambda lat: lat[0],
    )


def deferred_load_world(*, x0: float, debt: float, boost_effect: float, shed_effect: float,
                        drift: float, threshold: Optional[float], direction: str,
                        boost_debt: float = _BOOST_DEBT, shed_debt: float = _SHED_DEBT) -> Dict[str, Any]:
    """Mundo de carga diferida con estado (load, debt). Réplica de DeferredLoadScenario.

    La deuda acumulada empuja la carga hacia arriba cada paso: `boost` la inyecta
    (rebote diferido), `shed` la reduce (sostenible).
    """
    def init():
        return (x0, debt)

    def step(lat, iv):
        load, d = lat
        if iv == "boost_throughput":
            ld, dd = -boost_effect, +boost_debt
        elif iv == "shed_load":
            ld, dd = -shed_effect, -shed_debt
        else:
            ld, dd = 0.0, 0.0
        nd = dc.clamp(d + dd)
        return (dc.clamp(load + drift + ld + nd), nd)

    def value(lat):
        return lat[0] if direction == "minimize" else (1.0 - lat[0])

    def breached(lat):
        if threshold is None:
            return False
        return (lat[0] >= threshold) if direction == "minimize" else (lat[0] <= threshold)

    return dict(
        interventions=["boost_throughput", "shed_load"],
        init=init, step=step, value=value, breached=breached, observe=lambda lat: lat[0],
    )


def _build_world(state, model, mv, direction, threshold, obs):
    """Selecciona el world-model de imaginación según las intervenciones del escenario."""
    keys = set(model)
    # 1) Identificar el mundo que A11 sabe imaginar (por su set de intervenciones).
    if {"activate_cooling", "deactivate_cooling"} <= keys:
        kind = "thermal"
    elif {"boost_throughput", "shed_load"} <= keys:
        kind = "deferred"
    else:
        return None, "imagination_no_world_model"
    # 2) Validar la observación de la variable principal.
    x0 = obs.get(mv)
    if not isinstance(x0, (int, float)):
        return None, "imagination_no_observation"
    x0 = float(x0)
    if kind == "thermal":
        cooling = abs(dc.num(model.get("activate_cooling")))
        if cooling <= 0.0:
            return None, "imagination_no_cooling_signal"
        return thermal_world(
            x0=x0, cooling=cooling, drift=_ASSUMED_DRIFT, threshold=threshold,
            direction=direction, cooling_active=bool(obs.get("cooling_active", False)),
        ), None
    return deferred_load_world(
        x0=x0, debt=dc.num(obs.get("debt")),
        boost_effect=abs(dc.num(model.get("boost_throughput"))),
        shed_effect=abs(dc.num(model.get("shed_load"))),
        drift=_ASSUMED_DRIFT, threshold=threshold, direction=direction,
    ), None


def execute(state):
    if not ci.family_deep_enabled(FAMILY_ID):
        return _idle()

    model = ci._effect_model(state)
    mv = ci.main_variable(state)
    direction = ci.optimization_direction(state, ci.resolve_signature(state))
    obs = dc.safe_dict(state.get("observation"))
    md = dc.safe_dict(state.get("scenario_metadata"))
    threshold = md.get("alarm_threshold")
    threshold = float(threshold) if isinstance(threshold, (int, float)) else None

    world, reason = _build_world(state, model, mv, direction, threshold, obs)
    if world is None:
        return _idle_signal(reason)

    res = imagine(**world, chosen=state.get("intervention"), horizon=_HORIZON)

    breach_at = res.get("chosen_breaches_at")
    disagree = bool(res.get("disagrees_with_choice", False))
    status = "warn" if (breach_at is not None or disagree) else "ok"

    return {
        "family": FAMILY_ID,
        "status": status,
        "state_delta": {
            "imagination_active": True,
            "imagination_speculative": True,
            "imagination_recommended_intervention": res["recommended_intervention"],
            "imagination_recommended_terminal": res["recommended_terminal"],
            "imagination_chosen_breaches_at": breach_at,
            "imagination_chosen_terminal": res.get("chosen_terminal"),
            "imagination_disagrees_with_choice": disagree,
            "imagination_horizon": _HORIZON,
            "imagination_assumed_drift": _ASSUMED_DRIFT,
        },
        "confidence": 0.35,
        "cost": 0.6,
        "recommended_next_family": "PROB",
        "failure_mode": None,
        "artifacts": {
            "chosen_trajectory": res.get("chosen_trajectory"),
            "per_action_terminal": res["per_action_terminal"],
        },
    }
