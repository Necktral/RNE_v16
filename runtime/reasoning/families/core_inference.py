"""Motor de inferencia riguroso compartido para las familias core.

Reemplaza los stubs de ABD/ANA/CAU/CTF/PROB por inferencia real y determinista
sobre el estado del episodio (observación, transición factual/contrafactual,
intervención, firma causal del escenario, memoria recuperada y belief state).

Diseño híbrido:
- **Núcleo simbólico** (este módulo, funciones `abduce/analogize/causal_infer/
  counterfactual_check/calibrate`): determinista, en milisegundos, sin dependencias
  externas. Es el motor base, autoritativo para la decisión y la reproducibilidad.
- **Aumento LLM opcional** (`maybe_llm_augment`): bajo `ExternalReasonerGate` (solo en
  conflicto causal/contrafactual o ambigüedad) y **opt-in** vía
  `RNFE_CORE_FAMILIES_LLM=1`. Llama al OpenThinker3-7B (llama.cpp en /mnt/d) **como
  máximo una vez por episodio** (cacheado en el estado) y deja evidencia advisoria;
  NO altera la decisión simbólica determinista. Por defecto está desactivado, de modo
  que el comportamiento nominal y los benchmarks reproducibles no cambian.

El contrato con el resto del sistema se preserva: cada familia sigue poblando su clave
(`abd_hypothesis`/`ana_mapping`/`cau_link`/`ctf_checked`/`prob_calibrated`) con un valor
*truthy*, ahora con contenido real en vez de un flag.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Mapping, Optional


# ─────────────────────────────── helpers de estado ───────────────────────────

def _meta(state: Mapping[str, Any]) -> Dict[str, Any]:
    meta = state.get("_meta")
    return meta if isinstance(meta, dict) else {}


def features(state: Mapping[str, Any]) -> Dict[str, float]:
    feats = _meta(state).get("features")
    return feats if isinstance(feats, dict) else {}


def _safe_dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _num(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return min(max(x, lo), hi)


_SIG_CACHE: Dict[str, Any] = {}


def scenario_name(state: Mapping[str, Any]) -> str:
    md = _safe_dict(state.get("scenario_metadata"))
    name = md.get("scenario_name") or state.get("scenario")
    return name if isinstance(name, str) else ""


def main_variable(state: Mapping[str, Any]) -> str:
    md = _safe_dict(state.get("scenario_metadata"))
    mv = md.get("main_variable")
    if isinstance(mv, str) and mv:
        return mv
    obs = _safe_dict(state.get("observation"))
    for cand in ("world_level", "global_temp_mean", "temperature", "stock_level"):
        if cand in obs:
            return cand
    return "temperature"


def _value(d: Mapping[str, Any], main_var: str) -> Optional[float]:
    if main_var in d:
        return _num(d.get(main_var), None)
    for cand in ("global_temp_mean", "world_level"):
        if cand in d:
            return _num(d.get(cand), None)
    return None


def resolve_signature(state: Mapping[str, Any]):
    """Firma causal del escenario vía el registro (best-effort, cacheada)."""
    name = scenario_name(state)
    if not name:
        return None
    if name in _SIG_CACHE:
        return _SIG_CACHE[name]
    sig = None
    try:  # lazy + tolerante: no acopla import-time ni rompe en escenarios desconocidos
        from runtime.world.registry import get_scenario

        sig = get_scenario(name).causal_signature
    except Exception:
        sig = None
    _SIG_CACHE[name] = sig
    return sig


def optimization_direction(state: Mapping[str, Any], sig: Any) -> str:
    direction = getattr(sig, "optimization_direction", None)
    if isinstance(direction, str) and direction:
        return direction
    name = scenario_name(state).lower()
    if "resource" in name or "stock" in name:
        return "maximize"
    return "minimize"


def observed_transition(state: Mapping[str, Any]) -> Dict[str, Any]:
    mv = main_variable(state)
    obs = _safe_dict(state.get("observation"))
    factual = _safe_dict(state.get("updated_world") or state.get("factual"))
    counterfactual = _safe_dict(state.get("counterfactual"))
    x0 = _value(obs, mv)
    xf = _value(factual, mv)
    xcf = _value(counterfactual, mv)
    return {
        "main_var": mv,
        "x0": x0,
        "xf": xf,
        "xcf": xcf,
        "factual_delta": (xf - x0) if (xf is not None and x0 is not None) else None,
        "counterfactual_delta": (xcf - x0) if (xcf is not None and x0 is not None) else None,
        "effect": (xf - xcf) if (xf is not None and xcf is not None) else None,
    }


def _better(a: Optional[float], b: Optional[float], direction: str) -> Optional[bool]:
    """¿Es `a` al menos tan bueno como `b` según la dirección de optimización?"""
    if a is None or b is None:
        return None
    return a >= b if direction == "maximize" else a <= b


# ─────────────────────────────── CAU: inferencia causal ──────────────────────

def causal_infer(state: Mapping[str, Any]) -> Dict[str, Any]:
    tr = observed_transition(state)
    sig = resolve_signature(state)
    direction = optimization_direction(state, sig)
    intervention = state.get("intervention")
    effect = tr["effect"]

    helps_goal: Optional[bool] = None
    if effect is not None:
        helps_goal = (effect < 0) if direction == "minimize" else (effect > 0)

    strength = _clamp(abs(effect) * 5.0) if effect is not None else 0.0

    expected_dir = None
    if sig is not None:
        for eff in getattr(sig, "intervention_effects", ()) or ():
            if getattr(eff, "intervention_name", None) == intervention:
                expected_dir = getattr(eff, "expected_direction", None)
                break
    fd = tr["factual_delta"]
    observed_dir = None
    if fd is not None:
        observed_dir = "+" if fd > 1e-9 else ("-" if fd < -1e-9 else "0")
    direction_match = bool(
        expected_dir is not None and observed_dir is not None and expected_dir == observed_dir
    )

    cau_link = {
        "intervention": intervention,
        "target": tr["main_var"],
        "observed_effect": None if effect is None else round(effect, 6),
        "factual_delta": None if fd is None else round(fd, 6),
        "helps_goal": helps_goal,
        "expected_direction": expected_dir,
        "observed_direction": observed_dir,
        "direction_match": direction_match,
        "strength": round(strength, 4),
        "optimization_direction": direction,
    }
    confidence = _clamp(
        0.5
        + 0.3 * strength
        + (0.2 if helps_goal else (-0.1 if helps_goal is False else 0.0))
    )
    return {"state_delta": {"cau_link": cau_link}, "confidence": confidence}


# ─────────────────────────────── CTF: contrafactual ──────────────────────────

def counterfactual_check(state: Mapping[str, Any]) -> Dict[str, Any]:
    tr = observed_transition(state)
    direction = optimization_direction(state, resolve_signature(state))
    supports = _better(tr["xf"], tr["xcf"], direction)
    relation_kind = state.get("relation_kind")
    agreement: Optional[bool] = None
    if supports is not None and relation_kind in ("support", "contradiction"):
        agreement = ((relation_kind == "support") == bool(supports))

    ctf_checked = {
        "factual": tr["xf"],
        "counterfactual": tr["xcf"],
        "delta": None if tr["effect"] is None else round(tr["effect"], 6),
        "supports_choice": None if supports is None else bool(supports),
        "relation_kind": relation_kind,
        "agreement_with_relation_kind": agreement,
        "optimization_direction": direction,
    }
    confidence = _clamp(
        0.55
        + (0.25 if supports else (-0.1 if supports is False else 0.0))
        + (0.1 if agreement else 0.0)
    )
    return {"state_delta": {"ctf_checked": ctf_checked}, "confidence": confidence}


# ─────────────────────────────── ABD: abducción ──────────────────────────────

def abduce(state: Mapping[str, Any]) -> Dict[str, Any]:
    sig = resolve_signature(state)
    direction = optimization_direction(state, sig)
    md = _safe_dict(state.get("scenario_metadata"))
    obs = _safe_dict(state.get("observation"))
    alarm = bool(obs.get("alarm"))
    mv = main_variable(state)

    eff_map: Dict[str, Any] = {}
    if sig is not None:
        for eff in getattr(sig, "intervention_effects", ()) or ():
            eff_map[getattr(eff, "intervention_name", "")] = eff
    interventions = list(md.get("interventions") or list(eff_map.keys()))

    candidates: List[Dict[str, Any]] = []
    for interv in interventions:
        eff = eff_map.get(interv)
        score = 0.3
        if eff is not None:
            moves = getattr(eff, "expected_direction", None)
            good_dir = (moves == "-") if direction == "minimize" else (moves == "+")
            magnitude = _num(getattr(eff, "expected_magnitude", 0.0), 0.0) or 0.0
            role = getattr(eff, "semantic_role", "")
            score = 0.2 + 0.4 * (1.0 if good_dir else 0.0) + 0.3 * _clamp(magnitude * 5.0)
            if role == "corrective":
                score += 0.1
            if alarm and role == "corrective":
                score += 0.15
            if not alarm and role == "neutral":
                score += 0.10
        candidates.append(
            {
                "hypothesis": f"{interv}_explains_resolves_{mv}",
                "intervention": interv,
                "score": round(_clamp(score), 4),
            }
        )
    candidates.sort(key=lambda c: c["score"], reverse=True)
    top = candidates[0] if candidates else {"hypothesis": "no_candidates", "intervention": None, "score": 0.0}
    margin = (top["score"] - candidates[1]["score"]) if len(candidates) > 1 else top["score"]

    confidence = _clamp(0.4 + 0.4 * top["score"] + 0.2 * _clamp(margin * 3.0))
    return {
        "state_delta": {
            "abd_hypothesis": top,
            "abd_hypotheses": candidates,
            "abd_top_intervention": top["intervention"],
        },
        "confidence": confidence,
    }


# ─────────────────────────────── ANA: analogía ───────────────────────────────

def analogize(state: Mapping[str, Any]) -> Dict[str, Any]:
    memory = state.get("retrieved_memory") or state.get("memory_hits") or []
    obs = _safe_dict(state.get("observation"))
    props = {str(p) for p in (obs.get("propositions") or [])}

    best: Optional[Dict[str, Any]] = None
    if isinstance(memory, list) and memory:
        scored = sorted(
            (m for m in memory if isinstance(m, dict)),
            key=lambda m: _num(m.get("score"), 0.0) or 0.0,
            reverse=True,
        )
        if scored:
            top = scored[0]
            best = {
                "source": "memory",
                "memory_id": top.get("memory_id"),
                "scale": top.get("scale"),
                "alignment_score": round(_num(top.get("score"), 0.0) or 0.0, 4),
                "analogical_source": bool(top.get("analogical_source")),
            }
    if best is None:
        sig = resolve_signature(state)
        vocab = set(getattr(sig, "proposition_vocabulary", set()) or set())
        overlap = (len(props & vocab) / len(vocab)) if vocab else 0.0
        best = {
            "source": "scenario_self",
            "alignment_score": round(_clamp(overlap), 4),
            "matched_propositions": sorted(props & vocab),
        }

    confidence = _clamp(0.45 + 0.5 * (_num(best.get("alignment_score"), 0.0) or 0.0))
    return {"state_delta": {"ana_mapping": best}, "confidence": confidence}


# ─────────────────────────────── IND: inducción ──────────────────────────────

_TRANSFER_VERDICTS_OK = {"certified_transfer_safe", "certified_analogical_only"}


def _best_inherited_rule(inherited_rules: Any) -> "tuple[str | None, float] | None":
    """Mejor regla transferida usable: (best_intervention, confianza_atenuada).

    Confianza = confidence_lcb · (1 − info_loss) · overall_score. Solo cuenta si
    el veredicto de transferencia es ≥ analógico. Devuelve None si no hay ninguna.
    """
    if not isinstance(inherited_rules, (list, tuple)) or not inherited_rules:
        return None
    best: "tuple[str | None, float] | None" = None
    for entry in inherited_rules:
        if not isinstance(entry, Mapping):
            continue
        iv = entry.get("best_intervention")
        if not iv:
            continue
        transfer = entry.get("transfer") if isinstance(entry.get("transfer"), Mapping) else {}
        verdict = transfer.get("verdict") or entry.get("transfer_verdict")
        if verdict is not None and verdict not in _TRANSFER_VERDICTS_OK:
            continue
        lcb = entry.get("confidence_lcb")
        lcb = float(lcb) if isinstance(lcb, (int, float)) else 0.0
        info_loss = transfer.get("info_loss", 0.0)
        info_loss = float(info_loss) if isinstance(info_loss, (int, float)) else 0.0
        overall = transfer.get("overall_score", 1.0)
        overall = float(overall) if isinstance(overall, (int, float)) else 1.0
        attenuated = lcb * max(0.0, 1.0 - info_loss) * max(0.0, min(1.0, overall))
        if best is None or attenuated > best[1]:
            best = (str(iv), attenuated)
    return best


def induce(state: Mapping[str, Any]) -> Dict[str, Any]:
    """IND: induce una regularidad general a partir de ejemplos + firma causal.

    Reemplaza el stub idle. Generaliza sobre los episodios análogos recuperados
    (``retrieved_memory``) la regla «bajo alarma A, intervención X → relación R
    sobre la variable principal», con conteo de soporte y cota inferior de
    confianza (Agresti-Coull, la misma que usa PROB). Si no hay ejemplos, induce
    *a priori* desde el modelo de efectos declarado de la firma causal. Núcleo
    determinista, sin dependencias externas.
    """
    memory = state.get("retrieved_memory") or state.get("memory_hits") or []
    obs = _safe_dict(state.get("observation"))
    alarm = bool(obs.get("alarm"))
    mv = main_variable(state)
    sig = resolve_signature(state)
    direction = optimization_direction(state, sig)

    # 1) Inducción empírica desde ejemplos (generalización por enumeración).
    by_interv: Dict[str, Dict[str, int]] = {}
    support = total = 0
    for m in memory:
        if not isinstance(m, dict):
            continue
        struct = _safe_dict(m.get("structure"))
        rk = struct.get("relation_kind")
        iv = struct.get("intervention") or struct.get("intervention_label")
        if rk in ("support", "contradiction"):
            total += 1
            support += 1 if rk == "support" else 0
            if iv:
                slot = by_interv.setdefault(str(iv), {"support": 0, "total": 0})
                slot["total"] += 1
                slot["support"] += 1 if rk == "support" else 0

    if total > 0:
        p = support / total
        lcb = _beta_lcb(p, total)
        best_iv, best_rate, best_n = None, -1.0, 0
        for iv, s in by_interv.items():
            if s["total"] <= 0:
                continue
            rate = s["support"] / s["total"]
            if rate > best_rate or (rate == best_rate and s["total"] > best_n):
                best_iv, best_rate, best_n = iv, rate, s["total"]
        rule = {
            "source": "memory",
            "antecedent": {"alarm": alarm, "main_variable": mv},
            "consequent_relation": "support" if p >= 0.5 else "contradiction",
            "support_rate": round(p, 4),
            "support": support,
            "total": total,
            "confidence_lcb": round(lcb, 4),
            "best_intervention": best_iv,
            "best_intervention_support_rate": None if best_iv is None else round(best_rate, 4),
        }
        confidence = _clamp(0.40 + 0.50 * lcb)
        law_fit = _clamp(lcb)
    else:
        # 2) Inducción a priori desde el modelo de efectos de la firma causal.
        generalized: List[str] = []
        if sig is not None:
            for eff in getattr(sig, "intervention_effects", ()) or ():
                moves = getattr(eff, "expected_direction", None)
                good = (moves == "-") if direction == "minimize" else (moves == "+")
                role = getattr(eff, "semantic_role", "")
                if good or role == "corrective":
                    name = getattr(eff, "intervention_name", "")
                    if name:
                        generalized.append(name)
        rule = {
            "source": "causal_signature",
            "antecedent": {"alarm": alarm, "main_variable": mv},
            "consequent_relation": "support",
            "generalized_interventions": generalized,
            "support": 0,
            "total": 0,
            "confidence_lcb": 0.0,
            "best_intervention": generalized[0] if generalized else None,
        }
        confidence = _clamp(0.40 + 0.12 * len(generalized))
        law_fit = _clamp(0.20 + 0.10 * len(generalized))

        # 2b) Regla transferida por la ecología multi-organismo: sin memoria
        # empírica propia, una regla certificada de un par (transportada por el
        # morfismo causal) sirve de prior. Atenuada por (1−info_loss)·overall_score
        # y solo si el veredicto de transferencia es ≥ analógico. No reescribe el
        # comportamiento sin ecología (inherited_rules ausente ⇒ rama intacta).
        inherited = _best_inherited_rule(state.get("inherited_rules"))
        if inherited is not None:
            iv, transferred_conf = inherited
            if iv and transferred_conf > confidence:
                rule["source"] = "transferred"
                rule["best_intervention"] = iv
                rule["transferred_confidence"] = round(transferred_conf, 4)
                if iv not in generalized:
                    generalized.insert(0, iv)
                    rule["generalized_interventions"] = generalized
                confidence = _clamp(transferred_conf)
                law_fit = _clamp(max(law_fit, 0.30))

    generalization = (
        f"alarm={alarm} & interv='{rule.get('best_intervention')}' "
        f"=> {rule['consequent_relation']}({mv}) [n={rule['total']}]"
    )
    return {
        "state_delta": {
            "ind_rule": rule,
            "ind_generalization": generalization,
            "ind_support": rule["total"],
            "ind_confidence_lcb": rule["confidence_lcb"],
            "ind_law_fit_signal": round(law_fit, 4),
            "ind_best_intervention": rule.get("best_intervention"),
        },
        "confidence": confidence,
    }


# ──────────────────────── modelo de efectos declarado ────────────────────────

def _effect_model(state: Mapping[str, Any]) -> Dict[str, float]:
    """Δ esperado de la variable principal por intervención (firma causal)."""
    sig = resolve_signature(state)
    md = _safe_dict(state.get("scenario_metadata"))
    model: Dict[str, float] = {}
    if sig is not None:
        for eff in getattr(sig, "intervention_effects", ()) or ():
            name = getattr(eff, "intervention_name", "")
            moves = getattr(eff, "expected_direction", None)
            mag = abs(_num(getattr(eff, "expected_magnitude", 0.0), 0.0) or 0.0)
            if name:
                model[name] = mag if moves == "+" else (-mag if moves == "-" else 0.0)
    for interv in md.get("interventions") or ():
        model.setdefault(str(interv), 0.0)
    return model


def _goal_reached(x: float, threshold: Optional[float], direction: str) -> Optional[bool]:
    if threshold is None:
        return None
    return x < threshold if direction == "minimize" else x > threshold


# ─────────────────────────────── PLAN: planificación ─────────────────────────

def plan_search(state: Mapping[str, Any], *, horizon: int = 3) -> Dict[str, Any]:
    """PLAN: búsqueda hacia adelante sobre el modelo de efectos declarado.

    Reemplaza el stub idle. Enumera secuencias de intervenciones hasta
    ``horizon`` pasos proyectando la variable principal con los Δ esperados de
    la firma causal (clamp [0,1]); objetivo = cruzar al lado seguro del umbral
    de alarma según la dirección de optimización. Devuelve el plan más corto
    que alcanza el objetivo (desempate por mejor valor terminal) o, si ninguno
    llega, la mejor trayectoria alcanzable. Determinista, ≤ |I|^horizon nodos
    (los escenarios declaran 2-4 intervenciones ⇒ coste trivial).
    """
    model = _effect_model(state)
    mv = main_variable(state)
    direction = optimization_direction(state, resolve_signature(state))
    md = _safe_dict(state.get("scenario_metadata"))
    threshold = _num(md.get("alarm_threshold"), None)
    obs = _safe_dict(state.get("observation"))
    x0 = _value(obs, mv)

    if x0 is None or not model:
        return {
            "state_delta": {
                "plan_built": False,
                "plan": {"status": "no_model_or_observation", "steps": []},
            },
            "confidence": 0.2,
        }

    interventions = sorted(model)  # orden determinista
    sign = -1.0 if direction == "minimize" else 1.0

    def better(a: float, b: float) -> bool:
        return (a < b) if direction == "minimize" else (a > b)

    best_goal: Optional[Dict[str, Any]] = None  # plan más corto que llega
    best_any: Optional[Dict[str, Any]] = None  # mejor terminal aunque no llegue
    frontier: List[Dict[str, Any]] = [{"x": float(x0), "steps": [], "traj": [round(float(x0), 6)]}]
    for depth in range(1, max(1, int(horizon)) + 1):
        nxt: List[Dict[str, Any]] = []
        for node in frontier:
            for interv in interventions:
                x = min(1.0, max(0.0, node["x"] + model[interv]))
                cand = {
                    "x": x,
                    "steps": node["steps"] + [interv],
                    "traj": node["traj"] + [round(x, 6)],
                }
                if best_any is None or better(x, best_any["x"]):
                    best_any = cand
                if _goal_reached(x, threshold, direction) and (
                    best_goal is None
                    or len(cand["steps"]) < len(best_goal["steps"])
                    or (len(cand["steps"]) == len(best_goal["steps"]) and better(x, best_goal["x"]))
                ):
                    best_goal = cand
                nxt.append(cand)
        if best_goal is not None:
            break  # BFS por profundidad ⇒ el primero hallado es el más corto
        frontier = nxt

    chosen = best_goal or best_any or {"x": float(x0), "steps": [], "traj": [round(float(x0), 6)]}
    goal_reached = bool(best_goal is not None)
    improvement = sign * (chosen["x"] - float(x0))
    plan = {
        "status": "goal_reached" if goal_reached else "best_effort",
        "main_variable": mv,
        "optimization_direction": direction,
        "alarm_threshold": threshold,
        "x0": round(float(x0), 6),
        "steps": chosen["steps"],
        "projected_trajectory": chosen["traj"],
        "projected_terminal": round(chosen["x"], 6),
        "horizon": int(horizon),
        "first_action": chosen["steps"][0] if chosen["steps"] else None,
    }
    confidence = _clamp(
        0.45 + (0.35 if goal_reached else 0.0) + 0.20 * _clamp(improvement * 5.0)
    )
    return {
        "state_delta": {
            "plan_built": True,
            "plan": plan,
            "plan_goal_reached": goal_reached,
            "plan_first_action": plan["first_action"],
        },
        "confidence": confidence,
    }


# ─────────────────────────────── OPT: optimización ───────────────────────────

def optimize_choice(state: Mapping[str, Any], *, horizon: int = 3, effort_cost: float = 0.05) -> Dict[str, Any]:
    """OPT: argmin de un objetivo escalar sobre (intervención, nº de pasos).

    Reemplaza el stub idle. Para cada intervención aplicada k=1..horizon veces
    proyecta la variable principal y minimiza
    ``objetivo = término_de_valor + effort_cost·k`` donde el término de valor
    es x (minimize), 1−x (maximize) — menor es mejor en ambos casos. Devuelve
    la elección óptima con la tabla de alternativas (explicable). Determinista.
    """
    model = _effect_model(state)
    mv = main_variable(state)
    direction = optimization_direction(state, resolve_signature(state))
    obs = _safe_dict(state.get("observation"))
    x0 = _value(obs, mv)

    if x0 is None or not model:
        return {
            "state_delta": {
                "opt_solved": False,
                "opt_choice": {"status": "no_model_or_observation"},
            },
            "confidence": 0.2,
        }

    def value_term(x: float) -> float:
        return x if direction == "minimize" else (1.0 - x)

    alternatives: List[Dict[str, Any]] = []
    for interv in sorted(model):
        x = float(x0)
        for k in range(1, max(1, int(horizon)) + 1):
            x = min(1.0, max(0.0, x + model[interv]))
            alternatives.append(
                {
                    "intervention": interv,
                    "steps": k,
                    "projected": round(x, 6),
                    "objective": round(value_term(x) + effort_cost * k, 6),
                }
            )
    best = min(alternatives, key=lambda a: (a["objective"], a["steps"], a["intervention"]))
    baseline = value_term(float(x0))
    gain = baseline - (best["objective"] - effort_cost * best["steps"])
    choice = {
        "status": "ok",
        "main_variable": mv,
        "optimization_direction": direction,
        "x0": round(float(x0), 6),
        "intervention": best["intervention"],
        "steps": best["steps"],
        "projected": best["projected"],
        "objective": best["objective"],
        "effort_cost": effort_cost,
        "alternatives": alternatives,
    }
    confidence = _clamp(0.5 + 0.4 * _clamp(gain * 5.0))
    return {
        "state_delta": {
            "opt_solved": True,
            "opt_choice": choice,
            "opt_intervention": best["intervention"],
        },
        "confidence": confidence,
    }


# ─────────────────────────────── PROB: calibración ───────────────────────────

def _beta_lcb(p: float, n: int, z: float = 1.96) -> float:
    """Cota inferior de confianza (Agresti-Coull)."""
    if n <= 0:
        return 0.0
    n_t = n + z * z
    p_t = (p * n + z * z / 2.0) / n_t
    interval = z * math.sqrt(max(0.0, p_t * (1.0 - p_t) / n_t))
    return _clamp(p_t - interval)


def calibrate(state: Mapping[str, Any]) -> Dict[str, Any]:
    feats = features(state)
    cau = _safe_dict(state.get("cau_link"))
    ctf = _safe_dict(state.get("ctf_checked"))
    belief = _safe_dict(state.get("belief_state"))
    posterior_src = _safe_dict(belief.get("posterior"))

    if cau:
        helps = cau.get("helps_goal")
        cau_factor = 1.0 if helps else (0.5 if helps is None else 0.2)
        cau_sig = _clamp((_num(cau.get("strength"), 0.0) or 0.0) * cau_factor + 0.2)
    else:
        cau_sig = 0.5

    supports = ctf.get("supports_choice") if ctf else None
    ctf_sig = 0.9 if supports else (0.5 if supports is None else 0.2)

    ded_sig = 0.85 if state.get("ded_validated") else (0.6 if state.get("ded_conclusion") else 0.5)
    certainty = 1.0 - _clamp(_num(feats.get("uncertainty"), 0.25) or 0.25)
    prior_causal = _num(posterior_src.get("causal_support_confidence"), 0.5) or 0.5

    p = _clamp(
        0.25 * cau_sig
        + 0.25 * ctf_sig
        + 0.20 * ded_sig
        + 0.15 * certainty
        + 0.15 * prior_causal
    )
    n_obs = 4 + sum(
        1 for k in ("cau_link", "ctf_checked", "abd_hypothesis", "ded_conclusion") if state.get(k)
    )
    lcb = _beta_lcb(p, n_obs)

    posterior = {
        "point": round(p, 4),
        "lower_confidence_bound": round(lcb, 4),
        "n_observations": n_obs,
        "evidence": {
            "cau": round(cau_sig, 4),
            "ctf": round(ctf_sig, 4),
            "ded": round(ded_sig, 4),
            "certainty": round(certainty, 4),
            "prior_causal": round(prior_causal, 4),
        },
    }
    return {
        "state_delta": {
            "prob_calibrated": True,  # contrato truthy preservado
            "prob_posterior": posterior,
            "prob_point": round(p, 4),
            "prob_lcb": round(lcb, 4),
        },
        "confidence": round(p, 4),
    }


# ─────────────────────────── aumento LLM opcional (gated) ─────────────────────

_SENTINEL = object()
_AUG_KEY = "_core_llm_augmentation"


def llm_enabled() -> bool:
    return os.environ.get("RNFE_CORE_FAMILIES_LLM", "").strip().lower() in {"1", "true", "yes", "on"}


def _aug_max_tokens() -> int:
    try:
        return int(os.environ.get("RNFE_CORE_FAMILIES_LLM_MAX_TOKENS", "64"))
    except (TypeError, ValueError):
        return 64


def _detect_conflict(state: Mapping[str, Any]) -> bool:
    cau = _safe_dict(state.get("cau_link"))
    ctf = _safe_dict(state.get("ctf_checked"))
    feats = features(state)
    if ctf.get("agreement_with_relation_kind") is False:
        return True
    if cau.get("helps_goal") is False:
        return True
    if (_num(feats.get("contradiction_signal"), 0.0) or 0.0) >= 0.45:
        return True
    if (_num(feats.get("ambiguity_signal"), 0.0) or 0.0) >= 0.55:
        return True
    return False


def maybe_llm_augment(state: Dict[str, Any], *, family: str) -> Optional[Dict[str, Any]]:
    """Aumento neuronal opt-in y gated; a lo sumo UNA llamada por episodio (cacheada).

    Devuelve la augmentación (o None). No altera la decisión simbólica: solo añade
    evidencia advisoria del razonador externo cuando hay conflicto/ambigüedad real.
    """
    if not llm_enabled():
        return None

    cached = state.get(_AUG_KEY, _SENTINEL)
    if cached is not _SENTINEL:
        return cached  # ya resuelto este episodio (puede ser None)

    try:
        from runtime.reasoning.external_models.gating import (
            ExternalReasonerGate,
            ExternalReasonerGateInput,
        )
    except Exception:
        state[_AUG_KEY] = None
        return None

    regime = "causal_counterfactual_conflict" if _detect_conflict(state) else str(
        _meta(state).get("regime_label") or ""
    )
    cau = _safe_dict(state.get("cau_link"))
    gate = ExternalReasonerGate()
    decision = gate.evaluate(
        ExternalReasonerGateInput(
            regime=regime,
            core_intervention=str(state.get("intervention") or ""),
            causal_recommended_intervention=str(
                state.get("abd_top_intervention") or state.get("intervention") or ""
            ),
            counterfactual_recommended_intervention=(
                None if _safe_dict(state.get("ctf_checked")).get("supports_choice") in (True, None) else "ALT"
            ),
            core_confidence_proxy=_num(state.get("prob_point"), None),
            core_metrics={
                "intervention_precision": _num(cau.get("strength"), 0.0) or 0.0,
                "viability_margin": 0.5,
                "closure_stable": True,
            },
        )
    )
    if not decision.called:
        state[_AUG_KEY] = None
        return None

    try:
        from runtime.reasoning.external_models import LlamaCppClient

        client = state.get("_external_reasoner_client") or LlamaCppClient()
        prompt = _build_core_prompt(state, decision_reason=decision.reason)
        result = client.generate(
            prompt,
            max_tokens=_aug_max_tokens(),
            temperature=0.1,
            top_p=0.9,
            allow_cpu_fallback=bool(
                os.environ.get("RNFE_CORE_FAMILIES_LLM_CPU_FALLBACK", "").strip().lower()
                in {"1", "true", "yes", "on"}
            ),
        )
    except Exception as exc:  # pragma: no cover - degradación robusta
        aug = {"ok": False, "error_type": "augment_exception", "error_message": str(exc)[:200]}
        state[_AUG_KEY] = aug
        return aug

    if result.get("ok"):
        aug = {
            "ok": True,
            "trigger_family": family,
            "gate_reason": decision.reason,
            "raw_excerpt": str(result.get("output_text") or result.get("stdout") or "")[:700],
            "latency_s": _num(result.get("latency_s"), 0.0),
            "backend": result.get("backend"),
        }
    else:
        aug = {
            "ok": False,
            "trigger_family": family,
            "gate_reason": decision.reason,
            "error_type": result.get("error_type"),
            "error_message": result.get("error_message"),
        }
    state[_AUG_KEY] = aug
    return aug


def _build_core_prompt(state: Mapping[str, Any], *, decision_reason: str) -> str:
    import json

    tr = observed_transition(state)
    md = _safe_dict(state.get("scenario_metadata"))
    payload = {
        "task": "resolve_core_reasoning_conflict_json",
        "json_only": True,
        "gate_reason": decision_reason,
        "scenario": scenario_name(state),
        "main_variable": tr["main_var"],
        "allowed_interventions": md.get("interventions") or [],
        "chosen_intervention": state.get("intervention"),
        "observed": {"x0": tr["x0"], "factual": tr["xf"], "counterfactual": tr["xcf"]},
        "relation_kind": state.get("relation_kind"),
        "abd_top": state.get("abd_top_intervention"),
        "question": "Is the chosen intervention best supported? Reply short JSON with keys "
        "{best_intervention, confidence_0_1, one_line_reason}.",
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
