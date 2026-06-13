"""Evaluación de cierre triádico y trazabilidad por episodio."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from runtime.lotf import LOTFMin


# ───────────────────────────  CLOSURE PROFILES  ───────────────────────────────

@dataclass(frozen=True)
class ClosureProfile:
    """Perfil de cierre que define secuencia requerida y familias opcionales.

    Attributes:
        name: Nombre del perfil (e.g., 'baseline_fixed', 'adaptive_min').
        required_sequence: Lista ordenada de familias obligatorias.
        partial_order: Diccionario de dependencias parciales (familia -> debe ir antes de).
        optional_families: Familias adicionales que son válidas pero no obligatorias.
        prob_must_close: Si PROB debe ser la última familia en calibración.
    """

    name: str
    required_sequence: List[str]
    partial_order: Dict[str, Set[str]] = field(default_factory=dict)
    optional_families: Set[str] = field(default_factory=set)
    prob_must_close: bool = True


# Perfil baseline fijo: la secuencia exacta original
# Este perfil es un contrato duro que garantiza:
# - misma longitud exacta
# - mismo orden exacto
# - sin duplicados
# - sin intercalaciones de otras familias
BASELINE_FIXED_PROFILE = ClosureProfile(
    name="baseline_fixed",
    required_sequence=["ABD", "ANA", "CAU", "CTF", "DED", "PROB"],
    partial_order={
        "ABD": {"ANA", "CAU", "CTF", "DED", "PROB"},  # ABD debe ir antes de todos estos
        "ANA": {"CAU", "CTF", "DED", "PROB"},
        "CAU": {"CTF", "DED", "PROB"},
        "CTF": {"DED", "PROB"},
        "DED": {"PROB"},
    },
    optional_families=set(),  # baseline NO permite familias opcionales
    prob_must_close=True,
)

# Perfil adaptativo mínimo: orden parcial obligatorio + familias extra permitidas
# Este perfil permite ecología adaptativa:
# - Familias obligatorias en orden parcial
# - Familias opcionales intercaladas (DIA_ADV, HEUR, FAL_GUARD, EML_SR)
# - PROB debe cerrar la calibración
ADAPTIVE_MIN_PROFILE = ClosureProfile(
    name="adaptive_min",
    required_sequence=["ABD", "ANA", "CAU", "CTF", "DED", "PROB"],
    partial_order={
        "ABD": {"ANA", "CAU", "CTF", "DED", "PROB"},
        "ANA": {"CAU", "CTF", "DED", "PROB"},
        "CAU": {"CTF", "DED", "PROB"},
        "CTF": {"DED", "PROB"},
        "DED": {"PROB"},
    },
    optional_families={"DIA_ADV", "HEUR", "FAL_GUARD", "EML_SR", "IND", "PLAN", "OPT"},
    prob_must_close=True,
)

# Registro de perfiles disponibles
CLOSURE_PROFILES: Dict[str, ClosureProfile] = {
    "baseline_fixed": BASELINE_FIXED_PROFILE,
    "adaptive_min": ADAPTIVE_MIN_PROFILE,
}

# Mantener compatibilidad con código existente
REQUIRED_META_SEQUENCE = BASELINE_FIXED_PROFILE.required_sequence


# ───────────────────────────  VALIDATION HELPERS  ─────────────────────────────

def _has_episode_closed_event(storage, *, run_id: str | None, episode_id: str) -> bool:
    events = storage.list_events(run_id=run_id, limit=500)
    for item in events:
        if item.event_type != "episode.closed":
            continue
        payload = item.payload or {}
        if payload.get("episode_id") == episode_id:
            return True
    return False


# Contextos de tipos conocidos por escenario
KNOWN_TYPE_CONTEXTS: Dict[str, Dict[str, str]] = {
    "thermal_homeostasis": {"TEMP_HIGH": "bool", "ACTIVATE_COOLING": "bool"},
    "resource_management": {"STOCK_LOW": "bool", "START_PRODUCTION": "bool"},
}


def _infer_type_context(formula: str, episode: Dict[str, Any]) -> Dict[str, str]:
    """Infiere contexto de tipos para la fórmula.

    Args:
        formula: Fórmula LOTF.
        episode: Diccionario del episodio.

    Returns:
        Dict con símbolos y sus tipos.
    """
    # Intentar obtener escenario del episodio
    scenario_name = episode.get("scenario", "thermal_homeostasis")
    if scenario_name in KNOWN_TYPE_CONTEXTS:
        return KNOWN_TYPE_CONTEXTS[scenario_name]

    # Fallback: extraer símbolos de la fórmula y asumir bool
    # Esto es un parser simple para símbolos
    symbols = re.findall(r"[A-Z][A-Z_0-9]+", formula)
    return {sym: "bool" for sym in set(symbols)}


def _normalize_sequence(sequence: List[str]) -> List[str]:
    """Normaliza secuencia a uppercase."""
    return [fam.upper() for fam in sequence]


def _validate_partial_order(
    sequence: List[str],
    partial_order: Dict[str, Set[str]],
) -> bool:
    """Valida que la secuencia respeta el orden parcial.

    Args:
        sequence: Secuencia de familias ejecutadas.
        partial_order: Dict donde key debe aparecer antes que todos los items en value.

    Returns:
        True si el orden parcial se respeta.
    """
    positions = {fam: idx for idx, fam in enumerate(sequence)}
    for before_fam, after_fams in partial_order.items():
        if before_fam not in positions:
            continue  # Si no está en la secuencia, no verificamos
        before_pos = positions[before_fam]
        for after_fam in after_fams:
            if after_fam not in positions:
                continue
            if positions[after_fam] <= before_pos:
                return False
    return True


def _validate_required_families(
    sequence: List[str],
    required: List[str],
) -> bool:
    """Verifica que todas las familias requeridas estén presentes."""
    sequence_set = set(sequence)
    return all(fam in sequence_set for fam in required)


def _validate_prob_closes(sequence: List[str]) -> bool:
    """Verifica que PROB sea la última familia de calibración."""
    if not sequence:
        return False
    return sequence[-1] == "PROB"


def _validate_no_unknown_families(
    sequence: List[str],
    required: List[str],
    optional: Set[str],
) -> bool:
    """Verifica que no hay familias desconocidas en la secuencia."""
    valid_families = set(required) | optional
    return all(fam in valid_families for fam in sequence)


def _validate_exact_sequence(
    sequence: List[str],
    required: List[str],
) -> bool:
    """Verifica que la secuencia sea exactamente igual a la requerida.
    
    Para baseline_fixed: misma longitud, mismo orden, sin duplicados, sin intercalaciones.
    """
    return sequence == required


def validate_sequence_with_profile(
    sequence: List[str],
    profile: ClosureProfile,
) -> Dict[str, Any]:
    """Valida una secuencia contra un perfil de cierre.

    Args:
        sequence: Secuencia de familias ejecutadas.
        profile: Perfil de cierre a usar.

    Returns:
        Dict con resultado de validación y detalles.
    """
    normalized = _normalize_sequence(sequence)

    # Para baseline_fixed, usar validación exacta estricta
    if profile.name == "baseline_fixed":
        exact_match = _validate_exact_sequence(normalized, profile.required_sequence)
        checks = {
            "exact_sequence_match": exact_match,
            "required_families_present": _validate_required_families(
                normalized, profile.required_sequence
            ),
            "partial_order_respected": _validate_partial_order(
                normalized, profile.partial_order
            ),
            "no_unknown_families": _validate_no_unknown_families(
                normalized, profile.required_sequence, profile.optional_families
            ),
            "prob_closes_calibration": _validate_prob_closes(normalized),
        }
        passed = exact_match  # baseline_fixed requiere match exacto
    else:
        # Para otros perfiles (adaptive_min), usar validación flexible
        checks = {
            "required_families_present": _validate_required_families(
                normalized, profile.required_sequence
            ),
            "partial_order_respected": _validate_partial_order(
                normalized, profile.partial_order
            ),
            "no_unknown_families": _validate_no_unknown_families(
                normalized, profile.required_sequence, profile.optional_families
            ),
        }

        if profile.prob_must_close:
            checks["prob_closes_calibration"] = _validate_prob_closes(normalized)

        passed = all(checks.values())

    return {
        "profile": profile.name,
        "passed": passed,
        "checks": checks,
        "sequence_normalized": normalized,
        "required_families": profile.required_sequence,
        "optional_families": list(profile.optional_families),
    }


# ───────────────────────────  MAIN EVALUATOR  ─────────────────────────────────

def evaluate_episode_closure(
    *,
    storage,
    run_id: str | None,
    result: Dict[str, Any],
    closure_profile: str = "baseline_fixed",
) -> Dict[str, Any]:
    """Evalúa el cierre triádico de un episodio.

    Args:
        storage: Facade de almacenamiento.
        run_id: ID de corrida.
        result: Resultado del episodio con episode, smg_snapshot, etc.
        closure_profile: Nombre del perfil de cierre ('baseline_fixed' o 'adaptive_min').

    Returns:
        Dict con episode_id, checks, closure_passed, trace_integrity.
    """
    episode = result.get("episode", {})
    smg_snapshot = result.get("smg_snapshot", {})
    episode_id = episode.get("episode_id", "")
    context = episode.get("context", {})
    output = episode.get("result", {})
    trace = episode.get("trace", [])

    has_observation = isinstance(context.get("observation"), dict)
    has_signs = len(smg_snapshot.get("signs", [])) >= 2

    # Determinar contexto de tipos basado en el escenario
    formula_ok = False
    formula = context.get("formula")
    if isinstance(formula, str) and formula.strip():
        try:
            parsed = LOTFMin().parse(formula)
            # Inferir type_context de la fórmula o usar defaults conocidos
            type_context = _infer_type_context(formula, episode)
            LOTFMin().check(parsed, type_context)
            formula_ok = True
        except Exception:
            formula_ok = False

    has_intervention = isinstance(context.get("intervention"), str) and isinstance(
        context.get("counterfactual"), dict
    )
    has_episode_closed = bool(episode_id) and _has_episode_closed_event(
        storage, run_id=run_id, episode_id=episode_id
    )

    # Usar perfil de cierre para validar secuencia
    reasoning_sequence = output.get("reasoning_sequence", []) or []
    profile = CLOSURE_PROFILES.get(closure_profile, BASELINE_FIXED_PROFILE)

    sequence_validation = validate_sequence_with_profile(reasoning_sequence, profile)

    # Para compatibilidad: meta_trace_complete ahora usa el perfil
    meta_trace_complete = (
        isinstance(trace, list)
        and len(trace) >= len(profile.required_sequence)
        and sequence_validation["passed"]
    )
    trace_integrity = has_episode_closed and meta_trace_complete

    checks = {
        "observation_registered": has_observation,
        "signs_persisted": has_signs,
        "lotf_parse_check_ok": formula_ok,
        "factual_and_counterfactual_present": has_intervention,
        "episode_closed_event_present": has_episode_closed,
        "meta_trace_complete": meta_trace_complete,
    }
    closure_passed = all(checks.values())

    return {
        "episode_id": episode_id,
        "checks": checks,
        "closure_passed": closure_passed,
        "trace_integrity": trace_integrity,
        "closure_profile": closure_profile,
        "sequence_validation": sequence_validation,
    }
