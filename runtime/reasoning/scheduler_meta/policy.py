"""Política adaptativa y explicable para selección de familias."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .family_profiles import (
    AUGMENTER_FAMILIES,
    BACKBONE_FAMILIES,
    CONDITIONAL_SHADOW_FAMILIES,
    CORE_SEQUENCE,
    DELIBERATIVE_FAMILIES,
    EXT_OPEN_THINKER_ADMISSION,
    resolve_family_profile,
)


HARD_MAX_STEPS = 10
MANDATORY_FAMILY_FLOORS: Dict[str, List[str]] = {
    "homogeneous_safe": ["abd", "ded", "prob"],
    "heterogeneous_elevated": ["abd", "ana", "cau", "ctf"],
    "heterogeneous_warning": ["abd", "ana", "cau", "ctf", "ded"],
    "viability_edge": ["cau", "ctf", "ded", "prob"],
}
SAFE_FALLBACK_PROFILES: Dict[str, str] = {
    "homogeneous_safe": "core_only",
    "heterogeneous_elevated": "core_plus_heur",
    "heterogeneous_warning": "core_plus_guard",
    "viability_edge": "core_plus_guard",
}
OVERLAY_INSERTION_ORDER = ["heur", "ind", "dia_adv", "plan", "opt", "eml_sr", "ext_open_thinker", "fal_guard"]


def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return lo
    return min(max(numeric, lo), hi)


def _dedup(sequence: Iterable[str]) -> List[str]:
    out: List[str] = []
    for family in sequence:
        normalized = str(family or "").strip().lower()
        if normalized and normalized not in out:
            out.append(normalized)
    return out


def _upper_list(sequence: Iterable[str]) -> List[str]:
    return [str(family).strip().upper() for family in sequence if str(family).strip()]


def normalize_regime_label(
    regime_hint: str | None,
    *,
    features: Dict[str, float],
    ignore_vram: bool = False,
) -> str:
    hint = (regime_hint or "").strip().lower()
    if hint:
        if "viability" in hint or "edge" in hint:
            return "viability_edge"
        if not ignore_vram and "vram" in hint:
            return "vram_favorable"
        if "heterogeneous" in hint:
            if "warning" in hint:
                return "heterogeneous_warning"
            if "elevated" in hint:
                return "heterogeneous_elevated"
            return "heterogeneous_elevated"
        if "homogeneous" in hint:
            return "homogeneous_safe"

    viability_edge_signal = _clamp(features.get("viability_edge_signal", 0.0))
    heterogeneity_signal = _clamp(features.get("heterogeneity_signal", 0.0))
    vram_favorable_signal = _clamp(features.get("vram_favorable_signal", 0.0))

    if viability_edge_signal >= 0.70:
        return "viability_edge"
    if heterogeneity_signal >= 0.55 and _clamp(features.get("world_level_signal", 0.0)) >= 0.82:
        return "heterogeneous_warning"
    if heterogeneity_signal >= 0.40:
        return "heterogeneous_elevated"
    if not ignore_vram and vram_favorable_signal >= 0.62:
        return "vram_favorable"
    return "homogeneous_safe"


def resolve_regime_labels(
    *,
    regime_hint: str | None,
    features: Dict[str, float],
) -> Dict[str, str]:
    primary_regime_label = normalize_regime_label(regime_hint, features=features)
    if primary_regime_label == "vram_favorable":
        cognitive_regime_label = normalize_regime_label(None, features=features, ignore_vram=True)
        floor_regime_label = cognitive_regime_label
    else:
        cognitive_regime_label = primary_regime_label
        floor_regime_label = primary_regime_label
    return {
        "primary_regime_label": primary_regime_label,
        "cognitive_regime_label": cognitive_regime_label,
        "floor_regime_label": floor_regime_label,
    }


def mandatory_family_floor(regime_label: str | None) -> List[str]:
    normalized = str(regime_label or "").strip().lower()
    return list(MANDATORY_FAMILY_FLOORS.get(normalized, MANDATORY_FAMILY_FLOORS["homogeneous_safe"]))


def safe_fallback_profile(regime_label: str | None) -> str:
    normalized = str(regime_label or "").strip().lower()
    return SAFE_FALLBACK_PROFILES.get(normalized, "core_only")


def _base_score_map(features: Dict[str, float]) -> Dict[str, float]:
    return {
        "abd": 0.30 + (0.22 * _clamp(features.get("ambiguity_signal", 0.0))),
        "ana": 0.22 + (0.30 * _clamp(features.get("heterogeneity_signal", 0.0))),
        "cau": 0.22 + (0.30 * _clamp(features.get("causal_risk", 0.0))),
        "ctf": 0.20 + (0.32 * _clamp(features.get("causal_risk", 0.0))),
        "ded": 0.18 + (0.20 * (1.0 - _clamp(features.get("continuity_recent", 1.0)))),
        "prob": 0.25 + (0.25 * _clamp(features.get("uncertainty", 0.0))),
        "heur": 0.12 + (0.30 * _clamp(features.get("edge_pressure", 0.0))),
        "dia_adv": 0.10 + (0.30 * _clamp(features.get("contradiction_signal", 0.0))),
        "fal_guard": 0.10 + (0.30 * _clamp(features.get("fragility_risk_signal", 0.0))),
        "ind": 0.08 + (0.32 * _clamp(features.get("pattern_without_structure_signal", 0.0))),
        "eml_sr": (
            0.08
            + (0.30 * _clamp(features.get("symbolic_regularity", 0.0)))
            + (0.22 * _clamp(features.get("law_fit_signal", 0.0)))
        ),
        "ext_open_thinker": (
            0.08
            + (0.24 * _clamp(features.get("ambiguity_signal", 0.0)))
            + (0.24 * _clamp(features.get("contradiction_signal", 0.0)))
            + (0.18 * _clamp(features.get("viability_edge_signal", 0.0)))
        ),
        "plan": (
            0.10
            + (0.30 * _clamp(features.get("viability_edge_signal", 0.0)))
            + (0.12 * _clamp(features.get("world_level_signal", 0.0)))
        ),
        "opt": (
            0.10
            + (0.28 * _clamp(features.get("causal_risk", 0.0)))
            + (0.12 * _clamp(features.get("viability_edge_signal", 0.0)))
        ),
    }


def _regime_boosts(
    *,
    scores: Dict[str, float],
    features: Dict[str, float],
    regime_label: str,
) -> None:
    if regime_label.startswith("heterogeneous"):
        scores["ana"] += 0.25
        scores["cau"] += 0.20
        scores["ctf"] += 0.20
        scores["heur"] += 0.28
    if regime_label == "viability_edge":
        scores["cau"] += 0.24
        scores["ctf"] += 0.24
        scores["dia_adv"] += 0.22
        scores["fal_guard"] += 0.28
    if regime_label == "vram_favorable":
        scores["heur"] += 0.18
        scores["ana"] += 0.14
        scores["ind"] += 0.18
    if _clamp(features.get("ambiguity_signal", 0.0)) >= 0.55:
        scores["abd"] += 0.22
        scores["ana"] += 0.12
        scores["dia_adv"] += 0.22
    if _clamp(features.get("fragility_risk_signal", 0.0)) >= 0.50:
        scores["fal_guard"] += 0.30
        scores["dia_adv"] += 0.20
    if _clamp(features.get("pattern_without_structure_signal", 0.0)) >= 0.45:
        scores["heur"] += 0.18
        scores["ana"] += 0.12
        scores["ind"] += 0.20


def score_families(
    *,
    features: Dict[str, float],
    regime_label: str,
    allowed_families: List[str],
) -> Dict[str, float]:
    scores = _base_score_map(features)
    _regime_boosts(scores=scores, features=features, regime_label=regime_label)
    for family in allowed_families:
        scores.setdefault(family, 0.1)
    return {family: _clamp(scores.get(family, 0.1), 0.0, 3.0) for family in allowed_families}


def _should_activate_optional(
    *,
    family: str,
    features: Dict[str, float],
    regime_label: str,
    profile_name: str,
    allow_experimental: bool,
) -> bool:
    if family == "heur":
        return (
            regime_label.startswith("heterogeneous")
            or regime_label == "vram_favorable"
            or _clamp(features.get("edge_pressure", 0.0)) >= 0.58
            or _clamp(features.get("pattern_without_structure_signal", 0.0)) >= 0.45
        )
    if family == "dia_adv":
        return (
            regime_label == "viability_edge"
            or _clamp(features.get("ambiguity_signal", 0.0)) >= 0.55
            or _clamp(features.get("contradiction_signal", 0.0)) >= 0.45
        )
    if family == "fal_guard":
        return (
            regime_label == "viability_edge"
            or _clamp(features.get("fragility_risk_signal", 0.0)) >= 0.50
            or _clamp(features.get("contradiction_signal", 0.0)) >= 0.55
        )
    if family == "ind":
        return (
            profile_name == "full_family_exploration"
            and (
                regime_label == "vram_favorable"
                or _clamp(features.get("pattern_without_structure_signal", 0.0)) >= 0.40
            )
        )
    if family == "eml_sr":
        return (
            allow_experimental
            and (
                _clamp(features.get("symbolic_regularity", 0.0)) >= 0.40
                or _clamp(features.get("law_fit_signal", 0.0)) >= 0.40
            )
        )
    if family == "plan":
        return (
            regime_label == "viability_edge"
            or _clamp(features.get("viability_edge_signal", 0.0)) >= 0.60
        )
    if family == "opt":
        return (
            _clamp(features.get("causal_risk", 0.0)) >= 0.50
            or regime_label == "viability_edge"
        )
    return False


def _apply_overlay_directives(
    optional_families: List[str],
    overlay_directives: Dict[str, str] | None,
    *,
    allowed: List[str] | None = None,
) -> List[str]:
    """Aplica directivas on/off del selector guiado-por-recompensa.

    Solo gobierna familias opcionales: ``off`` las retira, ``on`` las fuerza
    si el perfil las permite (``allowed``). El núcleo nunca pasa por aquí.
    """
    if not overlay_directives:
        return optional_families
    normalized = {
        str(family).strip().lower(): str(action).strip().lower()
        for family, action in overlay_directives.items()
    }
    out = [family for family in optional_families if normalized.get(family) != "off"]
    allowed_set = set(allowed if allowed is not None else optional_families)
    for family, action in normalized.items():
        if action == "on" and family in allowed_set and family not in out:
            out.append(family)
    return out


def _sequence_for_non_adaptive_profile(
    *,
    profile_name: str,
    core_sequence: List[str],
    optional_families: List[str],
) -> List[str]:
    if profile_name in {
        "core_plus_external_reasoner",
        "core_plus_external_reasoner_guarded",
        "core_plus_deliberative",
        "core_plus_ind",
        "core_plus_plan",
        "core_plus_opt",
    }:
        sequence, _ = _compose_core_protected_sequence(
            overlays=optional_families,
            effective_max_steps=len(core_sequence) + len(optional_families),
            default_overlays=optional_families,
            scores={family: 1.0 for family in optional_families},
            trim_to_budget=False,
        )
        return sequence

    sequence = list(core_sequence)
    if not optional_families:
        return sequence
    insertion_index = 1 if sequence and sequence[0] == "abd" else 0
    for family in optional_families:
        sequence.insert(insertion_index, family)
        insertion_index += 1
    return _dedup(sequence)


def _contradiction_or_ambiguity_spike(features: Dict[str, float]) -> bool:
    return (
        _clamp(features.get("contradiction_signal", 0.0)) >= 0.45
        or _clamp(features.get("ambiguity_signal", 0.0)) >= 0.55
    )


def _fragility_spike(features: Dict[str, float]) -> bool:
    return (
        _clamp(features.get("fragility_risk_signal", 0.0)) >= 0.50
        or _clamp(features.get("contradiction_signal", 0.0)) >= 0.55
    )


def _should_activate_shadow(
    *,
    family: str,
    features: Dict[str, float],
    regime_label: str,
    profile_name: str,
    allow_experimental: bool,
) -> bool:
    if family == "ind":
        return (
            profile_name == "full_family_exploration"
            or (
                regime_label == "vram_favorable"
                and _clamp(features.get("pattern_without_structure_signal", 0.0)) >= 0.40
            )
        )
    if family == "eml_sr":
        return allow_experimental and (
            _clamp(features.get("symbolic_regularity", 0.0)) >= 0.40
            or _clamp(features.get("law_fit_signal", 0.0)) >= 0.40
        )
    if family == "plan":
        return (
            regime_label == "viability_edge"
            or _clamp(features.get("viability_edge_signal", 0.0)) >= 0.60
        )
    if family == "opt":
        return (
            _clamp(features.get("causal_risk", 0.0)) >= 0.50
            or regime_label == "viability_edge"
        )
    return False


def _admit_v2_overlays(
    *,
    profile_name: str,
    allowed_families: List[str],
    features: Dict[str, float],
    primary_regime_label: str,
    cognitive_regime_label: str,
    allow_experimental: bool,
    requested_max_steps: int,
) -> Dict[str, List[str]]:
    allowed = set(allowed_families)
    defaults: List[str] = []
    conditionals: List[str] = []
    shadows: List[str] = []

    def add_default(family: str) -> None:
        if family in allowed and family not in defaults:
            defaults.append(family)

    def add_conditional(family: str) -> None:
        if family in allowed and family not in defaults and family not in conditionals:
            conditionals.append(family)

    regime_for_defaults = cognitive_regime_label if primary_regime_label == "vram_favorable" else primary_regime_label
    if regime_for_defaults == "heterogeneous_elevated":
        add_default("heur")
    elif regime_for_defaults == "heterogeneous_warning":
        add_default("heur")
        add_default("fal_guard")
    elif regime_for_defaults == "viability_edge":
        add_default("fal_guard")
        add_default("dia_adv")

    if primary_regime_label == "vram_favorable":
        add_default("heur")

    if _contradiction_or_ambiguity_spike(features):
        add_conditional("dia_adv")
    if _fragility_spike(features):
        add_conditional("fal_guard")
    if (
        primary_regime_label == "viability_edge"
        and requested_max_steps >= len(CORE_SEQUENCE) + len(defaults) + 1
        and (
            _clamp(features.get("edge_pressure", 0.0)) >= 0.75
            or _clamp(features.get("pattern_without_structure_signal", 0.0)) >= 0.45
        )
    ):
        add_conditional("heur")
    if (
        primary_regime_label == "homogeneous_safe"
        and _clamp(features.get("edge_pressure", 0.0)) >= 0.75
    ):
        add_conditional("heur")

    for family in CONDITIONAL_SHADOW_FAMILIES:
        if family in allowed and _should_activate_shadow(
            family=family,
            features=features,
            regime_label=primary_regime_label,
            profile_name=profile_name,
            allow_experimental=allow_experimental,
        ):
            shadows.append(family)

    return {
        "default_overlays": _dedup(defaults),
        "conditional_overlays": _dedup(conditionals),
        "shadow_overlays": _dedup(shadows),
    }


def _insert_overlay(sequence: List[str], family: str) -> List[str]:
    seq = [item for item in sequence if item != family]
    if family == "heur":
        anchor = "abd"
    elif family == "ind":
        anchor = "heur" if "heur" in seq else "abd"
    elif family == "dia_adv":
        anchor = "ana"
    elif family == "eml_sr":
        anchor = "ctf"
    elif family == "ext_open_thinker":
        anchor = "ctf"
    elif family == "fal_guard":
        anchor = "ded"
    elif family == "plan":
        anchor = "ctf"
    elif family == "opt":
        anchor = "plan" if "plan" in seq else "ctf"
    else:
        anchor = "abd"

    if anchor in seq:
        insert_at = seq.index(anchor) + 1
    else:
        insert_at = 0
    seq.insert(insert_at, family)
    if "prob" in seq and seq[-1] != "prob":
        seq = [item for item in seq if item != "prob"] + ["prob"]
    return _dedup(seq)


def _compose_core_protected_sequence(
    *,
    overlays: Sequence[str],
    effective_max_steps: int,
    default_overlays: Sequence[str],
    scores: Dict[str, float],
    trim_to_budget: bool,
) -> tuple[List[str], Dict[str, Any]]:
    overlay_pool = _dedup(overlays)
    sequence = list(CORE_SEQUENCE)
    for family in OVERLAY_INSERTION_ORDER:
        if family in overlay_pool:
            sequence = _insert_overlay(sequence, family)

    dropped_for_budget: List[str] = []
    if trim_to_budget and len(sequence) > effective_max_steps:
        # Deliberativas se descartan primero junto a las shadow si no hay budget.
        shadow_drop = [
            family
            for family in list(CONDITIONAL_SHADOW_FAMILIES) + list(DELIBERATIVE_FAMILIES)
            if family in overlay_pool
        ]
        conditional_drop = [
            family for family in overlay_pool
            if family in AUGMENTER_FAMILIES and family not in set(default_overlays)
        ]
        default_drop = [family for family in default_overlays if family in overlay_pool]
        drop_priority = shadow_drop + sorted(
            conditional_drop,
            key=lambda family: (scores.get(family, 0.0), family),
        ) + sorted(
            default_drop,
            key=lambda family: (scores.get(family, 0.0), family),
        )

        retained = list(overlay_pool)
        for family in drop_priority:
            if len(CORE_SEQUENCE) + len(retained) <= effective_max_steps:
                break
            if family in retained:
                retained.remove(family)
                dropped_for_budget.append(family)
        sequence = list(CORE_SEQUENCE)
        for family in OVERLAY_INSERTION_ORDER:
            if family in retained:
                sequence = _insert_overlay(sequence, family)

    return sequence, {
        "overlay_pool": overlay_pool,
        "dropped_for_budget": dropped_for_budget,
    }


def _validate_partial_order(
    sequence: Sequence[str],
    *,
    enforce_overlay_anchors: bool,
) -> bool:
    index = {family: idx for idx, family in enumerate(sequence)}
    backbone = list(BACKBONE_FAMILIES)
    for earlier_idx, earlier in enumerate(backbone):
        if earlier not in index:
            continue
        for later in backbone[earlier_idx + 1 :]:
            if later in index and index[earlier] >= index[later]:
                return False

    if not enforce_overlay_anchors:
        return True

    if "heur" in index:
        if "abd" in index and index["heur"] <= index["abd"]:
            return False
        if "ana" in index and index["heur"] >= index["ana"]:
            return False
    if "ind" in index:
        anchor = "heur" if "heur" in index else "abd"
        if anchor in index and index["ind"] <= index[anchor]:
            return False
        if "ana" in index and index["ind"] >= index["ana"]:
            return False
    if "dia_adv" in index:
        if "ana" in index and index["dia_adv"] <= index["ana"]:
            return False
        if "cau" in index and index["dia_adv"] >= index["cau"]:
            return False
    if "eml_sr" in index:
        if "ctf" in index and index["eml_sr"] <= index["ctf"]:
            return False
        if "ded" in index and index["eml_sr"] >= index["ded"]:
            return False
    if "ext_open_thinker" in index:
        if "ctf" in index and index["ext_open_thinker"] <= index["ctf"]:
            return False
        if "ded" in index and index["ext_open_thinker"] >= index["ded"]:
            return False
    if "fal_guard" in index:
        if "ded" in index and index["fal_guard"] <= index["ded"]:
            return False
        if "prob" in index and index["fal_guard"] >= index["prob"]:
            return False
    if "plan" in index:
        if "ctf" in index and index["plan"] <= index["ctf"]:
            return False
        if "prob" in index and index["plan"] >= index["prob"]:
            return False
    if "opt" in index:
        opt_anchor = "plan" if "plan" in index else "ctf"
        if opt_anchor in index and index["opt"] <= index[opt_anchor]:
            return False
        if "prob" in index and index["opt"] >= index["prob"]:
            return False
    return True


def _evaluate_sequence(
    *,
    sequence: Sequence[str],
    mandatory_floor: Sequence[str],
    allowed_families: Sequence[str],
    effective_max_steps: int,
    enforce_overlay_anchors: bool,
) -> Dict[str, Any]:
    normalized = _dedup(sequence)
    allowed = set(allowed_families)
    active_optionals = [
        family for family in normalized
        if family not in set(BACKBONE_FAMILIES)
    ]
    missing_floor = [family for family in mandatory_floor if family not in normalized]
    missing_core = [family for family in CORE_SEQUENCE if family not in normalized]
    unknown_families = [family for family in normalized if family not in allowed]
    prob_last_ok = "prob" not in normalized or normalized[-1] == "prob"
    partial_order_ok = _validate_partial_order(
        normalized,
        enforce_overlay_anchors=enforce_overlay_anchors,
    )
    length_ok = len(CORE_SEQUENCE) <= len(normalized) <= effective_max_steps
    optional_displacement_detected = bool(active_optionals) and bool(missing_floor or missing_core)

    passed = (
        not missing_floor
        and not missing_core
        and not unknown_families
        and partial_order_ok
        and prob_last_ok
        and length_ok
    )
    return {
        "sequence": normalized,
        "missing_floor": missing_floor,
        "missing_core": missing_core,
        "unknown_families": unknown_families,
        "partial_order_ok": partial_order_ok,
        "prob_last_ok": prob_last_ok,
        "length_ok": length_ok,
        "optional_displacement_detected": optional_displacement_detected,
        "passed": passed,
    }


def _fallback_v2_sequence(
    *,
    primary_regime_label: str,
    floor_regime_label: str,
    allowed_families: Sequence[str],
    effective_max_steps: int,
    allow_experimental: bool,
    features: Dict[str, float],
    scores: Dict[str, float],
) -> tuple[str, List[str], List[str]]:
    fallback_profile_name = safe_fallback_profile(floor_regime_label)
    fallback_profile = resolve_family_profile(fallback_profile_name, mode="fixed")
    fallback_optional = [family for family in fallback_profile.optional_families if family in set(allowed_families)]
    if primary_regime_label == "vram_favorable" and "heur" in allowed_families and "heur" not in fallback_optional:
        fallback_optional.append("heur")
    if allow_experimental and "eml_sr" in allowed_families and _should_activate_shadow(
        family="eml_sr",
        features=features,
        regime_label="vram_favorable",
        profile_name="adaptive_family_ecology_v2",
        allow_experimental=allow_experimental,
    ):
        fallback_optional.append("eml_sr")

    fallback_sequence, _ = _compose_core_protected_sequence(
        overlays=fallback_optional,
        effective_max_steps=effective_max_steps,
        default_overlays=fallback_optional,
        scores=scores,
        trim_to_budget=True,
    )
    return fallback_profile_name, fallback_sequence, _dedup(fallback_optional)


def validate_reasoning_sequence(
    *,
    proposed_sequence: Sequence[str],
    allowed_families: Sequence[str],
    requested_max_steps: int,
    primary_regime_label: str,
    cognitive_regime_label: str,
    floor_regime_label: str,
    mandatory_family_floor: Sequence[str],
    default_overlays: Sequence[str],
    admitted_overlays: Sequence[str],
    scores: Dict[str, float],
    allow_experimental: bool,
    features: Dict[str, float],
    enforce_overlay_anchors: bool = True,
) -> Dict[str, Any]:
    effective_max_steps = max(len(CORE_SEQUENCE), min(HARD_MAX_STEPS, int(requested_max_steps)))
    budget_overridden_by_floor = int(requested_max_steps) < len(CORE_SEQUENCE)

    proposed_eval = _evaluate_sequence(
        sequence=proposed_sequence,
        mandatory_floor=mandatory_family_floor,
        allowed_families=allowed_families,
        effective_max_steps=effective_max_steps,
        enforce_overlay_anchors=enforce_overlay_anchors,
    )

    correction_steps: List[str] = []
    validated_sequence = list(proposed_eval["sequence"])
    fallback_used = False
    fallback_profile_name = None

    if not proposed_eval["passed"]:
        retained_overlays = [
            family for family in proposed_eval["sequence"]
            if family not in set(BACKBONE_FAMILIES) and family in set(allowed_families)
        ]
        validated_sequence, correction_meta = _compose_core_protected_sequence(
            overlays=retained_overlays,
            effective_max_steps=effective_max_steps,
            default_overlays=default_overlays,
            scores=scores,
            trim_to_budget=True,
        )
        if correction_meta["dropped_for_budget"]:
            correction_steps.append(
                "budget_trim:" + ",".join(_upper_list(correction_meta["dropped_for_budget"]))
            )
        if proposed_eval["missing_core"]:
            correction_steps.append(
                "restore_core:" + ",".join(_upper_list(proposed_eval["missing_core"]))
            )
        if proposed_eval["missing_floor"]:
            correction_steps.append(
                "restore_floor:" + ",".join(_upper_list(proposed_eval["missing_floor"]))
            )
        if proposed_eval["unknown_families"]:
            correction_steps.append(
                "drop_unknown:" + ",".join(_upper_list(proposed_eval["unknown_families"]))
            )
        if not correction_steps:
            correction_steps.append("recompose_core_protected_sequence")

    validated_eval = _evaluate_sequence(
        sequence=validated_sequence,
        mandatory_floor=mandatory_family_floor,
        allowed_families=allowed_families,
        effective_max_steps=effective_max_steps,
        enforce_overlay_anchors=enforce_overlay_anchors,
    )

    if not validated_eval["passed"]:
        fallback_profile_name, fallback_sequence, fallback_overlays = _fallback_v2_sequence(
            primary_regime_label=primary_regime_label,
            floor_regime_label=floor_regime_label,
            allowed_families=allowed_families,
            effective_max_steps=effective_max_steps,
            allow_experimental=allow_experimental,
            features=features,
            scores=scores,
        )
        validated_sequence = fallback_sequence
        validated_eval = _evaluate_sequence(
            sequence=validated_sequence,
            mandatory_floor=mandatory_family_floor,
            allowed_families=allowed_families,
            effective_max_steps=effective_max_steps,
            enforce_overlay_anchors=enforce_overlay_anchors,
        )
        fallback_used = True
        correction_steps.append(
            "fallback_safe_profile:" + fallback_profile_name
        )
        admitted_overlays = list(fallback_overlays)

    return {
        "primary_regime_label": primary_regime_label,
        "cognitive_regime_label": cognitive_regime_label,
        "floor_regime_label": floor_regime_label,
        "mandatory_family_floor": _upper_list(mandatory_family_floor),
        "proposed_sequence": _upper_list(proposed_eval["sequence"]),
        "validated_sequence": _upper_list(validated_eval["sequence"]),
        "proposed_passed": proposed_eval["passed"],
        "validated_passed": validated_eval["passed"],
        "missing_floor": _upper_list(proposed_eval["missing_floor"]),
        "missing_core": _upper_list(proposed_eval["missing_core"]),
        "partial_order_ok": proposed_eval["partial_order_ok"],
        "prob_last_ok": proposed_eval["prob_last_ok"],
        "length_ok": proposed_eval["length_ok"],
        "optional_displacement_detected": proposed_eval["optional_displacement_detected"],
        "autocorrected": _upper_list(proposed_eval["sequence"]) != _upper_list(validated_eval["sequence"]),
        "fallback_used": fallback_used,
        "budget_overridden_by_floor": budget_overridden_by_floor,
        "effective_max_steps": effective_max_steps,
        "unknown_families": _upper_list(proposed_eval["unknown_families"]),
        "admitted_overlays": _upper_list(admitted_overlays),
        "default_overlays": _upper_list(default_overlays),
        "correction_steps": correction_steps,
        "fallback_profile_name": fallback_profile_name,
    }


def _build_v2_sequence(
    *,
    features: Dict[str, float],
    allowed_families: List[str],
    requested_max_steps: int,
    regime_labels: Dict[str, str],
    allow_experimental: bool,
    scores: Dict[str, float],
) -> tuple[List[str], Dict[str, Any]]:
    overlays_by_role = _admit_v2_overlays(
        profile_name="adaptive_family_ecology_v2",
        allowed_families=allowed_families,
        features=features,
        primary_regime_label=regime_labels["primary_regime_label"],
        cognitive_regime_label=regime_labels["cognitive_regime_label"],
        allow_experimental=allow_experimental,
        requested_max_steps=requested_max_steps,
    )
    admitted_overlays = _dedup(
        overlays_by_role["default_overlays"]
        + overlays_by_role["conditional_overlays"]
        + overlays_by_role["shadow_overlays"]
    )
    protected_min_steps = min(
        HARD_MAX_STEPS,
        max(
            requested_max_steps,
            len(CORE_SEQUENCE) + len(admitted_overlays),
        ),
    )
    proposed_sequence, _ = _compose_core_protected_sequence(
        overlays=admitted_overlays,
        effective_max_steps=max(len(CORE_SEQUENCE), protected_min_steps),
        default_overlays=overlays_by_role["default_overlays"],
        scores=scores,
        trim_to_budget=False,
    )
    validation = validate_reasoning_sequence(
        proposed_sequence=proposed_sequence,
        allowed_families=allowed_families,
        requested_max_steps=protected_min_steps,
        primary_regime_label=regime_labels["primary_regime_label"],
        cognitive_regime_label=regime_labels["cognitive_regime_label"],
        floor_regime_label=regime_labels["floor_regime_label"],
        mandatory_family_floor=mandatory_family_floor(regime_labels["floor_regime_label"]),
        default_overlays=overlays_by_role["default_overlays"],
        admitted_overlays=admitted_overlays,
        scores=scores,
        allow_experimental=allow_experimental,
        features=features,
        enforce_overlay_anchors=True,
    )
    return [family.lower() for family in validation["validated_sequence"]], validation


def _legacy_adaptive_sequence(
    *,
    profile_name: str,
    core_sequence: List[str],
    optional_families: List[str],
    features: Dict[str, float],
    regime_label: str,
    allow_experimental: bool,
    scores: Dict[str, float],
    max_steps: int,
    overlay_directives: Dict[str, str] | None = None,
) -> List[str]:
    max_steps = max(max_steps, len(core_sequence))
    core = list(core_sequence)
    optional_candidates = list(optional_families)
    forced_on = {
        str(family).strip().lower()
        for family, action in (overlay_directives or {}).items()
        if str(action).strip().lower() == "on"
    }
    activated_optional: List[str] = []
    if profile_name == "full_family_exploration":
        activated_optional = sorted(
            optional_candidates,
            key=lambda family: (-scores.get(family, 0.0), family),
        )
    else:
        for family in optional_candidates:
            if family in forced_on or _should_activate_optional(
                family=family,
                features=features,
                regime_label=regime_label,
                profile_name=profile_name,
                allow_experimental=allow_experimental,
            ):
                activated_optional.append(family)
        activated_optional = sorted(
            activated_optional,
            key=lambda family: (-scores.get(family, 0.0), family),
        )
    sequence = _dedup([core[0]] + activated_optional + core[1:]) if core else _dedup(activated_optional)
    ranked_remaining = [
        family for family in sorted(optional_candidates + core, key=lambda f: (-scores.get(f, 0.0), f))
        if family not in sequence
    ]
    sequence.extend(ranked_remaining)
    sequence = _dedup(sequence)[:max_steps]
    if "prob" in core:
        if "prob" not in sequence and sequence:
            sequence[-1] = "prob"
        elif "prob" in sequence and sequence[-1] != "prob":
            sequence = [family for family in sequence if family != "prob"] + ["prob"]
            sequence = sequence[:max_steps]
            if sequence and sequence[-1] != "prob":
                sequence[-1] = "prob"
    return sequence


def select_sequence(
    *,
    features: Dict[str, float],
    budget: Dict[str, float],
    allow_experimental: bool = False,
    mode: str = "adaptive",
    profile_name: str | None = None,
    regime_hint: str | None = None,
    return_metadata: bool = False,
    overlay_directives: Dict[str, str] | None = None,
) -> Tuple[List[str], Dict[str, float], str] | Tuple[List[str], Dict[str, float], str, Dict[str, Any]]:
    active_mode = (mode or "fixed").strip().lower()
    profile = resolve_family_profile(profile_name, mode=active_mode)
    if "ext_open_thinker" in profile.optional_families:
        raise ValueError(
            "external_reasoner_profile_not_nominal:"
            f"{profile.name}:"
            f"status={EXT_OPEN_THINKER_ADMISSION.nominal_status}:"
            "use core_plus_external_reasoner_gated_v1 only in lab benchmark"
        )
    regime_labels = resolve_regime_labels(regime_hint=regime_hint, features=features)
    requested_max_steps = max(1, int(_clamp(budget.get("max_steps", 6), 1.0, 32.0)))

    allowed = list(profile.allowed_families)
    if not allow_experimental and "eml_sr" in allowed:
        allowed = [family for family in allowed if family != "eml_sr"]

    scores = score_families(
        features=features,
        regime_label=regime_labels["primary_regime_label"],
        allowed_families=allowed,
    )

    if active_mode != "adaptive" and not profile.adaptive:
        effective_optional = _apply_overlay_directives(
            [fam for fam in profile.optional_families if fam in allowed],
            overlay_directives,
            allowed=[fam for fam in profile.optional_families if fam in allowed],
        )
        effective_max_steps = max(requested_max_steps, len(profile.core_sequence) + len(effective_optional))
        sequence = _sequence_for_non_adaptive_profile(
            profile_name=profile.name,
            core_sequence=profile.core_sequence,
            optional_families=effective_optional,
        )
        validation = validate_reasoning_sequence(
            proposed_sequence=sequence,
            allowed_families=allowed,
            requested_max_steps=effective_max_steps,
            primary_regime_label=regime_labels["primary_regime_label"],
            cognitive_regime_label=regime_labels["cognitive_regime_label"],
            floor_regime_label=regime_labels["floor_regime_label"],
            mandatory_family_floor=mandatory_family_floor(regime_labels["floor_regime_label"]),
            default_overlays=effective_optional,
            admitted_overlays=effective_optional,
            scores=scores,
            allow_experimental=allow_experimental,
            features=features,
            enforce_overlay_anchors=profile.name in {
                "core_plus_external_reasoner",
                "core_plus_external_reasoner_guarded",
            },
        )
    elif profile.name == "adaptive_family_ecology_v2":
        sequence, validation = _build_v2_sequence(
            features=features,
            allowed_families=allowed,
            requested_max_steps=requested_max_steps,
            regime_labels=regime_labels,
            allow_experimental=allow_experimental,
            scores=scores,
        )
    else:
        effective_max_steps = max(requested_max_steps, len(profile.core_sequence))
        sequence = _legacy_adaptive_sequence(
            profile_name=profile.name,
            core_sequence=profile.core_sequence,
            optional_families=_apply_overlay_directives(
                [family for family in profile.optional_families if family in allowed],
                overlay_directives,
                allowed=[family for family in profile.optional_families if family in allowed],
            ),
            features=features,
            regime_label=regime_labels["primary_regime_label"],
            allow_experimental=allow_experimental,
            scores=scores,
            max_steps=effective_max_steps,
            overlay_directives=overlay_directives,
        )
        validation = validate_reasoning_sequence(
            proposed_sequence=sequence,
            allowed_families=allowed,
            requested_max_steps=effective_max_steps,
            primary_regime_label=regime_labels["primary_regime_label"],
            cognitive_regime_label=regime_labels["cognitive_regime_label"],
            floor_regime_label=regime_labels["floor_regime_label"],
            mandatory_family_floor=mandatory_family_floor(regime_labels["floor_regime_label"]),
            default_overlays=[],
            admitted_overlays=[
                family for family in sequence if family not in set(BACKBONE_FAMILIES)
            ],
            scores=scores,
            allow_experimental=allow_experimental,
            features=features,
            enforce_overlay_anchors=False,
        )

    sequence = _dedup(sequence)
    remaining = [family for family in allowed if family not in sequence]
    recommended_next = (
        sorted(remaining, key=lambda family: (-scores.get(family, 0.0), family))[0]
        if remaining
        else "prob"
    )

    metadata = {
        "profile_name": profile.name,
        "regime_label": regime_labels["primary_regime_label"],
        "allowed_families": allowed,
        "mode": active_mode,
        **regime_labels,
        "mandatory_family_floor": validation["mandatory_family_floor"],
        "proposed_sequence": validation["proposed_sequence"],
        "validated_sequence": validation["validated_sequence"],
        "sequence_validation": validation,
        "effective_max_steps": validation["effective_max_steps"],
    }
    if return_metadata:
        return sequence, scores, recommended_next, metadata
    return sequence, scores, recommended_next


def is_eml_experimental_enabled() -> bool:
    mode = os.environ.get("RNFE_EML_MODE", "disabled").strip().lower()
    if mode != "shadow":
        return False
    allowlist = os.environ.get("RNFE_META_EXPERIMENTAL_FAMILIES", "")
    enabled = {item.strip().lower() for item in allowlist.split(",") if item.strip()}
    return "eml_sr" in enabled
