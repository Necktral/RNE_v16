"""Política adaptativa y explicable para selección de familias."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple


FAMILY_POOL = [
    "abd",
    "ana",
    "cau",
    "ctf",
    "ded",
    "prob",
    "dia_adv",
    "heur",
    "fal_guard",
    "eml_sr",
]


def score_families(features: Dict[str, float]) -> Dict[str, float]:
    scores = {family: 0.1 for family in FAMILY_POOL}
    scores["abd"] += 0.3
    scores["ana"] += 0.2 + (0.2 * features["uncertainty"])
    scores["cau"] += 0.2 + (0.2 * features["causal_risk"])
    scores["ctf"] += 0.2 + (0.2 * features["causal_risk"])
    scores["ded"] += 0.2 + (0.2 * (1.0 - features["continuity_recent"]))
    scores["prob"] += 0.2 + (0.25 * features["uncertainty"])
    scores["heur"] += 0.1 + (0.35 * features["edge_pressure"])
    scores["dia_adv"] += 0.1 + (0.35 * features["contradiction_signal"])
    scores["fal_guard"] += 0.1 + (0.25 * features["contradiction_signal"])
    scores["eml_sr"] += (
        0.1
        + (0.3 * features.get("symbolic_regularity", 0.0))
        + (0.25 * features.get("law_fit_signal", 0.0))
    )
    return scores


def _dedup(sequence: List[str]) -> List[str]:
    out: List[str] = []
    for family in sequence:
        if family not in out:
            out.append(family)
    return out


def select_sequence(
    *,
    features: Dict[str, float],
    budget: Dict[str, float],
    allow_experimental: bool = False,
) -> Tuple[List[str], Dict[str, float], str]:
    scores = score_families(features)
    max_steps = int(budget["max_steps"])
    sequence: List[str] = ["abd"]
    if features["edge_pressure"] >= 0.7:
        sequence.append("heur")
    if features["contradiction_signal"] >= 0.45:
        sequence.extend(["dia_adv", "fal_guard"])
    if allow_experimental and (
        features.get("symbolic_regularity", 0.0) >= 0.4
        or features.get("law_fit_signal", 0.0) >= 0.4
    ):
        sequence.append("eml_sr")
    sequence.extend(["ana", "cau", "ctf", "ded", "prob"])

    ranked = sorted(
        [fam for fam in FAMILY_POOL if fam not in sequence],
        key=lambda fam: (-scores[fam], fam),
    )
    sequence.extend(ranked)
    sequence = _dedup(sequence)
    sequence = sequence[:max_steps]

    if "prob" not in sequence:
        sequence[-1] = "prob"
    elif sequence[-1] != "prob":
        sequence = [fam for fam in sequence if fam != "prob"] + ["prob"]

    remaining = [fam for fam in FAMILY_POOL if fam not in sequence]
    if remaining:
        recommended_next = sorted(remaining, key=lambda fam: (-scores[fam], fam))[0]
    else:
        recommended_next = "prob"
    return sequence, scores, recommended_next


def is_eml_experimental_enabled() -> bool:
    mode = os.environ.get("RNFE_EML_MODE", "disabled").strip().lower()
    if mode != "shadow":
        return False
    allowlist = os.environ.get("RNFE_META_EXPERIMENTAL_FAMILIES", "")
    enabled = {item.strip().lower() for item in allowlist.split(",") if item.strip()}
    return "eml_sr" in enabled
