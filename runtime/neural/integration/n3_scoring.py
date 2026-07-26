"""Single source of truth for bounded N3 memory score modulation."""

from __future__ import annotations

import math
from typing import Any, Mapping

SCALES = ("micro", "meso", "macro")


def n3_scale_multiplier(scale: str, signals: Mapping[str, float]) -> float:
    if scale not in SCALES or set(signals) - set(SCALES):
        raise ValueError("n3_scale_signal_invalid")
    value = float(signals.get(scale, 0.0))
    if not math.isfinite(value):
        raise ValueError("n3_scale_signal_nonfinite")
    return 0.75 + 0.25 * max(-1.0, min(1.0, value))


def n3_adjusted_score(candidate: Mapping[str, Any], signals: Mapping[str, float]) -> float:
    raw = candidate.get("canonical_score", candidate.get("score", 0.0))
    score = float(raw) * n3_scale_multiplier(str(candidate.get("scale")), signals)
    if not math.isfinite(score):
        raise ValueError("n3_adjusted_score_nonfinite")
    return score
