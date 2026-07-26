"""Escalas canónicas compartidas por planificación y transferencia."""

from __future__ import annotations

import math
from dataclasses import dataclass


def normalize_range(value: float, lower: float, upper: float) -> float:
    value, lower, upper = float(value), float(lower), float(upper)
    if not all(math.isfinite(item) for item in (value, lower, upper)):
        raise ValueError("scale_values_must_be_finite")
    if upper <= lower:
        raise ValueError("scale_range_must_be_non_degenerate")
    return max(0.0, min(1.0, (value - lower) / (upper - lower)))


@dataclass(frozen=True)
class ScaleTransform:
    kind: str
    source_lower: float = 0.0
    source_upper: float = 1.0
    target_lower: float = 0.0
    target_upper: float = 1.0
    source_reference: float | None = None
    target_reference: float | None = None
    slope: float | None = None
    intercept: float | None = None

    def __post_init__(self) -> None:
        allowed = {
            "identity",
            "affine",
            "range_normalized",
            "threshold_percentile",
            "magnitude_ratio",
        }
        if self.kind not in allowed:
            raise ValueError("scale_kind_invalid")
        values = (
            self.source_lower,
            self.source_upper,
            self.target_lower,
            self.target_upper,
            self.source_reference,
            self.target_reference,
            self.slope,
            self.intercept,
        )
        if any(item is not None and not math.isfinite(float(item)) for item in values):
            raise ValueError("scale_values_must_be_finite")
        if self.kind in {"range_normalized", "threshold_percentile"} and (
            self.source_upper <= self.source_lower
            or self.target_upper <= self.target_lower
        ):
            raise ValueError("scale_range_must_be_non_degenerate")
        if self.kind == "affine" and (self.slope is None or self.intercept is None):
            raise ValueError("affine_scale_requires_slope_and_intercept")
        if self.kind == "magnitude_ratio" and (
            self.source_reference in {None, 0.0} or self.target_reference is None
        ):
            raise ValueError("magnitude_ratio_requires_references")

    def apply(self, value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("scale_values_must_be_finite")
        if self.kind == "identity":
            return value
        if self.kind == "affine":
            assert self.slope is not None and self.intercept is not None
            return self.slope * value + self.intercept
        if self.kind in {"range_normalized", "threshold_percentile"}:
            position = normalize_range(value, self.source_lower, self.source_upper)
            return self.target_lower + position * (
                self.target_upper - self.target_lower
            )
        assert self.source_reference is not None
        assert self.target_reference is not None
        return value / self.source_reference * self.target_reference
