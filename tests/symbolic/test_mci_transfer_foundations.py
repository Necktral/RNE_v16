from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from runtime.symbolic.mci import (
    ScaleTransform,
    resource_with_energy_spec,
    thermal_battery_spec,
    thermal_battery_to_resource_energy,
)


@pytest.mark.parametrize(
    ("transform", "value", "expected"),
    [
        (ScaleTransform("identity"), 0.3, 0.3),
        (ScaleTransform("affine", slope=2.0, intercept=0.1), 0.3, 0.7),
        (
            ScaleTransform(
                "range_normalized",
                source_lower=0.0,
                source_upper=10.0,
                target_lower=0.0,
                target_upper=1.0,
            ),
            5.0,
            0.5,
        ),
        (ScaleTransform("threshold_percentile"), 0.3, 0.3),
        (
            ScaleTransform(
                "magnitude_ratio",
                source_reference=0.07,
                target_reference=0.08,
            ),
            0.07,
            0.08,
        ),
    ],
)
def test_scale_transforms(transform, value, expected):
    assert transform.apply(value) == pytest.approx(expected)


def test_scale_validation_rejects_invalid_contracts():
    with pytest.raises(ValueError):
        ScaleTransform("range_normalized", source_lower=1.0, source_upper=1.0)
    with pytest.raises(ValueError):
        ScaleTransform("affine")
    with pytest.raises(ValueError):
        ScaleTransform("magnitude_ratio", source_reference=0.0, target_reference=1.0)


def test_known_morphism_validates_and_hashes_deterministically():
    source = thermal_battery_spec()
    target = resource_with_energy_spec()
    first = thermal_battery_to_resource_energy(source, target)
    second = thermal_battery_to_resource_energy(source, target)
    assert first.sha256 == second.sha256
    assert first.objective_relation == "inverse"
    assert first.effect_map["thermal_battery/cooling"] == "resource_energy/production"
    with pytest.raises(FrozenInstanceError):
        first.version = "2"


def test_morphism_rejects_hash_and_type_divergence():
    source = thermal_battery_spec()
    target = resource_with_energy_spec()
    morphism = thermal_battery_to_resource_energy(source, target)
    with pytest.raises(ValueError, match="target_hash"):
        replace(morphism, target_spec_sha256="0" * 64).validate(source, target)
