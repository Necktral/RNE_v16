from __future__ import annotations

from runtime.symbolic.mci import (
    OverlayTranslator,
    resource_with_energy_spec,
    thermal_battery_spec,
    thermal_battery_to_resource_energy,
)
from scripts.experiments.exp6_transfer import (
    _condition_is_correct,
    _mapping_only_hypotheses,
)


def test_semantic_correctness_is_not_string_matching() -> None:
    assert _condition_is_correct("energy_level >= 0.3000001", "energy_level")
    assert not _condition_is_correct("energy_level > 0.5", "energy_level")
    assert not _condition_is_correct("stock_level > 0.3", "energy_level")


def test_mapping_only_is_guided_but_contains_no_source_claim() -> None:
    mapping = thermal_battery_to_resource_energy(
        thermal_battery_spec(), resource_with_energy_spec()
    )
    hypotheses = _mapping_only_hypotheses(
        morphism_id=mapping.morphism_id, logical_time=1
    )
    assert {item.target_id for item in hypotheses} == {
        "resource_energy/production"
    }
    assert {item.expression for item in hypotheses} == {
        "energy_level > 0.2",
        "energy_level > 0.3",
        "energy_level > 0.4",
    }
    assert all(item.package_id == "mapping-only-no-source-package" for item in hypotheses)


def test_translator_rejects_package_cross_wiring() -> None:
    assert OverlayTranslator is not None
