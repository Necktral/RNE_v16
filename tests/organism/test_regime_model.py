"""Tests para runtime/organism/regime_model.py — Modelo de regímenes latentes."""

from __future__ import annotations

import pytest

from runtime.organism.regime_model import (
    RegimeModel,
    RegimeComparisonResult,
    compare_regimes,
    THERMAL_REGIME,
    RESOURCE_REGIME,
    SCENARIO_TO_REGIME,
    get_regime_for_scenario,
)


class TestRegimeModel:
    def test_thermal_regime(self):
        assert THERMAL_REGIME.regime_id == "homeostatic_cooling"
        assert THERMAL_REGIME.optimization_geometry == "minimize"
        assert "thermal_homeostasis" in THERMAL_REGIME.scenario_instances

    def test_resource_regime(self):
        assert RESOURCE_REGIME.regime_id == "inventory_maximization"
        assert RESOURCE_REGIME.optimization_geometry == "maximize"
        assert "resource_management" in RESOURCE_REGIME.scenario_instances


class TestCompareRegimes:
    def test_same_regime(self):
        r = compare_regimes(THERMAL_REGIME, THERMAL_REGIME)
        assert r.compatibility == "same_regime"
        assert r.structural_distance == 0.0
        assert r.transport_feasibility > 0.5

    def test_thermal_vs_resource(self):
        r = compare_regimes(THERMAL_REGIME, RESOURCE_REGIME)
        # Different optimization geometry and polarity
        assert r.geometry_match is False
        assert r.polarity_match is False
        # Still same topology
        assert r.topology_match is True
        assert r.compatibility in ("compatible_regime", "transformable_regime", "non_transportable")

    def test_asymmetric_comparison(self):
        ab = compare_regimes(THERMAL_REGIME, RESOURCE_REGIME)
        ba = compare_regimes(RESOURCE_REGIME, THERMAL_REGIME)
        # Structural distance should be symmetric (same properties)
        assert ab.structural_distance == ba.structural_distance
        # But order of source/target differs
        assert ab.source_regime != ba.source_regime

    def test_transport_feasibility_range(self):
        r = compare_regimes(THERMAL_REGIME, RESOURCE_REGIME)
        assert 0.0 <= r.transport_feasibility <= 1.0

    def test_custom_regime_compatible(self):
        """Two regimes with identical structure should be compatible."""
        r1 = RegimeModel(
            regime_id="custom_a",
            control_topology="cascade",
            optimization_geometry="target_band",
        )
        r2 = RegimeModel(
            regime_id="custom_b",
            control_topology="cascade",
            optimization_geometry="target_band",
        )
        comp = compare_regimes(r1, r2)
        assert comp.compatibility == "compatible_regime"
        assert comp.structural_distance < 0.3


class TestGetRegimeForScenario:
    def test_known_scenarios(self):
        thermal = get_regime_for_scenario("thermal_homeostasis")
        assert thermal is not None
        assert thermal.regime_id == "homeostatic_cooling"

        resource = get_regime_for_scenario("resource_management")
        assert resource is not None
        assert resource.regime_id == "inventory_maximization"

    def test_unknown_scenario(self):
        assert get_regime_for_scenario("nonexistent") is None

    def test_scenario_to_regime_mapping(self):
        assert "thermal_homeostasis" in SCENARIO_TO_REGIME
        assert "resource_management" in SCENARIO_TO_REGIME
