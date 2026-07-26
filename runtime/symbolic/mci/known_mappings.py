"""Morfismos certificados construidos desde hashes reales."""

from __future__ import annotations

from .contracts import TransitionSpec
from .scales import ScaleTransform
from .structural_morphism import StructuralMorphism


def thermal_battery_to_resource_energy(
    source: TransitionSpec,
    target: TransitionSpec,
) -> StructuralMorphism:
    morphism = StructuralMorphism(
        morphism_id="thermal_battery_to_resource_energy_v1",
        version="1",
        source_spec_id=source.spec_id,
        target_spec_id=target.spec_id,
        source_spec_sha256=source.sha256,
        target_spec_sha256=target.sha256,
        variable_map={
            "temperature": "stock_level",
            "battery_level": "energy_level",
            "cooling_active": "production_active",
            "alarm": "scarcity_alert",
        },
        action_map={
            "activate_cooling": "start_production",
            "deactivate_cooling": "stop_production",
        },
        parameter_map={
            "alarm_threshold": "scarcity_threshold",
            "cooling_effect": "production_rate",
        },
        effect_map={
            "thermal_battery/cooling": "resource_energy/production",
            "thermal_battery/active_consumption": "resource_energy/production_cost",
            "thermal_battery/passive_recharge": "resource_energy/recovery",
        },
        alarm_map={"alarm": "scarcity_alert"},
        invariant_map={
            "temperature": "stock_level",
            "battery_level": "energy_level",
        },
        objective_relation="inverse",
        causal_polarity_map={
            "thermal_battery/cooling": "inverse",
            "thermal_battery/active_consumption": "same",
            "thermal_battery/passive_recharge": "same",
        },
        scale_transforms={
            "temperature": ScaleTransform("range_normalized"),
            "battery_level": ScaleTransform("threshold_percentile"),
            "alarm_threshold": ScaleTransform("threshold_percentile"),
            "cooling_effect": ScaleTransform(
                "magnitude_ratio",
                source_reference=float(source.parameters["cooling_effect"]),
                target_reference=float(target.parameters["production_rate"]),
            ),
        },
        evidence={
            "basis": "typed_isomorphism",
            "source_profile": "threshold_recovery_loop",
            "target_profile": "threshold_recovery_loop",
        },
    )
    morphism.validate(source, target)
    return morphism
