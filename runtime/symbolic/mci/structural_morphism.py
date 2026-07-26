"""Morfismos tipados y verificables entre Transition IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from runtime.symbolic.schemas import _freeze, sealed_sha256

from .contracts import TransitionSpec
from .scales import ScaleTransform


@dataclass(frozen=True)
class StructuralMorphism:
    morphism_id: str
    version: str
    source_spec_id: str
    target_spec_id: str
    source_spec_sha256: str
    target_spec_sha256: str
    variable_map: Mapping[str, str]
    action_map: Mapping[str, str]
    parameter_map: Mapping[str, str]
    effect_map: Mapping[str, str]
    alarm_map: Mapping[str, str]
    invariant_map: Mapping[str, str]
    objective_relation: str
    causal_polarity_map: Mapping[str, str]
    scale_transforms: Mapping[str, ScaleTransform]
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "variable_map",
            "action_map",
            "parameter_map",
            "effect_map",
            "alarm_map",
            "invariant_map",
            "causal_polarity_map",
            "scale_transforms",
            "evidence",
        ):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        if self.objective_relation not in {"same", "inverse"}:
            raise ValueError("objective_relation_invalid")
        if any(
            item not in {"same", "inverse"}
            for item in self.causal_polarity_map.values()
        ):
            raise ValueError("causal_polarity_invalid")

    @property
    def sha256(self) -> str:
        return sealed_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "morphism_id": self.morphism_id,
            "version": self.version,
            "source_spec_id": self.source_spec_id,
            "target_spec_id": self.target_spec_id,
            "source_spec_sha256": self.source_spec_sha256,
            "target_spec_sha256": self.target_spec_sha256,
            "variable_map": dict(self.variable_map),
            "action_map": dict(self.action_map),
            "parameter_map": dict(self.parameter_map),
            "effect_map": dict(self.effect_map),
            "alarm_map": dict(self.alarm_map),
            "invariant_map": dict(self.invariant_map),
            "objective_relation": self.objective_relation,
            "causal_polarity_map": dict(self.causal_polarity_map),
            "scale_transforms": {
                key: vars(value) for key, value in self.scale_transforms.items()
            },
            "evidence": dict(self.evidence),
        }

    def validate(self, source: TransitionSpec, target: TransitionSpec) -> None:
        if (source.spec_id, target.spec_id) != (
            self.source_spec_id,
            self.target_spec_id,
        ):
            raise ValueError("morphism_spec_id_mismatch")
        if source.sha256 != self.source_spec_sha256:
            raise ValueError("morphism_source_hash_mismatch")
        if target.sha256 != self.target_spec_sha256:
            raise ValueError("morphism_target_hash_mismatch")
        source_variables = {item.name: item for item in source.variables}
        target_variables = {item.name: item for item in target.variables}
        source_actions = {item.name for item in source.actions}
        target_actions = {item.name for item in target.actions}
        source_effects = {
            item.effect_id for equation in source.equations for item in equation.effects
        }
        target_effects = {
            item.effect_id for equation in target.equations for item in equation.effects
        }
        self._validate_map(self.variable_map, set(source_variables), set(target_variables), "variable")
        self._validate_map(self.action_map, source_actions, target_actions, "action")
        self._validate_map(
            self.parameter_map, set(source.parameters), set(target.parameters), "parameter"
        )
        self._validate_map(self.effect_map, source_effects, target_effects, "effect")
        self._validate_map(
            self.alarm_map, set(source_variables), set(target_variables), "alarm"
        )
        self._validate_map(
            self.invariant_map, set(source_variables), set(target_variables), "invariant"
        )
        if set(self.causal_polarity_map) - source_effects:
            raise ValueError("morphism_causal_polarity_id_unknown")
        for left, right in self.variable_map.items():
            if source_variables[left].kind != target_variables[right].kind:
                raise ValueError("morphism_variable_type_mismatch")
        expected_relation = (
            "same"
            if source.optimization_direction == target.optimization_direction
            else "inverse"
        )
        if self.objective_relation != expected_relation:
            raise ValueError("morphism_objective_relation_mismatch")
        for key in self.scale_transforms:
            if key not in self.variable_map and key not in self.parameter_map:
                raise ValueError("morphism_scale_target_unknown")

    @staticmethod
    def _validate_map(
        mapping: Mapping[str, str],
        source_ids: set[str],
        target_ids: set[str],
        label: str,
    ) -> None:
        if set(mapping) - source_ids or set(mapping.values()) - target_ids:
            raise ValueError(f"morphism_{label}_id_unknown")
        if len(set(mapping.values())) != len(mapping):
            raise ValueError(f"morphism_{label}_map_not_one_to_one")
