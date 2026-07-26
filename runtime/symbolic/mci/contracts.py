"""Contratos inmutables del Módulo de Cognición Integrada."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from runtime.symbolic.schemas import _Schema, _freeze, sealed_sha256


@dataclass(frozen=True)
class VariableSpec(_Schema):
    name: str
    kind: str
    lower: float = 0.0
    upper: float = 1.0


@dataclass(frozen=True)
class TermSpec(_Schema):
    source: str
    coefficient: float = 1.0
    parameter: str | None = None


@dataclass(frozen=True)
class EffectSpec(_Schema):
    effect_id: str
    action: str
    delta: float
    parameter: str | None = None
    precondition: str | None = None


@dataclass(frozen=True)
class EquationSpec(_Schema):
    target: str
    terms: tuple[TermSpec, ...]
    effects: tuple[EffectSpec, ...] = ()
    clamp: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        object.__setattr__(self, "effects", tuple(self.effects))


@dataclass(frozen=True)
class ActionSpec(_Schema):
    name: str
    assignments: Mapping[str, bool]

    def __post_init__(self) -> None:
        object.__setattr__(self, "assignments", _freeze(self.assignments))


@dataclass(frozen=True)
class InvariantSpec(_Schema):
    variable: str
    operator: str
    threshold_parameter: str


@dataclass(frozen=True)
class TransitionSpec(_Schema):
    spec_id: str
    version: str
    scenario: str
    variables: tuple[VariableSpec, ...]
    parameters: Mapping[str, float]
    actions: tuple[ActionSpec, ...]
    equations: tuple[EquationSpec, ...]
    alarm_variable: str
    alarm_source: str
    alarm_operator: str
    alarm_threshold_parameter: str
    main_variable: str
    optimization_direction: str
    safe_action: str
    invariants: tuple[InvariantSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "parameters", _freeze(self.parameters))
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(self, "equations", tuple(self.equations))
        object.__setattr__(self, "invariants", tuple(self.invariants))
        self.validate()

    @property
    def sha256(self) -> str:
        return sealed_sha256(self.to_dict())

    def validate(self) -> None:
        from .conditions import parse_condition

        variables = {item.name for item in self.variables}
        actions = {item.name for item in self.actions}
        if len(variables) != len(self.variables) or len(actions) != len(self.actions):
            raise ValueError("Variables o acciones duplicadas")
        if self.main_variable not in variables or self.alarm_variable not in variables:
            raise ValueError("Variable principal o alarma no declarada")
        if self.safe_action not in actions:
            raise ValueError("safe_action no declarada")
        if self.optimization_direction not in {"minimize", "maximize"}:
            raise ValueError("Dirección de optimización inválida")
        for equation in self.equations:
            if equation.target not in variables:
                raise ValueError(f"Target no declarado: {equation.target}")
            effect_ids: set[str] = set()
            for effect in equation.effects:
                if effect.effect_id in effect_ids:
                    raise ValueError(f"Effect duplicado: {effect.effect_id}")
                effect_ids.add(effect.effect_id)
                if effect.action not in actions:
                    raise ValueError("Effect referencia una acción desconocida")
                if effect.parameter is not None and effect.parameter not in self.parameters:
                    raise ValueError("Effect referencia un parámetro desconocido")
                if effect.precondition:
                    parse_condition(effect.precondition, variables)


@dataclass(frozen=True)
class CausalOverlay(_Schema):
    overlay_id: str
    version: int
    base_spec_sha256: str
    parameter_updates: Mapping[str, float]
    precondition_updates: Mapping[str, str | None]
    edge_updates: tuple[tuple[str, str, str], ...]
    evidence: Mapping[str, Any]
    train_mae_before: float
    train_mae_after: float
    holdout_mae_before: float
    holdout_mae_after: float
    parent_overlay_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_updates", _freeze(self.parameter_updates))
        object.__setattr__(self, "precondition_updates", _freeze(self.precondition_updates))
        object.__setattr__(self, "edge_updates", tuple(tuple(item) for item in self.edge_updates))
        object.__setattr__(self, "evidence", _freeze(self.evidence))


@dataclass(frozen=True)
class SMTPlanReport(_Schema):
    status: str
    horizon: int
    actions: tuple[str, ...]
    projected_states: tuple[Mapping[str, Any], ...]
    objective: float | None
    regret: float | None
    effort_cost: float | None
    risk: float | None
    constraint_ids: tuple[str, ...]
    tie_breaks: tuple[str, ...] = ("objective", "length", "actions")

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", tuple(self.actions))
        object.__setattr__(
            self, "projected_states", tuple(_freeze(item) for item in self.projected_states)
        )
        object.__setattr__(self, "constraint_ids", tuple(self.constraint_ids))
        object.__setattr__(self, "tie_breaks", tuple(self.tie_breaks))


@dataclass(frozen=True)
class Justification(_Schema):
    justification_id: str
    antecedent_ids: tuple[str, ...]
    consequence_id: str
    kind: str


@dataclass(frozen=True)
class BeliefNode(_Schema):
    belief_id: str
    proposition: str
    status: str
    confidence: float
    logical_time: int
    justification_ids: tuple[str, ...]


@dataclass(frozen=True)
class SelfModelReport(_Schema):
    signature: str
    success_probability: float
    expected_cost: float
    invariant_risk: float
    support_count: int
    decision: str
    safe_action: str


@dataclass(frozen=True)
class MCICommitReport(_Schema):
    schema: str
    spec_id: str
    spec_sha256: str
    overlay_id: str | None
    plan: SMTPlanReport
    causal_graph: tuple[tuple[str, str, str, str | None], ...]
    beliefs: tuple[BeliefNode, ...]
    belief_revision: Mapping[str, Any]
    self_model: SelfModelReport
    constraint_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "causal_graph", tuple(tuple(edge) for edge in self.causal_graph)
        )
        object.__setattr__(self, "beliefs", tuple(self.beliefs))
        object.__setattr__(self, "belief_revision", _freeze(self.belief_revision))
        object.__setattr__(self, "constraint_ids", tuple(self.constraint_ids))
