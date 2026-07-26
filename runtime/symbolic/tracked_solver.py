"""Wrapper mínimo de Z3 que registra toda restricción afirmada."""

from __future__ import annotations

from typing import Any, Mapping

from z3 import Bool, BoolRef, Not, Solver, sat, unsat

from .constraint_registry import ConstraintRegistry, constraint_id
from .schemas import ConstraintRecord, CoreReport


class TrackedSolver:
    def __init__(
        self,
        *,
        registry: ConstraintRegistry,
        replay_unit_id: str,
        logical_time: int,
    ):
        self.registry = registry
        self.replay_unit_id = replay_unit_id
        self.logical_time = int(logical_time)
        self.solver = Solver()
        self._ids: list[str] = []

    def _add(
        self,
        expression: BoolRef,
        *,
        kind: str,
        source: str,
        rule_id: str,
        provenance: dict[str, Any],
        hard_or_revisable: str,
    ) -> str:
        identifier = constraint_id(
            replay_unit_id=self.replay_unit_id,
            logical_time=self.logical_time,
            kind=kind,
            source=source,
            rule_id=rule_id,
        )
        record = ConstraintRecord(
            constraint_id=identifier,
            expression=str(expression),
            hard_or_revisable=hard_or_revisable,
            provenance=provenance,
            parent_ids=(),
            logical_time=self.logical_time,
            replay_unit_id=self.replay_unit_id,
        )
        self.registry.register(record)
        self.solver.assert_and_track(expression, Bool(identifier))
        if identifier not in self._ids:
            self._ids.append(identifier)
        return identifier

    def add_formula(
        self,
        expression: BoolRef,
        *,
        kind: str = "formula",
        source: str,
        rule_id: str,
    ) -> str:
        return self._add(
            expression,
            kind=kind,
            source=source,
            rule_id=rule_id,
            provenance={"kind": kind, "source": source, "rule_id": rule_id},
            hard_or_revisable="hard",
        )

    def add_assumption(
        self,
        symbol: BoolRef,
        value: bool,
        *,
        source: str,
        rule_id: str,
    ) -> str:
        expression = symbol if value else Not(symbol)
        return self._add(
            expression,
            kind="assumption",
            source=source,
            rule_id=rule_id,
            provenance={
                "kind": "assumption",
                "source": source,
                "rule_id": rule_id,
                "value": bool(value),
            },
            hard_or_revisable="revisable",
        )

    def check(
        self,
        *,
        need_model: bool = False,
        model_symbols: Mapping[str, BoolRef] | None = None,
    ) -> tuple[str, Any | None, tuple[str, ...], CoreReport]:
        result = self.solver.check()
        status = "SAT" if result == sat else "UNSAT" if result == unsat else "UNKNOWN"
        model = self.solver.model() if result == sat and need_model else None
        core_ids = tuple(sorted(str(item) for item in self.solver.unsat_core()))
        serialized_model = None
        if model is not None:
            if model_symbols is not None:
                serialized_model = {
                    symbol: str(model.evaluate(model_symbols[symbol], model_completion=True))
                    for symbol in sorted(model_symbols)
                }
            else:
                serialized_model = {
                    str(declaration): str(model[declaration])
                    for declaration in sorted(model.decls(), key=lambda item: str(item))
                    if not str(declaration).startswith("unit/")
                }
        report = self.registry.get_core_report(
            status=status,
            core_ids=core_ids,
            all_ids=self._ids,
            model=serialized_model,
        )
        return status, model, core_ids, report
