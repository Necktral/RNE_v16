"""Planner SMT determinista de horizonte acotado."""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping

from z3 import Solver, sat

from .compiler import TransitionCompiler
from .contracts import SMTPlanReport


class SMTPlanner:
    def __init__(
        self,
        compiler: TransitionCompiler,
        *,
        horizon: int = 3,
        effort_cost: float = 0.05,
        risk_cost: float = 0.25,
    ):
        self.compiler = compiler
        self.horizon = max(1, int(horizon))
        self.effort_cost = float(effort_cost)
        self.risk_cost = float(risk_cost)

    def plan(
        self,
        state: Mapping[str, Any],
        *,
        external_input: float,
    ) -> SMTPlanReport:
        actions = tuple(sorted(action.name for action in self.compiler.spec.actions))
        candidates: list[tuple[tuple[Any, ...], SMTPlanReport]] = []
        for length in range(1, self.horizon + 1):
            for sequence in product(actions, repeat=length):
                current = dict(state)
                projected: list[Mapping[str, Any]] = []
                constraint_ids: list[str] = []
                solver = Solver()
                for step, action in enumerate(sequence, 1):
                    _, constraints = self.compiler.z3_step(
                        current,
                        action=action,
                        external_input=external_input,
                        prefix=f"mci_t{step}",
                    )
                    for index, constraint in enumerate(constraints):
                        solver.add(constraint)
                        constraint_ids.append(f"mci/plan/t/{step}/constraint/{index}")
                    current = self.compiler.execute(
                        current, action=action, external_input=external_input
                    )
                    projected.append(current)
                if solver.check() != sat:
                    continue
                regret = self._regret(current)
                risk = self._risk(projected)
                effort = self.effort_cost * length
                objective = round(regret + effort + self.risk_cost * risk, 9)
                report = SMTPlanReport(
                    status="sat",
                    horizon=self.horizon,
                    actions=sequence,
                    projected_states=tuple(projected),
                    objective=objective,
                    regret=round(regret, 9),
                    effort_cost=round(effort, 9),
                    risk=round(risk, 9),
                    constraint_ids=tuple(constraint_ids),
                )
                candidates.append(((objective, length, sequence), report))
        if not candidates:
            return SMTPlanReport(
                status="unsat",
                horizon=self.horizon,
                actions=(),
                projected_states=(),
                objective=None,
                regret=None,
                effort_cost=None,
                risk=None,
                constraint_ids=(),
            )
        return min(candidates, key=lambda item: item[0])[1]

    def _regret(self, state: Mapping[str, Any]) -> float:
        spec = self.compiler.spec
        value = float(state[spec.main_variable])
        threshold = float(spec.parameters[spec.alarm_threshold_parameter])
        return max(0.0, value - threshold) if spec.optimization_direction == "minimize" else max(
            0.0, threshold - value
        )

    def _risk(self, states: list[Mapping[str, Any]]) -> float:
        if not states:
            return 1.0
        alarms = sum(bool(state[self.compiler.spec.alarm_variable]) for state in states)
        return alarms / len(states)
