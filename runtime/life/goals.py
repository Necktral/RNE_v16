"""Agenda autonoma de objetivos internos RNFE."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List

from runtime.storage.records import utc_now_iso

from .contracts import GoalState, VitalSignsSnapshot


class GoalManager:
    """Mantiene y actualiza los objetivos vivos del organismo."""

    def __init__(self, goals: Iterable[GoalState] | None = None):
        self.goals: List[GoalState] = list(goals or self.default_goals())

    @staticmethod
    def default_goals() -> list[GoalState]:
        return [
            GoalState.create(
                kind="survival",
                priority=1.0,
                horizon_episodes=1,
                success_metric="viability_margin>=0.45",
                risk_budget=0.10,
            ),
            GoalState.create(
                kind="continuity",
                priority=0.95,
                horizon_episodes=8,
                success_metric="identity_continuity>=0.60",
                risk_budget=0.15,
            ),
            GoalState.create(
                kind="risk_reduction",
                priority=0.85,
                horizon_episodes=4,
                success_metric="risk_score<0.60",
                risk_budget=0.12,
            ),
            GoalState.create(
                kind="cognitive_gain",
                priority=0.72,
                horizon_episodes=12,
                success_metric="cognitive_quality improving",
                risk_budget=0.30,
            ),
            GoalState.create(
                kind="memory_maintenance",
                priority=0.68,
                horizon_episodes=6,
                success_metric="memory_purity>=0.75",
                risk_budget=0.20,
            ),
            GoalState.create(
                kind="exploration",
                priority=0.42,
                horizon_episodes=16,
                success_metric="safe novelty under certified continuity",
                risk_budget=0.35,
                metadata={"cadence": "opportunistic"},
            ),
        ]

    @classmethod
    def from_payload(cls, payload: Iterable[dict] | None) -> "GoalManager":
        goals = []
        for item in payload or []:
            if isinstance(item, dict):
                goals.append(GoalState.from_dict(item))
        return cls(goals=goals or None)

    def active_goals(self) -> list[GoalState]:
        return [goal for goal in self.goals if goal.status == "active"]

    def highest_priority(self) -> GoalState:
        active = self.active_goals()
        if not active:
            self.goals = self.default_goals()
            active = self.active_goals()
        return max(active, key=lambda goal: goal.priority)

    def update_from_vitals(self, vitals: VitalSignsSnapshot) -> list[GoalState]:
        updated: list[GoalState] = []
        now = utc_now_iso()
        for goal in self.goals:
            progress = self._progress_for(goal, vitals)
            status = "satisfied" if progress >= 1.0 and goal.kind != "survival" else "active"
            if goal.kind == "survival":
                status = "active" if vitals.viability_margin > 0.0 else "failed"
            updated.append(
                replace(
                    goal,
                    progress=round(progress, 4),
                    status=status,
                    updated_at=now,
                )
            )
        self.goals = updated
        return list(self.goals)

    def to_payload(self) -> list[dict]:
        return [goal.to_dict() for goal in self.goals]

    @staticmethod
    def _progress_for(goal: GoalState, vitals: VitalSignsSnapshot) -> float:
        if goal.kind == "survival":
            return min(1.0, max(0.0, vitals.viability_margin / 0.45))
        if goal.kind == "continuity":
            return min(1.0, max(0.0, vitals.identity_continuity / 0.60))
        if goal.kind == "risk_reduction":
            return min(1.0, max(0.0, 1.0 - (vitals.risk_score / 0.60)))
        if goal.kind == "cognitive_gain":
            return min(1.0, max(0.0, vitals.cognitive_quality))
        if goal.kind == "memory_maintenance":
            return min(1.0, max(0.0, vitals.memory_purity / 0.75))
        if goal.kind == "exploration":
            if not vitals.certified or vitals.risk_score >= goal.risk_budget:
                return 0.0
            return min(1.0, max(0.0, 0.5 * vitals.cognitive_quality + 0.5 * vitals.memory_purity))
        return 0.0
