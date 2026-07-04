"""Supervisor soberano de autonomia operacional."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .contracts import AutonomyDecision, GoalState, VitalSignsSnapshot


@dataclass(frozen=True, slots=True)
class AutonomySupervisorConfig:
    allow_external_reasoner: bool = False
    exploration_interval: int = 16
    shutdown_on_irreversible: bool = False


class AutonomySupervisor:
    """Decide que debe hacer el organismo en el siguiente ciclo vital."""

    def __init__(self, config: AutonomySupervisorConfig | None = None):
        self.config = config or AutonomySupervisorConfig()

    def decide(
        self,
        *,
        vitals: VitalSignsSnapshot,
        goals: Iterable[GoalState],
        step_index: int,
        scenario: str,
        external_input: float | None = None,
        shutdown_requested: bool = False,
    ) -> AutonomyDecision:
        if shutdown_requested:
            return AutonomyDecision(
                action="shutdown",
                mode="shutdown_safe",
                reason="shutdown_requested",
                priority=1.0,
                scenario=scenario,
                external_input=external_input,
            )

        if vitals.episode_count <= 0:
            return AutonomyDecision(
                action="act",
                mode="normal",
                reason="genesis_requires_first_episode",
                priority=1.0,
                scenario=scenario,
                external_input=external_input,
            )

        if vitals.mode == "rollback":
            action = "shutdown" if self.config.shutdown_on_irreversible and not vitals.reversible else "rollback"
            return AutonomyDecision(
                action=action,
                mode="rollback" if action == "rollback" else "shutdown_safe",
                reason="viability_edge_or_irreversible_state",
                priority=1.0,
                scenario=scenario,
                external_input=external_input,
            )

        if vitals.mode == "quarantine":
            return AutonomyDecision(
                action="quarantine",
                mode="quarantine",
                reason="identity_or_viability_below_quarantine_threshold",
                priority=0.98,
                scenario=scenario,
                external_input=external_input,
            )

        if vitals.resource_pressure >= 0.90:
            return AutonomyDecision(
                action="sleep",
                mode="conservative",
                reason="resource_pressure_too_high",
                priority=0.92,
                scenario=scenario,
                external_input=external_input,
                directives={"cooldown": 1},
            )

        if vitals.memory_purity < 0.65 or vitals.accumulated_drift > 0.55:
            return AutonomyDecision(
                action="self_modify",
                mode="recovery",
                reason="degradation_candidate_for_controlled_self_modification",
                priority=0.90,
                scenario=scenario,
                external_input=external_input,
                directives={
                    "mutation_scope": "policy_or_memory_knobs",
                    "requires_checkpoint": True,
                    "requires_shadow_window": True,
                },
            )

        if vitals.mode == "recovery":
            return AutonomyDecision(
                action="act",
                mode="recovery",
                reason="recover_by_certified_episode_before_mutating",
                priority=0.84,
                scenario=scenario,
                external_input=external_input,
                directives={"closure_profile": "adaptive_min"},
            )

        if (
            self.config.allow_external_reasoner
            and vitals.cognitive_quality < 0.35
            and vitals.risk_score < 0.80
        ):
            return AutonomyDecision(
                action="consult_external",
                mode="conservative",
                reason="low_cognitive_quality_with_bounded_risk",
                priority=0.76,
                scenario=scenario,
                external_input=external_input,
                directives={
                    "external_reasoner": "gated",
                    "advisory_only": True,
                },
            )

        top_goal = self._top_goal(goals)
        cadence = max(1, int(self.config.exploration_interval))
        if (
            top_goal is not None
            and top_goal.kind == "exploration"
            and vitals.is_stable
            and step_index > 0
            and step_index % cadence == 0
        ):
            return AutonomyDecision(
                action="explore",
                mode="normal",
                reason="stable_state_allows_controlled_exploration",
                priority=top_goal.priority,
                scenario=scenario,
                external_input=external_input,
                directives={"closure_profile": "adaptive_min"},
            )

        return AutonomyDecision(
            action="act",
            mode=vitals.mode,
            reason="nominal_life_cycle",
            priority=0.70,
            scenario=scenario,
            external_input=external_input,
        )

    @staticmethod
    def _top_goal(goals: Iterable[GoalState]) -> GoalState | None:
        active = [goal for goal in goals if goal.status == "active"]
        if not active:
            return None
        return max(active, key=lambda goal: goal.priority)
