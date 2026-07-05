"""Life Kernel soberano para RNFE."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Sequence
from uuid import uuid4

from runtime.organism.lineage import LineageState
from runtime.organism.state import IdentityState, OrganismState
from runtime.conjunction import OperationalConjunctionLayer
from runtime.conjunction.contracts import ComputeTier, OperationalConjunctionResult
from runtime.storage import StorageFacade, get_storage
from runtime.storage.records import utc_now_iso
from runtime.world import ScenarioEpisodeRunner

from .checkpoints import CheckpointManager
from .contracts import (
    AutonomyDecision,
    GoalState,
    LifeStepResult,
    RestoredIdentity,
    VitalSignsSnapshot,
)
from .goals import GoalManager
from .persistence import OrganismPersistence
from .serialization import lineage_from_payload, organism_from_payload
from .supervisor import AutonomySupervisor, AutonomySupervisorConfig
from .vitals import VitalSignsService


@dataclass(frozen=True, slots=True)
class LifeKernelConfig:
    run_id: str | None = None
    scenarios: Sequence[str] = ("thermal_homeostasis", "resource_management")
    block_size: int = 8
    interval_s: float = 0.0
    max_steps: int = 0
    restore: bool = True
    checkpoint_interval: int = 1
    memory_filter_mode: str = "strict_same_scenario"
    closure_profile: str = "baseline_fixed"
    default_external_input: float = 0.04
    perturbation_spike: float = 0.14
    perturbation_period: int = 7
    allow_external_reasoner: bool = False
    exploration_interval: int = 16
    enable_msrc: bool = True
    enable_operational_conjunction: bool = True
    max_compute_tier: ComputeTier = "tier_2_specialized"
    autonomy_policy: str = "bounded"


class LifeKernel:
    """Proceso soberano que mantiene vida, identidad y continuidad."""

    def __init__(
        self,
        *,
        config: LifeKernelConfig | None = None,
        storage: StorageFacade | None = None,
    ):
        self.config = config or LifeKernelConfig()
        self.storage = storage or get_storage()
        self.persistence = OrganismPersistence(storage=self.storage)
        self.checkpoints = CheckpointManager(storage=self.storage)
        self.vitals_service = VitalSignsService(storage=self.storage)
        self.supervisor = AutonomySupervisor(
            AutonomySupervisorConfig(
                allow_external_reasoner=self.config.allow_external_reasoner,
                exploration_interval=self.config.exploration_interval,
            )
        )
        self.conjunction = (
            OperationalConjunctionLayer(storage=self.storage)
            if self.config.enable_operational_conjunction
            else None
        )
        self.run_id = self.config.run_id or f"life-{uuid4().hex[:12]}"
        self.total_steps = 0
        self.scenario_index = 0
        self.scenario_episode_index = 0
        self.organism_state: OrganismState | None = None
        self.lineage: LineageState | None = None
        self.goal_manager = GoalManager()
        self.last_vitals: VitalSignsSnapshot | None = None
        self.last_decision: AutonomyDecision | None = None
        self.last_operational: OperationalConjunctionResult | None = None
        self._runner: ScenarioEpisodeRunner | None = None
        self._runner_key: tuple[str, str, str] | None = None
        self._msrc_controller = None
        self._scale_state = None

        if self.config.restore:
            self._restore_initial_identity()
        if self.organism_state is None or self.lineage is None:
            self._genesis()

    @classmethod
    def run_forever(
        cls,
        config: LifeKernelConfig | None = None,
        *,
        storage: StorageFacade | None = None,
    ) -> "LifeKernel":
        kernel = cls(config=config, storage=storage)
        kernel.run_until_stopped()
        return kernel

    def run_until_stopped(self, *, max_steps: int | None = None) -> list[LifeStepResult]:
        limit = self.config.max_steps if max_steps is None else int(max_steps)
        results: list[LifeStepResult] = []
        while True:
            if limit and len(results) >= limit:
                break
            result = self.step()
            results.append(result)
            if result.decision.action == "shutdown":
                break
            if self.config.interval_s > 0:
                time.sleep(self.config.interval_s)
        return results

    def step(self, external_input: float | None = None) -> LifeStepResult:
        assert self.organism_state is not None
        assert self.lineage is not None

        scenario = self._current_scenario()
        input_value = (
            float(external_input)
            if external_input is not None
            else self._perturbation(self.total_steps)
        )
        pre_vitals = self.last_vitals or self.vitals_service.bootstrap(
            run_id=self.run_id,
            organism_state=self.organism_state,
            lineage=self.lineage,
        )
        decision = self.supervisor.decide(
            vitals=pre_vitals,
            goals=self.goal_manager.goals,
            step_index=self.total_steps,
            scenario=scenario,
            external_input=input_value,
        )
        operational = self._evaluate_operational_conjunction(
            scenario=scenario,
            decision=decision,
            vitals=pre_vitals,
            external_input=input_value,
        )
        decision = self._apply_operational_gate(
            decision=decision,
            operational=operational,
        )
        self.last_decision = decision

        if decision.action in {"shutdown", "sleep", "quarantine", "rollback"}:
            return self._handle_non_acting_decision(
                decision=decision,
                pre_vitals=pre_vitals,
                operational=operational,
            )

        checkpoint_before = None
        if decision.action == "self_modify":
            checkpoint_before = self._save_checkpoint(
                vitals=pre_vitals,
                decision=decision,
                reason="pre_self_modify",
            )

        closure_profile = str(decision.directives.get("closure_profile") or self.config.closure_profile)
        runner = self._runner_for(
            scenario=scenario,
            closure_profile=closure_profile,
            memory_filter_mode=self.config.memory_filter_mode,
        )
        episode_result = runner.run_episode(external_input=input_value)
        if operational is not None:
            episode_result["operational_conjunction"] = operational.to_dict()
        self.organism_state = runner.organism_state
        self.lineage = runner.lineage

        vitals = self.vitals_service.from_state(
            run_id=self.run_id,
            organism_state=self.organism_state,
            lineage=self.lineage,
            mode=decision.mode,
            episode_result=episode_result,
        )
        goals = self.goal_manager.update_from_vitals(vitals)
        msrc_result = self._maybe_run_msrc(
            episode_result=episode_result,
            vitals=vitals,
        )

        self.total_steps += 1
        self.scenario_episode_index += 1
        if self.scenario_episode_index >= max(1, int(self.config.block_size)):
            self.scenario_episode_index = 0
            self.scenario_index += 1
            self._runner = None
            self._runner_key = None

        self.last_vitals = vitals
        checkpoint = self._checkpoint_if_due(
            vitals=vitals,
            decision=decision,
            reason="step_completed",
        )
        self.storage.append_event(
            event_type="life.step.completed",
            run_id=self.run_id,
            source="life_kernel",
            payload={
                "step_index": self.total_steps,
                "scenario": scenario,
                "decision": decision.to_dict(),
                "vital_signs": vitals.to_dict(),
                "goals": [goal.to_dict() for goal in goals],
                "checkpoint_artifact_id": getattr(checkpoint, "artifact_id", None),
                "pre_mutation_checkpoint_artifact_id": getattr(checkpoint_before, "artifact_id", None),
                "msrc": msrc_result,
                "operational_conjunction": operational.to_dict() if operational else {},
            },
        )
        return LifeStepResult(
            run_id=self.run_id,
            step_index=self.total_steps,
            decision=decision,
            vital_signs=vitals,
            goals=list(goals),
            episode_result=episode_result,
            checkpoint_artifact_id=getattr(checkpoint, "artifact_id", None),
            msrc=msrc_result,
            operational=operational.to_dict() if operational else {},
        )

    def _evaluate_operational_conjunction(
        self,
        *,
        scenario: str,
        decision: AutonomyDecision,
        vitals: VitalSignsSnapshot,
        external_input: float | None,
    ) -> OperationalConjunctionResult | None:
        if self.conjunction is None:
            return None
        result = self.conjunction.evaluate_life_cycle(
            run_id=self.run_id,
            scenario=scenario,
            decision=decision,
            vitals=vitals,
            goals=self.goal_manager.goals,
            step_index=self.total_steps,
            external_input=external_input,
            allow_external_reasoner=self.config.allow_external_reasoner,
            max_compute_tier=self.config.max_compute_tier,
            autonomy_policy=self.config.autonomy_policy,
        )
        self.last_operational = result
        return result

    def _apply_operational_gate(
        self,
        *,
        decision: AutonomyDecision,
        operational: OperationalConjunctionResult | None,
    ) -> AutonomyDecision:
        if operational is None:
            return decision
        operational_payload = operational.to_dict()
        if operational.final_decision == "block":
            if decision.action in {"self_modify", "consult_external"}:
                return AutonomyDecision(
                    action="act",
                    mode="recovery" if decision.action == "self_modify" else "conservative",
                    reason=f"operational_conjunction_blocked_{decision.action}",
                    priority=decision.priority,
                    scenario=decision.scenario,
                    external_input=decision.external_input,
                    directives={
                        **decision.directives,
                        "blocked_action": decision.action,
                        "closure_profile": "adaptive_min",
                        "operational_conjunction": operational_payload,
                    },
                )
            evidence_kinds = set(operational.context_summary.get("available_evidence_kinds") or [])
            if decision.action == "rollback" and "healthy_checkpoint" not in evidence_kinds:
                return AutonomyDecision(
                    action="quarantine",
                    mode="quarantine",
                    reason="operational_conjunction_blocked_rollback_without_evidence",
                    priority=decision.priority,
                    scenario=decision.scenario,
                    external_input=decision.external_input,
                    directives={
                        **decision.directives,
                        "blocked_action": decision.action,
                        "operational_conjunction": operational_payload,
                    },
                )
        if operational.final_decision == "degrade":
            return AutonomyDecision(
                action=decision.action,
                mode="conservative" if decision.mode == "normal" else decision.mode,
                reason=f"{decision.reason};operational_conjunction_degraded",
                priority=decision.priority,
                scenario=decision.scenario,
                external_input=decision.external_input,
                directives={
                    **decision.directives,
                    "closure_profile": "baseline_fixed",
                    "operational_conjunction": operational_payload,
                },
            )
        return AutonomyDecision(
            action=decision.action,
            mode=decision.mode,
            reason=decision.reason,
            priority=decision.priority,
            scenario=decision.scenario,
            external_input=decision.external_input,
            directives={
                **decision.directives,
                "operational_conjunction": operational_payload,
            },
            decision_id=decision.decision_id,
            created_at=decision.created_at,
        )

    def _handle_non_acting_decision(
        self,
        *,
        decision: AutonomyDecision,
        pre_vitals: VitalSignsSnapshot,
        operational: OperationalConjunctionResult | None = None,
    ) -> LifeStepResult:
        assert self.organism_state is not None
        assert self.lineage is not None

        vitals = pre_vitals
        if decision.action == "rollback":
            restored = self._restore_latest_healthy_checkpoint()
            if restored:
                vitals = self.vitals_service.bootstrap(
                    run_id=self.run_id,
                    organism_state=self.organism_state,
                    lineage=self.lineage,
                )
        checkpoint = self._save_checkpoint(
            vitals=vitals,
            decision=decision,
            reason=f"non_acting_{decision.action}",
        )
        self.storage.append_event(
            event_type=f"life.{decision.action}",
            run_id=self.run_id,
            source="life_kernel",
            payload={
                "step_index": self.total_steps,
                "decision": decision.to_dict(),
                "vital_signs": vitals.to_dict(),
                "checkpoint_artifact_id": checkpoint.artifact_id,
                "operational_conjunction": operational.to_dict() if operational else {},
            },
        )
        return LifeStepResult(
            run_id=self.run_id,
            step_index=self.total_steps,
            decision=decision,
            vital_signs=vitals,
            goals=list(self.goal_manager.goals),
            episode_result=None,
            checkpoint_artifact_id=checkpoint.artifact_id,
            operational=operational.to_dict() if operational else {},
        )

    def _restore_initial_identity(self) -> None:
        restored = self.persistence.load_latest_identity(run_id=self.config.run_id)
        if restored is None:
            return
        self._apply_restored_identity(restored)
        self.storage.append_event(
            event_type="life.identity.restored",
            run_id=self.run_id,
            source="life_kernel",
            payload={
                "checkpoint_artifact_id": restored.checkpoint_artifact_id,
                "episode_count": restored.organism_state.episode_count,
                "total_steps": restored.total_steps,
            },
        )

    def _apply_restored_identity(self, restored: RestoredIdentity) -> None:
        self.run_id = restored.run_id
        self.organism_state = restored.organism_state
        self.lineage = restored.lineage
        self.goal_manager = GoalManager(restored.goals or None)
        self.last_vitals = restored.vital_signs
        self.total_steps = restored.total_steps
        self.scenario_index = restored.scenario_index
        payload = restored.checkpoint_payload
        self.scenario_episode_index = int(payload.get("scenario_episode_index", 0))
        scale_state = payload.get("scale_state")
        if isinstance(scale_state, dict) and scale_state.get("current_scale_id"):
            self._scale_state = self._scale_state_from_payload(scale_state)

    def _genesis(self) -> None:
        self.organism_state = OrganismState(
            state_id=f"state-0-{self.run_id}",
            timestamp=utc_now_iso(),
            active_regime="genesis",
            episode_count=0,
            identity=IdentityState(lineage_id=f"lineage-{self.run_id}"),
        )
        self.lineage = LineageState(lineage_id=f"lineage-{self.run_id}")
        from runtime.organism.constitution import OrganismConstitution

        self.lineage.record_genesis(OrganismConstitution(), timestamp=utc_now_iso())
        self.last_vitals = self.vitals_service.bootstrap(
            run_id=self.run_id,
            organism_state=self.organism_state,
            lineage=self.lineage,
        )
        self.storage.append_event(
            event_type="life.genesis",
            run_id=self.run_id,
            source="life_kernel",
            payload={
                "organism_state_id": self.organism_state.state_id,
                "lineage_id": self.lineage.lineage_id,
                "goals": [goal.to_dict() for goal in self.goal_manager.goals],
            },
        )

    def _runner_for(
        self,
        *,
        scenario: str,
        closure_profile: str,
        memory_filter_mode: str,
    ) -> ScenarioEpisodeRunner:
        assert self.organism_state is not None
        assert self.lineage is not None
        key = (scenario, closure_profile, memory_filter_mode)
        if self._runner is not None and self._runner_key == key:
            return self._runner
        self._runner = ScenarioEpisodeRunner(
            storage=self.storage,
            run_id=self.run_id,
            scenario=scenario,
            memory_filter_mode=memory_filter_mode,
            closure_profile=closure_profile,
            organism_state=self.organism_state,
            lineage=self.lineage,
        )
        self._runner_key = key
        return self._runner

    def _checkpoint_if_due(
        self,
        *,
        vitals: VitalSignsSnapshot,
        decision: AutonomyDecision,
        reason: str,
    ):
        interval = max(1, int(self.config.checkpoint_interval))
        if vitals.is_stable or self.total_steps % interval == 0:
            return self._save_checkpoint(vitals=vitals, decision=decision, reason=reason)
        return None

    def _save_checkpoint(
        self,
        *,
        vitals: VitalSignsSnapshot,
        decision: AutonomyDecision | None,
        reason: str,
    ):
        assert self.organism_state is not None
        assert self.lineage is not None
        return self.checkpoints.save_checkpoint(
            run_id=self.run_id,
            organism_state=self.organism_state,
            lineage=self.lineage,
            goals=list(self.goal_manager.goals),
            vital_signs=vitals,
            total_steps=self.total_steps,
            scenario_index=self.scenario_index,
            scenario_episode_index=self.scenario_episode_index,
            memory_filter_mode=self.config.memory_filter_mode,
            closure_profile=self.config.closure_profile,
            decision=decision,
            runner_knobs={
                "memory_filter_mode": self.config.memory_filter_mode,
                "closure_profile": self.config.closure_profile,
            },
            scale_state=self._scale_state.to_dict() if self._scale_state is not None else {},
            metadata={"reason": reason},
        )

    def _restore_latest_healthy_checkpoint(self) -> bool:
        loaded = self.checkpoints.load_latest_payload(run_id=self.run_id, healthy_only=True)
        if loaded is None:
            return False
        payload, artifact = loaded
        self.organism_state = organism_from_payload(payload.get("organism_state"))
        self.lineage = lineage_from_payload(payload.get("lineage"))
        self.goal_manager = GoalManager.from_payload(payload.get("goals"))
        self.total_steps = int(payload.get("total_steps", self.total_steps))
        self.scenario_index = int(payload.get("scenario_index", self.scenario_index))
        self.scenario_episode_index = int(payload.get("scenario_episode_index", 0))
        self._runner = None
        self._runner_key = None
        self.storage.append_event(
            event_type="life.rollback.restored_checkpoint",
            run_id=self.run_id,
            source="life_kernel",
            payload={"checkpoint_artifact_id": artifact.artifact_id},
        )
        return True

    def _current_scenario(self) -> str:
        scenarios = list(self.config.scenarios or ("thermal_homeostasis",))
        return scenarios[self.scenario_index % len(scenarios)]

    def _perturbation(self, step: int) -> float:
        period = max(1, int(self.config.perturbation_period))
        if step % period == period - 1:
            return float(self.config.perturbation_spike)
        return float(self.config.default_external_input) + 0.01 * float(step % 3)

    def _maybe_run_msrc(
        self,
        *,
        episode_result: Dict[str, Any],
        vitals: VitalSignsSnapshot,
    ) -> Dict[str, Any]:
        if not self.config.enable_msrc:
            return {}
        try:
            from runtime.control.msrc import MSRCController, ProbeResult, ScalePolicyState
            from runtime.control.msrc.vram_sampler import NullVRAMSampler
        except Exception:
            return {"status": "unavailable"}
        if self._scale_state is None:
            self._scale_state = ScalePolicyState(current_scale_id="1x1")
        if self._msrc_controller is None:
            self._msrc_controller = MSRCController(
                storage=self.storage,
                vram_sampler=NullVRAMSampler(),
            )

        episode = episode_result.get("episode") or {}
        episode_id = str(episode.get("episode_id") or f"life-step-{self.total_steps}")
        observation = ((episode.get("context") or {}).get("observation") or {})

        def probe_executor(target_scale_id: str):
            gain = max(0.0, vitals.cognitive_quality - 0.50)
            return ProbeResult(
                target_scale_id=target_scale_id,
                cognitive_gain_delta=round(gain, 4),
                viability_preserved=vitals.viability_margin >= 0.35,
                evidence_score=round(max(vitals.cognitive_quality, vitals.ioc_proxy), 4),
                outcome="positive" if gain > 0.03 and vitals.viability_margin >= 0.35 else "inconclusive",
                details={"source": "life_kernel_shadow_probe"},
            )

        try:
            out = self._msrc_controller.step(
                run_id=self.run_id,
                episode_id=episode_id,
                state=self._scale_state,
                observation=dict(observation),
                viability_margin=vitals.viability_margin,
                certification_verdict=(episode_result.get("certification") or {}).get("verdict"),
                metrics={
                    "cognitive_quality": vitals.cognitive_quality,
                    "risk_score": vitals.risk_score,
                    "memory_purity": vitals.memory_purity,
                },
                probe_executor=probe_executor,
            )
        except Exception as exc:
            self.storage.append_event(
                event_type="life.msrc.error",
                run_id=self.run_id,
                source="life_kernel",
                payload={"episode_id": episode_id, "error": f"{type(exc).__name__}: {exc}"},
            )
            return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        self._scale_state = out["state"]
        return {
            "selected_scale_id": out["selected_scale_id"],
            "action": out["action"].to_dict(),
            "estimate": out["estimate"].to_dict(),
            "state": self._scale_state.to_dict(),
        }

    @staticmethod
    def _scale_state_from_payload(payload: Dict[str, Any]):
        try:
            from runtime.control.msrc import ScalePolicyState

            return ScalePolicyState(
                current_scale_id=str(payload.get("current_scale_id", "1x1")),
                step_index=int(payload.get("step_index", 0)),
                upgrade_evidence=int(payload.get("upgrade_evidence", 0)),
                downgrade_evidence=int(payload.get("downgrade_evidence", 0)),
                cooldown_remaining=int(payload.get("cooldown_remaining", 0)),
                lock_remaining=int(payload.get("lock_remaining", 0)),
                probe_inflight_target=payload.get("probe_inflight_target"),
                last_actions=list(payload.get("last_actions") or []),
                oscillation_events=int(payload.get("oscillation_events", 0)),
                upgrade_regret=int(payload.get("upgrade_regret", 0)),
                downgrade_regret=int(payload.get("downgrade_regret", 0)),
                missed_upgrade_regret=int(payload.get("missed_upgrade_regret", 0)),
                regime_history=list(payload.get("regime_history") or []),
            )
        except Exception:
            return None
