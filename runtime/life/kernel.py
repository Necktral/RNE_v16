"""Life Kernel soberano para RNFE."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Sequence
from uuid import uuid4

from runtime.organism.identity import (
    CausalContext,
    causal_context_enabled,
    mint_lineage_id,
    mint_organism_id,
    mint_run_id,
    resolve_organism_id,
)
from runtime.organism.lineage import LineageState
from runtime.organism.state import IdentityState, OrganismState
from runtime.conjunction import OperationalConjunctionLayer
from runtime.conjunction.contracts import ComputeTier, OperationalConjunctionResult
from runtime.conjunction.execution import routing_enforced, tier_execution_directives
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

# B48: acciones que NO ejecutan episodio (las maneja la rama no-actuante de
# ``step()``). Cualquier otra acción llega al runner, así que bajo veredicto
# "block" debe transformarse o detenerse — jamás caer al retorno anotador.
NON_EPISODE_ACTIONS = frozenset({"shutdown", "sleep", "quarantine", "rollback"})


@dataclass(frozen=True, slots=True)
class LifeKernelConfig:
    run_id: str | None = None
    # B41: genoma persistente a vincular (resume/bind de un organismo conocido, p. ej.
    # aeon-01). None ⇒ el kernel resuelve por RNFE_ORGANISM_ID o acuña un genoma genuino.
    # Config gana sobre entorno (precedencia de génesis, §1.2).
    organism_id: str | None = None
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
        # B41 — tres ejes de identidad distintos (canon f2.4 A-M5/A-M8/A-M10):
        #   run_id      : la corrida (EFÍMERO; re-acuñado cada proceso salvo config).
        #   organism_id : el genoma (PERSISTE entre corridas; namespace de memoria M_t).
        #   lineage_id  : el linaje evolutivo μ_t (abarca varios organismos).
        # run_id ya NO es fuente de identidad persistente: solo marca esta ejecución.
        self.run_id = self.config.run_id or mint_run_id()
        self.organism_id: str = ""
        self.lineage_id: str = ""
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
        self._runner_key: tuple | None = None
        self._msrc_controller = None
        self._scale_state = None
        # Sensado de recursos (host + GPU), opt-in por RNFE_HOST_SENSING.
        self._host_sampler = None
        self._vram_sampler = None
        self._resource_snapshot: Dict[str, Any] = {}
        # Experiencia: memoria de golpes + lecciones del maestro (RNFE_EXPERIENCE / RNFE_TEACHER).
        from runtime.organism.experience import ExperienceStore, experience_enabled

        self._experience = ExperienceStore(storage=self.storage) if experience_enabled() else None
        self._experience_lessons: list = []
        self._teacher = None
        self._consecutive_quarantine = 0
        self._steps_since_musing = 0

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
        snapshot = self._sense_resources()
        snap = snapshot or None
        if self.last_vitals is None:
            pre_vitals = self.vitals_service.bootstrap(
                run_id=self.run_id,
                organism_state=self.organism_state,
                lineage=self.lineage,
                resource_snapshot=snap,
            )
        elif snap is not None:
            # Con sensado activo, refrescamos presión/modo para que el
            # supervisor reaccione en el mismo ciclo (rama sleep, conservative).
            pre_vitals = self.vitals_service.from_state(
                run_id=self.run_id,
                organism_state=self.organism_state,
                lineage=self.lineage,
                mode=self.last_vitals.mode,
                episode_result=None,
                resource_snapshot=snap,
            )
        else:
            pre_vitals = self.last_vitals
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

        if decision.action in NON_EPISODE_ACTIONS:
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
        memory_retrieval_limit: int | None = None
        external_reasoner_enabled = self.config.allow_external_reasoner
        routing_directive: Dict[str, Any] | None = None
        # B3: el tier del router se vuelve EJECUTABLE (gated por
        # RNFE_CONJUNCTION_ROUTING_ENFORCED). Off -> nada de esto aplica.
        if routing_enforced() and operational is not None:
            # B48: se ejecuta con el tier AUTORIZADO por el gate (execution_tier),
            # no con el tier en el que se validó la acción original. Una acción
            # crítica bloqueada y transformada valida en un tier y su reemplazo
            # seguro ejecuta en otro (separación validation_tier/execution_tier).
            execution_tier = (
                decision.directives.get("execution_tier")
                or operational.selected_compute_tier
            )
            exec_dir = tier_execution_directives(
                execution_tier,
                gpu_backed=self._route_gpu_backed(operational),
            )
            # La seguridad manda: si el gate/supervisor ya fijó un closure_profile
            # (block/degrade/recovery), se respeta; si no, lo fija el tier.
            if "closure_profile" not in decision.directives:
                closure_profile = exec_dir.closure_profile
            memory_retrieval_limit = exec_dir.memory_retrieval_limit
            external_reasoner_enabled = (
                external_reasoner_enabled and exec_dir.external_reasoner_enabled
            )
            routing_directive = exec_dir.to_dict()
        runner = self._runner_for(
            scenario=scenario,
            closure_profile=closure_profile,
            memory_filter_mode=self.config.memory_filter_mode,
            memory_retrieval_limit=memory_retrieval_limit,
        )
        # A3: el runner inyecta el snapshot de recursos en el contexto de razonamiento.
        runner.set_resource_signals(self._resource_snapshot)
        runner.set_external_reasoner_enabled(external_reasoner_enabled)
        # Experiencia: identidad cross-vida + lecciones del maestro para sesgar el episodio.
        runner.set_organism_id(self.run_id)
        runner.set_experience_lessons(self._experience_lessons)
        episode_result = runner.run_episode(external_input=input_value)
        self._consecutive_quarantine = 0  # actuó sano ⇒ ya no está atascado
        # Reflexión continua (E2): el maestro reflexiona si el episodio hirió, o cada
        # cierto tiempo — la reflexión es parte de su vida en todo momento. Off hot-path
        # nominal (solo cuando hay herida o toca el musing).
        if self._experience is not None:
            exp_info = (episode_result or {}).get("experience") or {}
            self._steps_since_musing += 1
            # Cooldown: la reflexión PROFUNDA (7B, ~5s) es proporcional, no cada golpe —
            # como mucho cada ~8 latidos, antes si la herida fue grave. La reflexión
            # ligera (recall/sesgo E3) ya es continua y barata en cada latido.
            severity = float(exp_info.get("severity", 0.0) or 0.0)
            if self._steps_since_musing >= 8 or (severity >= 0.7 and self._steps_since_musing >= 4):
                self._steps_since_musing = 0
                self._maybe_reflect()
        if routing_directive is not None:
            episode_result["routing_directive"] = routing_directive
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
            resource_snapshot=snap,
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

    def _sense_resources(self) -> Dict[str, Any]:
        """Snapshot de recursos host+GPU para el ciclo actual (opt-in).

        Off (``RNFE_HOST_SENSING`` sin setear) devuelve ``{}`` y todo el camino
        vivo permanece byte-idéntico. On, compone CPU/RAM del host con la VRAM
        real (``NvidiaVRAMSampler``) cuando hay GPU.
        """
        from runtime.control.msrc.host_sampler import (
            HostResourceSampler,
            build_resource_snapshot,
            host_sensing_enabled,
        )

        if not host_sensing_enabled():
            self._resource_snapshot = {}
            return {}
        if self._host_sampler is None:
            self._host_sampler = HostResourceSampler()
        if self._vram_sampler is None:
            try:
                from runtime.control.msrc.vram_sampler import NvidiaVRAMSampler

                self._vram_sampler = NvidiaVRAMSampler()
            except Exception:
                self._vram_sampler = None
        snapshot = build_resource_snapshot(
            host_sampler=self._host_sampler,
            vram_sampler=self._vram_sampler,
        )
        self._resource_snapshot = snapshot
        return snapshot

    @staticmethod
    def _route_gpu_backed(operational: OperationalConjunctionResult) -> bool:
        """True solo si el router marcó la ruta seleccionada como servida por GPU."""
        for item in reversed(operational.trace):
            if isinstance(item, dict) and item.get("stage") == "router.output":
                return bool(item.get("gpu_backed"))
        return False

    @staticmethod
    def _scenario_info(scenario: str) -> tuple[str, list]:
        """(main_variable, interventions) del escenario, best-effort."""
        try:
            from runtime.world import get_scenario

            sc = get_scenario(scenario)
            return str(sc.config.main_variable), list(sc.config.interventions)
        except Exception:
            return "", []

    def _record_nonacting_wound(self, *, decision: AutonomyDecision, vitals: VitalSignsSnapshot) -> None:
        """Graba el golpe de una decisión no-actuante (cuarentena/rollback/sleep) en el diario."""
        if self._experience is None:
            return
        from runtime.organism.experience import build_experience

        scenario = self._current_scenario()
        main_var, _ = self._scenario_info(scenario)
        exp = build_experience(
            organism_id=self.run_id,
            run_id=self.run_id,
            episode_id=f"life-{decision.action}-{self.total_steps}",
            scenario=scenario,
            regime="distress" if decision.action in {"quarantine", "rollback"} else "rest",
            main_variable=main_var,
            causal_status="",
            intervention="",
            viability_margin=vitals.viability_margin,
            ioc=vitals.ioc_proxy,
            risk=vitals.risk_score,
            reward=0.0,
            action=decision.action,
            certified=bool(vitals.certified),
            closure_passed=False,
            viability_delta=0.0,
        )
        self._experience.record(exp)

    def _maybe_reflect(self) -> None:
        """El maestro (7B) reflexiona sobre las heridas recientes y destila lecciones (E2)."""
        from runtime.organism.teacher import teacher_enabled

        if not teacher_enabled() or self._experience is None:
            return
        if self._teacher is None:
            from runtime.organism.teacher import Teacher

            self._teacher = Teacher(storage=self.storage, experience=self._experience)
        scenario = self._current_scenario()
        _, valid = self._scenario_info(scenario)
        if not valid:
            return
        try:
            lessons = self._teacher.reflect(organism_id=self.run_id, valid_interventions=valid)
        except Exception:
            lessons = []
        if lessons:
            self._experience_lessons = lessons
            self.storage.append_event(
                event_type="life.reflection",
                run_id=self.run_id,
                source="life_kernel",
                payload={"step_index": self.total_steps, "lessons": lessons},
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
            resource_snapshot=self._resource_snapshot or None,
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
        # Separación validation_tier/execution_tier (B48): validation_tier es el
        # tier en el que la conjunción VALIDÓ la acción original; execution_tier
        # es el tier en el que la decisión resultante queda AUTORIZADA a
        # ejecutar (None si el ciclo se detiene y no hay episodio).
        validation_tier = operational.selected_compute_tier
        if operational.final_decision == "block":
            if decision.action in {"self_modify", "consult_external"}:
                # La acción crítica bloqueada NO se ejecuta: se reemplaza por un
                # acto seguro que ejecuta en el tier más conservador, no en el
                # tier donde validó la acción original.
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
                        "validation_tier": validation_tier,
                        "execution_tier": "tier_0_deterministic",
                    },
                    # B39: la transformación preserva la identidad causal de la
                    # decisión original (mismo decision_id/created_at).
                    decision_id=decision.decision_id,
                    created_at=decision.created_at,
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
                        "validation_tier": validation_tier,
                        "execution_tier": None,
                    },
                    # B39: identidad causal preservada en la transformación.
                    decision_id=decision.decision_id,
                    created_at=decision.created_at,
                )
            if decision.action not in NON_EPISODE_ACTIONS:
                # B48 (H5): invariante total de bloqueo. Cualquier otra acción
                # ejecutante bajo veredicto "block" se DETIENE — jamás cae al
                # retorno anotador (que dejaría correr el episodio igual).
                return AutonomyDecision(
                    action="sleep",
                    mode="conservative",
                    reason=f"operational_conjunction_block_halted_{decision.action}",
                    priority=decision.priority,
                    scenario=decision.scenario,
                    external_input=decision.external_input,
                    directives={
                        **decision.directives,
                        "blocked_action": decision.action,
                        "operational_conjunction": operational_payload,
                        "validation_tier": validation_tier,
                        "execution_tier": None,
                    },
                    # B39: identidad causal preservada en la transformación.
                    decision_id=decision.decision_id,
                    created_at=decision.created_at,
                )
            # Acción no-ejecutante (shutdown/sleep/quarantine/rollback con
            # evidencia) bajo "block": ya se detiene sola en la rama no-actuante
            # de step(); cae al retorno anotador final sin ejecutar episodio.
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
                    "validation_tier": validation_tier,
                    "execution_tier": validation_tier,
                },
                # B39: identidad causal preservada en la transformación.
                decision_id=decision.decision_id,
                created_at=decision.created_at,
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
                "validation_tier": validation_tier,
                "execution_tier": (
                    validation_tier if decision.action not in NON_EPISODE_ACTIONS else None
                ),
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
        elif decision.action == "quarantine" and self._experience is not None:
            # E5 — refugio sano: una cuarentena atascada es un callejón sin salida (no
            # ejecuta episodios ⇒ no se recupera). Tras varias seguidas, el organismo rueda
            # a su último yo sano para sobrevivir y seguir aprendiendo de sus golpes.
            self._consecutive_quarantine += 1
            if self._consecutive_quarantine >= 3 and self._restore_latest_healthy_checkpoint():
                vitals = self.vitals_service.bootstrap(
                    run_id=self.run_id,
                    organism_state=self.organism_state,
                    lineage=self.lineage,
                )
                self._consecutive_quarantine = 0
                self.storage.append_event(
                    event_type="life.refuge",
                    run_id=self.run_id,
                    source="life_kernel",
                    payload={"step_index": self.total_steps, "reason": "quarantine_stuck_rollback_to_healthy"},
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
        # Experiencia: grabar el golpe de esta decisión no-actuante y —si es una herida
        # profunda— que el maestro (7B) reflexione y destile una lección (E1 + E2, off hot-path).
        self._record_nonacting_wound(decision=decision, vitals=vitals)
        if decision.action in {"quarantine", "rollback"}:
            self._maybe_reflect()
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
        # organism_id (genoma) por precedencia: config gana sobre entorno →
        # RNFE_ORGANISM_ID → ancestro (stub) → génesis genuina org-{uuid4}.
        self.organism_id = resolve_organism_id(self.config.organism_id) or mint_organism_id()
        # lineage_id (linaje μ_t): génesis genuina ⇒ linaje nuevo con un solo organismo.
        # Distinto del run_id (ya no se deriva de la corrida).
        self.lineage_id = mint_lineage_id()
        self.organism_state = OrganismState(
            state_id=f"state-0-{self.organism_id}",
            timestamp=utc_now_iso(),
            active_regime="genesis",
            episode_count=0,
            identity=IdentityState(lineage_id=self.lineage_id),
        )
        self.lineage = LineageState(lineage_id=self.lineage_id)
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
                "organism_id": self.organism_id,
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
        memory_retrieval_limit: int | None = None,
    ) -> ScenarioEpisodeRunner:
        assert self.organism_state is not None
        assert self.lineage is not None
        key = (scenario, closure_profile, memory_filter_mode, memory_retrieval_limit)
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
        # El tier ejecutable (B3) puede fijar el límite de recuperación de memoria;
        # None conserva el default del runner (byte-idéntico).
        if memory_retrieval_limit is not None:
            self._runner.memory_retrieval_limit = int(memory_retrieval_limit)
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
            # Con sensado activo y GPU real disponible, MSRC recibe la VRAM real;
            # de lo contrario mantiene el NullVRAMSampler (byte-idéntico).
            vram_sampler = NullVRAMSampler()
            if self._vram_sampler is not None and self._resource_snapshot.get("gpu_available"):
                vram_sampler = self._vram_sampler
            self._msrc_controller = MSRCController(
                storage=self.storage,
                vram_sampler=vram_sampler,
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
