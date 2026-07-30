"""Runner de episodio cognitivo genérico que soporta múltiples escenarios."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from runtime.certification.promotion_gate import PromotionGate
from runtime.lotf import LOTFMin
from runtime.memory.mfm_lite.retrieval import MemoryRetrieval
from runtime.organism.autoevolution import AutoEvolutionController
from runtime.organism.constitution import OrganismConstitution
from runtime.organism.identity import mint_lineage_id, mint_organism_id
from runtime.organism.lineage import LineageState
from runtime.organism.state import OrganismState, IdentityState, transition_organism_state
from runtime.organism.trajectory import OrganismTrajectory
from runtime.organism.viability import ViabilityKernel
from runtime.reasoning.context import build_reasoning_context, resolve_reasoning_mode
from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler
from runtime.reality.belief_state import BeliefState, build_belief_state
from runtime.smg import SMGMin
from runtime.storage import get_storage
from runtime.storage.records import utc_now_iso
from runtime.reasoning.families import a12 as a12_family
from runtime.world.intervention_override import (
    OverrideDecision,
    RolloutAssessment,
    evaluate_foresight_override,
    evaluate_override,
    is_actuation_enabled,
    outcome_effectiveness,
)
from runtime.world.scenario import ScenarioTransition
from runtime.world.causal_attestation import build_causal_attestation
from runtime.symbolic.eml import EMLRunner
from runtime.symbolic.acting_trace import ActingTraceCollector
from runtime.symbolic.constraint_registry import ConstraintRegistry, semantic_segment
from runtime.symbolic.core_reporter import persist_core_report
from runtime.symbolic.optimization_tracker import build_optimization_report
from runtime.symbolic.schemas import CausalGuardReport, sealed_sha256
from runtime.symbolic.tracked_solver import TrackedSolver
from runtime.symbolic.mci import (
    EdgeMapping,
    HypothesisMappingRegistry,
    HypothesisProvider,
    MCIRuntime,
    MCIPlanningConfig,
    deferred_load_spec,
    resource_spec,
    resource_with_energy_spec,
    thermal_spec,
    thermal_battery_spec,
    TransferredHypothesis,
)
from runtime.symbolic.mci.hypothesis_mapping import default_edge_mappings
from runtime.symbolic.mci.feedback import relay_feedback
from runtime.symbolic.mci.provider_adapter import adapt_hypotheses
from runtime.neural import NeuralMode, NeuralModelManifest, NeuralRuntime
from runtime.neural.contracts import canonical_sha256
from runtime.neural.n4_hypothesis_provider import N4HypothesisProvider
from runtime.neural.organs import N4CausalRankingBackend

from .scenario import CognitiveScenario, ScenarioObservation
from .registry import get_scenario, DEFAULT_SCENARIO


def _external_reasoner_runtime_flag() -> bool:
    """True si RNFE_EXTERNAL_REASONER_RUNTIME habilita el razonador externo vivo."""
    return os.environ.get("RNFE_EXTERNAL_REASONER_RUNTIME", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _resolve_mci_planning_config(
    explicit: MCIPlanningConfig | None,
) -> MCIPlanningConfig:
    if explicit is not None:
        return explicit
    horizon_raw = os.environ.get("RNFE_MCI_PLANNING_HORIZON")
    exact_raw = os.environ.get("RNFE_MCI_EXACT_HORIZON")
    mode_raw = os.environ.get("RNFE_MCI_OBJECTIVE_MODE")
    if horizon_raw is None and exact_raw is None and mode_raw is None:
        return MCIPlanningConfig()
    try:
        horizon = int(horizon_raw) if horizon_raw is not None else 3
        exact = (
            exact_raw.strip().lower() in {"1", "true", "yes", "on"}
            if exact_raw is not None
            else False
        )
        mode = mode_raw.strip() if mode_raw is not None else "terminal_regret"
        return MCIPlanningConfig(
            horizon=horizon,
            exact_horizon=exact,
            objective_mode=mode,
        )
    except (AttributeError, TypeError, ValueError):
        return MCIPlanningConfig()


class ScenarioEpisodeRunner:
    """Runner de episodio cognitivo que opera sobre escenarios parametrizables.

    Este runner es la versión generalizada de MinimalCognitiveEpisodeRunner
    que puede operar sobre cualquier escenario que implemente CognitiveScenario.
    """

    def __init__(
        self,
        *,
        storage=None,
        run_id: str | None = None,
        scenario: CognitiveScenario | str | None = None,
        scenario_kwargs: Dict[str, Any] | None = None,
        memory_filter_mode: str = "strict_same_scenario",
        closure_profile: str = "baseline_fixed",
        organism_state: OrganismState | None = None,
        lineage: LineageState | None = None,
        reward_guided=None,
        family_profile: str | None = None,
        mci_planning_config: MCIPlanningConfig | None = None,
        n4_artifact_path: Path | None = None,
    ):
        """Inicializa runner con escenario especificado.

        Args:
            storage: Storage facade opcional.
            run_id: ID de corrida.
            scenario: Escenario como instancia, nombre string, o None para default.
            scenario_kwargs: Kwargs para crear escenario si es string.
            memory_filter_mode: Modo de filtrado de memoria por escenario
                ('strict_same_scenario' o 'cross_scenario_analogical').
                El alias 'analogical' se normaliza automáticamente.
            closure_profile: Perfil de cierre a usar ('baseline_fixed' o 'adaptive_min').
            organism_state: Estado inicial del organismo (para continuar una vida
                a través de varios runners/regímenes, p. ej. el life-loop).
            lineage: LineageState compartido para continuidad generacional.
        """
        self.storage = storage or get_storage()
        self.run_id = run_id or f"run-{uuid4()}"
        self._trajectory_regime_label = DEFAULT_SCENARIO

        # Resolver escenario
        if scenario is None:
            self.scenario = get_scenario(DEFAULT_SCENARIO, **(scenario_kwargs or {}))
            self._trajectory_regime_label = DEFAULT_SCENARIO
        elif isinstance(scenario, str):
            self.scenario = get_scenario(scenario, **(scenario_kwargs or {}))
            self._trajectory_regime_label = scenario
        else:
            self.scenario = scenario
            self._trajectory_regime_label = self.scenario.config.name

        # Normalize memory filter mode alias
        if memory_filter_mode == "analogical":
            memory_filter_mode = "cross_scenario_analogical"
        _VALID_MEMORY_MODES = {"strict_same_scenario", "cross_scenario_analogical"}
        if memory_filter_mode not in _VALID_MEMORY_MODES:
            raise ValueError(
                f"memory_filter_mode inválido: '{memory_filter_mode}'. "
                f"Válidos: {sorted(_VALID_MEMORY_MODES)}"
            )
        self.memory_filter_mode = memory_filter_mode

        _VALID_CLOSURE_PROFILES = {"baseline_fixed", "adaptive_min"}
        if closure_profile not in _VALID_CLOSURE_PROFILES:
            raise ValueError(
                f"closure_profile inválido: '{closure_profile}'. "
                f"Válidos: {sorted(_VALID_CLOSURE_PROFILES)}"
            )
        self.closure_profile = closure_profile
        self.reasoning_mode = resolve_reasoning_mode(closure_profile)
        self.family_profile = str(
            family_profile
            or os.environ.get("RNFE_REASONING_FAMILY_PROFILE")
            or ""
        ).strip().lower()
        self._mci_planning_config = _resolve_mci_planning_config(
            mci_planning_config
        )
        self._n4_artifact_path = n4_artifact_path
        # Señales de recursos (host+GPU) inyectadas por el LifeKernel por ciclo.
        # Vacío por defecto -> el contexto de razonamiento no cambia (byte-idéntico).
        self._resource_signals: Dict[str, Any] = {}
        # Habilitación por-episodio del razonador externo (tier_3). El runtime
        # solo lo agenda si además el perfil admitido y el gate lo permiten (Bloque C).
        self._external_reasoner_enabled: bool = False
        # B41: sobre CausalContext del step (aditivo, gated por el kernel). None ⇒ no-op.
        self._causal_context: Dict[str, Any] | None = None
        # Experiencia: el organismo recuerda sus golpes y aprende (RNFE_EXPERIENCE).
        # B41: el namespace es organism_id (cross-vida). El runner NO acuña con convención
        # propia (nada de org-{run_id}): usa la función de acuñación compartida (SSOT). En
        # el life-loop, el kernel soberano lo REEMPLAZA vía set_organism_id con el genoma real.
        from runtime.organism.experience import ExperienceStore, experience_enabled

        self._organism_id: str = mint_organism_id()
        # Linaje standalone (solo si no llega uno del kernel): un único lineage_id
        # compartido entre el estado de génesis y el LineageState, vía la SSOT.
        _standalone_lineage_id = mint_lineage_id()
        self._experience = ExperienceStore(storage=self.storage) if experience_enabled() else None
        self._experience_lessons: List[Dict[str, Any]] = []

        self.smg = SMGMin(storage=self.storage, run_id=self.run_id)
        self.lotf = LOTFMin()
        self.scheduler = MetaScheduler(
            trace_store=self.storage,
            mode=self.reasoning_mode,
            family_profile=self.family_profile or None,
        )
        self._mci_runtime = self._build_mci_runtime()
        self._hypothesis_provider: HypothesisProvider | None = None
        self._hypothesis_mappings = HypothesisMappingRegistry()
        self._provider_hypothesis_metadata: dict[str, dict[str, str]] = {}
        self._configure_n4_closed_loop()
        self.memory_retrieval = MemoryRetrieval(storage=self.storage)
        self.promotion_gate = PromotionGate(storage=self.storage)
        self.eml_mode = os.environ.get("RNFE_EML_MODE", "disabled").strip().lower()
        self.eml_runner = EMLRunner(storage=self.storage)
        self._previous_belief: BeliefState | None = None

        # T5 SOVEREIGNTY: Initialize organism trajectory as primary runtime unit
        self._organism_state = organism_state or OrganismState(
            state_id=f"state-0-{self._organism_id}",
            timestamp=utc_now_iso(),
            active_regime="unknown",
            episode_count=0,
            identity=IdentityState(
                lineage_id=_standalone_lineage_id,
                constitution_hash="",
            ),
        )
        self._organism_trajectory = OrganismTrajectory(
            organism_id=self._organism_id,
            start_timestamp=utc_now_iso(),
        )
        self._constitution = OrganismConstitution()
        self._viability_kernel = ViabilityKernel(constitution=self._constitution)

        # R2 — vida: linaje activo (μₜ) + lazo de autoevolución (ρₜ).
        # Los mandos (knobs) son parámetros de comportamiento REALES del runner;
        # el controlador solo actúa bajo degradación sostenida, así que los
        # baselines sanos quedan numéricamente intactos.

        self.memory_retrieval_limit = 3
        if lineage is not None:
            self._lineage = lineage
        else:
            self._lineage = LineageState(lineage_id=_standalone_lineage_id)
            self._lineage.record_genesis(self._constitution, timestamp=utc_now_iso())
        self._autoevolution = AutoEvolutionController(
            run_id=self.run_id,
            storage=self.storage,
            lineage=self._lineage,
            knob_reader=lambda: {
                "memory_retrieval_limit": self.memory_retrieval_limit,
                "memory_filter_mode": self.memory_filter_mode,
            },
            knob_writer=self._apply_knob_changes,
        )

        # Multiplicación de ganancia (canon §8): selector de overlays guiado por
        # la recompensa semi-Markov. Apagado por defecto (disciplina sombra);
        # se activa con RNFE_REWARD_GUIDED_SELECTION=1.
        from runtime.reasoning.scheduler_meta.reward_guided import (
            RewardGuidedOverlaySelector,
            is_reward_guided_enabled,
        )

        self._reward_guided: RewardGuidedOverlaySelector | None = (
            reward_guided
            if reward_guided is not None
            else (
                RewardGuidedOverlaySelector(storage=self.storage)
                if is_reward_guided_enabled()
                else None
            )
        )
        # Reglas inducidas transferidas por una ecología multi-organismo
        # (modo reasoning_policy_plus_rules). None en el camino de un solo organismo.
        self._inherited_rules: list | None = None
    def _build_mci_runtime(self) -> MCIRuntime | None:
        """Construye el modelo del escenario sin estado global ni lógica duplicada."""
        name = self.scenario.config.name
        if name == "thermal_homeostasis":
            spec = thermal_spec(
                alarm_threshold=float(self.scenario.config.alarm_threshold),
                cooling_effect=float(getattr(self.scenario, "_cooling_effect")),
            )
        elif name == "resource_management":
            spec = resource_spec(
                scarcity_threshold=float(self.scenario.config.alarm_threshold),
                production_rate=float(getattr(self.scenario, "_production_rate")),
            )
        elif name == "deferred_load_trap":
            spec = deferred_load_spec(
                alarm_threshold=float(self.scenario.config.alarm_threshold),
                boost_effect=float(getattr(self.scenario, "_boost_effect")),
                shed_effect=float(getattr(self.scenario, "_shed_effect")),
                boost_debt=float(getattr(self.scenario, "_boost_debt")),
                shed_debt=float(getattr(self.scenario, "_shed_debt")),
            )
        elif name == "thermal_with_battery":
            spec = thermal_battery_spec(
                alarm_threshold=float(self.scenario.config.alarm_threshold),
                cooling_effect=float(getattr(self.scenario, "cooling_effect")),
                battery_discharge_rate=float(
                    getattr(self.scenario, "_battery_discharge_rate")
                ),
                battery_charge_rate=float(
                    getattr(self.scenario, "_battery_charge_rate")
                ),
            )
        elif name == "resource_with_energy":
            spec = resource_with_energy_spec(
                scarcity_threshold=float(self.scenario.config.alarm_threshold),
                production_rate=float(getattr(self.scenario, "_production_rate")),
                production_energy_cost=float(
                    getattr(self.scenario, "_production_energy_cost")
                ),
                recovery_rate=float(getattr(self.scenario, "_recovery_rate")),
            )
        else:
            return None
        runtime = MCIRuntime(
            spec,
            planning_config=self._mci_planning_config,
        )
        try:
            overlays = self.storage.list_events(
                limit=200,
                event_types=["mci.overlay.promoted"],
                run_id=self.run_id,
            )
            compatible = [
                event.payload
                for event in overlays
                if (event.payload or {}).get("base_spec_sha256") == spec.sha256
            ]
            if compatible:
                runtime.restore_overlay(
                    max(compatible, key=lambda item: int(item.get("version", 0)))
                )
            ledger_events = self.storage.list_events(
                limit=1,
                event_types=["mci.hypothesis.ledger"],
                run_id=self.run_id,
            )
            if ledger_events:
                ledger_payload = (ledger_events[0].payload or {}).get("entries") or ()
                runtime.restore_hypothesis_ledger(
                    tuple(dict(item) for item in ledger_payload)
                )
            transfer_events = self.storage.list_events(
                limit=1,
                event_types=["mci.transfer.ledger"],
                run_id=self.run_id,
            )
            if transfer_events:
                entries = (transfer_events[0].payload or {}).get("entries") or ()
                runtime.restore_transfer_ledger(tuple(dict(item) for item in entries))
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            pass
        return runtime

    @property
    def mci_active(self) -> bool:
        return self.family_profile in {
            "mci_integrated_v1",
            "mci_n4_closed_loop_v1",
        } and self._mci_runtime is not None

    def _configure_n4_closed_loop(self) -> None:
        if (
            self.family_profile != "mci_n4_closed_loop_v1"
            or self._mci_runtime is None
        ):
            return
        weights: dict[str, Any] = {
            "ranking_weights": (2.0, 1.5, 0.5, 0.4, 1.0, 1.5),
            "bias": -1.0,
            "temperature": 1.0,
        }
        trained = False
        provenance: dict[str, Any] = {}
        metrics: dict[str, Any] = {"scientific_gate_eligible": False}
        model_id = "n4-reference-ranking-v1"
        artifact_path = "reference/n4-ranking-v1.json"
        artifact_sha256 = canonical_sha256(weights)
        if self._n4_artifact_path is not None:
            raw_bytes = self._n4_artifact_path.read_bytes()
            payload = json.loads(raw_bytes)
            if payload.get("schema") not in {
                "n4-ranking-artifact.v1",
                "n4-ranking-artifact.v2",
            }:
                raise ValueError("n4_artifact_schema_invalid")
            schema = payload.get("schema")
            expected_kind = (
                "trained_multihead"
                if schema == "n4-ranking-artifact.v2"
                else "trained"
            )
            if payload.get("model_kind") != expected_kind:
                raise ValueError("n4_artifact_must_be_trained")
            promotable = (
                (payload.get("gates") or {}).get("promotable")
                if schema == "n4-ranking-artifact.v2"
                else payload.get("promotable")
            )
            if not bool(promotable):
                raise ValueError("n4_artifact_not_promotable")
            weights = payload
            trained = True
            provenance = dict(
                payload.get("training_provenance")
                or payload.get("training")
                or {}
            )
            metrics = {
                "scientific_gate_eligible": True,
                "validation": payload.get("validation_metrics")
                or payload.get("validation"),
                "holdout": payload.get("holdout_metrics")
                or payload.get("holdout"),
            }
            model_id = f"n4-trained-ranking-{hashlib.sha256(raw_bytes).hexdigest()[:12]}"
            artifact_path = self._n4_artifact_path.name
            artifact_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        manifest = NeuralModelManifest(
            organ="N4",
            capability="causal_hypothesis_ranking",
            model_id=model_id,
            version=(
                "2"
                if weights.get("schema") == "n4-ranking-artifact.v2"
                else "1"
            ),
            backend="python-deterministic",
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
            trained=trained,
            training_provenance=provenance,
            metrics=metrics,
        )
        provider = N4HypothesisProvider(
            runtime=NeuralRuntime(
                backend=N4CausalRankingBackend(
                    weights,
                    artifact_sha256=artifact_sha256,
                ),
                manifest=manifest,
                mode=NeuralMode.EXPERIMENTAL,
            ),
            spec=self._mci_runtime.base_spec,
            run_id=self.run_id,
        )
        try:
            calibration_events = self.storage.list_events(
                limit=1,
                event_types=["n4.calibration.snapshot"],
                run_id=self.run_id,
            )
            if calibration_events:
                snapshot = (calibration_events[0].payload or {}).get("snapshot")
                if isinstance(snapshot, dict):
                    provider.calibration.restore(snapshot)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            pass
        self.set_hypothesis_provider(provider)

    def ingest_mci_hypotheses(self, hypotheses: list[Any]) -> bool:
        """Encola hipótesis externas; no altera el baseline si MCI no está disponible."""
        if self._mci_runtime is None:
            return False
        self._mci_runtime.ingest_neural_hypotheses(hypotheses)
        return True

    def ingest_transferred_hypotheses(
        self, hypotheses: list[TransferredHypothesis]
    ) -> bool:
        """Encola conocimiento transferido como propuestas, nunca como autoridad."""
        if self._mci_runtime is None:
            return False
        self._mci_runtime.ingest_transferred_hypotheses(hypotheses)
        return True

    def get_mci_active_overlay(self) -> dict[str, Any] | None:
        """Snapshot público de solo lectura para empaquetado experimental."""
        if self._mci_runtime is None or self._mci_runtime.learner.active_overlay is None:
            return None
        return self._mci_runtime.learner.active_overlay.to_dict()

    def set_hypothesis_provider(
        self,
        provider: HypothesisProvider | None,
        *,
        edge_mappings: tuple[EdgeMapping, ...] = (),
    ) -> None:
        """Configura un emisor proposal-only sin alterar el perfil cognitivo."""
        if provider is not None and not isinstance(provider, HypothesisProvider):
            raise TypeError("provider no implementa HypothesisProvider")
        resolved_mappings = edge_mappings
        if not resolved_mappings and self._mci_runtime is not None:
            resolved_mappings = default_edge_mappings(
                self._mci_runtime.base_spec.spec_id
            )
        registry = HypothesisMappingRegistry(resolved_mappings)
        if self._mci_runtime is not None:
            registry.validate_for(self._mci_runtime.base_spec)
        self._hypothesis_provider = provider
        self._hypothesis_mappings = registry
        self._provider_hypothesis_metadata.clear()

    def _maybe_override_intervention(
        self,
        *,
        reasoning_state: Dict[str, Any],
        greedy_intervention: str,
        factual: Any,
        external_input: float,
    ) -> "tuple[OverrideDecision, Any]":
        """Decide el override determinista guardado (sombra: OFF salvo flag).

        Devuelve (decision, candidate_transition). La transición candidata se
        simula fresca (el contrafactual naive del runner no es la alterna real).
        """
        if not is_actuation_enabled():
            return OverrideDecision(fired=False, guard_reason="actuation_disabled"), None
        mv = self.scenario.config.main_variable
        try:
            direction = str(self.scenario.causal_signature.optimization_direction)
        except Exception:
            direction = "minimize"
        allowed = list(self.scenario.config.interventions)

        # 1) Override de PREVISIÓN (A11+A12): guard de horizonte, corre primero. A12
        # ya integró toda la evidencia (no-monotonía + Bayes-factor + ACT) y A11
        # certificó el breach diferido del greedy; el guard de un paso lo vetaría.
        #
        # El decisor A12 debe usar la traza COMPLETA. El scheduler puede ejecutarlo
        # antes que otras familias; para la ACTUACIÓN lo recomputamos sobre el estado
        # final (ya con todas las claves: imagination_*/ctf/cau/ded/prob). Gated
        # internamente por RNFE_A12_DEEP (idle ⇒ sin claves ⇒ no dispara).
        a12_delta = a12_family.execute(reasoning_state).get("state_delta", {})
        foresight_state = {**reasoning_state, **a12_delta} if a12_delta else reasoning_state
        foresight = evaluate_foresight_override(
            reasoning_state=foresight_state,
            allowed_interventions=allowed,
            greedy_intervention=greedy_intervention,
        )
        if foresight.fired:
            assert foresight.to_intervention is not None
            candidate = self.scenario.simulate_counterfactual(
                intervention=foresight.to_intervention, external_input=external_input
            )
            return foresight, candidate

        # 2) Override greedy guardado de UN paso (existente): conflicto estructural +
        # familia deliberativa (opt/plan/ind) + mejora inmediata certificada.
        sim_cache: Dict[str, ScenarioTransition] = {}

        def simulate_value(intervention: str) -> float:
            transition = self.scenario.simulate_counterfactual(
                intervention=intervention, external_input=external_input
            )
            sim_cache[intervention] = transition
            return float(transition.state.get(mv, 0.0))

        decision = evaluate_override(
            reasoning_state=reasoning_state,
            allowed_interventions=list(self.scenario.config.interventions),
            greedy_intervention=greedy_intervention,
            direction=direction,
            factual_value=float(factual.state.get(mv, 0.0)),
            simulate_value=simulate_value,
            **self._mci_rollout_guard_inputs(
                reasoning_state=reasoning_state,
                greedy_intervention=greedy_intervention,
                external_input=external_input,
            ),
        )
        candidate: Optional[ScenarioTransition] = (
            sim_cache.get(str(decision.to_intervention))
            if decision.fired and decision.to_intervention is not None
            else None
        )
        if (
            decision.fired
            and decision.to_intervention is not None
            and candidate is None
        ):
            candidate = self.scenario.simulate_counterfactual(
                intervention=decision.to_intervention,
                external_input=external_input,
            )
        return decision, candidate

    def _mci_rollout_guard_inputs(
        self,
        *,
        reasoning_state: Dict[str, Any],
        greedy_intervention: str,
        external_input: float,
    ) -> Dict[str, Any]:
        """Construye callbacks multi-step solo para un plan MCI concreto y no trivial."""
        if not self.mci_active or self._mci_runtime is None:
            return {}
        runtime = self._mci_runtime
        plan_payload = reasoning_state.get("mci_plan_report")
        if not isinstance(plan_payload, dict):
            return {}
        actions = tuple(str(item) for item in (plan_payload.get("actions") or ()))
        candidate = str(reasoning_state.get("mci_first_action") or "")
        if len(actions) <= 1 or not candidate or actions[0] != candidate:
            return {}
        horizon = min(5, len(actions))
        actions = actions[:horizon]
        observation = self.scenario.observe()
        initial_state = {
            **dict(observation.state),
            runtime.base_spec.alarm_variable: bool(observation.alarm),
        }
        def simulate_rollout(sequence) -> RolloutAssessment:
            report = runtime.evaluate_plan_sequence(
                initial_state,
                actions=tuple(str(item) for item in sequence),
                external_input=float(external_input),
            )
            violations = tuple(
                f"alarm/t/{index}"
                for index, state in enumerate(report.projected_states, 1)
                if bool(state.get(runtime.base_spec.alarm_variable))
            )
            return RolloutAssessment(
                objective=(
                    float(report.objective)
                    if report.objective is not None
                    else float("inf")
                ),
                invariant_violations=violations,
                status=report.status,
            )

        return {
            "candidate_sequence": actions,
            "baseline_sequence": (greedy_intervention,) * horizon,
            "simulate_rollout": simulate_rollout,
        }

    def _apply_knob_changes(self, changes: Dict[str, Any]) -> None:
        """Aplica una modificación aceptada sobre los mandos reales del runner."""
        if "memory_retrieval_limit" in changes:
            self.memory_retrieval_limit = max(1, int(changes["memory_retrieval_limit"]))
        if "memory_filter_mode" in changes:
            mode = str(changes["memory_filter_mode"])
            if mode in {"strict_same_scenario", "cross_scenario_analogical"}:
                self.memory_filter_mode = mode

    @property
    def organism_state(self) -> OrganismState:
        """Estado vivo del organismo (para continuarlo en otro runner/régimen)."""
        return self._organism_state

    @property
    def lineage(self) -> LineageState:
        """Linaje del organismo (continuidad generacional)."""
        return self._lineage

    def _build_scenario_metadata(self) -> Dict[str, Any]:
        """Construye metadata formal del escenario activo.

        Returns:
            Dict con identidad completa del escenario: nombre, versión,
            hash de configuración, variable principal, umbral e intervenciones.
        """
        cfg = self.scenario.config
        config_blob = json.dumps(
            {
                "name": cfg.name,
                "main_variable": cfg.main_variable,
                "alarm_threshold": cfg.alarm_threshold,
                "interventions": cfg.interventions,
                "formula_template": cfg.formula_template,
                "type_context": cfg.type_context,
            },
            sort_keys=True,
        )
        config_hash = hashlib.sha256(config_blob.encode()).hexdigest()[:12]
        return {
            "scenario_name": cfg.name,
            "scenario_version": "1.0",
            "scenario_config_hash": config_hash,
            "main_variable": cfg.main_variable,
            "alarm_threshold": cfg.alarm_threshold,
            "interventions": cfg.interventions,
        }

    def _build_eml_dataset(
        self,
        *,
        observation: Dict[str, Any],
        factual: Dict[str, Any],
        counterfactual: Dict[str, Any],
    ) -> list[dict[str, float]]:
        """Construye dataset EML a partir del escenario."""
        main_var = self.scenario.config.main_variable
        x = float(observation.get(main_var, 0.0))
        cf = float(counterfactual.get(main_var, x))
        y = float(factual.get(main_var, x))
        return [
            {"x": x, "cf": cf, "y": y},
            {"x": max(0.0, x - 0.02), "cf": cf, "y": y},
            {"x": min(1.0, x + 0.02), "cf": cf, "y": y},
        ]

    def set_organism_id(self, organism_id: str) -> None:
        """Fija el namespace de identidad (genoma) para la experiencia cross-vida.

        B41: el kernel soberano REEMPLAZA el organism_id acuñado por el runner con el
        genoma real; se propaga también a la trayectoria para que la identidad del
        organismo sea única en todo el runner (no queda el org-acuñado del constructor).
        Si llega vacío, se conserva el acuñado por la SSOT (nunca vuelve a run_id).
        """
        if organism_id:
            self._organism_id = str(organism_id)
            self._organism_trajectory.organism_id = self._organism_id

    def set_experience_lessons(self, lessons: List[Dict[str, Any]] | None) -> None:
        """Inyecta lecciones del maestro (7B) para sesgar el razonamiento vía IND."""
        self._experience_lessons = list(lessons) if lessons else []

    def set_causal_context(self, causal_context: Dict[str, Any] | None) -> None:
        """Inyecta el sobre CausalContext.v1 del step (aditivo, gated por el kernel).

        None ⇒ no-op byte-idéntico. Cuando está presente, su ``trace_group_id`` ata la
        cadena decisión→episodio→traza→certificado de ESTE step y viaja como clave
        aditiva en el evento ``episode.closed`` y en el contexto de razonamiento.
        """
        self._causal_context = dict(causal_context) if causal_context else None

    def _causal_context_signals(self) -> Dict[str, Any]:
        """Señales aditivas del sobre para el contexto de razonamiento (trazas)."""
        ctx = self._causal_context
        if not ctx:
            return {}
        tg = ctx.get("trace_group_id")
        return {"trace_group_id": tg} if tg else {}

    def _situation_signature(self, observation) -> str:
        """Firma de situación estable, consistente entre sesgo y grabación."""
        from runtime.organism.experience import situation_key

        regime = "alarm" if getattr(observation, "alarm", False) else "calm"
        return situation_key(
            scenario=self.scenario.config.name,
            regime=regime,
            main_variable=self.scenario.config.main_variable,
        )

    def _experience_biased_intervention(self, observation, intervention):
        """Si esta intervención hirió antes en esta situación, propone una mejor.

        Umbral de dolor mínimo + margen de mejora ⇒ el organismo solo cambia cuando
        hay una alternativa claramente menos dolorosa; nunca veto absoluto. La
        fuerza es proporcional a la cicatriz (más dolor ⇒ más probable superar el margen).
        """
        _MIN_SCAR = 0.5
        _MARGIN = 0.25
        sig = self._situation_signature(observation)
        wisdom = self._experience.wisdom(organism_id=self._organism_id, situation=sig)
        scar_here = float(wisdom.scar.get(intervention, 0.0))
        if scar_here < _MIN_SCAR:
            return None
        candidates = [iv for iv in self.scenario.config.interventions if iv != intervention]
        if not candidates:
            return None
        best = min(candidates, key=lambda iv: float(wisdom.scar.get(iv, 0.0)))
        if float(wisdom.scar.get(best, 0.0)) + _MARGIN < scar_here:
            return best
        return None

    def set_external_reasoner_enabled(self, enabled: bool) -> None:
        """Habilita/inhabilita el razonador externo para el próximo episodio.

        Es una condición NECESARIA pero no suficiente: el scheduler solo agenda
        ``ext_open_thinker`` si además el perfil admitido y el gate lo permiten.
        """
        self._external_reasoner_enabled = bool(enabled)

    def set_resource_signals(self, snapshot: Dict[str, Any] | None) -> None:
        """Inyecta el snapshot de recursos host+GPU del ciclo vital actual.

        Se traduce en señales de presión (cpu/mem/vram/thermal/gpu) dentro del
        contexto de razonamiento para que ``extract_context_features`` y el
        presupuesto reaccionen al hardware real. Snapshot vacío -> sin efecto.
        """
        self._resource_signals = dict(snapshot) if snapshot else {}

    def _resource_context_signals(self) -> Dict[str, Any]:
        """Mapea el snapshot de recursos a las claves que consume el scheduler."""
        snap = self._resource_signals
        if not snap:
            return {}
        signals: Dict[str, Any] = {}
        for key in (
            "cpu_pressure",
            "memory_pressure",
            "vram_pressure",
            "thermal_pressure",
            "gpu_load",
        ):
            if isinstance(snap.get(key), (int, float)):
                signals[key] = float(snap[key])
        if snap.get("gpu_available"):
            signals["gpu_available"] = True
        if isinstance(snap.get("gpu_acceleration"), (int, float)):
            signals["gpu_acceleration_signal"] = float(snap["gpu_acceleration"])
        if isinstance(snap.get("vram_headroom"), (int, float)):
            signals["vram_headroom"] = float(snap["vram_headroom"])
        return signals

    def run_episode(
        self,
        *,
        external_input: float = 0.04,
        replay_unit_id: str | None = None,
        trace_dir: str | Path | None = None,
    ) -> Dict[str, Any]:
        """Ejecuta un episodio cognitivo completo.

        Args:
            external_input: Entrada/perturbación externa para el escenario.

        Returns:
            Dict con episodio, smg_snapshot, reasoning, artifact, certification.
        """
        episode_id = f"episode-{uuid4()}"
        scenario_metadata = self._build_scenario_metadata()
        transition_ordinal = int(self._organism_state.episode_count) + 1
        scenario_seed = getattr(self.scenario, "seed", getattr(self.scenario, "_seed", None))
        resolved_replay_unit_id = replay_unit_id or (
            f"{self.scenario.config.name}/seed-"
            f"{scenario_seed if scenario_seed is not None else 'unavailable'}"
            f"/ep-{transition_ordinal}"
        )
        preaction_logical_time = transition_ordinal * 2 - 1
        trace_root = (
            Path(trace_dir)
            if trace_dir is not None
            else Path(self.storage.config.artifact_root) / "traces"
        )
        episode_trace_dir = trace_root / semantic_segment(resolved_replay_unit_id)
        constraint_registry = ConstraintRegistry(
            replay_unit_id=resolved_replay_unit_id,
            logical_time=preaction_logical_time,
        )
        tracked_solver = TrackedSolver(
            registry=constraint_registry,
            replay_unit_id=resolved_replay_unit_id,
            logical_time=preaction_logical_time,
        )
        acting_collector = ActingTraceCollector(
            replay_unit_id=resolved_replay_unit_id,
            preaction_logical_time=preaction_logical_time,
            trace_dir=episode_trace_dir,
        )

        # 1. Observar escenario
        observation = self.scenario.observe()
        observation_dict = self.scenario.to_observation_dict(observation)
        observation_ref = self.smg.add_observation(observation_dict)

        # 2. Crear signo principal
        main_proposition = self.scenario.get_main_proposition(observation)
        sign_main = self.smg.create_sign(
            proposition=main_proposition,
            observation_id=observation_ref.observation_id,
            metadata={self.scenario.config.main_variable: observation.state.get(
                self.scenario.config.main_variable
            )},
        )

        # 3. Generar y verificar fórmula LOTF
        formula = self.scenario.get_formula(observation)
        ast = self.lotf.parse(formula)
        self.lotf.check(ast, self.scenario.config.type_context)

        # 4. Consultar memoria
        memory_hits = self.memory_retrieval.retrieve(
            run_id=self.run_id,
            query={
                "proposition": main_proposition,
                "alarm": observation.alarm,
            },
            limit=self.memory_retrieval_limit,
            scenario_name=scenario_metadata["scenario_name"],
            scenario_filter_mode=self.memory_filter_mode,
        )

        # 5. Seleccionar intervención
        intervention = self.scenario.select_intervention(observation)
        if memory_hits:
            top = memory_hits[0].get("structure", {})
            if top.get("relation_kind") == "support" and observation.alarm:
                # Memoria soporta la intervención por alarma
                intervention = self.scenario.select_intervention(observation)
        # E3 — sabiduría ∝ daño: si esta situación ya lo hirió con esta intervención
        # y hay una alternativa con claramente menos dolor recordado, la evita (no
        # repetir errores). Fuerza proporcional a la cicatriz. RNFE_EXPERIENCE off ⇒ no-op.
        self._experience_bias = None
        if self._experience is not None:
            alternative = self._experience_biased_intervention(observation, intervention)
            if alternative is not None and alternative != intervention:
                self._experience_bias = {"avoided": intervention, "chose": alternative}
                intervention = alternative
        baseline_intervention = intervention
        acting_collector.set_baseline_action(baseline_intervention)

        # 6. Simular contrafactual (sin intervención o con opuesta)
        counter_intervention = (
            self.scenario.config.interventions[1]
            if len(self.scenario.config.interventions) > 1
            else self.scenario.config.interventions[0]
        )
        counterfactual = self.scenario.simulate_counterfactual(
            intervention=counter_intervention,
            external_input=external_input,
        )

        # 7. Computar la transición factual (greedy) por SIMULACIÓN (sin mutar). La
        # acción FINAL se aplica una sola vez tras la decisión de override (paso 9c), de
        # modo que el override REEMPLACE al greedy desde el estado pre-acción — no lo
        # apile — y los efectos colaterales ocultos del greedy (p.ej. deuda acumulada)
        # no se comprometan si la acción fue reemplazada.
        factual = self.scenario.simulate_counterfactual(
            intervention=intervention,
            external_input=external_input,
        )

        # 8. Crear signo de intervención y relación
        intervention_proposition = self.scenario.get_intervention_proposition(intervention)
        sign_intervention = self.smg.create_sign(
            proposition=intervention_proposition,
            observation_id=observation_ref.observation_id,
            metadata={"intervention": intervention},
        )

        relation_kind = self.scenario.evaluate_relation_kind(
            factual=factual,
            counterfactual=counterfactual,
        )
        relation = self.smg.link_signs(
            source_sign_id=sign_main.sign_id,
            target_sign_id=sign_intervention.sign_id,
            kind=relation_kind,
            metadata={
                f"factual_{self.scenario.config.main_variable}": factual.state.get(
                    self.scenario.config.main_variable
                ),
                f"counterfactual_{self.scenario.config.main_variable}": counterfactual.state.get(
                    self.scenario.config.main_variable
                ),
            },
        )
        counterfactual_dict = self.scenario.to_transition_dict(counterfactual)
        updated_world = self.scenario.to_transition_dict(factual)
        belief_input = asdict(self._previous_belief) if self._previous_belief else None
        causal_attestation = build_causal_attestation(
            scenario_name=scenario_metadata["scenario_name"],
            scenario_version=scenario_metadata.get("scenario_version"),
            main_variable=self.scenario.config.main_variable,
            intervention=intervention,
            observation=observation_dict,
            factual=updated_world,
            counterfactual=counterfactual_dict,
            relation_kind=relation_kind,
            signature=self.scenario.causal_signature,
        )

        # 9. Ejecutar scheduler de razonamiento
        reasoning_context = build_reasoning_context(
            episode_id=episode_id,
            run_id=self.run_id,
            observation=observation_dict,
            intervention=intervention,
            formula=formula,
            memory_hits=memory_hits,
            counterfactual=counterfactual_dict,
            updated_world=updated_world,
            relation_kind=relation_kind,
            scenario=self.scenario.config.name,
            scenario_metadata=scenario_metadata,
            belief_state=belief_input,
            closure_profile=self.closure_profile,
            reasoning_mode=self.reasoning_mode,
            extra_signals={
                "memory_filter_mode": self.memory_filter_mode,
                "causal_attestation": causal_attestation,
                **self._resource_context_signals(),
                **self._causal_context_signals(),
            },
        )
        # Objetos operativos privados: META los conserva para DED pero los excluye
        # de su snapshot serializable, evitando una segunda representación.
        reasoning_context["_constraint_registry"] = constraint_registry
        reasoning_context["_tracked_solver"] = tracked_solver
        reasoning_context["_replay_unit_id"] = resolved_replay_unit_id
        reasoning_context["_preaction_logical_time"] = preaction_logical_time
        if self.mci_active:
            reasoning_context["family_profile"] = self.family_profile
            reasoning_context["regime_hint"] = self._trajectory_regime_label
            reasoning_context["_mci_runtime"] = self._mci_runtime
            reasoning_context["_mci_external_input"] = float(external_input)
            if self._hypothesis_provider is not None and self._mci_runtime is not None:
                try:
                    provider_context = {
                        "state": {
                            key: value
                            for key, value in observation_dict.items()
                            if key not in {"propositions", "level"}
                        },
                        "candidate_action": intervention,
                        "spec_id": self._mci_runtime.base_spec.spec_id,
                        "recent_evidence": self._mci_runtime.learner.get_recent_evidence(10),
                        "logical_time": preaction_logical_time,
                    }
                    raw_hypotheses = tuple(
                        self._hypothesis_provider.infer_hypotheses(provider_context)
                    )
                    hypotheses = adapt_hypotheses(
                        raw_hypotheses,
                        registry=self._hypothesis_mappings,
                        spec=self._mci_runtime.base_spec,
                        recent_evidence=provider_context["recent_evidence"],
                        logical_time=preaction_logical_time,
                    )
                    for hypothesis in hypotheses:
                        self._provider_hypothesis_metadata[
                            hypothesis.hypothesis_id
                        ] = {
                            "provider": hypothesis.provider,
                            "model_ref": hypothesis.model_ref,
                        }
                    self.ingest_mci_hypotheses(hypotheses)
                except Exception:
                    # El proveedor es evidence-only: jamás bloquea razonamiento o acción.
                    pass
        overlay_directives: Dict[str, str] | None = None
        if self._reward_guided is not None:
            overlay_directives = self._reward_guided.directives(
                self.run_id, regime=self._trajectory_regime_label
            )
            reasoning_context["overlay_directives"] = overlay_directives
        # Reglas transferidas por la ecología (modo reasoning_policy_plus_rules):
        # IND las consulta en su rama a-priori. Sin ecología quedan en None.
        if self._inherited_rules:
            reasoning_context["inherited_rules"] = self._inherited_rules
        # C4: tier_3 pidió el razonador externo. Solo se solicita el perfil admitido
        # cuando además RNFE_EXTERNAL_REASONER_RUNTIME está on; el scheduler agenda
        # ext_open_thinker solo si el régimen valida la admisión, y degrada si no
        # (nunca crashea). Sin ambos flags -> perfil nominal (byte-idéntico).
        if (
            not self.mci_active
            and self._external_reasoner_enabled
            and _external_reasoner_runtime_flag()
        ):
            reasoning_context["family_profile"] = "core_plus_external_reasoner_gated_v1"
            reasoning_context.setdefault("regime_hint", self._trajectory_regime_label)
        reasoning = self.scheduler.run(reasoning_context)
        core_report = constraint_registry.latest_core_report
        if core_report is None:
            core_report = constraint_registry.get_core_report(
                status="UNKNOWN",
                core_ids=(),
                all_ids=(),
            )
        acting_collector.set_ded_report(core_report)
        opt_report = None
        if isinstance((reasoning.get("state") or {}).get("opt_choice"), dict):
            opt_report = build_optimization_report(
                reasoning_context,
                reasoning.get("state") or {},
                resolved_replay_unit_id,
                preaction_logical_time,
            )
        acting_collector.set_opt_report(opt_report)
        if self.mci_active:
            mci_evidence = (reasoning.get("state") or {}).get("mci_commit_report")
            if isinstance(mci_evidence, dict):
                sealed_mci_evidence = dict(mci_evidence)
                n4_evidence = getattr(
                    self._hypothesis_provider, "last_evidence", None
                )
                if (
                    self.family_profile == "mci_n4_closed_loop_v1"
                    and isinstance(n4_evidence, dict)
                ):
                    sealed_mci_evidence["n4_evidence"] = n4_evidence
                acting_collector.set_mci_evidence(sealed_mci_evidence)
        try:
            constraint_registry.export_jsonl(acting_collector.paths["constraints"])
            if core_report is not None:
                persist_core_report(core_report, acting_collector.paths["core_report"])
        except OSError:
            acting_collector.mark_persistence_degraded()

        # 9b. Override determinista guardado (actuación del razonamiento). Gated por
        # RNFE_REASONING_ACTUATES=1 (sombra OFF ⇒ camino nominal byte-idéntico). En
        # conflicto causal-contrafactual, si una familia recomienda la alterna y la
        # guarda certifica que es mejor (el contrafactual ya está simulado), se adopta.
        intervention_override, candidate_transition = self._maybe_override_intervention(
            reasoning_state=reasoning.get("state") or {},
            greedy_intervention=intervention,
            factual=factual,
            external_input=external_input,
        )
        if intervention_override.fired and candidate_transition is not None:
            # Conmutar: la alterna recomendada (simulada fresca) pasa a ser la
            # factual; el resultado greedy queda como contrafactual.
            counterfactual = factual
            counter_intervention = intervention
            factual = candidate_transition
            intervention = intervention_override.to_intervention
            relation_kind = self.scenario.evaluate_relation_kind(
                factual=factual, counterfactual=counterfactual
            )
            intervention_proposition = self.scenario.get_intervention_proposition(intervention)
            sign_intervention = self.smg.create_sign(
                proposition=intervention_proposition,
                observation_id=observation_ref.observation_id,
                metadata={"intervention": intervention, "via": "override"},
            )
            relation = self.smg.link_signs(
                source_sign_id=sign_main.sign_id,
                target_sign_id=sign_intervention.sign_id,
                kind=relation_kind,
                metadata={
                    f"factual_{self.scenario.config.main_variable}": factual.state.get(
                        self.scenario.config.main_variable
                    ),
                    f"counterfactual_{self.scenario.config.main_variable}": counterfactual.state.get(
                        self.scenario.config.main_variable
                    ),
                    "intervention_override": True,
                },
            )
            counterfactual_dict = self.scenario.to_transition_dict(counterfactual)
            updated_world = self.scenario.to_transition_dict(factual)
            causal_attestation = build_causal_attestation(
                scenario_name=scenario_metadata["scenario_name"],
                scenario_version=scenario_metadata.get("scenario_version"),
                main_variable=self.scenario.config.main_variable,
                intervention=intervention,
                observation=observation_dict,
                factual=updated_world,
                counterfactual=counterfactual_dict,
                relation_kind=relation_kind,
                signature=self.scenario.causal_signature,
            )
            self.storage.append_event(
                event_type="reasoning.intervention_override",
                run_id=self.run_id,
                source="scenario_episode_runner",
                payload={"episode_id": episode_id, **intervention_override.to_dict()},
            )

        reasoning_state = reasoning.get("state") or {}
        conflict_evidence = {
            "cau": reasoning_state.get("cau_link"),
            "ctf": reasoning_state.get("ctf_checked"),
        }
        guard_report = CausalGuardReport(
            fired=intervention_override.fired,
            from_intervention=intervention_override.from_intervention,
            to_intervention=intervention_override.to_intervention,
            margin_gain=float(intervention_override.margin_gain),
            guard_reason=intervention_override.guard_reason,
            conflict_evidence=conflict_evidence,
        )
        acting_collector.set_guard_report(guard_report)
        acting_trace = None
        governance_envelope = dict(reasoning.get("governance") or {})
        traceability = governance_envelope.get("traceability")
        if isinstance(traceability, dict):
            governance_envelope["traceability"] = {
                key: value for key, value in traceability.items() if key != "run_id"
            }
        try:
            acting_trace = acting_collector.seal(
                committed_action=intervention,
                governance_verdict={
                    "verdict": "admitted",
                    "envelope": governance_envelope,
                },
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            acting_collector.mark_persistence_degraded()

        # 9c. Aplicar la acción FINAL una sola vez desde el estado pre-acción. Hasta aquí
        # nada mutó el escenario (greedy y candidatas se computaron por simulación); esta
        # es la ÚNICA mutación que avanza el mundo, por la intervención efectivamente
        # elegida (greedy o la del override). Byte-idéntico con actuación OFF (final=greedy).
        observed_transition = self.scenario.factual_transition(
            intervention=intervention, external_input=external_input
        )

        # 10. Construir payload de episodio
        factual_delta = float(factual.state.get(self.scenario.config.main_variable, 0.0)) - float(
            observation.state.get(self.scenario.config.main_variable, 0.0)
        )
        counterfactual_delta = float(counterfactual.state.get(self.scenario.config.main_variable, 0.0)) - float(
            observation.state.get(self.scenario.config.main_variable, 0.0)
        )
        episode_payload = {
            "episode_id": episode_id,
            "timestamp": utc_now_iso(),
            "scenario": self.scenario.config.name,
            "scenario_metadata": scenario_metadata,
            "closure_profile": self.closure_profile,
            "context": {
                "observation": observation_dict,
                "formula": formula,
                "intervention": intervention,
                "counterfactual": counterfactual_dict,
                "retrieved_memory": memory_hits,
                "memory_rag_attestation": reasoning_context.get("memory_rag_attestation"),
                "memory_filter_mode": self.memory_filter_mode,
                "causal_attestation": causal_attestation,
                "closure_profile": self.closure_profile,
            },
            "result": {
                "updated_world": updated_world,
                "relation_kind": relation_kind,
                "reasoning_sequence": reasoning["sequence"],
                "factual_delta": factual_delta,
                "counterfactual_delta": counterfactual_delta,
                "intervention_effect": relation_kind,
                "alarm_transition": observation.alarm,
            },
            "trace": reasoning["trace"],
        }

        # 11. Persistir evento de cierre. B41: el sobre CausalContext viaja como clave
        # aditiva (gated). Ausente ⇒ episode_payload byte-idéntico a pre-B41.
        if self._causal_context is not None:
            episode_payload["causal_context"] = self._causal_context
        self.storage.append_event(
            event_type="episode.closed",
            payload=episode_payload,
            run_id=self.run_id,
            source="scenario_episode_runner",
        )

        # 12. Materializar artifact
        artifact_blob = json.dumps(
            {
                "episode": episode_payload,
                "smg_snapshot": self.smg.snapshot(),
                "relation": asdict(relation),
            },
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
        )
        artifact = self.storage.materialize_artifact(
            run_id=self.run_id,
            kind="episode_report",
            content=artifact_blob,
            filename=f"{episode_id}.json",
            metadata={
                "episode_id": episode_id,
                "scenario": self.scenario.config.name,
                "scenario_metadata": scenario_metadata,
            },
        )

        episode_result = {
            "episode": episode_payload,
            "smg_snapshot": self.smg.snapshot(),
            "reasoning": reasoning,
            "artifact": asdict(artifact),
            "run_id": self.run_id,
            "intervention_override": intervention_override.to_dict(),
            "acting_trace": {
                "sealed_hash": acting_trace.sealed_hash if acting_trace is not None else None,
                "replay_unit_id": resolved_replay_unit_id,
                "preaction_logical_time": preaction_logical_time,
                "trace_status": acting_collector.trace_status,
                "paths": acting_collector.paths,
            },
        }

        # 12b. Build and persist belief state
        current_belief = build_belief_state(episode_result=episode_result)
        belief_prior = self._previous_belief
        episode_result["belief_state"] = {
            "prior": asdict(belief_prior) if belief_prior else None,
            "posterior": asdict(current_belief),
        }
        self._previous_belief = current_belief

        # 12c. T5 SOVEREIGNTY: Transition organism state and append to trajectory
        previous_state = self._organism_state
        # B41: el state_id se ancla al GENOMA (organism_id), no a la corrida (run_id) —
        # coherente con la génesis del kernel (state-0-{organism_id}).
        new_state_id = f"state-{self._organism_state.episode_count + 1}-{self._organism_id}"
        regime = self._trajectory_regime_label

        self._organism_state = transition_organism_state(
            current=self._organism_state,
            episode_result=episode_result,
            regime=regime,
            new_state_id=new_state_id,
            timestamp=utc_now_iso(),
        )

        # Validate and assess viability
        constitutional_validation = self._constitution.validate(self._organism_state)
        viability_assessment = self._viability_kernel.assess(
            state=self._organism_state,
            previous_state=previous_state,
        )

        # Append to trajectory
        self._organism_trajectory.append_point(
            state=self._organism_state,
            regime=regime,
            episode_id=episode_id,
            timestamp=utc_now_iso(),
            constitutional_validation=constitutional_validation,
            viability_margin=viability_assessment.viability_margin,
        )

        # Add trajectory to episode result for certification
        episode_result["organism_trajectory"] = self._organism_trajectory.to_dict()
        episode_result["trajectory_window"] = self._organism_trajectory.get_window(window_size=5).to_dict() if False else None  # Will enable in certification update
        episode_result["constitutional_validation"] = {
            "is_valid": constitutional_validation.is_valid,
            "verdict": constitutional_validation.verdict,
            "hard_violation_count": constitutional_validation.hard_violation_count,
            "soft_violation_count": constitutional_validation.soft_violation_count,
            "margin_to_threshold": constitutional_validation.margin_to_threshold,
        }
        episode_result["viability_assessment"] = {
            "is_viable": viability_assessment.is_viable,
            "viability_margin": viability_assessment.viability_margin,
            "distance_to_edge": viability_assessment.distance_to_edge,
            "rollback_required": viability_assessment.rollback_required,
        }

        # 13. Certificación
        certification = self.promotion_gate.process_episode(
            run_id=self.run_id,
            episode_result=episode_result,
        )

        # 13b. R2 — lazo de autoevolución (ρₜ): el organismo observa su propio
        # certificado y decide si proponerse una modificación, monitorear una
        # activa, o ejecutar rollback al último checkpoint sano.
        evolution = self._autoevolution.observe_episode(
            organism_state=self._organism_state,
            episode_result=episode_result,
            certificate_metadata=certification["certificate"].metadata,
            certificate_verdict=certification["certificate"].verdict,
        )
        restored_state = evolution.pop("restored_state", None)
        if restored_state is not None:
            self._organism_state = restored_state
        episode_result["autoevolution"] = evolution
        episode_result["lineage"] = self._lineage.to_dict()

        # 13c. R3 — recompensa semi-Markov del razonamiento: el escalar de control
        # r = ΔIoC − λE·(coste/presupuesto) − λB·B_safe, reusando ΔIoC y B_safe del
        # certificado (R1) y el coste del trace. Se adjunta, persiste y — con
        # RNFE_REWARD_GUIDED_SELECTION=1 — GOBIERNA la ecología opcional del
        # siguiente episodio (selector guiado-por-recompensa).
        from runtime.reasoning.scheduler_meta.reward import (
            compute_episode_reward,
            reasoning_cost_from_trace,
        )

        cert_meta = certification["certificate"].metadata or {}
        cert_risk_plus = cert_meta.get("risk_plus") or {}
        # Efectividad del mundo: margen de seguridad del resultado factual
        # (committed, post-override) en la dirección de optimización. Cierra la
        # ceguera de ΔIoC*; pesa solo con RNFE_REWARD_LAMBDA_EFFECTIVENESS>0.
        try:
            effectiveness = outcome_effectiveness(
                value=float(observed_transition.state.get(self.scenario.config.main_variable, 0.0)),
                alarm_threshold=float(self.scenario.config.alarm_threshold),
                alarm_semantics=str(self.scenario.causal_signature.alarm_semantics),
            )
        except Exception:
            effectiveness = None
        try:
            observed_dict = self.scenario.to_transition_dict(observed_transition)
            observed_value = float(
                observed_transition.state.get(self.scenario.config.main_variable, 0.0)
            )
            predicted_value = float(
                factual.state.get(self.scenario.config.main_variable, 0.0)
            )
            certificate = certification["certificate"]
            decision = certification["decision"]
            certification_fingerprint = sealed_sha256(
                {
                    "replay_unit_id": resolved_replay_unit_id,
                    "verdict": certificate.verdict,
                    "promotion_candidate": certificate.promotion_candidate,
                    "ioc_proxy": certificate.ioc_proxy,
                    "risk_score": certificate.risk_score,
                    "decision_verdict": decision.verdict,
                }
            )
            outcome_link = acting_collector.link_outcome(
                outcome_observed=observed_dict,
                prediction_error=round(observed_value - predicted_value, 6),
                utility=round(effectiveness, 6) if effectiveness is not None else None,
                certification_ref=f"sha256:{certification_fingerprint}",
            )
            episode_result["acting_trace"]["outcome_link"] = outcome_link.to_dict()
            episode_result["acting_trace"]["certificate_id"] = certificate.certificate_id
        except (OSError, RuntimeError, TypeError, ValueError):
            acting_collector.mark_persistence_degraded()
        episode_result["acting_trace"]["trace_status"] = acting_collector.trace_status
        # ν = cau.helps_goal (¿la acción factual va en la dirección del objetivo?),
        # ya direction-aware desde core_inference. Criterio de viabilidad de primera
        # clase (cura J(h|X)); pesa solo con RNFE_REWARD_LAMBDA_NU>0.
        nu_helps_goal = ((reasoning.get("state") or {}).get("cau_link") or {}).get("helps_goal")
        reasoning_reward = compute_episode_reward(
            delta_ioc=cert_risk_plus.get("delta_ioc"),
            delta_ioc_star=(cert_meta.get("omega") or {}).get("delta_ioc_star"),
            reasoning_cost=reasoning_cost_from_trace(reasoning.get("trace") or []),
            cost_budget=reasoning.get("effective_max_steps"),
            b_safe=cert_risk_plus.get("b_safe"),
            effectiveness=effectiveness,
            nu=nu_helps_goal,
        )
        episode_result["reasoning_reward"] = reasoning_reward
        if self.mci_active and self._mci_runtime is not None:
            try:
                mci_outcome = self._mci_runtime.observe_outcome(
                    {
                        **self.scenario.to_transition_dict(observed_transition),
                        self._mci_runtime.base_spec.alarm_variable: bool(
                            observed_transition.alarm
                        ),
                    },
                    committed_action=intervention,
                    logical_time=preaction_logical_time + 1,
                    reasoning_cost=reasoning_cost_from_trace(reasoning.get("trace") or []),
                    decision_trace_sha256=(
                        acting_trace.sealed_hash if acting_trace is not None else None
                    ),
                )
                episode_result["mci_outcome"] = mci_outcome
                self.storage.append_event(
                    event_type="mci.outcome",
                    run_id=self.run_id,
                    source=self.family_profile,
                    payload={"episode_id": episode_id, **mci_outcome},
                )
                promoted_overlay = mci_outcome.get("promoted_overlay")
                if isinstance(promoted_overlay, dict):
                    self.storage.append_event(
                        event_type="mci.overlay.promoted",
                        run_id=self.run_id,
                        source=self.family_profile,
                        payload={"episode_id": episode_id, **promoted_overlay},
                    )
                ledger_payload = mci_outcome.get("hypothesis_ledger")
                if isinstance(ledger_payload, list):
                    self.storage.append_event(
                        event_type="mci.hypothesis.ledger",
                        run_id=self.run_id,
                        source=self.family_profile,
                        payload={
                            "episode_id": episode_id,
                            "entries": ledger_payload,
                            "logical_time": preaction_logical_time + 1,
                        },
                    )
                transfer_payload = mci_outcome.get("transfer_beliefs")
                if isinstance(transfer_payload, list):
                    self.storage.append_event(
                        event_type="mci.transfer.ledger",
                        run_id=self.run_id,
                        source=self.family_profile,
                        payload={
                            "episode_id": episode_id,
                            "entries": transfer_payload,
                            "logical_time": preaction_logical_time + 1,
                        },
                    )
                if self._hypothesis_provider is not None:
                    try:
                        evaluation_payload = tuple(
                            item
                            for item in (
                                mci_outcome.get("neural_hypotheses_evaluated") or ()
                            )
                            if isinstance(item, dict)
                        )
                        evaluations = relay_feedback(
                            self._hypothesis_provider, evaluation_payload
                        )
                        if self.family_profile == "mci_n4_closed_loop_v1":
                            mci_outcome["n4_calibration_reports"] = list(
                                getattr(
                                    self._hypothesis_provider,
                                    "last_calibration_reports",
                                    (),
                                )
                            )
                            for report in mci_outcome["n4_calibration_reports"]:
                                self.storage.append_event(
                                    event_type="n4.calibration.updated",
                                    run_id=self.run_id,
                                    source="n4-causal-ranker",
                                    payload={
                                        "episode_id": episode_id,
                                        **report,
                                        "logical_time": preaction_logical_time + 1,
                                    },
                                )
                            calibration = getattr(
                                self._hypothesis_provider, "calibration", None
                            )
                            if calibration is not None:
                                self.storage.append_event(
                                    event_type="n4.calibration.snapshot",
                                    run_id=self.run_id,
                                    source="n4-causal-ranker",
                                    payload={
                                        "episode_id": episode_id,
                                        "snapshot": calibration.snapshot(),
                                        "logical_time": preaction_logical_time + 1,
                                    },
                                )
                        for evaluation in evaluations:
                            metadata = self._provider_hypothesis_metadata.get(
                                evaluation.hypothesis_id, {}
                            )
                            self.storage.append_event(
                                event_type="neural.hypothesis.evaluated",
                                run_id=self.run_id,
                                source="mci_hypothesis_provider",
                                payload={
                                    "episode_id": episode_id,
                                    "provider": metadata.get("provider", "external"),
                                    "model_ref": metadata.get(
                                        "model_ref", "unavailable"
                                    ),
                                    **evaluation.to_dict(),
                                    "logical_time": preaction_logical_time + 1,
                                },
                            )
                            if evaluation.status != "pending":
                                self._provider_hypothesis_metadata.pop(
                                    evaluation.hypothesis_id, None
                                )
                    except Exception:
                        # Feedback y telemetría son fail-open y no cambian el outcome.
                        pass
            except (KeyError, OSError, RuntimeError, TypeError, ValueError):
                episode_result["mci_outcome"] = {
                    "status": "persistence_degraded",
                }
        executed_overlays = [
            family.lower()
            for family in (reasoning.get("sequence") or [])
            if family.lower() not in {"abd", "ana", "cau", "ctf", "ded", "prob"}
        ]
        if self._reward_guided is not None:
            self._reward_guided.observe(
                run_id=self.run_id,
                reward_block=reasoning_reward,
                executed_sequence=reasoning.get("sequence") or [],
                regime=regime,
            )
            episode_result["reward_guided"] = {
                "directives": overlay_directives or {},
                "executed_overlays": executed_overlays,
                **self._reward_guided.summary(self.run_id, regime=regime),
            }
        self.storage.append_event(
            event_type="reasoning.reward",
            run_id=self.run_id,
            source="meta_scheduler",
            payload={
                "episode_id": episode_id,
                # Overlays activos + régimen: permiten re-sembrar y estratificar la
                # evidencia del selector guiado-por-recompensa entre runners.
                "optional_overlays_active": executed_overlays,
                "regime_label": regime,
                **reasoning_reward,
            },
        )

        # E1 — Experiencia: destilar ESTE episodio (éxito o golpe) en el diario del
        # organismo, con firma de situación y severidad ∝ daño. RNFE_EXPERIENCE off ⇒ skip.
        if self._experience is not None:
            from runtime.organism.experience import build_experience

            cert = certification["certificate"]
            va = episode_result.get("viability_assessment") or {}
            prev_vm = float(getattr(getattr(previous_state, "viability", None), "viability_margin", va.get("viability_margin", 1.0)) or 1.0)
            vm = float(va.get("viability_margin", prev_vm))
            exp = build_experience(
                organism_id=self._organism_id,
                run_id=self.run_id,
                episode_id=episode_id,
                scenario=self.scenario.config.name,
                regime=("alarm" if getattr(observation, "alarm", False) else "calm"),
                main_variable=self.scenario.config.main_variable,
                causal_status="",
                intervention=intervention,
                viability_margin=vm,
                ioc=float(getattr(cert, "ioc_proxy", 0.0) or 0.0),
                risk=float(getattr(cert, "risk_score", 0.0) or 0.0),
                reward=float(reasoning_reward.get("reward", 0.0) or 0.0),
                action="act",
                certified=(getattr(cert, "verdict", None) == "certified"),
                closure_passed=bool((va.get("is_viable", True))),
                viability_delta=vm - prev_vm,
            )
            self._experience.record(exp)
            episode_result["experience"] = {
                "situation_key": exp.situation_key,
                "severity": exp.severity,
                "wound": exp.wound,
                "biased": self._experience_bias,
            }
            if self._experience_bias is not None:
                self.storage.append_event(
                    event_type="experience.applied",
                    run_id=self.run_id,
                    source="experience",
                    payload={"episode_id": episode_id, "situation_key": exp.situation_key,
                             **self._experience_bias},
                )

        # 14. EML shadow (opcional)
        eml_shadow = {"enabled": False, "status": "disabled"}
        if self.eml_mode == "shadow":
            dataset = self._build_eml_dataset(
                observation=observation_dict,
                factual=self.scenario.to_transition_dict(factual),
                counterfactual=self.scenario.to_transition_dict(counterfactual),
            )
            eml_out = self.eml_runner.run_shadow(
                run_id=self.run_id,
                episode_id=episode_id,
                rows=dataset,
            )
            top = eml_out["run"]["top_candidates"]
            eml_shadow = {
                "enabled": True,
                "status": "ok",
                "eml_run_id": eml_out["run"]["eml_run_id"],
                "candidate_count": eml_out["run"]["candidate_count"],
                "top_composite": top[0]["composite_score"] if top else 0.0,
                "top_expr_signature": str(top[0]["expr"]) if top else "",
                "artifacts": eml_out["artifacts"],
            }
            episode_result["episode"]["context"]["eml_shadow"] = {
                "eml_run_id": eml_shadow["eml_run_id"],
                "candidate_count": eml_shadow["candidate_count"],
                "top_composite": eml_shadow["top_composite"],
                "top_expr_signature": eml_shadow["top_expr_signature"],
            }

        return {
            **episode_result,
            "certification": {
                "certificate_id": certification["certificate"].certificate_id,
                "verdict": certification["certificate"].verdict,
                "promotion_candidate": certification["certificate"].promotion_candidate,
                "decision_verdict": certification["decision"].verdict,
            },
            "eml_shadow": eml_shadow,
        }
