"""Runner de episodio cognitivo genérico que soporta múltiples escenarios."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from typing import Any, Dict
from uuid import uuid4

from runtime.certification.promotion_gate import PromotionGate
from runtime.lotf import LOTFMin
from runtime.memory.mfm_lite.retrieval import MemoryRetrieval
from runtime.organism.autoevolution import AutoEvolutionController
from runtime.organism.constitution import OrganismConstitution
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
from runtime.world.intervention_override import (
    OverrideDecision,
    evaluate_override,
    is_actuation_enabled,
    outcome_effectiveness,
)
from runtime.symbolic.eml import EMLRunner

from .scenario import CognitiveScenario, ScenarioObservation
from .registry import get_scenario, DEFAULT_SCENARIO


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

        self.smg = SMGMin(storage=self.storage, run_id=self.run_id)
        self.lotf = LOTFMin()
        self.scheduler = MetaScheduler(trace_store=self.storage, mode=self.reasoning_mode)
        self.memory_retrieval = MemoryRetrieval(storage=self.storage)
        self.promotion_gate = PromotionGate(storage=self.storage)
        self.eml_mode = os.environ.get("RNFE_EML_MODE", "disabled").strip().lower()
        self.eml_runner = EMLRunner(storage=self.storage)
        self._previous_belief: BeliefState | None = None

        # T5 SOVEREIGNTY: Initialize organism trajectory as primary runtime unit
        self._organism_state = organism_state or OrganismState(
            state_id=f"state-0-{self.run_id}",
            timestamp=utc_now_iso(),
            active_regime="unknown",
            episode_count=0,
            identity=IdentityState(
                lineage_id=f"lineage-{self.run_id}",
                constitution_hash="",
            ),
        )
        self._organism_trajectory = OrganismTrajectory(
            organism_id=f"org-{self.run_id}",
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
            self._lineage = LineageState(lineage_id=f"lineage-{self.run_id}")
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
        sim_cache: Dict[str, Any] = {}

        def simulate_value(intervention: str) -> float:
            transition = self.scenario.factual_transition(
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
        )
        candidate = sim_cache.get(decision.to_intervention) if decision.fired else None
        return decision, candidate

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

    def run_episode(self, *, external_input: float = 0.04) -> Dict[str, Any]:
        """Ejecuta un episodio cognitivo completo.

        Args:
            external_input: Entrada/perturbación externa para el escenario.

        Returns:
            Dict con episodio, smg_snapshot, reasoning, artifact, certification.
        """
        episode_id = f"episode-{uuid4()}"
        scenario_metadata = self._build_scenario_metadata()

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

        # 7. Ejecutar transición factual
        factual = self.scenario.factual_transition(
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
        )
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
        reasoning = self.scheduler.run(reasoning_context)

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
            self.storage.append_event(
                event_type="reasoning.intervention_override",
                run_id=self.run_id,
                source="scenario_episode_runner",
                payload={"episode_id": episode_id, **intervention_override.to_dict()},
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

        # 11. Persistir evento de cierre
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
        new_state_id = f"state-{self._organism_state.episode_count + 1}-{self.run_id}"
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
                value=float(factual.state.get(self.scenario.config.main_variable, 0.0)),
                alarm_threshold=float(self.scenario.config.alarm_threshold),
                alarm_semantics=str(self.scenario.causal_signature.alarm_semantics),
            )
        except Exception:
            effectiveness = None
        reasoning_reward = compute_episode_reward(
            delta_ioc=cert_risk_plus.get("delta_ioc"),
            delta_ioc_star=(cert_meta.get("omega") or {}).get("delta_ioc_star"),
            reasoning_cost=reasoning_cost_from_trace(reasoning.get("trace") or []),
            cost_budget=reasoning.get("effective_max_steps"),
            b_safe=cert_risk_plus.get("b_safe"),
            effectiveness=effectiveness,
        )
        episode_result["reasoning_reward"] = reasoning_reward
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
