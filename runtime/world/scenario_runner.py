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
from runtime.organism.constitution import OrganismConstitution
from runtime.organism.state import OrganismState, IdentityState, transition_organism_state
from runtime.organism.trajectory import OrganismTrajectory
from runtime.organism.viability import ViabilityKernel
from runtime.reasoning.context import build_reasoning_context, resolve_reasoning_mode
from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler
from runtime.reality.belief_state import BeliefState, build_belief_state
from runtime.smg import SMGMin
from runtime.storage import get_storage
from runtime.storage.records import utc_now_iso
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
        self._organism_state = OrganismState(
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
            limit=3,
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
        reasoning = self.scheduler.run(
            build_reasoning_context(
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
