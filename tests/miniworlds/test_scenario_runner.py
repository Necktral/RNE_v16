"""Tests para escenarios cognitivos parametrizables."""

from pathlib import Path

import pytest

from runtime.storage import StorageConfig, StorageFactory
from runtime.world import (
    CognitiveScenario,
    ThermalScenario,
    ResourceScenario,
    ScenarioEpisodeRunner,
    get_scenario,
    list_scenarios,
    SCENARIO_REGISTRY,
    DEFAULT_SCENARIO,
)


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "scenario.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


#: Argumento de arranque de cada escenario (su variable principal).
_INITIAL_KWARG = {
    ThermalScenario: "initial_temperature",
    ResourceScenario: "initial_stock",
}


def _relation(cls, initial, factual_intervention, counter_intervention, external):
    """`relation_kind` del contraste entre dos acciones desde el MISMO estado inicial.

    Ambas ramas se simulan sobre instancias frescas: el contrafactual debe contrastar con
    el factual desde el estado pre-acción, no arrastrar la mutación de la otra rama.
    """
    scenario = cls(**{_INITIAL_KWARG[cls]: initial})
    factual = scenario.simulate_counterfactual(
        intervention=factual_intervention, external_input=external
    )
    counterfactual = scenario.simulate_counterfactual(
        intervention=counter_intervention, external_input=external
    )
    return scenario.evaluate_relation_kind(factual=factual, counterfactual=counterfactual)


class TestScenarioRegistry:
    """Tests para el registro de escenarios."""

    def test_thermal_scenario_is_registered(self):
        """Escenario térmico está registrado."""
        assert "thermal_homeostasis" in SCENARIO_REGISTRY

    def test_resource_scenario_is_registered(self):
        """Escenario de recursos está registrado."""
        assert "resource_management" in SCENARIO_REGISTRY

    def test_default_scenario_is_thermal(self):
        """El escenario por defecto es thermal_homeostasis."""
        assert DEFAULT_SCENARIO == "thermal_homeostasis"

    def test_get_scenario_returns_instance(self):
        """get_scenario retorna instancia del escenario."""
        scenario = get_scenario("thermal_homeostasis")
        assert isinstance(scenario, CognitiveScenario)
        assert isinstance(scenario, ThermalScenario)

    def test_get_scenario_with_kwargs(self):
        """get_scenario acepta kwargs de configuración."""
        scenario = get_scenario("thermal_homeostasis", initial_temperature=0.9)
        obs = scenario.observe()
        assert obs.state["temperature"] == 0.9

    def test_get_scenario_raises_for_unknown(self):
        """get_scenario lanza error para escenario desconocido."""
        with pytest.raises(ValueError, match="no encontrado"):
            get_scenario("unknown_scenario")

    def test_list_scenarios_returns_configs(self):
        """list_scenarios retorna configuraciones."""
        configs = list_scenarios()
        assert "thermal_homeostasis" in configs
        assert "resource_management" in configs
        assert configs["thermal_homeostasis"].main_variable == "temperature"
        assert configs["resource_management"].main_variable == "stock_level"


class TestThermalScenario:
    """Tests para el escenario térmico."""

    def test_observe_returns_observation(self):
        """observe() retorna ScenarioObservation."""
        scenario = ThermalScenario()
        obs = scenario.observe()
        assert "temperature" in obs.state
        assert isinstance(obs.propositions, list)

    def test_factual_transition_updates_state(self):
        """factual_transition() actualiza estado."""
        scenario = ThermalScenario(initial_temperature=0.9)
        obs_before = scenario.observe()
        result = scenario.factual_transition(intervention="activate_cooling", external_input=0.03)
        obs_after = scenario.observe()

        assert result.state["cooling_active"] is True
        assert obs_after.state["temperature"] < obs_before.state["temperature"]

    def test_counterfactual_does_not_mutate_state(self):
        """simulate_counterfactual() no muta estado."""
        scenario = ThermalScenario(initial_temperature=0.9)
        obs_before = scenario.observe()
        _ = scenario.simulate_counterfactual(intervention="activate_cooling", external_input=0.03)
        obs_after = scenario.observe()

        assert obs_before.state["temperature"] == obs_after.state["temperature"]

    def test_get_formula_returns_template(self):
        """get_formula() retorna plantilla LOTF."""
        scenario = ThermalScenario()
        obs = scenario.observe()
        formula = scenario.get_formula(obs)
        assert "TEMP_HIGH" in formula
        assert "ACTIVATE_COOLING" in formula

    def test_support_when_only_the_factual_stays_safe(self):
        """support = el factual mantuvo seguro donde el contrafactual HABRÍA ROTO.

        0.86 en alarma (umbral 0.85), calor 0.04:
          enfriar   -> 0.86 + 0.04 - 0.07 = 0.83  (SEGURO)
          no actuar -> 0.86 + 0.04        = 0.90  (ROTO)
        El contrafactual DISCRIMINA y la acción elegida es la que salva: evidencia ganada.
        """
        kind = _relation(ThermalScenario, 0.86, "activate_cooling", "deactivate_cooling", 0.04)
        assert kind == "support"

    def test_contradiction_when_only_the_factual_breaks(self):
        """contradiction = el factual rompió donde el contrafactual habría mantenido seguro.

        0.82 SIN alarma, calor 0.04. La política reactiva no actúa (correcto según su
        propia regla) y aun así cruza el umbral:
          no actuar -> 0.82 + 0.04        = 0.86  (ROTO: >= 0.85)
          enfriar   -> 0.82 + 0.04 - 0.07 = 0.79  (SEGURO)
        Evidencia REAL de que su modelo causal falla: su política llega un paso tarde.
        """
        kind = _relation(ThermalScenario, 0.82, "deactivate_cooling", "activate_cooling", 0.04)
        assert kind == "contradiction"

    def test_no_discriminating_evidence_when_both_stay_safe(self):
        """Ambas acciones dejan al organismo seguro ⇒ el contrafactual NO ENSEÑA NADA.

        0.50, calor 0.04: no actuar -> 0.54 y enfriar -> 0.47. Las dos bajo el umbral.
        Bajo el monótono viejo esto era `contradiction` (0.54 > 0.47 ⇒ "perdiste"): el
        organismo se acusaba de contradecir su modelo causal por NO actuar estando cómodo.
        No hay soporte causal que medir: se DECLARA, no se puntúa.
        """
        kind = _relation(ThermalScenario, 0.50, "deactivate_cooling", "activate_cooling", 0.04)
        assert kind == "no_discriminating_evidence"

    def test_no_discriminating_evidence_when_both_break(self):
        """Ambas acciones rompen el objetivo ⇒ tampoco hay nada que aprender.

        0.90, calor 0.03: enfriar -> 0.86 y no actuar -> 0.93. Las DOS en alarma.
        Enfriar quedó "más frío", pero no salvó al organismo: contra el objetivo
        regulatorio el contrafactual no discrimina. Bajo el monótono viejo esto era
        `support` — un soporte causal 0.90 afirmado en un episodio donde la intervención
        NO logró el objetivo.
        """
        kind = _relation(ThermalScenario, 0.90, "activate_cooling", "deactivate_cooling", 0.03)
        assert kind == "no_discriminating_evidence"


class TestResourceScenario:
    """Tests para el escenario de recursos."""

    def test_observe_returns_observation(self):
        """observe() retorna ScenarioObservation."""
        scenario = ResourceScenario()
        obs = scenario.observe()
        assert "stock_level" in obs.state
        assert isinstance(obs.propositions, list)

    def test_factual_transition_updates_state(self):
        """factual_transition() actualiza estado."""
        scenario = ResourceScenario(initial_stock=0.15)
        obs_before = scenario.observe()
        result = scenario.factual_transition(intervention="start_production", external_input=0.03)
        obs_after = scenario.observe()

        assert result.state["production_active"] is True
        assert obs_after.state["stock_level"] > obs_before.state["stock_level"]

    def test_counterfactual_does_not_mutate_state(self):
        """simulate_counterfactual() no muta estado."""
        scenario = ResourceScenario(initial_stock=0.15)
        obs_before = scenario.observe()
        _ = scenario.simulate_counterfactual(intervention="start_production", external_input=0.03)
        obs_after = scenario.observe()

        assert obs_before.state["stock_level"] == obs_after.state["stock_level"]

    def test_inverse_causality_to_thermal(self):
        """Recursos tiene causalidad inversa al térmico (LOW -> ACTIVATE)."""
        scenario = ResourceScenario(initial_stock=0.15, scarcity_threshold=0.20)
        obs = scenario.observe()

        # En escasez, debe activar producción
        assert obs.alarm is True
        intervention = scenario.select_intervention(obs)
        assert intervention == "start_production"

    def test_support_for_resources_without_inverting_any_monotone(self):
        """El MISMO criterio vale en recursos, sin override y sin invertir nada.

        Recursos tiene `alarm_semantics='threshold_below'` (escasez = alarma abajo), la
        semántica opuesta a térmico. El criterio no compara valores: compara si cada acción
        deja al organismo DENTRO de su región segura, que cada escenario ya juzga con su
        propia alarma. Por eso `ResourceScenario` ya no necesita el override que antes daba
        vuelta el `<=` en `>=` — duplicaba el mismo error con el signo cambiado.

        0.17 en escasez (umbral 0.20), consumo 0.04:
          producir  -> 0.17 - 0.04 + 0.08 = 0.21  (SEGURO: > 0.20)
          no actuar -> 0.17 - 0.04        = 0.13  (ROTO)
        """
        kind = _relation(ResourceScenario, 0.17, "start_production", "stop_production", 0.04)
        assert kind == "support"

    def test_no_discriminating_evidence_for_resources_when_both_break(self):
        """0.15, consumo 0.03: producir -> 0.20 (aún en escasez) y parar -> 0.12. Ambas rotas.

        Producir dejó MÁS stock, pero no sacó al organismo de la escasez: contra el objetivo
        regulatorio no hay discriminación. El override monótono viejo (`factual >= ctf`)
        cantaba `support` acá — soporte causal 0.90 en un episodio que no logró nada.
        """
        kind = _relation(ResourceScenario, 0.15, "start_production", "stop_production", 0.03)
        assert kind == "no_discriminating_evidence"


class TestScenarioEpisodeRunner:
    """Tests para el runner de episodios con escenarios."""

    def test_runner_with_default_thermal_scenario(self, tmp_path: Path):
        """Runner funciona con escenario térmico por defecto."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(storage=storage, run_id="run-thermal-default")
        result = runner.run_episode(external_input=0.05)

        assert result["episode"]["scenario"] == "thermal_homeostasis"
        assert result["episode"]["result"]["reasoning_sequence"] == [
            "ABD", "ANA", "CAU", "CTF", "DED", "PROB"
        ]
        storage.close()

    def test_runner_with_resource_scenario(self, tmp_path: Path):
        """Runner funciona con escenario de recursos."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-resource",
            scenario="resource_management",
        )
        result = runner.run_episode(external_input=0.03)

        assert result["episode"]["scenario"] == "resource_management"
        assert "stock_level" in result["episode"]["context"]["observation"]
        storage.close()

    def test_runner_with_scenario_instance(self, tmp_path: Path):
        """Runner funciona con instancia de escenario."""
        storage = _storage(tmp_path)
        scenario = ThermalScenario(initial_temperature=0.95)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-instance",
            scenario=scenario,
        )
        result = runner.run_episode(external_input=0.02)

        assert result["episode"]["scenario"] == "thermal_homeostasis"
        storage.close()

    def test_runner_persists_events_and_artifacts(self, tmp_path: Path):
        """Runner persiste eventos y artifacts."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-persist-test",
            scenario="thermal_homeostasis",
        )
        result = runner.run_episode(external_input=0.05)

        # Verificar evento
        events = storage.list_events(run_id="run-persist-test", limit=50)
        closed_events = [e for e in events if e.event_type == "episode.closed"]
        assert len(closed_events) >= 1

        # Verificar artifact
        artifact_path = Path(result["artifact"]["abs_path"])
        assert artifact_path.exists()

        storage.close()

    def test_runner_passes_enriched_reasoning_context(self, tmp_path: Path):
        """Scheduler recibe contexto enriquecido reusable desde ScenarioEpisodeRunner."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-reasoning-context",
            scenario="thermal_homeostasis",
        )
        result = runner.run_episode(external_input=0.05)

        reasoning_context = result["episode"]["trace"][0]["detail"]["reasoning_context"]
        assert reasoning_context["formula"]
        assert reasoning_context["memory_hits"] == reasoning_context["retrieved_memory"]
        assert "counterfactual" in reasoning_context
        assert "updated_world" in reasoning_context
        assert "factual" in reasoning_context
        assert reasoning_context["relation_kind"] in {"support", "contradiction"}
        assert reasoning_context["scenario"] == "thermal_homeostasis"
        assert reasoning_context["scenario_metadata"]["scenario_name"] == "thermal_homeostasis"
        assert reasoning_context["reasoning_mode"] == "fixed"
        ded_step = next(step for step in result["episode"]["trace"] if step["family"] == "DED")
        assert ded_step["detail"]["artifacts"]["solver_result"] == "sat"
        assert ded_step["detail"]["artifacts"]["formula_normalized"] == "TEMP_HIGH -> ACTIVATE_COOLING"
        assert ded_step["detail"]["artifacts"]["z3_expression"]
        storage.close()

    def test_runner_second_episode_includes_prior_belief_state_in_reasoning_context(self, tmp_path: Path):
        """Segundo episodio expone belief_state previo al scheduler."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-belief-context",
            scenario="thermal_homeostasis",
        )
        runner.run_episode(external_input=0.04)
        second = runner.run_episode(external_input=0.04)

        reasoning_context = second["episode"]["trace"][0]["detail"]["reasoning_context"]
        assert "belief_state" in reasoning_context
        assert reasoning_context["belief_state"]["scenario_name"] == "thermal_homeostasis"
        storage.close()

    def test_runner_both_scenarios_produce_valid_episodes(self, tmp_path: Path):
        """Ambos escenarios producen episodios válidos."""
        storage = _storage(tmp_path)

        # Térnico
        runner1 = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-compare-thermal",
            scenario="thermal_homeostasis",
        )
        result1 = runner1.run_episode(external_input=0.05)

        # Recursos
        runner2 = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-compare-resource",
            scenario="resource_management",
        )
        result2 = runner2.run_episode(external_input=0.03)

        # Ambos deben tener episodio cerrado con veredicto válido
        valid_verdicts = ["PASSED", "CONDITIONALLY_PASSED", "certified"]
        assert result1["certification"]["verdict"] in valid_verdicts
        assert result2["certification"]["verdict"] in valid_verdicts

        # Ambos deben tener secuencia de razonamiento
        assert len(result1["episode"]["result"]["reasoning_sequence"]) >= 6
        assert len(result2["episode"]["result"]["reasoning_sequence"]) >= 6

        storage.close()
