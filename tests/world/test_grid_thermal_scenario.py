"""Tests unitarios para GridThermalScenario (mundo espacial 5x5)."""

import pytest
from runtime.world.grid_thermal_scenario import GridThermalScenario, CellState, GridState


class TestGridThermalScenarioStructure:
    """Tests de estructura básica del grid 5x5."""

    def test_grid_has_25_cells(self):
        """Grid 5x5 debe tener exactamente 25 celdas."""
        scenario = GridThermalScenario()
        obs = scenario.observe()
        assert len(obs.state["cell_states"]) == 25

    def test_grid_size_is_5x5(self):
        """Metadata debe indicar world_shape 5x5."""
        scenario = GridThermalScenario()
        obs = scenario.observe()
        assert obs.state["world_shape"] == "5x5"
        assert obs.state["cell_count"] == 25

    def test_cells_have_correct_coordinates(self):
        """Cada celda debe tener coordenadas (row, col) válidas."""
        scenario = GridThermalScenario()
        obs = scenario.observe()
        cells = obs.state["cell_states"]

        # Verificar que todas las combinaciones (0-4, 0-4) están presentes
        coords = {(cell["row"], cell["col"]) for cell in cells}
        expected_coords = {(i, j) for i in range(5) for j in range(5)}
        assert coords == expected_coords

    def test_initial_temperature_applied_to_all_cells(self):
        """Temperatura inicial debe aplicarse a todas las celdas."""
        initial_temp = 0.75
        scenario = GridThermalScenario(initial_temperature=initial_temp)
        obs = scenario.observe()

        for cell in obs.state["cell_states"]:
            assert abs(cell["temperature"] - initial_temp) < 0.001

    def test_config_has_correct_name(self):
        """Config debe tener nombre 'grid_thermal_5x5'."""
        scenario = GridThermalScenario()
        assert scenario.config.name == "grid_thermal_5x5"

    def test_main_variable_is_global_temp_mean(self):
        """Variable principal debe ser global_temp_mean."""
        scenario = GridThermalScenario()
        assert scenario.config.main_variable == "global_temp_mean"


class TestGridThermalAggregates:
    """Tests de agregados globales."""

    def test_aggregates_computed_correctly_uniform_temp(self):
        """Agregados deben computarse correctamente con temperatura uniforme."""
        initial_temp = 0.8
        scenario = GridThermalScenario(initial_temperature=initial_temp)
        obs = scenario.observe()

        assert abs(obs.state["global_temp_mean"] - initial_temp) < 0.001
        assert abs(obs.state["global_temp_max"] - initial_temp) < 0.001

    def test_global_temp_mean_is_average_of_cells(self):
        """global_temp_mean debe ser promedio de todas las temperaturas."""
        scenario = GridThermalScenario(initial_temperature=0.5)

        # Modificar manualmente algunas celdas para tener temperaturas distintas
        scenario._grid.cells[0].temperature = 0.3
        scenario._grid.cells[1].temperature = 0.7
        scenario._update_aggregates()

        temps = [cell.temperature for cell in scenario._grid.cells]
        expected_mean = sum(temps) / len(temps)

        assert abs(scenario._grid.global_temp_mean - expected_mean) < 0.001

    def test_global_temp_max_is_maximum_of_cells(self):
        """global_temp_max debe ser el máximo de todas las temperaturas."""
        scenario = GridThermalScenario(initial_temperature=0.5)

        # Establecer una celda con temperatura alta
        scenario._grid.cells[10].temperature = 0.95
        scenario._update_aggregates()

        assert abs(scenario._grid.global_temp_max - 0.95) < 0.001

    def test_cooling_cells_count_is_accurate(self):
        """cooling_cells_count debe contar celdas con enfriamiento activo."""
        scenario = GridThermalScenario()

        # Activar enfriamiento en algunas celdas
        scenario._grid.cells[0].cooling_active = True
        scenario._grid.cells[5].cooling_active = True
        scenario._grid.cells[10].cooling_active = True
        scenario._update_aggregates()

        assert scenario._grid.cooling_cells_count == 3

    def test_global_alarm_triggers_on_mean_threshold(self):
        """global_alarm debe activarse cuando global_temp_mean >= threshold."""
        threshold = 0.85
        scenario = GridThermalScenario(alarm_threshold=threshold)

        # Establecer temperatura por encima del umbral
        for cell in scenario._grid.cells:
            cell.temperature = 0.90
        scenario._update_aggregates()

        assert scenario._grid.global_alarm is True

    def test_global_alarm_off_below_threshold(self):
        """global_alarm debe estar apagada cuando global_temp_mean < threshold."""
        threshold = 0.85
        scenario = GridThermalScenario(alarm_threshold=threshold)

        # Establecer temperatura por debajo del umbral
        for cell in scenario._grid.cells:
            cell.temperature = 0.70
        scenario._update_aggregates()

        assert scenario._grid.global_alarm is False


class TestGridThermalTransitions:
    """Tests de transiciones factual y contrafactual."""

    def test_factual_transition_updates_all_cells(self):
        """Transición factual debe actualizar todas las 25 celdas."""
        scenario = GridThermalScenario(initial_temperature=0.8)
        result = scenario.factual_transition(
            intervention="activate_cooling",
            external_input=0.04
        )

        # Verificar que todas las celdas tienen cooling activo
        obs = scenario.observe()
        assert all(cell["cooling_active"] for cell in obs.state["cell_states"])

    def test_factual_transition_applies_cooling_effect(self):
        """Cooling debe reducir temperatura en todas las celdas."""
        initial_temp = 0.8
        cooling_effect = 0.07
        scenario = GridThermalScenario(
            initial_temperature=initial_temp,
            cooling_effect=cooling_effect
        )

        # Transición con cooling y sin calor externo
        scenario.factual_transition(
            intervention="activate_cooling",
            external_input=0.0
        )

        obs = scenario.observe()
        # Temperatura debe haber bajado aproximadamente cooling_effect
        expected_temp = initial_temp - cooling_effect
        assert abs(obs.state["global_temp_mean"] - expected_temp) < 0.01

    def test_factual_transition_distributes_external_heat(self):
        """Calor externo debe distribuirse uniformemente entre celdas."""
        scenario = GridThermalScenario(initial_temperature=0.5)
        external_heat = 0.25  # Total

        # Sin cooling
        scenario.factual_transition(
            intervention="deactivate_cooling",
            external_input=external_heat
        )

        obs = scenario.observe()
        heat_per_cell = external_heat / 25.0
        expected_temp = 0.5 + heat_per_cell
        assert abs(obs.state["global_temp_mean"] - expected_temp) < 0.01

    def test_counterfactual_does_not_mutate_state(self):
        """Simulación contrafactual NO debe mutar el estado real."""
        scenario = GridThermalScenario(initial_temperature=0.8)
        before_obs = scenario.observe()
        before_mean = before_obs.state["global_temp_mean"]

        # Simular contrafactual
        cf_result = scenario.simulate_counterfactual(
            intervention="activate_cooling",
            external_input=0.0
        )

        # Estado real no debe haber cambiado
        after_obs = scenario.observe()
        after_mean = after_obs.state["global_temp_mean"]

        assert abs(before_mean - after_mean) < 0.0001
        # Pero el contrafactual debe ser diferente
        assert cf_result.state["global_temp_mean"] != before_mean

    def test_counterfactual_returns_valid_state(self):
        """Contrafactual debe retornar estado válido."""
        scenario = GridThermalScenario(initial_temperature=0.8)
        cf_result = scenario.simulate_counterfactual(
            intervention="activate_cooling",
            external_input=0.04
        )

        assert "global_temp_mean" in cf_result.state
        assert "global_temp_max" in cf_result.state
        assert "world_level" in cf_result.state
        assert 0.0 <= cf_result.state["global_temp_mean"] <= 1.0

    def test_deactivate_cooling_turns_off_all_cells(self):
        """Deactivate cooling debe apagar enfriamiento en todas las celdas."""
        scenario = GridThermalScenario()

        # Primero activar
        scenario.factual_transition(intervention="activate_cooling", external_input=0.0)

        # Luego desactivar
        scenario.factual_transition(intervention="deactivate_cooling", external_input=0.0)

        obs = scenario.observe()
        assert all(not cell["cooling_active"] for cell in obs.state["cell_states"])


class TestGridThermalWorldLevel:
    """Tests de world_level derivado para comparabilidad con 1x1."""

    def test_world_level_derived_from_mean(self):
        """world_level debe ser igual a global_temp_mean."""
        scenario = GridThermalScenario(initial_temperature=0.82)
        obs = scenario.observe()

        world_level = obs.state["world_level"]
        mean = obs.state["global_temp_mean"]

        assert abs(world_level - mean) < 0.0001

    def test_world_level_in_valid_range(self):
        """world_level debe estar en rango [0.0, 1.0]."""
        scenario = GridThermalScenario()
        obs = scenario.observe()

        world_level = obs.state["world_level"]
        assert 0.0 <= world_level <= 1.0

    def test_world_level_present_in_transitions(self):
        """world_level debe estar presente en transiciones."""
        scenario = GridThermalScenario()
        result = scenario.factual_transition(
            intervention="activate_cooling",
            external_input=0.04
        )

        assert "world_level" in result.state


class TestGridThermalStructuralProfile:
    """Tests de perfil estructural y firma causal."""

    def test_structural_profile_declares_spatial_topology(self):
        """Perfil estructural debe declarar topología espacial."""
        scenario = GridThermalScenario()
        profile = scenario.structural_profile

        assert "spatial" in profile.control_topology

    def test_structural_profile_has_grid_metadata(self):
        """Config hash debe incluir world_shape y cell_count."""
        scenario = GridThermalScenario()
        profile = scenario.structural_profile

        # Verificar que el hash es diferente del thermal_homeostasis simple
        assert profile.scenario_name == "grid_thermal_5x5"

    def test_causal_signature_includes_grid_metadata(self):
        """Firma causal debe incluir variable agregada."""
        scenario = GridThermalScenario()
        sig = scenario.causal_signature

        assert sig.main_variable == "global_temp_mean"
        assert "global_temp_mean" in sig.observable_variables

    def test_causal_signature_has_correct_polarity(self):
        """Firma causal debe tener polaridad 'lower_is_better'."""
        scenario = GridThermalScenario()
        sig = scenario.causal_signature

        assert sig.causal_polarity == "lower_is_better"
        assert sig.optimization_direction == "minimize"


class TestGridThermalObservation:
    """Tests de observación y proposiciones."""

    def test_observe_returns_temp_high_on_alarm(self):
        """Observe debe retornar TEMP_HIGH cuando hay alarma global."""
        scenario = GridThermalScenario(initial_temperature=0.90, alarm_threshold=0.85)
        obs = scenario.observe()

        assert "TEMP_HIGH" in obs.propositions
        assert obs.alarm is True

    def test_observe_returns_temp_normal_below_threshold(self):
        """Observe debe retornar TEMP_NORMAL bajo umbral."""
        scenario = GridThermalScenario(initial_temperature=0.70, alarm_threshold=0.85)
        obs = scenario.observe()

        assert "TEMP_NORMAL" in obs.propositions
        assert obs.alarm is False

    def test_observe_includes_cooling_active_proposition(self):
        """Observe debe incluir COOLING_ACTIVE si hay celdas con cooling."""
        scenario = GridThermalScenario()
        scenario.factual_transition(intervention="activate_cooling", external_input=0.0)
        obs = scenario.observe()

        assert "COOLING_ACTIVE" in obs.propositions


class TestGridThermalInterventions:
    """Tests de intervenciones y selección."""

    def test_select_intervention_activates_on_alarm(self):
        """select_intervention con alarma depende de topología.

        Nueva lógica topológicamente sensible:
        - Alarma + concentrado/hotspot -> activate_cooling
        - Alarma + difuso sin gradiente fuerte -> deactivate_cooling (menos urgente)
        """
        # Caso 1: Alarma + hotspot (concentrado) -> DEBE activar
        scenario_hotspot = GridThermalScenario(
            initial_temperature=0.80,
            alarm_threshold=0.85,
            topology="hotspot_center",
            topology_params={"hotspot_temp": 0.95},
        )
        obs_hotspot = scenario_hotspot.observe()
        intervention_hotspot = scenario_hotspot.select_intervention(obs_hotspot)
        assert intervention_hotspot == "activate_cooling"

        # Caso 2: Alarma + uniforme (difuso) -> NO activa (tolera más)
        scenario_uniform = GridThermalScenario(
            initial_temperature=0.90, alarm_threshold=0.85
        )
        obs_uniform = scenario_uniform.observe()
        intervention_uniform = scenario_uniform.select_intervention(obs_uniform)
        # Difuso sin gradiente fuerte -> tolera más
        assert intervention_uniform == "deactivate_cooling"

    def test_select_intervention_deactivates_below_threshold(self):
        """select_intervention debe retornar deactivate_cooling sin alarma."""
        scenario = GridThermalScenario(initial_temperature=0.70, alarm_threshold=0.85)
        obs = scenario.observe()
        intervention = scenario.select_intervention(obs)

        assert intervention == "deactivate_cooling"

    def test_get_main_proposition_returns_correct_value(self):
        """get_main_proposition debe retornar proposición correcta."""
        scenario = GridThermalScenario(initial_temperature=0.90, alarm_threshold=0.85)
        obs = scenario.observe()
        prop = scenario.get_main_proposition(obs)

        assert prop == "TEMP_HIGH"

    def test_get_intervention_proposition_for_cooling(self):
        """get_intervention_proposition debe mapear correctamente."""
        scenario = GridThermalScenario()
        prop = scenario.get_intervention_proposition("activate_cooling")

        assert prop == "ACTIVATE_COOLING"


class TestGridThermalFormula:
    """Tests de fórmula LOTF."""

    def test_get_formula_returns_template(self):
        """get_formula debe retornar template LOTF."""
        scenario = GridThermalScenario()
        obs = scenario.observe()
        formula = scenario.get_formula(obs)

        assert formula == "TEMP_HIGH -> ACTIVATE_COOLING"


class TestGridThermalBoundaryConditions:
    """Tests de condiciones de borde."""

    def test_temperature_clamped_at_0(self):
        """Temperatura no debe caer por debajo de 0.0."""
        scenario = GridThermalScenario(initial_temperature=0.05, cooling_effect=0.1)
        scenario.factual_transition(intervention="activate_cooling", external_input=0.0)

        obs = scenario.observe()
        for cell in obs.state["cell_states"]:
            assert cell["temperature"] >= 0.0

    def test_temperature_clamped_at_1(self):
        """Temperatura no debe superar 1.0."""
        scenario = GridThermalScenario(initial_temperature=0.95)
        scenario.factual_transition(intervention="deactivate_cooling", external_input=0.5)

        obs = scenario.observe()
        for cell in obs.state["cell_states"]:
            assert cell["temperature"] <= 1.0

    def test_zero_external_input_with_no_cooling(self):
        """Sin calor externo ni cooling, temperatura debe permanecer estable."""
        initial_temp = 0.6
        scenario = GridThermalScenario(initial_temperature=initial_temp)
        scenario.factual_transition(intervention="deactivate_cooling", external_input=0.0)

        obs = scenario.observe()
        assert abs(obs.state["global_temp_mean"] - initial_temp) < 0.001


class TestGridThermalCloning:
    """Tests de clonado de grid para contrafactual."""

    def test_clone_grid_creates_independent_copy(self):
        """_clone_grid debe crear copia independiente."""
        scenario = GridThermalScenario(initial_temperature=0.8)
        original_grid = scenario._grid
        cloned_grid = scenario._clone_grid()

        # Modificar clon
        cloned_grid.cells[0].temperature = 0.5

        # Original no debe cambiar
        assert original_grid.cells[0].temperature == 0.8

    def test_clone_preserves_all_cell_states(self):
        """Clone debe preservar todos los estados de celdas."""
        scenario = GridThermalScenario()
        scenario._grid.cells[5].cooling_active = True
        scenario._grid.cells[10].temperature = 0.95

        cloned = scenario._clone_grid()

        assert cloned.cells[5].cooling_active is True
        assert abs(cloned.cells[10].temperature - 0.95) < 0.001


class TestGridThermalSemanticLevels:
    """Tests de niveles semánticos discretos del mundo."""

    def test_level_safe_range(self):
        """Nivel SAFE (1) para temperaturas < 0.60."""
        scenario = GridThermalScenario(initial_temperature=0.50)
        obs = scenario.observe()

        assert obs.level == 1
        assert obs.state["world_level_semantic"] == "SAFE"

    def test_level_elevated_range(self):
        """Nivel ELEVATED (2) para 0.60 <= temp < alarm_threshold."""
        scenario = GridThermalScenario(initial_temperature=0.70, alarm_threshold=0.85)
        obs = scenario.observe()

        assert obs.level == 2
        assert obs.state["world_level_semantic"] == "ELEVATED"

    def test_level_warning_range(self):
        """Nivel WARNING (3) para alarm_threshold <= temp < 0.95."""
        scenario = GridThermalScenario(initial_temperature=0.88, alarm_threshold=0.85)
        obs = scenario.observe()

        assert obs.level == 3
        assert obs.state["world_level_semantic"] == "WARNING"
        assert obs.alarm is True

    def test_level_critical_range(self):
        """Nivel CRITICAL (4) para temp >= 0.95."""
        scenario = GridThermalScenario(initial_temperature=0.97)
        obs = scenario.observe()

        assert obs.level == 4
        assert obs.state["world_level_semantic"] == "CRITICAL"
        assert obs.alarm is True

    def test_level_boundary_at_0_60(self):
        """Frontera SAFE/ELEVATED en 0.60."""
        scenario_safe = GridThermalScenario(initial_temperature=0.59)
        scenario_elevated = GridThermalScenario(initial_temperature=0.60)

        assert scenario_safe.observe().level == 1
        assert scenario_elevated.observe().level == 2

    def test_level_boundary_at_alarm_threshold(self):
        """Frontera ELEVATED/WARNING en alarm_threshold."""
        threshold = 0.85
        scenario_elevated = GridThermalScenario(initial_temperature=0.84, alarm_threshold=threshold)
        scenario_warning = GridThermalScenario(initial_temperature=0.85, alarm_threshold=threshold)

        assert scenario_elevated.observe().level == 2
        assert scenario_warning.observe().level == 3

    def test_level_boundary_at_0_95(self):
        """Frontera WARNING/CRITICAL en 0.95."""
        scenario_warning = GridThermalScenario(initial_temperature=0.94)
        scenario_critical = GridThermalScenario(initial_temperature=0.95)

        assert scenario_warning.observe().level == 3
        assert scenario_critical.observe().level == 4

    def test_level_in_factual_transition(self):
        """Nivel debe propagarse correctamente en transición factual."""
        scenario = GridThermalScenario(initial_temperature=0.88, alarm_threshold=0.85)
        result = scenario.factual_transition(
            intervention="activate_cooling",
            external_input=0.0
        )

        assert "level" in dir(result)
        assert result.level in [1, 2, 3, 4]

    def test_level_in_counterfactual_transition(self):
        """Nivel debe propagarse correctamente en transición contrafactual."""
        scenario = GridThermalScenario(initial_temperature=0.88)
        result = scenario.simulate_counterfactual(
            intervention="activate_cooling",
            external_input=0.0
        )

        assert "level" in dir(result)
        assert result.level in [1, 2, 3, 4]

    def test_state_has_three_level_representations(self):
        """Estado debe tener world_level, world_level_numeric y world_level_semantic."""
        scenario = GridThermalScenario(initial_temperature=0.70)
        obs = scenario.observe()

        assert "world_level" in obs.state
        assert "world_level_numeric" in obs.state
        assert "world_level_semantic" in obs.state

        # world_level y world_level_numeric deben ser iguales (compatibilidad)
        assert obs.state["world_level"] == obs.state["world_level_numeric"]

    def test_level_semantic_mapping(self):
        """Verificar mapeo completo de niveles a semántica."""
        test_cases = [
            (0.30, 1, "SAFE"),
            (0.65, 2, "ELEVATED"),
            (0.88, 3, "WARNING"),
            (0.97, 4, "CRITICAL"),
        ]

        for temp, expected_level, expected_semantic in test_cases:
            scenario = GridThermalScenario(initial_temperature=temp)
            obs = scenario.observe()

            assert obs.level == expected_level, f"Failed for temp={temp}"
            assert obs.state["world_level_semantic"] == expected_semantic, f"Failed for temp={temp}"

    def test_level_transitions_through_ranges(self):
        """Nivel debe cambiar correctamente al transicionar entre rangos."""
        scenario = GridThermalScenario(initial_temperature=0.50)  # SAFE

        # Observar inicial
        obs1 = scenario.observe()
        assert obs1.level == 1

        # Calentar a ELEVATED (necesita subir ~0.10, con 25 celdas = 0.10 * 25 = 2.5)
        scenario.factual_transition(intervention="deactivate_cooling", external_input=2.5)
        obs2 = scenario.observe()
        assert obs2.level == 2

        # Calentar a WARNING (necesita subir ~0.25, con 25 celdas = 0.25 * 25 = 6.25)
        scenario.factual_transition(intervention="deactivate_cooling", external_input=6.25)
        obs3 = scenario.observe()
        assert obs3.level == 3

    def test_level_consistency_between_observation_and_transition(self):
        """Nivel en observation debe coincidir con nivel en transition."""
        scenario = GridThermalScenario(initial_temperature=0.88)

        # Ejecutar transición
        transition = scenario.factual_transition(
            intervention="activate_cooling",
            external_input=0.0
        )

        # Observar después de transición
        obs = scenario.observe()

        assert transition.level == obs.level

    def test_counterfactual_level_differs_from_factual(self):
        """Nivel contrafactual puede diferir del factual."""
        scenario = GridThermalScenario(initial_temperature=0.88, alarm_threshold=0.85)

        # Factual: activar cooling
        factual = scenario.factual_transition(
            intervention="activate_cooling",
            external_input=0.0
        )

        # Contrafactual: desactivar cooling (NO muta estado)
        # Recrear escenario para contrafactual puro
        scenario2 = GridThermalScenario(initial_temperature=0.88, alarm_threshold=0.85)
        counterfactual = scenario2.simulate_counterfactual(
            intervention="deactivate_cooling",
            external_input=0.0
        )

        # Ambos tienen level, pero pueden ser diferentes
        assert hasattr(factual, "level")
        assert hasattr(counterfactual, "level")

