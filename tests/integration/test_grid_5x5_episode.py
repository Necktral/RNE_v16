"""Tests de integración para GridThermalScenario con ScenarioEpisodeRunner."""

import json
from pathlib import Path

from runtime.storage import StorageConfig, StorageFactory
from runtime.world.scenario_runner import ScenarioEpisodeRunner


def _storage(tmp_path: Path):
    """Crea storage para tests."""
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "grid_5x5.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


class TestGrid5x5Integration:
    """Tests de integración del escenario grid_thermal_5x5 con el runtime completo."""

    def test_scenario_runner_executes_5x5_episode(self, tmp_path: Path):
        """ScenarioEpisodeRunner debe ejecutar episodio completo con grid_thermal_5x5."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-5x5-test",
            scenario="grid_thermal_5x5"
        )
        result = runner.run_episode(external_input=0.04)

        assert result["episode"]["scenario"] == "grid_thermal_5x5"
        # world_shape y cell_count están en observation, no en scenario_metadata
        obs = result["episode"]["context"]["observation"]
        assert obs["world_shape"] == "5x5"
        assert obs["cell_count"] == 25
        storage.close()

    def test_5x5_episode_creates_artifact(self, tmp_path: Path):
        """Episodio 5x5 debe crear artifact en disco."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-5x5-artifact",
            scenario="grid_thermal_5x5"
        )
        result = runner.run_episode(external_input=0.04)

        assert "artifact" in result
        artifact_path = Path(result["artifact"]["abs_path"])
        assert artifact_path.exists()
        storage.close()

    def test_5x5_artifact_contains_cell_states(self, tmp_path: Path):
        """Artifact debe contener estados de las 25 celdas."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-5x5-cells",
            scenario="grid_thermal_5x5"
        )
        result = runner.run_episode(external_input=0.04)

        artifact_path = result["artifact"]["abs_path"]
        with open(artifact_path) as f:
            artifact_data = json.load(f)

        assert "cell_states" in artifact_data["episode"]["context"]["observation"]
        cell_states = artifact_data["episode"]["context"]["observation"]["cell_states"]
        assert len(cell_states) == 25
        storage.close()

    def test_5x5_episode_emits_closed_event(self, tmp_path: Path):
        """Episodio 5x5 debe emitir evento episode.closed."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-5x5-event",
            scenario="grid_thermal_5x5"
        )
        runner.run_episode(external_input=0.04)

        events = storage.list_events(run_id="run-5x5-event", limit=10)
        closed_events = [e for e in events if e.event_type == "episode.closed"]
        assert len(closed_events) > 0

        # Verificar que el evento tiene metadata de 5x5
        event_payload = closed_events[0].payload
        assert event_payload["scenario"] == "grid_thermal_5x5"
        storage.close()

    def test_5x5_episode_gets_certified(self, tmp_path: Path):
        """Episodio 5x5 debe pasar por certificación."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-5x5-cert",
            scenario="grid_thermal_5x5"
        )
        result = runner.run_episode(external_input=0.04)

        assert "certification" in result
        assert "certificate_id" in result["certification"]
        assert "verdict" in result["certification"]
        assert result["certification"]["verdict"] in ["passed", "failed"]
        storage.close()

    def test_5x5_episode_has_world_level(self, tmp_path: Path):
        """Episodio 5x5 debe incluir world_level en observación."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-5x5-level",
            scenario="grid_thermal_5x5"
        )
        result = runner.run_episode(external_input=0.04)

        observation = result["episode"]["context"]["observation"]
        assert "world_level" in observation
        assert "global_temp_mean" in observation

        # world_level debe ser igual a global_temp_mean
        world_level = observation["world_level"]
        mean = observation["global_temp_mean"]
        assert abs(world_level - mean) < 0.001
        storage.close()

    def test_5x5_with_baseline_fixed_profile(self, tmp_path: Path):
        """5x5 debe funcionar con baseline_fixed (default)."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-5x5-baseline",
            scenario="grid_thermal_5x5",
            closure_profile="baseline_fixed"
        )
        result = runner.run_episode(external_input=0.04)

        obs = result["episode"]["context"]["observation"]
        assert obs["world_shape"] == "5x5"
        assert result["episode"]["closure_profile"] == "baseline_fixed"
        storage.close()

    def test_5x5_with_strict_memory_mode(self, tmp_path: Path):
        """5x5 debe funcionar con strict_same_scenario (default)."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-5x5-strict",
            scenario="grid_thermal_5x5",
            memory_filter_mode="strict_same_scenario"
        )

        # Primer episodio
        result1 = runner.run_episode(external_input=0.04)
        # Segundo episodio
        result2 = runner.run_episode(external_input=0.04)

        # Memoria debe estar vacía en el primer episodio
        memory1 = result1["episode"]["context"]["retrieved_memory"]
        assert len(memory1) == 0

        # Segundo episodio puede tener memoria del primero (mismo escenario)
        # pero depende de si la proposición coincide
        storage.close()

    def test_5x5_does_not_pollute_1x1_memory(self, tmp_path: Path):
        """Memoria de 5x5 NO debe contaminar memoria de 1x1 en strict mode."""
        storage = _storage(tmp_path)

        # Ejecutar episodio 1x1
        runner_1x1 = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-1x1-memory",
            scenario="thermal_homeostasis",
            memory_filter_mode="strict_same_scenario"
        )
        runner_1x1.run_episode(external_input=0.04)

        # Ejecutar episodio 5x5
        runner_5x5 = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-5x5-memory",
            scenario="grid_thermal_5x5",
            memory_filter_mode="strict_same_scenario"
        )
        result_5x5 = runner_5x5.run_episode(external_input=0.04)

        # Memoria 5x5 NO debe recuperar episodios 1x1
        memory_hits = result_5x5["episode"]["context"]["retrieved_memory"]
        for hit in memory_hits:
            # Si hay hits, deben ser del mismo escenario
            if "scenario_name" in hit.get("metadata", {}):
                assert hit["metadata"]["scenario_name"] == "grid_thermal_5x5"

        storage.close()

    def test_5x5_multiple_episodes_maintain_continuity(self, tmp_path: Path):
        """Múltiples episodios 5x5 deben mantener continuidad."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-5x5-multi",
            scenario="grid_thermal_5x5"
        )

        results = []
        for i in range(3):
            result = runner.run_episode(external_input=0.04)
            results.append(result)

        # Verificar que todos los episodios se ejecutaron
        assert len(results) == 3

        # Verificar que todos son del mismo escenario
        for result in results:
            assert result["episode"]["scenario"] == "grid_thermal_5x5"

        storage.close()

    def test_5x5_episode_includes_organism_trajectory(self, tmp_path: Path):
        """Episodio 5x5 debe incluir organism_trajectory."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-5x5-traj",
            scenario="grid_thermal_5x5"
        )
        result = runner.run_episode(external_input=0.04)

        assert "organism_trajectory" in result
        trajectory = result["organism_trajectory"]
        assert "organism_id" in trajectory
        assert "points" in trajectory
        storage.close()

    def test_5x5_artifact_size_reasonable(self, tmp_path: Path):
        """Artifact de 5x5 debe ser razonable en tamaño (< 100KB para un episodio)."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-5x5-size",
            scenario="grid_thermal_5x5"
        )
        result = runner.run_episode(external_input=0.04)

        artifact_path = Path(result["artifact"]["abs_path"])
        artifact_size = artifact_path.stat().st_size

        # Artifact de un episodio no debería exceder 100KB
        assert artifact_size < 100 * 1024, f"Artifact too large: {artifact_size} bytes"
        storage.close()


class TestGrid5x5Compatibility:
    """Tests de compatibilidad con runtime existente."""

    def test_5x5_scenario_config_valid(self, tmp_path: Path):
        """Config de 5x5 debe ser válido según contrato ScenarioConfig."""
        from runtime.world.registry import get_scenario

        scenario = get_scenario("grid_thermal_5x5")
        config = scenario.config

        assert config.name == "grid_thermal_5x5"
        assert config.main_variable == "global_temp_mean"
        assert len(config.interventions) == 2
        assert "activate_cooling" in config.interventions
        assert config.alarm_threshold > 0.0

    def test_5x5_structural_profile_valid(self, tmp_path: Path):
        """Perfil estructural de 5x5 debe ser válido."""
        from runtime.world.registry import get_scenario

        scenario = get_scenario("grid_thermal_5x5")
        profile = scenario.structural_profile

        assert profile.scenario_name == "grid_thermal_5x5"
        assert profile.main_variable == "global_temp_mean"
        assert profile.optimization_direction == "minimize"

    def test_5x5_causal_signature_valid(self, tmp_path: Path):
        """Firma causal de 5x5 debe ser válida."""
        from runtime.world.registry import get_scenario

        scenario = get_scenario("grid_thermal_5x5")
        sig = scenario.causal_signature

        assert sig.scenario_name == "grid_thermal_5x5"
        assert sig.main_variable == "global_temp_mean"
        assert len(sig.intervention_effects) == 2
        assert len(sig.causal_edges) == 3
