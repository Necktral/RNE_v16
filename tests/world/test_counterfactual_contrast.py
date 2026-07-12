"""B5 — el contrafactual debe ser un CONTRASTE, no un índice fijo.

El contrafactual existe para responder "¿qué habría pasado si NO hubiera hecho
ESTO?". El runner lo tomaba como ``interventions[1]`` — un índice fijo. Y como
TODAS las políticas de RNFE devuelven ``interventions[0]`` bajo alarma y
``interventions[1]`` si no, el contrafactual colisionaba con el factual en todo
el régimen de calma: el organismo se comparaba contra sí mismo.

Colisión medida sobre main (20 episodios por escenario):
    thermal_homeostasis   45%
    resource_management   55%
    grid_thermal_5x5     100%   <- nunca tuvo contraste, jamás
    deferred_load_trap     0%   (en esa corrida; colisiona si load < 0.6)

Con factual == contrafactual, ``factual_delta == counterfactual_delta`` y
``scale_estimator._compute_epistemic_insufficiency`` obtiene ``conflict = 0``: el
organismo se declara epistémicamente suficiente precisamente cuando no tiene
contraste alguno.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.storage import StorageConfig, StorageFactory
from runtime.world.scenario_runner import ScenarioEpisodeRunner
from runtime.world.thermal_scenario import ThermalScenario


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "ctf.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


class _SingleInterventionScenario(ThermalScenario):
    """Escenario degenerado: una sola intervención ⇒ NO existe alterna posible.

    No es un escenario del registry: existe para ejercitar el camino de
    "contraste NO DISPONIBLE", que es distinto de "contraste cero".
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._config.interventions = ["activate_cooling"]

    def select_intervention(self, observation) -> str:
        return "activate_cooling"


class TestCounterfactualIsAContrast:
    """El contrafactual debe DIFERIR de la acción factual."""

    def test_second_intervention_as_factual_does_not_collide(self, tmp_path: Path):
        """EL TEST OBLIGATORIO: forzar a la política a elegir la SEGUNDA intervención.

        Con temperatura bajo el umbral, ThermalScenario elige
        ``deactivate_cooling`` = ``interventions[1]`` = exactamente el contrafactual
        fijo de antes. En main esto producía factual == contrafactual y delta 0
        por colisión. Ahora el contrafactual debe ser la OPUESTA.
        """
        storage = _storage(tmp_path)
        scenario = ThermalScenario(
            initial_temperature=0.30, alarm_threshold=0.85
        )
        runner = ScenarioEpisodeRunner(
            storage=storage, run_id="run-b5-second", scenario=scenario
        )
        result = runner.run_episode(external_input=0.04)
        res = result["episode"]["result"]
        contrast = res["counterfactual_contrast"]

        # La política eligió la SEGUNDA intervención como acción factual.
        assert result["episode"]["context"]["intervention"] == "deactivate_cooling"
        assert contrast["factual_intervention"] == "deactivate_cooling"

        # ...y el contrafactual NO es esa misma: es la opuesta.
        assert contrast["counter_intervention"] == "activate_cooling"
        assert contrast["counter_intervention"] != contrast["factual_intervention"]
        assert contrast["available"] is True
        assert contrast["reason"] == "opposite_intervention"

        # El delta NO es 0 por colisión: hay contraste real.
        assert res["counterfactual_delta"] is not None
        assert res["factual_delta"] != res["counterfactual_delta"]
        storage.close()

    def test_first_intervention_as_factual_also_contrasts(self, tmp_path: Path):
        """Bajo alarma la política elige la PRIMERA: el contrafactual es la segunda."""
        storage = _storage(tmp_path)
        scenario = ThermalScenario(
            initial_temperature=0.95, alarm_threshold=0.85
        )
        runner = ScenarioEpisodeRunner(
            storage=storage, run_id="run-b5-first", scenario=scenario
        )
        result = runner.run_episode(external_input=0.04)
        res = result["episode"]["result"]
        contrast = res["counterfactual_contrast"]

        assert contrast["factual_intervention"] == "activate_cooling"
        assert contrast["counter_intervention"] == "deactivate_cooling"
        assert res["factual_delta"] != res["counterfactual_delta"]
        storage.close()

    @pytest.mark.parametrize(
        "scenario_name",
        [
            "thermal_homeostasis",
            "resource_management",
            "grid_thermal_5x5",
            "deferred_load_trap",
        ],
    )
    def test_no_collision_across_episodes_in_every_scenario(
        self, scenario_name: str, tmp_path: Path
    ):
        """En NINGÚN episodio de NINGÚN escenario el contrafactual iguala al factual.

        grid_thermal_5x5 es el caso crítico: en main colisionaba el 100% de los
        episodios (su política devuelve deactivate_cooling = interventions[1]
        salvo hotspot), así que su conflicto epistémico era estructuralmente 0.
        """
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id=f"run-b5-{scenario_name}",
            scenario=scenario_name,
        )
        for _ in range(6):
            result = runner.run_episode(external_input=0.04)
            res = result["episode"]["result"]
            contrast = res["counterfactual_contrast"]

            assert contrast["available"] is True
            assert contrast["counter_intervention"] != contrast["factual_intervention"], (
                f"{scenario_name}: el contrafactual colisionó con el factual "
                f"({contrast['factual_intervention']})"
            )
            assert res["counterfactual_delta"] is not None
        storage.close()

    def test_grid_thermal_has_real_conflict_signal(self, tmp_path: Path):
        """grid_thermal_5x5: el conflicto factual-contrafactual deja de ser 0 estructural."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage, run_id="run-b5-grid", scenario="grid_thermal_5x5"
        )
        gaps = []
        for _ in range(5):
            res = runner.run_episode(external_input=0.04)["episode"]["result"]
            gaps.append(abs(res["factual_delta"] - res["counterfactual_delta"]))

        # En main TODOS estos gaps eran exactamente 0.0 (colisión 100%).
        assert any(g > 0.0 for g in gaps), f"todos los gaps siguen en 0: {gaps}"
        storage.close()


class TestContrastUnavailableIsDeclaredNotFaked:
    """Sin alterna posible: contraste NO DISPONIBLE != contraste cero."""

    def test_single_intervention_declares_unavailable(self, tmp_path: Path):
        """Un escenario de una sola intervención no puede contrastar: se DECLARA."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-b5-single",
            scenario=_SingleInterventionScenario(
                initial_temperature=0.95, alarm_threshold=0.85
            ),
        )
        res = runner.run_episode(external_input=0.04)["episode"]["result"]
        contrast = res["counterfactual_contrast"]

        assert contrast["available"] is False
        assert contrast["counter_intervention"] is None
        assert contrast["reason"] == "no_alternative_intervention"
        # AUSENCIA, no un cero favorable.
        assert res["counterfactual_delta"] is None
        assert contrast["unmeasured_fields"] == ["counterfactual_delta"]
        # El factual sí se midió: no se pierde información real.
        assert res["factual_delta"] is not None
        storage.close()


class TestAbsentContrastIsNotCertainty:
    """`_compute_epistemic_insufficiency` no debe leer la ausencia como certeza."""

    def _estimator(self):
        from runtime.control.msrc.scale_estimator import ScaleEstimator

        return ScaleEstimator()

    def _insufficiency(self, metrics):
        est = self._estimator()
        return est._compute_epistemic_insufficiency(
            current_scale_id="micro",
            observation={"propositions": ["A", "B"]},
            metrics=metrics,
            expected_spatial_complexity=0.0,
        )

    def test_absent_counterfactual_does_not_lower_insufficiency(self):
        """Sin contraste, la insuficiencia NO puede quedar por debajo de la medida con contraste nulo.

        Ésta es la trampa: si el contrafactual ausente se rellena con 0.0, el
        conflicto da 0 y ARRASTRA el score hacia abajo (0.30 de peso en "todo
        bien"). El organismo se declara suficiente por no haber mirado.
        """
        signals = {
            "contradiction_signal": 0.8,
            "uncertainty": 0.6,
            "factual_delta": 0.05,
        }
        # Contraste NO disponible (declarado por el runner).
        absent_score, absent_bd = self._insufficiency(
            {**signals, "counterfactual_delta": None}
        )
        # Contraste disponible pero idéntico (el bug viejo: colisión ⇒ conflict = 0).
        collided_score, collided_bd = self._insufficiency(
            {**signals, "counterfactual_delta": 0.05}
        )

        assert collided_bd["factual_counterfactual_conflict"] == 0.0
        assert collided_bd["conflict_measured"] == 1.0
        assert absent_bd["conflict_measured"] == 0.0

        # La ausencia NO se premia: el score no baja respecto del conflicto-cero.
        assert absent_score > collided_score

    def test_measured_conflict_path_is_unchanged(self):
        """Con conflicto medible, el score es el de siempre (pesos suman 1.0)."""
        score, bd = self._insufficiency(
            {
                "factual_delta": 0.10,
                "counterfactual_delta": -0.10,
                "contradiction_signal": 0.5,
                "uncertainty": 0.5,
            }
        )
        # conflict = min(|0.10 - (-0.10)| / 0.2, 1.0) = 1.0
        expected = (
            0.30 * 1.0 + 0.18 * 0.5 + 0.14 * 0.5 + 0.10 * 0.0
            + 0.08 * 0.0 + 0.10 * 0.0 + 0.10 * 0.0
        )
        assert bd["factual_counterfactual_conflict"] == pytest.approx(1.0)
        assert score == pytest.approx(expected)

