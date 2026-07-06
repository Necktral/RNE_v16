"""Cableado del override de PREVISIÓN (A11+A12) a la decisión viva del runner.

A diferencia del override greedy (guard de un paso), éste usa el guard de HORIZONTE:
A12 adoptó una alternativa y A11 certificó que el greedy cruza la alarma en el
horizonte. Gated por RNFE_REASONING_ACTUATES (sombra OFF ⇒ byte-idéntico).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from runtime.storage import StorageConfig, StorageFactory
from runtime.world import ScenarioEpisodeRunner
from runtime.world.deferred_load_scenario import DeferredLoadScenario
from runtime.world.intervention_override import evaluate_foresight_override


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "fov.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "art",
        prefer_postgres_reads=False,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


_ALLOWED = ["boost_throughput", "shed_load"]


class TestPureForesightGuard:
    def _state(self, **over):
        base = {
            "a12_adopted_alternative": True,
            "a12_decision": "shed_load",
            "imagination_chosen_breaches_at": 3,
        }
        base.update(over)
        return base

    def test_fires_on_a12_adoption_and_a11_breach(self):
        d = evaluate_foresight_override(
            reasoning_state=self._state(), allowed_interventions=_ALLOWED,
            greedy_intervention="boost_throughput",
        )
        assert d.fired is True
        assert d.driver_family == "a12"
        assert d.to_intervention == "shed_load"
        assert d.guard_reason == "foresight_horizon"

    def test_no_fire_without_a12_adoption(self):
        d = evaluate_foresight_override(
            reasoning_state=self._state(a12_adopted_alternative=False),
            allowed_interventions=_ALLOWED, greedy_intervention="boost_throughput",
        )
        assert d.fired is False and d.guard_reason == "a12_no_adoption"

    def test_no_fire_without_foresight_breach(self):
        d = evaluate_foresight_override(
            reasoning_state=self._state(imagination_chosen_breaches_at=None),
            allowed_interventions=_ALLOWED, greedy_intervention="boost_throughput",
        )
        assert d.fired is False and d.guard_reason == "no_foresight_breach"

    def test_no_fire_when_decision_matches_greedy(self):
        d = evaluate_foresight_override(
            reasoning_state=self._state(a12_decision="boost_throughput"),
            allowed_interventions=_ALLOWED, greedy_intervention="boost_throughput",
        )
        assert d.fired is False and d.guard_reason == "a12_matches_greedy"

    def test_no_fire_when_decision_not_allowed(self):
        d = evaluate_foresight_override(
            reasoning_state=self._state(a12_decision="teleport"),
            allowed_interventions=_ALLOWED, greedy_intervention="boost_throughput",
        )
        assert d.fired is False and d.guard_reason == "a12_decision_not_allowed"


class TestRunnerWiring:
    def _runner(self, tmp_path):
        return ScenarioEpisodeRunner(
            scenario=DeferredLoadScenario(initial_load=0.70),
            storage=_storage(tmp_path),
            run_id="fov",
            closure_profile="adaptive_min",
        )

    def test_disabled_by_default_no_override(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RNFE_REASONING_ACTUATES", raising=False)
        runner = self._runner(tmp_path)
        factual = runner.scenario.factual_transition(intervention="boost_throughput", external_input=0.04)
        decision, candidate = runner._maybe_override_intervention(
            reasoning_state={"a12_adopted_alternative": True, "a12_decision": "shed_load",
                             "imagination_chosen_breaches_at": 3},
            greedy_intervention="boost_throughput", factual=factual, external_input=0.04,
        )
        assert decision.fired is False and decision.guard_reason == "actuation_disabled"
        assert candidate is None

    def test_foresight_override_fires_and_returns_candidate(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RNFE_REASONING_ACTUATES", "1")
        runner = self._runner(tmp_path)
        factual = runner.scenario.factual_transition(intervention="boost_throughput", external_input=0.04)
        decision, candidate = runner._maybe_override_intervention(
            reasoning_state={"a12_adopted_alternative": True, "a12_decision": "shed_load",
                             "imagination_chosen_breaches_at": 3},
            greedy_intervention="boost_throughput", factual=factual, external_input=0.04,
        )
        assert decision.fired is True
        assert decision.driver_family == "a12"
        assert decision.to_intervention == "shed_load"
        assert candidate is not None
        assert candidate.state["load"] < 1.0  # shed aplicado (no saturado)

    def test_falls_through_to_greedy_without_a12(self, tmp_path, monkeypatch):
        # Sin adopción de A12, el camino de previsión no dispara y cae al greedy
        # (que sin conflicto estructural devuelve no_conflict).
        monkeypatch.setenv("RNFE_REASONING_ACTUATES", "1")
        monkeypatch.delenv("RNFE_A12_DEEP", raising=False)
        monkeypatch.delenv("RNFE_REASONING_DEEP", raising=False)
        runner = self._runner(tmp_path)
        factual = runner.scenario.factual_transition(intervention="boost_throughput", external_input=0.04)
        decision, _ = runner._maybe_override_intervention(
            reasoning_state={},  # sin a12 ni conflicto
            greedy_intervention="boost_throughput", factual=factual, external_input=0.04,
        )
        assert decision.fired is False
        assert decision.guard_reason in ("a12_no_adoption", "no_conflict")


class TestLiveEpisodeWiring:
    """El cableado completo a través del ScenarioEpisodeRunner real."""

    def _env(self, monkeypatch, *, actuate: bool):
        monkeypatch.setenv("RNFE_REASONING_MODE", "fixed")
        monkeypatch.setenv("RNFE_REASONING_FAMILY_PROFILE", "core_plus_imagination_a12")
        monkeypatch.setenv("RNFE_REASONING_MAX_STEPS", "10")
        monkeypatch.setenv("RNFE_IMAGINATION_DEEP", "1")
        monkeypatch.setenv("RNFE_A12_DEEP", "1")
        if actuate:
            monkeypatch.setenv("RNFE_REASONING_ACTUATES", "1")
        else:
            monkeypatch.delenv("RNFE_REASONING_ACTUATES", raising=False)

    def _run(self, tmp_path):
        runner = ScenarioEpisodeRunner(
            scenario=DeferredLoadScenario(initial_load=0.70),
            storage=_storage(tmp_path), run_id="live", closure_profile="adaptive_min",
        )
        return runner.run_episode(external_input=0.04)

    def test_override_fires_and_applies_shed(self, tmp_path, monkeypatch):
        self._env(monkeypatch, actuate=True)
        res = self._run(tmp_path)
        ov = res["intervention_override"]
        assert ov["fired"] is True
        assert ov["driver_family"] == "a12"
        assert ov["to_intervention"] == "shed_load"
        assert ov["guard_reason"] == "foresight_horizon"
        # La acción aplicada cambió a la previsora.
        assert res["episode"]["context"]["intervention"] == "shed_load"

    def test_shadow_off_keeps_greedy_trap(self, tmp_path, monkeypatch):
        # Con actuación OFF (sombra), el camino nominal aplica el greedy (boost, trampa).
        self._env(monkeypatch, actuate=False)
        res = self._run(tmp_path)
        assert res["intervention_override"]["fired"] is False
        assert res["episode"]["context"]["intervention"] == "boost_throughput"
