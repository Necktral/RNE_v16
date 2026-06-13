"""Tests de Ωₜ: obstrucción de coherencia multi-contexto + IoC* (canon f2.4 §4)."""

from __future__ import annotations

import json
from pathlib import Path

from runtime.certification.coherence_obstruction import (
    CoherenceObstructionTracker,
    Section,
    cycle_error,
    section_divergence,
    section_from_certificate,
    section_from_episode_result,
)
from runtime.storage import StorageConfig, StorageFactory


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "omega.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


def _section(
    *,
    scenario="thermal_homeostasis",
    symbols=("TEMP_HIGH", "ACTIVATE_COOLING"),
    relation="support",
    formula="TEMP_HIGH -> ACTIVATE_COOLING",
    value=0.5,
) -> Section:
    tokens = formula.split()
    return Section(
        scenario=scenario,
        symbols=frozenset(symbols),
        relation_kind=relation,
        formula_norm=" ".join(tokens),
        formula_tokens=frozenset(tokens),
        main_variable="temperature",
        value=value,
        alarm_threshold=0.85,
    )


def _episode_result(
    *,
    scenario="thermal_homeostasis",
    formula="TEMP_HIGH -> ACTIVATE_COOLING",
    relation="support",
    temp=0.5,
    ded_validated=True,
    direction_match=True,
    agreement=True,
):
    return {
        "episode": {
            "scenario": scenario,
            "scenario_metadata": {
                "scenario_name": scenario,
                "main_variable": "temperature",
                "alarm_threshold": 0.85,
            },
            "context": {"formula": formula, "observation": {"temperature": temp}},
            "result": {"updated_world": {"temperature": temp}, "relation_kind": relation},
        },
        "reasoning": {
            "state": {
                "ded_validated": ded_validated,
                "cau_link": {"direction_match": direction_match},
                "ctf_checked": {"agreement_with_relation_kind": agreement},
            }
        },
    }


class TestSectionDivergence:
    def test_identical_sections_diverge_zero(self):
        a, b = _section(), _section()
        out = section_divergence(a, b)
        assert out["divergence"] == 0.0
        assert out["d_s"] == 0.0 and out["d_f"] == 0.0 and out["d_w"] == 0.0
        assert out["cross_context"] is False

    def test_disjoint_symbols_raise_d_s(self):
        out = section_divergence(_section(), _section(symbols=("STOCK_LOW", "RESTOCK")))
        assert out["d_s"] >= 0.7  # Jaccard 0 ⇒ 0.7·1.0

    def test_relation_kind_mismatch_contributes(self):
        out = section_divergence(_section(relation="support"), _section(relation="contradiction"))
        assert abs(out["d_s"] - 0.3) < 1e-9  # solo el término de relación

    def test_formula_change_raises_d_f(self):
        out = section_divergence(
            _section(), _section(formula="STOCK_LOW -> RESTOCK", symbols=("STOCK_LOW", "RESTOCK"))
        )
        assert out["d_f"] > 0.5

    def test_world_distance_is_value_gap(self):
        out = section_divergence(_section(value=0.2), _section(value=0.9))
        assert abs(out["d_w"] - 0.7) < 1e-9

    def test_missing_world_value_is_neutral(self):
        out = section_divergence(_section(value=None), _section(value=0.9))
        assert out["d_w"] == 0.5

    def test_real_cross_scenario_morphism_penalizes_adversarial(self):
        # thermal (minimize) vs resource (maximize): morfismo real adversarial.
        from runtime.world.morphism_engine import MorphismEngine
        from runtime.world.registry import get_scenario

        m = MorphismEngine().compute_morphism(
            get_scenario("thermal_homeostasis").causal_signature,
            get_scenario("resource_management").causal_signature,
        )
        a = _section()
        b = _section(
            scenario="resource_management",
            symbols=("STOCK_LOW", "START_PRODUCTION"),
            formula="STOCK_LOW -> START_PRODUCTION",
            value=0.52,
        )
        out = section_divergence(a, b, morphism=m)
        assert out["cross_context"] is True
        assert out["morphism_class"] == "adversarial"
        assert out["d_w"] >= 0.70  # piso adversarial aunque los valores estén cerca
        json.dumps(out)


class TestCycleError:
    def test_coherent_episode_closes_cycle(self):
        out = cycle_error(_episode_result())
        assert out["error"] == 0.0
        assert out["s_to_f"] == 0.0  # símbolos en el vocabulario del escenario real
        assert out["f_to_m"] == 0.0
        assert out["m_to_s"] == 0.0

    def test_ded_failure_breaks_f_to_m(self):
        out = cycle_error(_episode_result(ded_validated=False))
        assert out["f_to_m"] == 1.0
        assert out["error"] > 0.3

    def test_causal_disagreement_breaks_m_to_s(self):
        out = cycle_error(_episode_result(direction_match=False, agreement=False))
        assert out["m_to_s"] == 1.0

    def test_foreign_symbols_break_s_to_f(self):
        out = cycle_error(_episode_result(formula="ALIEN_PROP -> OTHER_THING"))
        assert out["s_to_f"] == 1.0

    def test_missing_signals_are_neutral_not_crash(self):
        out = cycle_error({"episode": {}, "reasoning": {}})
        assert out["error"] == 0.5
        json.dumps(out)


class TestTracker:
    def test_first_episode_has_no_pairs(self):
        tracker = CoherenceObstructionTracker()
        block = tracker.assess(run_id="r", episode_result=_episode_result(), ioc_value=0.88)
        assert block["n_window"] == 0
        assert block["pairwise_mean"] == 0.0
        assert block["omega"] == 0.0  # ciclo coherente + sin pares
        assert block["ioc_star"] == 0.88
        assert block["delta_ioc_star"] is None

    def test_ioc_star_formula(self):
        tracker = CoherenceObstructionTracker(lambda_omega=0.30, lambda_cycle=0.50)
        tracker.assess(run_id="r", episode_result=_episode_result(temp=0.5), ioc_value=0.9)
        block = tracker.assess(
            run_id="r", episode_result=_episode_result(temp=0.9), ioc_value=0.9
        )
        # d_w = 0.4, d_s = d_f = 0 ⇒ pares = 0.4/3; ciclo = 0
        expected_omega = (0.4 / 3.0)
        assert abs(block["omega"] - expected_omega) < 1e-6
        assert abs(block["ioc_star"] - (0.9 - 0.30 * expected_omega)) < 1e-6
        assert block["delta_ioc_star"] is not None

    def test_window_is_bounded(self):
        tracker = CoherenceObstructionTracker(window=3)
        for i in range(10):
            block = tracker.assess(
                run_id="r", episode_result=_episode_result(temp=0.5), ioc_value=0.9
            )
        assert block["n_window"] == 3

    def test_seeds_window_from_certificates(self, tmp_path):
        from runtime.world import ScenarioEpisodeRunner

        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage, run_id="omega-seed", scenario="thermal_homeostasis"
        )
        for _ in range(3):
            runner.run_episode()
        # Tracker NUEVO sobre el mismo run: siembra la ventana de los certificados.
        tracker = CoherenceObstructionTracker(storage=storage)
        block = tracker.assess(
            run_id="omega-seed", episode_result=_episode_result(temp=0.5), ioc_value=0.9
        )
        assert block["n_window"] == 3

    def test_block_is_json_safe(self):
        tracker = CoherenceObstructionTracker()
        for _ in range(3):
            block = tracker.assess(
                run_id="r", episode_result=_episode_result(), ioc_value=0.85
            )
        json.dumps(block)


class TestCertificateSymmetry:
    def test_live_and_persisted_sections_match(self, tmp_path):
        from runtime.world import ScenarioEpisodeRunner

        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage, run_id="omega-sym", scenario="thermal_homeostasis"
        )
        result = runner.run_episode()
        live = section_from_episode_result(result)
        cert = storage.list_episode_certificates(run_id="omega-sym", limit=1)[0]
        persisted = section_from_certificate(cert)
        assert live.scenario == persisted.scenario
        assert live.symbols == persisted.symbols
        assert live.formula_norm == persisted.formula_norm
        assert live.relation_kind == persisted.relation_kind
        assert live.value == persisted.value


class TestGateIntegration:
    def test_healthy_run_low_omega_shadow_mode(self, tmp_path):
        from runtime.world import ScenarioEpisodeRunner

        runner = ScenarioEpisodeRunner(
            storage=_storage(tmp_path), run_id="omega-int", scenario="thermal_homeostasis"
        )
        for _ in range(4):
            runner.run_episode()
        cert = runner.storage.list_episode_certificates(run_id="omega-int", limit=1)[0]
        block = cert.metadata["omega"]
        assert block["schema"] == "omega.v1"
        assert block["omega"] < 0.05  # run coherente
        assert block["cycle"]["error"] == 0.0
        assert abs(block["ioc_star"] - cert.ioc_proxy) < 0.05
        # Sombra: el veredicto clásico sigue gobernado por ioc_proxy.
        assert cert.verdict == "certified"
        json.dumps(block)

    def test_regime_switch_spikes_omega_with_morphism(self, tmp_path):
        from runtime.world import ScenarioEpisodeRunner

        storage = _storage(tmp_path)
        r1 = ScenarioEpisodeRunner(
            storage=storage, run_id="omega-x", scenario="thermal_homeostasis"
        )
        for _ in range(3):
            r1.run_episode()
        r2 = ScenarioEpisodeRunner(
            storage=storage,
            run_id="omega-x",
            scenario="resource_management",
            organism_state=r1.organism_state,
            lineage=r1.lineage,
        )
        r2.run_episode()
        cert = storage.list_episode_certificates(run_id="omega-x", limit=1)[0]
        block = cert.metadata["omega"]
        assert block["cross_context"] is True
        assert block["omega"] > 0.3  # el cruce de régimen cuesta coherencia
        classes = {p.get("morphism_class") for p in block["pair_divergences"]}
        assert "adversarial" in classes  # morfismo real cableado al camino vivo

    def test_reward_uses_delta_ioc_star(self, tmp_path):
        from runtime.world import ScenarioEpisodeRunner

        runner = ScenarioEpisodeRunner(
            storage=_storage(tmp_path), scenario="thermal_homeostasis"
        )
        for _ in range(3):
            result = runner.run_episode()
        rr = result["reasoning_reward"]
        assert rr["delta_used"] == "delta_ioc_star"
        assert rr["delta_ioc_star"] is not None
