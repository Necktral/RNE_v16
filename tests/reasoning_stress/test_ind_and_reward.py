"""R3a: IND real (inducción) + recompensa semi-Markov del razonamiento."""

from __future__ import annotations

import json
from pathlib import Path

from runtime.reasoning.families import core_inference as ci
import runtime.reasoning.families.ind as IND
from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler
from runtime.reasoning.scheduler_meta.reward import (
    compute_episode_reward,
    reasoning_cost_from_trace,
    summarize_rewards,
)
from runtime.storage import StorageConfig, StorageFactory


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "r3.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


def _state(memory=None, intervention="activate_cooling", alarm=True):
    return {
        "observation": {"alarm": alarm, "temperature": 0.9},
        "scenario": "thermal_homeostasis",
        "scenario_metadata": {
            "scenario_name": "thermal_homeostasis",
            "main_variable": "temperature",
        },
        "intervention": intervention,
        "retrieved_memory": memory or [],
    }


class TestIndInduction:
    def test_no_longer_a_stub(self):
        out = IND.execute(_state())
        assert out["status"] == "ok"
        assert out["state_delta"]  # ya no es {}
        assert "ind_rule" in out["state_delta"]

    def test_empirical_generalization_from_memory(self):
        mem = [
            {"structure": {"relation_kind": "support", "intervention": "activate_cooling"}},
            {"structure": {"relation_kind": "support", "intervention": "activate_cooling"}},
            {"structure": {"relation_kind": "support", "intervention": "activate_cooling"}},
            {"structure": {"relation_kind": "contradiction", "intervention": "deactivate_cooling"}},
        ]
        rule = ci.induce(_state(memory=mem))["state_delta"]["ind_rule"]
        assert rule["source"] == "memory"
        assert rule["total"] == 4
        assert rule["support"] == 3
        assert rule["consequent_relation"] == "support"
        assert rule["best_intervention"] == "activate_cooling"
        assert rule["best_intervention_support_rate"] == 1.0
        # LCB conservadora: por debajo de la tasa puntual.
        assert 0.0 <= rule["confidence_lcb"] <= rule["support_rate"]

    def test_confidence_grows_with_more_consistent_evidence(self):
        few = [{"structure": {"relation_kind": "support", "intervention": "activate_cooling"}}] * 3
        many = [{"structure": {"relation_kind": "support", "intervention": "activate_cooling"}}] * 30
        lcb_few = ci.induce(_state(memory=few))["state_delta"]["ind_confidence_lcb"]
        lcb_many = ci.induce(_state(memory=many))["state_delta"]["ind_confidence_lcb"]
        assert lcb_many > lcb_few

    def test_apriori_fallback_from_causal_signature(self):
        # Sin memoria: induce desde el modelo de efectos de la firma causal real.
        out = ci.induce(_state(memory=[]))
        rule = out["state_delta"]["ind_rule"]
        assert rule["source"] == "causal_signature"
        assert rule["total"] == 0
        # El escenario térmico declara una intervención correctiva (activate_cooling).
        assert rule["best_intervention"] is not None
        assert "activate_cooling" in rule["generalized_interventions"]

    def test_runs_through_scheduler_contract(self, tmp_path):
        # Secuencia explícita ⇒ el scheduler ejecuta IND y propaga su state_delta.
        sched = MetaScheduler(sequence=["ind"], mode="fixed")
        result = sched.run(
            {
                "observation": {"alarm": True, "temperature": 0.9},
                "scenario": "thermal_homeostasis",
                "scenario_metadata": {
                    "scenario_name": "thermal_homeostasis",
                    "main_variable": "temperature",
                },
                "intervention": "activate_cooling",
                "retrieved_memory": [
                    {"structure": {"relation_kind": "support", "intervention": "activate_cooling"}},
                ],
            }
        )
        assert result["sequence"] == ["IND"]
        assert result["state"]["ind_best_intervention"] == "activate_cooling"
        assert result["trace"][0]["status"] == "ok"

    def test_state_delta_is_json_safe(self):
        json.dumps(ci.induce(_state(memory=[{"structure": {"relation_kind": "support"}}]))["state_delta"])


class TestReward:
    def test_reward_formula(self):
        r = compute_episode_reward(delta_ioc=0.02, reasoning_cost=6.0, cost_budget=6.0, b_safe=None)
        # r = 0.02 − 0.10·(6/6) − 0 = -0.08
        assert abs(r["reward"] - (-0.08)) < 1e-6
        assert r["energy_term"] == 0.1
        assert r["bsafe_penalty"] == 0.0
        assert r["dissipation_term"] == 0.0

    def test_none_delta_treated_as_zero(self):
        r = compute_episode_reward(delta_ioc=None, reasoning_cost=0.0, cost_budget=6.0, b_safe=None)
        assert r["delta_ioc"] is None
        assert r["reward"] == 0.0

    def test_bsafe_violation_penalizes(self):
        r = compute_episode_reward(
            delta_ioc=0.05, reasoning_cost=0.0, cost_budget=6.0,
            b_safe={"violated": True, "value": None},
        )
        assert r["bsafe_penalty"] == 0.5
        assert r["reward"] < 0.0  # 0.05 − 0 − 0.5

    def test_higher_cost_lowers_reward(self):
        cheap = compute_episode_reward(delta_ioc=0.1, reasoning_cost=1.0, cost_budget=6.0, b_safe=None)
        pricey = compute_episode_reward(delta_ioc=0.1, reasoning_cost=6.0, cost_budget=6.0, b_safe=None)
        assert cheap["reward"] > pricey["reward"]

    def test_cost_from_trace(self):
        trace = [
            {"detail": {"cost": 1.0}},
            {"detail": {"cost": 0.9}},
            {"detail": {}},  # sin coste ⇒ 1.0 por defecto
        ]
        assert abs(reasoning_cost_from_trace(trace) - 2.9) < 1e-9


class TestRewardNuDecomposition:
    """PR1 Bucle A — ν=cau.helps_goal como criterio aditivo de primera clase (cura J(h|X))."""

    def test_nu_byte_identical_when_lambda_zero(self):
        """λ_ν=0 (default) ⇒ el escalar `reward` es idéntico independientemente de ν."""
        base = compute_episode_reward(delta_ioc=0.03, reasoning_cost=2.0, cost_budget=6.0, b_safe=None)
        with_nu = compute_episode_reward(
            delta_ioc=0.03, reasoning_cost=2.0, cost_budget=6.0, b_safe=None, nu=True,
        )
        assert with_nu["reward"] == base["reward"]
        assert with_nu["schema"] == "reasoning_reward.v3"
        assert with_nu["lambda_nu"] == 0.0
        assert with_nu["nu"] == 1.0
        assert with_nu["nu_term"] == 0.0

    def test_nu_helps_goal_adds_lambda(self):
        """Con λ_ν=1.0 y helps_goal=True, ν suma exactamente λ_ν al reward."""
        without = compute_episode_reward(
            delta_ioc=0.0, reasoning_cost=0.0, cost_budget=6.0, b_safe=None,
            nu=True, lambda_nu=0.0,
        )
        with_nu = compute_episode_reward(
            delta_ioc=0.0, reasoning_cost=0.0, cost_budget=6.0, b_safe=None,
            nu=True, lambda_nu=1.0,
        )
        assert abs(with_nu["reward"] - without["reward"] - 1.0) < 1e-9
        assert with_nu["nu_term"] == 1.0

    def test_nu_false_or_none_no_bonus(self):
        """helps_goal=False o None ⇒ ν=0 ⇒ sin bonus aunque λ_ν>0."""
        false_nu = compute_episode_reward(
            delta_ioc=0.0, reasoning_cost=0.0, cost_budget=6.0, b_safe=None,
            nu=False, lambda_nu=1.0,
        )
        none_nu = compute_episode_reward(
            delta_ioc=0.0, reasoning_cost=0.0, cost_budget=6.0, b_safe=None,
            nu=None, lambda_nu=1.0,
        )
        assert false_nu["nu_term"] == 0.0
        assert false_nu["reward"] == 0.0
        assert none_nu["nu_term"] == 0.0
        assert none_nu["nu"] is None

    def test_nu_float_is_clamped(self):
        """ν float se clampa a [0,1]."""
        hi = compute_episode_reward(
            delta_ioc=0.0, reasoning_cost=0.0, cost_budget=6.0, b_safe=None,
            nu=5.0, lambda_nu=1.0,
        )
        assert hi["nu"] == 1.0 and hi["nu_term"] == 1.0

    def test_env_flag_default_off(self, monkeypatch):
        """Sin RNFE_REWARD_LAMBDA_NU en el entorno, λ_ν=0 y byte-idéntico."""
        monkeypatch.delenv("RNFE_REWARD_LAMBDA_NU", raising=False)
        r = compute_episode_reward(delta_ioc=0.02, reasoning_cost=6.0, cost_budget=6.0, b_safe=None, nu=True)
        assert abs(r["reward"] - (-0.08)) < 1e-6  # idéntico a test_reward_formula
        assert r["lambda_nu"] == 0.0

    def test_env_flag_activates(self, monkeypatch):
        """RNFE_REWARD_LAMBDA_NU=1.0 activa el término vía entorno."""
        monkeypatch.setenv("RNFE_REWARD_LAMBDA_NU", "1.0")
        r = compute_episode_reward(delta_ioc=0.0, reasoning_cost=0.0, cost_budget=6.0, b_safe=None, nu=True)
        assert r["lambda_nu"] == 1.0
        assert abs(r["reward"] - 1.0) < 1e-9

    def test_summary(self):
        rewards = [{"reward": 0.1}, {"reward": -0.2}, {"reward": 0.3}]
        s = summarize_rewards(rewards)
        assert s["n"] == 3
        assert abs(s["cumulative"] - 0.2) < 1e-9
        assert s["max"] == 0.3 and s["min"] == -0.2

    def test_empty_summary(self):
        assert summarize_rewards([])["n"] == 0


class TestRunnerIntegration:
    def test_episode_carries_reward_and_persists_event(self, tmp_path):
        from runtime.world import ScenarioEpisodeRunner

        runner = ScenarioEpisodeRunner(
            storage=_storage(tmp_path), scenario="thermal_homeostasis"
        )
        for _ in range(3):
            result = runner.run_episode()

        r = result["reasoning_reward"]
        assert r["schema"] == "reasoning_reward.v3"
        assert isinstance(r["reward"], float)
        assert r["dissipation_term"] == 0.0  # D_t aún en R4
        assert "nu" in r and "lambda_nu" in r  # término de viabilidad ν (PR1 Bucle A)
        assert r["lambda_nu"] == 0.0  # default OFF ⇒ nominal byte-idéntico
        json.dumps(r)

        events = runner.storage.list_events(
            run_id=runner.run_id, event_types=["reasoning.reward"]
        )
        assert len(events) == 3
