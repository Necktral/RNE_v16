"""R3b: PLAN/OPT reales + perfil deliberativo en la política."""

from __future__ import annotations

import json

from runtime.reasoning.families import core_inference as ci
import runtime.reasoning.families.plan as PLAN
import runtime.reasoning.families.opt as OPT
from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler


def _state(temp=0.9, alarm=True, threshold=0.85):
    return {
        "observation": {"alarm": alarm, "temperature": temp, "propositions": ["TEMP_HIGH"]},
        "scenario": "thermal_homeostasis",
        "scenario_metadata": {
            "scenario_name": "thermal_homeostasis",
            "main_variable": "temperature",
            "alarm_threshold": threshold,
            "interventions": ["activate_cooling", "deactivate_cooling"],
        },
        "intervention": "activate_cooling",
    }


class TestPlanSearch:
    def test_no_longer_a_stub(self):
        out = PLAN.execute(_state())
        assert out["status"] == "ok"
        assert out["state_delta"]["plan_built"] is True

    def test_finds_shortest_plan_to_goal(self):
        plan = ci.plan_search(_state(temp=0.9))["state_delta"]["plan"]
        # Térmico: cooling baja 0.07/paso ⇒ 0.9→0.83 < 0.85 en UN paso.
        assert plan["status"] == "goal_reached"
        assert plan["steps"] == ["activate_cooling"]
        assert plan["projected_terminal"] < 0.85

    def test_multi_step_plan_when_farther_from_goal(self):
        plan = ci.plan_search(_state(temp=0.97))["state_delta"]["plan"]
        assert plan["status"] == "goal_reached"
        assert len(plan["steps"]) == 2  # 0.97→0.90→0.83
        assert all(s == "activate_cooling" for s in plan["steps"])

    def test_best_effort_when_goal_unreachable_in_horizon(self):
        # Umbral imposible en 3 pasos (0.07·3=0.21 máx de bajada desde 0.9).
        plan = ci.plan_search(_state(temp=0.9, threshold=0.10))["state_delta"]["plan"]
        assert plan["status"] == "best_effort"
        assert plan["projected_terminal"] < 0.9  # aún así mejora
        assert plan["first_action"] == "activate_cooling"

    def test_graceful_without_observation(self):
        out = ci.plan_search({"scenario_metadata": {"interventions": ["a"]}})
        assert out["state_delta"]["plan_built"] is False

    def test_json_safe(self):
        json.dumps(ci.plan_search(_state())["state_delta"])


class TestOptimizeChoice:
    def test_no_longer_a_stub(self):
        out = OPT.execute(_state())
        assert out["status"] == "ok"
        assert out["state_delta"]["opt_solved"] is True

    def test_picks_corrective_intervention(self):
        choice = ci.optimize_choice(_state())["state_delta"]["opt_choice"]
        assert choice["intervention"] == "activate_cooling"
        assert choice["projected"] < 0.9

    def test_effort_cost_limits_steps(self):
        cheap = ci.optimize_choice(_state(), effort_cost=0.30)["state_delta"]["opt_choice"]
        eager = ci.optimize_choice(_state(), effort_cost=0.01)["state_delta"]["opt_choice"]
        # Esfuerzo caro ⇒ menos pasos que esfuerzo barato.
        assert cheap["steps"] <= eager["steps"]

    def test_alternatives_table_is_explicable(self):
        choice = ci.optimize_choice(_state())["state_delta"]["opt_choice"]
        assert len(choice["alternatives"]) == 2 * 3  # 2 intervenciones × horizonte 3
        best = min(choice["alternatives"], key=lambda a: a["objective"])
        assert best["objective"] == choice["objective"]

    def test_json_safe(self):
        json.dumps(ci.optimize_choice(_state())["state_delta"])


class TestPolicyActivation:
    def test_deliberative_profile_orders_plan_opt_after_ctf(self):
        sched = MetaScheduler(mode="fixed", family_profile="core_plus_deliberative")
        result = sched.run(_state())
        seq = result["sequence"]
        assert "PLAN" in seq and "OPT" in seq
        assert seq.index("PLAN") > seq.index("CTF")
        assert seq.index("OPT") > seq.index("PLAN")
        assert seq[-1] == "PROB"
        assert result["sequence_validation"]["validated_passed"] is True
        assert result["sequence_validation"]["unknown_families"] == []
        # Los resultados deliberativos quedan en el estado compartido.
        assert result["state"]["plan_built"] is True
        assert result["state"]["opt_solved"] is True

    def test_default_ecology_unaffected(self):
        sched = MetaScheduler(mode="adaptive", family_profile="adaptive_family_ecology_v2")
        result = sched.run(
            {"observation": {"temperature": 0.4, "propositions": ["A", "B"]}, "uncertainty": 0.2}
        )
        assert result["sequence"] == ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]
        assert "PLAN" not in result["sequence"] and "OPT" not in result["sequence"]

    def test_exploration_profile_admits_plan_opt_on_viability_edge(self):
        sched = MetaScheduler(
            mode="adaptive", family_profile="full_family_exploration", max_steps=10
        )
        result = sched.run(
            {**_state(), "regime_hint": "viability_edge", "edge_pressure": 0.7, "uncertainty": 0.5}
        )
        # Admisión: PLAN/OPT entran en la propuesta. Con 8 opcionales y presupuesto
        # 10 (6 núcleo + 4) el recorte validado no cabe a todas; lo que se EJECUTA
        # es la secuencia validada (recortada), preservando el núcleo. Ejecutar la
        # propuesta sin recortar rompería el cierre que la certificación cobra.
        assert "PLAN" in result["proposed_sequence"]
        assert "OPT" in result["proposed_sequence"]
        executed = set(result["sequence"])
        assert {"ABD", "ANA", "CAU", "CTF", "DED", "PROB"} <= executed
