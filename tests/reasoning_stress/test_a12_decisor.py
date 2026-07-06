"""A12 decisor lógico-probabilístico — Fase 1 (shadow).

Cubre:
  - OFF byte-idéntico (idle exacto).
  - Núcleo `decide()`: retracción defeasible, adopción por Bayes-factor, ACT abstain,
    y la composición con A11 (retracta el consenso lineal ante la previsión de breach).
  - `execute()` ON sobre una traza sintética tipo trampa.
"""

import pytest

from runtime.reasoning.families import a12 as A12


def _decide(**over):
    base = dict(
        default="boost_throughput",
        witnesses={},
        defeaters={"imagination_breach": False, "ctf_disagree": False,
                   "cau_not_help": False, "ded_unsat": False},
        prob_point=0.7, prob_lcb=0.6,
    )
    base.update(over)
    return A12.decide(**base)


class TestOff:
    def test_idle_when_flags_off(self, monkeypatch):
        monkeypatch.delenv("RNFE_A12_DEEP", raising=False)
        monkeypatch.delenv("RNFE_REASONING_DEEP", raising=False)
        out = A12.execute({"intervention": "boost_throughput"})
        assert out == {"family": "A12", "status": "idle",
                       "state_delta": {}, "confidence": 0.0, "cost": 0.0}


class TestDecideCore:
    def test_no_defeaters_keeps_default(self):
        r = _decide(witnesses={"opt": "boost_throughput"})
        assert r["a12_default_defeated"] is False
        assert r["a12_adopted_alternative"] is False
        assert r["a12_decision"] == "boost_throughput"

    def test_a11_breach_retracts_and_adopts_alternative(self):
        # Composición con A11: opt/plan/ind votan boost (consenso lineal = default),
        # pero A11 recomienda shed y predice breach del default → A12 retracta y adopta shed.
        r = _decide(
            witnesses={"opt": "boost_throughput", "plan": "boost_throughput",
                       "ind": "boost_throughput", "imagination": "shed_load"},
            defeaters={"imagination_breach": True, "ctf_disagree": False,
                       "cau_not_help": False, "ded_unsat": False},
            prob_point=0.6, prob_lcb=0.6,
        )
        assert r["a12_default_defeated"] is True
        assert "imagination_breach" in r["a12_defeaters"]
        assert r["a12_adopted_alternative"] is True
        assert r["a12_decision"] == "shed_load"
        assert r["a12_bayes_factor"] >= 3.0
        assert r["a12_act"] == "commit"

    def test_weak_evidence_does_not_adopt(self):
        # Un solo derrotador débil y sin testigo alternativo → no hay candidata → no adopta.
        r = _decide(
            witnesses={"opt": "boost_throughput"},
            defeaters={"imagination_breach": False, "ctf_disagree": True,
                       "cau_not_help": False, "ded_unsat": False},
        )
        assert r["a12_default_defeated"] is True
        assert r["a12_adopted_alternative"] is False   # no hay alternativa que supere el BF
        assert r["a12_decision"] == "boost_throughput"

    def test_abstains_when_low_confidence(self):
        # Default derrotado con candidata, pero baja confianza y poca evidencia → abstain.
        r = _decide(
            witnesses={"heur": "shed_load"},
            defeaters={"imagination_breach": False, "ctf_disagree": True,
                       "cau_not_help": False, "ded_unsat": False},
            prob_point=0.2, prob_lcb=0.2,
        )
        assert r["a12_default_defeated"] is True
        assert r["a12_act"] == "abstain"
        assert r["a12_adopted_alternative"] is False
        assert r["a12_decision"] == "boost_throughput"   # honesto: no confiado, no cambia

    def test_deterministic(self):
        kw = dict(
            witnesses={"opt": "boost_throughput", "imagination": "shed_load"},
            defeaters={"imagination_breach": True, "ctf_disagree": True,
                       "cau_not_help": False, "ded_unsat": False},
        )
        assert _decide(**kw) == _decide(**kw)


class TestExecuteOn:
    def test_trap_trace_adopts_shed(self, monkeypatch):
        monkeypatch.setenv("RNFE_A12_DEEP", "1")
        state = {
            "intervention": "boost_throughput",
            # consenso lineal (deliberativas) = boost, el default
            "opt_intervention": "boost_throughput",
            "plan_first_action": "boost_throughput",
            "ind_best_intervention": "boost_throughput",
            # A11: previsión de breach + recomienda shed
            "imagination_recommended_intervention": "shed_load",
            "imagination_chosen_breaches_at": 3,
            "prob_point": 0.6, "prob_lcb": 0.6,
        }
        out = A12.execute(state)
        sd = out["state_delta"]
        assert sd["a12_decision"] == "shed_load"
        assert sd["a12_adopted_alternative"] is True
        assert sd["a12_default_defeated"] is True
        assert out["status"] == "warn"
        import json
        json.dumps(out)
        assert A12.execute(state) == out   # determinista

    def test_clean_trace_keeps_default(self, monkeypatch):
        monkeypatch.setenv("RNFE_A12_DEEP", "1")
        state = {
            "intervention": "activate_cooling",
            "opt_intervention": "activate_cooling",
            "prob_point": 0.8, "prob_lcb": 0.7,
        }
        out = A12.execute(state)
        assert out["state_delta"]["a12_decision"] == "activate_cooling"
        assert out["state_delta"]["a12_adopted_alternative"] is False
        assert out["status"] == "ok"


class TestCompositionEndToEnd:
    """Escenario real → A11 real → A12 real: la decisión de A12 maneja el mundo."""

    def test_a11_a12_composition_avoids_trap(self, monkeypatch):
        monkeypatch.setenv("RNFE_IMAGINATION_DEEP", "1")
        monkeypatch.setenv("RNFE_A12_DEEP", "1")
        from runtime.world.deferred_load_scenario import DeferredLoadScenario
        from runtime.reasoning.families import imagination as IMAG

        sc = DeferredLoadScenario(alarm_threshold=0.85)
        sig = sc.causal_signature

        def linear_greedy():  # lo que ven las deliberativas y el reactivo: boost
            lin = {
                e.intervention_name: (-e.expected_magnitude if e.expected_direction == "-" else e.expected_magnitude)
                for e in sig.intervention_effects
            }
            return min(lin, key=lambda k: lin[k])

        breaches = 0
        adopted = 0
        for _ in range(25):
            obs = sc.observe()
            reactive = linear_greedy()  # boost_throughput (cae en la trampa)
            state = {
                "scenario": "deferred_load_trap",
                "scenario_metadata": {
                    "scenario_name": "deferred_load_trap", "main_variable": "load",
                    "alarm_threshold": 0.85, "interventions": ["boost_throughput", "shed_load"],
                },
                "observation": dict(obs.state),
                "intervention": reactive,
            }
            # A11 imagina y escribe imagination_* en el estado
            state.update(IMAG.execute(state)["state_delta"])
            # consenso lineal de las deliberativas = boost (= default)
            state["opt_intervention"] = reactive
            state["plan_first_action"] = reactive
            state["ind_best_intervention"] = reactive
            state["prob_point"] = 0.6
            state["prob_lcb"] = 0.6
            # A12 decide leyendo toda la traza (incl. la previsión de A11)
            sd = A12.execute(state)["state_delta"]
            final = sd["a12_decision"] if sd.get("a12_adopted_alternative") else reactive
            if sd.get("a12_adopted_alternative"):
                adopted += 1
            tr = sc.factual_transition(intervention=final, external_input=0.04)
            if tr.alarm:
                breaches += 1

        assert adopted > 0            # A12 retracta boost y adopta shed vía Bayes-factor
        assert breaches == 0          # la composición A11→A12 evita la trampa end-to-end
