"""R3d: overlays profundizadas (HEUR/FAL_GUARD/DIA_ADV/NESY/EVO_SEARCH).

Todas gated: OFF (flags sin setear) ⇒ comportamiento clásico byte-idéntico
(las stubs NESY/EVO_SEARCH siguen idle; heur/fal_guard/dia_adv conservan sus
claves clásicas). ON (RNFE_REASONING_DEEP o RNFE_<FAMILIA>_DEEP) ⇒ implementación
real de alta calidad sobre el estado acumulado. Espeja el estilo de
`test_ind_and_reward.py`.
"""

from __future__ import annotations

import copy

from runtime.reasoning.families import core_inference as ci
import runtime.reasoning.families.heur as HEUR
import runtime.reasoning.families.fal_guard as FAL
import runtime.reasoning.families.dia_adv as DIA
import runtime.reasoning.families.nesy as NESY
import runtime.reasoning.families.evo_search as EVO
from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler
from runtime.reasoning.scheduler_meta.family_profiles import resolve_family_profile


def _state():
    return {
        "scenario": "thermal_homeostasis",
        "observation": {"temperature": 0.9, "propositions": ["TEMP_HIGH"], "alarm": True},
        "updated_world": {"temperature": 0.83},
        "counterfactual": {"temperature": 0.9},
        "intervention": "activate_cooling",
        "relation_kind": "support",
        "scenario_metadata": {
            "scenario_name": "thermal_homeostasis", "main_variable": "temperature",
            "alarm_threshold": 0.85, "interventions": ["activate_cooling", "deactivate_cooling"],
        },
        "cau_link": {"helps_goal": True, "direction_match": True, "strength": 0.35,
                     "expected_direction": "-", "observed_direction": "-"},
        "ctf_checked": {"supports_choice": True, "agreement_with_relation_kind": True},
        "abd_hypotheses": [{"intervention": "activate_cooling", "score": 0.8},
                           {"intervention": "deactivate_cooling", "score": 0.3}],
        "abd_top_intervention": "activate_cooling", "opt_intervention": "activate_cooling",
        "plan_first_action": "activate_cooling", "ind_best_intervention": "activate_cooling",
        "ded_validated": True, "ded_conclusion": "ACTIVATE_COOLING",
        "prob_posterior": {"point": 0.72, "lower_confidence_bound": 0.55},
        "_meta": {"features": {"edge_pressure": 0.5, "uncertainty": 0.3, "contradiction_signal": 0.2,
                               "ambiguity_signal": 0.1, "hardware_pressure": 0.4}},
    }


class TestMasterFlag:
    def test_off_by_default(self, monkeypatch):
        for f in ("RNFE_REASONING_DEEP", "RNFE_HEUR_DEEP", "RNFE_NESY_DEEP"):
            monkeypatch.delenv(f, raising=False)
        assert ci.family_deep_enabled("HEUR") is False
        assert ci.family_deep_enabled("NESY") is False

    def test_master_enables_all(self, monkeypatch):
        monkeypatch.setenv("RNFE_REASONING_DEEP", "1")
        assert ci.family_deep_enabled("HEUR") is True
        assert ci.family_deep_enabled("EVO_SEARCH") is True
        assert ci.ana_structural_enabled() is True  # el maestro también cubre ANA/CTF

    def test_specific_flag(self, monkeypatch):
        monkeypatch.delenv("RNFE_REASONING_DEEP", raising=False)
        monkeypatch.setenv("RNFE_FAL_GUARD_DEEP", "1")
        assert ci.family_deep_enabled("FAL_GUARD") is True
        assert ci.family_deep_enabled("HEUR") is False


class TestHeur:
    def test_off_classic(self, monkeypatch):
        monkeypatch.delenv("RNFE_REASONING_DEEP", raising=False)
        sd = HEUR.execute(_state())["state_delta"]
        assert set(sd) == {"heur_triage_fast", "heur_uncertainty_hint"}

    def test_deep_consensus(self, monkeypatch):
        monkeypatch.setenv("RNFE_REASONING_DEEP", "1")
        out = HEUR.execute(_state())
        assert out["state_delta"]["heur_recommended_intervention"] == "activate_cooling"
        assert out["state_delta"]["heur_source_agreement"] == 1.0
        assert "vote_table" in out["artifacts"]


class TestFalGuard:
    def test_off_classic(self, monkeypatch):
        monkeypatch.delenv("RNFE_REASONING_DEEP", raising=False)
        sd = FAL.execute(_state())["state_delta"]
        assert set(sd) == {"fal_guard_clean", "fallacy_risk"}

    def test_deep_clean_when_consistent(self, monkeypatch):
        monkeypatch.setenv("RNFE_REASONING_DEEP", "1")
        out = FAL.execute(_state())
        assert out["state_delta"]["fallacies"] == []
        assert out["state_delta"]["fal_guard_clean"] is True

    def test_deep_detects_acting_against_causal(self, monkeypatch):
        monkeypatch.setenv("RNFE_REASONING_DEEP", "1")
        st = _state(); st["cau_link"]["helps_goal"] = False
        out = FAL.execute(st)
        assert "acting_against_causal_evidence" in out["state_delta"]["fallacies"]
        assert out["status"] == "warn"
        assert out["failure_mode"] == "reasoning_fallacies_detected"


class TestDiaAdv:
    def test_off_classic(self, monkeypatch):
        monkeypatch.delenv("RNFE_REASONING_DEEP", raising=False)
        sd = DIA.execute(_state())["state_delta"]
        assert set(sd) == {"adversarial_challenge_active", "adversarial_pressure"}

    def test_deep_dialectic(self, monkeypatch):
        monkeypatch.setenv("RNFE_REASONING_DEEP", "1")
        out = DIA.execute(_state())
        assert out["state_delta"]["dialectic_thesis"] == "activate_cooling"
        assert out["state_delta"]["dialectic_thesis_survives"] is True
        assert "objections" in out["artifacts"]


class TestNesy:
    def test_off_idle(self, monkeypatch):
        monkeypatch.delenv("RNFE_REASONING_DEEP", raising=False)
        out = NESY.execute(_state())
        assert out["status"] == "idle" and out["state_delta"] == {}

    def test_deep_coherent(self, monkeypatch):
        monkeypatch.setenv("RNFE_REASONING_DEEP", "1")
        out = NESY.execute(_state())
        assert out["state_delta"]["nesy_coherent"] is True

    def test_deep_dissonance(self, monkeypatch):
        monkeypatch.setenv("RNFE_REASONING_DEEP", "1")
        st = _state(); st["ded_validated"] = False
        out = NESY.execute(st)
        assert "symbolic_refutation" in out["state_delta"]["nesy_dissonance"]
        assert out["status"] == "warn"


class TestEvoSearch:
    def test_off_idle(self, monkeypatch):
        monkeypatch.delenv("RNFE_REASONING_DEEP", raising=False)
        out = EVO.execute(_state())
        assert out["status"] == "idle" and out["state_delta"] == {}

    def test_deep_solves_and_deterministic(self, monkeypatch):
        monkeypatch.setenv("RNFE_REASONING_DEEP", "1")
        a = EVO.execute(_state())
        b = EVO.execute(_state())
        assert a["state_delta"]["evo_solved"] is True
        assert a["state_delta"]["evo_first_action"] == "activate_cooling"
        assert a["state_delta"]["evo_goal_reached"] is True
        assert a == b  # sembrado desde el estado ⇒ reproducible


class TestSchedulerWiring:
    def test_new_profile_includes_upgraded_stubs(self):
        prof = resolve_family_profile("full_family_deep_v1", mode="fixed")
        assert "nesy" in prof.allowed_families
        assert "evo_search" in prof.allowed_families

    def test_overlays_execute_via_scheduler(self, monkeypatch):
        monkeypatch.setenv("RNFE_REASONING_DEEP", "1")
        seq = ["abd", "ana", "cau", "ctf", "prob", "nesy", "evo_search"]
        result = MetaScheduler(sequence=seq, mode="fixed").run(_state())
        assert "NESY" in result["sequence"]
        assert "EVO_SEARCH" in result["sequence"]
        assert result["state"]["evo_solved"] is True
        assert "nesy_coherent" in result["state"]


class TestCauMechanism:
    def test_off_no_mechanism(self, monkeypatch):
        monkeypatch.delenv("RNFE_REASONING_DEEP", raising=False)
        monkeypatch.delenv("RNFE_CAU_DEEP", raising=False)
        cau = ci.causal_infer(_state())["state_delta"]["cau_link"]
        assert "causal_mechanism" not in cau

    def test_deep_traces_dag_to_alarm(self, monkeypatch):
        monkeypatch.setenv("RNFE_CAU_DEEP", "1")
        cau = ci.causal_infer(_state())["state_delta"]["cau_link"]
        mech = cau["causal_mechanism"]
        assert mech["path"][-1] == "alarm"
        assert mech["goal_consistent"] is True  # activate_cooling reduce la alarma


class TestEmlSrDataset:
    def test_off_synthetic(self, monkeypatch):
        monkeypatch.delenv("RNFE_REASONING_DEEP", raising=False)
        monkeypatch.delenv("RNFE_EML_SR_DEEP", raising=False)
        import runtime.reasoning.families.eml_sr as EML
        rows = EML._dataset_from_state(_state())
        assert len(rows) == 3  # clásico: 3 filas ±0.01

    def test_deep_sweep_from_effect_model(self, monkeypatch):
        monkeypatch.setenv("RNFE_EML_SR_DEEP", "1")
        import runtime.reasoning.families.eml_sr as EML
        rows = EML._dataset_from_state(_state())
        assert len(rows) == 6
        # y = x + Δ_correctivo (Δ<0) ⇒ respuesta real, no ruido
        assert all(r["y"] <= r["x"] for r in rows)
        assert len({r["x"] for r in rows}) > 1  # x varía de verdad
