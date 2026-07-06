"""R3c: ANA estructural (SME-lite) + CTF re-simulado — profundización opt-in.

Destino en el repo: tests/reasoning_stress/test_deep_families.py

Flags OFF por defecto ⇒ comportamiento clásico byte-idéntico (la ecología por
defecto ABD/ANA/CAU/CTF/DED/PROB y los baselines no se mueven). ON ⇒ ANA por
alineación estructural (sistematicidad) y CTF por re-simulación con el modelo de
efectos declarado (dominancia de la elección). Espeja el estilo de
`test_ind_and_reward.py` / `test_plan_opt_deliberative.py`.
"""

from __future__ import annotations

import json

from runtime.reasoning.families import core_inference as ci
import runtime.reasoning.families.ana as ANA
import runtime.reasoning.families.ctf as CTF
from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler


def _state(temp=0.9, factual=0.83, counterfactual=0.9, intervention="activate_cooling"):
    return {
        "observation": {"alarm": True, "temperature": temp, "propositions": ["TEMP_HIGH"]},
        "scenario": "thermal_homeostasis",
        "scenario_metadata": {
            "scenario_name": "thermal_homeostasis",
            "main_variable": "temperature",
            "alarm_threshold": 0.85,
            "interventions": ["activate_cooling", "deactivate_cooling"],
        },
        "intervention": intervention,
        "relation_kind": "support",
        "updated_world": {"temperature": factual},
        "counterfactual": {"temperature": counterfactual},
        "retrieved_memory": [
            {"memory_id": "m1", "score": 0.80, "scale": "meso",
             "structure": {"relation_kind": "support", "intervention": "activate_cooling",
                           "propositions": ["TEMP_HIGH"]}},
            {"memory_id": "m2", "score": 0.62, "scale": "micro",
             "structure": {"relation_kind": "contradiction", "intervention": "deactivate_cooling"}},
        ],
    }


class TestGating:
    def test_flags_off_by_default(self, monkeypatch):
        monkeypatch.delenv("RNFE_ANA_STRUCTURAL", raising=False)
        monkeypatch.delenv("RNFE_CTF_RESIM", raising=False)
        assert ci.ana_structural_enabled() is False
        assert ci.ctf_resim_enabled() is False

    def test_flags_activate(self, monkeypatch):
        monkeypatch.setenv("RNFE_ANA_STRUCTURAL", "1")
        monkeypatch.setenv("RNFE_CTF_RESIM", "on")
        assert ci.ana_structural_enabled() is True
        assert ci.ctf_resim_enabled() is True


class TestAnaStructural:
    def test_classic_shape_when_off(self, monkeypatch):
        monkeypatch.delenv("RNFE_ANA_STRUCTURAL", raising=False)
        ana = ci.analogize(_state())["state_delta"]["ana_mapping"]
        assert ana["source"] == "memory"
        assert ana["memory_id"] == "m1"        # mejor score de recuperación
        assert "alignment_mode" not in ana     # marcador estructural ausente (clásico)

    def test_structural_alignment_when_on(self, monkeypatch):
        monkeypatch.setenv("RNFE_ANA_STRUCTURAL", "1")
        out = ci.analogize(_state())["state_delta"]
        ana = out["ana_mapping"]
        assert ana["alignment_mode"] == "structural"
        assert ana["mapped_intervention"] == "activate_cooling"
        assert "intervention_match" in ana["correspondences"]
        assert "relation_consistent" in ana["correspondences"]
        assert ana["conflicts"] == 0
        assert len(out["ana_structural_candidates"]) == 2

    def test_self_map_without_memory_when_on(self, monkeypatch):
        monkeypatch.setenv("RNFE_ANA_STRUCTURAL", "1")
        st = _state()
        st["retrieved_memory"] = []
        ana = ci.analogize(st)["state_delta"]["ana_mapping"]
        assert ana["source"] == "scenario_self"
        assert ana["alignment_mode"] == "structural"
        assert "relation_coverage" in ana

    def test_scheduler_wiring(self, monkeypatch):
        monkeypatch.setenv("RNFE_ANA_STRUCTURAL", "1")
        result = MetaScheduler(sequence=["ana"], mode="fixed").run(_state())
        assert result["sequence"] == ["ANA"]
        assert result["state"]["ana_mapping"]["alignment_mode"] == "structural"
        assert result["trace"][0]["status"] == "ok"

    def test_json_safe(self, monkeypatch):
        monkeypatch.setenv("RNFE_ANA_STRUCTURAL", "1")
        json.dumps(ci.analogize(_state())["state_delta"])


class TestCtfResim:
    def test_classic_when_off(self, monkeypatch):
        monkeypatch.delenv("RNFE_CTF_RESIM", raising=False)
        ctf = ci.counterfactual_check(_state())["state_delta"]["ctf_checked"]
        assert ctf["supports_choice"] is True
        assert "resim" not in ctf              # sin re-simulación en la ruta clásica

    def test_resimulation_when_on(self, monkeypatch):
        monkeypatch.setenv("RNFE_CTF_RESIM", "1")
        ctf = ci.counterfactual_check(_state())["state_delta"]["ctf_checked"]
        assert ctf["supports_choice"] is True  # clásico preservado
        assert ctf["resim"]["status"] == "ok"
        assert ctf["resim"]["chosen_dominates"] is True
        assert set(ctf["resim"]["projected_counterfactuals"]) == {
            "activate_cooling", "deactivate_cooling"
        }
        assert ctf["resim_corroborates_factual"] is True

    def test_non_dominant_choice_flagged(self, monkeypatch):
        monkeypatch.setenv("RNFE_CTF_RESIM", "1")
        ctf = ci.counterfactual_check(
            _state(intervention="deactivate_cooling")
        )["state_delta"]["ctf_checked"]
        assert ctf["resim"]["chosen_dominates"] is False

    def test_scheduler_wiring(self, monkeypatch):
        monkeypatch.setenv("RNFE_CTF_RESIM", "1")
        result = MetaScheduler(sequence=["ctf"], mode="fixed").run(_state())
        assert result["sequence"] == ["CTF"]
        assert "resim" in result["state"]["ctf_checked"]
        assert result["trace"][0]["status"] == "ok"

    def test_json_safe(self, monkeypatch):
        monkeypatch.setenv("RNFE_CTF_RESIM", "1")
        json.dumps(ci.counterfactual_check(_state())["state_delta"])


class TestContractPreserved:
    def test_family_wrappers_when_deep(self, monkeypatch):
        monkeypatch.setenv("RNFE_ANA_STRUCTURAL", "1")
        monkeypatch.setenv("RNFE_CTF_RESIM", "1")
        for mod in (ANA, CTF):
            res = mod.execute(_state())
            assert res["family"] == mod.FAMILY_ID
            assert res["status"] == "ok"
            assert isinstance(res["state_delta"], dict) and res["state_delta"]
            assert 0.0 <= res["confidence"] <= 1.0
            assert res["cost"] >= 0.0
