"""A11 IMAGINATION — Fase 1 (shadow): previsión de consecuencia diferida.

Cubre:
  - OFF byte-idéntico (idle exacto).
  - Núcleo puro `imagine()` sobre un mundo sintético con estado: detecta breach
    diferido, recomienda la acción con previsión, marca desacuerdo, y es determinista.
  - `execute()` ON sobre un escenario térmico (helpers de core_inference mockeados):
    misma señal end-to-end, contrato de familia respetado, determinista.
"""

import pytest

from runtime.reasoning.families import imagination as IMAG


# ----------------------------- OFF: byte-idéntico -----------------------------

class TestOff:
    def test_idle_when_flags_off(self, monkeypatch):
        monkeypatch.delenv("RNFE_IMAGINATION_DEEP", raising=False)
        monkeypatch.delenv("RNFE_REASONING_DEEP", raising=False)
        out = IMAG.execute({"observation": {"t": 0.8}, "scenario_metadata": {}, "intervention": "x"})
        assert out == {
            "family": "IMAGINATION", "status": "idle",
            "state_delta": {}, "confidence": 0.0, "cost": 0.0,
        }


# ------------------------- Núcleo puro imagine() ------------------------------

def _synthetic_thermal(*, x0=0.80, drift=0.03, cooling=0.07, threshold=0.85):
    """Mundo con estado: cooling_active persiste; desactivar deja subir la temp."""
    def init():
        return (x0, False)

    def step(lat, iv):
        t, ca = lat
        if iv == "activate_cooling":
            ca = True
        elif iv == "deactivate_cooling":
            ca = False
        t2 = min(1.0, max(0.0, t + drift - (cooling if ca else 0.0)))
        return (t2, ca)

    return dict(
        interventions=["activate_cooling", "deactivate_cooling"],
        init=init, step=step,
        value=lambda lat: lat[0],
        breached=lambda lat: lat[0] >= threshold,
        observe=lambda lat: lat[0],
    )


class TestImagineCore:
    def test_detects_delayed_breach_and_recommends_foresight(self):
        w = _synthetic_thermal(x0=0.80, drift=0.03, threshold=0.85)
        res = IMAG.imagine(chosen="deactivate_cooling", horizon=20, **w)
        # Recomienda enfriar (previsión), pese a que Δ lineal veía "desactivar" como neutral.
        assert res["recommended_intervention"] == "activate_cooling"
        # Desactivar cruza la alarma: 0.80 -> 0.83 (paso 1) -> 0.86 (paso 2) >= 0.85.
        assert res["chosen_breaches_at"] == 2
        assert res["disagrees_with_choice"] is True
        # Enfriar termina muy por debajo; desactivar saturado arriba.
        assert res["per_action_terminal"]["activate_cooling"] < res["per_action_terminal"]["deactivate_cooling"]

    def test_no_breach_no_disagreement_when_choice_is_foresightful(self):
        w = _synthetic_thermal(x0=0.80, drift=0.03, threshold=0.85)
        res = IMAG.imagine(chosen="activate_cooling", horizon=20, **w)
        assert res["recommended_intervention"] == "activate_cooling"
        assert res["chosen_breaches_at"] is None
        assert res["disagrees_with_choice"] is False

    def test_deterministic(self):
        w = _synthetic_thermal()
        a = IMAG.imagine(chosen="deactivate_cooling", horizon=20, **w)
        b = IMAG.imagine(chosen="deactivate_cooling", horizon=20, **w)
        assert a == b

    def test_already_breached_at_step_1(self):
        w = _synthetic_thermal(x0=0.90, drift=0.03, threshold=0.85)  # ya sobre el umbral
        res = IMAG.imagine(chosen="deactivate_cooling", horizon=10, **w)
        assert res["chosen_breaches_at"] == 1


# --------------------------- execute() ON (térmico) ---------------------------

def _mock_thermal_ci(monkeypatch):
    monkeypatch.setenv("RNFE_IMAGINATION_DEEP", "1")
    monkeypatch.setattr(IMAG.ci, "_effect_model",
                        lambda s: {"activate_cooling": -0.07, "deactivate_cooling": 0.0})
    monkeypatch.setattr(IMAG.ci, "main_variable", lambda s: "temp")
    monkeypatch.setattr(IMAG.ci, "resolve_signature", lambda s: None)
    monkeypatch.setattr(IMAG.ci, "optimization_direction", lambda s, sig: "minimize")


class TestExecuteOn:
    def _state(self, chosen, temp=0.80):
        return {
            "observation": {"temp": temp, "cooling_active": False},
            "scenario_metadata": {"alarm_threshold": 0.85},
            "intervention": chosen,
        }

    def test_foresight_warns_and_disagrees_on_myopic_choice(self, monkeypatch):
        _mock_thermal_ci(monkeypatch)
        out = IMAG.execute(self._state("deactivate_cooling"))
        sd = out["state_delta"]
        assert sd["imagination_active"] is True
        assert sd["imagination_speculative"] is True
        assert sd["imagination_recommended_intervention"] == "activate_cooling"
        assert sd["imagination_chosen_breaches_at"] == 2
        assert sd["imagination_disagrees_with_choice"] is True
        assert out["status"] == "warn"

    def test_ok_when_choice_matches_foresight(self, monkeypatch):
        _mock_thermal_ci(monkeypatch)
        out = IMAG.execute(self._state("activate_cooling"))
        assert out["status"] == "ok"
        assert out["state_delta"]["imagination_disagrees_with_choice"] is False
        assert out["state_delta"]["imagination_chosen_breaches_at"] is None

    def test_contract_and_determinism(self, monkeypatch):
        _mock_thermal_ci(monkeypatch)
        st = self._state("deactivate_cooling")
        out = IMAG.execute(st)
        # Contrato de familia
        assert out["family"] == "IMAGINATION"
        assert 0.0 <= out["confidence"] <= 1.0
        assert out["cost"] >= 0.0
        assert out["recommended_next_family"] == "PROB"
        # JSON-safe
        import json
        json.dumps(out)
        # Determinista
        assert IMAG.execute(st) == out

    def test_idle_signal_when_not_thermal(self, monkeypatch):
        monkeypatch.setenv("RNFE_IMAGINATION_DEEP", "1")
        monkeypatch.setattr(IMAG.ci, "_effect_model", lambda s: {"buy": 0.1, "sell": -0.1})
        out = IMAG.execute({"observation": {"x": 0.5}, "scenario_metadata": {}, "intervention": "buy"})
        assert out["status"] == "idle"
        assert out["state_delta"] == {"imagination_active": False}
        assert out["failure_mode"] == "imagination_no_world_model"


# ------------------------- Compuerta gated (Fase 3) ---------------------------

class TestGate:
    def _sd(self, **over):
        base = {
            "imagination_active": True,
            "imagination_disagrees_with_choice": True,
            "imagination_chosen_breaches_at": 3,
            "imagination_recommended_intervention": "shed_load",
        }
        base.update(over)
        return base

    def test_opens_when_all_conditions_met(self):
        g = IMAG.gate(self._sd(), checkpoint_healthy=True, risk=0.2)
        assert g == {"override": True, "intervention": "shed_load", "reason": "gated_override"}

    def test_refuses_without_healthy_checkpoint(self):
        g = IMAG.gate(self._sd(), checkpoint_healthy=False, risk=0.2)
        assert g["override"] is False and g["reason"] == "no_healthy_checkpoint"

    def test_refuses_when_risk_too_high(self):
        g = IMAG.gate(self._sd(), checkpoint_healthy=True, risk=0.9)
        assert g["override"] is False and g["reason"] == "risk_too_high"

    def test_refuses_when_no_predicted_breach(self):
        g = IMAG.gate(self._sd(imagination_chosen_breaches_at=None), checkpoint_healthy=True, risk=0.2)
        assert g["override"] is False and g["reason"] == "no_predicted_breach"

    def test_refuses_when_no_disagreement(self):
        g = IMAG.gate(self._sd(imagination_disagrees_with_choice=False), checkpoint_healthy=True, risk=0.2)
        assert g["override"] is False and g["reason"] == "no_disagreement"

    def test_refuses_when_inactive(self):
        g = IMAG.gate({"imagination_active": False}, checkpoint_healthy=True, risk=0.2)
        assert g["override"] is False and g["reason"] == "imagination_inactive"
