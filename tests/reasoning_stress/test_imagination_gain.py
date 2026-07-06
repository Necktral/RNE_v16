"""Medición de ganancia de A11 en el mundo-trampa `deferred_load_trap`.

Compara dos políticas sobre el mismo escenario:
  - REACTIVA: greedy sobre el effect-model LINEAL (elige la mayor reducción inmediata
    = boost_throughput) — lo que ve el núcleo reactivo/1-paso → cae en la trampa.
  - PREVISORA: sigue la recomendación de A11 (imagina H pasos el mundo con deuda →
    shed_load) → evita el rebote.

Cuantifica: breaches de alarma y carga media/terminal. La previsión debe evitar la
trampa. Es la evidencia que justifica mover A11 a ejecución gated (Fase 3).
"""

import pytest

from runtime.world.deferred_load_scenario import DeferredLoadScenario
from runtime.reasoning.families import imagination as IMAG


EXT = 0.04
STEPS = 25
THRESHOLD = 0.85


def _run(policy_fn):
    sc = DeferredLoadScenario(alarm_threshold=THRESHOLD)
    breaches, loads = 0, []
    for _ in range(STEPS):
        obs = sc.observe()
        iv = policy_fn(sc, obs)
        tr = sc.factual_transition(intervention=iv, external_input=EXT)
        loads.append(tr.state["load"])
        if tr.alarm:
            breaches += 1
    return {
        "breaches": breaches,
        "final_load": round(loads[-1], 4),
        "mean_load": round(sum(loads) / len(loads), 4),
    }


def _reactive(sc, obs):
    """Greedy sobre la magnitud inmediata de la firma lineal (lo que ve el reactivo)."""
    sig = sc.causal_signature
    linear = {
        e.intervention_name: (-e.expected_magnitude if e.expected_direction == "-" else e.expected_magnitude)
        for e in sig.intervention_effects
    }
    return min(linear, key=lambda k: linear[k])  # mayor reducción inmediata → boost


def _foresight(sc, obs):
    """Sigue la recomendación de A11 imaginando el mundo con deuda."""
    world = IMAG.deferred_load_world(
        x0=obs.state["load"],
        debt=obs.state.get("debt", 0.0),
        boost_effect=0.15,
        shed_effect=0.05,
        drift=EXT,
        threshold=sc.alarm_threshold,
        direction="minimize",
    )
    res = IMAG.imagine(**world, horizon=IMAG._HORIZON)
    return res["recommended_intervention"]


class TestImaginationGain:
    def test_reactive_falls_into_trap(self):
        r = _run(_reactive)
        assert r["breaches"] > 0                 # el reactivo cruza la alarma
        assert _reactive(DeferredLoadScenario(), DeferredLoadScenario().observe()) == "boost_throughput"

    def test_foresight_avoids_trap(self):
        f = _run(_foresight)
        assert f["breaches"] == 0                 # la previsión no cruza la alarma
        assert _foresight(DeferredLoadScenario(), DeferredLoadScenario().observe()) == "shed_load"

    def test_foresight_beats_reactive(self, capsys):
        reactive = _run(_reactive)
        foresight = _run(_foresight)
        # Ganancia medible: menos breaches y menor carga terminal/media.
        assert foresight["breaches"] < reactive["breaches"]
        assert foresight["final_load"] < reactive["final_load"]
        assert foresight["mean_load"] < reactive["mean_load"]
        # Deja el número visible en el reporte de tests.
        with capsys.disabled():
            print(f"\n[A11 gain @ deferred_load_trap, {STEPS} pasos]")
            print(f"  REACTIVA (boost): breaches={reactive['breaches']:2d}  "
                  f"final={reactive['final_load']}  mean={reactive['mean_load']}")
            print(f"  PREVISORA (A11):  breaches={foresight['breaches']:2d}  "
                  f"final={foresight['final_load']}  mean={foresight['mean_load']}")
