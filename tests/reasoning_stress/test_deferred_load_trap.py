"""Mundo-trampa de consecuencia diferida — prueba que la trampa temporal existe.

Verifica, a nivel de dinámica del mundo (sin stack de razonamiento):
  - `boost_throughput` se ve MEJOR que `shed_load` en el paso 1 (menor carga) →
    el núcleo reactivo/1-paso lo prefiere.
  - Repetir `boost_throughput` **cruza la alarma** en pocos pasos (rebote por deuda).
  - Repetir `shed_load` se mantiene seguro indefinidamente.
  - El escenario está registrado y su firma causal favorece linealmente a boost.
"""

import pytest

from runtime.world.deferred_load_scenario import DeferredLoadScenario
from runtime.world import registry


EXT = 0.04  # entrada externa por paso (base_drift), como usa el episode runner


def _roll(policy: str, steps: int, *, initial_load=0.70, threshold=0.85):
    sc = DeferredLoadScenario(initial_load=initial_load, alarm_threshold=threshold)
    loads, breached_at = [], None
    for i in range(steps):
        tr = sc.factual_transition(intervention=policy, external_input=EXT)
        loads.append(tr.state["load"])
        if breached_at is None and tr.alarm:
            breached_at = i + 1
    return loads, breached_at


class TestTrapDynamics:
    def test_boost_looks_best_at_step_1(self):
        """La trampa es tentadora: boost baja más la carga en el primer paso."""
        sc_b = DeferredLoadScenario()
        sc_s = DeferredLoadScenario()
        load_boost = sc_b.factual_transition(intervention="boost_throughput", external_input=EXT).state["load"]
        load_shed = sc_s.factual_transition(intervention="shed_load", external_input=EXT).state["load"]
        assert load_boost < load_shed  # reactivo/1-paso preferiría boost

    def test_repeated_boost_breaches_alarm(self):
        """Consecuencia diferida: repetir boost rebota y cruza la alarma."""
        loads, breached_at = _roll("boost_throughput", steps=10)
        assert breached_at is not None
        assert breached_at <= 5  # rebota rápido
        assert max(loads) >= 0.85

    def test_repeated_shed_stays_safe(self):
        """La acción previsora se mantiene bajo el umbral indefinidamente."""
        loads, breached_at = _roll("shed_load", steps=30)
        assert breached_at is None
        assert max(loads) < 0.85

    def test_boost_worse_than_shed_long_horizon(self):
        """A 10 pasos, boost termina muy por encima de shed (la trampa se paga)."""
        boost_loads, _ = _roll("boost_throughput", steps=10)
        shed_loads, _ = _roll("shed_load", steps=10)
        assert boost_loads[-1] > shed_loads[-1]


class TestRegistrationAndSignature:
    def test_registered_and_loadable(self):
        sc = registry.get_scenario("deferred_load_trap")
        assert isinstance(sc, DeferredLoadScenario)
        assert sc.config.main_variable == "load"
        assert set(sc.config.interventions) == {"boost_throughput", "shed_load"}

    def test_linear_signature_favors_the_trap(self):
        """La firma causal (magnitud inmediata) hace que boost parezca el mejor arreglo."""
        sig = DeferredLoadScenario().causal_signature
        mags = {e.intervention_name: e.expected_magnitude for e in sig.intervention_effects}
        # boost tiene mayor magnitud correctiva inmediata que shed → el effect-model
        # lineal (core_inference._effect_model) lo prefiere, cayendo en la trampa.
        assert mags["boost_throughput"] > mags["shed_load"]

    def test_counterfactual_does_not_mutate(self):
        sc = DeferredLoadScenario()
        before = sc.observe().state["load"]
        sc.simulate_counterfactual(intervention="boost_throughput", external_input=EXT)
        assert sc.observe().state["load"] == before
