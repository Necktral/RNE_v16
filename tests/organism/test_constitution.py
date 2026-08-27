"""Tests para runtime/organism/constitution.py — Constitución del organismo."""

from __future__ import annotations

import pytest

from runtime.organism.state import (
    OrganismBeliefState,
    OrganismState,
    PolicyState,
    IdentityState,
    ViabilityState,
)
from runtime.organism.constitution import (
    ConstitutionalInvariant,
    ConstitutionalValidation,
    OrganismConstitution,
    HARD_INVARIANTS,
    SOFT_INVARIANTS,
    MUTABLE_COMPONENTS,
    IMMUTABLE_COMPONENTS,
)


class TestOrganismConstitution:
    def test_default_constitution(self):
        c = OrganismConstitution()
        assert len(c.hard_invariants) == len(HARD_INVARIANTS)
        assert len(c.soft_invariants) == len(SOFT_INVARIANTS)

    def test_healthy_state_validates(self):
        c = OrganismConstitution()
        s = OrganismState(
            belief=OrganismBeliefState(
                alarm_probability=0.1,
                intervention_efficacy=0.8,
                causal_support_confidence=0.9,
                memory_purity_estimate=0.95,
                trace_integrity_confidence=0.85,
            ),
            identity=IdentityState(lineage_id="L1"),
        )
        v = c.validate(s)
        assert v.is_valid is True
        assert v.verdict == "valid"
        assert v.hard_violation_count == 0

    def test_low_memory_purity_triggers_hard_violation(self):
        c = OrganismConstitution()
        s = OrganismState(
            belief=OrganismBeliefState(
                memory_purity_estimate=0.20,
                causal_support_confidence=0.9,
                trace_integrity_confidence=0.8,
            ),
            identity=IdentityState(lineage_id="L1"),
        )
        v = c.validate(s)
        assert v.is_valid is False
        assert v.hard_violation_count > 0
        names = [vl.invariant_name for vl in v.violations]
        assert "min_memory_purity" in names

    def test_low_trace_integrity_triggers_hard_violation(self):
        c = OrganismConstitution()
        s = OrganismState(
            belief=OrganismBeliefState(
                trace_integrity_confidence=0.10,
                memory_purity_estimate=0.9,
                causal_support_confidence=0.9,
            ),
            identity=IdentityState(lineage_id="L1"),
        )
        v = c.validate(s)
        assert v.is_valid is False
        names = [vl.invariant_name for vl in v.violations]
        assert "min_trace_integrity" in names

    def test_high_degradation_triggers_hard_violation(self):
        c = OrganismConstitution()
        s = OrganismState(
            belief=OrganismBeliefState(
                causal_support_confidence=0.9,
                memory_purity_estimate=0.9,
                trace_integrity_confidence=0.9,
            ),
            viability=ViabilityState(accumulated_degradation=0.90),
            identity=IdentityState(lineage_id="L1"),
        )
        v = c.validate(s)
        assert v.is_valid is False
        names = [vl.invariant_name for vl in v.violations]
        assert "baseline_not_degraded" in names

    def test_no_rollback_triggers_hard_violation(self):
        c = OrganismConstitution()
        s = OrganismState(
            belief=OrganismBeliefState(
                causal_support_confidence=0.9,
                memory_purity_estimate=0.9,
                trace_integrity_confidence=0.9,
            ),
            viability=ViabilityState(rollback_readiness=False),
            identity=IdentityState(lineage_id="L1"),
        )
        v = c.validate(s)
        assert v.is_valid is False
        names = [vl.invariant_name for vl in v.violations]
        assert "rollback_available" in names

    def test_empty_lineage_triggers_hard_violation(self):
        c = OrganismConstitution()
        s = OrganismState(
            belief=OrganismBeliefState(
                causal_support_confidence=0.9,
                memory_purity_estimate=0.9,
                trace_integrity_confidence=0.9,
            ),
            identity=IdentityState(lineage_id=""),
        )
        v = c.validate(s)
        assert v.is_valid is False
        names = [vl.invariant_name for vl in v.violations]
        assert "lineage_coherent" in names

    def test_high_policy_drift_triggers_soft_violation(self):
        c = OrganismConstitution()
        s = OrganismState(
            belief=OrganismBeliefState(
                causal_support_confidence=0.9,
                memory_purity_estimate=0.9,
                trace_integrity_confidence=0.9,
            ),
            policy=PolicyState(accumulated_drift=0.70),
            identity=IdentityState(lineage_id="L1"),
        )
        v = c.validate(s)
        # Still valid (soft violation only)
        assert v.is_valid is True
        assert v.soft_violation_count > 0
        names = [vl.invariant_name for vl in v.violations if vl.severity == "soft"]
        assert "policy_stability" in names or "drift_tolerance" in names

    def test_multiple_hard_violations_yield_rollback(self):
        c = OrganismConstitution()
        s = OrganismState(
            belief=OrganismBeliefState(
                memory_purity_estimate=0.10,
                trace_integrity_confidence=0.10,
                causal_support_confidence=0.05,
            ),
            viability=ViabilityState(
                accumulated_degradation=0.95,
                rollback_readiness=False,
            ),
            identity=IdentityState(lineage_id=""),
        )
        v = c.validate(s)
        assert v.verdict == "rollback"
        assert v.hard_violation_count >= 3

    def test_mutable_immutable(self):
        c = OrganismConstitution()
        assert c.is_mutable("transport_parameters")
        assert c.is_immutable("baseline_semantics")
        assert not c.is_mutable("baseline_semantics")
        assert not c.is_immutable("transport_parameters")

    def test_constitution_hash_deterministic(self):
        c = OrganismConstitution()
        h1 = c.constitution_hash()
        h2 = c.constitution_hash()
        assert h1 == h2
        assert len(h1) == 16


class TestLaAleyPermiteEquivocarse:
    """P12.5 — el organismo puede DESCUBRIR que su modelo causal era falso, y sobrevivir.

    La ley separa dos TIPOS que antes iban en el mismo producto duro:
      - FACULTADES (`trace`, `purity`): ¿puedo saber algo, en absoluto?  Rotas ⇒ cuarentena.
      - HALLAZGO (`causal_support`): ¿qué dice mi última medición del mundo?  Cambia ⇒
        actualizo el modelo, NO me muero.
    """

    @staticmethod
    def _state(*, causal, trace=0.80, purity=1.0) -> OrganismState:
        return OrganismState(
            belief=OrganismBeliefState(
                alarm_probability=0.1,
                intervention_efficacy=0.8,
                causal_support_confidence=causal,
                memory_purity_estimate=purity,
                trace_integrity_confidence=trace,
            ),
            identity=IdentityState(lineage_id="L1"),
        )

    def test_measured_contradiction_does_not_quarantine(self):
        """EL TEST QUE FALSIFICA EL ARREGLO.

        `belief_state` emite causal = 0.20 EXACTO cuando el contrafactual mide una
        CONTRADICCIÓN (`reality/belief_state.py:226`).  Con las facultades PERFECTAS:

            LEY VIEJA (triádico = causal × trace × purity):
                0.20 × 1.0 × 1.0 = 0.20  <  0.50  ⇒  VIOLACIÓN HARD ⇒ cuarentena.
            LEY NUEVA (aparato = trace × purity):
                1.0 × 1.0 = 1.00  >=  0.50        ⇒  sin violación.

        Con la ley vieja este test se pone ROJO por aritmética pura, sin tocar un umbral.
        Es la prueba de que el organismo tenía PROHIBIDO percibir una contradicción.
        """
        c = OrganismConstitution()
        v = c.validate(self._state(causal=0.20, trace=1.0, purity=1.0))

        assert v.hard_violation_count == 0, [x.invariant_name for x in v.violations]
        assert v.is_valid is True
        assert v.verdict == "valid"
        assert v.is_fully_verified is True  # nada quedó sin mirar

        # Y en particular NO por haberse aflojado el piso causal: 0.20 == el piso, y el
        # invariante del hallazgo compara con `<` estricto ⇒ 0.20 NO es violación.
        assert c.config["min_causal_support"] == 0.20
        assert c.config["triadic_closure_threshold"] == 0.50

    def test_contradiction_with_real_runner_beliefs_does_not_quarantine(self):
        """Los valores REALES del runner vivo (trace=0.80, purity≈0.97-1.0)."""
        c = OrganismConstitution()
        v = c.validate(self._state(causal=0.20, trace=0.80, purity=0.97))
        # Ley vieja: 0.20 × 0.80 × 0.97 = 0.155 < 0.50 ⇒ cuarentena (era el bug vivo).
        # Ley nueva: 0.80 × 0.97 = 0.776 >= 0.50 ⇒ sano.
        assert v.hard_violation_count == 0
        assert v.verdict == "valid"

    def test_corrupt_trace_is_still_rejected(self):
        """LA COMPUERTA DEL APARATO NO SE AFLOJÓ: traza corrupta ⇒ sigue en cuarentena.

        Sacar el hallazgo causal del producto NO puede volver permisivo el gate de las
        facultades.  Una traza rota es una facultad rota: el organismo no puede saber nada,
        y ahí la cuarentena es la respuesta CORRECTA.
        """
        c = OrganismConstitution()
        v = c.validate(self._state(causal=0.90, trace=0.10, purity=1.0))  # causal PERFECTO
        names = [x.invariant_name for x in v.violations]
        assert v.is_valid is False
        assert v.verdict in ("quarantine", "rollback")
        # Lo cazan DOS invariantes independientes: el mínimo y el producto del aparato.
        assert "min_trace_integrity" in names  # 0.10 < 0.30
        assert "triadic_closure" in names      # 0.10 × 1.0 = 0.10 < 0.50

    def test_corrupt_memory_is_still_rejected(self):
        c = OrganismConstitution()
        v = c.validate(self._state(causal=0.90, trace=1.0, purity=0.20))
        names = [x.invariant_name for x in v.violations]
        assert v.is_valid is False
        assert "min_memory_purity" in names   # 0.20 < 0.40
        assert "triadic_closure" in names     # 1.0 × 0.20 = 0.20 < 0.50

    def test_apparatus_gate_still_binds_between_the_minimums(self):
        """El producto del aparato NO es redundante con los mínimos individuales.

        trace=0.55 y purity=0.60 pasan AMBOS mínimos (0.30 / 0.40) pero su producto
        (0.33) no llega al umbral: el organismo está degradado en las DOS facultades a la
        vez, y eso el producto sí lo caza.
        """
        c = OrganismConstitution()
        v = c.validate(self._state(causal=0.90, trace=0.55, purity=0.60))
        names = [x.invariant_name for x in v.violations]
        assert "min_trace_integrity" not in names
        assert "min_memory_purity" not in names
        assert "triadic_closure" in names  # 0.55 × 0.60 = 0.33 < 0.50
        assert v.is_valid is False

    def test_unmeasured_faculty_abstains_and_does_not_approve(self):
        """Facultad NO MEDIDA ⇒ abstención: ni dispara ni aprueba (patrón de 3 estados)."""
        c = OrganismConstitution()
        v = c.validate(self._state(causal=0.90, trace=None, purity=1.0))
        assert "triadic_closure" in v.abstained_invariants
        assert "triadic_closure" not in [x.invariant_name for x in v.violations]
        assert "trace_integrity_confidence" in v.unmeasured_axes
        # `is_valid` sólo dice "no se detectó nada malo"; NO dice "verificado sano".
        assert v.is_fully_verified is False

    def test_unmeasured_causal_axis_no_longer_blocks_the_apparatus_check(self):
        """El eje causal no medido ya NO impide evaluar el cierre del aparato.

        Antes, `causal=None` forzaba a `triadic_closure` a abstenerse (el eje entraba en su
        producto).  Ahora las facultades se pueden mirar igual: sólo el HALLAZGO se abstiene.
        """
        c = OrganismConstitution()
        v = c.validate(self._state(causal=None, trace=0.80, purity=0.98))
        assert v.abstained_invariants == ("factual_counterfactual_coherence",)
        assert "triadic_closure" not in v.abstained_invariants
        assert v.hard_violation_count == 0
        assert v.is_fully_verified is False  # el hallazgo sigue sin poder mirarse

    def test_causal_finding_is_exposed_and_named(self):
        """La señal se MIDE, se EXPONE y se NOMBRA — aunque hoy NO TENGA CONSUMIDOR."""
        c = OrganismConstitution()
        contradiction = c.validate(self._state(causal=0.20))
        assert contradiction.causal_finding == 0.20
        assert contradiction.causal_finding_measured is True

        support = c.validate(self._state(causal=0.90))
        assert support.causal_finding == 0.90

        unmeasured = c.validate(self._state(causal=None))
        assert unmeasured.causal_finding is None
        assert unmeasured.causal_finding_measured is False

    def test_causal_floor_still_fires_below_the_declared_floor(self):
        """El hallazgo conserva su invariante: por DEBAJO del piso, sí es violación."""
        c = OrganismConstitution()
        v = c.validate(self._state(causal=0.05))  # 0.05 < 0.20
        names = [x.invariant_name for x in v.violations]
        assert "factual_counterfactual_coherence" in names
        assert v.is_valid is False

    def test_live_producer_can_never_trigger_the_causal_floor(self):
        """⚠ PIN DEL HALLAZGO REPORTADO: el invariante del piso causal es INALCANZABLE.

        El productor vivo (`reality/belief_state.py:221-230` → `trajectory_state_machine`)
        emite EXACTAMENTE tres valores para el eje causal.  NINGUNO es `< 0.20`.  ⇒ En el
        camino vivo, `factual_counterfactual_coherence` NO PUEDE DISPARAR.

        Este test NO lo arregla (fabricar un número para que un detector pueda dispararse
        sería la enfermedad).  Lo PINEA para que el día que alguien cambie el mapeo —p.ej.
        emitiendo un GRADO de contradicción en vez de la constante 0.20— este test se ponga
        rojo y la decisión sea consciente y humana, no un efecto colateral.
        """
        c = OrganismConstitution()
        for produced in (0.90, 0.20, None):  # el set COMPLETO del productor vivo
            v = c.validate(self._state(causal=produced, trace=1.0, purity=1.0))
            names = [x.invariant_name for x in v.violations]
            assert "factual_counterfactual_coherence" not in names, produced


class TestConstitutionalValidation:
    def test_quarantine_verdict(self):
        """One or two hard violations → quarantine."""
        c = OrganismConstitution()
        s = OrganismState(
            belief=OrganismBeliefState(
                memory_purity_estimate=0.10,
                trace_integrity_confidence=0.9,
                causal_support_confidence=0.9,
            ),
            identity=IdentityState(lineage_id="L1"),
        )
        v = c.validate(s)
        # Only memory purity + triadic closure might be violated
        assert v.verdict in ("quarantine", "rollback")
        assert v.is_valid is False
