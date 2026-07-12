"""Constitución operativa del organismo RNFE.

Define la ley viva del organismo en código.  Contiene:
- Invariantes duros: no pueden violarse sin cuarentena o rollback
- Invariantes blandos: pueden relajarse bajo sandbox
- Reglas de mutación: qué puede cambiar el organismo
- Reglas de no-mutación: qué requiere aprobación superior

La constitución convierte a PromotionGate en una parte de la corte,
no en toda la corte.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Literal, Tuple

from .state import OrganismState


# ── Invariant types ──────────────────────────────────────────────────────────

InvariantSeverity = Literal["hard", "soft"]

@dataclass(frozen=True)
class ConstitutionalInvariant:
    """Invariante constitucional del organismo.

    Attributes:
        name: Nombre único del invariante.
        severity: 'hard' (cuarentena/rollback si violado) o 'soft' (relajable en sandbox).
        description: Descripción del invariante.
        check_fn_name: Nombre de la función de chequeo.
    """

    name: str
    severity: InvariantSeverity
    description: str
    check_fn_name: str = ""


@dataclass(frozen=True)
class InvariantViolation:
    """Violación de un invariante detectada."""

    invariant_name: str
    severity: InvariantSeverity
    evidence_value: float
    threshold: float
    description: str


@dataclass(frozen=True)
class InvariantAbstention:
    """Invariante que NO PUDO EVALUARSE porque le faltó un eje de evidencia.

    No es una violación (no dispara: sería falso pánico) y **no es una aprobación**
    (sería la mentira original). Es la tercera salida: el invariante se abstiene y lo
    declara. **Un invariante que no pudo evaluarse NO es un invariante satisfecho.**

    Attributes:
        invariant_name: Invariante que se abstuvo.
        severity: Severidad que HABRÍA tenido de haber podido evaluarse.
        unmeasured_axes: Ejes de evidencia ausentes, por nombre.
        description: Motivo legible.
    """

    invariant_name: str
    severity: InvariantSeverity
    unmeasured_axes: Tuple[str, ...]
    description: str


#: Resultado de un chequeo: violación, abstención, o nada (satisfecho y verificado).
CheckOutcome = "InvariantViolation | InvariantAbstention | None"


@dataclass(frozen=True)
class ConstitutionalValidation:
    """Resultado de validación constitucional.

    Attributes:
        is_valid: True si no se DETECTÓ ninguna violación hard.  **No significa
            "verificado sano"**: si `abstained_invariants` no está vacío, hay invariantes
            que no pudieron evaluarse.  Para "sano y verificado" está `is_fully_verified`.
        verdict: 'valid', 'quarantine', 'rollback'.
        violations: Lista de violaciones detectadas.
        abstentions: Invariantes que no pudieron evaluarse por falta de evidencia.
        hard_violation_count: Número de violaciones hard.
        soft_violation_count: Número de violaciones soft.
        margin_to_threshold: Margen mínimo sobre cualquier invariante hard EVALUABLE.
        causal_finding: HALLAZGO causal del episodio (ver `causal_finding` abajo).
    """

    is_valid: bool
    verdict: Literal["valid", "quarantine", "rollback"]
    violations: Tuple[InvariantViolation, ...]
    hard_violation_count: int
    soft_violation_count: int
    margin_to_threshold: float
    abstentions: Tuple[InvariantAbstention, ...] = ()

    #: HALLAZGO causal del episodio: ``belief.causal_support_confidence`` tal cual se midió
    #: (``0.90`` soporte, ``0.20`` contradicción, ``None`` = el contrafactual no discriminó).
    #:
    #: ⚠ **SEÑAL SIN CONSUMIDOR — LEER ANTES DE ASUMIR QUE ESTÁ RESUELTO (P12.5).**
    #:
    #: "Contradicción persistente ⇒ mi modelo causal está mal ⇒ debería volverme conservador
    #: (bajar la ganancia de mi razonamiento)" es una señal REAL y VALIOSA.  **Hoy no tiene a
    #: dónde ir.**  No existe el órgano que module la ganancia del razonamiento con esta
    #: señal, y este paquete NO lo inventa (es P-TALLO).
    #:
    #: Antes de P12.5 el hallazgo SÍ tenía un consumidor — pero era el equivocado: entraba en
    #: el producto de ``triadic_closure`` y mandaba a CUARENTENA al organismo por el mero
    #: hecho de percibir una contradicción.  Eso no era "consumir la señal", era prohibirla.
    #: P12.5 lo sacó de ahí.  El único invariante que hoy mira el eje causal
    #: (``factual_counterfactual_coherence``) NO PUEDE DISPARAR con el set de valores que el
    #: productor emite (ver ``_check_coherence``).
    #:
    #: ⇒ NINGÚN INVARIANTE CONSTITUCIONAL actúa hoy sobre el eje causal: se MIDE, se EXPONE
    #: y se PERSISTE, y ninguna compuerta lo consume.  Que esto no se lea como "ya está
    #: resuelto" es el motivo de este comentario.
    #:
    #: OJO — el alcance de esa frase es la CONSTITUCIÓN, no el organismo entero.  El eje
    #: causal SÍ tiene consumidores fuera de acá, y hay que saberlo antes de tocarlo:
    #: ``state.py`` y ``reality/belief_state.py`` lo pesan 0.25 en ``composite_confidence``;
    #: ``reasoning/families/core_inference.py`` lo usa en la calibración; ``reality/transport.py``
    #: y ``reality/regime_renormalization.py`` lo proyectan; ``certification/transfer_assessment.py``
    #: lo lee.  Lo que falta no es un lector: es un consumidor que DECIDA algo con él.
    causal_finding: float | None = None

    @property
    def causal_finding_measured(self) -> bool:
        """True si el episodio pudo MEDIR el eje causal (el contrafactual discriminó)."""
        return self.causal_finding is not None

    @property
    def abstained_invariants(self) -> Tuple[str, ...]:
        """Nombres de los invariantes que no pudieron evaluarse."""
        return tuple(a.invariant_name for a in self.abstentions)

    @property
    def unmeasured_axes(self) -> Tuple[str, ...]:
        """Ejes de evidencia ausentes que forzaron alguna abstención."""
        axes: List[str] = []
        for abstention in self.abstentions:
            for axis in abstention.unmeasured_axes:
                if axis not in axes:
                    axes.append(axis)
        return tuple(axes)

    @property
    def is_fully_verified(self) -> bool:
        """True sólo si NO hubo violaciones hard **y** todos los invariantes se evaluaron.

        `is_valid` responde "¿se detectó algo malo?".  Esta responde "¿se pudo mirar?".
        Confundirlas es leer "sin violación" como "verificado sano".
        """
        return self.is_valid and not self.abstentions


# ── Hard invariant checks ────────────────────────────────────────────────────

def _check_triadic_closure(state: OrganismState, config: Dict[str, float]):
    """Cierre del APARATO: trace integrity × memory purity.  ¿Puedo saber algo, siquiera?

    P12.5 — ERROR DE CATEGORÍA CORREGIDO.  Este producto multiplicaba TRES factores:
    ``causal_support × trace_integrity × memory_purity``.  Pero esos factores NO SON DEL
    MISMO TIPO:

      - ``trace_integrity`` y ``memory_purity`` son **FACULTADES**: responden "¿mi aparato
        de conocer está sano?".  Si se rompen, el organismo no puede saber NADA —
        cuarentena es la respuesta CORRECTA.
      - ``causal_support`` es un **HALLAZGO**: el veredicto actual de una medición sobre el
        mundo ("mi modelo causal predijo bien / predijo mal").  Es una creencia sobre el
        mundo, no una capacidad del sujeto.

    Multiplicarlos en un mismo producto duro volvía *"aprendí que mi modelo causal era
    falso"* **indistinguible** de *"tengo la memoria corrupta"*.  Y con los números reales
    era peor que una confusión: era una PROHIBICIÓN.  ``belief_state`` emite
    ``causal = 0.20`` cuando el contrafactual mide una CONTRADICCIÓN.  Con facultades
    PERFECTAS (trace = purity = 1.0):

        0.20 × 1.0 × 1.0 = 0.20  <  0.50  ⇒  VIOLACIÓN HARD, por pura aritmética.

    Es decir: el organismo tenía **constitucionalmente prohibido descubrir que su modelo
    causal era falso**.  Cualquier episodio en que midiera una contradicción lo mandaba a
    cuarentena, aunque su aparato de conocer estuviera intacto.  El umbral de coherencia
    causal (``min_causal_support = 0.20``, ``_check_coherence``) declara 0.20 como PISO
    ACEPTABLE (dispara con ``< 0.20``, estricto); este invariante lo castigaba igual.  Dos
    invariantes hard que se contradecían exactamente en el valor que el organismo emite
    cuando PERCIBE.

    El arreglo es ESTRUCTURAL, no numérico: el hallazgo sale del producto de las facultades.
    El umbral (0.50) NO SE TOCA; el mapeo ``contradiction → 0.20`` NO SE TOCA.  El hallazgo
    causal conserva su propio invariante (``_check_coherence``), que es el honesto: sólo
    dispara si el soporte cae POR DEBAJO del piso declarado.  **Medir una contradicción no
    es un delito constitucional: es percibir.**

    Tres salidas (patrón de P12):
      - facultad NO MEDIDA ⇒ **abstención**: no dispara (falso pánico) y NO APRUEBA (la
        mentira original).  Hoy ambas facultades están tipadas ``float`` y el productor vivo
        (``trajectory_state_machine``) siempre las mide, así que esta rama NO SE ALCANZA en
        el camino vivo — existe para que el día que una facultad pueda faltar el invariante
        se ABSTENGA en vez de aprobar por defecto.  (Se ejercita en tests pasando ``None``.)
      - producto < umbral ⇒ violación hard.
      - producto >= umbral ⇒ satisfecho y verificado.

    NOMBRE: ``triadic_closure`` es un MISNOMER HISTÓRICO y se conserva a propósito.  El
    "cierre triádico" del canon (A3, ``canon/normative/CANON_RNFE_v3_2_rc1.md``) es el ciclo
    **Significado → Forma → Mundo → Significado** — que NO es este producto de confianzas, y
    que de hecho se mide en otro lado (``certification/coherence_obstruction.cycle_error``,
    el término |Φ_{M→S}·Φ_{F→M}·Φ_{S→F} − I|).  El nombre no se cambia acá porque viaja en
    ``IdentityState.active_invariants`` (``lineage.py:263``), se persiste
    (``snapshot.py:85``), alimenta la comparación de continuidad identitaria
    (``state.py``, ``distance_to``) y entra en ``constitution_hash()``: renombrarlo mutaría la
    identidad de todo organismo ya persistido.  Es deuda de nombre, reportada, no tapada.

    ADVERTENCIA para el que toque este umbral: en el CAMINO VIVO este producto es HOY EL
    ÚNICO detector de traza degradada, porque ``min_trace_integrity`` (0.30) es INALCANZABLE
    — el productor (``reality/belief_state.py``: ``0.80 if trace else 0.40``) tiene piso 0.40,
    así que "no tengo traza" ya saca 0.40 y ese invariante nunca puede disparar.

    Antes de P12.5 la barra efectiva sobre ``trace × purity`` no era 0.50 sino ``0.50 / causal``
    (= 0.5556 en episodios de soporte): un umbral que NADIE declaró y que fluctuaba con una
    variable sin relación con las facultades.  Ahora el umbral declarado ES el efectivo.  Eso
    admite organismos en la banda [0.50, 0.5556) que la ley vieja rechazaba; el peor es
    ``trace=0.50, purity=1.00`` — o sea SIN TRAZA (0.40) más el bonus de certificación (+0.10).
    Subir este número a 0.5556 sería fabricar una constante para preservar un accidente: el
    bug real está en el PRODUCTOR (chequea si la lista está vacía, no la integridad; y un
    certificado le suma +0.10 a una traza que no existe).  Ver backlog.
    """
    threshold = config.get("triadic_closure_threshold", 0.50)
    trace = state.belief.trace_integrity_confidence
    purity = state.belief.memory_purity_estimate

    unmeasured = tuple(
        axis
        for axis, value in (
            ("trace_integrity_confidence", trace),
            ("memory_purity_estimate", purity),
        )
        if value is None
    )
    if unmeasured:
        return InvariantAbstention(
            invariant_name="triadic_closure",
            severity="hard",
            unmeasured_axes=unmeasured,
            description=(
                "Cierre del aparato NO EVALUABLE: no se midió "
                f"{', '.join(unmeasured)}. Sin violación detectada y sin invariante "
                "satisfecho: una facultad que nadie pudo mirar no es una facultad sana."
            ),
        )

    value = trace * purity
    if value < threshold:
        return InvariantViolation(
            invariant_name="triadic_closure",
            severity="hard",
            evidence_value=round(value, 4),
            threshold=threshold,
            description=(
                f"Apparatus closure product (trace × purity) {value:.4f} < {threshold}"
            ),
        )
    return None


def _check_memory_purity(state: OrganismState, config: Dict[str, float]):
    """FACULTAD — pureza de memoria >= mínimo.  Se abstiene si no se midió."""
    threshold = config.get("min_memory_purity", 0.40)
    purity = state.belief.memory_purity_estimate
    if purity is None:
        return InvariantAbstention(
            invariant_name="min_memory_purity",
            severity="hard",
            unmeasured_axes=("memory_purity_estimate",),
            description=(
                "Pureza de memoria NO EVALUABLE: el eje no se midió. Sin violación "
                "detectada y sin invariante satisfecho."
            ),
        )
    if purity < threshold:
        return InvariantViolation(
            invariant_name="min_memory_purity",
            severity="hard",
            evidence_value=round(purity, 4),
            threshold=threshold,
            description=f"Memory purity {purity:.4f} < {threshold}",
        )
    return None


def _check_trace_integrity(state: OrganismState, config: Dict[str, float]):
    """FACULTAD — integridad de traza >= mínimo.  Se abstiene si no se midió.

    Mismo patrón de tres estados que el cierre del aparato: una facultad que nadie pudo
    mirar NO es una facultad sana (aprobarla sería la mentira original), pero tampoco es
    una facultad rota (dispararla sería falso pánico).  Se declara y se sigue.
    """
    threshold = config.get("min_trace_integrity", 0.30)
    trace = state.belief.trace_integrity_confidence
    if trace is None:
        return InvariantAbstention(
            invariant_name="min_trace_integrity",
            severity="hard",
            unmeasured_axes=("trace_integrity_confidence",),
            description=(
                "Integridad de traza NO EVALUABLE: el eje no se midió. Sin violación "
                "detectada y sin invariante satisfecho."
            ),
        )
    if trace < threshold:
        return InvariantViolation(
            invariant_name="min_trace_integrity",
            severity="hard",
            evidence_value=round(trace, 4),
            threshold=threshold,
            description=f"Trace integrity {trace:.4f} < {threshold}",
        )
    return None


def _check_baseline_integrity(state: OrganismState, config: Dict[str, float]) -> InvariantViolation | None:
    """Baseline no degradado."""
    max_degradation = config.get("max_degradation", 0.80)
    if state.viability.accumulated_degradation >= max_degradation:
        return InvariantViolation(
            invariant_name="baseline_not_degraded",
            severity="hard",
            evidence_value=round(state.viability.accumulated_degradation, 4),
            threshold=max_degradation,
            description=f"Accumulated degradation {state.viability.accumulated_degradation:.4f} >= {max_degradation}",
        )
    return None


def _check_rollback_available(state: OrganismState, config: Dict[str, float]) -> InvariantViolation | None:
    if not state.viability.rollback_readiness:
        return InvariantViolation(
            invariant_name="rollback_available",
            severity="hard",
            evidence_value=0.0,
            threshold=1.0,
            description="Rollback not available",
        )
    return None


def _check_coherence(state: OrganismState, config: Dict[str, float]):
    """Coherencia factual/contrafactual: causal_support >= threshold.  El HALLAZGO.

    Éste es el invariante honesto del eje causal, y el ÚNICO que lo juzga (P12.5 sacó el
    hallazgo del producto de facultades, ``_check_triadic_closure``).  Sólo dispara si el
    soporte causal cae POR DEBAJO del piso declarado — ``min_causal_support = 0.20``,
    comparación ESTRICTA: **``causal == 0.20`` NO es violación**.  La constitución declara
    0.20 como piso aceptable, y medir una contradicción (que es exactamente 0.20) no es un
    delito constitucional.

    Si el eje causal no se midió, no puede evaluarse: se abstiene y lo declara — no dispara
    (falso pánico) y no aprueba (mentira).  Sin contrafactual discriminante no hay coherencia
    factual/contrafactual que juzgar: no hay contraste.

    ⚠ HALLAZGO REPORTADO, NO "ARREGLADO" (P12.5).  Con el productor vivo actual este
    invariante **NO PUEDE DISPARAR**.  El único escritor de ``belief.causal_support_confidence``
    en el camino vivo es ``trajectory_state_machine.advance_state`` (línea ~71), que lo toma
    de ``build_belief_state`` (``reality/belief_state.py:221-230``), cuyo mapeo emite EXACTAMENTE
    tres valores:

        support                    → 0.90
        contradiction              → 0.20   (== umbral ⇒ NO dispara: la comparación es `<`)
        no_discriminating_evidence → None   (⇒ abstención)

    Ninguno es ``< 0.20``.  ⇒ En el camino vivo, ``factual_counterfactual_coherence`` es un
    invariante hard **INALCANZABLE** (sí es alcanzable desde estados sintéticos: un test que
    construya ``OrganismState`` con ``causal=0.05`` lo dispara, y ``test_constitution`` lo hace).
    NO se "arregla" acá inventando un número ni moviendo el umbral: fabricar un valor para que
    un detector pueda dispararse es la misma enfermedad que este paquete vino a desarmar.  Que
    el mapeo deba emitir un grado de contradicción (y no una constante 0.20) es una decisión
    del humano, no del ejecutor.  Queda REPORTADO.
    """
    threshold = config.get("min_causal_support", 0.20)
    causal = state.belief.causal_support_confidence
    if causal is None:
        return InvariantAbstention(
            invariant_name="factual_counterfactual_coherence",
            severity="hard",
            unmeasured_axes=("causal_support_confidence",),
            description=(
                "Coherencia factual/contrafactual NO EVALUABLE: el contrafactual no "
                "discriminó, no hay contraste que juzgar."
            ),
        )
    if causal < threshold:
        return InvariantViolation(
            invariant_name="factual_counterfactual_coherence",
            severity="hard",
            evidence_value=round(causal, 4),
            threshold=threshold,
            description=f"Causal support {causal:.4f} < {threshold}",
        )
    return None


def _check_lineage_coherent(state: OrganismState, config: Dict[str, float]) -> InvariantViolation | None:
    if not state.identity.lineage_id:
        return InvariantViolation(
            invariant_name="lineage_coherent",
            severity="hard",
            evidence_value=0.0,
            threshold=1.0,
            description="Lineage ID is empty",
        )
    return None


# ── Soft invariant checks ────────────────────────────────────────────────────

def _check_policy_stability(state: OrganismState, config: Dict[str, float]) -> InvariantViolation | None:
    threshold = config.get("max_policy_drift", 0.50)
    if state.policy.accumulated_drift > threshold:
        return InvariantViolation(
            invariant_name="policy_stability",
            severity="soft",
            evidence_value=round(state.policy.accumulated_drift, 4),
            threshold=threshold,
            description=f"Policy drift {state.policy.accumulated_drift:.4f} > {threshold}",
        )
    return None


def _check_continuity_minimum(state: OrganismState, config: Dict[str, float]) -> InvariantViolation | None:
    threshold = config.get("min_continuity", 0.40)
    composite = state.belief.composite_confidence
    if composite < threshold:
        return InvariantViolation(
            invariant_name="continuity_minimum",
            severity="soft",
            evidence_value=round(composite, 4),
            threshold=threshold,
            description=f"Composite confidence {composite:.4f} < {threshold}",
        )
    return None


def _check_recovery_cost(state: OrganismState, config: Dict[str, float]) -> InvariantViolation | None:
    threshold = config.get("max_recovery_debt", 0.70)
    if state.viability.recovery_debt > threshold:
        return InvariantViolation(
            invariant_name="max_recovery_cost",
            severity="soft",
            evidence_value=round(state.viability.recovery_debt, 4),
            threshold=threshold,
            description=f"Recovery debt {state.viability.recovery_debt:.4f} > {threshold}",
        )
    return None


def _check_drift_tolerance(state: OrganismState, config: Dict[str, float]) -> InvariantViolation | None:
    threshold = config.get("max_drift_rate", 0.60)
    drift = state.policy.accumulated_drift
    if drift > threshold:
        return InvariantViolation(
            invariant_name="drift_tolerance",
            severity="soft",
            evidence_value=round(drift, 4),
            threshold=threshold,
            description=f"Drift rate {drift:.4f} > {threshold}",
        )
    return None


# ── Hard and soft invariant registries ───────────────────────────────────────

#: Invariantes hard.  Los NOMBRES son identidad persistida (``IdentityState.active_invariants``,
#: ``constitution_hash()``): no se renombran sin migrar organismos.  Las descripciones sí dicen
#: la verdad sobre qué mide cada uno y de qué TIPO es (P12.5: FACULTAD vs HALLAZGO).
HARD_INVARIANTS: Tuple[ConstitutionalInvariant, ...] = (
    ConstitutionalInvariant(
        "triadic_closure",
        "hard",
        "FACULTAD — cierre del aparato: trace × purity >= threshold. "
        "(Nombre histórico: NO es el cierre triádico σ→f→w→σ del canon A3.)",
    ),
    ConstitutionalInvariant("min_memory_purity", "hard", "FACULTAD — Memory purity >= minimum"),
    ConstitutionalInvariant("min_trace_integrity", "hard", "FACULTAD — Trace integrity >= minimum"),
    ConstitutionalInvariant("baseline_not_degraded", "hard", "FACULTAD — Accumulated degradation < maximum"),
    ConstitutionalInvariant("rollback_available", "hard", "FACULTAD — Rollback must be ready"),
    ConstitutionalInvariant(
        "factual_counterfactual_coherence",
        "hard",
        "HALLAZGO — Causal support >= min_causal_support (piso declarado). "
        "Medir una contradicción NO es violarlo. Hoy inalcanzable con el productor vivo.",
    ),
    ConstitutionalInvariant("lineage_coherent", "hard", "FACULTAD — Lineage ID must be non-empty"),
)

SOFT_INVARIANTS: Tuple[ConstitutionalInvariant, ...] = (
    ConstitutionalInvariant("policy_stability", "soft", "Policy drift within tolerance"),
    ConstitutionalInvariant("continuity_minimum", "soft", "Composite confidence above minimum"),
    ConstitutionalInvariant("max_recovery_cost", "soft", "Recovery debt below maximum"),
    ConstitutionalInvariant("drift_tolerance", "soft", "Drift rate within tolerance"),
)

_HARD_CHECKS = [
    _check_triadic_closure,
    _check_memory_purity,
    _check_trace_integrity,
    _check_baseline_integrity,
    _check_rollback_available,
    _check_coherence,
    _check_lineage_coherent,
]

_SOFT_CHECKS = [
    _check_policy_stability,
    _check_continuity_minimum,
    _check_recovery_cost,
    _check_drift_tolerance,
]


# ── Mutation rules ───────────────────────────────────────────────────────────

MUTABLE_COMPONENTS: FrozenSet[str] = frozenset({
    "transport_parameters",
    "selection_policy",
    "benchmark_policy_experimental",
    "reasoning_activation_policy",
    "memory_scoring_secondary_weights",
    "analogical_lab_parameters",
})

IMMUTABLE_COMPONENTS: FrozenSet[str] = frozenset({
    "baseline_semantics",
    "constitutional_invariants",
    "baseline_fixed",
    "constitutional_purity_minimum",
    "lineage_identity_anchor",
})


# ── Constitution ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OrganismConstitution:
    """Constitución operativa del organismo.

    Define la ley viva del organismo: invariantes, reglas de mutación,
    y configuración de umbrales constitucionales.
    """

    hard_invariants: Tuple[ConstitutionalInvariant, ...] = HARD_INVARIANTS
    soft_invariants: Tuple[ConstitutionalInvariant, ...] = SOFT_INVARIANTS
    mutable_components: FrozenSet[str] = MUTABLE_COMPONENTS
    immutable_components: FrozenSet[str] = IMMUTABLE_COMPONENTS
    config: Dict[str, float] = field(default_factory=lambda: {
        "triadic_closure_threshold": 0.50,
        "min_memory_purity": 0.40,
        "min_trace_integrity": 0.30,
        "max_degradation": 0.80,
        "min_causal_support": 0.20,
        "max_policy_drift": 0.50,
        "min_continuity": 0.40,
        "max_recovery_debt": 0.70,
        "max_drift_rate": 0.60,
    })

    def validate(self, state: OrganismState) -> ConstitutionalValidation:
        """Valida el estado del organismo contra la constitución.

        Un chequeo tiene TRES resultados posibles: violación, abstención (no pudo
        evaluarse por falta de evidencia) o silencio (satisfecho y verificado). Las
        abstenciones NO cuentan como violación ni como aprobación: se declaran.

        Los invariantes hard se dividen en dos TIPOS (P12.5), y la distinción es la ley:

          - **FACULTADES** (¿puedo saber algo, en absoluto?): ``triadic_closure``,
            ``min_trace_integrity``, ``min_memory_purity``, ``rollback_available``,
            ``lineage_coherent``, ``baseline_not_degraded``.  Facultad rota ⇒ cuarentena
            es CORRECTA: un organismo que no puede conocer no puede corregirse.
          - **HALLAZGO** (¿qué dice mi última medición del mundo?):
            ``factual_counterfactual_coherence``.  Un hallazgo que cambia ⇒ **actualizo el
            modelo, no me muero**.  Sólo se juzga contra su propio piso declarado.

        El hallazgo se EXPONE en ``causal_finding`` — y hoy **no tiene consumidor**
        (ver el comentario de ese campo).

        Returns:
            ConstitutionalValidation con violaciones, abstenciones, verdict y hallazgo causal.
        """
        violations: List[InvariantViolation] = []
        abstentions: List[InvariantAbstention] = []

        for check in _HARD_CHECKS + _SOFT_CHECKS:
            outcome = check(state, self.config)
            if isinstance(outcome, InvariantAbstention):
                abstentions.append(outcome)
            elif outcome is not None:
                violations.append(outcome)

        hard_count = sum(1 for v in violations if v.severity == "hard")
        soft_count = sum(1 for v in violations if v.severity == "soft")

        if hard_count > 0:
            verdict = "rollback" if hard_count >= 3 else "quarantine"
        else:
            verdict = "valid"

        # Margen al umbral: la menor distancia de un invariante hard EVALUABLE a su
        # umbral. Los ejes que NO SE MIDIERON no aportan margen (no se puede medir la
        # distancia a un umbral que no se pudo evaluar) — y tampoco lo inflan.
        margin = 1.0
        for v in violations:
            if v.severity == "hard":
                margin = min(margin, v.threshold - v.evidence_value)
        if not violations:
            margins = [
                1.0
                - state.viability.accumulated_degradation
                / max(self.config.get("max_degradation", 0.80), 0.01),
            ]
            for value, key, default in (
                (state.belief.memory_purity_estimate, "min_memory_purity", 0.40),
                (state.belief.trace_integrity_confidence, "min_trace_integrity", 0.30),
                (state.belief.causal_support_confidence, "min_causal_support", 0.20),
            ):
                if value is not None:
                    margins.append(value - self.config.get(key, default))
            margin = min(margins)

        return ConstitutionalValidation(
            is_valid=hard_count == 0,
            verdict=verdict,
            violations=tuple(violations),
            hard_violation_count=hard_count,
            soft_violation_count=soft_count,
            margin_to_threshold=round(margin, 4),
            abstentions=tuple(abstentions),
            # El HALLAZGO causal viaja nombrado, sin consumidor y sin aprobación implícita.
            causal_finding=state.belief.causal_support_confidence,
        )

    def is_mutable(self, component: str) -> bool:
        """Verifica si un componente puede ser mutado."""
        return component in self.mutable_components

    def is_immutable(self, component: str) -> bool:
        """Verifica si un componente es inmutable."""
        return component in self.immutable_components

    def constitution_hash(self) -> str:
        """Hash SHA256 del contenido constitucional."""
        import hashlib
        import json
        blob = json.dumps({
            "hard": [i.name for i in self.hard_invariants],
            "soft": [i.name for i in self.soft_invariants],
            "mutable": sorted(self.mutable_components),
            "immutable": sorted(self.immutable_components),
            "config": {k: v for k, v in sorted(self.config.items())},
        }, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]
