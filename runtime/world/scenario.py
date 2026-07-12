"""Interfaz base para escenarios cognitivos mínimos."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, TYPE_CHECKING

from runtime.smg.smg_min import (  # SSOT del vocabulario de relaciones semióticas
    CONTRADICTION,
    NO_DISCRIMINATING_EVIDENCE,
    SUPPORT,
)

if TYPE_CHECKING:
    from .compatibility import ScenarioStructuralProfile
    from .causal_signature import ScenarioCausalSignature


@dataclass
class ScenarioObservation:
    """Observación del estado del escenario.

    Attributes:
        state: Diccionario con estado observable del mundo.
        propositions: Lista de proposiciones inferibles del estado.
        alarm: Indicador de condición de alarma/umbral.
        level: Nivel discreto del mundo (1=SAFE, 2=ELEVATED, 3=WARNING, 4=CRITICAL).
    """

    state: Dict[str, Any]
    propositions: List[str]
    alarm: bool
    level: int = 1


@dataclass
class ScenarioTransition:
    """Resultado de transición del escenario.

    Attributes:
        state: Nuevo estado después de la transición.
        propositions: Proposiciones actualizadas.
        alarm: Estado de alarma actualizado.
        level: Nivel discreto del mundo (1=SAFE, 2=ELEVATED, 3=WARNING, 4=CRITICAL).
    """

    state: Dict[str, Any]
    propositions: List[str]
    alarm: bool
    level: int = 1


@dataclass
class ScenarioConfig:
    """Configuración de un escenario cognitivo.

    Attributes:
        name: Nombre único del escenario.
        description: Descripción del escenario.
        main_variable: Variable principal del escenario.
        alarm_threshold: Umbral para activar alarma.
        interventions: Lista de intervenciones válidas.
        formula_template: Plantilla de fórmula LOTF.
        type_context: Contexto de tipos para checker LOTF.
    """

    name: str
    description: str
    main_variable: str
    alarm_threshold: float
    interventions: List[str]
    formula_template: str
    type_context: Dict[str, str]


class CognitiveScenario(ABC):
    """Interfaz base para escenarios cognitivos mínimos.

    Un escenario cognitivo define:
    - Estado observable del mundo
    - Transiciones factual y contrafactual
    - Proposiciones y fórmulas LOTF
    - Mapeo a signos SMG
    """

    @property
    @abstractmethod
    def config(self) -> ScenarioConfig:
        """Retorna configuración del escenario."""
        ...

    @property
    @abstractmethod
    def structural_profile(self) -> ScenarioStructuralProfile:
        """Retorna perfil estructural para evaluación de compatibilidad."""
        ...

    @property
    @abstractmethod
    def causal_signature(self) -> ScenarioCausalSignature:
        """Retorna firma causal completa para morfismos dirigidos."""
        ...

    @abstractmethod
    def observe(self) -> ScenarioObservation:
        """Observa el estado actual del escenario.

        Returns:
            ScenarioObservation con estado, proposiciones y alarma.
        """
        ...

    @abstractmethod
    def factual_transition(
        self,
        *,
        intervention: str,
        external_input: float,
    ) -> ScenarioTransition:
        """Ejecuta transición factual con intervención.

        Args:
            intervention: Intervención a aplicar.
            external_input: Entrada externa (perturbación).

        Returns:
            ScenarioTransition con nuevo estado.
        """
        ...

    @abstractmethod
    def simulate_counterfactual(
        self,
        *,
        intervention: str,
        external_input: float,
    ) -> ScenarioTransition:
        """Simula transición contrafactual sin mutar estado.

        Args:
            intervention: Intervención hipotética.
            external_input: Entrada externa simulada.

        Returns:
            ScenarioTransition con estado simulado.
        """
        ...

    @abstractmethod
    def get_formula(self, observation: ScenarioObservation) -> str:
        """Genera fórmula LOTF para la observación.

        Args:
            observation: Observación actual.

        Returns:
            String con fórmula LOTF.
        """
        ...

    @abstractmethod
    def select_intervention(self, observation: ScenarioObservation) -> str:
        """Selecciona intervención apropiada para la observación.

        Args:
            observation: Observación actual.

        Returns:
            Nombre de la intervención a aplicar.
        """
        ...

    @abstractmethod
    def get_main_proposition(self, observation: ScenarioObservation) -> str:
        """Obtiene proposición principal de la observación.

        Args:
            observation: Observación actual.

        Returns:
            Proposición principal (e.g., 'TEMP_HIGH').
        """
        ...

    @abstractmethod
    def get_intervention_proposition(self, intervention: str) -> str:
        """Obtiene proposición correspondiente a la intervención.

        Args:
            intervention: Intervención aplicada.

        Returns:
            Proposición de intervención (e.g., 'ACTIVATE_COOLING').
        """
        ...

    def evaluate_relation_kind(
        self,
        *,
        factual: ScenarioTransition,
        counterfactual: ScenarioTransition,
    ) -> str:
        """Evalúa el soporte causal del contraste factual/contrafactual.

        CRITERIO ÚNICO PARA TODOS LOS ESCENARIOS (no se sobreescribe): se juzga
        contra el **objetivo regulatorio** que el escenario declara —mantenerse en
        la región segura, es decir FUERA de alarma— y no contra un monótono.

        Por qué NO el monótono. La implementación anterior comparaba el valor de la
        variable principal (`factual <= counterfactual` ⇒ support) asumiendo "más
        frío siempre es mejor". Pero la firma causal declara a la vez
        `optimization_direction` (monótono) y `alarm_semantics` + `alarm_threshold`
        (umbral), y **son objetivos distintos**. La política real del organismo es
        la segunda: sólo actúa bajo alarma. Juzgar sus actos contra la primera lo
        hacía perder en calma —no-actuar siempre queda "peor" que actuar— y
        acusarse de `contradiction` justo cuando cumplía su política.

        `ScenarioTransition.alarm` es el juicio que CADA escenario ya emite sobre su
        propia región segura, con su propio umbral y su propia semántica
        (`threshold_above` / `threshold_below`). Compararlo es leer el objetivo
        declarado, no inventar uno nuevo: por eso el criterio vale igual para
        térmico, recursos, grid y carga diferida, y por eso ningún escenario
        necesita override.

        Las TRES salidas:

        - ``support``: el factual mantuvo al organismo seguro donde el contrafactual
          habría roto el objetivo. **Evidencia causal ganada**: la acción elegida
          marcó la diferencia.
        - ``contradiction``: el factual rompió el objetivo donde el contrafactual lo
          habría mantenido seguro. **Evidencia real de que el modelo causal falla**.
        - ``no_discriminating_evidence``: ambas acciones dejan al organismo del mismo
          lado del objetivo (ambas seguras, o ambas rotas). El contrafactual **no
          enseña nada**: no hay soporte causal que medir. NO es `support` (falsa
          salud) ni `contradiction` (falso pánico); es un NO MEDIDO que se **declara**
          —mismo idioma que `checks_applied`, `unmeasured_vitals`, `unverified_fields`
          y `unmeasured_fields` en el resto del organismo.

        El tercer estado también absorbe, sin fingir, el caso en que el contraste no
        existe (contrafactual == factual porque el escenario no admite alterna): sin
        contraste no puede haber evidencia discriminante.

        Args:
            factual: Transición factual.
            counterfactual: Transición contrafactual.

        Returns:
            ``'support'``, ``'contradiction'`` o ``'no_discriminating_evidence'``.
        """
        factual_safe = not bool(factual.alarm)
        counterfactual_safe = not bool(counterfactual.alarm)

        if factual_safe == counterfactual_safe:
            # El contrafactual no discrimina: ambos caminos caen del mismo lado del
            # objetivo. No hay nada que aprender de este acto. Se declara.
            return NO_DISCRIMINATING_EVIDENCE
        if factual_safe:
            return SUPPORT
        return CONTRADICTION

    def to_observation_dict(self, observation: ScenarioObservation) -> Dict[str, Any]:
        """Convierte observación a diccionario para persistencia.

        Args:
            observation: Observación del escenario.

        Returns:
            Diccionario con estado y metadata.
        """
        return {
            **observation.state,
            "alarm": observation.alarm,
            "propositions": observation.propositions,
            "scenario": self.config.name,
        }

    def to_transition_dict(self, transition: ScenarioTransition) -> Dict[str, Any]:
        """Convierte transición a diccionario para persistencia.

        Args:
            transition: Transición del escenario.

        Returns:
            Diccionario con estado y metadata.
        """
        return {
            **transition.state,
            "alarm": transition.alarm,
        }
