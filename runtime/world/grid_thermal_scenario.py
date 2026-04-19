"""Escenario térmico espacial 5x5 - primer mundo grid estructurado.

EXPERIMENTAL: GridThermalScenario es el primer mundo espacial 5x5.
Status: MVP estable, pero no production-ready.
No usar como default hasta aprobación explícita.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import List

from .compatibility import ScenarioStructuralProfile
from .causal_signature import (
    CausalEdge,
    InterventionEffect,
    ScenarioCausalSignature,
)
from .scenario import (
    CognitiveScenario,
    ScenarioConfig,
    ScenarioObservation,
    ScenarioTransition,
)


@dataclass
class CellState:
    """Estado de una celda individual en el grid 5x5."""

    row: int
    col: int
    temperature: float  # 0.0 - 1.0
    cooling_active: bool


@dataclass
class GridState:
    """Estado completo del grid 5x5."""

    cells: List[CellState]  # 25 elementos
    global_temp_mean: float  # agregado global
    global_temp_max: float  # agregado global
    global_alarm: bool  # derivado del agregado
    cooling_cells_count: int  # agregado estructural


class GridThermalScenario(CognitiveScenario):
    """Escenario homeostático de control de temperatura sobre grid 5x5.

    Este escenario modela un sistema espacial de control de temperatura con:
    - Grid: 5x5 (25 celdas)
    - Variable principal agregada: global_temp_mean (0.0 - 1.0)
    - Variable local por celda: temperature (0.0 - 1.0)
    - Intervenciones globales: activate_cooling, deactivate_cooling
    - Umbral de alarma: configurable (default 0.85) sobre agregado global
    - Dinámica: enfriamiento reduce temperatura por celda, calor externo distribuido

    La intervención es global (aplica a todas las celdas).
    La observación incluye tanto estado local (25 celdas) como agregados globales.
    """

    def __init__(
        self,
        *,
        initial_temperature: float = 0.82,
        alarm_threshold: float = 0.85,
        cooling_effect: float = 0.07,
        grid_size: int = 5,
    ):
        """Inicializa escenario térmico espacial.

        Args:
            initial_temperature: Temperatura inicial para todas las celdas (0.0-1.0).
            alarm_threshold: Umbral de alarma sobre global_temp_mean.
            cooling_effect: Efecto del enfriamiento por paso por celda.
            grid_size: Tamaño del grid (default 5 para 5x5).
        """
        self._alarm_threshold = alarm_threshold
        self._cooling_effect = cooling_effect
        self._grid_size = grid_size
        self._cell_count = grid_size * grid_size

        # Inicializar grid con todas las celdas al mismo estado
        cells = [
            CellState(
                row=i,
                col=j,
                temperature=initial_temperature,
                cooling_active=False,
            )
            for i in range(grid_size)
            for j in range(grid_size)
        ]

        self._grid = GridState(
            cells=cells,
            global_temp_mean=initial_temperature,
            global_temp_max=initial_temperature,
            global_alarm=initial_temperature >= alarm_threshold,
            cooling_cells_count=0,
        )

        self._config = ScenarioConfig(
            name="grid_thermal_5x5",
            description=f"Control de temperatura homeostático sobre grid {grid_size}x{grid_size}",
            main_variable="global_temp_mean",
            alarm_threshold=alarm_threshold,
            interventions=["activate_cooling", "deactivate_cooling"],
            formula_template="TEMP_HIGH -> ACTIVATE_COOLING",
            type_context={"TEMP_HIGH": "bool", "ACTIVATE_COOLING": "bool"},
        )

    @property
    def config(self) -> ScenarioConfig:
        return self._config

    @property
    def structural_profile(self) -> ScenarioStructuralProfile:
        """Perfil estructural para evaluación de compatibilidad."""
        cfg = self._config
        config_blob = json.dumps(
            {
                "name": cfg.name,
                "main_variable": cfg.main_variable,
                "alarm_threshold": cfg.alarm_threshold,
                "interventions": cfg.interventions,
                "formula_template": cfg.formula_template,
                "type_context": cfg.type_context,
                "world_shape": f"{self._grid_size}x{self._grid_size}",
                "cell_count": self._cell_count,
            },
            sort_keys=True,
        )
        config_hash = hashlib.sha256(config_blob.encode()).hexdigest()[:12]
        return ScenarioStructuralProfile(
            scenario_name=cfg.name,
            scenario_version="1.0",
            scenario_config_hash=config_hash,
            control_topology="threshold_single_loop_spatial",
            optimization_direction="minimize",
            intervention_semantics=tuple(cfg.interventions),
            counterfactual_policy="opposite_intervention",
            relation_polarity="lower_is_better",
            main_variable=cfg.main_variable,
        )

    @property
    def causal_signature(self) -> ScenarioCausalSignature:
        """Firma causal completa para morfismos dirigidos."""
        cfg = self._config
        return ScenarioCausalSignature(
            scenario_name=cfg.name,
            scenario_version="1.0",
            observable_variables=frozenset({"global_temp_mean", "global_temp_max", "cooling_cells_count"}),
            control_variables=frozenset({"cooling_active"}),
            main_variable="global_temp_mean",
            optimization_direction="minimize",
            causal_polarity="lower_is_better",
            alarm_semantics="threshold_above",
            intervention_effects=(
                InterventionEffect(
                    intervention_name="activate_cooling",
                    target_variable="global_temp_mean",
                    expected_direction="-",
                    expected_magnitude=self._cooling_effect,
                    semantic_role="corrective",
                ),
                InterventionEffect(
                    intervention_name="deactivate_cooling",
                    target_variable="global_temp_mean",
                    expected_direction="+",
                    expected_magnitude=0.0,
                    semantic_role="neutral",
                ),
            ),
            counterfactual_policy="opposite_intervention",
            counterfactual_variable="global_temp_mean",
            causal_edges=(
                CausalEdge(source="external_heat", target="global_temp_mean", polarity="+"),
                CausalEdge(source="cooling_active", target="global_temp_mean", polarity="-"),
                CausalEdge(source="global_temp_mean", target="alarm", polarity="+"),
            ),
            proposition_vocabulary=frozenset({
                "TEMP_HIGH", "TEMP_NORMAL", "COOLING_ACTIVE", "ACTIVATE_COOLING", "KEEP_IDLE",
            }),
        )

    @property
    def alarm_threshold(self) -> float:
        """Umbral de alarma para compatibilidad con código existente."""
        return self._alarm_threshold

    def _update_aggregates(self) -> None:
        """Recomputa agregados globales desde estado de celdas."""
        temps = [cell.temperature for cell in self._grid.cells]
        self._grid.global_temp_mean = sum(temps) / len(temps)
        self._grid.global_temp_max = max(temps)
        self._grid.global_alarm = self._grid.global_temp_mean >= self._alarm_threshold
        self._grid.cooling_cells_count = sum(1 for cell in self._grid.cells if cell.cooling_active)

    def _build_aggregate_state(self) -> dict:
        """Construye diccionario de estado agregado para observación."""
        world_level_numeric = self._derive_world_level_numeric()
        world_level_discrete = self._derive_world_level(world_level_numeric)
        world_level_semantic = self._level_to_semantic(world_level_discrete)

        return {
            "global_temp_mean": self._grid.global_temp_mean,
            "global_temp_max": self._grid.global_temp_max,
            "cooling_cells_count": self._grid.cooling_cells_count,
            "world_level": world_level_numeric,  # Compatibilidad 1x1 (numérico)
            "world_level_numeric": world_level_numeric,  # Explícito
            "world_level_semantic": world_level_semantic,  # Categoría semántica
        }

    def _build_cell_states_list(self) -> List[dict]:
        """Construye lista de estados de celdas para metadata."""
        return [
            {
                "row": cell.row,
                "col": cell.col,
                "temperature": cell.temperature,
                "cooling_active": cell.cooling_active,
            }
            for cell in self._grid.cells
        ]

    def _derive_world_level(self, global_temp_mean: float) -> int:
        """Deriva nivel discreto del mundo desde temperatura media global.

        Umbrales basados en alarm_threshold y severidad de crisis:
        - SAFE (1): 0.00 <= temp < 0.60
        - ELEVATED (2): 0.60 <= temp < alarm_threshold (0.85)
        - WARNING (3): alarm_threshold <= temp < 0.95
        - CRITICAL (4): 0.95 <= temp <= 1.00

        Args:
            global_temp_mean: Temperatura media global [0.0, 1.0].

        Returns:
            Nivel discreto 1-4.
        """
        if global_temp_mean >= 0.95:
            return 4  # CRITICAL
        elif global_temp_mean >= self._alarm_threshold:
            return 3  # WARNING
        elif global_temp_mean >= 0.60:
            return 2  # ELEVATED
        else:
            return 1  # SAFE

    def _level_to_semantic(self, level: int) -> str:
        """Convierte nivel discreto a etiqueta semántica.

        Args:
            level: Nivel discreto 1-4.

        Returns:
            Etiqueta semántica.
        """
        mapping = {
            1: "SAFE",
            2: "ELEVATED",
            3: "WARNING",
            4: "CRITICAL",
        }
        return mapping.get(level, "UNKNOWN")

    def _derive_world_level_numeric(self) -> float:
        """Deriva nivel numérico (alias de global_temp_mean para compatibilidad 1x1).

        Returns:
            Temperatura media global [0.0, 1.0].
        """
        return self._grid.global_temp_mean

    def observe(self) -> ScenarioObservation:
        """Observa el estado actual del grid.

        Returns:
            ScenarioObservation con estado agregado y proposiciones.
            cell_states se incluyen en metadata persistida, no en state principal.
        """
        temp_high = self._grid.global_alarm
        propositions = ["TEMP_HIGH"] if temp_high else ["TEMP_NORMAL"]
        if self._grid.cooling_cells_count > 0:
            propositions.append("COOLING_ACTIVE")

        # Estado principal: agregados globales
        state = self._build_aggregate_state()
        # Añadir cell_states para trazabilidad
        state["cell_states"] = self._build_cell_states_list()
        state["world_shape"] = f"{self._grid_size}x{self._grid_size}"
        state["cell_count"] = self._cell_count

        # Derivar nivel discreto para contrato formal
        level = self._derive_world_level(self._grid.global_temp_mean)

        return ScenarioObservation(
            state=state,
            propositions=propositions,
            alarm=self._grid.global_alarm,
            level=level,
        )

    def _clone_grid(self) -> GridState:
        """Clona el grid completo para simulación contrafactual."""
        cloned_cells = [
            CellState(
                row=cell.row,
                col=cell.col,
                temperature=cell.temperature,
                cooling_active=cell.cooling_active,
            )
            for cell in self._grid.cells
        ]
        return GridState(
            cells=cloned_cells,
            global_temp_mean=self._grid.global_temp_mean,
            global_temp_max=self._grid.global_temp_max,
            global_alarm=self._grid.global_alarm,
            cooling_cells_count=self._grid.cooling_cells_count,
        )

    def _compute_transition(
        self,
        grid: GridState,
        *,
        intervention: str,
        external_input: float,
    ) -> GridState:
        """Computa transición de estado sobre un grid.

        Args:
            grid: Grid a transicionar (puede ser self._grid o clon).
            intervention: Intervención a aplicar globalmente.
            external_input: Perturbación externa (calor) a distribuir.

        Returns:
            Grid actualizado (mismo objeto mutado).
        """
        # 1. Aplicar intervención global a todas las celdas
        if intervention == "activate_cooling":
            for cell in grid.cells:
                cell.cooling_active = True
        elif intervention == "deactivate_cooling":
            for cell in grid.cells:
                cell.cooling_active = False

        # 2. Aplicar dinámica térmica por celda
        heat_delta_per_cell = external_input / self._cell_count  # Distribución uniforme
        for cell in grid.cells:
            cooling_delta = self._cooling_effect if cell.cooling_active else 0.0
            cell.temperature = max(0.0, min(1.0, cell.temperature + heat_delta_per_cell - cooling_delta))

        # 3. Recomputar agregados globales
        temps = [cell.temperature for cell in grid.cells]
        grid.global_temp_mean = sum(temps) / len(temps)
        grid.global_temp_max = max(temps)
        grid.global_alarm = grid.global_temp_mean >= self._alarm_threshold
        grid.cooling_cells_count = sum(1 for cell in grid.cells if cell.cooling_active)

        return grid

    def factual_transition(
        self,
        *,
        intervention: str,
        external_input: float,
    ) -> ScenarioTransition:
        """Ejecuta transición factual con intervención global.

        Args:
            intervention: Intervención a aplicar (activate_cooling o deactivate_cooling).
            external_input: Entrada externa (calor) distribuida uniformemente.

        Returns:
            ScenarioTransition con nuevo estado agregado.
        """
        self._compute_transition(
            self._grid,
            intervention=intervention,
            external_input=external_input,
        )

        state = self._build_aggregate_state()
        state["cell_states"] = self._build_cell_states_list()
        state["world_shape"] = f"{self._grid_size}x{self._grid_size}"
        state["cell_count"] = self._cell_count

        # Derivar nivel discreto para contrato formal
        level = self._derive_world_level(self._grid.global_temp_mean)

        return ScenarioTransition(
            state=state,
            propositions=self.observe().propositions,
            alarm=self._grid.global_alarm,
            level=level,
        )

    def simulate_counterfactual(
        self,
        *,
        intervention: str,
        external_input: float,
    ) -> ScenarioTransition:
        """Simula transición contrafactual sin mutar estado real.

        Args:
            intervention: Intervención hipotética.
            external_input: Entrada externa simulada.

        Returns:
            ScenarioTransition con estado simulado.
        """
        # Clonar grid para simulación
        simulated_grid = self._clone_grid()

        # Computar transición sobre el clon
        self._compute_transition(
            simulated_grid,
            intervention=intervention,
            external_input=external_input,
        )

        # Construir estado agregado desde el clon
        temps = [cell.temperature for cell in simulated_grid.cells]
        world_level_numeric = simulated_grid.global_temp_mean
        world_level_discrete = self._derive_world_level(world_level_numeric)
        world_level_semantic = self._level_to_semantic(world_level_discrete)

        state = {
            "global_temp_mean": simulated_grid.global_temp_mean,
            "global_temp_max": simulated_grid.global_temp_max,
            "cooling_cells_count": simulated_grid.cooling_cells_count,
            "world_level": world_level_numeric,
            "world_level_numeric": world_level_numeric,
            "world_level_semantic": world_level_semantic,
        }

        propositions = [
            "TEMP_HIGH" if simulated_grid.global_alarm else "TEMP_NORMAL"
        ]

        return ScenarioTransition(
            state=state,
            propositions=propositions,
            alarm=simulated_grid.global_alarm,
            level=world_level_discrete,
        )

    def get_formula(self, observation: ScenarioObservation) -> str:
        """Genera fórmula LOTF para la observación."""
        return self._config.formula_template

    def select_intervention(self, observation: ScenarioObservation) -> str:
        """Selecciona intervención apropiada basada en alarma global."""
        if observation.alarm:
            return "activate_cooling"
        return "deactivate_cooling"

    def get_main_proposition(self, observation: ScenarioObservation) -> str:
        """Obtiene proposición principal de la observación."""
        if observation.alarm:
            return "TEMP_HIGH"
        return "TEMP_NORMAL"

    def get_intervention_proposition(self, intervention: str) -> str:
        """Obtiene proposición correspondiente a la intervención."""
        if intervention == "activate_cooling":
            return "ACTIVATE_COOLING"
        return "KEEP_IDLE"


# Factory function para compatibilidad
def create_grid_thermal_scenario(
    *,
    initial_temperature: float = 0.82,
    alarm_threshold: float = 0.85,
    grid_size: int = 5,
) -> GridThermalScenario:
    """Crea escenario térmico grid con configuración por defecto."""
    return GridThermalScenario(
        initial_temperature=initial_temperature,
        alarm_threshold=alarm_threshold,
        grid_size=grid_size,
    )
