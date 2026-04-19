"""Escenario térmico espacial 5x5 - primer mundo grid estructurado.

EXPERIMENTAL: GridThermalScenario es el primer mundo espacial 5x5.
Status: MVP estable, pero no production-ready.
No usar como default hasta aprobación explícita.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Literal

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
        topology: Optional[
            Literal[
                "uniform",
                "hotspot_center",
                "hotspot_corner",
                "gradient_ns",
                "gradient_ew",
                "checkerboard",
                "quadrants",
            ]
        ] = None,
        topology_params: Optional[Dict[str, Any]] = None,
    ):
        """Inicializa escenario térmico espacial.

        Args:
            initial_temperature: Temperatura inicial base (0.0-1.0).
            alarm_threshold: Umbral de alarma sobre global_temp_mean.
            cooling_effect: Efecto del enfriamiento por paso por celda.
            grid_size: Tamaño del grid (default 5 para 5x5).
            topology: Topología de inicialización espacial (None = uniform).
            topology_params: Parámetros específicos de topología.
        """
        self._alarm_threshold = alarm_threshold
        self._cooling_effect = cooling_effect
        self._grid_size = grid_size
        self._cell_count = grid_size * grid_size

        # Inicializar grid con topología especificada
        cells = self._initialize_topology(
            initial_temperature,
            topology or "uniform",
            topology_params or {},
        )

        # Computar agregados iniciales
        temps = [cell.temperature for cell in cells]
        global_temp_mean = sum(temps) / len(temps)
        global_temp_max = max(temps)
        global_alarm = global_temp_mean >= alarm_threshold

        self._grid = GridState(
            cells=cells,
            global_temp_mean=global_temp_mean,
            global_temp_max=global_temp_max,
            global_alarm=global_alarm,
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

    def _initialize_topology(
        self,
        base_temp: float,
        topology: str,
        params: Dict[str, Any],
    ) -> List[CellState]:
        """Inicializa celdas con topología específica.

        Args:
            base_temp: Temperatura base.
            topology: Tipo de topología.
            params: Parámetros de topología.

        Returns:
            Lista de CellState inicializadas.
        """
        if topology == "uniform":
            return self._topology_uniform(base_temp)
        elif topology == "hotspot_center":
            return self._topology_hotspot_center(base_temp, params)
        elif topology == "hotspot_corner":
            return self._topology_hotspot_corner(base_temp, params)
        elif topology == "gradient_ns":
            return self._topology_gradient_ns(base_temp, params)
        elif topology == "gradient_ew":
            return self._topology_gradient_ew(base_temp, params)
        elif topology == "checkerboard":
            return self._topology_checkerboard(base_temp, params)
        elif topology == "quadrants":
            return self._topology_quadrants(base_temp, params)
        else:
            # Default: uniform
            return self._topology_uniform(base_temp)

    def _topology_uniform(self, temp: float) -> List[CellState]:
        """Topología uniforme: todas las celdas a la misma temperatura."""
        return [
            CellState(row=i, col=j, temperature=temp, cooling_active=False)
            for i in range(self._grid_size)
            for j in range(self._grid_size)
        ]

    def _topology_hotspot_center(
        self, base_temp: float, params: Dict[str, Any]
    ) -> List[CellState]:
        """Hotspot en el centro del grid."""
        hotspot_temp = params.get("hotspot_temp", 0.95)
        hotspot_radius = params.get("hotspot_radius", 1)
        center = self._grid_size // 2

        cells = []
        for i in range(self._grid_size):
            for j in range(self._grid_size):
                dist = abs(i - center) + abs(j - center)  # Manhattan distance
                temp = hotspot_temp if dist <= hotspot_radius else base_temp
                cells.append(CellState(row=i, col=j, temperature=temp, cooling_active=False))
        return cells

    def _topology_hotspot_corner(
        self, base_temp: float, params: Dict[str, Any]
    ) -> List[CellState]:
        """Hotspot en esquina superior izquierda."""
        hotspot_temp = params.get("hotspot_temp", 0.95)
        hotspot_size = params.get("hotspot_size", 2)

        cells = []
        for i in range(self._grid_size):
            for j in range(self._grid_size):
                temp = hotspot_temp if (i < hotspot_size and j < hotspot_size) else base_temp
                cells.append(CellState(row=i, col=j, temperature=temp, cooling_active=False))
        return cells

    def _topology_gradient_ns(
        self, base_temp: float, params: Dict[str, Any]
    ) -> List[CellState]:
        """Gradiente Norte-Sur (caliente arriba, frío abajo)."""
        temp_range = params.get("temp_range", 0.3)
        max_temp = min(1.0, base_temp + temp_range / 2)
        min_temp = max(0.0, base_temp - temp_range / 2)

        cells = []
        for i in range(self._grid_size):
            # Interpolación lineal
            t = i / (self._grid_size - 1)
            temp = max_temp - t * (max_temp - min_temp)
            for j in range(self._grid_size):
                cells.append(CellState(row=i, col=j, temperature=temp, cooling_active=False))
        return cells

    def _topology_gradient_ew(
        self, base_temp: float, params: Dict[str, Any]
    ) -> List[CellState]:
        """Gradiente Este-Oeste (caliente derecha, frío izquierda)."""
        temp_range = params.get("temp_range", 0.3)
        max_temp = min(1.0, base_temp + temp_range / 2)
        min_temp = max(0.0, base_temp - temp_range / 2)

        cells = []
        for i in range(self._grid_size):
            for j in range(self._grid_size):
                t = j / (self._grid_size - 1)
                temp = min_temp + t * (max_temp - min_temp)
                cells.append(CellState(row=i, col=j, temperature=temp, cooling_active=False))
        return cells

    def _topology_checkerboard(
        self, base_temp: float, params: Dict[str, Any]
    ) -> List[CellState]:
        """Patrón de tablero de ajedrez (celdas alternadas calientes/frías)."""
        delta = params.get("delta", 0.15)
        cells = []
        for i in range(self._grid_size):
            for j in range(self._grid_size):
                temp = base_temp + delta if (i + j) % 2 == 0 else base_temp - delta
                temp = max(0.0, min(1.0, temp))
                cells.append(CellState(row=i, col=j, temperature=temp, cooling_active=False))
        return cells

    def _topology_quadrants(
        self, base_temp: float, params: Dict[str, Any]
    ) -> List[CellState]:
        """Cuatro cuadrantes con temperaturas diferentes."""
        temps = params.get("temps", [0.7, 0.85, 0.75, 0.9])  # [NW, NE, SW, SE]
        mid = self._grid_size // 2

        cells = []
        for i in range(self._grid_size):
            for j in range(self._grid_size):
                if i < mid and j < mid:
                    temp = temps[0]  # NW
                elif i < mid and j >= mid:
                    temp = temps[1]  # NE
                elif i >= mid and j < mid:
                    temp = temps[2]  # SW
                else:
                    temp = temps[3]  # SE
                cells.append(CellState(row=i, col=j, temperature=temp, cooling_active=False))
        return cells

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

        # Computar métricas espaciales
        spatial_metrics = self._compute_spatial_metrics()

        return {
            # Agregados globales (compatibilidad 1x1)
            "global_temp_mean": self._grid.global_temp_mean,
            "global_temp_max": self._grid.global_temp_max,
            "cooling_cells_count": self._grid.cooling_cells_count,
            "world_level": world_level_numeric,
            "world_level_numeric": world_level_numeric,
            "world_level_semantic": world_level_semantic,
            # Métricas espaciales (nuevo 5x5)
            **spatial_metrics,
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

    def _detect_hotspots(self) -> Tuple[int, float, List[Tuple[int, int]]]:
        """Detecta hotspots en el grid (celdas significativamente más calientes).

        Un hotspot es una celda cuya temperatura está al menos 0.1 por encima
        de la media global Y >= alarm_threshold. Esto evita falsos positivos
        en campos uniformemente altos.

        Returns:
            (hotspot_count, hotspot_peak, hotspot_locations)
        """
        hotspots = []
        peak = 0.0
        mean_temp = self._grid.global_temp_mean
        hotspot_delta_threshold = 0.1  # Temperatura relativa sobre la media

        for cell in self._grid.cells:
            # Hotspot: significativamente por encima de la media Y en zona de alarma
            is_significantly_hot = cell.temperature >= mean_temp + hotspot_delta_threshold
            is_in_alarm_zone = cell.temperature >= self._alarm_threshold

            if is_significantly_hot and is_in_alarm_zone:
                hotspots.append((cell.row, cell.col))
                peak = max(peak, cell.temperature)

        return len(hotspots), peak, hotspots

    def _compute_thermal_gradient(self) -> Tuple[float, str]:
        """Computa gradiente térmico dominante.

        Returns:
            (gradient_strength, gradient_axis)
            gradient_strength: [0.0, 1.0] magnitud del gradiente
            gradient_axis: "NS", "EW", "NE_SW", "NW_SE", "NONE"
        """
        # Gradientes direccionales
        temps_by_row = [[] for _ in range(self._grid_size)]
        temps_by_col = [[] for _ in range(self._grid_size)]

        for cell in self._grid.cells:
            temps_by_row[cell.row].append(cell.temperature)
            temps_by_col[cell.col].append(cell.temperature)

        # Promedio por fila/columna
        row_means = [sum(temps) / len(temps) for temps in temps_by_row]
        col_means = [sum(temps) / len(temps) for temps in temps_by_col]

        # Gradiente NS: diferencia entre extremos norte-sur
        gradient_ns = abs(row_means[0] - row_means[-1])
        # Gradiente EW: diferencia entre extremos este-oeste
        gradient_ew = abs(col_means[0] - col_means[-1])

        # Gradiente diagonal (simplificado: esquinas)
        corners = [
            self._grid.cells[0].temperature,  # NW
            self._grid.cells[self._grid_size - 1].temperature,  # NE
            self._grid.cells[-self._grid_size].temperature,  # SW
            self._grid.cells[-1].temperature,  # SE
        ]
        gradient_ne_sw = abs(corners[1] - corners[2])
        gradient_nw_se = abs(corners[0] - corners[3])

        # Identificar gradiente dominante
        gradients = {
            "NS": gradient_ns,
            "EW": gradient_ew,
            "NE_SW": gradient_ne_sw,
            "NW_SE": gradient_nw_se,
        }
        max_gradient = max(gradients.values())

        if max_gradient < 0.05:  # Umbral de insignificancia
            return 0.0, "NONE"

        dominant_axis = max(gradients, key=gradients.get)
        return max_gradient, dominant_axis

    def _compute_quadrant_distribution(self) -> Tuple[List[float], float]:
        """Computa distribución por cuadrantes y desequilibrio.

        Returns:
            (quadrant_temps, quadrant_imbalance)
            quadrant_temps: [NW, NE, SW, SE] temperaturas medias
            quadrant_imbalance: [0.0, 1.0] desviación estándar entre cuadrantes
        """
        # Dividir grid en 4 cuadrantes
        mid = self._grid_size // 2
        quadrants = {
            "NW": [],
            "NE": [],
            "SW": [],
            "SE": [],
        }

        for cell in self._grid.cells:
            if cell.row < mid and cell.col < mid:
                quadrants["NW"].append(cell.temperature)
            elif cell.row < mid and cell.col >= mid:
                quadrants["NE"].append(cell.temperature)
            elif cell.row >= mid and cell.col < mid:
                quadrants["SW"].append(cell.temperature)
            else:
                quadrants["SE"].append(cell.temperature)

        # Temperatura media por cuadrante
        quadrant_temps = [
            sum(quadrants["NW"]) / len(quadrants["NW"]) if quadrants["NW"] else 0.0,
            sum(quadrants["NE"]) / len(quadrants["NE"]) if quadrants["NE"] else 0.0,
            sum(quadrants["SW"]) / len(quadrants["SW"]) if quadrants["SW"] else 0.0,
            sum(quadrants["SE"]) / len(quadrants["SE"]) if quadrants["SE"] else 0.0,
        ]

        # Desequilibrio: desviación estándar entre cuadrantes
        mean_quad = sum(quadrant_temps) / 4
        variance = sum((t - mean_quad) ** 2 for t in quadrant_temps) / 4
        imbalance = math.sqrt(variance)

        return quadrant_temps, imbalance

    def _compute_concentration_index(self) -> float:
        """Computa índice de concentración espacial del calor.

        Basado en entropía espacial inversa:
        - 0.0: calor difuso (uniforme)
        - 1.0: calor concentrado (hotspots)

        Returns:
            Índice de concentración [0.0, 1.0].
        """
        temps = [cell.temperature for cell in self._grid.cells]
        mean_temp = sum(temps) / len(temps)

        # Varianza normalizada
        variance = sum((t - mean_temp) ** 2 for t in temps) / len(temps)
        max_variance = mean_temp * (1.0 - mean_temp)  # Máximo para distribución binaria

        if max_variance < 1e-6:
            return 0.0

        # Normalizar varianza a [0, 1]
        concentration = min(1.0, variance / max_variance)
        return concentration

    def _compute_spatial_entropy(self) -> float:
        """Computa entropía espacial de la distribución de temperatura.

        Returns:
            Entropía normalizada [0.0, 1.0].
        """
        temps = [cell.temperature for cell in self._grid.cells]

        # Discretizar temperaturas en bins
        bins = 10
        counts = [0] * bins
        for t in temps:
            bin_idx = min(bins - 1, int(t * bins))
            counts[bin_idx] += 1

        # Calcular entropía
        total = sum(counts)
        entropy = 0.0
        for count in counts:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        # Normalizar por entropía máxima
        max_entropy = math.log2(bins)
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def _compute_spatial_metrics(self) -> Dict[str, Any]:
        """Computa todas las métricas espaciales del grid.

        Returns:
            Diccionario con 13 métricas espaciales.
        """
        temps = [cell.temperature for cell in self._grid.cells]
        mean_temp = self._grid.global_temp_mean

        # 1. Dispersión
        variance = sum((t - mean_temp) ** 2 for t in temps) / len(temps)
        temp_std = math.sqrt(variance)

        # 2-3. Hotspots
        hotspot_count, hotspot_peak, hotspot_locations = self._detect_hotspots()

        # Clasificación de hotspot
        hotspot_central = False
        hotspot_peripheral = False
        if hotspot_count > 0:
            center = self._grid_size // 2
            for row, col in hotspot_locations:
                if abs(row - center) <= 1 and abs(col - center) <= 1:
                    hotspot_central = True
                else:
                    hotspot_peripheral = True

        # 4-5. Gradiente térmico
        gradient_strength, gradient_axis = self._compute_thermal_gradient()

        # 6. Índice de concentración
        heat_concentration_index = self._compute_concentration_index()

        # 7-8. Distribución por cuadrantes
        quadrant_temps, quadrant_imbalance = self._compute_quadrant_distribution()

        # 9. Celdas en zona crítica
        cells_above_warning = sum(1 for t in temps if t >= 0.95)

        # 10. Entropía espacial
        spatial_entropy = self._compute_spatial_entropy()

        # 11. Diferencia max-min (rango térmico)
        thermal_range = self._grid.global_temp_max - min(temps)

        # 12-13. Estadísticas de activación de cooling
        cooling_coverage = self._grid.cooling_cells_count / self._cell_count

        return {
            "temp_std": temp_std,
            "hotspot_count": hotspot_count,
            "hotspot_peak": hotspot_peak,
            "hotspot_central": hotspot_central,
            "hotspot_peripheral": hotspot_peripheral,
            "gradient_strength": gradient_strength,
            "gradient_axis": gradient_axis,
            "heat_concentration_index": heat_concentration_index,
            "quadrant_temps": quadrant_temps,
            "quadrant_imbalance": quadrant_imbalance,
            "cells_above_warning": cells_above_warning,
            "spatial_entropy": spatial_entropy,
            "thermal_range": thermal_range,
            "cooling_coverage": cooling_coverage,
        }

    def _generate_spatial_propositions(self, spatial_metrics: Dict[str, Any]) -> List[str]:
        """Genera proposiciones espaciales desde métricas.

        Args:
            spatial_metrics: Diccionario con métricas espaciales.

        Returns:
            Lista de proposiciones espaciales.
        """
        props = []

        # 1. Proposiciones de hotspot
        hotspot_count = spatial_metrics["hotspot_count"]
        if hotspot_count > 0:
            props.append("HOTSPOT_DETECTED")
            if hotspot_count == 1:
                props.append("SINGLE_HOTSPOT")
            else:
                props.append("MULTI_HOTSPOT")

            # Localización de hotspot
            if spatial_metrics["hotspot_central"]:
                props.append("HOTSPOT_CENTRAL")
            if spatial_metrics["hotspot_peripheral"]:
                props.append("HOTSPOT_PERIPHERAL")

            # Intensidad de hotspot
            if spatial_metrics["hotspot_peak"] >= 0.95:
                props.append("HOTSPOT_CRITICAL")

        # 2. Proposiciones de gradiente térmico
        gradient_axis = spatial_metrics["gradient_axis"]
        gradient_strength = spatial_metrics["gradient_strength"]
        if gradient_axis != "NONE" and gradient_strength > 0.1:
            props.append("THERMAL_GRADIENT")
            props.append(f"THERMAL_GRADIENT_{gradient_axis}")

            if gradient_strength >= 0.3:
                props.append("STRONG_GRADIENT")

        # 3. Proposiciones de concentración de calor
        concentration = spatial_metrics["heat_concentration_index"]
        if concentration >= 0.6:
            props.append("CONCENTRATED_HEAT")
        elif concentration <= 0.2:
            props.append("DIFFUSE_HEAT")

        # 4. Proposiciones de desequilibrio regional
        imbalance = spatial_metrics["quadrant_imbalance"]
        if imbalance >= 0.15:
            props.append("REGIONAL_IMBALANCE")

        # 5. Proposiciones de zona crítica
        cells_above_warning = spatial_metrics["cells_above_warning"]
        if cells_above_warning > 0:
            props.append("CRITICAL_ZONE_PRESENT")
            if cells_above_warning >= 5:
                props.append("CRITICAL_ZONE_EXTENSIVE")

        # 6. Proposiciones de dispersión
        temp_std = spatial_metrics["temp_std"]
        if temp_std >= 0.15:
            props.append("HIGH_TEMPERATURE_VARIANCE")
        elif temp_std <= 0.03:
            props.append("UNIFORM_TEMPERATURE")

        # 7. Proposiciones de rango térmico
        thermal_range = spatial_metrics["thermal_range"]
        if thermal_range >= 0.4:
            props.append("EXTREME_THERMAL_RANGE")

        # 8. Proposiciones de entropía espacial
        entropy = spatial_metrics["spatial_entropy"]
        if entropy >= 0.8:
            props.append("HIGH_SPATIAL_ENTROPY")
        elif entropy <= 0.3:
            props.append("LOW_SPATIAL_ENTROPY")

        # 9. Proposiciones de cobertura de cooling
        coverage = spatial_metrics["cooling_coverage"]
        if coverage > 0:
            props.append("COOLING_ACTIVE")
            if coverage >= 0.8:
                props.append("COOLING_WIDESPREAD")
            elif coverage <= 0.3:
                props.append("COOLING_LOCALIZED")

        return props

    def observe(self) -> ScenarioObservation:
        """Observa el estado actual del grid.

        Returns:
            ScenarioObservation con estado agregado y proposiciones.
            cell_states se incluyen en metadata persistida, no en state principal.
        """
        # Estado principal: agregados globales + métricas espaciales
        state = self._build_aggregate_state()
        state["cell_states"] = self._build_cell_states_list()
        state["world_shape"] = f"{self._grid_size}x{self._grid_size}"
        state["cell_count"] = self._cell_count

        # Proposiciones legacy (compatibilidad 1x1)
        temp_high = self._grid.global_alarm
        propositions = ["TEMP_HIGH"] if temp_high else ["TEMP_NORMAL"]

        # Proposiciones espaciales (nuevo 5x5)
        spatial_metrics = self._compute_spatial_metrics()
        spatial_props = self._generate_spatial_propositions(spatial_metrics)
        propositions.extend(spatial_props)

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
            ScenarioTransition con estado simulado incluyendo métricas espaciales.
        """
        # Clonar grid para simulación
        simulated_grid = self._clone_grid()

        # Computar transición sobre el clon
        self._compute_transition(
            simulated_grid,
            intervention=intervention,
            external_input=external_input,
        )

        # Construir estado agregado CON métricas espaciales
        # (Necesitamos guardar el grid actual temporalmente)
        original_grid = self._grid
        self._grid = simulated_grid

        # Usar _build_aggregate_state para incluir todas las métricas espaciales
        state = self._build_aggregate_state()

        # Restaurar grid original
        self._grid = original_grid

        # Derivar nivel discreto
        world_level_discrete = self._derive_world_level(simulated_grid.global_temp_mean)

        # Generar proposiciones espaciales del estado contrafactual
        spatial_metrics = self._compute_spatial_metrics_from_grid(simulated_grid)
        spatial_props = self._generate_spatial_propositions(spatial_metrics)

        # Proposiciones legacy
        temp_high = simulated_grid.global_alarm
        propositions = ["TEMP_HIGH"] if temp_high else ["TEMP_NORMAL"]
        propositions.extend(spatial_props)

        return ScenarioTransition(
            state=state,
            propositions=propositions,
            alarm=simulated_grid.global_alarm,
            level=world_level_discrete,
        )

    def _compute_spatial_metrics_from_grid(self, grid: GridState) -> Dict[str, Any]:
        """Versión auxiliar de _compute_spatial_metrics que opera sobre un grid arbitrario.

        Args:
            grid: GridState a analizar.

        Returns:
            Diccionario con métricas espaciales.
        """
        original_grid = self._grid
        self._grid = grid
        metrics = self._compute_spatial_metrics()
        self._grid = original_grid
        return metrics

    def get_formula(self, observation: ScenarioObservation) -> str:
        """Genera fórmula LOTF para la observación."""
        return self._config.formula_template

    def select_intervention(self, observation: ScenarioObservation) -> str:
        """Selecciona intervención apropiada considerando estructura espacial.

        Lógica topológicamente sensible:
        - Zona crítica presente -> SIEMPRE activar (riesgo inminente)
        - Alarma + calor concentrado -> URGENTE activar (hotspot peligroso)
        - Alarma + calor difuso -> MODERADO (distribuido, menos urgente)
        - Sin alarma + hotspot detectado -> PREVENTIVO activar
        - Sin alarma + uniforme -> Mantener idle

        Args:
            observation: Observación actual con proposiciones espaciales.

        Returns:
            Nombre de la intervención a aplicar.
        """
        props = observation.propositions

        # Regla 1: Zona crítica -> SIEMPRE activar (células >= 0.95)
        if "CRITICAL_ZONE_PRESENT" in props:
            return "activate_cooling"

        # Regla 2: Alarma + calor concentrado -> URGENTE
        if observation.alarm and "CONCENTRATED_HEAT" in props:
            return "activate_cooling"

        # Regla 3: Alarma + hotspot -> URGENTE (riesgo localizado)
        if observation.alarm and "HOTSPOT_DETECTED" in props:
            return "activate_cooling"

        # Regla 4: Alarma + calor difuso -> MODERADO
        # (Decisión diferenciada: difuso es menos urgente que concentrado)
        if observation.alarm and "DIFFUSE_HEAT" in props:
            # Con calor difuso, evaluar si gradiente fuerte sugiere tendencia
            if "STRONG_GRADIENT" in props:
                return "activate_cooling"
            # Sin gradiente fuerte, tolerar más (no isomórfico con concentrado)
            return "deactivate_cooling"

        # Regla 5: Sin alarma pero hotspot detectado -> PREVENTIVO
        if "HOTSPOT_DETECTED" in props:
            return "activate_cooling"

        # Regla 6: Default - sin alarma, sin hotspot -> idle
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
