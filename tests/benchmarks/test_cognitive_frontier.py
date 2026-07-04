"""Test de Frontera Cognitiva: Caracterización del gradiente 1x1 → 5x5.

Fase 0: Test de No-Isomorfismo (Gate Crítico)
==============================================

Este test determina si el GridThermalScenario 5x5 usa activamente la información
espacial para cognición, o si es cognitivamente isomorfo al 1x1 (solo serializa
la estructura espacial como overhead sin usarla).

Criterio de éxito:
- spatial_information_usage > 0.2
- Al menos 5/10 pares de estados muestran diferencias significativas

Si falla: El 5x5 NO aporta cognición espacial → PARAR y rediseñar.
Si pasa: El 5x5 usa información espacial → CONTINUAR con Fase 1.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import json

from runtime.storage import StorageConfig, StorageFactory
from runtime.world.scenario_runner import ScenarioEpisodeRunner
from runtime.world.grid_thermal_scenario import GridThermalScenario


def _storage(tmp_path: Path):
    """Crea storage para tests de frontera cognitiva."""
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "cognitive_frontier.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


class SpatialStateGenerator:
    """Generador de estados espaciales con topologías diversas."""

    @staticmethod
    def uniform(target_mean: float, grid_size: int = 5) -> List[List[float]]:
        """Genera grid uniforme con temperatura constante.

        Args:
            target_mean: Temperatura objetivo (será exacta).
            grid_size: Tamaño del grid.

        Returns:
            Grid 2D con temperatura uniforme.
        """
        return [[target_mean for _ in range(grid_size)] for _ in range(grid_size)]

    @staticmethod
    def hotspot_center(target_mean: float, grid_size: int = 5, hotspot_delta: float = 0.20) -> List[List[float]]:
        """Genera grid con hotspot central y periferia más fría.

        El agregado global mantiene target_mean, pero con alta varianza espacial.

        Args:
            target_mean: Temperatura media objetivo.
            grid_size: Tamaño del grid.
            hotspot_delta: Diferencia de temperatura del hotspot respecto a la media.

        Returns:
            Grid 2D con hotspot central.
        """
        grid = []
        center = grid_size // 2
        total_cells = grid_size * grid_size

        # Calcular cuántas celdas son hotspot (centro 3x3 en 5x5 = 9 celdas)
        hotspot_cells = min(9, total_cells // 3)
        peripheral_cells = total_cells - hotspot_cells

        # Resolver para T_hot y T_cold tal que la media sea target_mean
        # hotspot_cells * T_hot + peripheral_cells * T_cold = total_cells * target_mean
        # T_hot = target_mean + hotspot_delta
        # Entonces: T_cold = (total_cells * target_mean - hotspot_cells * T_hot) / peripheral_cells

        T_hot = min(1.0, target_mean + hotspot_delta)
        T_cold = (total_cells * target_mean - hotspot_cells * T_hot) / peripheral_cells
        T_cold = max(0.0, T_cold)

        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                # Centro 3x3
                if abs(i - center) <= 1 and abs(j - center) <= 1:
                    row.append(T_hot)
                else:
                    row.append(T_cold)
            grid.append(row)

        return grid

    @staticmethod
    def gradient_ns(target_mean: float, grid_size: int = 5) -> List[List[float]]:
        """Genera gradiente Norte (frío) a Sur (caliente).

        Args:
            target_mean: Temperatura media objetivo.
            grid_size: Tamaño del grid.

        Returns:
            Grid 2D con gradiente N-S.
        """
        grid = []
        # Gradiente lineal: T(row) = T_min + (T_max - T_min) * (row / (grid_size - 1))
        # Media: (T_min + T_max) / 2 = target_mean
        # Usamos rango [target_mean - 0.15, target_mean + 0.15]
        T_min = max(0.0, target_mean - 0.15)
        T_max = min(1.0, target_mean + 0.15)

        for i in range(grid_size):
            temp = T_min + (T_max - T_min) * (i / (grid_size - 1))
            row = [temp for _ in range(grid_size)]
            grid.append(row)

        return grid

    @staticmethod
    def gradient_ew(target_mean: float, grid_size: int = 5) -> List[List[float]]:
        """Genera gradiente Este (frío) a Oeste (caliente).

        Args:
            target_mean: Temperatura media objetivo.
            grid_size: Tamaño del grid.

        Returns:
            Grid 2D con gradiente E-O.
        """
        grid = []
        T_min = max(0.0, target_mean - 0.15)
        T_max = min(1.0, target_mean + 0.15)

        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                temp = T_min + (T_max - T_min) * (j / (grid_size - 1))
                row.append(temp)
            grid.append(row)

        return grid

    @staticmethod
    def checkerboard(target_mean: float, grid_size: int = 5, delta: float = 0.10) -> List[List[float]]:
        """Genera patrón de tablero de ajedrez.

        Args:
            target_mean: Temperatura media objetivo.
            grid_size: Tamaño del grid.
            delta: Diferencia entre celdas calientes y frías.

        Returns:
            Grid 2D con patrón checkerboard.
        """
        grid = []
        T_high = min(1.0, target_mean + delta / 2)
        T_low = max(0.0, target_mean - delta / 2)

        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                if (i + j) % 2 == 0:
                    row.append(T_high)
                else:
                    row.append(T_low)
            grid.append(row)

        return grid

    @staticmethod
    def quadrants_alternating(target_mean: float, grid_size: int = 5) -> List[List[float]]:
        """Genera cuadrantes alternos calientes/fríos.

        Args:
            target_mean: Temperatura media objetivo.
            grid_size: Tamaño del grid.

        Returns:
            Grid 2D con cuadrantes alternos.
        """
        grid = []
        mid = grid_size // 2
        T_high = min(1.0, target_mean + 0.12)
        T_low = max(0.0, target_mean - 0.12)

        for i in range(grid_size):
            row = []
            for j in range(grid_size):
                # NE y SO calientes, NO y SE fríos
                if (i < mid and j >= mid) or (i >= mid and j < mid):
                    row.append(T_high)
                else:
                    row.append(T_low)
            grid.append(row)

        return grid

    @classmethod
    def create_state_pair(
        cls,
        target_mean: float,
        topology_a: str,
        topology_b: str,
        grid_size: int = 5,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Crea par de estados con mismo agregado pero distinta topología.

        Args:
            target_mean: Temperatura media objetivo para ambos.
            topology_a: Tipo de topología A ('uniform', 'hotspot', 'gradient_ns', etc).
            topology_b: Tipo de topología B.
            grid_size: Tamaño del grid.

        Returns:
            Tupla (grid_a, grid_b) con misma media global pero distinta distribución.
        """
        generators = {
            'uniform': cls.uniform,
            'hotspot': cls.hotspot_center,
            'gradient_ns': cls.gradient_ns,
            'gradient_ew': cls.gradient_ew,
            'checkerboard': cls.checkerboard,
            'quadrants': cls.quadrants_alternating,
        }

        if topology_a not in generators or topology_b not in generators:
            raise ValueError(f"Topología no reconocida: {topology_a} o {topology_b}")

        grid_a = generators[topology_a](target_mean, grid_size)
        grid_b = generators[topology_b](target_mean, grid_size)

        return grid_a, grid_b


def apply_spatial_state_to_scenario(scenario: GridThermalScenario, spatial_grid: List[List[float]]) -> None:
    """Aplica un estado espacial dado a un escenario 5x5.

    Args:
        scenario: Escenario GridThermalScenario a modificar.
        spatial_grid: Grid 2D con temperaturas por celda.
    """
    grid_size = len(spatial_grid)
    for i in range(grid_size):
        for j in range(grid_size):
            cell_idx = i * grid_size + j
            scenario._grid.cells[cell_idx].temperature = spatial_grid[i][j]

    # Recomputar agregados
    scenario._update_aggregates()


def compute_spatial_information_usage(
    pairs_results: List[Dict[str, Any]]
) -> float:
    """Calcula la métrica de uso de información espacial.

    spatial_information_usage = proporción de pares donde las decisiones difieren
    a pesar de tener el mismo agregado global.

    Args:
        pairs_results: Lista de resultados de pares de episodios.

    Returns:
        Valor entre 0.0 y 1.0.
        - 0.0: Isomorfismo completo (5x5 ignora estructura espacial)
        - > 0.2: Umbral mínimo de uso espacial
        - > 0.3: Uso espacial moderado
        - > 0.6: Uso espacial fuerte
    """
    if not pairs_results:
        return 0.0

    different_count = 0

    for pair in pairs_results:
        # Contar par como "diferente" si AL MENOS UNO de estos difiere
        if pair.get('propositions_differ', False):
            different_count += 1
        elif pair.get('intervention_differs', False):
            different_count += 1
        elif pair.get('level_differs', False):
            different_count += 1

    return different_count / len(pairs_results)


class TestCognitiveFrontierPhase0:
    """Fase 0: Test de No-Isomorfismo (Gate Crítico)."""

    def test_spatial_state_generator_produces_valid_grids(self):
        """Validar que el generador produce grids con agregados correctos."""
        target_mean = 0.75
        grid_size = 5

        # Test uniform
        grid = SpatialStateGenerator.uniform(target_mean, grid_size)
        actual_mean = sum(sum(row) for row in grid) / (grid_size * grid_size)
        assert abs(actual_mean - target_mean) < 0.01

        # Test hotspot
        grid = SpatialStateGenerator.hotspot_center(target_mean, grid_size)
        actual_mean = sum(sum(row) for row in grid) / (grid_size * grid_size)
        assert abs(actual_mean - target_mean) < 0.05  # Más tolerancia por redondeo

        # Test gradientes
        grid = SpatialStateGenerator.gradient_ns(target_mean, grid_size)
        actual_mean = sum(sum(row) for row in grid) / (grid_size * grid_size)
        assert abs(actual_mean - target_mean) < 0.05

    def test_no_isomorphism_gate(self, tmp_path: Path):
        """Test de No-Isomorfismo: ¿El 5x5 usa información espacial?

        Este es el gate crítico. Si falla, el 5x5 es cognitivamente isomorfo al 1x1.

        Criterio de éxito:
        - spatial_information_usage > 0.2
        - Al menos 5/10 pares muestran diferencias en decisiones
        """
        storage = _storage(tmp_path)

        # Definir 10 pares de estados con mismo agregado, distinta topología
        test_pairs = [
            (0.75, 'uniform', 'hotspot', "Uniforme vs Hotspot central"),
            (0.80, 'uniform', 'gradient_ns', "Uniforme vs Gradiente N-S"),
            (0.70, 'uniform', 'gradient_ew', "Uniforme vs Gradiente E-O"),
            (0.82, 'uniform', 'checkerboard', "Uniforme vs Checkerboard"),
            (0.78, 'uniform', 'quadrants', "Uniforme vs Cuadrantes"),
            (0.75, 'gradient_ns', 'gradient_ew', "Gradiente N-S vs E-O"),
            (0.80, 'hotspot', 'checkerboard', "Hotspot vs Checkerboard"),
            (0.72, 'gradient_ns', 'quadrants', "Gradiente N-S vs Cuadrantes"),
            (0.85, 'hotspot', 'gradient_ns', "Hotspot vs Gradiente N-S"),
            (0.77, 'checkerboard', 'quadrants', "Checkerboard vs Cuadrantes"),
        ]

        pairs_results = []
        pairs_with_differences = 0

        for idx, (target_mean, topo_a, topo_b, description) in enumerate(test_pairs):
            print(f"\n=== Par {idx + 1}/10: {description} (mean={target_mean:.2f}) ===")

            # Generar par de grids
            grid_a, grid_b = SpatialStateGenerator.create_state_pair(
                target_mean, topo_a, topo_b
            )

            # Crear escenarios y aplicar estados
            scenario_a = GridThermalScenario(initial_temperature=0.5, alarm_threshold=0.85)
            scenario_b = GridThermalScenario(initial_temperature=0.5, alarm_threshold=0.85)

            apply_spatial_state_to_scenario(scenario_a, grid_a)
            apply_spatial_state_to_scenario(scenario_b, grid_b)

            # Verificar que agregados son iguales
            mean_a = scenario_a._grid.global_temp_mean
            mean_b = scenario_b._grid.global_temp_mean
            print(f"  Global means: A={mean_a:.4f}, B={mean_b:.4f}, diff={abs(mean_a - mean_b):.4f}")

            # Ejecutar observación en ambos
            obs_a = scenario_a.observe()
            obs_b = scenario_b.observe()

            # Comparar proposiciones
            props_differ = set(obs_a.propositions) != set(obs_b.propositions)
            print(f"  Propositions A: {obs_a.propositions}")
            print(f"  Propositions B: {obs_b.propositions}")
            print(f"  Propositions differ: {props_differ}")

            # Comparar intervención seleccionada
            intervention_a = scenario_a.select_intervention(obs_a)
            intervention_b = scenario_b.select_intervention(obs_b)
            intervention_differs = intervention_a != intervention_b
            print(f"  Intervention A: {intervention_a}")
            print(f"  Intervention B: {intervention_b}")
            print(f"  Intervention differs: {intervention_differs}")

            # Comparar nivel discreto (world_level)
            level_a = obs_a.level
            level_b = obs_b.level
            level_differs = level_a != level_b
            print(f"  Level A: {level_a} ({obs_a.state['world_level_semantic']})")
            print(f"  Level B: {level_b} ({obs_b.state['world_level_semantic']})")
            print(f"  Level differs: {level_differs}")

            # Registrar resultados
            any_difference = props_differ or intervention_differs or level_differs
            if any_difference:
                pairs_with_differences += 1
                print(f"  ✓ DIFERENCIA detectada")
            else:
                print(f"  ✗ Sin diferencias cognitivas")

            pairs_results.append({
                'pair_id': idx + 1,
                'description': description,
                'target_mean': target_mean,
                'topology_a': topo_a,
                'topology_b': topo_b,
                'mean_a': mean_a,
                'mean_b': mean_b,
                'propositions_a': list(obs_a.propositions),
                'propositions_b': list(obs_b.propositions),
                'propositions_differ': props_differ,
                'intervention_a': intervention_a,
                'intervention_b': intervention_b,
                'intervention_differs': intervention_differs,
                'level_a': level_a,
                'level_b': level_b,
                'level_differs': level_differs,
                'any_difference': any_difference,
            })

        # Calcular spatial_information_usage
        spatial_usage = compute_spatial_information_usage(pairs_results)

        # Persistir resultados completos para análisis
        import json
        results_file = tmp_path / "phase0_gate_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'spatial_information_usage': spatial_usage,
                'pairs_with_differences': pairs_with_differences,
                'total_pairs': len(test_pairs),
                'pairs': pairs_results,
            }, f, indent=2)
        print(f"\n📊 Resultados detallados guardados en: {results_file}")

        print(f"\n{'='*70}")
        print(f"RESULTADOS DEL TEST DE NO-ISOMORFISMO")
        print(f"{'='*70}")
        print(f"Pares con diferencias: {pairs_with_differences}/10")
        print(f"spatial_information_usage: {spatial_usage:.3f}")
        print(f"{'='*70}")

        # Evaluar criterio de éxito
        gate_passed = spatial_usage > 0.2 and pairs_with_differences >= 5

        if gate_passed:
            print("✅ GATE PASSED: El 5x5 USA información espacial para cognición")
            print("   → Continuar con Fase 1 (Instrumentación)")
        else:
            print("❌ GATE FAILED: El 5x5 es cognitivamente ISOMORFO al 1x1")
            print("   → PARAR y rediseñar antes de instrumentar")
            print(f"\n   Razón del fallo:")
            if spatial_usage <= 0.2:
                print(f"   - spatial_information_usage ({spatial_usage:.3f}) ≤ 0.2")
            if pairs_with_differences < 5:
                print(f"   - Pares con diferencias ({pairs_with_differences}) < 5")

        storage.close()

        # Assert para pytest
        assert gate_passed, (
            f"Test de No-Isomorfismo FALLÓ: "
            f"spatial_usage={spatial_usage:.3f} (requiere >0.2), "
            f"diferencias={pairs_with_differences}/10 (requiere ≥5)"
        )

        return spatial_usage, pairs_results
