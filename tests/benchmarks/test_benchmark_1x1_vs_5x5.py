"""Test Benchmark 1x1 vs 5x5: Protocolo experimental comparativo.

Este test ejecuta el benchmark completo definido en la especificación de Fase 1.
Marca con pytest.mark.requires_extended_bench para ejecución explícita.
"""

import pytest
from pathlib import Path
import yaml
from datetime import datetime

from runtime.world.grid_thermal_scenario import GridThermalScenario
from .benchmark_runner import BenchmarkRunner, BenchmarkConfig


@pytest.fixture
def benchmark_output_dir(tmp_path):
    """Directorio para resultados de benchmark."""
    output_dir = tmp_path / "benchmark_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@pytest.fixture
def benchmark_config():
    """Carga configuración de benchmark."""
    config_file = Path(__file__).parent / "benchmark_config.yaml"
    with open(config_file) as f:
        return yaml.safe_load(f)


@pytest.mark.requires_extended_bench
class TestBenchmark1x1vs5x5:
    """Suite de benchmarks comparativos 1x1 vs 5x5."""

    def test_baseline_comparison_1x1(self, benchmark_output_dir, benchmark_config):
        """Ejecuta baseline 1x1 (100 episodios)."""
        baseline_cfg = benchmark_config['baseline']
        scenario_cfg = baseline_cfg['scenarios'][0]  # 1x1

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = benchmark_output_dir / scenario_cfg['name'] / run_id

        config = BenchmarkConfig(
            scenario_name=scenario_cfg['name'],
            scenario_class=GridThermalScenario,
            scenario_params={
                'grid_size': scenario_cfg['grid_size'],
                'initial_temperature': scenario_cfg['initial_temperature'],
                'alarm_threshold': scenario_cfg['alarm_threshold'],
                'cooling_effect': scenario_cfg['cooling_effect'],
            },
            episodes=baseline_cfg['episodes_per_scenario'],
            base_seed=baseline_cfg['base_seed'],
            max_steps=baseline_cfg['max_steps'],
            output_dir=output_dir,
        )

        runner = BenchmarkRunner(benchmark_output_dir)
        summary = runner.run_benchmark(config)

        # Assertions básicas
        assert summary['total_episodes'] == 100
        assert summary['success_rate'] > 0.5  # Al menos 50% de éxito

        print(f"\n{'='*70}")
        print(f"RESUMEN BASELINE 1x1")
        print(f"{'='*70}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Avg IVC-R: {summary['avg_metrics'].get('ivc_r', 0.0):.3f}")
        print(f"Avg wall time: {summary['avg_metrics'].get('wall_time_ms', 0.0):.1f}ms")
        print(f"{'='*70}\n")

    def test_baseline_comparison_5x5_uniform(self, benchmark_output_dir, benchmark_config):
        """Ejecuta baseline 5x5 uniform (100 episodios)."""
        baseline_cfg = benchmark_config['baseline']
        scenario_cfg = baseline_cfg['scenarios'][1]  # 5x5 uniform

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = benchmark_output_dir / scenario_cfg['name'] / run_id

        config = BenchmarkConfig(
            scenario_name=scenario_cfg['name'],
            scenario_class=GridThermalScenario,
            scenario_params={
                'grid_size': scenario_cfg['grid_size'],
                'initial_temperature': scenario_cfg['initial_temperature'],
                'alarm_threshold': scenario_cfg['alarm_threshold'],
                'cooling_effect': scenario_cfg['cooling_effect'],
                'topology': scenario_cfg['topology'],
            },
            episodes=baseline_cfg['episodes_per_scenario'],
            base_seed=baseline_cfg['base_seed'],
            max_steps=baseline_cfg['max_steps'],
            output_dir=output_dir,
        )

        runner = BenchmarkRunner(benchmark_output_dir)
        summary = runner.run_benchmark(config)

        # Assertions
        assert summary['total_episodes'] == 100
        assert summary['success_rate'] > 0.5

        print(f"\n{'='*70}")
        print(f"RESUMEN BASELINE 5x5 UNIFORM")
        print(f"{'='*70}")
        print(f"Success rate: {summary['success_rate']:.2%}")
        print(f"Avg IVC-R: {summary['avg_metrics'].get('ivc_r', 0.0):.3f}")
        print(f"Avg wall time: {summary['avg_metrics'].get('wall_time_ms', 0.0):.1f}ms")
        print(f"Avg spatial usage: {summary['avg_metrics'].get('spatial_information_usage', 0.0):.3f}")
        print(f"{'='*70}\n")

    def test_heterogeneous_5x5(self, benchmark_output_dir, benchmark_config):
        """Ejecuta 5x5 con topologías heterogéneas (100 episodios)."""
        hetero_cfg = benchmark_config['heterogeneous_5x5']
        scenario_base = hetero_cfg['scenario_base']
        topology_dist = hetero_cfg['topology_distribution']

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = benchmark_output_dir / scenario_base['name'] / run_id

        # Ejecutar episodios con diferentes topologías
        all_results = []
        episode_idx = 0

        for topology, count in topology_dist.items():
            print(f"\nEjecutando {count} episodios con topología: {topology}")

            config = BenchmarkConfig(
                scenario_name=f"{scenario_base['name']}_{topology}",
                scenario_class=GridThermalScenario,
                scenario_params={
                    'grid_size': scenario_base['grid_size'],
                    'initial_temperature': scenario_base['initial_temperature'],
                    'alarm_threshold': scenario_base['alarm_threshold'],
                    'cooling_effect': scenario_base['cooling_effect'],
                    'topology': topology,
                },
                episodes=count,
                base_seed=hetero_cfg['base_seed'] + episode_idx,
                max_steps=hetero_cfg['max_steps'],
                output_dir=output_dir / topology,
            )

            runner = BenchmarkRunner(benchmark_output_dir)
            summary = runner.run_benchmark(config)
            all_results.append((topology, summary))

            episode_idx += count

        # Resumen consolidado
        print(f"\n{'='*70}")
        print(f"RESUMEN HETEROGENEOUS 5x5")
        print(f"{'='*70}")

        for topology, summary in all_results:
            print(f"\n{topology}:")
            print(f"  Success rate: {summary['success_rate']:.2%}")
            print(f"  Avg IVC-R: {summary['avg_metrics'].get('ivc_r', 0.0):.3f}")
            print(f"  Avg spatial usage: {summary['avg_metrics'].get('spatial_information_usage', 0.0):.3f}")

        print(f"\n{'='*70}\n")

    def test_level_sweep_1x1(self, benchmark_output_dir, benchmark_config):
        """Barrido por niveles del mundo (1x1)."""
        level_cfg = benchmark_config['level_sweep']
        grid_size = 1

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_by_level = {}

        for level_spec in level_cfg['scenarios']:
            level_name = level_spec['name']
            print(f"\nEjecutando nivel {level_name} (1x1)...")

            config = BenchmarkConfig(
                scenario_name=f"grid_thermal_1x1_{level_name.lower()}",
                scenario_class=GridThermalScenario,
                scenario_params={
                    'grid_size': grid_size,
                    'initial_temperature': level_spec['initial_temperature'],
                    'alarm_threshold': level_spec['alarm_threshold'],
                    'cooling_effect': level_cfg['cooling_effect'],
                },
                episodes=level_spec['episodes'],
                base_seed=level_cfg['base_seed'],
                max_steps=level_cfg['max_steps'],
                output_dir=benchmark_output_dir / f"level_sweep_1x1" / run_id / level_name.lower(),
            )

            runner = BenchmarkRunner(benchmark_output_dir)
            summary = runner.run_benchmark(config)
            results_by_level[level_name] = summary

        # Resumen por nivel
        print(f"\n{'='*70}")
        print(f"LEVEL SWEEP 1x1")
        print(f"{'='*70}")

        for level_name, summary in results_by_level.items():
            print(f"\n{level_name}:")
            print(f"  Success rate: {summary['success_rate']:.2%}")
            print(f"  Avg IVC-R: {summary['avg_metrics'].get('ivc_r', 0.0):.3f}")

        print(f"\n{'='*70}\n")

    def test_level_sweep_5x5(self, benchmark_output_dir, benchmark_config):
        """Barrido por niveles del mundo (5x5)."""
        level_cfg = benchmark_config['level_sweep']
        grid_size = 5

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        results_by_level = {}

        for level_spec in level_cfg['scenarios']:
            level_name = level_spec['name']
            print(f"\nEjecutando nivel {level_name} (5x5)...")

            config = BenchmarkConfig(
                scenario_name=f"grid_thermal_5x5_{level_name.lower()}",
                scenario_class=GridThermalScenario,
                scenario_params={
                    'grid_size': grid_size,
                    'initial_temperature': level_spec['initial_temperature'],
                    'alarm_threshold': level_spec['alarm_threshold'],
                    'cooling_effect': level_cfg['cooling_effect'],
                    'topology': 'uniform',  # Usar uniform para comparabilidad
                },
                episodes=level_spec['episodes'],
                base_seed=level_cfg['base_seed'] + 1000,  # Offset para 5x5
                max_steps=level_cfg['max_steps'],
                output_dir=benchmark_output_dir / f"level_sweep_5x5" / run_id / level_name.lower(),
            )

            runner = BenchmarkRunner(benchmark_output_dir)
            summary = runner.run_benchmark(config)
            results_by_level[level_name] = summary

        # Resumen por nivel
        print(f"\n{'='*70}")
        print(f"LEVEL SWEEP 5x5")
        print(f"{'='*70}")

        for level_name, summary in results_by_level.items():
            print(f"\n{level_name}:")
            print(f"  Success rate: {summary['success_rate']:.2%}")
            print(f"  Avg IVC-R: {summary['avg_metrics'].get('ivc_r', 0.0):.3f}")
            print(f"  Avg spatial usage: {summary['avg_metrics'].get('spatial_information_usage', 0.0):.3f}")

        print(f"\n{'='*70}\n")
