"""Benchmark Runner: Ejecuta experimentos comparativos 1x1 vs 5x5.

Orquesta la ejecución de episodios, captura de métricas y persistencia de resultados.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import time
import traceback
import uuid
from datetime import datetime

from runtime.storage import StorageConfig, StorageFactory
from runtime.world.scenario_runner import ScenarioEpisodeRunner
from runtime.world.grid_thermal_scenario import GridThermalScenario

from .metrics_cognitive_quality import compute_all_cognitive_metrics
from .metrics_operational_cost import compute_all_operational_cost_metrics
from .metrics_ivc_r import compute_ivc_r_from_episode
from .failure_taxonomy import classify_episode_failures


class BenchmarkConfig:
    """Configuración de benchmark."""

    def __init__(
        self,
        scenario_name: str,
        scenario_class: type,
        scenario_params: Dict[str, Any],
        episodes: int,
        base_seed: int,
        max_steps: int,
        output_dir: Path,
    ):
        self.scenario_name = scenario_name
        self.scenario_class = scenario_class
        self.scenario_params = scenario_params
        self.episodes = episodes
        self.base_seed = base_seed
        self.max_steps = max_steps
        self.output_dir = output_dir


class EpisodeResult:
    """Resultado de un episodio individual."""

    def __init__(self, episode_id: str, scenario_name: str, seed: int):
        self.episode_id = episode_id
        self.scenario_name = scenario_name
        self.seed = seed
        self.outcome = None
        self.error = None
        self.trace = []
        self.trace_length = 0
        self.cierre_rate = 0.0
        self.continuity_score = 0.0
        self.counterfactual = None
        self.metadata = {}
        self.metrics = {}
        self.wall_time_ms = 0.0
        self.start_time = None
        self.end_time = None

    def to_dict(self) -> Dict[str, Any]:
        """Convierte resultado a diccionario."""
        return {
            'episode_id': self.episode_id,
            'scenario': self.scenario_name,
            'seed': self.seed,
            'outcome': self.outcome,
            'error': self.error,
            'trace_length': self.trace_length,
            'cierre_rate': self.cierre_rate,
            'continuity_score': self.continuity_score,
            'counterfactual': self.counterfactual,
            'metadata': self.metadata,
            'wall_time_ms': self.wall_time_ms,
            'timestamp': self.start_time.isoformat() if self.start_time else None,
            **self.metrics,
        }


class BenchmarkRunner:
    """Ejecuta benchmarks comparativos entre escenarios."""

    def __init__(self, output_root: Path):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def run_benchmark(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Ejecuta benchmark completo según configuración.

        Args:
            config: Configuración del benchmark.

        Returns:
            Diccionario con resultados agregados.
        """
        print(f"\n{'='*70}")
        print(f"EJECUTANDO BENCHMARK: {config.scenario_name}")
        print(f"{'='*70}")
        print(f"Episodios: {config.episodes}")
        print(f"Base seed: {config.base_seed}")
        print(f"Max steps: {config.max_steps}")
        print(f"Output: {config.output_dir}")
        print(f"{'='*70}\n")

        # Crear directorio de salida
        config.output_dir.mkdir(parents=True, exist_ok=True)

        # Ejecutar episodios
        results = []
        for i in range(config.episodes):
            seed = config.base_seed + i
            episode_id = str(uuid.uuid4())

            print(f"Episodio {i+1}/{config.episodes} (seed={seed})...", end=" ", flush=True)

            try:
                result = self.run_single_episode(
                    config=config,
                    episode_id=episode_id,
                    seed=seed,
                )
                results.append(result)
                print(f"✓ {result.outcome} ({result.trace_length} steps, {result.wall_time_ms:.1f}ms)")

            except Exception as e:
                print(f"✗ ERROR: {str(e)}")
                # Crear resultado de error
                error_result = EpisodeResult(episode_id, config.scenario_name, seed)
                error_result.outcome = 'error'
                error_result.error = str(e)
                results.append(error_result)

        # Persistir resultados
        self.persist_results(results, config.output_dir)

        # Generar resumen
        summary = self.generate_summary(results, config)

        return summary

    def run_single_episode(
        self,
        config: BenchmarkConfig,
        episode_id: str,
        seed: int,
    ) -> EpisodeResult:
        """Ejecuta un episodio individual con captura completa de métricas.

        Args:
            config: Configuración del benchmark.
            episode_id: ID único del episodio.
            seed: Semilla para reproducibilidad.

        Returns:
            EpisodeResult con métricas completas.
        """
        result = EpisodeResult(episode_id, config.scenario_name, seed)
        result.start_time = datetime.now()

        # Iniciar timer
        t0 = time.perf_counter()

        # Crear storage temporal
        temp_storage = self._create_temp_storage()

        try:
            # Crear escenario
            scenario = config.scenario_class(**config.scenario_params)

            # Guardar metadata
            result.metadata = {
                'initial_temperature': config.scenario_params.get('initial_temperature', 0.82),
                'alarm_threshold': config.scenario_params.get('alarm_threshold', 0.85),
                'cooling_effect': config.scenario_params.get('cooling_effect', 0.07),
                'grid_size': config.scenario_params.get('grid_size', 1),
                'topology': config.scenario_params.get('topology'),
            }

            # Ejecutar episodio
            runner = ScenarioEpisodeRunner(
                scenario=scenario,
                storage=temp_storage,
                max_steps=config.max_steps,
            )

            # TODO: Capturar trace completo
            # Por ahora, ejecutar y extraer métricas básicas
            episode_data = runner.run_episode()

            result.outcome = 'success' if episode_data.get('closed', False) else 'failure'
            result.trace_length = len(episode_data.get('trace', []))
            result.cierre_rate = 1.0 if episode_data.get('closed', False) else 0.0
            result.continuity_score = episode_data.get('continuity_score', 0.0)
            result.trace = episode_data.get('trace', [])

            # Extraer contrafactual si existe
            if 'counterfactual' in episode_data:
                result.counterfactual = episode_data['counterfactual']

        except Exception as e:
            result.outcome = 'error'
            result.error = f"{type(e).__name__}: {str(e)}"
            result.trace_length = 0

        finally:
            temp_storage.close()

        # Finalizar timer
        t1 = time.perf_counter()
        result.wall_time_ms = (t1 - t0) * 1000.0
        result.end_time = datetime.now()

        # Calcular métricas
        episode_dict = result.to_dict()
        episode_dict['trace'] = result.trace
        episode_dict['counterfactual'] = result.counterfactual

        # Grupo 2: Calidad cognitiva
        cognitive_metrics = compute_all_cognitive_metrics(episode_dict)
        result.metrics.update(cognitive_metrics)

        # Grupo 3: Costo operativo
        operational_metrics = compute_all_operational_cost_metrics(episode_dict)
        result.metrics.update(operational_metrics)

        # Grupo 4: IVC-R
        ivc_r_result = compute_ivc_r_from_episode({**episode_dict, **result.metrics})
        result.metrics['ivc_r'] = ivc_r_result['ivc_r']
        result.metrics['ivc_r_log'] = ivc_r_result['ivc_r_log']
        result.metrics['ivc_r_components'] = ivc_r_result['components']

        # Grupo 5: Clasificación de fallos
        failure_classification = classify_episode_failures({**episode_dict, **result.metrics})
        result.metrics['failure_primary'] = failure_classification['failure_primary']
        result.metrics['failure_secondary'] = failure_classification['failure_secondary']

        return result

    def persist_results(self, results: List[EpisodeResult], output_dir: Path):
        """Persiste resultados en formato JSONL.

        Args:
            results: Lista de resultados de episodios.
            output_dir: Directorio de salida.
        """
        episodes_file = output_dir / "episodes.jsonl"

        with open(episodes_file, 'w') as f:
            for result in results:
                # No incluir trace completo en JSONL (muy pesado)
                result_dict = result.to_dict()
                result_dict.pop('trace', None)  # Trace va a archivo separado si se requiere

                f.write(json.dumps(result_dict, default=str) + '\n')

        print(f"\n📊 Resultados guardados en: {episodes_file}")

    def generate_summary(self, results: List[EpisodeResult], config: BenchmarkConfig) -> Dict[str, Any]:
        """Genera resumen estadístico de resultados.

        Args:
            results: Lista de resultados.
            config: Configuración del benchmark.

        Returns:
            Diccionario con resumen.
        """
        total = len(results)
        successful = sum(1 for r in results if r.outcome == 'success')
        failed = total - successful

        # Calcular promedios de métricas (solo episodios exitosos)
        success_results = [r for r in results if r.outcome == 'success']

        if success_results:
            avg_metrics = {}
            metric_keys = [
                'cierre_rate', 'continuity_score', 'intervention_precision',
                'proposition_diversity', 'spatial_information_usage',
                'wall_time_ms', 'artifact_size_bytes', 'ivc_r',
            ]

            for key in metric_keys:
                values = [r.metrics.get(key, 0.0) for r in success_results if key in r.metrics]
                if values:
                    avg_metrics[key] = sum(values) / len(values)
                else:
                    avg_metrics[key] = 0.0
        else:
            avg_metrics = {key: 0.0 for key in metric_keys}

        # Agregación de fallos
        from .failure_taxonomy import aggregate_failure_distribution
        failure_dist = aggregate_failure_distribution([r.to_dict() for r in results])

        summary = {
            'scenario': config.scenario_name,
            'total_episodes': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0.0,
            'avg_metrics': avg_metrics,
            'failure_distribution': failure_dist,
        }

        # Guardar resumen
        summary_file = config.output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"📊 Resumen guardado en: {summary_file}")

        return summary

    def _create_temp_storage(self):
        """Crea storage temporal para episodio."""
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())

        config = StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(temp_dir / "temp.db"),
            postgres_dsn=None,
            artifact_root=temp_dir / "artifacts",
            prefer_postgres_reads=False,
            strict_dual_write=False,
        )

        return StorageFactory.create_facade(config)
