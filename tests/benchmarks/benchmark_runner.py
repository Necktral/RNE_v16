"""Benchmark Runner: Ejecuta experimentos comparativos 1x1 vs 5x5.

CORREGIDO: Alineado con el contrato real de ScenarioEpisodeRunner.
No asume claves ficticias. Adapta el payload real del runtime a formato benchmark.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import time
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
        max_steps: int,  # Mantenido para config, pero NO se pasa al runtime
        output_dir: Path,
    ):
        self.scenario_name = scenario_name
        self.scenario_class = scenario_class
        self.scenario_params = scenario_params
        self.episodes = episodes
        self.base_seed = base_seed
        self.max_steps = max_steps  # Usado solo para documentación
        self.output_dir = output_dir


class EpisodeResult:
    """Resultado de un episodio individual adaptado desde runtime."""

    def __init__(self, episode_id: str, scenario_name: str, seed: int):
        self.episode_id = episode_id
        self.scenario_name = scenario_name
        self.seed = seed
        self.outcome = None  # 'success', 'failure', 'error'
        self.error = None

        # Datos del runtime (estructura real)
        self.runtime_payload = None

        # Métricas derivadas
        self.certification_verdict = None
        self.is_viable = None
        self.viability_margin = None
        self.artifact_path = None
        self.artifact_size_bytes = None
        self.reasoning_trace_length = None
        self.organism_trajectory = None

        self.metadata = {}
        self.metrics = {}
        self.wall_time_ms = 0.0
        self.start_time = None
        self.end_time = None

    def to_dict(self) -> Dict[str, Any]:
        """Convierte resultado a diccionario para análisis."""
        return {
            'episode_id': self.episode_id,
            'scenario': self.scenario_name,
            'seed': self.seed,
            'outcome': self.outcome,
            'error': self.error,
            'certification_verdict': self.certification_verdict,
            'is_viable': self.is_viable,
            'viability_margin': self.viability_margin,
            'artifact_path': self.artifact_path,
            'artifact_size_bytes': self.artifact_size_bytes,
            'reasoning_trace_length': self.reasoning_trace_length,
            'metadata': self.metadata,
            'wall_time_ms': self.wall_time_ms,
            'timestamp': self.start_time.isoformat() if self.start_time else None,
            **self.metrics,
        }


def adapt_runtime_result_to_benchmark(runtime_result: Dict[str, Any]) -> Dict[str, Any]:
    """Adapta el payload real del runtime a formato benchmark.

    El runtime retorna:
    - episode: {...}
    - smg_snapshot: {...}
    - reasoning: {...}
    - artifact: {...}
    - run_id: str
    - organism_trajectory: {...}
    - constitutional_validation: {...}
    - viability_assessment: {...}
    - certification: {...}
    - eml_shadow: {...}

    Esta función normaliza a un formato benchmark coherente.
    """
    adapted = {
        'episode_id': runtime_result.get('episode', {}).get('episode_id'),
        'run_id': runtime_result.get('run_id'),
        'scenario_name': runtime_result.get('episode', {}).get('scenario'),

        # Certification
        'certification_verdict': runtime_result.get('certification', {}).get('verdict'),
        'promotion_candidate': runtime_result.get('certification', {}).get('promotion_candidate'),

        # Viability
        'is_viable': runtime_result.get('viability_assessment', {}).get('is_viable'),
        'viability_margin': runtime_result.get('viability_assessment', {}).get('viability_margin'),
        'distance_to_edge': runtime_result.get('viability_assessment', {}).get('distance_to_edge'),

        # Artifact
        'artifact_path': runtime_result.get('artifact', {}).get('abs_path'),

        # Reasoning trace (del scheduler, NO del mundo)
        'reasoning_sequence': runtime_result.get('reasoning', {}).get('sequence', []),

        # Organism trajectory
        'organism_trajectory': runtime_result.get('organism_trajectory'),

        # Episode context
        'observation': runtime_result.get('episode', {}).get('context', {}).get('observation'),
        'updated_world': runtime_result.get('episode', {}).get('result', {}).get('updated_world'),
        'counterfactual_world': runtime_result.get('episode', {}).get('context', {}).get('counterfactual'),

        # Raw payload para métricas que lo necesiten
        'raw_runtime_result': runtime_result,
    }

    return adapted


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
                print(f"✓ {result.outcome} ({result.certification_verdict}, {result.wall_time_ms:.1f}ms)")

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

        CORREGIDO: Usa contrato real de ScenarioEpisodeRunner.

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

            # Crear runner SIN max_steps (no existe en el contrato)
            runner = ScenarioEpisodeRunner(
                scenario=scenario,
                storage=temp_storage,
                run_id=f"bench-{episode_id[:8]}",
            )

            # Ejecutar UN SOLO EPISODIO (un paso cognitivo)
            runtime_result = runner.run_episode(external_input=0.04)

            # Adaptar resultado del runtime a formato benchmark
            adapted_result = adapt_runtime_result_to_benchmark(runtime_result)

            # Guardar payload completo
            result.runtime_payload = runtime_result

            # Extraer métricas del runtime
            cert_verdict = adapted_result['certification_verdict']
            result.certification_verdict = cert_verdict
            result.is_viable = adapted_result['is_viable']
            result.viability_margin = adapted_result['viability_margin']
            result.artifact_path = adapted_result['artifact_path']
            result.organism_trajectory = adapted_result['organism_trajectory']

            # Calcular artifact size si existe
            if result.artifact_path and Path(result.artifact_path).exists():
                result.artifact_size_bytes = Path(result.artifact_path).stat().st_size

            # Reasoning trace length
            result.reasoning_trace_length = len(adapted_result['reasoning_sequence'])

            # Determinar outcome según certificación
            if cert_verdict in ['passed', 'certified']:
                result.outcome = 'success'
            else:
                result.outcome = 'failure'

        except Exception as e:
            result.outcome = 'error'
            result.error = f"{type(e).__name__}: {str(e)}"

        finally:
            temp_storage.close()

        # Finalizar timer
        t1 = time.perf_counter()
        result.wall_time_ms = (t1 - t0) * 1000.0
        result.end_time = datetime.now()

        # Calcular métricas CORREGIDAS (sobre payload real)
        if result.runtime_payload:
            adapted_payload = adapt_runtime_result_to_benchmark(result.runtime_payload)

            # Grupo 2: Calidad cognitiva (ADAPTADO)
            cognitive_metrics = compute_all_cognitive_metrics(adapted_payload)
            result.metrics.update(cognitive_metrics)

            # Grupo 3: Costo operativo (ADAPTADO)
            operational_metrics = compute_all_operational_cost_metrics({
                **adapted_payload,
                'wall_time_ms': result.wall_time_ms,
                'artifact_size_bytes': result.artifact_size_bytes,
                'reasoning_trace_length': result.reasoning_trace_length,
            })
            result.metrics.update(operational_metrics)

            # Grupo 4: IVC-R (ADAPTADO)
            # IVC-R requiere métricas ya calculadas
            ivc_r_input = {
                **result.to_dict(),
                **result.metrics,
                'cierre_rate': 1.0 if result.outcome == 'success' else 0.0,
                'continuity_score': result.viability_margin if result.viability_margin else 0.0,
            }
            ivc_r_result = compute_ivc_r_from_episode(ivc_r_input)
            result.metrics['ivc_r'] = ivc_r_result.get('ivc_r', 0.0)
            result.metrics['ivc_r_log'] = ivc_r_result.get('ivc_r_log', 0.0)
            result.metrics['ivc_r_components'] = ivc_r_result.get('components', {})

            # Grupo 5: Clasificación de fallos (ADAPTADO)
            failure_classification = classify_episode_failures({
                **adapted_payload,
                **result.to_dict(),
                **result.metrics,
            })
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
                result_dict = result.to_dict()
                # No incluir runtime_payload completo (muy pesado)
                result_dict.pop('runtime_payload', None)
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
        failed = sum(1 for r in results if r.outcome == 'failure')
        errors = sum(1 for r in results if r.outcome == 'error')

        # Calcular promedios de métricas (solo episodios exitosos)
        success_results = [r for r in results if r.outcome == 'success']

        if success_results:
            avg_metrics = {}
            metric_keys = [
                'intervention_precision',
                'proposition_diversity',
                'spatial_information_usage',
                'wall_time_ms',
                'artifact_size_bytes',
                'reasoning_trace_length',
                'ivc_r',
            ]

            for key in metric_keys:
                values = []
                for r in success_results:
                    if key == 'wall_time_ms':
                        values.append(r.wall_time_ms)
                    elif key == 'artifact_size_bytes':
                        if r.artifact_size_bytes:
                            values.append(r.artifact_size_bytes)
                    elif key == 'reasoning_trace_length':
                        if r.reasoning_trace_length:
                            values.append(r.reasoning_trace_length)
                    elif key in r.metrics:
                        val = r.metrics[key]
                        if val is not None:
                            values.append(val)

                if values:
                    avg_metrics[key] = sum(values) / len(values)
                else:
                    avg_metrics[key] = 0.0

            # Viability margin promedio
            viability_values = [r.viability_margin for r in success_results if r.viability_margin is not None]
            avg_metrics['viability_margin'] = sum(viability_values) / len(viability_values) if viability_values else 0.0
        else:
            avg_metrics = {key: 0.0 for key in metric_keys}
            avg_metrics['viability_margin'] = 0.0

        # Agregación de fallos
        from .failure_taxonomy import aggregate_failure_distribution
        failure_dist = aggregate_failure_distribution([r.to_dict() for r in results])

        summary = {
            'scenario': config.scenario_name,
            'total_episodes': total,
            'successful': successful,
            'failed': failed,
            'errors': errors,
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
