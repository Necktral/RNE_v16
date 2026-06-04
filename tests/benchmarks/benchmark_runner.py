"""Benchmark Runner: Ejecuta experimentos comparativos 1x1 vs 5x5.

CORREGIDO: Alineado con el contrato real de ScenarioEpisodeRunner.
No asume claves ficticias. Adapta el payload real del runtime a formato benchmark.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os
import time
import uuid

from runtime.reasoning.scheduler_meta.family_metrics import (
    aggregate_family_activation_counts,
    aggregate_family_dict_metric,
    build_family_sensitive_bundle,
)
from runtime.storage import StorageConfig, StorageFactory
from runtime.world.grid_thermal_scenario import GridThermalScenario
from runtime.world.scenario_runner import ScenarioEpisodeRunner

from .failure_taxonomy import classify_episode_failures
from .metrics_cognitive_quality import compute_all_cognitive_metrics
from .metrics_ivc_r import compute_ivc_r_from_episode
from .metrics_operational_cost import compute_all_operational_cost_metrics


@contextmanager
def _temporary_reasoning_env(updates: Dict[str, str | None]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
        run_id: Optional[str] = None,
        reasoning_mode: Optional[str] = None,
        family_profile: Optional[str] = None,
        regime_label: Optional[str] = None,
        reasoning_max_steps: Optional[int] = None,
        closure_profile: Optional[str] = None,
    ):
        self.scenario_name = scenario_name
        self.scenario_class = scenario_class
        self.scenario_params = scenario_params
        self.episodes = episodes
        self.base_seed = base_seed
        self.max_steps = max_steps  # Usado solo para documentación
        self.output_dir = output_dir
        self.run_id = run_id
        self.reasoning_mode = reasoning_mode
        self.family_profile = family_profile
        self.regime_label = regime_label
        self.reasoning_max_steps = reasoning_max_steps
        self.closure_profile = closure_profile


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

        # Configuración de razonamiento (observada/forzada)
        self.reasoning_mode = None
        self.family_profile = None
        self.regime_label = None

        # Bundle family-sensitive
        self.family_sensitive: Dict[str, Any] = {}

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
            'reasoning_mode': self.reasoning_mode,
            'family_profile': self.family_profile,
            'regime_label': self.regime_label,
            'metadata': self.metadata,
            'wall_time_ms': self.wall_time_ms,
            'timestamp': self.start_time.isoformat() if self.start_time else None,
            # Proxy metrics for analysis compatibility
            'success_rate': 1.0 if self.outcome == 'success' else 0.0,
            **self.metrics,
            **self.family_sensitive,
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
    reasoning = runtime_result.get('reasoning', {}) or {}
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
        'reasoning_sequence': reasoning.get('sequence', []),
        'reasoning_trace': reasoning.get('trace', []),
        'reasoning_mode': reasoning.get('mode'),
        'family_profile': reasoning.get('family_profile'),
        'regime_label': reasoning.get('regime_label'),
        'primary_regime_label': reasoning.get('primary_regime_label'),
        'cognitive_regime_label': reasoning.get('cognitive_regime_label'),
        'floor_regime_label': reasoning.get('floor_regime_label'),
        'mandatory_family_floor': reasoning.get('mandatory_family_floor', []),
        'proposed_sequence': reasoning.get('proposed_sequence', []),
        'validated_sequence': reasoning.get('validated_sequence', []),
        'sequence_validation': reasoning.get('sequence_validation', {}),
        'effective_max_steps': reasoning.get('effective_max_steps'),

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

    def __init__(self, output_root: Path, storage_config: Optional[StorageConfig] = None):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.storage_config = storage_config
        self._active_storage = None
        self._active_run_id = None

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
        print(f"Reasoning mode/profile/regime: {config.reasoning_mode}/{config.family_profile}/{config.regime_label}")
        print(f"{'='*70}\n")

        # Crear directorio de salida
        config.output_dir.mkdir(parents=True, exist_ok=True)

        benchmark_run_id = self._build_benchmark_run_id(config)
        self._active_storage = self._create_benchmark_storage()
        self._active_run_id = benchmark_run_id

        try:
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
                    print(
                        f"✓ {result.outcome} ({result.certification_verdict}, "
                        f"{result.wall_time_ms:.1f}ms, profile={result.family_profile})"
                    )

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

            # Persistir resumen agregado en DB dedicada de storage
            self.persist_benchmark_summary(
                summary=summary,
                results=results,
                config=config,
                benchmark_run_id=benchmark_run_id,
            )

            return summary
        finally:
            if self._active_storage is not None:
                self._active_storage.close()
            self._active_storage = None
            self._active_run_id = None

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
        result.reasoning_mode = config.reasoning_mode
        result.family_profile = config.family_profile
        result.regime_label = config.regime_label

        # Iniciar timer
        t0 = time.perf_counter()

        storage = self._active_storage
        owns_storage = False
        if storage is None:
            # Fallback para compatibilidad si run_single_episode se invoca en aislamiento.
            storage = self._create_benchmark_storage()
            owns_storage = True

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
                'reasoning_mode': config.reasoning_mode,
                'family_profile': config.family_profile,
                'regime_label': config.regime_label,
                'reasoning_max_steps': config.reasoning_max_steps,
                'closure_profile': config.closure_profile,
            }

            env_updates = {
                'RNFE_REASONING_MODE': config.reasoning_mode,
                'RNFE_REASONING_FAMILY_PROFILE': config.family_profile,
                'RNFE_REASONING_REGIME_HINT': config.regime_label,
                'RNFE_REASONING_MAX_STEPS': (
                    str(config.reasoning_max_steps) if config.reasoning_max_steps is not None else None
                ),
            }
            with _temporary_reasoning_env(env_updates):
                # Crear runner SIN max_steps (no existe en el contrato)
                closure_profile = config.closure_profile
                if not closure_profile:
                    profile_name = (config.family_profile or "").strip().lower()
                    closure_profile = "baseline_fixed" if profile_name in {"", "core_only"} else "adaptive_min"
                runner = ScenarioEpisodeRunner(
                    scenario=scenario,
                    storage=storage,
                    run_id=self._active_run_id or f"bench-{episode_id[:8]}",
                    closure_profile=closure_profile,
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
            result.reasoning_mode = adapted_result.get('reasoning_mode') or result.reasoning_mode
            result.family_profile = adapted_result.get('family_profile') or result.family_profile
            result.regime_label = adapted_result.get('regime_label') or result.regime_label

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
            if owns_storage:
                storage.close()

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
            def _to_nonnegative(value: Any) -> float:
                try:
                    return max(float(value), 0.0)
                except (TypeError, ValueError):
                    return 0.0

            ivc_r_input = {
                **result.to_dict(),
                **result.metrics,
                'cierre_rate': 1.0 if result.outcome == 'success' else 0.0,
                'continuity_score': _to_nonnegative(result.viability_margin),
                # intervention_precision puede ser negativa si la intervención fue perjudicial.
                # Para IVC-R se acota a dominio no negativo para evitar log(<0).
                'intervention_precision': _to_nonnegative(result.metrics.get('intervention_precision')),
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

            # Grupo 6: Señales family-sensitive
            final_metrics = {
                'ivc_r': result.metrics.get('ivc_r', 0.0),
                'intervention_precision': result.metrics.get('intervention_precision', 0.0),
                'viability_margin': result.viability_margin,
                'reasoning_trace_length': result.reasoning_trace_length,
                'success_rate': 1.0 if result.outcome == 'success' else 0.0,
                'spatial_information_usage': result.metrics.get('spatial_information_usage', 0.0),
            }
            family_bundle = build_family_sensitive_bundle(
                reasoning_sequence=adapted_payload.get('reasoning_sequence', []),
                reasoning_trace=adapted_payload.get('reasoning_trace', []),
                profile_name=result.family_profile,
                mode=result.reasoning_mode or config.reasoning_mode or 'fixed',
                final_metrics=final_metrics,
                proposed_sequence=adapted_payload.get('proposed_sequence', []),
                validated_sequence=adapted_payload.get('validated_sequence', []),
                sequence_validation=adapted_payload.get('sequence_validation', {}),
            )
            result.family_sensitive = family_bundle
            result.metrics['family_mix_entropy'] = float(family_bundle.get('family_mix_entropy', 0.0))
            result.metrics['family_optional_count'] = int(family_bundle.get('family_optional_count', 0))
            result.metrics['family_optional_used_flag'] = 1.0 if family_bundle.get('family_optional_used_flag') else 0.0

        return result

    def persist_results(self, results: List[EpisodeResult], output_dir: Path):
        """Persiste resultados en formato JSONL.

        Args:
            results: Lista de resultados de episodios.
            output_dir: Directorio de salida.
        """
        episodes_file = output_dir / "episodes.jsonl"

        with open(episodes_file, 'w', encoding='utf-8') as f:
            for result in results:
                result_dict = result.to_dict()
                # No incluir runtime_payload completo (muy pesado)
                result_dict.pop('runtime_payload', None)
                f.write(json.dumps(result_dict, default=str, ensure_ascii=True) + '\n')

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

        # Calcular promedios de métricas en episodios no-error para comparabilidad entre perfiles.
        valid_results = [r for r in results if r.outcome != 'error']
        non_error_rows = [r.to_dict() for r in results if r.outcome != 'error']
        metric_keys = [
            'intervention_precision',
            'proposition_diversity',
            'spatial_information_usage',
            'wall_time_ms',
            'artifact_size_bytes',
            'reasoning_trace_length',
            'ivc_r',
            'family_mix_entropy',
        ]

        if valid_results:
            avg_metrics = {}

            for key in metric_keys:
                values = []
                for r in valid_results:
                    if key == 'wall_time_ms':
                        values.append(r.wall_time_ms)
                    elif key == 'artifact_size_bytes':
                        if r.artifact_size_bytes is not None:
                            values.append(r.artifact_size_bytes)
                    elif key == 'reasoning_trace_length':
                        if r.reasoning_trace_length is not None:
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
            viability_values = [r.viability_margin for r in valid_results if r.viability_margin is not None]
            avg_metrics['viability_margin'] = sum(viability_values) / len(viability_values) if viability_values else 0.0
        else:
            avg_metrics = {key: 0.0 for key in metric_keys}
            avg_metrics['viability_margin'] = 0.0

        # Agregación family-sensitive
        family_activation_counts = aggregate_family_activation_counts(non_error_rows)
        family_contribution_proxy = aggregate_family_dict_metric(non_error_rows, 'family_contribution_proxy')
        family_delta_ivc_r = aggregate_family_dict_metric(non_error_rows, 'family_delta_ivc_r')
        family_delta_intervention_precision = aggregate_family_dict_metric(
            non_error_rows,
            'family_delta_intervention_precision',
        )
        family_delta_viability_margin = aggregate_family_dict_metric(non_error_rows, 'family_delta_viability_margin')
        family_delta_reasoning_trace_length = aggregate_family_dict_metric(
            non_error_rows,
            'family_delta_reasoning_trace_length',
        )
        family_delta_success_rate = aggregate_family_dict_metric(non_error_rows, 'family_delta_success_rate')
        family_delta_spatial_usage = aggregate_family_dict_metric(
            non_error_rows,
            'family_delta_spatial_information_usage',
        )
        optional_usage_rate = 0.0
        if non_error_rows:
            optional_usage_rate = (
                sum(1.0 for row in non_error_rows if row.get('family_optional_used_flag'))
                / len(non_error_rows)
            )
        backbone_floor_satisfied_rate = (
            sum(1.0 for row in non_error_rows if row.get('backbone_floor_satisfied_flag')) / len(non_error_rows)
            if non_error_rows
            else 0.0
        )
        sequence_validation_fail_rate = (
            sum(1.0 for row in non_error_rows if row.get('sequence_validation_fail_flag')) / len(non_error_rows)
            if non_error_rows
            else 0.0
        )
        fallback_to_safe_sequence_rate = (
            sum(1.0 for row in non_error_rows if row.get('fallback_to_safe_sequence_flag')) / len(non_error_rows)
            if non_error_rows
            else 0.0
        )
        optional_displacement_rate = (
            sum(1.0 for row in non_error_rows if row.get('optional_displacement_flag')) / len(non_error_rows)
            if non_error_rows
            else 0.0
        )
        closure_break_rate = (
            sum(1.0 for row in non_error_rows if row.get('closure_break_flag')) / len(non_error_rows)
            if non_error_rows
            else 0.0
        )

        # Agregación de fallos
        from .failure_taxonomy import aggregate_failure_distribution

        failure_dist = aggregate_failure_distribution([r.to_dict() for r in results])

        summary = {
            'scenario': config.scenario_name,
            'reasoning_mode': config.reasoning_mode,
            'family_profile': config.family_profile,
            'regime_label': config.regime_label,
            'total_episodes': total,
            'successful': successful,
            'failed': failed,
            'errors': errors,
            'success_rate': successful / total if total > 0 else 0.0,
            'avg_metrics': avg_metrics,
            'failure_distribution': failure_dist,
            'optional_family_usage_rate': optional_usage_rate,
            'backbone_floor_satisfied_rate': backbone_floor_satisfied_rate,
            'sequence_validation_fail_rate': sequence_validation_fail_rate,
            'fallback_to_safe_sequence_rate': fallback_to_safe_sequence_rate,
            'optional_displacement_rate': optional_displacement_rate,
            'closure_break_rate': closure_break_rate,
            'family_specific_activation_counts': family_activation_counts,
            'family_impact_aggregates': {
                'family_contribution_proxy': family_contribution_proxy,
                'family_delta_ivc_r': family_delta_ivc_r,
                'family_delta_intervention_precision': family_delta_intervention_precision,
                'family_delta_viability_margin': family_delta_viability_margin,
                'family_delta_reasoning_trace_length': family_delta_reasoning_trace_length,
                'family_delta_success_rate': family_delta_success_rate,
                'family_delta_spatial_information_usage': family_delta_spatial_usage,
            },
        }

        # Guardar resumen
        summary_file = config.output_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str, ensure_ascii=True)

        print(f"📊 Resumen guardado en: {summary_file}")

        return summary

    def persist_benchmark_summary(
        self,
        *,
        summary: Dict[str, Any],
        results: List[EpisodeResult],
        config: BenchmarkConfig,
        benchmark_run_id: str,
    ) -> None:
        """Persiste resumen agregado en reality_bench_runs."""
        if self._active_storage is None:
            return

        collapse_count = sum(1 for r in results if r.is_viable is False)
        summary_for_db = {
            **summary,
            'proxy_mapping': {
                'closure_rate': 'success_rate',
                'continuity_mean': 'viability_margin',
            },
            'benchmark_run_id': benchmark_run_id,
            'runtime_contract_mode': 'runtime_to_benchmark',
        }

        self._active_storage.write_reality_bench_run(
            bench_run_id=benchmark_run_id,
            run_id=benchmark_run_id,
            total_episodes=int(summary.get('total_episodes', 0)),
            closure_rate=float(summary.get('success_rate', 0.0)),
            continuity_mean=float(summary.get('avg_metrics', {}).get('viability_margin', 0.0)),
            collapse_count=collapse_count,
            gate_profile=config.scenario_name,
            passed=bool(summary.get('errors', 0) == 0 and summary.get('successful', 0) > 0),
            summary=summary_for_db,
        )

    def _build_benchmark_run_id(self, config: BenchmarkConfig) -> str:
        if config.run_id:
            return config.run_id

        scenario_token = config.scenario_name.replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        return f"bench-{scenario_token}-{timestamp}-{suffix}"

    def _create_benchmark_storage(self):
        """Crea storage persistente para ejecución de benchmark."""
        config = self.storage_config or StorageConfig.from_env()
        return StorageFactory.create_facade(config)
