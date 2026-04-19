"""Grupo 3: Métricas de Costo Operativo.

Métricas que miden el costo computacional y de recursos:
- episode_wall_time_ms: Tiempo real de ejecución
- counterfactual_overhead_ratio: Ratio tiempo CF vs factual
- memory_pressure_mb: Memoria incremental usada
- scheduler_cpu_time_ms: Tiempo de CPU del scheduler
- artifact_size_bytes: Tamaño del artifact serializado
- trace_length: Número de pasos del episodio
"""

from typing import Dict, Any
import json
import sys


def compute_episode_wall_time_ms(episode: Dict[str, Any]) -> float:
    """Extrae tiempo real de ejecución del episodio.

    Args:
        episode: Diccionario con metadata de timing.

    Returns:
        Tiempo en milisegundos.
    """
    # El tiempo puede venir en diferentes formatos dependiendo de cómo se capture
    wall_time_ms = episode.get('wall_time_ms')

    if wall_time_ms is not None:
        return float(wall_time_ms)

    # Intentar calcular desde timestamps si están disponibles
    start_time = episode.get('start_time')
    end_time = episode.get('end_time')

    if start_time is not None and end_time is not None:
        return (end_time - start_time) * 1000.0

    # Fallback: retornar 0.0 si no hay información
    return 0.0


def compute_counterfactual_overhead_ratio(episode: Dict[str, Any]) -> float:
    """Calcula ratio de tiempo contrafactual vs factual.

    Formula: ratio = counterfactual_time_ms / factual_time_ms

    Args:
        episode: Diccionario con información de timing.

    Returns:
        Ratio en [1.0, ∞]. Retorna 0.0 si no hay CF o factual_time es 0.
    """
    factual_time = episode.get('factual_time_ms')
    cf_time = episode.get('counterfactual_time_ms')

    # Verificar que exista información de contrafactual
    if 'counterfactual' not in episode or episode.get('counterfactual') is None:
        return 0.0

    # Si no hay tiempos específicos, intentar inferir del wall_time total
    if factual_time is None or cf_time is None:
        # Fallback: asumir que wall_time incluye ambos
        # CF típicamente toma 1.5-2x del factual
        return 0.0

    if factual_time == 0:
        return 0.0

    ratio = cf_time / factual_time

    return ratio


def compute_memory_pressure_mb(episode: Dict[str, Any]) -> float:
    """Extrae memoria incremental usada durante episodio.

    Args:
        episode: Diccionario con metadata de memoria.

    Returns:
        Memoria en megabytes.
    """
    memory_mb = episode.get('memory_pressure_mb')

    if memory_mb is not None:
        return float(memory_mb)

    # Fallback: calcular desde delta si está disponible
    memory_delta_bytes = episode.get('memory_delta_bytes')

    if memory_delta_bytes is not None:
        return memory_delta_bytes / (1024 * 1024)

    return 0.0


def compute_scheduler_cpu_time_ms(episode: Dict[str, Any]) -> float:
    """Extrae tiempo de CPU dedicado al scheduler.

    Args:
        episode: Diccionario con metadata de timing del scheduler.

    Returns:
        Tiempo en milisegundos.
    """
    scheduler_time = episode.get('scheduler_cpu_time_ms')

    if scheduler_time is not None:
        return float(scheduler_time)

    # Fallback: si hay trace con timing por step del scheduler
    trace = episode.get('trace', [])
    total_scheduler_time = 0.0

    for step in trace:
        step_scheduler_time = step.get('scheduler_time_ms', 0.0)
        total_scheduler_time += step_scheduler_time

    return total_scheduler_time


def compute_artifact_size_bytes(episode: Dict[str, Any]) -> int:
    """Calcula tamaño del artifact serializado completo.

    Args:
        episode: Diccionario completo del episodio.

    Returns:
        Tamaño en bytes.
    """
    artifact_size = episode.get('artifact_size_bytes')

    if artifact_size is not None:
        return int(artifact_size)

    # Calcular serializando el episodio completo
    try:
        serialized = json.dumps(episode, default=str)
        size = len(serialized.encode('utf-8'))
        return size
    except Exception:
        # Fallback: estimar desde sys.getsizeof (menos preciso)
        return sys.getsizeof(episode)


def compute_trace_length(episode: Dict[str, Any]) -> int:
    """Extrae número de pasos del episodio.

    Args:
        episode: Diccionario con trace.

    Returns:
        Número de pasos.
    """
    # Verificar si ya está calculado
    trace_length = episode.get('trace_length')

    if trace_length is not None:
        return int(trace_length)

    # Calcular desde trace
    trace = episode.get('trace', [])
    return len(trace)


def compute_all_operational_cost_metrics(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula todas las métricas de costo operativo para un episodio.

    Args:
        episode: Diccionario completo del episodio.

    Returns:
        Diccionario con todas las métricas del Grupo 3.
    """
    return {
        'wall_time_ms': compute_episode_wall_time_ms(episode),
        'cf_overhead_ratio': compute_counterfactual_overhead_ratio(episode),
        'memory_pressure_mb': compute_memory_pressure_mb(episode),
        'scheduler_cpu_time_ms': compute_scheduler_cpu_time_ms(episode),
        'artifact_size_bytes': compute_artifact_size_bytes(episode),
        'trace_length': compute_trace_length(episode),
    }
