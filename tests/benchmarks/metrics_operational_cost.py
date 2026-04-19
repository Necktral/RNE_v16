"""Grupo 3: Métricas de Costo Operativo.

CORREGIDO: Usa solo señales observables reales del runtime.

Métricas que miden el costo computacional y de recursos:
- episode_wall_time_ms: Tiempo real de ejecución (observable directo)
- artifact_size_bytes: Tamaño del artifact serializado (observable directo)
- reasoning_trace_length: Número de pasos del scheduler (observable directo)

NO instrumentamos:
- counterfactual_overhead_ratio: No existe timing por factual vs CF
- memory_pressure_mb: No está instrumentado en runtime
- scheduler_cpu_time_ms: No existe timing separado del scheduler
"""

from typing import Dict, Any


def compute_episode_wall_time_ms(episode: Dict[str, Any]) -> float:
    """Extrae tiempo real de ejecución del episodio.

    CORREGIDO: Lee directamente el campo observado.

    Args:
        episode: Diccionario adaptado con wall_time_ms.

    Returns:
        Tiempo en milisegundos.
    """
    wall_time_ms = episode.get('wall_time_ms')

    if wall_time_ms is not None:
        return float(wall_time_ms)

    # Si no está presente, retornar 0.0
    return 0.0


def compute_artifact_size_bytes(episode: Dict[str, Any]) -> int:
    """Extrae tamaño del artifact serializado.

    CORREGIDO: Lee directamente el campo observado.

    Args:
        episode: Diccionario adaptado con artifact_size_bytes.

    Returns:
        Tamaño en bytes.
    """
    artifact_size = episode.get('artifact_size_bytes')

    if artifact_size is not None:
        return int(artifact_size)

    return 0


def compute_reasoning_trace_length(episode: Dict[str, Any]) -> int:
    """Extrae longitud del reasoning trace del scheduler.

    CORREGIDO: Lee directamente el campo observado.

    Args:
        episode: Diccionario adaptado con reasoning_trace_length.

    Returns:
        Número de pasos del scheduler.
    """
    trace_length = episode.get('reasoning_trace_length')

    if trace_length is not None:
        return int(trace_length)

    # Fallback: intentar calcular desde reasoning_sequence si está presente
    reasoning_seq = episode.get('reasoning_sequence', [])
    if reasoning_seq:
        return len(reasoning_seq)

    return 0


def compute_all_operational_cost_metrics(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula todas las métricas de costo operativo para un episodio.

    CORREGIDO: Solo métricas realmente observadas.

    Args:
        episode: Diccionario adaptado del episodio.

    Returns:
        Diccionario con métricas observables del Grupo 3.
    """
    return {
        'wall_time_ms': compute_episode_wall_time_ms(episode),
        'artifact_size_bytes': compute_artifact_size_bytes(episode),
        'reasoning_trace_length': compute_reasoning_trace_length(episode),
    }
