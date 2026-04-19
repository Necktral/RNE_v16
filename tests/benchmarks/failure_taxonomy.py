"""Grupo 5: Taxonomía de Fallos.

Clasificación sistemática de fallos en episodios:
- Categorías primarias: timeout, error, counterfactual_failed, etc.
- Causas secundarias: high_initial_temp, weak_cooling, spatial_complexity, etc.
"""

from typing import Dict, Any, List, Optional


# Categorías primarias de fallos
class FailureCategory:
    TIMEOUT = 'timeout'
    ERROR = 'error'
    COUNTERFACTUAL_FAILED = 'counterfactual_failed'
    BOTH_FAILED = 'both_failed'
    ALARM_PERSISTENT = 'alarm_persistent'
    OSCILLATION = 'oscillation'


# Causas secundarias de fallos
class FailureCause:
    HIGH_INITIAL_TEMP = 'high_initial_temp'
    WEAK_COOLING = 'weak_cooling'
    TIGHT_THRESHOLD = 'tight_threshold'
    SCHEDULER_OVERHEAD = 'scheduler_overhead'
    SPATIAL_COMPLEXITY = 'spatial_complexity'


def classify_failure_primary(episode: Dict[str, Any]) -> Optional[str]:
    """Clasifica la categoría primaria de fallo de un episodio.

    Args:
        episode: Diccionario del episodio.

    Returns:
        Categoría de fallo o None si episodio cerró exitosamente.
    """
    # Verificar si el episodio cerró exitosamente
    outcome = episode.get('outcome')
    if outcome == 'success':
        cierre_rate = episode.get('cierre_rate', 0.0)
        if cierre_rate >= 1.0:
            return None

    # Verificar error explícito
    if episode.get('error') or outcome == 'error':
        return FailureCategory.ERROR

    # Verificar timeout
    max_steps = episode.get('max_steps', 50)
    trace_length = episode.get('trace_length', 0)

    if trace_length >= max_steps:
        # Verificar si es por alarma persistente
        if _has_persistent_alarm(episode):
            return FailureCategory.ALARM_PERSISTENT
        else:
            return FailureCategory.TIMEOUT

    # Verificar oscillation (ciclo detectado)
    if _has_oscillation(episode):
        return FailureCategory.OSCILLATION

    # Verificar fallos de contrafactual
    cf = episode.get('counterfactual')
    factual_closed = episode.get('cierre_rate', 0.0) >= 1.0

    if cf is not None:
        cf_closed = cf.get('closed', False)

        if factual_closed and not cf_closed:
            return FailureCategory.COUNTERFACTUAL_FAILED
        elif not factual_closed and not cf_closed:
            return FailureCategory.BOTH_FAILED

    # Si llegamos aquí y outcome != success, clasificar como timeout genérico
    if outcome != 'success':
        return FailureCategory.TIMEOUT

    return None


def classify_failure_secondary(episode: Dict[str, Any]) -> List[str]:
    """Clasifica las causas secundarias de fallo.

    Args:
        episode: Diccionario del episodio.

    Returns:
        Lista de causas secundarias (puede estar vacía).
    """
    causes = []

    metadata = episode.get('metadata', {})

    # High initial temperature
    initial_temp = metadata.get('initial_temperature', 0.0)
    if initial_temp > 0.95:
        causes.append(FailureCause.HIGH_INITIAL_TEMP)

    # Weak cooling
    cooling_effect = metadata.get('cooling_effect', 0.07)
    if cooling_effect < 0.05:
        causes.append(FailureCause.WEAK_COOLING)

    # Tight threshold
    alarm_threshold = metadata.get('alarm_threshold', 0.85)
    diff = alarm_threshold - initial_temp
    if 0 < diff < 0.05:
        causes.append(FailureCause.TIGHT_THRESHOLD)

    # Scheduler overhead
    wall_time = episode.get('wall_time_ms', 0.0)
    scheduler_time = episode.get('scheduler_cpu_time_ms', 0.0)

    if wall_time > 0 and (scheduler_time / wall_time) > 0.5:
        causes.append(FailureCause.SCHEDULER_OVERHEAD)

    # Spatial complexity (solo para 5x5)
    spatial_coherence = episode.get('spatial_coherence_index')
    if spatial_coherence is not None and spatial_coherence < 0.3:
        causes.append(FailureCause.SPATIAL_COMPLEXITY)

    return causes


def _has_persistent_alarm(episode: Dict[str, Any]) -> bool:
    """Verifica si la alarma nunca se desactivó en X pasos consecutivos.

    Args:
        episode: Diccionario del episodio.

    Returns:
        True si alarma persistente detectada.
    """
    trace = episode.get('trace', [])

    if len(trace) < 10:
        return False

    # Verificar últimos 10 pasos
    consecutive_alarm = 0
    threshold = 10

    for step in trace[-threshold:]:
        obs = step.get('observation', {})
        alarm = obs.get('alarm', False)

        if alarm:
            consecutive_alarm += 1
        else:
            consecutive_alarm = 0

    return consecutive_alarm >= threshold


def _has_oscillation(episode: Dict[str, Any]) -> bool:
    """Verifica si hay ciclo detectado (estado repetido >3 veces).

    Args:
        episode: Diccionario del episodio.

    Returns:
        True si oscillation detectada.
    """
    trace = episode.get('trace', [])

    if len(trace) < 10:
        return False

    # Usar hash simplificado del estado
    state_hashes = []

    for step in trace:
        obs = step.get('observation', {})
        state = obs.get('state', {})

        # Crear hash basado en temperatura media y alarma
        temp = state.get('global_temp_mean', state.get('temperature', 0.0))
        alarm = obs.get('alarm', False)

        # Redondear temperatura para detectar oscilación
        temp_rounded = round(temp, 2)
        state_hash = (temp_rounded, alarm)

        state_hashes.append(state_hash)

    # Contar repeticiones
    from collections import Counter
    counts = Counter(state_hashes)

    # Si algún estado aparece >3 veces, hay oscilación
    max_count = max(counts.values()) if counts else 0

    return max_count > 3


def classify_episode_failures(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Clasifica fallos completos de un episodio.

    Args:
        episode: Diccionario del episodio.

    Returns:
        Diccionario con failure_primary y failure_secondary.
    """
    return {
        'failure_primary': classify_failure_primary(episode),
        'failure_secondary': classify_failure_secondary(episode),
    }


def aggregate_failure_distribution(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Agrega distribución de fallos de múltiples episodios.

    Args:
        episodes: Lista de episodios con clasificación de fallos.

    Returns:
        Diccionario con estadísticas agregadas.
    """
    total_episodes = len(episodes)
    total_failures = 0

    primary_counts = {}
    secondary_counts = {}

    for episode in episodes:
        primary = episode.get('failure_primary')

        if primary is not None:
            total_failures += 1
            primary_counts[primary] = primary_counts.get(primary, 0) + 1

        secondary_list = episode.get('failure_secondary', [])
        for cause in secondary_list:
            secondary_counts[cause] = secondary_counts.get(cause, 0) + 1

    return {
        'total_episodes': total_episodes,
        'total_failures': total_failures,
        'failure_rate': total_failures / total_episodes if total_episodes > 0 else 0.0,
        'failure_distribution': primary_counts,
        'secondary_causes': secondary_counts,
    }
