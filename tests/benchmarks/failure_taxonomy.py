"""Grupo 5: Taxonomía de Fallos.

CORREGIDO: Adaptado al contrato real del runtime (single-step).

El runtime retorna certification_verdict que determina success/failure:
- 'passed' o 'certified' → success
- otros valores → failure

Categorías primarias de fallos (basadas en señales reales):
- ERROR: exception durante ejecución
- CERTIFICATION_FAILED: certificación no pasó
- VIABILITY_FAILED: no viable según viability_assessment
- BOTH_FAILED: certificación y viabilidad fallaron

Causas secundarias (basadas en metadata observable):
- HIGH_INITIAL_TEMP: temperatura inicial > 0.95
- WEAK_COOLING: cooling_effect < 0.05
- TIGHT_THRESHOLD: margen entre initial_temp y alarm_threshold < 0.05
- SPATIAL_COMPLEXITY: escenario 5x5 con topología compleja

NO clasificamos (requieren multi-step trace):
- TIMEOUT: no hay concepto de timeout en single-step
- ALARM_PERSISTENT: no hay trace de alarmas
- OSCILLATION: no hay trace multi-step
- SCHEDULER_OVERHEAD: no hay timing separado del scheduler
"""

from typing import Dict, Any, List, Optional


# Categorías primarias de fallos
class FailureCategory:
    ERROR = 'error'
    CERTIFICATION_FAILED = 'certification_failed'
    VIABILITY_FAILED = 'viability_failed'
    BOTH_FAILED = 'both_failed'


# Causas secundarias de fallos
class FailureCause:
    HIGH_INITIAL_TEMP = 'high_initial_temp'
    WEAK_COOLING = 'weak_cooling'
    TIGHT_THRESHOLD = 'tight_threshold'
    SPATIAL_COMPLEXITY = 'spatial_complexity'


def classify_failure_primary(episode: Dict[str, Any]) -> Optional[str]:
    """Clasifica la categoría primaria de fallo de un episodio.

    CORREGIDO: Usa señales reales del runtime.

    Args:
        episode: Diccionario del episodio con certification_verdict y is_viable.

    Returns:
        Categoría de fallo o None si episodio pasó certificación.
    """
    # Verificar outcome directo
    outcome = episode.get('outcome')

    # ERROR explícito
    if outcome == 'error' or episode.get('error'):
        return FailureCategory.ERROR

    # SUCCESS: certificación pasó
    if outcome == 'success':
        return None

    # Si no es success, determinar por qué falló
    cert_verdict = episode.get('certification_verdict')
    is_viable = episode.get('is_viable', True)

    # Certificación falló
    cert_failed = cert_verdict not in ['passed', 'certified']

    # Viabilidad falló
    viability_failed = not is_viable

    if cert_failed and viability_failed:
        return FailureCategory.BOTH_FAILED
    elif viability_failed:
        return FailureCategory.VIABILITY_FAILED
    elif cert_failed:
        return FailureCategory.CERTIFICATION_FAILED

    # Si outcome es 'failure' pero no podemos clasificar, usar CERTIFICATION_FAILED
    if outcome == 'failure':
        return FailureCategory.CERTIFICATION_FAILED

    return None


def classify_failure_secondary(episode: Dict[str, Any]) -> List[str]:
    """Clasifica las causas secundarias de fallo.

    CORREGIDO: Usa solo metadata observable.

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

    # Spatial complexity (solo para 5x5)
    grid_size = metadata.get('grid_size', 1)
    topology = metadata.get('topology')

    if grid_size == 5 and topology not in ['uniform', None]:
        # Topologías heterogéneas pueden aumentar complejidad
        if topology in ['hotspot_center', 'gradient_ns', 'gradient_ew', 'checkerboard']:
            causes.append(FailureCause.SPATIAL_COMPLEXITY)

    return causes


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
