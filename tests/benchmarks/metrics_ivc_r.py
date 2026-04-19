"""Grupo 4: IVC-R (Índice de Valor Cognitivo Robusto).

Métrica compuesta que evalúa valor cognitivo neto considerando:
- Calidad de cierre (cierre_rate)
- Continuidad (continuity_score)
- Precisión de intervenciones
- Diversidad proposicional
- Costo temporal

Usa log-espacio para evitar división por cero y manejar valores extremos.
"""

from typing import Dict, Any, List
import math
import random


# Pesos por defecto para componentes IVC-R
DEFAULT_WEIGHTS = {
    'cierre': 0.35,
    'continuidad': 0.25,
    'precision': 0.20,
    'diversity': 0.10,
    'costo': 0.10,
}

# Regularización para evitar log(0)
EPSILON = 1e-6


def compute_ivc_r(
    cierre_rate: float,
    continuity_score: float,
    intervention_precision: float,
    proposition_diversity: float,
    episode_wall_time_ms: float,
    weights: Dict[str, float] = None,
) -> Dict[str, Any]:
    """Calcula IVC-R en log-espacio con descomposición por componentes.

    Formula:
    IVC-R = exp(
        w_cierre * log(cierre_rate + ε) +
        w_continuidad * log(continuity_score + ε) +
        w_precision * log(intervention_precision + ε) +
        w_diversity * log(proposition_diversity + ε) -
        w_costo * log(episode_wall_time_ms + ε)
    )

    Args:
        cierre_rate: Tasa de cierre [0, 1].
        continuity_score: Score de continuidad [0, 1].
        intervention_precision: Precisión de intervenciones [0, 1].
        proposition_diversity: Entropía de proposiciones [0, ∞].
        episode_wall_time_ms: Tiempo de ejecución en ms.
        weights: Pesos personalizados (opcional).

    Returns:
        Diccionario con IVC-R y descomposición.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Normalizar proposition_diversity a [0, 1] para comparabilidad
    # Asumimos max_diversity ≈ 4.0 (log2(16) para ~16 props únicas)
    max_diversity = 4.0
    normalized_diversity = min(proposition_diversity / max_diversity, 1.0)

    # Normalizar wall_time a escala comparable (baseline ~50ms)
    baseline_time = 50.0
    normalized_time = episode_wall_time_ms / baseline_time

    # Calcular componentes en log-espacio
    log_components = {
        'cierre': weights['cierre'] * math.log(cierre_rate + EPSILON),
        'continuidad': weights['continuidad'] * math.log(continuity_score + EPSILON),
        'precision': weights['precision'] * math.log(intervention_precision + EPSILON),
        'diversity': weights['diversity'] * math.log(normalized_diversity + EPSILON),
        'costo': -weights['costo'] * math.log(normalized_time + EPSILON),  # Negativo
    }

    # Sumar componentes
    ivc_r_log = sum(log_components.values())

    # Exponenciar para obtener IVC-R final
    ivc_r = math.exp(ivc_r_log)

    # Calcular valores normalizados de cada componente
    raw_components = {
        'cierre': cierre_rate,
        'continuidad': continuity_score,
        'precision': intervention_precision,
        'diversity': normalized_diversity,
        'costo': normalized_time,
    }

    return {
        'ivc_r': ivc_r,
        'ivc_r_log': ivc_r_log,
        'components': raw_components,
        'component_contributions': log_components,
    }


def compute_ivc_r_from_episode(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula IVC-R desde un episodio completo.

    Args:
        episode: Diccionario con métricas ya calculadas.

    Returns:
        Resultado de IVC-R con descomposición.
    """
    # Extraer métricas necesarias
    cierre_rate = episode.get('cierre_rate', 0.0)
    continuity_score = episode.get('continuity_score', 0.0)
    intervention_precision = episode.get('intervention_precision', 0.0)
    proposition_diversity = episode.get('proposition_diversity', 0.0)
    wall_time_ms = episode.get('wall_time_ms', 50.0)

    return compute_ivc_r(
        cierre_rate=cierre_rate,
        continuity_score=continuity_score,
        intervention_precision=intervention_precision,
        proposition_diversity=proposition_diversity,
        episode_wall_time_ms=wall_time_ms,
    )


def compute_ivc_r_with_confidence_intervals(
    episodes: List[Dict[str, Any]],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """Calcula IVC-R con intervalos de confianza via bootstrap.

    Args:
        episodes: Lista de episodios con métricas.
        n_bootstrap: Número de resamples bootstrap.
        confidence_level: Nivel de confianza (default 95%).

    Returns:
        Diccionario con media, IC inferior/superior, std.
    """
    if not episodes:
        return {
            'ivc_r_mean': 0.0,
            'ivc_r_ci_lower': 0.0,
            'ivc_r_ci_upper': 0.0,
            'ivc_r_std': 0.0,
        }

    # Calcular IVC-R para cada episodio original
    ivc_r_values = []
    for episode in episodes:
        result = compute_ivc_r_from_episode(episode)
        ivc_r_values.append(result['ivc_r'])

    # Bootstrap resampling
    bootstrap_means = []

    for _ in range(n_bootstrap):
        # Resample con reemplazo
        sample = random.choices(ivc_r_values, k=len(ivc_r_values))
        bootstrap_means.append(sum(sample) / len(sample))

    # Calcular percentiles para IC
    bootstrap_means.sort()
    alpha = 1.0 - confidence_level
    lower_idx = int(n_bootstrap * (alpha / 2))
    upper_idx = int(n_bootstrap * (1 - alpha / 2))

    ivc_r_mean = sum(ivc_r_values) / len(ivc_r_values)
    ivc_r_ci_lower = bootstrap_means[lower_idx]
    ivc_r_ci_upper = bootstrap_means[upper_idx]

    # Calcular std
    variance = sum((x - ivc_r_mean) ** 2 for x in ivc_r_values) / len(ivc_r_values)
    ivc_r_std = math.sqrt(variance)

    return {
        'ivc_r_mean': ivc_r_mean,
        'ivc_r_ci_lower': ivc_r_ci_lower,
        'ivc_r_ci_upper': ivc_r_ci_upper,
        'ivc_r_std': ivc_r_std,
    }
