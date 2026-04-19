"""Grupo 2: Métricas de Calidad Cognitiva.

CORREGIDO: Adaptado al contrato real del runtime.

El runtime NO retorna trace de múltiples pasos del mundo.
Cada episodio es UN SOLO paso cognitivo con:
- observation: estado inicial
- updated_world: estado final post-intervención
- counterfactual_world: mundo contrafactual (si existe)
- organism_trajectory: trayectoria con puntos de viabilidad
- reasoning_sequence: pasos del scheduler (NO pasos del mundo)

Métricas disponibles:
- intervention_precision: Eficacia de intervención (delta temperatura)
- proposition_diversity: Entropía de proposiciones en observación
- spatial_information_usage: Uso de proposiciones espaciales (solo 5x5)

Métricas NO disponibles (requieren multi-step trace):
- factual_counterfactual_divergence: No hay trace multi-step
- world_level_transitions: Solo un paso, no hay transiciones
- spatial_coherence_index: Solo un snapshot, no hay evolución temporal
"""

from typing import Dict, Any, Optional
import math
from collections import Counter


def compute_intervention_precision(episode: Dict[str, Any]) -> Optional[float]:
    """Calcula precisión de intervención (beneficio térmico obtenido).

    CORREGIDO: Compara observation vs updated_world (un solo paso).

    Formula: precision = (temp_initial - temp_final) / temp_initial
    Si temp_final < temp_initial → precisión positiva (beneficioso)

    Args:
        episode: Diccionario adaptado con observation y updated_world.

    Returns:
        Precisión en [-∞, 1.0] donde:
        - > 0: intervención beneficiosa (redujo temperatura)
        - = 0: sin efecto
        - < 0: intervención perjudicial (aumentó temperatura)
        - None: si no hay datos
    """
    observation = episode.get('observation')
    updated_world = episode.get('updated_world')

    if not observation or not updated_world:
        return None

    # Obtener temperatura inicial y final
    # Para 1x1: observation['temperature']
    # Para 5x5: observation['world_level'] (global_temp_mean)
    temp_initial = observation.get('world_level') or observation.get('temperature')
    temp_final = updated_world.get('world_level') or updated_world.get('temperature')

    if temp_initial is None or temp_final is None:
        return None

    if temp_initial == 0.0:
        return None

    # Precisión: proporción de reducción térmica
    precision = (temp_initial - temp_final) / temp_initial

    return precision


def compute_proposition_diversity(episode: Dict[str, Any]) -> float:
    """Calcula entropía de Shannon del conjunto de proposiciones observadas.

    CORREGIDO: Lee proposiciones de observation (snapshot único).

    Formula: diversity = -Σ(p_i * log2(p_i)) donde p_i = freq(prop_i)

    Args:
        episode: Diccionario adaptado con observation.

    Returns:
        Entropía en [0.0, log2(N)] donde N = proposiciones únicas.
    """
    observation = episode.get('observation')

    if not observation:
        return 0.0

    propositions = observation.get('propositions', [])

    if not propositions:
        return 0.0

    # Contar frecuencia de cada proposición
    prop_counts = Counter(propositions)

    # Calcular probabilidades
    total = len(propositions)
    probabilities = [count / total for count in prop_counts.values()]

    # Calcular entropía de Shannon
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def compute_spatial_information_usage(episode: Dict[str, Any]) -> float:
    """Calcula proporción de proposiciones espaciales activas.

    CORREGIDO: Lee proposiciones de observation (snapshot único).

    Esta métrica mide si el sistema usa información espacial
    disponible en escenarios 5x5.

    Args:
        episode: Diccionario adaptado con observation.

    Returns:
        Uso de información espacial en [0.0, 1.0].
    """
    observation = episode.get('observation')

    if not observation:
        return 0.0

    propositions = observation.get('propositions', [])

    if not propositions:
        return 0.0

    # Proposiciones que indican uso de información espacial
    spatial_indicators = [
        'HOTSPOT_DETECTED',
        'MULTI_HOTSPOT',
        'HOTSPOT_CENTRAL',
        'HOTSPOT_PERIPHERAL',
        'HOTSPOT_CRITICAL',
        'THERMAL_GRADIENT',
        'THERMAL_GRADIENT_NS',
        'THERMAL_GRADIENT_EW',
        'THERMAL_GRADIENT_NE_SW',
        'THERMAL_GRADIENT_NW_SE',
        'STRONG_GRADIENT',
        'CONCENTRATED_HEAT',
        'DIFFUSE_HEAT',
        'CRITICAL_ZONE_PRESENT',
        'CRITICAL_ZONE_CENTRAL',
        'CRITICAL_ZONE_PERIPHERAL',
        'CRITICAL_ZONE_EXTENSIVE',
        'REGIONAL_IMBALANCE',
        'QUADRANT_IMBALANCE',
        'UNIFORM_TEMPERATURE',
        'LOW_SPATIAL_ENTROPY',
        'HIGH_SPATIAL_ENTROPY',
    ]

    # Contar proposiciones espaciales
    spatial_count = sum(1 for prop in propositions if prop in spatial_indicators)

    # Ratio de proposiciones espaciales
    return spatial_count / len(propositions)


def compute_all_cognitive_metrics(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula todas las métricas de calidad cognitiva para un episodio.

    CORREGIDO: Solo métricas compatibles con episodio single-step.

    Args:
        episode: Diccionario adaptado del episodio.

    Returns:
        Diccionario con métricas observables del Grupo 2.
    """
    return {
        'intervention_precision': compute_intervention_precision(episode),
        'proposition_diversity': compute_proposition_diversity(episode),
        'spatial_information_usage': compute_spatial_information_usage(episode),
    }
