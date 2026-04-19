"""Grupo 2: Métricas de Calidad Cognitiva.

Métricas que evalúan la calidad de las decisiones y cognición del sistema:
- factual_counterfactual_divergence: Distancia entre factual y mejor contrafactual
- intervention_precision: Proporción de intervenciones beneficiosas
- proposition_diversity: Entropía de proposiciones observadas
- world_level_transitions: Transiciones entre niveles del mundo
- spatial_coherence_index: Correlación espacial entre celdas vecinas (solo 5x5)
- spatial_information_usage: Proporción de decisiones dependientes de topología
"""

from typing import Dict, Any, List, Optional
import math
from collections import Counter


def compute_factual_cf_divergence(episode: Dict[str, Any]) -> Optional[float]:
    """Calcula divergencia entre trayectoria factual y mejor contrafactual.

    Formula: divergence = (steps_factual - steps_counterfactual) / steps_factual

    Args:
        episode: Diccionario con datos del episodio (debe incluir trace).

    Returns:
        Divergencia en [-∞, 1.0] donde:
        - 1.0: factual es óptimo
        - 0.0: ambos iguales
        - <0: factual peor que contrafactual
        - None: si contrafactual no está disponible o falló
    """
    # Verificar que exista información de contrafactual
    if 'counterfactual' not in episode or episode.get('counterfactual') is None:
        return None

    cf = episode['counterfactual']

    # Verificar que contrafactual cerró exitosamente
    if not cf.get('closed', False):
        return None

    factual_steps = episode.get('trace_length', episode.get('steps', 0))
    cf_steps = cf.get('steps', 0)

    if factual_steps == 0:
        return None

    divergence = (factual_steps - cf_steps) / factual_steps

    return divergence


def compute_intervention_precision(episode: Dict[str, Any]) -> float:
    """Calcula precisión de intervenciones (proporción beneficiosa vs perjudicial).

    Clasifica cada intervención como:
    - Beneficiosa: reduce temperatura o distancia a safe zone
    - Neutral: sin cambio significativo (<0.01)
    - Perjudicial: incrementa temperatura o alarma

    Args:
        episode: Diccionario con trace de episodio.

    Returns:
        Precisión en [0.0, 1.0].
    """
    trace = episode.get('trace', [])

    if not trace or len(trace) < 2:
        return 0.0

    beneficial = 0
    neutral = 0
    harmful = 0

    for i in range(len(trace) - 1):
        step = trace[i]
        next_step = trace[i + 1]

        # Verificar si hubo intervención
        if 'intervention' not in step or step['intervention'] is None:
            continue

        # Obtener temperatura antes y después
        temp_before = step.get('state', {}).get('global_temp_mean',
                                                  step.get('state', {}).get('temperature', 0.0))
        temp_after = next_step.get('state', {}).get('global_temp_mean',
                                                      next_step.get('state', {}).get('temperature', 0.0))

        delta = temp_after - temp_before

        # Clasificar intervención
        if abs(delta) < 0.01:
            neutral += 1
        elif delta < 0:  # Temperatura bajó
            beneficial += 1
        else:  # Temperatura subió
            harmful += 1

    total = beneficial + neutral + harmful

    if total == 0:
        return 0.0

    # Considerar neutrales como medio punto
    precision = (beneficial + 0.5 * neutral) / total

    return precision


def compute_proposition_diversity(episode: Dict[str, Any]) -> float:
    """Calcula entropía de Shannon del conjunto de proposiciones observadas.

    Formula: diversity = -Σ(p_i * log2(p_i)) donde p_i = freq(prop_i)

    Args:
        episode: Diccionario con trace de episodio.

    Returns:
        Entropía en [0.0, log2(N)] donde N = proposiciones únicas.
    """
    trace = episode.get('trace', [])

    if not trace:
        return 0.0

    # Contar frecuencia de cada proposición
    prop_counts = Counter()

    for step in trace:
        obs = step.get('observation', {})
        props = obs.get('propositions', [])

        for prop in props:
            prop_counts[prop] += 1

    if not prop_counts:
        return 0.0

    # Calcular probabilidades
    total = sum(prop_counts.values())
    probabilities = [count / total for count in prop_counts.values()]

    # Calcular entropía de Shannon
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


def compute_world_level_transitions(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Analiza transiciones entre niveles discretos del mundo.

    Args:
        episode: Diccionario con trace de episodio.

    Returns:
        Diccionario con:
        - total_transitions: int
        - upward: int (SAFE→ELEVATED, etc.)
        - downward: int (CRITICAL→WARNING, etc.)
        - stable: int (mismo nivel)
        - transition_matrix: list[list[int]] (4x4 desde/hacia)
    """
    trace = episode.get('trace', [])

    # Mapeo de niveles a índices
    level_map = {
        'SAFE': 1,
        'ELEVATED': 2,
        'WARNING': 3,
        'CRITICAL': 4,
    }

    # Inicializar contadores
    transitions = {
        'total_transitions': 0,
        'upward': 0,
        'downward': 0,
        'stable': 0,
        'transition_matrix': [[0 for _ in range(4)] for _ in range(4)],
    }

    if len(trace) < 2:
        return transitions

    for i in range(len(trace) - 1):
        # Obtener nivel actual y siguiente
        current_obs = trace[i].get('observation', {})
        next_obs = trace[i + 1].get('observation', {})

        current_level_name = current_obs.get('state', {}).get('world_level_semantic', 'SAFE')
        next_level_name = next_obs.get('state', {}).get('world_level_semantic', 'SAFE')

        current_level = level_map.get(current_level_name, 1)
        next_level = level_map.get(next_level_name, 1)

        transitions['total_transitions'] += 1

        # Actualizar matriz de transición (0-indexed)
        transitions['transition_matrix'][current_level - 1][next_level - 1] += 1

        # Clasificar transición
        if next_level > current_level:
            transitions['upward'] += 1
        elif next_level < current_level:
            transitions['downward'] += 1
        else:
            transitions['stable'] += 1

    return transitions


def compute_spatial_coherence_index(episode: Dict[str, Any]) -> Optional[float]:
    """Calcula correlación entre temperatura de celdas vecinas (solo 5x5).

    Formula: coherence = Σ(correlation(cell_i, neighbors(cell_i))) / N_cells
    Vecindad: 4-vecinos (norte, sur, este, oeste)

    Args:
        episode: Diccionario con trace de episodio (debe ser 5x5).

    Returns:
        Coherencia en [-1.0, 1.0], típicamente [0.5, 1.0].
        None si no es 5x5 o no hay datos espaciales.
    """
    trace = episode.get('trace', [])

    if not trace:
        return None

    # Verificar que es un escenario 5x5
    first_step = trace[0]
    state = first_step.get('observation', {}).get('state', {})

    # Verificar si tiene estructura de grid
    if 'cells' not in state:
        return None

    cells = state['cells']

    # Inferir grid_size
    grid_size = int(math.sqrt(len(cells)))

    if grid_size != 5:
        return None

    # Calcular coherencia promedio a través de todos los steps
    coherences = []

    for step in trace:
        state = step.get('observation', {}).get('state', {})
        cells = state.get('cells', [])

        if len(cells) != 25:
            continue

        # Para cada celda, calcular correlación con vecinos
        step_coherence = 0.0
        cell_count = 0

        for idx, cell in enumerate(cells):
            row = idx // grid_size
            col = idx % grid_size
            temp = cell.get('temperature', 0.0)

            # Obtener vecinos (4-vecindad)
            neighbors = []
            if row > 0:  # norte
                neighbors.append(cells[(row - 1) * grid_size + col].get('temperature', 0.0))
            if row < grid_size - 1:  # sur
                neighbors.append(cells[(row + 1) * grid_size + col].get('temperature', 0.0))
            if col > 0:  # oeste
                neighbors.append(cells[row * grid_size + (col - 1)].get('temperature', 0.0))
            if col < grid_size - 1:  # este
                neighbors.append(cells[row * grid_size + (col + 1)].get('temperature', 0.0))

            if not neighbors:
                continue

            # Correlación simple: 1 - diferencia absoluta promedio
            avg_neighbor_temp = sum(neighbors) / len(neighbors)
            similarity = 1.0 - abs(temp - avg_neighbor_temp)

            step_coherence += similarity
            cell_count += 1

        if cell_count > 0:
            coherences.append(step_coherence / cell_count)

    if not coherences:
        return None

    # Retornar coherencia promedio del episodio
    return sum(coherences) / len(coherences)


def compute_spatial_information_usage(episode: Dict[str, Any]) -> float:
    """Calcula proporción de pasos donde decisión depende de topología espacial.

    Esta métrica requiere comparar decisiones con y sin información espacial.
    Para simplificar, se calcula como proporción de proposiciones espaciales
    activas respecto al total.

    Args:
        episode: Diccionario con trace de episodio.

    Returns:
        Uso de información espacial en [0.0, 1.0].
    """
    trace = episode.get('trace', [])

    if not trace:
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

    steps_with_spatial = 0
    total_steps = len(trace)

    for step in trace:
        obs = step.get('observation', {})
        props = obs.get('propositions', [])

        # Verificar si alguna proposición espacial está presente
        if any(prop in spatial_indicators for prop in props):
            steps_with_spatial += 1

    if total_steps == 0:
        return 0.0

    return steps_with_spatial / total_steps


def compute_all_cognitive_metrics(episode: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula todas las métricas de calidad cognitiva para un episodio.

    Args:
        episode: Diccionario completo del episodio.

    Returns:
        Diccionario con todas las métricas del Grupo 2.
    """
    return {
        'factual_cf_divergence': compute_factual_cf_divergence(episode),
        'intervention_precision': compute_intervention_precision(episode),
        'proposition_diversity': compute_proposition_diversity(episode),
        'world_level_transitions': compute_world_level_transitions(episode),
        'spatial_coherence_index': compute_spatial_coherence_index(episode),
        'spatial_information_usage': compute_spatial_information_usage(episode),
    }
