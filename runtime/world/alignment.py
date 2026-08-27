"""Alineamiento bipartito dirigido entre componentes de dos escenarios.

Computa scores de alineamiento dirigido entre intervenciones y proposiciones
de dos escenarios.

- ``align_interventions``: asignación ÓPTIMA por algoritmo húngaro
  (``_hungarian_assignment``, scipy-free, implementación interna) sobre una
  matriz de costo que combina dirección, magnitud y rol semántico del efecto.
- ``align_propositions``: NO es una asignación — es Jaccard sobre los
  vocabularios (los items se emparejan por identidad de nombre, no por costo).
- ``align_causal_graphs``: fracción de aristas coincidentes con igual polaridad.

B7: el docstring previo prometía "asignación óptima" y "algoritmo húngaro"
mientras la implementación era ``_greedy_assignment``, que no garantiza el
óptimo. Ahora el húngaro está efectivamente implementado y la promesa es cierta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Sequence, Tuple

from .causal_signature import InterventionEffect, CausalEdge


# ── Data contracts ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AlignmentPair:
    """Par de elementos alineados entre source y target."""

    source_item: str
    target_item: str
    cost: float


@dataclass(frozen=True)
class AlignmentResult:
    """Resultado del alineamiento bipartito."""

    pairs: Tuple[AlignmentPair, ...]
    total_cost: float
    normalized_score: float  # 1.0 - (total_cost / max_possible_cost)
    source_unmatched: FrozenSet[str]
    target_unmatched: FrozenSet[str]
    coverage: float  # fraction of elements matched


# ── Intervention alignment ───────────────────────────────────────────────────

def _intervention_distance(
    src: InterventionEffect,
    tgt: InterventionEffect,
) -> float:
    """Distancia entre dos efectos de intervención [0, 1].

    Combina:
    - direction match (0 o 0.3)
    - magnitude similarity
    - semantic role match (0 o 0.2)
    """
    cost = 0.0
    # Direction
    if src.expected_direction != tgt.expected_direction:
        cost += 0.35
    # Magnitude
    cost += 0.35 * abs(src.expected_magnitude - tgt.expected_magnitude)
    # Semantic role
    if src.semantic_role != tgt.semantic_role:
        cost += 0.30
    return min(1.0, cost)


def align_interventions(
    source: Sequence[InterventionEffect],
    target: Sequence[InterventionEffect],
) -> AlignmentResult:
    """Alineamiento bipartito dirigido de intervenciones.

    Usa asignación ÓPTIMA (algoritmo húngaro, ``_hungarian_assignment``).

    B7: antes usaba ``_greedy_assignment``, que NO garantiza el óptimo — ni
    siquiera en 2x2. Contraejemplo real: costs=[[0.0, 0.4], [0.4, 1.0]]; greedy
    toma la celda mínima (0.0) y queda forzado a 1.0 ⇒ coste 1.0, mientras que el
    óptimo es 0.4+0.4=0.8. El greedy es una COTA SUPERIOR del coste óptimo, o sea
    una COTA INFERIOR del ``normalized_score``: subestimaba el alineamiento.

    Esto no era cosmético: ``morphism_engine`` deriva de acá el
    ``effect_alignment_score`` (β=0.30 del score dirigido) y la corte
    constitucional clasifica compatibilidad con ese score (``morphism_failure``
    dispara bajo 0.35). Un score subestimado puede declarar incompatibles a dos
    escenarios que no lo son.
    """
    if not source and not target:
        return AlignmentResult(
            pairs=(), total_cost=0.0, normalized_score=1.0,
            source_unmatched=frozenset(), target_unmatched=frozenset(),
            coverage=1.0,
        )
    if not source or not target:
        all_src = frozenset(e.intervention_name for e in source)
        all_tgt = frozenset(e.intervention_name for e in target)
        return AlignmentResult(
            pairs=(), total_cost=1.0, normalized_score=0.0,
            source_unmatched=all_src, target_unmatched=all_tgt,
            coverage=0.0,
        )

    # Build cost matrix
    n_src = len(source)
    n_tgt = len(target)
    costs = [[0.0] * n_tgt for _ in range(n_src)]
    for i, s in enumerate(source):
        for j, t in enumerate(target):
            costs[i][j] = _intervention_distance(s, t)

    # Optimal (Hungarian) minimum-cost assignment
    pairs, used_src, used_tgt = _hungarian_assignment(costs, n_src, n_tgt)

    alignment_pairs = tuple(
        AlignmentPair(
            source_item=source[i].intervention_name,
            target_item=target[j].intervention_name,
            cost=costs[i][j],
        )
        for i, j in pairs
    )

    total_cost = sum(p.cost for p in alignment_pairs)
    max_pairs = max(n_src, n_tgt)
    # Unmatched elements get cost 1.0 each
    unmatched_count = max_pairs - len(pairs)
    effective_cost = total_cost + unmatched_count * 1.0
    max_cost = max_pairs * 1.0
    normalized = max(0.0, 1.0 - (effective_cost / max_cost)) if max_cost > 0 else 1.0

    src_matched = {i for i, _ in pairs}
    tgt_matched = {j for _, j in pairs}

    return AlignmentResult(
        pairs=alignment_pairs,
        total_cost=round(total_cost, 4),
        normalized_score=round(normalized, 4),
        source_unmatched=frozenset(
            source[i].intervention_name for i in range(n_src) if i not in src_matched
        ),
        target_unmatched=frozenset(
            target[j].intervention_name for j in range(n_tgt) if j not in tgt_matched
        ),
        coverage=round(len(pairs) / max_pairs, 4) if max_pairs > 0 else 1.0,
    )


# ── Proposition alignment ────────────────────────────────────────────────────

def align_propositions(
    source_vocab: FrozenSet[str],
    target_vocab: FrozenSet[str],
) -> AlignmentResult:
    """Alineamiento de vocabularios de proposiciones.

    Más simple que intervenciones: usa Jaccard extendido con
    penalización por tamaño asimétrico.
    """
    if not source_vocab and not target_vocab:
        return AlignmentResult(
            pairs=(), total_cost=0.0, normalized_score=1.0,
            source_unmatched=frozenset(), target_unmatched=frozenset(),
            coverage=1.0,
        )

    intersection = source_vocab & target_vocab
    union = source_vocab | target_vocab

    pairs = tuple(
        AlignmentPair(source_item=p, target_item=p, cost=0.0)
        for p in sorted(intersection)
    )
    # Asymmetry penalty: larger set loses more
    size_ratio = len(intersection) / len(union) if union else 1.0

    return AlignmentResult(
        pairs=pairs,
        total_cost=round(1.0 - size_ratio, 4),
        normalized_score=round(size_ratio, 4),
        source_unmatched=frozenset(source_vocab - target_vocab),
        target_unmatched=frozenset(target_vocab - source_vocab),
        coverage=round(len(intersection) / max(len(source_vocab), len(target_vocab), 1), 4),
    )


# ── Causal graph alignment ──────────────────────────────────────────────────

def align_causal_graphs(
    source_edges: Sequence[CausalEdge],
    target_edges: Sequence[CausalEdge],
) -> float:
    """Computes alignment score between two causal DAGs [0, 1].

    Matches edges by (source, target) pair and compares polarity.
    Returns fraction of matching edges with same polarity.
    """
    if not source_edges and not target_edges:
        return 1.0
    if not source_edges or not target_edges:
        return 0.0

    src_map = {(e.source, e.target): e for e in source_edges}
    tgt_map = {(e.source, e.target): e for e in target_edges}

    all_keys = set(src_map.keys()) | set(tgt_map.keys())
    if not all_keys:
        return 1.0

    matches = 0
    partial = 0
    for key in all_keys:
        if key in src_map and key in tgt_map:
            if src_map[key].polarity == tgt_map[key].polarity:
                matches += 1
            else:
                partial += 0.5
        # Missing in one side = 0 contribution

    return round((matches + partial) / len(all_keys), 4)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _hungarian_assignment(
    costs: list[list[float]],
    n_rows: int,
    n_cols: int,
) -> Tuple[list[Tuple[int, int]], set[int], set[int]]:
    """Asignación de costo mínimo ÓPTIMA (algoritmo húngaro / Kuhn-Munkres).

    Implementación O(n^2·m) por caminos aumentantes cortos con potenciales
    (variante Jonker-Volgenant del húngaro), sin dependencias externas.
    Emparejamiento máximo de cardinalidad ``min(n_rows, n_cols)`` de costo total
    mínimo — GARANTIZADO óptimo, a diferencia del greedy anterior.

    Determinismo: ante varias asignaciones de igual costo total, la elección
    queda fijada por el orden de índices (recorridos ascendentes y desempate por
    ``<`` estricto), de modo que el resultado es reproducible.

    Args:
        costs: Matriz de costos ``n_rows x n_cols``.
        n_rows: Filas (source).
        n_cols: Columnas (target).

    Returns:
        (pares (fila, col), filas usadas, columnas usadas).
    """
    if n_rows == 0 or n_cols == 0:
        return [], set(), set()

    # El algoritmo requiere n_rows <= n_cols: si no, se transpone y se deshace al final.
    transposed = n_rows > n_cols
    if transposed:
        matrix = [[costs[i][j] for i in range(n_rows)] for j in range(n_cols)]
        n, m = n_cols, n_rows
    else:
        matrix = [row[:] for row in costs]
        n, m = n_rows, n_cols

    INF = float("inf")
    # Potenciales duales (u sobre filas, v sobre columnas). Índices 1-based con
    # una fila/columna centinela en 0 (convención clásica del algoritmo).
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)      # p[j] = fila asignada a la columna j (0 = libre)
    way = [0] * (m + 1)    # árbol de caminos aumentantes

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        # Deshacer el camino aumentante encontrado.
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    pairs: list[Tuple[int, int]] = []
    for j in range(1, m + 1):
        if p[j] != 0:
            row, col = p[j] - 1, j - 1
            pairs.append((col, row) if transposed else (row, col))

    pairs.sort()
    used_rows = {i for i, _ in pairs}
    used_cols = {j for _, j in pairs}
    return pairs, used_rows, used_cols
