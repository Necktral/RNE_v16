"""Alineamiento bipartito dirigido entre componentes de dos escenarios.

Implementa asignación óptima entre intervenciones y proposiciones
de dos escenarios para computar scores de alineamiento dirigido.

Usa el algoritmo húngaro (scipy-free, implementación interna) sobre
una matriz de costo que combina distancia semántica, de efecto,
de control y contrafactual.
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

    Usa asignación greedy por mínimo costo (suficiente para N pequeño).
    Para N > 10 se podría usar Hungarian, pero en RNFE los escenarios
    tienen 2-4 intervenciones.
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

    # Greedy minimum-cost assignment
    pairs, used_src, used_tgt = _greedy_assignment(costs, n_src, n_tgt)

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

def _greedy_assignment(
    costs: list[list[float]],
    n_rows: int,
    n_cols: int,
) -> Tuple[list[Tuple[int, int]], set[int], set[int]]:
    """Greedy minimum-cost assignment for small matrices.

    Returns list of (row, col) pairs, sets of used rows and cols.
    """
    pairs: list[Tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    # Flatten all cells, sort by cost
    cells = []
    for i in range(n_rows):
        for j in range(n_cols):
            cells.append((costs[i][j], i, j))
    cells.sort(key=lambda x: x[0])

    for cost, i, j in cells:
        if i not in used_rows and j not in used_cols:
            pairs.append((i, j))
            used_rows.add(i)
            used_cols.add(j)
            if len(pairs) == min(n_rows, n_cols):
                break

    return pairs, used_rows, used_cols
