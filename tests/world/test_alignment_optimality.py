"""B7 — el alineamiento de intervenciones debe ser ÓPTIMO, no greedy.

``alignment.py`` prometía en su docstring "asignación óptima" y "algoritmo
húngaro" mientras ejecutaba ``_greedy_assignment``. El greedy NO garantiza el
óptimo — ni siquiera en matrices 2x2, que es el tamaño real de todos los
escenarios de RNFE. Estos tests fijan la garantía que el docstring promete.

El consumidor es ``morphism_engine`` (effect_alignment_score, β=0.30) y, vía
morfismos, la corte constitucional (``morphism_failure`` dispara bajo 0.35): un
score subestimado puede declarar incompatibles a dos escenarios que no lo son.
"""

from __future__ import annotations

import itertools
import random

import pytest

from runtime.world.alignment import _hungarian_assignment, align_interventions
from runtime.world.causal_signature import InterventionEffect


def _brute_force_min_cost(costs, n_rows, n_cols) -> float:
    """Costo mínimo por fuerza bruta sobre todas las asignaciones posibles."""
    k = min(n_rows, n_cols)
    best = None
    for rows in itertools.permutations(range(n_rows), k):
        for cols in itertools.permutations(range(n_cols), k):
            total = sum(costs[i][j] for i, j in zip(rows, cols))
            if best is None or total < best:
                best = total
    return best if best is not None else 0.0


def _greedy_min_cost(costs, n_rows, n_cols) -> float:
    """El greedy que ESTABA en main. Se conserva sólo como oráculo del test:
    demuestra que la asignación anterior era estrictamente peor en casos reales.
    """
    cells = sorted(
        (costs[i][j], i, j) for i in range(n_rows) for j in range(n_cols)
    )
    used_r: set[int] = set()
    used_c: set[int] = set()
    total = 0.0
    for cost, i, j in cells:
        if i not in used_r and j not in used_c:
            total += cost
            used_r.add(i)
            used_c.add(j)
            if len(used_r) == min(n_rows, n_cols):
                break
    return total


class TestHungarianOptimality:
    def test_beats_greedy_on_the_2x2_counterexample(self):
        """El caso que mata al greedy: tomar la celda mínima lo condena al peor par.

        costs=[[0.0, 0.4], [0.4, 1.0]]:
          - greedy: toma (0,0)=0.0 y queda forzado a (1,1)=1.0  -> 1.0
          - óptimo: (0,1)+(1,0) = 0.4+0.4                        -> 0.8
        """
        costs = [[0.0, 0.4], [0.4, 1.0]]
        pairs, _, _ = _hungarian_assignment(costs, 2, 2)
        hungarian_cost = sum(costs[i][j] for i, j in pairs)

        assert hungarian_cost == pytest.approx(0.8)
        assert _greedy_min_cost(costs, 2, 2) == pytest.approx(1.0)
        # Estrictamente mejor: el test falla si alguien reinstala el greedy.
        assert hungarian_cost < _greedy_min_cost(costs, 2, 2)

    @pytest.mark.parametrize("seed", range(25))
    def test_matches_brute_force_optimum(self, seed: int):
        """Sobre matrices aleatorias, el húngaro iguala al óptimo por fuerza bruta."""
        rng = random.Random(seed)
        n_rows = rng.randint(1, 5)
        n_cols = rng.randint(1, 5)
        costs = [
            [round(rng.uniform(0.0, 1.0), 3) for _ in range(n_cols)]
            for _ in range(n_rows)
        ]

        pairs, used_rows, used_cols = _hungarian_assignment(costs, n_rows, n_cols)
        got = sum(costs[i][j] for i, j in pairs)

        assert got == pytest.approx(_brute_force_min_cost(costs, n_rows, n_cols))
        # Emparejamiento válido: cardinalidad máxima, sin repetir filas ni columnas.
        assert len(pairs) == min(n_rows, n_cols)
        assert len(used_rows) == len(pairs)
        assert len(used_cols) == len(pairs)

    def test_never_worse_than_greedy(self):
        """El húngaro nunca es peor que el greedy, y a menudo es mejor."""
        rng = random.Random(1234)
        strictly_better = 0
        for _ in range(300):
            n_rows = rng.randint(2, 4)
            n_cols = rng.randint(2, 4)
            costs = [
                [round(rng.uniform(0.0, 1.0), 3) for _ in range(n_cols)]
                for _ in range(n_rows)
            ]
            pairs, _, _ = _hungarian_assignment(costs, n_rows, n_cols)
            hung = sum(costs[i][j] for i, j in pairs)
            greedy = _greedy_min_cost(costs, n_rows, n_cols)
            assert hung <= greedy + 1e-9
            if hung < greedy - 1e-9:
                strictly_better += 1
        # No es una diferencia teórica: en matrices al azar el greedy pierde seguido.
        assert strictly_better > 0

    def test_deterministic_under_ties(self):
        """Empate total: la asignación debe ser reproducible."""
        costs = [[0.5, 0.5], [0.5, 0.5]]
        results = {tuple(_hungarian_assignment(costs, 2, 2)[0]) for _ in range(50)}
        assert len(results) == 1

    def test_rectangular_matrices_both_orientations(self):
        """Rectangulares en ambos sentidos (más filas que columnas y viceversa)."""
        wide = [[0.9, 0.1, 0.5], [0.2, 0.8, 0.7]]
        pairs, _, _ = _hungarian_assignment(wide, 2, 3)
        assert len(pairs) == 2
        assert sum(wide[i][j] for i, j in pairs) == pytest.approx(
            _brute_force_min_cost(wide, 2, 3)
        )

        tall = [[0.9, 0.2], [0.1, 0.8], [0.5, 0.7]]
        pairs, _, _ = _hungarian_assignment(tall, 3, 2)
        assert len(pairs) == 2
        assert all(0 <= i < 3 and 0 <= j < 2 for i, j in pairs)
        assert sum(tall[i][j] for i, j in pairs) == pytest.approx(
            _brute_force_min_cost(tall, 3, 2)
        )


class TestAlignInterventionsUsesOptimum:
    def test_identical_signatures_align_perfectly(self):
        """Efectos idénticos ⇒ costo 0 y score 1.0."""
        source = [
            InterventionEffect("s_a", "v", "-", 0.10, "corrective"),
            InterventionEffect("s_b", "v", "+", 0.90, "neutral"),
        ]
        target = [
            InterventionEffect("t_a", "v", "-", 0.10, "corrective"),
            InterventionEffect("t_b", "v", "+", 0.90, "neutral"),
        ]
        result = align_interventions(source, target)
        assert result.normalized_score == pytest.approx(1.0)
        assert result.total_cost == pytest.approx(0.0)
        assert result.coverage == pytest.approx(1.0)

    def test_optimum_on_real_signatures_and_greedy_was_suboptimal(self):
        """En los escenarios REALES el húngaro alcanza el óptimo; el greedy no lo hacía."""
        from runtime.world.registry import get_scenario
        from runtime.world.alignment import _intervention_distance

        names = [
            "thermal_homeostasis",
            "resource_management",
            "grid_thermal_5x5",
            "deferred_load_trap",
        ]
        sigs = {n: get_scenario(n).causal_signature for n in names}
        improved = 0
        for src_name, tgt_name in itertools.permutations(names, 2):
            src = sigs[src_name].intervention_effects
            tgt = sigs[tgt_name].intervention_effects
            costs = [[_intervention_distance(a, b) for b in tgt] for a in src]
            n_r, n_c = len(src), len(tgt)

            pairs, _, _ = _hungarian_assignment(costs, n_r, n_c)
            hung = sum(costs[i][j] for i, j in pairs)
            greedy = _greedy_min_cost(costs, n_r, n_c)
            optimal = _brute_force_min_cost(costs, n_r, n_c)

            assert hung == pytest.approx(optimal), f"{src_name}->{tgt_name}"
            assert hung <= greedy + 1e-9
            if hung < greedy - 1e-9:
                improved += 1

        # Hecho MEDIDO (P12/B7): el greedy era subóptimo en 4 de los 12 cruces
        # reales — los que involucran deferred_load_trap, el único escenario con
        # dos intervenciones correctivas. No era una preocupación teórica.
        assert improved == 4
