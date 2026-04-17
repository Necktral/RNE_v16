"""Búsqueda acotada y determinista de candidatos EML."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable, List

from .tree import ExprNode


@dataclass(frozen=True, slots=True)
class SearchLimits:
    max_depth: int = 3
    max_candidates: int = 64
    max_evals: int = 512
    seed: int = 0


def _base_atoms(var_names: list[str]) -> list[ExprNode]:
    atoms = [ExprNode(op="const", value=0.0), ExprNode(op="const", value=1.0)]
    atoms.extend(ExprNode(op="var", name=name) for name in var_names)
    return atoms


def generate_candidates(
    *,
    var_names: Iterable[str],
    limits: SearchLimits,
) -> list[ExprNode]:
    names = sorted([name for name in var_names if isinstance(name, str)])
    if not names:
        names = ["x"]
    rng = Random(limits.seed)
    candidates: List[ExprNode] = _base_atoms(names)
    frontier = list(candidates)
    binary_ops = ["add", "sub", "mul", "div"]
    unary_ops = ["pow2", "log1p", "exp"]
    eval_budget = 0

    for depth in range(2, limits.max_depth + 1):
        if len(candidates) >= limits.max_candidates:
            break
        next_frontier: List[ExprNode] = []
        sample_left = frontier[:]
        sample_right = candidates[:]
        rng.shuffle(sample_left)
        rng.shuffle(sample_right)

        for left in sample_left:
            if len(candidates) >= limits.max_candidates or eval_budget >= limits.max_evals:
                break
            for op in unary_ops:
                node = ExprNode(op=op, left=left)
                if node.depth() == depth:
                    candidates.append(node)
                    next_frontier.append(node)
                    eval_budget += 1
                    if len(candidates) >= limits.max_candidates:
                        break
            if len(candidates) >= limits.max_candidates or eval_budget >= limits.max_evals:
                break
            for right in sample_right:
                for op in binary_ops:
                    node = ExprNode(op=op, left=left, right=right)
                    if node.depth() == depth:
                        candidates.append(node)
                        next_frontier.append(node)
                        eval_budget += 1
                        if (
                            len(candidates) >= limits.max_candidates
                            or eval_budget >= limits.max_evals
                        ):
                            break
                if len(candidates) >= limits.max_candidates or eval_budget >= limits.max_evals:
                    break
            if len(candidates) >= limits.max_candidates or eval_budget >= limits.max_evals:
                break
        frontier = next_frontier or frontier
    return candidates[: limits.max_candidates]

