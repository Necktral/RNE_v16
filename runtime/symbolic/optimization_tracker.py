"""Adaptador puro del resultado OPT al contrato simbólico."""

from __future__ import annotations

from typing import Any, Mapping

from .schemas import OptimizationCandidate, OptimizationDecisionReport


def build_optimization_report(
    state: Mapping[str, Any],
    result: Mapping[str, Any],
    replay_unit_id: str,
    logical_time: int,
) -> OptimizationDecisionReport:
    """Construye un reporte sin ejecutar ni alterar OPT."""

    del replay_unit_id, logical_time
    choice = result.get("opt_choice") if isinstance(result, Mapping) else None
    if not isinstance(choice, Mapping):
        choice = {}
    effort = float(choice.get("effort_cost", 0.0) or 0.0)
    candidates = tuple(
        OptimizationCandidate(
            intervention=str(item.get("intervention", "")),
            steps=int(item.get("steps", 0) or 0),
            projected=float(item.get("projected", 0.0) or 0.0),
            objective=float(item.get("objective", 0.0) or 0.0),
            effort_cost=effort,
        )
        for item in choice.get("alternatives", ())
        if isinstance(item, Mapping)
    )
    winner = None
    if choice.get("status") == "ok":
        winner = OptimizationCandidate(
            intervention=str(choice.get("intervention", "")),
            steps=int(choice.get("steps", 0) or 0),
            projected=float(choice.get("projected", 0.0) or 0.0),
            objective=float(choice.get("objective", 0.0) or 0.0),
            effort_cost=effort,
        )
    return OptimizationDecisionReport(
        x0=float(choice["x0"]) if choice.get("x0") is not None else None,
        direction=str(choice.get("optimization_direction", "")),
        candidates=candidates,
        winner=winner,
        tie_breaks=("objective", "steps", "intervention"),
    )
