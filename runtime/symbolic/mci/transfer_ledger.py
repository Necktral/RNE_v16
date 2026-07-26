"""Ledger separado para no modificar contratos históricos del JTMS."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from .contracts import NeuralHypothesisEvaluation
from .transfer_package import TransferredHypothesis


@dataclass(frozen=True)
class TransferBelief:
    hypothesis_id: str
    package_id: str
    morphism_id: str
    status: str
    belief_origin: str
    logical_time: int
    evaluation_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return vars(self)


class TransferBeliefLedger:
    def __init__(self) -> None:
        self._beliefs: dict[str, TransferBelief] = {}

    def propose(self, item: TransferredHypothesis) -> None:
        previous = self._beliefs.get(item.hypothesis_id)
        candidate = TransferBelief(
            hypothesis_id=item.hypothesis_id,
            package_id=item.package_id,
            morphism_id=item.morphism_id,
            status="REVISABLE",
            belief_origin="structural-transfer",
            logical_time=item.logical_time,
        )
        if previous is not None and previous != candidate:
            raise ValueError("transfer_belief_collision")
        self._beliefs[item.hypothesis_id] = candidate

    def apply(self, evaluation: NeuralHypothesisEvaluation) -> None:
        previous = self._beliefs.get(evaluation.hypothesis_id)
        if previous is None:
            return
        status = {
            "promoted": "IN",
            "rejected": "OUT",
            "invalid_expression": "OUT",
            "degraded": "REVISABLE",
            "pending": "REVISABLE",
        }.get(evaluation.status, "REVISABLE")
        self._beliefs[evaluation.hypothesis_id] = replace(
            previous, status=status, evaluation_reason=evaluation.reason
        )

    def snapshot(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._beliefs[key].to_dict() for key in sorted(self._beliefs))

    def restore(self, items: tuple[Mapping[str, Any], ...]) -> None:
        self._beliefs = {
            str(item["hypothesis_id"]): TransferBelief(
                hypothesis_id=str(item["hypothesis_id"]),
                package_id=str(item["package_id"]),
                morphism_id=str(item["morphism_id"]),
                status=str(item["status"]),
                belief_origin=str(item["belief_origin"]),
                logical_time=int(item["logical_time"]),
                evaluation_reason=(
                    str(item["evaluation_reason"])
                    if item.get("evaluation_reason") is not None
                    else None
                ),
            )
            for item in items
        }
