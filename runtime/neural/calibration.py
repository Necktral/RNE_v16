"""Calibración contextual online sin actualización de pesos N4."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from runtime.neural.contracts import canonical_sha256
from runtime.symbolic.mci.contracts import NeuralHypothesisEvaluation


@dataclass(frozen=True)
class CalibrationReport:
    context_key: str
    alpha: float
    beta: float
    temperature: float
    posterior_success: float
    calibration_ref: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_key": self.context_key,
            "alpha": self.alpha,
            "beta": self.beta,
            "temperature": self.temperature,
            "posterior_success": self.posterior_success,
            "calibration_ref": self.calibration_ref,
        }


class N4ProviderCalibration:
    def __init__(self) -> None:
        self._counts: dict[str, list[float]] = {}
        self._hypothesis_context: dict[str, str] = {}

    def register(self, hypothesis_id: str, context_key: str) -> None:
        self._hypothesis_context[hypothesis_id] = context_key
        self._counts.setdefault(context_key, [1.0, 1.0])

    def calibrate(self, probability: float, context_key: str) -> float:
        alpha, beta = self._counts.get(context_key, [1.0, 1.0])
        posterior = alpha / (alpha + beta)
        temperature = self._temperature(alpha, beta)
        probability = min(1.0 - 1e-9, max(1e-9, float(probability)))
        logit = math.log(probability / (1.0 - probability))
        scaled = 1.0 / (1.0 + math.exp(-logit / temperature))
        return round(0.5 * scaled + 0.5 * posterior, 6)

    def update(
        self, evaluations: Sequence[NeuralHypothesisEvaluation]
    ) -> tuple[CalibrationReport, ...]:
        touched: set[str] = set()
        for evaluation in evaluations:
            if evaluation.status == "pending":
                continue
            context = self._hypothesis_context.get(evaluation.hypothesis_id)
            if context is None:
                continue
            alpha, beta = self._counts.setdefault(context, [1.0, 1.0])
            if evaluation.status == "promoted":
                alpha += 1.0
            elif evaluation.status in {"rejected", "invalid_expression"}:
                beta += 1.0
            self._counts[context] = [alpha, beta]
            touched.add(context)
        return tuple(self.report(key) for key in sorted(touched))

    def report(self, context_key: str) -> CalibrationReport:
        alpha, beta = self._counts.get(context_key, [1.0, 1.0])
        payload = {
            "context_key": context_key,
            "alpha": alpha,
            "beta": beta,
            "temperature": self._temperature(alpha, beta),
        }
        return CalibrationReport(
            context_key=context_key,
            alpha=alpha,
            beta=beta,
            temperature=self._temperature(alpha, beta),
            posterior_success=round(alpha / (alpha + beta), 9),
            calibration_ref=f"calibration-{canonical_sha256(payload)[:24]}",
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "n4-calibration.v1",
            "counts": {
                key: {"alpha": value[0], "beta": value[1]}
                for key, value in sorted(self._counts.items())
            },
        }

    def restore(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema") != "n4-calibration.v1":
            raise ValueError("calibration_schema_invalid")
        restored: dict[str, list[float]] = {}
        for key, raw in (payload.get("counts") or {}).items():
            alpha, beta = float(raw["alpha"]), float(raw["beta"])
            if alpha <= 0 or beta <= 0:
                raise ValueError("calibration_counts_invalid")
            restored[str(key)] = [alpha, beta]
        self._counts = restored

    @staticmethod
    def _temperature(alpha: float, beta: float) -> float:
        support = alpha + beta - 2.0
        return round(max(0.65, 1.5 / math.sqrt(1.0 + support / 4.0)), 6)
