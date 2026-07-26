"""Proveedor N4 proposal-only para el receptor genérico del MCI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from runtime.neural.calibration import N4ProviderCalibration
from runtime.neural.contracts import (
    NeuralInferenceRequest,
    ResourceSnapshot,
    canonical_sha256,
)
from runtime.neural.hypothesis_generator import (
    StructuralCandidate,
    StructuralHypothesisGenerator,
)
from runtime.neural.integration import MCIToN4GraphBuilder
from runtime.neural.runtime import NeuralRuntime
from runtime.symbolic.mci.contracts import (
    NeuralHypothesisEvaluation,
    TransitionSpec,
)
from runtime.symbolic.mci.provider_protocol import ExternalHypothesis


@dataclass(frozen=True)
class N4ProviderConfig:
    max_hypotheses_per_episode: int = 2
    minimum_probability: float = 0.55
    beam_width: int = 24

    def __post_init__(self) -> None:
        if self.max_hypotheses_per_episode < 1 or self.beam_width < 1:
            raise ValueError("n4_provider_budget_invalid")
        if not 0.0 <= self.minimum_probability <= 1.0:
            raise ValueError("n4_minimum_probability_invalid")


class N4HypothesisProvider:
    def __init__(
        self,
        *,
        runtime: NeuralRuntime,
        spec: TransitionSpec,
        run_id: str,
        config: N4ProviderConfig | None = None,
        calibration: N4ProviderCalibration | None = None,
    ) -> None:
        self.runtime = runtime
        self.spec = spec
        self.run_id = run_id
        self.config = config or N4ProviderConfig()
        self.calibration = calibration or N4ProviderCalibration()
        self.generator = StructuralHypothesisGenerator(
            beam_width=self.config.beam_width
        )
        self.graph_builder = MCIToN4GraphBuilder()
        self.last_evidence: dict[str, Any] | None = None
        self.last_calibration_reports: tuple[dict[str, Any], ...] = ()
        self._emitted_hypothesis_ids: set[str] = set()

    def infer_hypotheses(
        self, context: Mapping[str, Any]
    ) -> Sequence[ExternalHypothesis]:
        recent = tuple(context.get("recent_evidence") or ())
        candidates = self.generator.generate(spec=self.spec, evidence=recent)
        graph = self.graph_builder.build(
            spec=self.spec,
            evidence=recent,
            state=context.get("state") or {},
            candidate_action=str(context.get("candidate_action") or ""),
            logical_time=int(context.get("logical_time", 0)),
        )
        candidate_payload = [item.ranking_payload() for item in candidates]
        candidate_set_sha256 = canonical_sha256(candidate_payload)
        inference_id = (
            f"n4/{self.run_id}/t-{int(context.get('logical_time', 0))}/"
            f"{candidate_set_sha256[:12]}"
        )
        result = self.runtime.infer(
            NeuralInferenceRequest(
                inference_id=inference_id,
                run_id=self.run_id,
                payload={
                    "schema": "n4-mci-request.v1",
                    "graph": graph.graph,
                    "candidates": candidate_payload,
                },
                logical_time=int(context.get("logical_time", 0)),
                resources=self._resources(context),
            )
        )
        by_id = {item.hypothesis_id: item for item in candidates}
        proposals: list[ExternalHypothesis] = []
        ranked_ids: list[str] = []
        for ranking in result.candidate_output.get("rankings", ()):
            candidate = by_id.get(str(ranking.get("hypothesis_id")))
            if (
                candidate is None
                or candidate.hypothesis_id in self._emitted_hypothesis_ids
            ):
                continue
            context_key = self._context_key(candidate)
            probability = self.calibration.calibrate(
                float(ranking.get("probability_valid", 0.0)), context_key
            )
            if probability < self.config.minimum_probability:
                continue
            self.calibration.register(candidate.hypothesis_id, context_key)
            ranked_ids.append(candidate.hypothesis_id)
            proposals.append(
                ExternalHypothesis(
                    hypothesis_id=candidate.hypothesis_id,
                    kind=candidate.kind,
                    target_id=candidate.target_id,
                    expression=candidate.expression,
                    proposed_value=candidate.proposed_value,
                    confidence=probability,
                    evidence_refs=candidate.evidence_refs,
                    provider="n4-causal-ranker",
                    model_ref=self.runtime.manifest.model_id,
                )
            )
            self._emitted_hypothesis_ids.add(candidate.hypothesis_id)
            if len(proposals) >= self.config.max_hypotheses_per_episode:
                break
        calibration_ref = canonical_sha256(self.calibration.snapshot())
        self.last_evidence = {
            "schema": "n4_evidence.v1",
            "inference_id": inference_id,
            "input_graph_sha256": graph.graph_sha256,
            "candidate_set_sha256": candidate_set_sha256,
            "model_manifest_sha256": result.model_manifest_sha256,
            "model_ref": self.runtime.manifest.model_id,
            "ranked_hypothesis_ids": ranked_ids,
            "uncertainty": result.uncertainty,
            "fallback_used": result.fallback_used,
            "fallback_reason": result.fallback_reason,
            "calibration_ref": f"sha256:{calibration_ref}",
        }
        return tuple(proposals)

    def receive_feedback(
        self, evaluations: Sequence[NeuralHypothesisEvaluation]
    ) -> None:
        self.last_calibration_reports = tuple(
            item.to_dict() for item in self.calibration.update(evaluations)
        )

    @staticmethod
    def _resources(context: Mapping[str, Any]) -> ResourceSnapshot:
        raw = context.get("resources") or {}
        return ResourceSnapshot(
            cpu_pressure=float(raw.get("cpu_pressure", 0.0)),
            memory_pressure=float(raw.get("memory_pressure", 0.0)),
            thermal_pressure=float(raw.get("thermal_pressure", 0.0)),
            budget_available=bool(raw.get("budget_available", True)),
        )

    def _context_key(self, candidate: StructuralCandidate) -> str:
        return f"{self.spec.scenario}|{candidate.kind}|{candidate.target_id}"
