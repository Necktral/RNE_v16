"""Adaptadores canónicos N1-N6 para la frontera simbiótica viva.

Cada adaptador llama el módulo dueño del órgano o una política de referencia
declarada. Ninguno selecciona acciones, certifica, muta grafos ni promociona memoria.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from runtime.neural.contracts import BackendOutput, NeuralInferenceRequest, NeuralMode
from runtime.neural.organs.n4_causal import (
    GRAPH_SCHEMA_VERSION,
    CausalMessagePassingBackend,
)
from runtime.neural.organs.n5_ingest import DeterministicChunker

from .contracts import SymbiosisIdentity


OPTIONAL_FAMILIES = ("IND", "PLAN", "OPT", "NESY", "IMAGINATION", "EVO_SEARCH")


class CanonicalOrganAdapter(Protocol):
    organ: str
    capability: str
    authority_ceiling: NeuralMode
    consumer: str
    reference_id: str

    def infer(
        self, request: NeuralInferenceRequest, context: Mapping[str, Any]
    ) -> BackendOutput: ...

    def fallback(self, identity: SymbiosisIdentity) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class N1ReferencePolicy:
    """Política contextual determinista; no se presenta como MLP entrenado."""

    min_utility: float = 0.25
    max_families: int = 2

    def evaluate(self, inputs: Mapping[str, Any], temporal: Mapping[str, Any]) -> BackendOutput:
        observation = inputs.get("observation") or {}
        resources = inputs.get("resources") or {}
        alarm = bool(observation.get("alarm"))
        uncertainty = float(temporal.get("uncertainty", 1.0) or 1.0)
        pressure = max(
            float(resources.get("cpu_pressure", 0.0) or 0.0),
            float(resources.get("memory_pressure", 0.0) or 0.0),
            float(resources.get("thermal_pressure", 0.0) or 0.0),
        )
        ranked = []
        for family in OPTIONAL_FAMILIES:
            score = {
                "IND": 0.25 + 0.20 * bool(inputs.get("memory_hits")),
                "PLAN": 0.35 + 0.30 * alarm,
                "OPT": 0.30 + 0.20 * alarm,
                "NESY": 0.35 + 0.20 * uncertainty,
                "IMAGINATION": 0.20 + 0.25 * alarm,
                "EVO_SEARCH": 0.15 + 0.10 * alarm,
            }[family]
            score -= 0.45 * pressure
            ranked.append({"family": family, "score": round(score, 6)})
        ranked.sort(key=lambda row: (-row["score"], row["family"]))
        proposed = [
            row["family"] for row in ranked if row["score"] > self.min_utility
        ][: self.max_families]
        decision = "ACTIVATE" if proposed else "ABSTAIN"
        return BackendOutput(
            candidate_output={
                "status": "ok" if proposed else "abstained",
                "backend": "n1_reference_policy",
                "classification": "reference",
                "trained_model": False,
                "activation": {"decision": decision, "families": proposed},
                "proposed_families": proposed,
                "ranked": ranked,
                "features": {
                    "alarm": alarm,
                    "uncertainty": uncertainty,
                    "resource_pressure": pressure,
                    "memory_count": len(inputs.get("memory_hits", [])),
                },
                "scheduler_authority": True,
            },
            confidence=max([row["score"] for row in ranked], default=0.0),
            uncertainty=uncertainty,
            cost={"families_scored": len(ranked), "ram_mb": 0.05, "vram_mb": 0.0},
        )


class N1Adapter:
    organ = "N1"
    capability = "family_routing_proposal"
    authority_ceiling = NeuralMode.SHADOW
    consumer = "MetaSchedulerComparator"
    reference_id = "rnfe:N1:family_routing_proposal:reference-policy-v1"

    def __init__(self) -> None:
        self.policy = N1ReferencePolicy()

    def infer(self, request: NeuralInferenceRequest, context: Mapping[str, Any]) -> BackendOutput:
        return self.policy.evaluate(context["inputs"], context.get("n3_temporal") or {})

    def fallback(self, identity: SymbiosisIdentity) -> Mapping[str, Any]:
        return {"status": "disabled", "proposed_families": []}


class N2Adapter:
    organ = "N2"
    capability = "semantic_neurosymbolic_candidate"
    authority_ceiling = NeuralMode.SHADOW
    consumer = "DED+LOT-F+NESY"
    reference_id = "rnfe:N2:semantic_neurosymbolic_candidate:nesy-v1"

    def infer(self, request: NeuralInferenceRequest, context: Mapping[str, Any]) -> BackendOutput:
        from runtime.reasoning.families import nesy

        state = dict(context.get("reasoning_state") or {})
        inputs = context["inputs"]
        lotf_valid = bool(context.get("lotf_valid"))
        nesy_result = nesy.execute({**state, "_symbiotic_n2_verify": True})
        nesy_delta = dict(nesy_result.get("state_delta") or {})
        ded_verified = bool(state.get("ded_validated"))
        nesy_verified = bool(nesy_delta.get("nesy_coherent"))
        verified = ded_verified and lotf_valid and nesy_verified
        verdict = "accepted_as_shadow_evidence" if verified else "rejected_by_symbolic_verifier"
        return BackendOutput(
            candidate_output={
                "status": "ok",
                "proposition": str(state.get("ded_conclusion") or inputs.get("formula") or ""),
                "verified": verified,
                "verdict": verdict,
                "verification": {"DED": ded_verified, "LOT-F": lotf_valid, "NESY": nesy_verified},
                "nesy": nesy_result,
                "authority": "DED+LOT-F+NESY",
            },
            confidence=float(nesy_result.get("confidence", 0.0) or 0.0),
            uncertainty=0.0 if verified else 1.0,
            cost={"verifiers": 3, "ram_mb": 0.05, "vram_mb": 0.0},
        )

    def fallback(self, identity: SymbiosisIdentity) -> Mapping[str, Any]:
        return {"status": "disabled", "verified": False}


class N3Adapter:
    organ = "N3"
    capability = "temporal_reference_state"
    authority_ceiling = NeuralMode.PROVISIONAL
    consumer = "next_episode+MFM+continuity"
    reference_id = "rnfe:N3:temporal_reference_state:reference-filter-v1"

    def __init__(self) -> None:
        self._states: dict[tuple[str, str, str], dict[str, Any]] = {}

    @staticmethod
    def state_key(identity: SymbiosisIdentity) -> tuple[str, str, str]:
        return (identity.organism_id, identity.scenario_id, identity.lineage_id)

    def infer(self, request: NeuralInferenceRequest, context: Mapping[str, Any]) -> BackendOutput:
        identity: SymbiosisIdentity = context["identity"]
        inputs = context["inputs"]
        key = self.state_key(identity)
        observation = inputs.get("observation") or {}
        metadata = inputs.get("scenario_metadata") or {}
        main_variable = str(metadata.get("main_variable") or "temperature")
        value = _number(observation.get(main_variable)) or 0.0
        previous = self._states.get(key)
        previous_value = _number((previous or {}).get("value"))
        trend = 0.0 if previous_value is None else value - previous_value
        count = int((previous or {}).get("episode_count", 0)) + 1
        state = {
            "status": "ok",
            "backend": "reference_temporal_filter",
            "classification": "reference",
            "mamba2_active": False,
            "state_key": list(key),
            "previous_state": previous,
            "value": value,
            "trend": trend,
            "uncertainty": 1.0 / (count + 1.0),
            "episode_count": count,
            "provenance": identity.episode_id,
            "version": "reference-temporal-v1",
            "summary": f"{main_variable}={value:.6f};trend={trend:+.6f};n={count}",
        }
        self._states[key] = state
        return BackendOutput(
            candidate_output=state,
            confidence=1.0 - state["uncertainty"],
            uncertainty=state["uncertainty"],
            cost={"state_entries": len(self._states), "ram_mb": 0.1, "vram_mb": 0.0},
        )

    def fallback(self, identity: SymbiosisIdentity) -> Mapping[str, Any]:
        return {"status": "disabled", "state_key": list(self.state_key(identity))}

    def export_state(self) -> dict[str, Any]:
        return {
            "schema_version": "n3-temporal-checkpoint-v1",
            "entries": [
                {"state_key": list(key), "state": dict(state)}
                for key, state in sorted(self._states.items())
            ],
        }

    def restore_state(self, payload: Mapping[str, Any] | None) -> int:
        data = dict(payload or {})
        if data.get("schema_version") != "n3-temporal-checkpoint-v1":
            return 0
        restored = 0
        for item in data.get("entries") or []:
            if not isinstance(item, Mapping):
                continue
            raw_key, raw_state = item.get("state_key"), item.get("state")
            if (
                not isinstance(raw_key, (list, tuple))
                or len(raw_key) != 3
                or not all(str(value or "").strip() for value in raw_key)
                or not isinstance(raw_state, Mapping)
            ):
                continue
            key = tuple(str(value) for value in raw_key)
            state = dict(raw_state)
            if list(key) != list(state.get("state_key") or []):
                continue
            self._states[key] = state
            restored += 1
        return restored


class N4Adapter:
    organ = "N4"
    capability = "typed_causal_proposal"
    authority_ceiling = NeuralMode.SHADOW
    consumer = "CAU+CTF+C-GWM comparator"
    reference_id = "rnfe:N4:typed_causal_proposal:frozen-contract-v1"

    def __init__(self) -> None:
        self.backend = CausalMessagePassingBackend()
        self.backend.load_frozen_reference_contract()

    def infer(self, request: NeuralInferenceRequest, context: Mapping[str, Any]) -> BackendOutput:
        identity: SymbiosisIdentity = context["identity"]
        graph = self._graph(identity, context["inputs"])
        graph_request = NeuralInferenceRequest(
            inference_id=request.inference_id,
            run_id=request.run_id,
            organ=request.organ,
            capability=request.capability,
            payload={"graph": graph},
            seed=request.seed,
            scope=request.scope,
            resources=request.resources,
            causal_context=request.causal_context,
        )
        return self.backend.infer(graph_request)

    def fallback(self, identity: SymbiosisIdentity) -> Mapping[str, Any]:
        return {"status": "disabled", "relations": []}

    @staticmethod
    def _graph(identity: SymbiosisIdentity, inputs: Mapping[str, Any]) -> dict[str, Any]:
        attestation = dict(inputs.get("causal_attestation") or {})
        observation = inputs.get("observation") or {}
        metadata = inputs.get("scenario_metadata") or {}
        variable = str(metadata.get("main_variable") or "temperature")
        base = _number(observation.get(variable)) or 0.0
        factual = _number(attestation.get("factual_delta"))
        counter = _number(attestation.get("counterfactual_delta"))
        if factual is None and _number(attestation.get("factual_value")) is not None:
            factual = float(attestation["factual_value"]) - base
        if counter is None and _number(attestation.get("counterfactual_value")) is not None:
            counter = float(attestation["counterfactual_value"]) - base
        effect = 0.0 if factual is None or counter is None else factual - counter
        supports_choice = attestation.get("supports_choice")
        direction = (
            1.0 if supports_choice is True else -1.0 if supports_choice is False else (1.0 if effect >= 0.0 else -1.0)
        )
        signed = direction * min(1.0, abs(effect))
        edge_type = "causal_negative" if signed < 0.0 else "causal_positive"
        if signed == 0.0:
            signed = 1e-9
        confidence = min(1.0, abs(effect) * 5.0) if factual is not None and counter is not None else 0.0
        provenance = f"causal_attestation:{identity.episode_id}"
        return {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "scenario_id": identity.scenario_id,
            "nodes": [
                {"id": "intervention", "node_type": "intervention", "feature_vector": [1.0, 0.0, factual or 0.0, counter or 0.0], "provenance": provenance, "scenario_id": identity.scenario_id, "schema_version": GRAPH_SCHEMA_VERSION},
                {"id": variable, "node_type": "world_variable", "feature_vector": [0.0, base, factual or 0.0, counter or 0.0], "provenance": provenance, "scenario_id": identity.scenario_id, "schema_version": GRAPH_SCHEMA_VERSION},
            ],
            "edges": [
                {"id": f"effect:{variable}", "source": "intervention", "target": variable, "edge_type": edge_type, "signed_strength": signed, "confidence": confidence, "provenance": provenance, "canonical": True, "schema_version": GRAPH_SCHEMA_VERSION}
            ],
        }

    @staticmethod
    def compare(candidate: Any, state: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(candidate, Mapping) or not candidate.get("relations"):
            return {"verdict": "disabled_or_no_candidate", "agreement": None}
        relation = candidate["relations"][0]
        signed_effect = _number(relation.get("signed_expected_effect"))
        n4_relation = "support" if signed_effect is not None and signed_effect >= 0.0 else "conflict"
        cau, ctf = state.get("cau_link") or {}, state.get("ctf_checked") or {}
        canonical_support = bool(cau.get("helps_goal")) and ctf.get("supports_choice") is not False
        canonical_relation = "support" if canonical_support else "conflict"
        agreement = n4_relation == canonical_relation
        return {
            "verdict": "agreement" if agreement else "disagreement",
            "agreement": agreement,
            "n4_relation": n4_relation,
            "canonical_relation": canonical_relation,
            "backend_disagreement": relation.get("canonical_disagreement"),
            "authorities": ["CAU", "CTF", "C-GWM"],
            "decision_influence": "none",
        }


class N5Adapter:
    organ = "N5"
    capability = "deterministic_ingestion"
    authority_ceiling = NeuralMode.PROVISIONAL
    consumer = "SMG+MFM"
    reference_id = "rnfe:N5:deterministic_ingestion:chunker-v1"

    def __init__(self) -> None:
        self.chunker = DeterministicChunker(max_bytes=256)

    def infer(self, request: NeuralInferenceRequest, context: Mapping[str, Any]) -> BackendOutput:
        inputs = context["inputs"]
        memory_text = " ".join(
            json.dumps(item.get("structure", {}), sort_keys=True, default=str)[:512]
            for item in inputs.get("memory_hits", [])[:3]
        )
        content = "\n".join(
            part for part in (str(inputs.get("proposition") or ""), str(inputs.get("formula") or ""), memory_text) if part
        )[:2048]
        chunks = [chunk.to_dict() for chunk in self.chunker.chunk(content)]
        return BackendOutput(
            candidate_output={
                "status": "ok", "backend": "deterministic_chunker", "hnet_active": False,
                "fallback_declared": "hnet_artifact_unavailable", "chunks": chunks,
                "memory_candidates": [{"chunk": chunk, "provenance": "scenario_observation+formula+authorized_memory", "promotion": "requires_existing_mfm_gate"} for chunk in chunks],
            },
            confidence=1.0, uncertainty=0.0,
            cost={"chunks": len(chunks), "ram_mb": 0.25, "vram_mb": 0.0},
        )

    def fallback(self, identity: SymbiosisIdentity) -> Mapping[str, Any]:
        return {"chunks": [], "memory_candidates": [], "fallback": "disabled"}


class N6Adapter:
    organ = "N6"
    capability = "structural_evolution_proposal"
    authority_ceiling = NeuralMode.SHADOW
    consumer = "sandbox+certification+autoevolution"
    reference_id = "rnfe:N6:structural_evolution_proposal:bounded-reference-v1"

    def infer(self, request: NeuralInferenceRequest, context: Mapping[str, Any]) -> BackendOutput:
        identity: SymbiosisIdentity = context["identity"]
        viability = context.get("viability") or {}
        reasoning = context.get("reasoning") or {}
        margin = float(viability.get("viability_margin", 0.0) or 0.0)
        trace = reasoning.get("trace") or []
        cost = sum(float((item.get("detail") or {}).get("cost", 0.0) or 0.0) for item in trace)
        proposal = None
        if margin < 0.75 or cost > 5.0:
            proposal = {
                "schema_version": "n6-structural-proposal-v1",
                "proposal_id": f"n6-{identity.episode_id}",
                "mutation_type": "optional_family_budget", "target": "reasoning_optional_budget",
                "value": "decrease_one", "expected_gain": round(max(0.01, (0.75 - margin) * 0.2), 6),
                "rollback_token": f"shadow-{identity.trace_group_id}", "lineage_id": identity.lineage_id,
            }
        sandbox = {
            "evaluated": proposal is not None,
            "safe": proposal is not None and proposal["mutation_type"] == "optional_family_budget",
            "applied": False,
            "reason": "shadow_no_mutation" if proposal is not None else "no_degradation_trigger",
        }
        return BackendOutput(
            candidate_output={
                "status": "proposed" if proposal is not None else "abstained",
                "proposal": proposal, "sandbox": sandbox,
                "sandbox_verdict": "shadow_safe_not_applied" if sandbox["safe"] else "abstained",
                "applied": False, "consumers": ["sandbox", "certification", "autoevolution_evidence"],
            },
            confidence=0.5 if proposal is not None else 1.0,
            uncertainty=0.5 if proposal is not None else 0.0,
            cost={"evaluations": 1, "ram_mb": 0.05, "vram_mb": 0.0},
        )

    def fallback(self, identity: SymbiosisIdentity) -> Mapping[str, Any]:
        return {"status": "disabled", "proposal": None, "applied": False}


def canonical_adapter_registry() -> dict[str, CanonicalOrganAdapter]:
    adapters: list[CanonicalOrganAdapter] = [
        N1Adapter(), N2Adapter(), N3Adapter(), N4Adapter(), N5Adapter(), N6Adapter()
    ]
    registry = {adapter.organ: adapter for adapter in adapters}
    expected = {f"N{index}" for index in range(1, 7)}
    if len(registry) != len(adapters) or set(registry) != expected:
        raise RuntimeError("canonical_neural_adapter_registry_incomplete_or_duplicated")
    return registry


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
