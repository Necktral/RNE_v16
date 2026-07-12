"""Frontera única entre el episodio vivo y los órganos neuronales N1-N6."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

from runtime.neural.config import NeuralRuntimeConfig
from runtime.neural.contracts import (
    BackendOutput,
    CausalContextView,
    InferenceScope,
    NeuralInferenceRequest,
    NeuralMode,
    ResourceSnapshot,
)
from runtime.neural.organs.n5_ingest import DeterministicChunker
from runtime.neural.registry import LazyBackendRegistry
from runtime.neural.runtime import NeuralRuntime

from .contracts import (
    OrganTrace,
    SymbiosisIdentity,
    SymbiosisTrace,
    canonical_sha256,
)


_OPTIONAL_FAMILIES = ("IND", "PLAN", "OPT", "NESY", "IMAGINATION", "EVO_SEARCH")


@dataclass(slots=True)
class _EpisodeSession:
    trace: SymbiosisTrace
    inputs: dict[str, Any]
    entries: dict[str, OrganTrace] = field(default_factory=dict)


class SymbioticNeuralCoordinator:
    """Coordina evidencia neuronal sin duplicar scheduler ni autoridad.

    Los algoritmos integrados son deterministas y están declarados como referencias;
    H-Net, Mamba2 y los pesos no entrenados no se activan. N0 conserva OFF, fallback,
    presupuestos, trazas y techo de autoridad para cada ejecución.
    """

    def __init__(
        self,
        *,
        storage: Any,
        config: NeuralRuntimeConfig | None = None,
    ) -> None:
        self.storage = storage
        self.config = config or NeuralRuntimeConfig.from_env()
        self.runtime = NeuralRuntime(
            config=self.config,
            registry=LazyBackendRegistry(artifact_root=Path(".")),
            storage=storage,
        )
        self._sessions: dict[str, _EpisodeSession] = {}
        self._temporal: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._disabled_organs = {
            item.strip().upper()
            for item in os.environ.get("RNFE_NEURAL_DISABLED_ORGANS", "").split(",")
            if item.strip()
        }

    def begin_episode(
        self,
        *,
        identity: SymbiosisIdentity,
        observation: Mapping[str, Any],
        formula: str,
        proposition: str,
        memory_hits: list[dict[str, Any]],
        scenario_metadata: Mapping[str, Any],
        causal_attestation: Mapping[str, Any],
        resources: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        resource_snapshot = ResourceSnapshot.from_mapping(resources)
        inputs = {
            "observation": dict(observation),
            "formula": formula,
            "proposition": proposition,
            "memory_hits": list(memory_hits),
            "scenario_metadata": dict(scenario_metadata),
            "causal_attestation": dict(causal_attestation),
            "resources": asdict(resource_snapshot),
        }
        session = _EpisodeSession(trace=SymbiosisTrace(identity=identity), inputs=inputs)
        self._sessions[identity.episode_id] = session

        n5 = self._execute(
            session,
            organ="N5",
            capability="deterministic_ingestion",
            authority_ceiling=NeuralMode.PROVISIONAL,
            consumer="SMG+MFM",
            producer=lambda _request: self._n5(inputs),
            fallback={"chunks": [], "memory_candidates": [], "fallback": "disabled"},
        )
        session.entries["N5"].consumer_verdict = (
            "disabled" if self.config.mode is NeuralMode.OFF else "consumed_by_SMG+MFM"
        )
        n3 = self._execute(
            session,
            organ="N3",
            capability="temporal_reference_state",
            authority_ceiling=NeuralMode.PROVISIONAL,
            consumer="next_episode+MFM+continuity",
            producer=lambda _request: self._n3(identity, inputs),
            fallback={"status": "disabled", "state_key": list(self._temporal_key(identity))},
        )
        session.entries["N3"].consumer_verdict = (
            "disabled"
            if self.config.mode is NeuralMode.OFF
            else "consumed_by_next_reasoning+MFM+continuity"
        )
        n1 = self._execute(
            session,
            organ="N1",
            capability="family_routing_proposal",
            authority_ceiling=NeuralMode.SHADOW,
            consumer="MetaSchedulerComparator",
            producer=lambda _request: self._n1(inputs, n3),
            fallback={"status": "disabled", "proposed_families": []},
        )
        self._execute(
            session,
            organ="N4",
            capability="typed_causal_proposal",
            authority_ceiling=NeuralMode.SHADOW,
            consumer="CAU+CTF+C-GWM comparator",
            producer=lambda _request: self._n4(identity, inputs),
            fallback={"status": "disabled", "relations": []},
        )
        return {
            "schema_version": "neural-symbiosis-signals-v1",
            "trace_group_id": identity.trace_group_id,
            "n1_proposal": n1,
            "n3_temporal": n3,
            "n5_ingestion": n5,
        }

    def consume_reasoning(
        self,
        *,
        episode_id: str,
        reasoning: Mapping[str, Any],
        lotf_valid: bool,
    ) -> dict[str, Any]:
        session = self._session(episode_id)
        state = dict(reasoning.get("state") or {})
        selected = [str(item).upper() for item in reasoning.get("sequence") or []]

        n1 = session.entries["N1"]
        proposed = set((n1.candidate or {}).get("proposed_families") or [])
        overlap = sorted(proposed.intersection(selected))
        n1.consumer_verdict = (
            "disabled"
            if n1.effective_mode == NeuralMode.OFF.value
            else f"compared:overlap={','.join(overlap) or 'none'}"
        )

        n2 = self._execute(
            session,
            organ="N2",
            capability="semantic_neurosymbolic_candidate",
            authority_ceiling=NeuralMode.SHADOW,
            consumer="DED+LOT-F+NESY",
            producer=lambda _request: self._n2(state, session.inputs, lotf_valid),
            fallback={"status": "disabled", "verified": False},
        )
        session.entries["N2"].consumer_verdict = str(n2.get("verdict", "disabled"))

        n4_entry = session.entries["N4"]
        n4_comparison = self._compare_n4(n4_entry.candidate, state)
        n4_entry.consumer_verdict = n4_comparison["verdict"]
        if isinstance(n4_entry.candidate, dict):
            n4_entry.candidate["canonical_comparison"] = n4_comparison
            n4_entry.candidate_hash = canonical_sha256(n4_entry.candidate)

        return {
            "n1_scheduler_comparison": {
                "proposed": sorted(proposed),
                "scheduler_selected": selected,
                "overlap": overlap,
                "scheduler_authority_preserved": True,
            },
            "n2_verification": n2,
            "n4_comparison": n4_comparison,
        }

    def prepare_certification(
        self,
        *,
        episode_id: str,
        viability: Mapping[str, Any],
        reasoning: Mapping[str, Any],
    ) -> dict[str, Any]:
        session = self._session(episode_id)
        n6 = self._execute(
            session,
            organ="N6",
            capability="structural_evolution_proposal",
            authority_ceiling=NeuralMode.SHADOW,
            consumer="sandbox+certification+autoevolution",
            producer=lambda _request: self._n6(session, viability, reasoning),
            fallback={"status": "disabled", "proposal": None, "applied": False},
        )
        session.entries["N6"].consumer_verdict = str(n6.get("sandbox_verdict", "disabled"))
        health = asdict(self.runtime.trace_health)
        session.trace.trace_health = health
        return self.certification_block(episode_id)

    def certification_block(self, episode_id: str) -> dict[str, Any]:
        session = self._session(episode_id)
        entries = [entry.to_dict(include_candidate=True) for entry in session.trace.organs]
        return {
            "schema_version": "neural-symbiosis-certificate-v1",
            "trace_group_id": session.trace.identity.trace_group_id,
            "runtime": {
                "organ": "N0",
                "mode": self.config.mode.value,
                "loaded_artifacts": self.runtime.registry.loaded_count,
                "trace_health": asdict(self.runtime.trace_health),
            },
            "organs_executed": [
                row["organ"] for row in entries if row["effective_mode"] != NeuralMode.OFF.value
            ],
            "mode": self.config.mode.value,
            "candidates": entries,
            "abstentions": [row["organ"] for row in entries if row["abstained"]],
            "fallbacks": [
                {"organ": row["organ"], "reason": row["fallback_reason"]}
                for row in entries
                if row["fallback_reason"]
            ],
            "disagreements": [
                row["candidate"].get("canonical_comparison")
                for row in entries
                if row["organ"] == "N4" and isinstance(row.get("candidate"), dict)
            ],
            "costs": {row["organ"]: row["cost"] for row in entries},
            "authority_effective": {
                row["organ"]: row["authority_ceiling"] for row in entries
            },
            "trace_completeness": session.trace.is_complete,
            "trace_health": asdict(self.runtime.trace_health),
            "resource_snapshot": dict(session.inputs.get("resources") or {}),
            "verdict_influence": "none",
        }

    def export_temporal_state(self) -> dict[str, Any]:
        """Serializa solo el estado N3 determinista para el checkpoint soberano."""

        return {
            "schema_version": "n3-temporal-checkpoint-v1",
            "entries": [
                {"state_key": list(key), "state": dict(state)}
                for key, state in sorted(self._temporal.items())
            ],
        }

    def restore_temporal_state(self, payload: Mapping[str, Any] | None) -> int:
        """Restaura N3 validando la clave fuerte; ignora entradas mal formadas."""

        data = dict(payload or {})
        if data.get("schema_version") != "n3-temporal-checkpoint-v1":
            return 0
        restored = 0
        for item in data.get("entries") or []:
            if not isinstance(item, Mapping):
                continue
            raw_key = item.get("state_key")
            raw_state = item.get("state")
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
            self._temporal[key] = state
            restored += 1
        return restored

    def finalize_episode(
        self,
        *,
        episode_id: str,
        outcome: Mapping[str, Any],
        certificate: Mapping[str, Any],
        reward: Mapping[str, Any],
    ) -> dict[str, Any]:
        session = self._session(episode_id)
        n1 = session.entries["N1"]
        n1.consumer_verdict = (
            f"{n1.consumer_verdict}|reward={float(reward.get('reward', 0.0) or 0.0):.4f}"
            f"|certificate={certificate.get('verdict')}"
        )
        n4 = session.entries["N4"]
        n4.consumer_verdict = f"{n4.consumer_verdict}|certificate_metadata=consumed"
        n6 = session.entries["N6"]
        n6.consumer_verdict = f"{n6.consumer_verdict}|certificate={certificate.get('verdict')}"
        session.trace.episode_result = {
            "intervention": outcome.get("intervention"),
            "relation_kind": outcome.get("relation_kind"),
            "reward": reward.get("reward"),
        }
        session.trace.certificate = dict(certificate)
        session.trace.trace_health = asdict(self.runtime.trace_health)
        payload = session.trace.to_dict(include_candidates=True)
        persisted = self.runtime.persist_symbiosis_event(
            event_type="neural.symbiosis.completed",
            payload=payload,
            run_id=session.trace.identity.run_id,
        )
        payload["trace_persisted"] = persisted
        payload["trace_health"] = asdict(self.runtime.trace_health)
        return payload

    def _execute(
        self,
        session: _EpisodeSession,
        *,
        organ: str,
        capability: str,
        authority_ceiling: NeuralMode,
        consumer: str,
        producer: Callable[[NeuralInferenceRequest], Any],
        fallback: Any,
    ) -> Any:
        identity = session.trace.identity
        request = NeuralInferenceRequest(
            inference_id=f"sym-{organ.lower()}-{uuid4().hex[:12]}",
            run_id=identity.run_id,
            organ=organ,
            capability=capability,
            payload={"identity": identity.to_dict(), "inputs": session.inputs},
            scope=InferenceScope.LIVE,
            resources=ResourceSnapshot.from_mapping(session.inputs.get("resources")),
            causal_context=CausalContextView(
                organism_id=identity.organism_id,
                decision_id=identity.decision_id,
                episode_id=identity.episode_id,
                trace_id=identity.trace_group_id,
            ),
        )
        result = self.runtime.infer_reference(
            request=request,
            producer=producer,
            fallback_output=fallback,
            reference_id=f"rnfe:{organ}:{capability}:reference-v1",
            authority_ceiling=authority_ceiling,
            enabled=organ not in self._disabled_organs,
        )
        candidate = result.candidate_output
        entry = OrganTrace(
            identity=identity,
            organ=organ,
            capability=capability,
            requested_mode=result.requested_mode.value,
            effective_mode=result.effective_mode.value,
            authority_ceiling=authority_ceiling.value,
            input_hash=canonical_sha256(request.payload),
            candidate_hash=canonical_sha256(candidate) if candidate is not None else None,
            consumer=consumer,
            consumer_verdict=("disabled" if result.effective_mode is NeuralMode.OFF else "pending"),
            latency_ms=result.latency_ms,
            ram_mb=_number(result.cost.get("ram_mb")),
            vram_mb=_number(result.cost.get("vram_mb")),
            fallback_reason=result.fallback_reason,
            manifest_sha256=result.manifest_sha256,
            artifact_sha256=None,
            candidate=candidate,
            abstained=bool(isinstance(candidate, Mapping) and candidate.get("status") == "abstained"),
            cost=result.cost,
        )
        session.entries[organ] = entry
        session.trace.organs.append(entry)
        return candidate if candidate is not None else fallback

    @staticmethod
    def _n5(inputs: Mapping[str, Any]) -> BackendOutput:
        memory_text = " ".join(
            json.dumps(item.get("structure", {}), sort_keys=True, default=str)[:512]
            for item in inputs.get("memory_hits", [])[:3]
        )
        content = "\n".join(
            part
            for part in (
                str(inputs.get("proposition") or ""),
                str(inputs.get("formula") or ""),
                memory_text,
            )
            if part
        )[:2048]
        chunks = [chunk.to_dict() for chunk in DeterministicChunker(max_bytes=256).chunk(content)]
        candidates = [
            {
                "chunk": chunk,
                "provenance": "scenario_observation+formula+authorized_memory",
                "promotion": "requires_existing_mfm_gate",
            }
            for chunk in chunks
        ]
        return BackendOutput(
            candidate_output={
                "status": "ok",
                "backend": "deterministic_chunker",
                "hnet_active": False,
                "fallback_declared": "hnet_artifact_unavailable",
                "chunks": chunks,
                "memory_candidates": candidates,
            },
            confidence=1.0,
            uncertainty=0.0,
            cost={"chunks": len(chunks), "ram_mb": 0.25, "vram_mb": 0.0},
        )

    def _n3(self, identity: SymbiosisIdentity, inputs: Mapping[str, Any]) -> BackendOutput:
        key = self._temporal_key(identity)
        observation = inputs.get("observation") or {}
        metadata = inputs.get("scenario_metadata") or {}
        main_variable = str(metadata.get("main_variable") or "temperature")
        value = _number(observation.get(main_variable)) or 0.0
        previous = self._temporal.get(key)
        previous_value = _number((previous or {}).get("value"))
        trend = 0.0 if previous_value is None else value - previous_value
        count = int((previous or {}).get("episode_count", 0)) + 1
        state = {
            "status": "ok",
            "backend": "reference_temporal_filter",
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
        self._temporal[key] = state
        return BackendOutput(
            candidate_output=state,
            confidence=1.0 - state["uncertainty"],
            uncertainty=state["uncertainty"],
            cost={"state_entries": len(self._temporal), "ram_mb": 0.1, "vram_mb": 0.0},
        )

    @staticmethod
    def _n1(inputs: Mapping[str, Any], temporal: Mapping[str, Any]) -> BackendOutput:
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
        for family in _OPTIONAL_FAMILIES:
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
        proposed = [row["family"] for row in ranked if row["score"] > 0.25][:2]
        status = "ok" if proposed else "abstained"
        return BackendOutput(
            candidate_output={
                "status": status,
                "backend": "deterministic_context_router",
                "trained_model": False,
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

    @staticmethod
    def _n4(identity: SymbiosisIdentity, inputs: Mapping[str, Any]) -> BackendOutput:
        attestation = dict(inputs.get("causal_attestation") or {})
        observation = inputs.get("observation") or {}
        metadata = inputs.get("scenario_metadata") or {}
        main_variable = str(metadata.get("main_variable") or "temperature")
        base = _number(observation.get(main_variable)) or 0.0
        factual_value = _number(attestation.get("factual_value"))
        counter_value = _number(attestation.get("counterfactual_value"))
        factual_delta = _number(attestation.get("factual_delta"))
        counter_delta = _number(attestation.get("counterfactual_delta"))
        if factual_delta is None and factual_value is not None:
            factual_delta = factual_value - base
        if counter_delta is None and counter_value is not None:
            counter_delta = counter_value - base
        insufficient = factual_delta is None or counter_delta is None
        supports_choice = attestation.get("supports_choice")
        predicted_relation = "unknown" if insufficient else (
            "support" if supports_choice is True else "conflict"
        )
        candidate = {
            "schema_version": "n4-live-causal-proposal-v1",
            "scenario_id": identity.scenario_id,
            "input_snapshot_hash": canonical_sha256(attestation),
            "relations": [
                {
                    "source": "intervention",
                    "target": main_variable,
                    "edge_type": "interventional_effect",
                    "factual_delta": factual_delta,
                    "counterfactual_delta": counter_delta,
                    "predicted_relation": predicted_relation,
                    "confidence": 0.0 if insufficient else min(1.0, abs((factual_delta or 0.0) - (counter_delta or 0.0)) * 5.0),
                }
            ],
            "authority": {
                "proposal_only": True,
                "may_mutate_graph": False,
                "may_choose_intervention": False,
                "may_authorize_action": False,
                "authoritative_systems": ["CAU", "CTF", "C-GWM", "LOT-F", "DED"],
            },
            "fallback": {
                "required": insufficient,
                "route": "CAU+CTF+C-GWM",
                "reason": "insufficient_live_snapshot" if insufficient else None,
            },
        }
        return BackendOutput(
            candidate_output=candidate,
            confidence=candidate["relations"][0]["confidence"],
            uncertainty=1.0 - candidate["relations"][0]["confidence"],
            cost={"relations": 1, "ram_mb": 0.05, "vram_mb": 0.0},
        )

    @staticmethod
    def _n2(
        state: Mapping[str, Any],
        inputs: Mapping[str, Any],
        lotf_valid: bool,
    ) -> BackendOutput:
        from runtime.reasoning.families import nesy

        nesy_result = nesy.execute({**dict(state), "_symbiotic_n2_verify": True})
        nesy_delta = dict(nesy_result.get("state_delta") or {})
        ded_verified = bool(state.get("ded_validated"))
        nesy_verified = bool(nesy_delta.get("nesy_coherent"))
        verified = ded_verified and lotf_valid and nesy_verified
        proposition = str(state.get("ded_conclusion") or inputs.get("formula") or "")
        verdict = "accepted_as_shadow_evidence" if verified else "rejected_by_symbolic_verifier"
        return BackendOutput(
            candidate_output={
                "status": "ok",
                "proposition": proposition,
                "verified": verified,
                "verdict": verdict,
                "verification": {
                    "DED": ded_verified,
                    "LOT-F": bool(lotf_valid),
                    "NESY": nesy_verified,
                },
                "nesy": nesy_result,
                "authority": "DED+LOT-F+NESY",
            },
            confidence=float(nesy_result.get("confidence", 0.0) or 0.0),
            uncertainty=0.0 if verified else 1.0,
            cost={"verifiers": 3, "ram_mb": 0.05, "vram_mb": 0.0},
        )

    @staticmethod
    def _compare_n4(candidate: Any, state: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(candidate, Mapping) or not candidate.get("relations"):
            return {"verdict": "disabled_or_no_candidate", "agreement": None}
        relation = candidate["relations"][0]
        n4_relation = relation.get("predicted_relation")
        cau = state.get("cau_link") or {}
        ctf = state.get("ctf_checked") or {}
        canonical_support = bool(cau.get("helps_goal")) and ctf.get("supports_choice") is not False
        canonical_relation = "support" if canonical_support else "conflict"
        agreement = n4_relation == canonical_relation
        return {
            "verdict": "agreement" if agreement else "disagreement",
            "agreement": agreement,
            "n4_relation": n4_relation,
            "canonical_relation": canonical_relation,
            "authorities": ["CAU", "CTF", "C-GWM"],
            "decision_influence": "none",
        }

    @staticmethod
    def _n6(
        session: _EpisodeSession,
        viability: Mapping[str, Any],
        reasoning: Mapping[str, Any],
    ) -> BackendOutput:
        margin = float(viability.get("viability_margin", 0.0) or 0.0)
        trace = reasoning.get("trace") or []
        cost = sum(float((item.get("detail") or {}).get("cost", 0.0) or 0.0) for item in trace)
        proposal = None
        status = "abstained"
        if margin < 0.75 or cost > 5.0:
            proposal = {
                "schema_version": "n6-structural-proposal-v1",
                "proposal_id": f"n6-{session.trace.identity.episode_id}",
                "mutation_type": "optional_family_budget",
                "target": "reasoning_optional_budget",
                "value": "decrease_one",
                "expected_gain": round(max(0.01, (0.75 - margin) * 0.2), 6),
                "rollback_token": f"shadow-{session.trace.identity.trace_group_id}",
                "lineage_id": session.trace.identity.lineage_id,
            }
            status = "proposed"
        sandbox = {
            "evaluated": proposal is not None,
            "safe": proposal is not None and proposal["mutation_type"] == "optional_family_budget",
            "applied": False,
            "reason": "shadow_no_mutation" if proposal is not None else "no_degradation_trigger",
        }
        return BackendOutput(
            candidate_output={
                "status": status,
                "proposal": proposal,
                "sandbox": sandbox,
                "sandbox_verdict": "shadow_safe_not_applied" if sandbox["safe"] else "abstained",
                "applied": False,
                "consumers": ["sandbox", "certification", "autoevolution_evidence"],
            },
            confidence=0.5 if proposal is not None else 1.0,
            uncertainty=0.5 if proposal is not None else 0.0,
            cost={"evaluations": 1, "ram_mb": 0.05, "vram_mb": 0.0},
        )

    @staticmethod
    def _temporal_key(identity: SymbiosisIdentity) -> tuple[str, str, str]:
        return (identity.organism_id, identity.scenario_id, identity.lineage_id)

    def _session(self, episode_id: str) -> _EpisodeSession:
        try:
            return self._sessions[episode_id]
        except KeyError as exc:
            raise KeyError(f"symbiosis_episode_unknown:{episode_id}") from exc


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
