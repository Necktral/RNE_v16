"""Frontera única entre el episodio vivo y los órganos neuronales N1-N6."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from runtime.neural.config import NeuralRuntimeConfig
from runtime.neural.contracts import (
    CausalContextView,
    InferenceScope,
    NeuralInferenceRequest,
    NeuralMode,
    ResourceSnapshot,
)
from runtime.neural.registry import LazyBackendRegistry
from runtime.neural.runtime import NeuralRuntime

from .adapters import N3Adapter, N4Adapter, canonical_adapter_registry
from .contracts import (
    OrganTrace,
    SymbiosisIdentity,
    SymbiosisTrace,
    canonical_sha256,
)


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
        self._adapters = canonical_adapter_registry()
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
            context={"identity": identity, "inputs": inputs},
        )
        session.entries["N5"].consumer_verdict = (
            "disabled" if self.config.mode is NeuralMode.OFF else "consumed_by_SMG+MFM"
        )
        n3 = self._execute(
            session,
            organ="N3",
            context={"identity": identity, "inputs": inputs},
        )
        session.entries["N3"].consumer_verdict = (
            "disabled"
            if self.config.mode is NeuralMode.OFF
            else "consumed_by_next_reasoning+MFM+continuity"
        )
        n1 = self._execute(
            session,
            organ="N1",
            context={"identity": identity, "inputs": inputs, "n3_temporal": n3},
        )
        self._execute(
            session,
            organ="N4",
            context={"identity": identity, "inputs": inputs},
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
            context={
                "identity": session.trace.identity,
                "inputs": session.inputs,
                "reasoning_state": state,
                "lotf_valid": lotf_valid,
            },
        )
        session.entries["N2"].consumer_verdict = str(n2.get("verdict", "disabled"))

        n4_entry = session.entries["N4"]
        n4_adapter = self._adapters["N4"]
        if not isinstance(n4_adapter, N4Adapter):
            raise RuntimeError("canonical_n4_adapter_type_mismatch")
        n4_comparison = n4_adapter.compare(n4_entry.candidate, state)
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
            context={
                "identity": session.trace.identity,
                "inputs": session.inputs,
                "viability": viability,
                "reasoning": reasoning,
            },
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

        adapter = self._adapters["N3"]
        if not isinstance(adapter, N3Adapter):
            raise RuntimeError("canonical_n3_adapter_type_mismatch")
        return adapter.export_state()

    def restore_temporal_state(self, payload: Mapping[str, Any] | None) -> int:
        """Restaura N3 validando la clave fuerte; ignora entradas mal formadas."""

        adapter = self._adapters["N3"]
        if not isinstance(adapter, N3Adapter):
            raise RuntimeError("canonical_n3_adapter_type_mismatch")
        return adapter.restore_state(payload)

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
        context: Mapping[str, Any],
    ) -> Any:
        adapter = self._adapters[organ]
        identity = session.trace.identity
        request = NeuralInferenceRequest(
            inference_id=f"sym-{organ.lower()}-{uuid4().hex[:12]}",
            run_id=identity.run_id,
            organ=organ,
            capability=adapter.capability,
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
            producer=lambda inference_request: adapter.infer(inference_request, context),
            fallback_output=adapter.fallback(identity),
            reference_id=adapter.reference_id,
            authority_ceiling=adapter.authority_ceiling,
            enabled=organ not in self._disabled_organs,
        )
        candidate = result.candidate_output
        entry = OrganTrace(
            identity=identity,
            organ=organ,
            capability=adapter.capability,
            requested_mode=result.requested_mode.value,
            effective_mode=result.effective_mode.value,
            authority_ceiling=adapter.authority_ceiling.value,
            input_hash=canonical_sha256(request.payload),
            candidate_hash=canonical_sha256(candidate) if candidate is not None else None,
            consumer=adapter.consumer,
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
        return candidate if candidate is not None else adapter.fallback(identity)

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
