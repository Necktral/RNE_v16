"""Replay shadow puro de ventanas vitales con adaptadores candidatos."""

from __future__ import annotations

from typing import Any

from runtime.neural.contracts import (
    CausalContextView,
    InferenceScope,
    NeuralInferenceRequest,
    ResourceSnapshot,
)
from runtime.neural.integration.adapters import canonical_adapter_registry
from runtime.neural.integration.contracts import SymbiosisIdentity, canonical_sha256

from .trajectory_window import DynamicTrajectoryWindow


REPLAY_SCHEMA_VERSION = "organism-trajectory-replay-v1"


class ShadowTrajectoryReplay:
    """No recibe storage ni autoridades vivas; sólo compara propuestas."""

    def replay(self, window: DynamicTrajectoryWindow, *, organ_id: str) -> dict[str, Any]:
        adapter = canonical_adapter_registry()[organ_id]
        rows = []
        for transition in window.transitions:
            replay_context = dict(transition.neural_state_delta.get("replay_context") or {})
            historical = next(
                (
                    item for item in transition.action_proposals
                    if item.get("organ") == organ_id
                ),
                None,
            )
            if not replay_context or historical is None:
                rows.append({
                    "transition_id": transition.transition_id,
                    "evidence_insufficient": True,
                    "contract_errors": ["historical_replay_context_or_candidate_missing"],
                })
                continue
            identity = SymbiosisIdentity(
                trace_group_id=transition.trace_group_id,
                organism_id=window.organism_id,
                lineage_id=window.lineage_id,
                run_id=transition.state_after.run_id,
                episode_id=transition.state_after.episode_id,
                scenario_id=str(replay_context.get("scenario_id") or "unknown"),
            )
            request = NeuralInferenceRequest(
                inference_id=f"replay-{transition.transition_id}-{organ_id.lower()}",
                run_id=identity.run_id,
                organ=organ_id,
                capability=adapter.capability,
                payload={"identity": identity.to_dict(), "inputs": replay_context.get("inputs") or {}},
                scope=InferenceScope.LAB,
                resources=ResourceSnapshot.from_mapping(replay_context.get("resources")),
                causal_context=CausalContextView(
                    organism_id=identity.organism_id,
                    episode_id=identity.episode_id,
                    trace_id=identity.trace_group_id,
                ),
            )
            context = {
                "identity": identity,
                "inputs": replay_context.get("inputs") or {},
                "n3_temporal": replay_context.get("n3_temporal") or {},
                "reasoning_state": replay_context.get("reasoning_state") or {},
                "lotf_valid": replay_context.get("lotf_valid"),
                "viability": replay_context.get("viability") or {},
                "reasoning": replay_context.get("reasoning") or {},
            }
            try:
                output = adapter.infer(request, context)
                candidate_hash = canonical_sha256(output.candidate_output)
                rows.append({
                    "transition_id": transition.transition_id,
                    "historical_candidate_hash": historical.get("candidate_hash"),
                    "replay_candidate_hash": candidate_hash,
                    "proposal_diverged": candidate_hash != historical.get("candidate_hash"),
                    "uncertainty_difference": None,
                    "cost_difference": None,
                    "causal_agreement": None,
                    "observable_regret": transition.reward.get("reward"),
                    "contract_errors": [],
                    "regime_compatibility": transition.regime_after.get(
                        "compatibility_with_previous_regime"
                    ),
                    "evidence_insufficient": False,
                    "estimated_impact_only": True,
                })
            except Exception as exc:
                rows.append({
                    "transition_id": transition.transition_id,
                    "evidence_insufficient": True,
                    "contract_errors": [f"{type(exc).__name__}:{exc}"],
                })
        return {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "window_id": window.window_id,
            "adapter": organ_id,
            "authority_effect": "none",
            "writes_performed": False,
            "original_chain_untouched": True,
            "results": rows,
        }
