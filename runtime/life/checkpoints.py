"""Checkpointing canonico del Life Kernel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from runtime.organism.lineage import LineageState
from runtime.organism.state import OrganismState
from runtime.storage.records import ArtifactRecord, utc_now_iso

from .contracts import AutonomyDecision, GoalState, VitalSignsSnapshot
from .serialization import jsonable, lineage_to_payload, organism_to_payload


LIFE_CHECKPOINT_KIND = "life_checkpoint"
LIFE_CHECKPOINT_VERSION = "1.0.0"


class CheckpointManager:
    """Persiste y recupera checkpoints vivos como artifacts indexados."""

    def __init__(self, *, storage):
        self.storage = storage

    def save_checkpoint(
        self,
        *,
        run_id: str,
        organism_state: OrganismState,
        lineage: LineageState,
        goals: list[GoalState],
        vital_signs: VitalSignsSnapshot,
        total_steps: int,
        scenario_index: int,
        scenario_episode_index: int,
        memory_filter_mode: str,
        closure_profile: str,
        decision: AutonomyDecision | None = None,
        runner_knobs: Dict[str, Any] | None = None,
        scale_state: Dict[str, Any] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        checkpoint_id = f"life-ckpt-{uuid4().hex[:12]}"
        payload = {
            "version": LIFE_CHECKPOINT_VERSION,
            "checkpoint_id": checkpoint_id,
            "created_at": utc_now_iso(),
            "run_id": run_id,
            "total_steps": int(total_steps),
            "scenario_index": int(scenario_index),
            "scenario_episode_index": int(scenario_episode_index),
            "memory_filter_mode": memory_filter_mode,
            "closure_profile": closure_profile,
            "organism_state": organism_to_payload(organism_state),
            "lineage": lineage_to_payload(lineage),
            "goals": [goal.to_dict() for goal in goals],
            "vital_signs": vital_signs.to_dict(),
            "decision": decision.to_dict() if decision is not None else None,
            "runner_knobs": dict(runner_knobs or {}),
            "scale_state": dict(scale_state or {}),
            "metadata": dict(metadata or {}),
        }
        blob = json.dumps(jsonable(payload), ensure_ascii=True, sort_keys=True, indent=2)
        artifact = self.storage.materialize_artifact(
            run_id=run_id,
            kind=LIFE_CHECKPOINT_KIND,
            content=blob,
            filename=f"{checkpoint_id}.json",
            mime_type="application/json",
            metadata={
                "checkpoint_id": checkpoint_id,
                "episode_count": organism_state.episode_count,
                "total_steps": total_steps,
                "mode": vital_signs.mode,
                "healthy": vital_signs.is_stable,
            },
        )
        self.storage.append_event(
            event_type="life.checkpoint.saved",
            run_id=run_id,
            source="life_kernel",
            payload={
                "checkpoint_id": checkpoint_id,
                "artifact_id": artifact.artifact_id,
                "episode_count": organism_state.episode_count,
                "mode": vital_signs.mode,
                "healthy": vital_signs.is_stable,
            },
        )
        return artifact

    def load_latest_payload(
        self,
        *,
        run_id: str | None = None,
        healthy_only: bool = False,
    ) -> tuple[Dict[str, Any], ArtifactRecord] | None:
        artifacts = self.storage.list_artifacts(
            run_id=run_id,
            kind=LIFE_CHECKPOINT_KIND,
            limit=20,
        )
        for artifact in artifacts:
            if healthy_only and not bool((artifact.metadata or {}).get("healthy")):
                continue
            path = Path(artifact.abs_path)
            if not path.exists() or not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and payload.get("version"):
                return payload, artifact
        return None
