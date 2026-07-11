"""Persistencia de identidad soberana del organismo."""

from __future__ import annotations

from runtime.organism.identity import legacy_organism_id

from .checkpoints import CheckpointManager
from .contracts import GoalState, RestoredIdentity, VitalSignsSnapshot
from .serialization import lineage_from_payload, organism_from_payload


class OrganismPersistence:
    """API publica para reconstruir identidad viva desde storage."""

    def __init__(self, *, storage):
        self.storage = storage
        self.checkpoints = CheckpointManager(storage=storage)

    def load_latest_identity(
        self,
        *,
        run_id: str | None = None,
        organism_id: str | None = None,
    ) -> RestoredIdentity | None:
        # B41: si se da organism_id, el descubrimiento se scopea por GENOMA (cross-corrida);
        # si no, se conserva la mecánica legada por run_id.
        loaded = self.checkpoints.load_latest_payload(run_id=run_id, organism_id=organism_id)
        if loaded is None:
            return None
        payload, artifact = loaded
        goals = [
            GoalState.from_dict(item)
            for item in payload.get("goals", [])
            if isinstance(item, dict)
        ]
        vital_payload = payload.get("vital_signs")
        vital_signs = (
            VitalSignsSnapshot.from_dict(vital_payload)
            if isinstance(vital_payload, dict)
            else None
        )
        lineage = lineage_from_payload(payload.get("lineage"))
        # Genoma restaurado: payload.organism_id, con fallback legado organism_id := run_id
        # (§4.1, mapeo identidad — la memoria vieja ya está namespaceada por ese valor).
        restored_organism_id = legacy_organism_id(payload)
        restored_lineage_id = str(
            payload.get("lineage_id")
            or getattr(lineage, "lineage_id", "")
            or f"lin-legacy-{restored_organism_id}"
        )
        return RestoredIdentity(
            run_id=str(payload.get("run_id") or artifact.run_id or "unknown"),
            organism_state=organism_from_payload(payload.get("organism_state")),
            lineage=lineage,
            goals=goals,
            vital_signs=vital_signs,
            total_steps=int(payload.get("total_steps", 0)),
            scenario_index=int(payload.get("scenario_index", 0)),
            checkpoint_payload=payload,
            checkpoint_artifact_id=artifact.artifact_id,
            organism_id=restored_organism_id,
            lineage_id=restored_lineage_id,
        )
