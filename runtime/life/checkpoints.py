"""Checkpointing canonico del Life Kernel."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from runtime.core.checkpoint_kinds import LIFE_CHECKPOINT_KIND
from runtime.organism.lineage import LineageState
from runtime.organism.state import OrganismState
from runtime.storage.records import ArtifactRecord, utc_now_iso

from .contracts import AutonomyDecision, GoalState, VitalSignsSnapshot
from .serialization import jsonable, lineage_to_payload, organism_to_payload


logger = logging.getLogger(__name__)

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
        organism_id: str | None = None,
        lineage_id: str | None = None,
        decision: AutonomyDecision | None = None,
        runner_knobs: Dict[str, Any] | None = None,
        scale_state: Dict[str, Any] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        checkpoint_id = f"life-ckpt-{uuid4().hex[:12]}"
        # B41 (decisión 4/6): el organism_id vive en el PAYLOAD del checkpoint (no se
        # toca el IdentityState frozen). Fallback legado: si el kernel no lo provee,
        # el genoma es el run_id bajo el que se escribió (organism_id := run_id).
        organism_id = str(organism_id or run_id)
        lineage_id = str(lineage_id or getattr(lineage, "lineage_id", "") or "")
        payload = {
            "version": LIFE_CHECKPOINT_VERSION,
            "checkpoint_id": checkpoint_id,
            "created_at": utc_now_iso(),
            "run_id": run_id,
            "organism_id": organism_id,
            "lineage_id": lineage_id,
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
                # B41: organism_id/lineage_id en metadata ⇒ descubrimiento por genoma
                # (decisión 5, filtro por metadata; path-segment diferido, cero migración).
                "organism_id": organism_id,
                "lineage_id": lineage_id,
                "episode_count": organism_state.episode_count,
                "total_steps": total_steps,
                "mode": vital_signs.mode,
                "healthy": vital_signs.is_restorable,
            },
        )
        self.storage.append_event(
            event_type="life.checkpoint.saved",
            run_id=run_id,
            source="life_kernel",
            payload={
                "checkpoint_id": checkpoint_id,
                "artifact_id": artifact.artifact_id,
                "organism_id": organism_id,
                "lineage_id": lineage_id,
                "episode_count": organism_state.episode_count,
                "mode": vital_signs.mode,
                "healthy": vital_signs.is_restorable,
            },
        )
        return artifact

    def load_latest_payload(
        self,
        *,
        run_id: str | None = None,
        organism_id: str | None = None,
        healthy_only: bool = False,
    ) -> tuple[Dict[str, Any], ArtifactRecord] | None:
        # B41 (§3.8): el descubrimiento se puede scopear por GENOMA (organism_id) en vez
        # de por corrida. Con run_id efímero, "el último checkpoint de cualquier corrida"
        # es incorrecto; debe ser "el último de ESTE organismo". Cuando se da organism_id,
        # escaneamos todos los checkpoints (run_id=None) y filtramos por el genoma vía
        # metadata (fallback legado: el artifact.run_id ES el organism_id legado, §4.1).
        # Sin organism_id, se conserva la mecánica por run_id (byte-idéntico).
        scan_run_id = None if organism_id else run_id
        # Al buscar un REFUGIO sano (healthy_only) o al scopear por organismo escaneamos
        # mucho más hondo: tras un período largo dañado (cuarentena) o a través de varias
        # corridas efímeras, el último yo sano del organismo puede estar lejos en el
        # tiempo, y debe seguir siendo alcanzable — sobrevivir es condición de aprender.
        artifacts = self.storage.list_artifacts(
            run_id=scan_run_id,
            kind=LIFE_CHECKPOINT_KIND,
            limit=400 if (healthy_only or organism_id) else 20,
        )
        for artifact in artifacts:
            if healthy_only and not bool((artifact.metadata or {}).get("healthy")):
                continue
            if organism_id is not None:
                # metadata.organism_id (post-B41) o, para checkpoints legados sin ese
                # campo, el run_id de la corrida bajo la que se escribió (== genoma legado).
                art_org = str((artifact.metadata or {}).get("organism_id") or artifact.run_id or "")
                if art_org != organism_id:
                    continue
            path = Path(artifact.abs_path)
            if not path.exists() or not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or not payload.get("version"):
                continue
            # B73: si esto es la búsqueda de un REFUGIO, la salud se re-deriva del
            # CONTENIDO del archivo, no del flag de metadata (ver _payload_is_restorable).
            if healthy_only and not self._payload_is_restorable(
                payload, artifact_id=getattr(artifact, "artifact_id", "?")
            ):
                continue
            return payload, artifact
        return None

    @staticmethod
    def _payload_is_restorable(payload: Dict[str, Any], *, artifact_id: str = "?") -> bool:
        """B73 — el refugio se decide sobre el archivo, no sobre el flag guardado.

        ``metadata["healthy"]`` se escribe al GUARDAR, con el snapshot completo en memoria:
        es un índice para no leer todos los artifacts, no una prueba de que el archivo que
        estamos por adoptar siga sano. Si el checkpoint se trunca o se corrompe después de
        escrito, el flag sigue diciendo ``True``. Ese flag era la ÚNICA compuerta del camino
        E5 (``kernel._restore_latest_healthy_checkpoint``), que adoptaba el payload sin
        volver a mirarle los vitales jamás.

        Ahora los vitales se re-derivan del payload real: un checkpoint con vitales
        ausentes/truncadas ya no puede pasar por refugio (``from_dict`` los marca NO
        VERIFICADOS y ``is_restorable`` se cierra). Un checkpoint sano re-deriva
        ``is_restorable=True`` y sigue siendo elegible exactamente como antes.

        FAIL-SAFE **Y TOTAL**: un candidato malo se **saltea**, jamás mata la búsqueda.
        ``from_dict`` no es total —un ``vital_signs`` JSON-válido pero MAL TIPADO
        (``"risk_score": "high"``) revienta en ``float(...)``— y esta función corre dentro
        del loop de selección de refugio: si la excepción escapara, abortaría la búsqueda
        ENTERA y un checkpoint **sano** que estuviera detrás del corrupto **nunca se
        alcanzaría**. Endurecer la compuerta no puede volverla frágil.

        Y el rechazo **deja constancia**: negarse a refugiarse sin decir por qué es el mismo
        callejón sin salida (cuarentena muda) que el camino E5 existe para romper.
        """
        vital_payload = payload.get("vital_signs")
        if not isinstance(vital_payload, dict):
            logger.warning(
                "[life.refuge] candidato %s RECHAZADO: bloque 'vital_signs' ausente o no-dict "
                "(checkpoint truncado o ajeno). No sirve de refugio.",
                artifact_id,
            )
            return False
        try:
            vitals = VitalSignsSnapshot.from_dict(vital_payload)
        except Exception as exc:  # noqa: BLE001 - candidato ilegible: se saltea, no se muere
            logger.warning(
                "[life.refuge] candidato %s RECHAZADO: vitales ilegibles (%s: %s). "
                "Se saltea y se sigue buscando; NO se aborta la búsqueda de refugio.",
                artifact_id,
                type(exc).__name__,
                exc,
            )
            return False
        if vitals.is_restorable:
            return True
        logger.warning(
            "[life.refuge] candidato %s RECHAZADO: %s",
            artifact_id,
            vitals.restorability_report(),
        )
        return False
