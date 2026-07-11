"""SSOT de acuñación de identidad del organismo (B41 · P-CADENA-CAUSAL).

Tres ejes de identidad con ciclos de vida distintos (canon f2.4: A-M5 continuidad,
A-M8 herencia como medida μ_t, A-M10 existencia por continuidad):

- ``run_id``      — la corrida (EFÍMERO, operativo): marca de ESTA ejecución.
- ``organism_id`` — el genoma (PERSISTE entre corridas): ADN portable, namespace
  de memoria viva y experiencia (``M_t``).
- ``lineage_id``  — el linaje evolutivo μ_t (abarca varios organismos): la medida
  de herencia (D8, reaparición viable entre semillas/entornos/corridas).

**Una única función de acuñación compartida (SSOT, decisión ratificada 1):** el
kernel soberano acuña ``organism_id``/``lineage_id`` con estas funciones; el runner
los RECIBE vía ``set_organism_id`` y NUNCA acuña con convención propia; el runner
standalone llama a ESTAS MISMAS funciones. Se elimina la divergencia histórica
``org-{run_id}`` (runner) vs. la convención del kernel.

Este módulo vive en ``runtime.organism`` a propósito: tanto ``runtime.world`` (runner)
como ``runtime.life`` (kernel) ya dependen de ``runtime.organism``; ubicarlo aquí evita
el ciclo ``world -> life`` (el kernel ya importa el runner). No importa ``world`` ni
``life``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional
from uuid import uuid4

RUN_ID_PREFIX = "life-"
ORGANISM_ID_PREFIX = "org-"
LINEAGE_ID_PREFIX = "lin-"

_TRUE = {"1", "true", "yes", "on"}


# ── Acuñación de los tres ejes ────────────────────────────────────────────────

def mint_run_id() -> str:
    """Acuña un ``run_id`` EFÍMERO para la corrida actual (re-acuñado cada proceso)."""
    return f"{RUN_ID_PREFIX}{uuid4().hex[:12]}"


def mint_organism_id() -> str:
    """Acuña un ``organism_id`` genuinamente nuevo (genoma portable, NO derivado del run)."""
    return f"{ORGANISM_ID_PREFIX}{uuid4().hex}"


def mint_lineage_id() -> str:
    """Acuña un ``lineage_id`` genuinamente nuevo (linaje con un solo organismo)."""
    return f"{LINEAGE_ID_PREFIX}{uuid4().hex}"


# ── Precedencia de génesis de ``organism_id`` (decisión ratificada 3) ─────────

def resolve_parent_organism_id() -> Optional[str]:
    """STUB documentado (B41 §1.2, origen 2): herencia de ancestro como interfaz
    SIN implementar.

    Cuando exista *fork con descendencia*, esta interfaz devolverá el
    ``parent_organism_id`` para que el nuevo organismo herede el ``lineage_id`` del
    ancestro y acuñe ``organism_id`` propio (D8). Hoy NO hay fork vivo cableado:
    devuelve ``None``. No se elimina para dejar el punto de extensión explícito.
    """
    return None


def resolve_organism_id(config_organism_id: str | None = None) -> Optional[str]:
    """Resuelve el ``organism_id`` de génesis por precedencia (config gana sobre entorno).

    Orden: config (``LifeKernelConfig.organism_id``) → entorno (``RNFE_ORGANISM_ID``)
    → ancestro (stub, ver :func:`resolve_parent_organism_id`) → ``None``.

    ``None`` significa "no hay genoma conocido a vincular" ⇒ el llamador acuña un
    genoma genuinamente nuevo con :func:`mint_organism_id`.
    """
    if config_organism_id:
        return str(config_organism_id)
    env = os.environ.get("RNFE_ORGANISM_ID", "").strip()
    if env:
        return env
    # Nivel ancestro: interfaz sin implementar (stub). None ⇒ génesis genuina.
    resolve_parent_organism_id()
    return None


def legacy_organism_id(payload: Dict[str, Any] | None) -> str:
    """Deriva el ``organism_id`` de un checkpoint payload, con fallback legado.

    Compatibilidad hacia atrás (decisión ratificada 6, §4.1): para todo artefacto
    pre-B41 sin campo ``organism_id``, el genoma legado ES exactamente el ``run_id``
    bajo el que fue escrito (mapeo identidad, cero bytes movidos):
    ``organism_id_legacy := run_id``.
    """
    data = payload or {}
    return str(data.get("organism_id") or data.get("run_id") or "unknown")


# ── CausalContext.v1 — sobre de correlación de auditoría (decisión 7) ─────────

def causal_context_enabled() -> bool:
    """True si el sobre ``CausalContext`` viaja en los payloads (aditivo, gated).

    Off por defecto (``RNFE_CAUSAL_CONTEXT`` sin setear) ⇒ la clave está ausente y
    el comportamiento es byte-idéntico a pre-B41.
    """
    return os.environ.get("RNFE_CAUSAL_CONTEXT", "").strip().lower() in _TRUE


def mint_trace_group_id(*, organism_id: str, step_index: int) -> str:
    """Acuña el hilo que ata decisión↔episodio↔traza↔certificado↔promoción de un step."""
    return f"tg-{organism_id}-{step_index}-{uuid4().hex[:8]}"


@dataclass(frozen=True, slots=True)
class CausalContext:
    """Sobre de correlación de auditoría, estable a través de corridas y máquinas.

    Contrato inmutable, ``frozen``, versionado. Se acuña 1× por ``step()`` y viaja
    como clave ADITIVA en payloads de eventos/runner/trazas. Con la feature ausente
    (``causal_context_enabled() is False``) NUNCA se inyecta ⇒ byte-idéntico.

    Reconstrucción de la cadena (objetivo B44/B45): dado un ``organism_id``, agrupar
    por ``trace_group_id`` reconstruye cada episodio; seguir ``parent_trace_group_id``
    reconstruye el árbol de rollbacks/forks — sin ordenar por timestamp.
    """

    schema_version: str = "causal_context.v1"
    organism_id: str = ""
    lineage_id: str = ""
    run_id: str = ""
    trace_group_id: str = ""
    parent_trace_group_id: str | None = None
    decision_id: str | None = None
    step_index: int = -1

    @classmethod
    def for_step(
        cls,
        *,
        organism_id: str,
        lineage_id: str,
        run_id: str,
        step_index: int,
        decision_id: str | None = None,
        parent_trace_group_id: str | None = None,
        trace_group_id: str | None = None,
    ) -> "CausalContext":
        """Acuña el sobre de un ``step()``: un ``trace_group_id`` fresco por episodio."""
        return cls(
            organism_id=str(organism_id or ""),
            lineage_id=str(lineage_id or ""),
            run_id=str(run_id or ""),
            trace_group_id=trace_group_id
            or mint_trace_group_id(organism_id=str(organism_id or ""), step_index=int(step_index)),
            parent_trace_group_id=parent_trace_group_id,
            decision_id=decision_id,
            step_index=int(step_index),
        )

    def with_decision(self, decision_id: str | None) -> "CausalContext":
        """Copia el sobre fijando el ``decision_id`` (cierra decisión→episodio)."""
        return CausalContext(
            schema_version=self.schema_version,
            organism_id=self.organism_id,
            lineage_id=self.lineage_id,
            run_id=self.run_id,
            trace_group_id=self.trace_group_id,
            parent_trace_group_id=self.parent_trace_group_id,
            decision_id=decision_id,
            step_index=self.step_index,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "CausalContext":
        data = payload or {}
        return cls(
            schema_version=str(data.get("schema_version") or "causal_context.v1"),
            organism_id=str(data.get("organism_id") or ""),
            lineage_id=str(data.get("lineage_id") or ""),
            run_id=str(data.get("run_id") or ""),
            trace_group_id=str(data.get("trace_group_id") or ""),
            parent_trace_group_id=data.get("parent_trace_group_id"),
            decision_id=data.get("decision_id"),
            step_index=int(data.get("step_index", -1)),
        )
