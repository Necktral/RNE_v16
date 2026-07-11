"""B41 — separación de ejes de identidad del organismo (P-CADENA-CAUSAL).

Tests aditivos: los tres ejes (run_id efímero / organism_id genoma persistente /
lineage_id linaje) se acuñan distintos; el restore re-keya a organism_id (persiste)
mientras run_id cambia y la memoria queda intacta; dos vidas con el mismo organism_id
comparten experiencia cross-vida.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from runtime.life import LifeKernel, LifeKernelConfig
from runtime.organism.identity import (
    LINEAGE_ID_PREFIX,
    ORGANISM_ID_PREFIX,
    RUN_ID_PREFIX,
)
from runtime.storage import StorageConfig, StorageFactory


def _storage(tmp_path: Path, name: str = "life.db"):
    return StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(tmp_path / name),
            postgres_dsn=None,
            artifact_root=tmp_path / "artifacts",
            prefer_postgres_reads=False,
            strict_dual_write=False,
        )
    )


def test_genesis_mints_distinct_identity_axes(tmp_path: Path):
    """Génesis acuña run_id, organism_id y lineage_id genuinamente distintos."""
    storage = _storage(tmp_path)
    kernel = LifeKernel(
        config=LifeKernelConfig(
            run_id="life-axes",
            scenarios=("thermal_homeostasis",),
            restore=False,
            enable_msrc=False,
        ),
        storage=storage,
    )

    # Los tres ejes son distintos entre sí (el defecto B41: los tres colapsaban en run_id).
    assert kernel.run_id == "life-axes"
    assert kernel.organism_id and kernel.organism_id != kernel.run_id
    assert kernel.lineage_id and kernel.lineage_id != kernel.run_id
    assert kernel.organism_id != kernel.lineage_id

    # Convención SSOT: organism_id genuino (org-), lineage_id genuino (lin-), NO derivados del run.
    assert kernel.organism_id.startswith(ORGANISM_ID_PREFIX)
    assert kernel.lineage_id.startswith(LINEAGE_ID_PREFIX)
    assert kernel.run_id not in kernel.organism_id
    assert kernel.run_id not in kernel.lineage_id

    # El estado y el linaje quedan anclados al genoma, no a la corrida.
    assert kernel.organism_state is not None
    assert kernel.organism_state.state_id == f"state-0-{kernel.organism_id}"
    assert kernel.lineage is not None
    assert kernel.lineage.lineage_id == kernel.lineage_id


def test_ephemeral_run_id_minted_when_not_configured(tmp_path: Path):
    """Sin config.run_id, el run_id es efímero (prefijo life-) y no colapsa con el genoma."""
    storage = _storage(tmp_path)
    kernel = LifeKernel(
        config=LifeKernelConfig(
            scenarios=("thermal_homeostasis",),
            restore=False,
            enable_msrc=False,
        ),
        storage=storage,
    )
    assert kernel.run_id.startswith(RUN_ID_PREFIX)
    assert kernel.organism_id != kernel.run_id


def test_config_organism_id_wins_over_env(tmp_path: Path, monkeypatch):
    """Precedencia de génesis: config gana sobre entorno (RNFE_ORGANISM_ID)."""
    monkeypatch.setenv("RNFE_ORGANISM_ID", "org-from-env")
    storage = _storage(tmp_path)
    kernel = LifeKernel(
        config=LifeKernelConfig(
            run_id="life-precedence",
            organism_id="org-from-config",
            scenarios=("thermal_homeostasis",),
            restore=False,
            enable_msrc=False,
        ),
        storage=storage,
    )
    assert kernel.organism_id == "org-from-config"


def test_env_organism_id_used_when_no_config(tmp_path: Path, monkeypatch):
    """Sin config.organism_id, se usa RNFE_ORGANISM_ID (segundo en la precedencia)."""
    monkeypatch.setenv("RNFE_ORGANISM_ID", "org-env-genome")
    storage = _storage(tmp_path)
    kernel = LifeKernel(
        config=LifeKernelConfig(
            run_id="life-env",
            scenarios=("thermal_homeostasis",),
            restore=False,
            enable_msrc=False,
        ),
        storage=storage,
    )
    assert kernel.organism_id == "org-env-genome"


def test_restore_scopes_to_organism(tmp_path: Path):
    """Re-keying del restore: organism_id persiste, run_id cambia, memoria intacta.

    Dos organismos (org-A, org-B) en el MISMO storage con run_id efímero: al restaurar
    org-A se recupera SU checkpoint (no el de org-B), el organism_id se mantiene como
    clave persistente y el run_id se re-acuña nuevo.
    """
    storage = _storage(tmp_path)

    # org-A vive DOS pasos; org-B vive UNO (para distinguir qué genoma se restaura).
    a1 = LifeKernel(
        config=LifeKernelConfig(
            organism_id="org-A", scenarios=("thermal_homeostasis",),
            restore=False, enable_msrc=False,
        ),
        storage=storage,
    )
    a1.step(external_input=0.05)
    a1.step(external_input=0.04)
    a1_run_id = a1.run_id
    a1_episodes = a1.organism_state.episode_count

    b1 = LifeKernel(
        config=LifeKernelConfig(
            organism_id="org-B", scenarios=("thermal_homeostasis",),
            restore=False, enable_msrc=False,
        ),
        storage=storage,
    )
    b1.step(external_input=0.05)

    # run_id efímero: la corrida A tiene su propia marca (no colapsa con el genoma).
    assert a1_run_id.startswith(RUN_ID_PREFIX)
    assert a1_run_id != "org-A"

    # Restaurar org-A: nueva corrida, mismo genoma.
    a2 = LifeKernel(
        config=LifeKernelConfig(
            organism_id="org-A", scenarios=("thermal_homeostasis",),
            restore=True, enable_msrc=False,
        ),
        storage=storage,
    )

    # organism_id PERSISTE como clave; run_id CAMBIA (re-acuñado, efímero).
    assert a2.organism_id == "org-A"
    assert a2.run_id != a1_run_id
    # Se restauró el genoma de A (2 episodios), NO el de B (1 episodio) ⇒ scoping correcto.
    assert a2.organism_state.episode_count == a1_episodes == 2
    assert a2.total_steps == 2

    # Memoria/estado intactos: un paso más continúa la vida de A sin discontinuidad.
    result = a2.step(external_input=0.04)
    assert result.step_index == 3
    assert a2.organism_state.episode_count == a1_episodes + 1

    # El evento life.identity.restored carga la genealogía de corridas (organism_id +
    # run_id anterior + run_id nuevo).
    events = storage.list_events(
        run_id=a2.run_id, event_types=["life.identity.restored"], limit=5
    )
    assert events
    payload = events[0].payload
    assert payload["organism_id"] == "org-A"
    assert payload["previous_run_id"] == a1_run_id
    assert payload["run_id"] == a2.run_id
    assert payload["previous_run_id"] != payload["run_id"]


def test_cross_run_experience_via_organism_id(tmp_path: Path, monkeypatch):
    """Dos vidas (run_id distinto) con el MISMO organism_id comparten experiencia."""
    monkeypatch.setenv("RNFE_EXPERIENCE", "1")
    storage = _storage(tmp_path)

    life_a = LifeKernel(
        config=LifeKernelConfig(
            organism_id="org-shared", run_id="life-a",
            scenarios=("thermal_homeostasis",), restore=False, enable_msrc=False,
        ),
        storage=storage,
    )
    life_a.step(external_input=0.05)

    life_b = LifeKernel(
        config=LifeKernelConfig(
            organism_id="org-shared", run_id="life-b",
            scenarios=("thermal_homeostasis",), restore=False, enable_msrc=False,
        ),
        storage=storage,
    )
    life_b.step(external_input=0.05)

    from runtime.organism.experience import ExperienceStore

    store = ExperienceStore(storage=storage)
    shared = store.recall(organism_id="org-shared")
    # Ambas vidas grabaron bajo el mismo genoma ⇒ la experiencia es compartida.
    assert len(shared) >= 2

    # Aislamiento: otro genoma no ve nada de esta experiencia.
    assert store.recall(organism_id="org-other") == []

    # La procedencia de corrida difiere (run_id distinto) bajo el mismo namespace de genoma.
    records = storage.retrieve_memory_records(run_id="org-shared", scales=["experience"])
    run_ids = {(r.metadata or {}).get("run_id") for r in records}
    assert {"life-a", "life-b"} <= run_ids


def test_causal_context_absent_by_default(tmp_path: Path, monkeypatch):
    """Feature ausente (RNFE_CAUSAL_CONTEXT off): la clave no viaja ⇒ byte-idéntico."""
    monkeypatch.delenv("RNFE_CAUSAL_CONTEXT", raising=False)
    storage = _storage(tmp_path)
    kernel = LifeKernel(
        config=LifeKernelConfig(
            run_id="life-noctx", organism_id="org-noctx",
            scenarios=("thermal_homeostasis",), restore=False, enable_msrc=False,
        ),
        storage=storage,
    )
    kernel.step(external_input=0.05)
    events = storage.list_events(
        run_id="life-noctx", event_types=["life.step.completed"], limit=5
    )
    assert events
    assert "causal_context" not in events[0].payload


def test_causal_context_injected_when_enabled(tmp_path: Path, monkeypatch):
    """Con RNFE_CAUSAL_CONTEXT=1 el sobre viaja aditivamente en life.step.completed."""
    monkeypatch.setenv("RNFE_CAUSAL_CONTEXT", "1")
    storage = _storage(tmp_path)
    kernel = LifeKernel(
        config=LifeKernelConfig(
            run_id="life-ctx", organism_id="org-ctx",
            scenarios=("thermal_homeostasis",), restore=False, enable_msrc=False,
        ),
        storage=storage,
    )
    kernel.step(external_input=0.05)
    events = storage.list_events(
        run_id="life-ctx", event_types=["life.step.completed"], limit=5
    )
    assert events
    ctx = events[0].payload.get("causal_context")
    assert ctx is not None
    assert ctx["schema_version"] == "causal_context.v1"
    # El sobre carga los tres ejes del organismo + el hilo del step.
    assert ctx["organism_id"] == "org-ctx"
    assert ctx["run_id"] == "life-ctx"
    assert ctx["lineage_id"] == kernel.lineage_id
    assert ctx["trace_group_id"].startswith("tg-org-ctx-")
    assert ctx["decision_id"]
