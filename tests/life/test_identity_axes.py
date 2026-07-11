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
