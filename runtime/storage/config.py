"""Configuracion de la capa de storage RNFE."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal


StorageMode = Literal["sqlite", "postgres", "hybrid"]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class StorageConfig:
    mode: StorageMode
    sqlite_db_path: str
    postgres_dsn: str | None
    artifact_root: Path
    prefer_postgres_reads: bool
    strict_dual_write: bool

    @classmethod
    def from_env(cls) -> "StorageConfig":
        mode = os.environ.get("RNFE_STORAGE_MODE", "sqlite").strip().lower()
        if mode not in {"sqlite", "postgres", "hybrid"}:
            mode = "sqlite"

        sqlite_db_path = os.environ.get("AEON_EVENT_DB", "aeon_event_log.db")
        postgres_dsn = os.environ.get("RNFE_POSTGRES_DSN")
        
        # Eliminar hardcode /mnt/d: usar directorio portátil por defecto
        default_artifact_root = os.path.join(os.getcwd(), "rnfe_artifacts")
        artifact_root = Path(
            os.environ.get("RNFE_ARTIFACT_ROOT", default_artifact_root)
        )
        
        prefer_postgres_reads = _env_bool("RNFE_STORAGE_PREFER_POSTGRES_READS", True)
        strict_dual_write = _env_bool("RNFE_STORAGE_STRICT_DUAL_WRITE", False)
        config = cls(
            mode=mode,  # type: ignore[arg-type]
            sqlite_db_path=sqlite_db_path,
            postgres_dsn=postgres_dsn,
            artifact_root=artifact_root,
            prefer_postgres_reads=prefer_postgres_reads,
            strict_dual_write=strict_dual_write,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.mode in {"postgres", "hybrid"} and not self.postgres_dsn:
            raise ValueError(
                "RNFE_POSTGRES_DSN es obligatorio cuando RNFE_STORAGE_MODE=postgres|hybrid"
            )

