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


_DOTENV_LOADED = False


def _load_dotenv_once() -> None:
    """Carga un ``.env`` del repo (si existe) para variables RNFE/AEON aún no definidas.

    Sin dependencias externas. Las variables ya presentes en el entorno tienen
    prioridad (sólo rellena las ausentes), de modo que el storage use PostgreSQL por
    defecto al correr desde el repo sin sourcing manual, pero el entorno explícito
    (p. ej. el aislamiento de tests) siga mandando.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    candidates = []
    try:
        candidates.append(Path(__file__).resolve().parents[2] / ".env")
    except Exception:
        pass
    candidates.append(Path.cwd() / ".env")
    seen: set = set()
    for env_path in candidates:
        if env_path in seen:
            continue
        seen.add(env_path)
        try:
            if not env_path.is_file():
                continue
            for raw in env_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
            break
        except Exception:
            continue


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
        _load_dotenv_once()
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

