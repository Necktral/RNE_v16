"""Factory para construir backends/facade de storage."""

from __future__ import annotations

from .config import StorageConfig
from .facade import StorageFacade
from .interfaces import StorageBackend


class StorageFactory:
    @staticmethod
    def create_backend(config: StorageConfig) -> StorageBackend:
        if config.mode == "sqlite":
            from .backends.sqlite_store import SQLiteStorageBackend

            return SQLiteStorageBackend(config.sqlite_db_path)
        if config.mode == "postgres":
            if not config.postgres_dsn:
                raise ValueError("postgres_dsn es obligatorio para modo postgres")
            from .backends.postgres_store import PostgresStorageBackend

            return PostgresStorageBackend(config.postgres_dsn)
        if config.mode == "hybrid":
            if not config.postgres_dsn:
                raise ValueError("postgres_dsn es obligatorio para modo hybrid")
            from .backends.hybrid_store import HybridStorageBackend
            from .backends.postgres_store import PostgresStorageBackend
            from .backends.sqlite_store import SQLiteStorageBackend

            primary = PostgresStorageBackend(config.postgres_dsn)
            fallback = SQLiteStorageBackend(config.sqlite_db_path)
            return HybridStorageBackend(
                primary=primary,
                fallback=fallback,
                prefer_primary_reads=config.prefer_postgres_reads,
                strict_dual_write=config.strict_dual_write,
            )
        raise ValueError(f"Modo de storage no soportado: {config.mode}")

    @staticmethod
    def create_facade(config: StorageConfig) -> StorageFacade:
        backend = StorageFactory.create_backend(config)
        return StorageFacade(backend=backend, config=config)
