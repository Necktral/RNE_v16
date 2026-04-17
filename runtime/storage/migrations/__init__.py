"""Migraciones de persistencia runtime/storage."""

from __future__ import annotations


def migrate_sqlite_ledger_to_postgres(*args, **kwargs):
    from .sqlite_to_postgres import migrate_sqlite_ledger_to_postgres as _impl

    return _impl(*args, **kwargs)


__all__ = ["migrate_sqlite_ledger_to_postgres"]
