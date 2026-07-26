"""Persistencia canónica de reportes de núcleo."""

from __future__ import annotations

from pathlib import Path

from .schemas import CoreReport, append_canonical_jsonl


def persist_core_report(report: CoreReport, path: str | Path) -> Path:
    return append_canonical_jsonl(path, report)
