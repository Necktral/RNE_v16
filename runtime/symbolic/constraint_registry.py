"""Registro de restricciones con ámbito estricto de episodio."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from .schemas import ConstraintRecord, CoreReport, append_canonical_jsonl_many


def semantic_segment(value: object) -> str:
    """Escapa un segmento de ID sin permitir que introduzca separadores."""

    return quote(str(value), safe="-._~")


def constraint_id(
    *,
    replay_unit_id: str,
    logical_time: int,
    kind: str,
    source: str,
    rule_id: str,
) -> str:
    return (
        f"unit/{semantic_segment(replay_unit_id)}/t/{int(logical_time)}"
        f"/kind/{semantic_segment(kind)}/source/{semantic_segment(source)}"
        f"/rule/{semantic_segment(rule_id)}"
    )


class ConstraintRegistry:
    """Colecciona constraints y reportes, rechazando colisiones ambiguas."""

    def __init__(self, *, replay_unit_id: str, logical_time: int):
        self.replay_unit_id = replay_unit_id
        self.logical_time = int(logical_time)
        self._records: dict[str, ConstraintRecord] = {}
        self._reports: list[CoreReport] = []

    @property
    def records(self) -> tuple[ConstraintRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    @property
    def latest_core_report(self) -> CoreReport | None:
        return self._reports[-1] if self._reports else None

    def register(self, record: ConstraintRecord) -> None:
        previous = self._records.get(record.constraint_id)
        if previous is not None and previous != record:
            raise ValueError(f"Colisión de constraint_id: {record.constraint_id}")
        self._records[record.constraint_id] = record

    def get_core_report(
        self,
        status: str,
        core_ids: Iterable[str],
        all_ids: Iterable[str],
        model: dict[str, Any] | None = None,
    ) -> CoreReport:
        report = CoreReport(
            replay_unit_id=self.replay_unit_id,
            logical_time=self.logical_time,
            status=str(status).upper(),
            core_ids=tuple(sorted(str(item) for item in core_ids)),
            all_constraint_ids=tuple(sorted(str(item) for item in all_ids)),
            model=model,
        )
        self._reports.append(report)
        return report

    def export_jsonl(self, path: str | Path) -> Path:
        return append_canonical_jsonl_many(path, self.records)
