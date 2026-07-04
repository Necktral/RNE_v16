from pathlib import Path

from runtime.core.event_log_sqlite import EventLogSQLite
from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler


def test_runtime_smoke_without_archive_dependency(tmp_path: Path):
    db_path = tmp_path / "smoke.db"
    ledger = EventLogSQLite(str(db_path))
    ledger.log_event("smoke", {"ok": True})
    rows = ledger.get_events(limit=1)
    assert rows and rows[0]["event"] == "smoke"

    scheduler = MetaScheduler()
    result = scheduler.run({"seed": 1})
    assert result["meta_family"] == "META"
    assert result["sequence"] == ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]

