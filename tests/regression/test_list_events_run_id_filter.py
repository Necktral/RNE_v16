"""Regresión: SQLiteStorageBackend.list_events filtra run_id en SQL antes del LIMIT.

Bug histórico: el ledger aplicaba `ORDER BY id DESC LIMIT ?` de forma **global** y el
filtro por run_id se hacía en Python *después*. Si los eventos del run no estaban entre
los `limit` más recientes globales, se perdían — y como `trace_integrity` /
certificación dependen de encontrar el evento `episode.closed` del run, un episodio
válido podía fallar por un artefacto de storage. Postgres ya filtraba en SQL.
"""

from runtime.storage.backends.sqlite_store import SQLiteStorageBackend
from runtime.storage.records import StoredEvent


def test_list_events_run_id_survives_global_limit(tmp_path):
    backend = SQLiteStorageBackend(str(tmp_path / "events.db"))

    # 1 evento del run objetivo...
    backend.append_event(
        StoredEvent(event_type="episode.closed", payload={}, run_id="R-target")
    )
    # ...seguido de MUCHOS eventos de otro run (ruido) que superan el limit global.
    for i in range(50):
        backend.append_event(
            StoredEvent(event_type="noise", payload={"i": i}, run_id="R-noise")
        )

    # Con el bug, list_events(limit=10) traía los 10 ruidos más recientes y perdía el
    # objetivo. Ahora el filtro run_id va en SQL antes del LIMIT.
    found = backend.list_events(run_id="R-target", limit=10)
    assert len(found) == 1, f"se perdió el evento del run por el LIMIT global: {found}"
    assert found[0].event_type == "episode.closed"
    assert found[0].run_id == "R-target"

    # El ruido se recupera por su propio run, sin contaminar al objetivo.
    noise = backend.list_events(run_id="R-noise", limit=10)
    assert len(noise) == 10
    assert all(e.run_id == "R-noise" for e in noise)

    # Sin run_id: comportamiento global intacto (los más recientes hasta el limit).
    recent = backend.list_events(limit=5)
    assert len(recent) == 5
