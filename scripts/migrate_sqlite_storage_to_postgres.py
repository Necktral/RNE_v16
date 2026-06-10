"""Migración COMPLETA del storage AEON de SQLite a PostgreSQL.

Extiende `runtime/storage/migrations/sqlite_to_postgres.py` (que sólo migra el
ledger `events`): migra también las 16 tablas auxiliares. El mapeo de columnas es
algorítmico — el único delta SQLite→PG es el sufijo `_jsonb` de las columnas JSON
(`metadata→metadata_jsonb`, `risk_json→risk_jsonb`, `*_artifacts→*_artifacts_jsonb`,
…) y unas pocas columnas TIMESTAMPTZ que PG rellena por DEFAULT (`created_at`/
`updated_at`). Sin claves foráneas → cualquier orden sirve. Idempotente
(`ON CONFLICT DO NOTHING`). Verifica conteos por tabla al final.

Uso:
    python scripts/migrate_sqlite_storage_to_postgres.py --sqlite-db <ruta> --postgres-dsn <dsn>
"""

from __future__ import annotations

import argparse
import json
import sqlite3

import psycopg

from runtime.storage.backends.postgres_store import PostgresStorageBackend
from runtime.storage.migrations.sqlite_to_postgres import (
    migrate_sqlite_ledger_to_postgres,
)


def _pg_table_for(sqlite_table: str) -> str:
    return "ledger_events" if sqlite_table == "events" else sqlite_table


def _map_pg_col(scol: str, pg_cols: set[str]) -> str | None:
    """Columna PG destino para una columna SQLite (regla de renombrado JSON)."""
    if scol in pg_cols:
        return scol
    if scol.endswith("_json") and (scol[:-5] + "_jsonb") in pg_cols:
        return scol[:-5] + "_jsonb"
    if (scol + "_jsonb") in pg_cols:
        return scol + "_jsonb"
    return None


def _migrate_aux_table(scon: sqlite3.Connection, pgcon, table: str) -> int:
    pgt = _pg_table_for(table)
    scols = [r[1] for r in scon.execute(f"PRAGMA table_info('{table}')")]
    pgtypes = {
        r[0]: r[1]
        for r in pgcon.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name=%s",
            (pgt,),
        )
    }
    pg_cols = set(pgtypes)
    pairs = []  # (scol, pgcol, is_jsonb, is_bool)
    for sc in scols:
        pc = _map_pg_col(sc, pg_cols)
        if pc is None:
            raise RuntimeError(f"[{table}] columna SQLite sin destino PG: {sc}")
        pairs.append((sc, pc, pgtypes[pc] == "jsonb", pgtypes[pc] == "boolean"))

    target_cols = [p[1] for p in pairs]
    placeholders = ["%s::jsonb" if p[2] else "%s" for p in pairs]
    sql = (
        f'INSERT INTO {pgt} ({", ".join(target_cols)}) '
        f'VALUES ({", ".join(placeholders)}) ON CONFLICT DO NOTHING'
    )

    sidx = {c: i for i, c in enumerate(scols)}
    quoted = ", ".join(f'"{c}"' for c in scols)
    rows = scon.execute(f'SELECT {quoted} FROM "{table}"').fetchall()

    data = []
    for row in rows:
        vals = []
        for (sc, _pc, is_jsonb, is_bool) in pairs:
            v = row[sidx[sc]]
            if is_jsonb:
                if v in (None, ""):
                    v = "{}"
                elif not isinstance(v, str):
                    v = json.dumps(v)
            elif is_bool and v is not None:
                v = bool(v)
            vals.append(v)
        data.append(vals)

    if data:
        with pgcon.cursor() as cur:
            cur.executemany(sql, data)
        pgcon.commit()
    return len(data)


def migrate(sqlite_db: str, postgres_dsn: str) -> None:
    # Construir el backend PG crea el esquema (las 18 tablas) vía _ensure_schema.
    PostgresStorageBackend(postgres_dsn).close()

    scon = sqlite3.connect(sqlite_db)
    stabs = [
        r[0]
        for r in scon.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]

    # 1) events -> ledger_events (mapeo especial: uuid/hash/legacy refs).
    n_events = migrate_sqlite_ledger_to_postgres(
        sqlite_db_path=sqlite_db, postgres_dsn=postgres_dsn
    )
    print(f"  events -> ledger_events: {n_events} filas")

    # 2) resto de tablas auxiliares (copia genérica).
    pgcon = psycopg.connect(postgres_dsn)
    for t in stabs:
        if t == "events":
            continue
        n = _migrate_aux_table(scon, pgcon, t)
        print(f"  {t}: {n} filas")

    # 3) verificación de conteos.
    print("\n=== VERIFICACIÓN (SQLite vs PG) ===")
    ok = True
    for t in stabs:
        pgt = _pg_table_for(t)
        s_n = scon.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        p_n = pgcon.execute(f"SELECT COUNT(*) FROM {pgt}").fetchone()[0]
        flag = "OK" if p_n >= s_n else "‼ FALTAN"
        if p_n < s_n:
            ok = False
        print(f"  {t:28s} SQLite={s_n:6d}  PG[{pgt}]={p_n:6d}  {flag}")
    pgcon.close()
    scon.close()
    print("\nRESULTADO:", "✅ migración verificada" if ok else "❌ faltan filas")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite-db", required=True)
    ap.add_argument("--postgres-dsn", required=True)
    migrate(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
