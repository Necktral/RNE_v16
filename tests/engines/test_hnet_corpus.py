"""Tests del extractor de corpus.

El test que importa es `test_ningun_run_id_cruza_el_split`: si un `run_id` aparece
en train y en val, la evaluación del fine-tuning miente y todo lo demás sobra.
Se construye una DB sintética con el mismo esquema real (incluida la ausencia de
columna `run_id` en `events`) para poder falsificarlo sin tocar la DB del
organismo.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from lab.hnet_chunker.corpus import (
    DB_SOURCES,
    CorpusBuilder,
    MixtureSpec,
    load_shard,
    split_of,
)


@pytest.fixture()
def fake_db(tmp_path):
    """Mismo esquema que aeon_event_log.db, incluido el detalle que muerde:
    `events` NO tiene columna run_id — vive dentro del payload."""
    db = tmp_path / "fake.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE events (id INTEGER, event_type TEXT, payload TEXT, timestamp TEXT);
        CREATE TABLE reasoning_traces (trace_id TEXT, run_id TEXT, step_index INTEGER,
            family TEXT, status TEXT, detail TEXT, trace_ts TEXT);
        CREATE TABLE memory_records (memory_id TEXT, run_id TEXT, episode_id TEXT, scale TEXT,
            structure_json TEXT, ttl_seconds INTEGER, no_interference INTEGER,
            certificate_id TEXT, ioc_proxy REAL, support_count INTEGER, metadata TEXT,
            created_at TEXT);
        CREATE TABLE organism_snapshots (snapshot_id TEXT, run_id TEXT, episode_id TEXT,
            trajectory_id TEXT, regime TEXT, snapshot_json TEXT, metadata TEXT, created_at TEXT);
        """
    )
    for i in range(60):
        run = f"run-{i % 12:03d}"
        con.execute(
            "INSERT INTO events VALUES (?,?,?,?)",
            (i, "smg.sign_created", json.dumps({"run_id": run, "i": i, "blob": "x" * 40}), "t"),
        )
        con.execute(
            "INSERT INTO reasoning_traces VALUES (?,?,?,?,?,?,?)",
            (f"tr{i}", run, 0, "DED", "ok", json.dumps({"family": "DED", "i": i}), "t"),
        )
        con.execute(
            "INSERT INTO memory_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"m{i}", run, "e", "s", json.dumps({"m": i}), 0, 0, "c", 0.0, 0, "{}", "t"),
        )
        con.execute(
            "INSERT INTO organism_snapshots VALUES (?,?,?,?,?,?,?,?)",
            (f"s{i}", run, "e", "t", "r", json.dumps({"s": i}), "{}", "t"),
        )
    # duplicado EXACTO de un payload, para verificar la dedup
    con.execute(
        "INSERT INTO events VALUES (?,?,?,?)",
        (999, "smg.sign_created", json.dumps({"run_id": "run-000", "i": 0, "blob": "x" * 40}), "t"),
    )
    con.commit()
    con.close()
    return db


def test_ningun_run_id_cruza_el_split(fake_db, tmp_path):
    """EL test. Un episodio produce eventos, trazas, memorias y snapshots con los
    MISMOS UUIDs. Si el split fuera aleatorio, el modelo vería en train el mismo
    episodio que se le pregunta en val, y el F1 de generalización sería ficción."""
    out = tmp_path / "corpus"
    CorpusBuilder(out, db_path=fake_db, repo_root=tmp_path / "vacio").build()

    train_runs, val_runs = set(), set()
    for table, _c, _r in DB_SOURCES:
        for split, bag in (("train", train_runs), ("val", val_runs)):
            shard = out / f"db_{table}.{split}.jsonl"
            if shard.exists():
                bag.update(d.run_id for d in load_shard(shard))

    assert train_runs and val_runs, "el split degeneró: una de las dos partes quedó vacía"
    assert not (train_runs & val_runs), f"FUGA de run_id entre train y val: {train_runs & val_runs}"


def test_el_split_es_el_mismo_en_las_cuatro_tablas(fake_db, tmp_path):
    out = tmp_path / "corpus"
    CorpusBuilder(out, db_path=fake_db, repo_root=tmp_path / "vacio").build()
    asignacion: dict[str, str] = {}
    for table, _c, _r in DB_SOURCES:
        for split in ("train", "val"):
            shard = out / f"db_{table}.{split}.jsonl"
            if not shard.exists():
                continue
            for d in load_shard(shard):
                prev = asignacion.setdefault(d.run_id, split)
                assert prev == split, f"{d.run_id} cayó en {prev} y en {split}"


def test_events_saca_el_run_id_del_payload_porque_no_hay_columna(fake_db, tmp_path):
    """`events` no tiene columna run_id. Si el extractor lo asumiera, todos los
    eventos quedarían sin split y se irían silenciosamente a un lado."""
    with sqlite3.connect(fake_db) as con:
        cols = {r[1] for r in con.execute("PRAGMA table_info(events)")}
    assert "run_id" not in cols

    b = CorpusBuilder(tmp_path / "c", db_path=fake_db, repo_root=tmp_path / "vacio")
    docs = list(b.iter_db("events", "payload", None))
    assert docs and all(d.run_id and d.run_id.startswith("run-") for d in docs)


def test_dedup_por_sha256(fake_db, tmp_path):
    b = CorpusBuilder(tmp_path / "c", db_path=fake_db, repo_root=tmp_path / "vacio")
    docs = list(b.iter_db("events", "payload", None))
    assert len(docs) == 60, "el payload duplicado exacto (61 filas) debió deduplicarse"
    assert len({d.sha256 for d in docs}) == len(docs)


def test_la_db_se_abre_read_only(fake_db, tmp_path):
    b = CorpusBuilder(tmp_path / "c", db_path=fake_db, repo_root=tmp_path / "vacio")
    con = b._connect()
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        con.execute("DELETE FROM events")
    con.close()


def test_manifiesto_lleva_procedencia_por_shard(fake_db, tmp_path):
    out = tmp_path / "corpus"
    man = CorpusBuilder(out, db_path=fake_db, repo_root=tmp_path / "vacio").build()
    assert man["shards"], "sin shards no hay procedencia"
    for s in man["shards"]:
        assert s["table"] or s["source"].startswith("repo")
        assert s["n_records"] > 0 and s["n_bytes"] > 0
        assert len(s["file_sha256"]) == 64
        assert s["content_kind"] in ("json", "prose", "code_py")
        assert s["split"] in ("train", "val")
    assert man["mixture"].startswith("NO DEFINIDA")


def test_el_split_es_determinista():
    a = [split_of(f"run-{i}") for i in range(200)]
    b = [split_of(f"run-{i}") for i in range(200)]
    assert a == b
    assert 0.05 < a.count("val") / len(a) < 0.30, "la fracción de val se fue de rango"


def test_mixture_sin_pesos_explota():
    """La proporción estructura:prosa es un PARÁMETRO. Un default acá sería una
    decisión de entrenamiento tomada a ciegas y disfrazada de constante."""
    with pytest.raises(ValueError, match="pesos explícitos"):
        MixtureSpec()
    m = MixtureSpec(weights={"json": 1.0, "prose": 3.0})
    assert m.normalized() == {"json": 0.25, "prose": 0.75}


def test_repo_files_se_clasifican_por_extension(tmp_path):
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "runtime").mkdir(parents=True)
    (repo / "docs" / "a.md").write_text("# titulo\n\ntexto\n")
    (repo / "runtime" / "b.py").write_text("x = 1\n")
    b = CorpusBuilder(tmp_path / "c", db_path=tmp_path / "nada.db", repo_root=repo)
    kinds = {d.content_kind for d in b.iter_repo()}
    assert kinds == {"prose", "code_py"}
