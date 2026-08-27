"""Extractor del corpus: la experiencia real del organismo + la prosa/código del repo.

═══════════════════════════════════════════════════════════════════════════════
SPLIT POR run_id — no es un detalle, es la diferencia entre medir y mentir
═══════════════════════════════════════════════════════════════════════════════

Un episodio del organismo produce, para el MISMO `run_id`: eventos, trazas de
razonamiento, registros de memoria y snapshots — todos con las mismas entidades,
los mismos UUIDs y las mismas claves.  Si se parte al azar, un `snapshot_json` de
train y un `payload` de val comparten literalmente el UUID del episodio.  El
modelo memoriza el episodio y la evaluación reporta una generalización que no
existe.

Por eso el split es POR `run_id`, HASHEADO, y COMPARTIDO entre las cuatro tablas:
un `run_id` cae entero de un lado o entero del otro.

  - `events` NO TIENE columna `run_id`.  El esquema es (id, event_type, payload,
    timestamp).  El `run_id` está DENTRO del payload JSON, como clave de primer
    nivel.  Verificado: 33.246/33.246 filas lo tienen.  Se extrae de ahí.
    (Esto contradice la lectura literal de "split por run_id" como si fuera una
    columna: para `events` hay que parsear.)

  - La PROSA Y EL CÓDIGO DEL REPO NO TIENEN `run_id` y no pueden tenerlo.  Se
    parten por hash del path relativo.  LÍMITE HONESTO: dos archivos casi
    idénticos (p. ej. dos tests hermanos) pueden caer a lados distintos.  El
    split por directorio sería más estricto pero dejaría familias enteras fuera
    de train.  Se elige el hash de path y se declara el límite; no se disimula.

═══════════════════════════════════════════════════════════════════════════════
LA MEZCLA ESTRUCTURA:PROSA ES UN PARÁMETRO, NO UNA CONSTANTE
═══════════════════════════════════════════════════════════════════════════════

Medido: 461 MB de estructura (JSON) contra ~5 MB de prosa+código.  Un desbalance
de ~87:1.  Entrenar con esa proporción es enseñarle al modelo a segmentar JSON y
a olvidarse de leer.

Este módulo NO elige la mezcla.  Emite los shards SEPARADOS por fuente y expone
`MixtureSpec`, que EXIGE pesos explícitos.  La proporción correcta se decide
MIDIENDO con el evaluador (F1 en prosa vs F1 en JSON a distintas mezclas), no
adivinando acá.  `MixtureSpec()` sin pesos levanta excepción a propósito.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Literal

__all__ = [
    "DB_SOURCES",
    "REPO_SOURCES",
    "CorpusBuilder",
    "Document",
    "MixtureSpec",
    "ShardManifest",
    "split_of",
]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_db() -> Path:
    """La DB vive en el checkout principal, no en el worktree. Overrideable por env."""
    env = os.environ.get("RNFE_EVENT_LOG_DB")
    if env:
        return Path(env)
    local = REPO_ROOT / "aeon_event_log.db"
    if local.exists():
        return local
    return Path.home() / "Desarrollo" / "RNE_v16" / "aeon_event_log.db"


DEFAULT_DB = _default_db()

Split = Literal["train", "val"]

# (tabla, columna de contenido, columna de run_id | None -> está dentro del JSON)
DB_SOURCES: tuple[tuple[str, str, str | None], ...] = (
    ("events", "payload", None),  # run_id vive DENTRO del payload
    ("reasoning_traces", "detail", "run_id"),
    ("memory_records", "structure_json", "run_id"),
    ("organism_snapshots", "snapshot_json", "run_id"),
)

REPO_SOURCES: tuple[str, ...] = (
    "canon",
    "docs",
    "runtime",
    "tests",
    "scripts",
    "ai/obsidian-mind",
)

VAL_FRACTION = 0.15
SPLIT_SALT = "rnfe-hnet-chunker-v1"


def split_of(key: str, *, val_fraction: float = VAL_FRACTION, salt: str = SPLIT_SALT) -> Split:
    """Split determinista y estable: hash(salt|key) -> [0,1).

    Determinista a propósito: reconstruir el corpus mañana tiene que dar el mismo
    split, o las comparaciones antes/después no comparan nada.
    """
    h = hashlib.sha256(f"{salt}|{key}".encode()).digest()
    x = int.from_bytes(h[:8], "big") / 2**64
    return "val" if x < val_fraction else "train"


@dataclass(frozen=True)
class Document:
    doc_id: str
    source: str  # "db:events.payload" | "repo:runtime/**.py"
    content_kind: str  # "json" | "prose" | "code_py"  -> qué verdad de campo aplica
    run_id: str | None
    split: Split
    sha256: str
    n_bytes: int
    text: str

    def as_record(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class ShardManifest:
    """Procedencia por shard. Nada de corpus anónimo."""

    path: str
    source: str
    table: str | None
    column: str | None
    content_kind: str
    split: Split
    n_records: int
    n_bytes: int
    n_run_ids: int
    file_sha256: str


@dataclass
class MixtureSpec:
    """Pesos de muestreo por fuente. SIN DEFAULT — a propósito.

    El desbalance medido es ~87:1 estructura:prosa.  Cualquier default que
    pusiéramos acá sería una decisión de entrenamiento tomada a ciegas y
    disfrazada de constante.  Los pesos se determinan con el evaluador.
    """

    weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError(
                "MixtureSpec exige pesos explícitos por fuente. La proporción "
                "estructura:prosa es un PARÁMETRO a determinar MIDIENDO con el "
                "evaluador (lab/hnet_chunker/evaluate.py), no una constante."
            )
        bad = [k for k, v in self.weights.items() if v < 0]
        if bad:
            raise ValueError(f"pesos negativos: {bad}")
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("la suma de los pesos debe ser > 0")

    def normalized(self) -> dict[str, float]:
        total = sum(self.weights.values())
        return {k: v / total for k, v in self.weights.items()}


class CorpusBuilder:
    """Extrae, deduplica, parte por run_id y escribe shards JSONL con manifiesto."""

    def __init__(
        self,
        out_dir: Path,
        *,
        db_path: Path = DEFAULT_DB,
        repo_root: Path = REPO_ROOT,
        val_fraction: float = VAL_FRACTION,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.db_path = Path(db_path)
        self.repo_root = Path(repo_root)
        self.val_fraction = val_fraction
        self.manifests: list[ShardManifest] = []

    # ── fuentes ────────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        """READ-ONLY. La DB del organismo no se toca ni por accidente."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"no existe la DB: {self.db_path}")
        return sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)

    def iter_db(self, table: str, column: str, run_col: str | None) -> Iterator[Document]:
        con = self._connect()
        try:
            cur = con.cursor()
            cols = f"{column}" if run_col is None else f"{column}, {run_col}"
            seen: set[str] = set()
            for row in cur.execute(f"SELECT {cols} FROM {table}"):  # noqa: S608 - tablas fijas
                text = row[0]
                if not text:
                    continue
                if run_col is None:
                    # events: el run_id está DENTRO del payload JSON.
                    try:
                        run_id = json.loads(text).get("run_id")
                    except (json.JSONDecodeError, AttributeError):
                        run_id = None
                else:
                    run_id = row[1]

                raw = text.encode("utf-8")
                digest = hashlib.sha256(raw).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)

                # Sin run_id no hay split defendible: el documento se DESCARTA,
                # no se manda a train "por las dudas".
                if not run_id:
                    continue

                yield Document(
                    doc_id=f"{table}:{digest[:16]}",
                    source=f"db:{table}.{column}",
                    content_kind="json",
                    run_id=run_id,
                    split=split_of(run_id, val_fraction=self.val_fraction),
                    sha256=digest,
                    n_bytes=len(raw),
                    text=text,
                )
        finally:
            con.close()

    def iter_repo(self) -> Iterator[Document]:
        seen: set[str] = set()
        for rel in REPO_SOURCES:
            base = self.repo_root / rel
            if not base.exists():
                continue
            for path in sorted(base.rglob("*")):
                if not path.is_file() or path.suffix not in (".py", ".md"):
                    continue
                if "__pycache__" in path.parts or ".pytest_cache" in path.parts:
                    continue
                try:
                    text = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                if not text.strip():
                    continue
                raw = text.encode("utf-8")
                digest = hashlib.sha256(raw).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                relpath = str(path.relative_to(self.repo_root))
                yield Document(
                    doc_id=f"repo:{digest[:16]}",
                    source=f"repo:{rel}",
                    content_kind="code_py" if path.suffix == ".py" else "prose",
                    run_id=None,
                    split=split_of(relpath, val_fraction=self.val_fraction),
                    sha256=digest,
                    n_bytes=len(raw),
                    text=text,
                )

    # ── escritura ──────────────────────────────────────────────────────────────

    def _write_shard(
        self,
        docs: list[Document],
        *,
        name: str,
        source: str,
        table: str | None,
        column: str | None,
        content_kind: str,
        split: Split,
    ) -> ShardManifest | None:
        if not docs:
            return None
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{name}.{split}.jsonl"
        h = hashlib.sha256()
        n_bytes = 0
        with path.open("w", encoding="utf-8") as fh:
            for d in docs:
                line = json.dumps(d.as_record(), ensure_ascii=False) + "\n"
                fh.write(line)
                h.update(line.encode("utf-8"))
                n_bytes += d.n_bytes
        man = ShardManifest(
            path=str(path.relative_to(self.out_dir)),
            source=source,
            table=table,
            column=column,
            content_kind=content_kind,
            split=split,
            n_records=len(docs),
            n_bytes=n_bytes,
            n_run_ids=len({d.run_id for d in docs if d.run_id}),
            file_sha256=h.hexdigest(),
        )
        self.manifests.append(man)
        return man

    def build(self, *, limit_per_source: int | None = None) -> dict:
        self.manifests = []
        for table, column, run_col in DB_SOURCES:
            buckets: dict[Split, list[Document]] = {"train": [], "val": []}
            for i, doc in enumerate(self.iter_db(table, column, run_col)):
                if limit_per_source is not None and i >= limit_per_source:
                    break
                buckets[doc.split].append(doc)
            for split, docs in buckets.items():
                self._write_shard(
                    docs,
                    name=f"db_{table}",
                    source=f"db:{table}.{column}",
                    table=table,
                    column=column,
                    content_kind="json",
                    split=split,
                )

        repo_buckets: dict[tuple[str, Split], list[Document]] = {}
        for i, doc in enumerate(self.iter_repo()):
            if limit_per_source is not None and i >= limit_per_source:
                break
            repo_buckets.setdefault((doc.content_kind, doc.split), []).append(doc)
        for (kind, split), docs in sorted(repo_buckets.items()):
            self._write_shard(
                docs,
                name=f"repo_{kind}",
                source="repo",
                table=None,
                column=None,
                content_kind=kind,
                split=split,
            )

        manifest = {
            "corpus_version": "v1",
            "db_path": str(self.db_path),
            "db_sha256_note": "la DB se abre read-only; no se hashea entera (635 MB)",
            "repo_root": str(self.repo_root),
            "repo_commit": _git_head(self.repo_root),
            "val_fraction": self.val_fraction,
            "split_salt": SPLIT_SALT,
            "split_policy": {
                "db": "hash(run_id) — compartido entre las 4 tablas; events lo extrae del payload",
                "repo": "hash(path relativo) — límite: archivos casi idénticos pueden separarse",
            },
            "dedup": "sha256 del contenido utf-8, dentro de cada fuente",
            "mixture": "NO DEFINIDA. Es un parámetro: ver MixtureSpec / evaluate.py",
            "shards": [asdict(m) for m in self.manifests],
            "totals": {
                "n_records": sum(m.n_records for m in self.manifests),
                "n_bytes": sum(m.n_bytes for m in self.manifests),
                "by_kind": _by(self.manifests, "content_kind"),
                "by_split": _by(self.manifests, "split"),
            },
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return manifest


def _by(mans: list[ShardManifest], attr: str) -> dict:
    out: dict[str, dict[str, int]] = {}
    for m in mans:
        k = getattr(m, attr)
        slot = out.setdefault(k, {"n_records": 0, "n_bytes": 0})
        slot["n_records"] += m.n_records
        slot["n_bytes"] += m.n_bytes
    return out


def _git_head(root: Path) -> str | None:
    head = root / ".git"
    try:
        if head.is_file():  # worktree: .git es un archivo con gitdir:
            gitdir = Path(head.read_text().split("gitdir:")[1].strip())
            head = gitdir
        ref = (head / "HEAD").read_text().strip()
        if ref.startswith("ref: "):
            return ref[5:]
        return ref
    except (OSError, IndexError):
        return None


def load_shard(path: Path) -> Iterator[Document]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            yield Document(**rec)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Extrae el corpus del organismo para el chunker")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "hnet_corpus"))
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--limit-per-source", type=int, default=None)
    args = ap.parse_args()

    builder = CorpusBuilder(Path(args.out), db_path=Path(args.db))
    man = builder.build(limit_per_source=args.limit_per_source)
    print(json.dumps(man["totals"], indent=2))
    for s in man["shards"]:
        print(f"  {s['path']:34s} {s['n_records']:7d} docs  {s['n_bytes']/1e6:8.2f} MB  runs={s['n_run_ids']}")
    print(f"\nmanifiesto: {Path(args.out) / 'manifest.json'}")


if __name__ == "__main__":
    main()
