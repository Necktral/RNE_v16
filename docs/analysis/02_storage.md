# 02 — `runtime/storage/` (capa de persistencia)

El módulo **más central** del runtime (25 importadores). Persistencia *backend-agnóstica* con
3 backends y un *facade* estable.

```
runtime/storage/
├── __init__.py        # singleton get_storage()/reset_storage() + re-exports
├── config.py          # StorageConfig (from_env, validate)
├── factory.py         # StorageFactory.create_backend/create_facade
├── facade.py          # StorageFacade (API estable, 737 LOC)
├── interfaces.py      # 10 Protocols → StorageBackend
├── records.py         # 16 dataclasses de record
├── backends/
│   ├── sqlite_store.py    # 1437 LOC — SQLite + ledger legacy
│   ├── postgres_store.py  # 1386 LOC — PostgreSQL (psycopg)
│   ├── hybrid_store.py    # 412 LOC — dual-write PG+SQLite
│   └── postgres/schema.sql # DDL Postgres (283 líneas)
└── migrations/sqlite_to_postgres.py  # migración del ledger
```

## Arquitectura

- **Singleton** (`__init__.py`): `get_storage()` con `threading.Lock`, `StorageConfig.from_env()`,
  `reset_storage()`. Patrón correcto y thread-safe a nivel de construcción.
- **Config** (`config.py`): `RNFE_STORAGE_MODE` (sqlite|postgres|hybrid, default sqlite),
  `AEON_EVENT_DB`, `RNFE_POSTGRES_DSN`, `RNFE_ARTIFACT_ROOT` (default `cwd/rnfe_artifacts`),
  `RNFE_STORAGE_PREFER_POSTGRES_READS` (def True), `RNFE_STORAGE_STRICT_DUAL_WRITE` (def **False**).
  `validate()` exige DSN en modos postgres/hybrid.
- **Factory** (`factory.py`): imports perezosos de backends (no carga `psycopg` en modo sqlite). En
  `hybrid`: primary=Postgres, fallback=SQLite.
- **Facade**: 1 método por operación, copias defensivas (`dict(... or {})`), normaliza tipos.
- **Interfaces**: 10 `Protocol` `@runtime_checkable` componen `StorageBackend` (l.274-289).
- **Backends**: conexión-por-operación (sin pool). SQLite con `threading.Lock` en escrituras.

---

## Hallazgos

### [BUG] Upsert parcial en Postgres `write_transfer_assessment` (postgres_store.py:828-831)
En `ON CONFLICT (assessment_id) DO UPDATE` solo actualiza `transfer_verdict` y `metadata_jsonb`.
**No** actualiza `memory_purity_score`, `transition_stability_score`, `compatibility_class`,
`source/target_scenario`. SQLite usa `INSERT OR REPLACE` (sqlite_store.py:946) que reemplaza **todo**.
➡️ Re-escribir un assessment con el mismo id deja **Postgres con scores viejos y SQLite con scores
nuevos**: divergencia silenciosa entre backends. Es el único upsert de Postgres que no copia todas
las columnas (los demás sí). Corregir el `DO UPDATE` para incluir todas las columnas.

### [BUG] `SQLiteStorageBackend.list_events` filtra `run_id` después del LIMIT (sqlite_store.py:338-360)
`list_events` pide al ledger legacy `get_events(limit=limit, ...)` que devuelve **los `limit`
eventos más recientes globales** (event_log_sqlite.py:79-97, `ORDER BY id DESC LIMIT ?`), y solo
*después* filtra por `run_id` en Python (l.357) y aplica `events[-limit:]`. Si los eventos de ese
`run_id` no están entre los últimos `limit` globales, **se pierden**. Postgres filtra `run_id` en
SQL (postgres_store.py:123-125), así que devuelve resultados correctos. → divergencia + bug de
correctitud en SQLite. (El filtro `event_types=[]` NO es bug: `[]` es falsy y se ignora → devuelve todos.)

### [RIESGO] Dual-write no estricto traga fallos parciales sin loguear (hybrid_store.py:45-79)
`_dual_write` escribe en primary y fallback capturando excepciones. Con `strict_dual_write=False`
(**el default**), si una escritura falla y la otra no, **devuelve el resultado bueno y sigue** sin
ningún log. Los dos stores divergen en silencio. No hay `logging` en todo el archivo. En modo hybrid
de producción esto degrada la garantía de consistencia sin observabilidad. Recomendado: loguear el
error parcial siempre (incluso en no-estricto) y exponer métrica de desincronización.

### [RIESGO] `_read_with_fallback` confunde "vacío" con "fallo" (hybrid_store.py:81-90)
Si el store preferido devuelve una lista **vacía válida** (no hay filas), `if rows:` es falsy y
cae al segundo store. Consecuencias: (a) un resultado legítimamente vacío del primary se
sobrescribe con datos (posiblemente desfasados) del fallback; (b) toda consulta sin resultados
golpea ambos backends (coste x2). La excepción tampoco se loguea (l.88).

### [DISEÑO] `list_events`: fuentes estructuralmente distintas por backend
SQLite lee de la tabla legacy `events` vía `EventLogSQLite` (sin `event_id`, sin `payload_hash`,
sin dedup); Postgres lee de `ledger_events` (uuid `event_id`, `payload_hash`, dedup por
`UNIQUE(legacy_db_path, legacy_event_id)`). El `append_event` también es asimétrico (SQLite no
puebla `event_id`/`payload_hash` en el record devuelto; Postgres sí). En hybrid, qué backend
responde cambia la forma/at conjunto de eventos.

### [DISEÑO] Tabla `runs` huérfana (schema.sql:1-7)
Postgres define `runs(run_id, status, started_at, ended_at, context_json)` pero **ningún método**
de los backends la lee o escribe. SQLite no la tiene. Tabla muerta a nivel de la capa de storage.

### [DISEÑO] Formato de timestamp inconsistente
`records.utc_now_iso()` usa `datetime.now(timezone.utc).isoformat()` (tz-aware, `+00:00`), pero el
ledger legacy `EventLogSQLite.log_event` usa `datetime.utcnow().isoformat()` (naive, sin offset, y
`utcnow()` está deprecado en 3.12+). Mezcla de formatos en la columna timestamp del ledger.

### [DISEÑO] Orden de listado inconsistente entre stores
`list_telemetry_snapshots`, `list_reasoning_traces` y `list_reality_assessments` ordenan **ASC**
(más antiguo primero); el resto ordena **DESC**. El llamador debe conocer el orden por tabla.

### [DISEÑO/RIESGO] SQLite sin WAL ni busy_timeout (sqlite_store.py)
Conexión por operación + `threading.Lock` solo en escrituras (las lecturas `list_*` no toman el
lock). No hay `PRAGMA journal_mode=WAL` ni `busy_timeout`. Dentro de un proceso queda casi
serializado, pero **múltiples procesos** (p. ej. benchmarks en paralelo escribiendo el mismo .db)
pueden disparar `database is locked` sin reintento. La DB por defecto es relativa al cwd
(`aeon_event_log.db`); en la raíz del repo hay una de ~890 MB (posible artefacto pesado versionado).

### [DISEÑO] `created_at` actualizado distinto en Postgres ON CONFLICT
`telemetry_snapshots`/`reasoning_traces`/`sessions` ponen `created_at`/`updated_at = now()` al
hacer update; el resto conserva `excluded.created_at`. Semántica de "created_at" no uniforme.

### [DISEÑO] `register_artifact` lee el fichero completo aunque ya tenga sha256+size (facade.py:140)
`content = path.read_bytes()` es incondicional. Cuando lo llama `materialize_artifact`
(facade.py:189) ya se pasó `sha256` y `size_bytes`, así que esa lectura completa es I/O
desperdiciado (doble lectura del artefacto). Además no hay hashing en streaming: artefactos grandes
se cargan enteros en memoria.

### [DISEÑO] Contratos formales incompletos vs. records persistidos
`records.py` define 16 records, pero `contracts/*.schema.json` solo cubre ~7 de ellos
(`event`, `telemetry_snapshot`, `reasoning_trace`, `artifact_index`, `session_bridge`,
`reality_assessment`, `episode certificate`, `memory_record`). **No hay JSON schema** para
`reality_bench_run`, `promotion_decision`, `transfer_assessment`, `organism_snapshot`,
`trajectory_window`, `trajectory_flow_report`, `renormalization_event`,
`constitutional_risk_state`, `failure_atlas_event`. Los contratos formales van por detrás del modelo.

### [DISEÑO] `TransferAssessmentStore` no re-exportado (`__init__.py`)
Está definido en `interfaces.py:167` y compuesto en `StorageBackend` (l.284), pero `__init__.py`
re-exporta solo 9 de los 10 Protocols (omite `TransferAssessmentStore`). Inconsistencia menor de API.

### [DISEÑO] Migración solo cubre el ledger
`sqlite_to_postgres.py` migra **solo** la tabla `events` (idempotente vía `event_id=legacy::path::id`
+ `ON CONFLICT DO NOTHING`). No migra telemetry/certificates/memory/trajectory/etc. Un movimiento
completo SQLite→Postgres requeriría más. (El nombre acota a "ledger", pero conviene anotarlo.)

---

## Aspectos positivos
- Separación limpia facade / interfaces / backends; factory con imports perezosos.
- Copias defensivas y casteo de tipos consistente en el facade.
- Esquema con índices `(run_id, …, created_at)` razonables en ambos backends.
- Paridad de tablas auxiliares completa entre SQLite y Postgres (16/16); migración idempotente del ledger.
- `_safe_json_load` robusto ante JSON corrupto.

## Prioridad de corrección sugerida
1. **[BUG]** upsert parcial Postgres `transfer_assessments` (divergencia de datos).
2. **[BUG]** `list_events` run_id post-LIMIT en SQLite (resultados incompletos).
3. **[RIESGO]** logging/observabilidad de divergencia en dual-write y fallback de lecturas.
