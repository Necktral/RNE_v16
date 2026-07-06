---
title: ORIENTACIONES_FASE0_PARA_CODEX
status: experimental
version: 1.0.0
date: 2026-07-05
owner: Wis
depends_on:
  - CANON_RNFE_v3_2_rc1.md
  - RUNTIME_SSOT_v1.md
  - 2026-07-05_planos_integracion_rne16_opus.md
  - docs/analysis/19_operational_conjunction.md
supersedes: []
notes:
  - Capa D (laboratorio/dirección). NO-NORMATIVO. Concreta la Fase 0 del plano de Fable
    (2026-07-05_planos_integracion_rne16_opus.md §2) en instrucciones ejecutables para Codex.
  - Autor de esta capa de orientación - Opus. Implementador - Codex. No contiene código
    de implementación; contiene anclas archivo:línea, contratos objetivo y criterios de
    aceptación falsables.
  - Todas las anclas archivo:línea fueron verificadas contra el árbol de trabajo el 2026-07-05.
---

# Orientaciones de Fase 0 (Cimientos) para Codex

## 0. Para quién es esto y cómo leerlo

Cadena de trabajo del proyecto: **Fable diseña** el plano maestro
([2026-07-05_planos_integracion_rne16_opus.md](2026-07-05_planos_integracion_rne16_opus.md))
→ **Opus concreta** estas orientaciones → **Codex implementa** el código y los tests.

Codex ya viene construyendo el working set actual (la capa de atestación/gobernanza, ~555
líneas sin commitear). Este documento **no reinicia** ese trabajo: lo reconoce (§2), lo
enciende (WS0) y lo completa con los cimientos que faltan (WS1–WS7).

Cada workstream (WS) trae: **objetivo**, **anclas** verificadas, **qué construir**, **cómo
conecta con el WIP**, **disciplina de flag**, **criterios de aceptación falsables** y
**tests**. Implementá en el orden de §11. No hace falta pedir permiso entre WS; sí respetar
el mandato de §1.

---

## 1. Mandato de ejecución para Codex (vinculante)

1. **Precedencia del canon.** `canon/normative/CANON_RNFE_v3_2_rc1.md` es SSOT. No violar
   axiomas A1–A9 ni los invariantes del núcleo. En especial A8 (observabilidad primaria) y
   A9 (economía de razón).
2. **Sombra-primero, nominal byte-idéntico.** Toda capacidad que cambie conducta nace tras
   un flag `RNFE_*` **off por defecto**. Con el flag off, una corrida seedeada debe producir
   **las mismas decisiones** que HEAD. Enriquecer payloads de eventos con claves nuevas
   **sí** está permitido (eventos aditivos); cambiar qué decide el organismo **no**, salvo
   con flag on.
3. **Cero dependencias nuevas en el camino vivo.** `runtime/` sigue siendo Python puro +
   stdlib (más `z3` y `psycopg` ya admitidos). `psutil` solo como **opcional con fallback
   stdlib** (WS0). Nada del camino vivo puede requerir GPU.
4. **Contratos antes que lógica.** Toda pieza nueva: dataclass frozen JSON-friendly (patrón
   `runtime/conjunction/contracts.py` y `runtime/life/contracts.py`), y evento append-only
   vía `StorageFacade`. Si cruza frontera persistente, schema en `contracts/`.
5. **Intocables sin ADR + campaña propia:** el orden `CORE_SEQUENCE`
   (abd→ana→cau→ctf→ded→prob, PROB último), el **contrato de `ScenarioEpisodeRunner`**
   (invariante `ADR_MSRC`: no reintroducir `max_steps` ni cambiar su firma pública), las 3
   métricas de continuidad (`ADR_CONTINUITY`), y la frontera `runtime ↛ exocortex ↛ archive`
   (verificada por `tests/contracts/test_boundary_rules.py`).
6. **Test de paridad obligatorio** por WS que toque el camino vivo: una corrida smoke con el
   flag off comparada contra baseline (mismas decisiones). Ver §10.
7. **Hardware objetivo:** PC común, CPU multinúcleo, 8–16 GB RAM, GPU ≤8 GB **opcional**.

---

## 2. Reconocimiento del WIP (alimentar, no reescribir)

El working set actual ya construyó una **capa consumidora** completa y verde (1050 tests
colectan; `tests/world/test_causal_attestation.py` + `tests/conjunction/` = 16 verde). NO la
reescribas; los WS de abajo la **alimentan**.

| Pieza del WIP | Esquema | Ubicación | Estado |
|---|---|---|---|
| Atestación causal factual/contrafactual | `causal_attestation.v1` | `runtime/world/causal_attestation.py`; cableada en `scenario_runner.py:382,469,508`, `min_cognitive_episode.py:124`, consumida en `promotion_gate.py:129`, `context_features.py:114`, `governance.py:169` | vivo, verde |
| Atestación de recuperación memory-RAG | `memory_rag_attestation.v1` | `runtime/memory/mfm_lite/retrieval.py::summarize_retrieval_hits`; en `reasoning/context.py:58,66` | vivo, verde |
| Envolvente de gobernanza META | `reasoning_governance.v1` | `runtime/reasoning/scheduler_meta/governance.py::build_governance_envelope`; vivo en `meta_scheduler.py:275,286,306` | vivo, verde |
| Plan de degradación táctico | `degradation_plan.v1` | `runtime/reasoning/scheduler_meta/degradation.py::build_degradation_plan` | vivo vía governance |
| Política de autonomía | `AutonomyPolicy`/`AutonomyMode` | `runtime/conjunction/contracts.py`; `.resolve()` con blockers; validador `_autonomy_policy` en `validators.py`; compensación `degrade_autonomy_scope` | vivo, verde |
| Features de presión de hardware | (features) | `context_features.py:155-205`, `budgeting.py`, `policy.py` | vivo pero **sin datos** |

**El problema central que Fase 0 resuelve:** esa capa **corre sobre ceros**.
`context_features.py:155-205` lee `cpu_pressure`/`memory_pressure`/`vram_pressure`/
`thermal_pressure`/`gpu_load` desde el `context` (o `telemetry`/`observation`) — y **ninguna
ruta viva escribe esas claves**, así que default 0.0. `vitals.py:78-83` deriva
`resource_pressure` de `risk_plus.b_safe.pressure` (clave que `compute_b_safe` no produce; ver
`risk_engine.py:98-128`) o de `episode_result.resource_pressure` (nunca escrito) → 0.0.
Consecuencia medible: nunca disparan `sleep` (`supervisor.py:75`, `>=0.90`),
`mode_for_vitals` conservative (`vitals.py:36`, `>=0.80`), downgrade de conjunción
(`service.py:199`, `>=0.90`) ni `resource_pressure_limit` de la AutonomyPolicy (`>=0.85`).
**WS0 enciende todo eso.**

---

## 3. WS0 — Keystone: productor de recursos reales

**Objetivo.** Que el organismo sienta su cuerpo: producir presión de recursos real y
alimentar con ella la capa consumidora ya construida, sin romper byte-idéntico nominal.

**Anclas.**
- Interfaz a imitar: `NvidiaVRAMSampler.sample()` y `NullVRAMSampler.sample()` en
  `runtime/control/msrc/vram_sampler.py:24-68,145-157` — devuelven un `dict` con
  `available`, `source`, `vram_headroom`, `vram_pressure`, `vram_fragmentation_risk`,
  `vram_opportunity_score`, `sample_ts`.
- Consumidor 1 (vitals): `runtime/life/vitals.py:78-83` (derivación de `resource_pressure`).
- Consumidor 2 (features, ya listo): `runtime/reasoning/scheduler_meta/context_features.py:155-205`
  (lee `cpu_pressure`/`memory_pressure`/`vram_pressure`/`gpu_load`/`thermal_pressure`).
- Punto de inyección al contexto: `build_reasoning_context(..., extra_signals=...)` en
  `runtime/reasoning/context.py:38-55,88-91` (los items de `extra_signals` se copian al
  contexto de nivel superior si la clave no existe).
- Construcción MSRC a intervenir: `runtime/life/kernel.py:546-555`
  (`vram_sampler=NullVRAMSampler()`).

**Qué construir.**
1. **Nuevo** `runtime/control/msrc/host_sampler.py` con `HostResourceSampler`:
   - `.sample()` devuelve un dict **superset** compatible con el de los VRAM samplers:
     `available`, `source`, `sample_ts`, `cpu_pressure`, `memory_pressure`, `swap_pressure`,
     `thermal_pressure`, `vram_pressure`, `vram_headroom`, `gpu_load`, `gpu_available`.
   - Fuentes: `psutil` si `import psutil` funciona; **fallback stdlib puro** a
     `os.getloadavg()`/`/proc/loadavg` (cpu), `/proc/meminfo` (memoria/swap),
     `/sys/class/thermal/thermal_zone*/temp` (termia, opcional). Sin GPU →
     `vram_pressure=0.0`, `gpu_available=False` (reusar `NvidiaVRAMSampler` si el
     `ResourceProfile` futuro declara GPU; por ahora GPU ausente es lo normal).
   - **Caché TTL** con el patrón de `deque`/`sample_ts` de `NvidiaVRAMSampler` (no re-leer
     `/proc` más de una vez por ~0.5 s).
   - `cpu_pressure`/`memory_pressure` normalizados a [0,1]; `_clamp` local.
2. **Contrato** `HostResourceSnapshot` (frozen JSON-friendly) — puede vivir junto al sampler
   o en `runtime/life/contracts.py`. Debe tener `.to_dict()`.
3. **Gate** `RNFE_HOST_SENSING` (off por defecto). Off → el sampler devuelve un snapshot de
   ceros con `available=False` (equivalente a `NullVRAMSampler` para el eje de host), y todo
   queda byte-idéntico. On → sensado real.

**Cómo conecta con el WIP (cableado en 3 consumidores existentes).**
- `runtime/life/kernel.py`: instanciar un `HostResourceSampler` una vez por kernel. Cuando el
  flag está on, pasar `MSRCController(vram_sampler=...)` un sampler compuesto (host + nvidia si
  hay) en `kernel.py:551-555` en vez de `NullVRAMSampler`.
- `runtime/life/vitals.py`: agregar parámetro opcional `resource_snapshot: dict | None` a
  `from_state(...)`. Cuando llega snapshot (flag on), `resource_pressure` = máx de
  `cpu_pressure`/`memory_pressure`/`vram_pressure`/`thermal_pressure` del snapshot, en vez de
  la derivación de `risk_plus`/`episode_result` (líneas 78-83). Flag off / sin snapshot →
  ruta actual intacta.
- Contexto de razonamiento: el `LifeKernel` pasa el snapshot al `ScenarioEpisodeRunner`
  (por atributo del runner o parámetro nuevo de `run_episode`; ver WS3 que ya toca la firma),
  y el runner lo inyecta como `extra_signals={"cpu_pressure":..., "memory_pressure":...,
  "vram_pressure":..., "thermal_pressure":..., "gpu_load":..., "gpu_available":...}` en
  `build_reasoning_context`. `extract_context_features` ya los levanta sin cambios.

**Criterios de aceptación (falsables).**
- Con `RNFE_HOST_SENSING` sin setear: `scripts/life_kernel.py --run-id smoke --max-steps 8
  --no-restore` produce decisiones idénticas a HEAD; suite completa verde en CPU.
- Con `RNFE_HOST_SENSING=1`: `life.step.completed.vital_signs.resource_pressure` > 0 y varía
  con la carga real del host; `context.cpu_pressure`/`memory_pressure` llegan no-cero a
  `extract_context_features` (verificable por un evento/traza que exponga las features).
- Con un sampler fake que inyecta presión ≥0.90: el `AutonomySupervisor` emite `sleep`
  (`supervisor.py:75`) y la conjunción degrada (constraint `resource_pressure` en
  `service.py:199`) — hoy imposible de testear.
- `HostResourceSampler` corre sin `psutil` instalado (test con `psutil` ausente vía
  monkeypatch del import).

**Tests.** `tests/control/test_host_sampler.py` (determinismo con fuentes mockeadas, fallback
sin psutil, clamp [0,1], TTL); `tests/life/test_vitals_resource_pressure.py` (off = actual;
on = derivado del snapshot; presión inyectada → mode conservative).

---

## 4. WS1 — Endurecimiento de SQLite (WAL + busy_timeout)

**Objetivo.** Evitar `database is locked` bajo la amplificación de escritura por episodio
(SMG + EventBus + court + memoria) sin cambiar lógica.

**Anclas.** Conexiones desnudas `sqlite3.connect(path)` sin pragmas en:
`runtime/storage/backends/sqlite_store.py:55-56` (`_connect`) y
`runtime/core/event_log_sqlite.py:29,72,101`. Cero pragmas en el repo (verificado por grep).

**Qué construir.** En cada punto de conexión, aplicar tras `connect`, de forma idempotente:
`PRAGMA journal_mode=WAL`, `PRAGMA busy_timeout=5000`, `PRAGMA synchronous=NORMAL`.
Centralizar en un helper (`_connect` de `sqlite_store` y el equivalente de `event_log_sqlite`)
para no dispersar. No cambia ninguna consulta ni resultado.

**Disciplina.** Sin flag (mejora de fiabilidad pura). Nota operativa: WAL crea archivos
`-wal`/`-shm` junto a la DB; documentarlo en `docs/analysis/02_storage.md`.

**Criterios.** Tras abrir cualquier store SQLite, `PRAGMA journal_mode` devuelve `wal` y
`PRAGMA busy_timeout` ≥ 5000. Suite de storage verde. Un test de escritura concurrente
(2 hilos/procesos) que hoy podría lockear, ya no lockea.

**Tests.** `tests/storage/test_sqlite_pragmas.py`.

---

## 5. WS2 — Divergencia de dual-write observable + paridad menor

**Objetivo.** Que el híbrido deje de tragar fallos parciales en silencio (viola A8) y cerrar
la divergencia menor de `transfer_assessments`.

**Anclas.**
- `runtime/storage/backends/hybrid_store.py:45-79` (`_dual_write`: con `strict_dual_write=False`
  descarta la excepción del backend fallido) y `:81-90` (`_read_with_fallback`: `if rows:`
  trata "vacío válido" como fallo → cae al otro backend).
- `runtime/storage/backends/postgres_store.py:820-838` (transfer_assessments: el `DO UPDATE
  SET` omite `created_at`, único campo divergente vs. `INSERT OR REPLACE` de
  `sqlite_store.py:945-968`).
- Config default: `config.py:76` (`mode="sqlite"`), `config.py:90` (`strict_dual_write=False`).

**Qué construir.**
1. En `_dual_write`, cuando un backend falla y el otro tiene éxito (con
   `strict_dual_write=False`), **asentar un evento** `storage.write_divergence` en el backend
   vivo con `{method, backend_failed, key/id si disponible, payload_hash, error_str}`, en vez
   de descartar la excepción. No re-lanzar (sigue siendo best-effort), pero deja rastro
   auditable.
2. En `_read_with_fallback`, distinguir "vacío válido" de "fallo": solo caer al segundo
   backend ante **excepción**, no ante lista vacía exitosa.
3. Paridad menor: agregar `created_at = EXCLUDED.created_at` al `DO UPDATE SET` de
   transfer_assessments en `postgres_store.py:820-838` para alinear con SQLite.

**Disciplina.** Evento aditivo en la ruta de fallo (nominal sin fallo = byte-idéntico). Sin
flag.

**Criterios.** Test con backend Postgres forzado a fallar: la escritura llega a SQLite y se
asienta `storage.write_divergence` con el hash del payload; el otro camino (fallback vacío)
ya no consulta el segundo backend ante un resultado vacío legítimo; re-escribir un
`assessment_id` existente converge SQLite/PG salvo por diseño explícito.

**Tests.** `tests/storage/test_hybrid_divergence.py` (requires_postgres para el caso PG; el
caso vacío-vs-fallo es SQLite puro).

---

## 6. WS3 — Sobre de correlación causal (`CausalContext`)

**Objetivo.** Que la cadena decisión→episodio→traza→certificado se reconstruya por IDs, no por
adyacencia temporal. Es prerequisito de la auditoría reproducible (Fase 1) y de la atribución
causal.

**Anclas (todo confirmado ausente hoy).**
- `runtime/life/kernel.py:213-228`: `life.step.completed` **sin** `episode_id`.
- `runtime/world/scenario_runner.py:494-521` (payload) y `:524-529` (append): `episode.closed`
  **sin** `decision_id`/`step_index`.
- `runtime/world/scenario_runner.py:285`: `def run_episode(self, *, external_input: float = 0.04)`
  — **sin** `causal_context`.
- `runtime/reasoning/scheduler_meta/meta_scheduler.py:313-329`: cada traza recibe un `uuid4`
  **por paso**; no hay `trace_id` por episodio.
- No existe `CausalContext` (grep en `runtime/life/contracts.py`, `runtime/storage/records.py`).

**Qué construir.**
1. `CausalContext` frozen en `runtime/life/contracts.py`: `run_id`, `step_index`,
   `decision_id`, `episode_id: str | None`, `parent_event_id: str | None`. Con `.to_dict()`.
2. `LifeKernel.step` construye el `CausalContext` tras la decisión del supervisor (ya tiene
   `decision.decision_id`, `self.total_steps`, `self.run_id`) y lo pasa a
   `run_episode(causal_context=...)`. Aprovechar el mismo parámetro para pasar el
   `resource_snapshot` de WS0 (o agregar ambos).
3. `scenario_runner.run_episode` acepta `causal_context: CausalContext | None = None`
   (kw-only, default None → no rompe el contrato: llamadas existentes sin el arg siguen
   válidas). Incluye `decision_id`/`step_index`/`parent_event_id` en `episode_payload`
   (`:494-521`) y genera **un** `trace_id` por episodio, propagado al `MetaScheduler` para que
   todas las `ReasoningTraceRecord` del episodio lo compartan (hoy `facade.py:108` mintea uno
   por llamada; pasar `trace_id` explícito).
4. `LifeKernel` agrega `episode_id`, `certificate_id`, `trace_id` al payload de
   `life.step.completed` (`kernel.py:213-228`) desde `episode_result`.

**Disciplina.** Aditivo (nuevas claves de payload y un parámetro kw-only opcional). Sin flag.
Verificar que el contrato público de `run_episode` sigue satisfaciendo `ADR_MSRC` (no se
reintroduce `max_steps`, la firma solo gana un kw-only opcional).

**Criterios.** Para cada `life.step.completed` con acción act/explore existe exactamente un
`episode.closed` con el mismo `decision_id` y `step_index` (query al ledger, sin timestamps);
todas las trazas de un episodio comparten un único `trace_id` y ese `trace_id` aparece en el
`EpisodeCertificateRecord` del episodio.

**Tests.** `tests/life/test_causal_context_propagation.py` (cadena por IDs en una corrida de
N pasos); `tests/reasoning/test_trace_id_per_episode.py`.

---

## 7. WS4 — Paridad e índices del ledger + `find_events`

**Objetivo.** Elevar el ledger SQLite a fuente de verdad de primera clase (paridad con
`ledger_events` de Postgres) y eliminar el escaneo O(n) del cierre triádico.

**Anclas.**
- Tabla `events` SQLite con solo 4 columnas: `runtime/core/event_log_sqlite.py:32-38`
  (`id, event_type, payload, timestamp`). `payload_hash` nunca se computa en SQLite.
- Postgres schema-rico: `runtime/storage/backends/postgres/schema.sql:9-21` (`ledger_events`
  con `event_id`, `run_id`, `payload_hash` NOT NULL, `source`, índices `:149-153`).
- No existe `find_events` (grep). Única superficie: `get_events(limit, event_types, run_id)`
  (`event_log_sqlite.py:79`) y `list_events` en facade/backends.
- Escaneo O(n): `runtime/reality/evaluator.py:84-92` (`_has_episode_closed_event` pide 500
  eventos y filtra en Python por `episode_id`; falso negativo si el run supera 500 eventos).

**Qué construir.**
1. Migración idempotente (patrón `eve_type→event_type` ya existente en `event_log_sqlite.py`)
   que agrega a `events` columnas `event_id` (UNIQUE), `run_id`, `source`, `payload_hash`, y
   columnas o índices para `episode_id`/`decision_id` (extraídos del payload al escribir, o
   índices sobre `json_extract`). Abrir una DB legacy no debe fallar y debe ser idempotente.
2. `append_event` computa `payload_hash` (SHA-256, mismo patrón que `postgres_store._payload_hash`
   en `postgres_store.py:37-39`) y persiste las columnas; devuelve el `StoredEvent` completo
   con su `seq` (rowid) como orden total.
3. `find_events(event_type, *, episode_id=None, decision_id=None, run_id=None, limit)` indexado
   en `StorageFacade` + `interfaces.py`, con paridad SQLite/Postgres.
4. `reality/evaluator.py:_has_episode_closed_event` pasa a usar `find_events(event_type=
   "episode.closed", episode_id=..., run_id=...)` — sin escaneo lineal, sin falso negativo.

**Disciplina.** La migración es aditiva sobre el esquema; los eventos ganan columnas derivadas
del payload existente (sin cambio de conducta). Sin flag. Cuidar retro-compat: DBs viejas sin
las columnas se migran al abrir.

**Criterios.** Una `aeon_event_log.db` legacy se abre sin error y la migración es idempotente
(dos aperturas no alteran esquema ni datos); `append_event` devuelve `event_id` y
`payload_hash` idénticos en SQLite y Postgres para el mismo payload; `_has_episode_closed_event`
encuentra el `episode.closed` aunque haya >500 eventos posteriores en el run (test que siembra
600 eventos de ruido).

**Tests.** `tests/storage/test_ledger_parity.py`, `tests/storage/test_find_events.py`,
`tests/reality/test_episode_closed_indexed.py`.

---

## 8. WS5 — Reality gate dentro del bucle vivo (shadow)

**Objetivo.** Cerrar la brecha "el bucle vivo nunca invoca `RealityValidationService`": que el
organismo evalúe su propio contacto con la realidad (closure_rate/continuidad/colapsos) durante
la vida, no solo en benchmark.

**Anclas.**
- `LifeKernel` no importa ni llama `RealityValidationService` (grep en `runtime/life/` = 0).
- `runtime/reality/service.py:22` (`GATE_PROFILES`), `:57` (`evaluate_episode_result`); hoy
  invocado solo por CLI/hook/benchmark/tests.
- `runtime/world/scenario_runner.py:604`: `trajectory_window` deshabilitado con `if False`
  ("Will enable in certification update").

**Qué construir.**
1. Que el episodio vivo escriba también un `reality_assessment` por episodio: `PromotionGate`
   (que ya corre por episodio) o el runner llama `RealityValidationService.evaluate_episode_result`.
2. `LifeKernel` agrega un chequeo ventaneado cada `reality_gate_interval` ciclos (config
   nueva, flag `enable_reality_gate` default **shadow=True** pero sin cambiar decisiones) que
   agrega closure_rate/continuity_mean/collapse_count del run contra `GATE_PROFILES['ci']`
   reusando `_evaluate_gate`.
3. El resultado entra a `VitalSignsService` como señal `reality_gate_ok` (patrón de señal
   compuesta como `identity_continuity`), disponible para el supervisor **sin gobernar todavía**
   (shadow). Registrar evento `reality.gate.evaluated`.
4. Opcional acotado: habilitar `trajectory_window` (`scenario_runner.py:604`) si el reality
   assessment lo requiere; si no, dejarlo para Fase 1 y documentarlo.

**Disciplina.** Shadow por defecto: se computa y persiste, no cambia decisiones. Encenderlo
como gate real (que el supervisor degrade ante `reality_gate_ok=False`) es Ola 1, tras
evidencia.

**Criterios.** Una corrida de ≥12 ciclos persiste ≥12 `reality_assessment` (vía
`storage.list_reality_assessments`) y ≥1 `reality.gate.evaluated` con el resumen ventaneado;
un test que fuerza colapsos sintéticos produce `reality_gate_ok=False` en vitals sin alterar la
decisión (shadow). Nominal byte-idéntico en decisiones.

**Tests.** `tests/life/test_reality_gate_shadow.py`.

---

## 9. WS6 — Consolidar el WIP (contratos, CLI, contexto)

**Objetivo.** Cerrar los cabos sueltos del working set actual para que la capa de gobernanza
quede consistente y alcanzable.

**Anclas.**
- `scripts/life_kernel.py:18-37`: **no** expone `--autonomy-policy` pese a existir
  `LifeKernelConfig.autonomy_policy` (default `"bounded"`) — la capacidad governed_unbounded no
  es alcanzable desde el CLI.
- `autonomy_policy` no se puebla en el contexto de razonamiento (grep: solo llega por la ruta
  de conjunción/life-kernel), así que la rama `context.get("autonomy_policy")` de
  `governance.py:172` queda vacía en la ruta del scenario_runner puro.
- `governance.py:60-74`: `_degradation_level` definido pero **nunca llamado** (código muerto;
  el `level` real viene de `build_degradation_plan`).

**Qué construir.**
1. `scripts/life_kernel.py`: agregar `--autonomy-policy {bounded,governed_unbounded}` (default
   `bounded`) y pasarlo a `LifeKernelConfig`.
2. Poblar `autonomy_policy` en el contexto de razonamiento (via `extra_signals` en
   `build_reasoning_context`, o donde el runner arma el contexto) para que la envolvente de
   gobernanza lo vea en ambas rutas.
3. Elevar las atestaciones (`causal_attestation.v1`, `memory_rag_attestation.v1`) a **contratos
   frozen tipados** (hoy son dicts con `schema` string) — dataclass en `runtime/world/` y
   `runtime/memory/`, con `.to_dict()`, sin cambiar el shape serializado (test de igualdad de
   dict). Retro-formalizar sus schemas en `contracts/` si van a persistir en certificado.
4. Eliminar el código muerto `_degradation_level` (`governance.py:60-74`) o cablearlo si tenía
   intención; documentar la decisión.
5. Registrar los 4 schemas nuevos (`causal_attestation.v1`, `memory_rag_attestation.v1`,
   `reasoning_governance.v1`, `degradation_plan.v1`) en `contracts/` como documentación
   machine-readable + test de existencia/parseo (patrón `tests/contracts/test_contract_schemas.py`).

**Disciplina.** Cambios estructurales de contrato = ADR corto. Serialización byte-idéntica de
las atestaciones (test de igualdad dict antes/después de tipar).

**Criterios.** `scripts/life_kernel.py --autonomy-policy governed_unbounded` alcanza la rama de
autonomía gobernada; `governance.py` recibe `autonomy_policy` en ambas rutas; los 4 schemas
existen en `contracts/` y parsean; cero código muerto en `governance.py`.

**Tests.** `tests/reasoning/test_governance_autonomy_context.py`,
`tests/contracts/test_attestation_schemas.py`.

---

## 10. WS7 — Actualizar `RUNTIME_SSOT_v1` (normativa)

**Objetivo.** Que el SSOT del runtime reconozca la arquitectura viva real. Hoy declara
`RuntimeRunner`/`OrchestratorLifecycle` (legacy) como núcleo normativo y no menciona
`LifeKernel` ni la capa de conjunción — esto rompe el `depends_on` de todo contrato nuevo
(invariante de gobierno 3.2.1 del canon).

**Anclas.** `canon/normative/RUNTIME_SSOT_v1.md` (desfasado); `docs/analysis/19_operational_conjunction.md`
(diseño de la conjunción); `runtime/life/kernel.py` (núcleo real).

**Qué construir.** Actualizar `RUNTIME_SSOT_v1.md` (o emitir `RUNTIME_SSOT_v2.md` que lo
supersede, con `supersedes:` y tabla de equivalencias) reconociendo: `LifeKernel` como bucle
vital soberano, `OperationalConjunctionLayer` como gate operacional por decisión, y el
`ScenarioEpisodeRunner` como unidad de episodio certificable. ADR asociado en `governance/adr/`.
Mover `docs/adr/ADR_EXT_OPEN_THINKER_CONFLICT_RESOLVER.md` a `governance/adr/` (fuente única de
ADRs, hallazgo del mapa).

**Disciplina.** Cambio documental normativo → requiere ADR (canon §3.2.2) y no puede
contradecir axiomas. No toca código.

**Criterios.** El SSOT nombra `LifeKernel` + conjunción como núcleo; existe ADR de la
actualización; `depends_on` de las piezas nuevas de Fase 0 puede encadenar al SSOT
actualizado; los ADRs viven en un solo directorio.

---

## 11. Secuencia y gates de salida

Orden recomendado (dependencias reales):

```
WS0 (keystone)  →  WS3 (correlación)  →  WS1 + WS2 (storage, paralelizables)
   →  WS4 (ledger + find_events)  →  WS5 (reality gate shadow)
   →  WS6 (consolidar WIP)  →  WS7 (SSOT)
```

Razón del orden: WS0 enciende la capa ya construida (máximo valor, destraba los tests de los
gates dormidos). WS3 es el sustrato de trazabilidad que WS4 y WS5 consumen. WS1/WS2 son
independientes y baratos. WS4 necesita el patrón de columnas de WS3. WS5 se apoya en el ledger
indexado de WS4. WS6/WS7 cierran consistencia y normativa.

**Gate de salida por WS:** los criterios de aceptación de cada sección, más test de paridad
byte-idéntica (flag off) y suite completa verde en CPU sin GPU/Postgres.

**Gate de salida global de Fase 0** (Fable §2): una corrida seedeada de `scripts/life_kernel.py`
(a) produce cadenas decisión→episodio→certificado reconstruibles al 100 % por IDs; (b) con
`RNFE_HOST_SENSING=1`, `resource_pressure > 0` variando con la carga y la rama `sleep`
alcanzable; (c) reality gate emitiendo en shadow; (d) suite completa verde en CPU; (e) conducta
nominal byte-idéntica a HEAD salvo eventos aditivos; (f) `PRAGMA journal_mode=wal` activo.

---

## 12. Protocolo de PR para Codex (definición de terminado)

Por cada WS, la PR entrega:
1. **ADR** si es estructural (contexto → decisión → hipótesis falsable → costo en hardware
   objetivo → plan de rollback), en `governance/adr/`.
2. **Contratos** (dataclass frozen + schema en `contracts/` si persiste + eventos) antes que
   la lógica.
3. **Tests**: unitarios del subsistema + el **test de paridad byte-idéntica** con flag off +
   regresión. Suite completa verde en CPU pura.
4. **Evidencia**: si el WS cambia conducta bajo flag on, una corrida seedeada documentada
   (smoke o campaña) con artefactos en `data/reports/` (formato `manifest.json`/`REPORT.md`/
   `verdict.json`).
5. **Doc de análisis**: `docs/analysis/NN_*.md` del módulo tocado anota el cambio; el índice no
   queda desincronizado.

Definición de terminado: criterios de aceptación cumplidos y verificados, flag off por defecto,
gate de salida del WS verde, evidencia archivada, ninguna regla de §1 violada.

---

*Capa D: puede tensionar capas superiores solo vía ADR y ruta de promoción (canon §1.2). Estas
orientaciones concretan la Fase 0 del plano de Fable sin contradecirla; toda desviación que
Codex necesite se resuelve con ADR, no de facto.*
