# 10 — Resto de `runtime/` (lotf, smg, memory, symbolic, telemetry, agents, utils)

Subpaquetes pequeños; varios en el **camino vivo del episodio**, otros legacy/stub.

## Camino vivo (sano)
- **`lotf/lotf_min.py`** (`LOTFMin`): parser de descenso recursivo booleano (NOT/AND/OR/`->`) +
  type-checker. Precedencia correcta (`->` > OR > AND > NOT), paréntesis balanceados,
  símbolos tipados contra `type_env`. Limpio y correcto. (Operadores en MAYÚSCULA; `->` implicación.)
- **`smg/smg_min.py`** (`SMGMin`): grafo de signos en memoria (observations/signs/relations) que
  **persiste cada operación como evento** (`smg.observation_added`/`sign_created`/`relation_created`).
  → **[RIESGO] amplificación de escritura**: ≥3 eventos por episodio, sumados a `court_runtime` y
  `event_bus`; agrava el bloqueo SQLite ([02_storage.md](02_storage.md)).
- **`memory/mfm_lite/`**: memoria multi-escala. `retrieval.MemoryRetrieval` (scoring por overlap
  Jaccard de tokens + filtro de escenario strict/analogical con penalización 0.5 y
  `retrieval_metrics` para observabilidad), `condenser` (micro/meso/macro), `episode_store`
  (write_*), `promotion.MacroPromotion`. Limpio. (Recordatorio: el runner recupera memoria pero
  **no la usa** — el no-op de [04_world.md](04_world.md).)
- **`symbolic/eml/`**: regresión simbólica "shadow". `tree.ExprNode` (AST), `search.generate_candidates`,
  `scoring.score_candidate`, `runner.EMLRunner`, `advisory`. Y **`safe_eval.py`**: evaluador numérico
  **genuinamente seguro** sobre el AST (ops whitelisted add/sub/mul/div/pow2/log1p/exp, chequeo de
  dominio, clipping, sin `eval` de Python). No hay riesgo de inyección — buen diseño.

## Legacy / stub / roto
### [BUG] `agents/fenix_agent.py` no importa
`from .rssm_lite import RSSMLite` (fenix_agent.py:4) — **no existe** `runtime/agents/rssm_lite.py`
(solo `runtime/core/rssm_lite2.py` con clase `RSSMLite2`). Importar `FenixAgent` lanza
`ModuleNotFoundError`. Coherente con `trainer_fenix` roto ([03_core.md](03_core.md)). Además tiene
un `print("🧠 FenixAgent v2 …")` en `forward`. Agente legacy no funcional.

### [DISEÑO] DOS `EpistemeMeter` distintos
- `runtime/core/episteme.py::EpistemeMeter` (torch, Fisher matricial, "PID cuántico", 423 LOC).
- `runtime/telemetry/episteme/episteme_meter.py::EpistemeMeter` (numpy, EMA, mutual_info; 70 LOC)
  con varios stubs (`get_global_efficiency→1.0`, `get_accumulated_energy→0.0`, `apply_noise→log`) y
  defaults `np.random.uniform` si faltan datos. `pynvml.nvmlInit()` en el constructor.
Dos implementaciones del mismo concepto con APIs y semánticas distintas. Confusión de nombres.

### [DISEÑO] utils son stubs de migración + shim de typo
- `utils/resilience.ResilienceMechanism`: métodos (`initiate_memory_pruning`, `compress_memory`)
  que devuelven dicts de estado pero **no hacen nada** ("Implementación simple para evitar ruptura").
- `utils/resilence.py` (sic): shim que reexporta `resilience` para mantener un **typo histórico**.
- `utils/logging_utils.py`: puente a `telemetry.logging.logging_utils`.

### Telemetría
- `telemetry/collector.py`, `snapshot_service.py`, `logging/logging_utils.py`: utilidades de
  recolección/snapshot usadas por el training loop legacy. `telemetry/event_bus.py` y
  `event_log_sqlite.py` son shims de 4 líneas hacia `core/`.

### `memory/persistence/persistence.py` (230)
`StatePreserver` (el que usa `shutdown_logic`). Persistencia de estado legacy (alias
`src.persistence`).

## Veredicto
Los módulos del **camino vivo del episodio** (lotf, smg, mfm_lite, eml+safe_eval) son **limpios y
correctos**; `safe_eval` destaca por ser realmente seguro. La deuda está en lo legacy:
`fenix_agent` roto, doble `EpistemeMeter`, `utils` stub, y la amplificación de escritura del SMG.
