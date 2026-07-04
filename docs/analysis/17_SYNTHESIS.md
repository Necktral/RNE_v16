# 17 — Síntesis cruzada del análisis

Cierre del análisis línea-por-línea de **todo** el repositorio rnfe.v15. Consolida los 16
documentos previos. Fuente de verdad: el código (los `.md` históricos están desfasados).

---

## 1. La historia arquitectónica: dos generaciones conviviendo

El repo es la fusión de **dos arquitecturas** en distinto estado de madurez:

### (A) Legacy "AEON FENIX-Δ" — el AGI orquestado original (deuda)
`runtime/core/` (Orchestrator, training_loop, QuantumDistributedTrainer, probabilistic_models,
loss, planner, episteme), `runtime/evolution/` (neurogénesis/poda/NAS), `runtime/control/homeostasis/`
(termodinámica/apagado), `exocortex/channels/cli/aeon_main_loop`, y todo `archive/`. Características:
- Física **"cuántica/termodinámica" decorativa** (ruido gaussiano/senoidal con nombres grandilocuentes).
- Entrena sobre **datos aleatorios**; alcanzable solo desde el CLI histórico.
- Partes **rotas** (`data/loader`, `trainer_fenix`, `fenix_agent`, `fase0_cert`) y **stubs**
  (protocolos de apagado solo loguean).

### (B) Nueva arquitectura "organismo / RTCME" — el sistema vivo de los benchmarks
`runtime/{world, reality, organism, reasoning, control/msrc, certification, storage, memory,
symbolic, lotf, smg}` + `scripts/` + `contracts/`. Características:
- Diseño **limpio y coherente**: dataclasses inmutables, contratos formales, gobernanza
  constitucional T5, posteriores bayesianos, control multi-escala (MSRC), morfismos causales.
- Es lo que ejercitan las campañas (`data/benchmarks/*`).

`engines/hnet` (H-Net) y `engines/mamba_vendor` (Mamba SSM) son **terceros** adaptados, aislados
del organismo (solo `generate.py`).

## 2. Flujo del pipeline vivo (un episodio)

```
ScenarioEpisodeRunner.run_episode (world/scenario_runner.py)
 ├─ scenario.observe() → SMGMin.add_observation/create_sign (smg)  [persiste eventos]
 ├─ LOTFMin.parse/check (lotf)                                     [Z3 real solo en DED]
 ├─ MemoryRetrieval.retrieve (memory/mfm_lite)                     [⚠ resultado IGNORADO]
 ├─ scenario.factual/counterfactual transition
 ├─ MetaScheduler.run (reasoning/scheduler_meta)                   [familias core = STUBS]
 │    └─ policy.select_sequence → familias.{abd..prob,ded,ext_open_thinker}
 ├─ build_belief_state (reality/belief_state)
 ├─ transition_organism_state + ViabilityKernel + Constitution (organism)
 └─ PromotionGate.process_episode (certification)
      ├─ evaluate_episode_closure (reality/evaluator) → list_events  [⚠ bug SQLite]
      ├─ ContinuityGuard + IoCProxy
      ├─ ConstitutionalCourtRuntime.ingest_episode (organism/court)  [T5 ON por defecto]
      ├─ CertificateBuilder → write_episode_certificate              [doble escritura si T5]
      └─ memoria micro/meso/macro (si certified)
```

MSRC (`control/msrc`) corre en paralelo como controlador de escala (1x1↔5x5). Todo persiste vía
`storage` (SQLite/Postgres/hybrid).

## 3. Hallazgos consolidados por severidad

### 🔴 [BUG] — corrección
1. **`postgres_store.write_transfer_assessment` upsert parcial** (solo verdict+metadata) → diverge
   de SQLite (`INSERT OR REPLACE` total). [02]
2. **`SQLiteStorageBackend.list_events` filtra `run_id` post-LIMIT** → consultas por run pierden
   filas; **acopla un fallo de storage a `trace_integrity`/certificación**. [02][05][09]
3. **`loss.CompositeLoss.forward` hace `.detach().requires_grad_(True)`** → rompe el backprop. [03]
4. **`data/loader.py` importa `aeon_fenix_delta`** inexistente → ImportError. [03]
5. **`trainer_fenix` y `agents/fenix_agent`** rotos (DummyEnv stub / `rssm_lite` inexistente). [03][10]
6. **`fase0_cert.py` invoca `run_aeon.py` inexistente** (+ `shell=True`). [14]
7. **Incompatibilidad de `HealthStatus`** entre `thermodynamic_governor` y `shutdown_logic`
   (campos distintos) → AttributeError si se cablean. [08]

### 🟠 [RIESGO] — fiabilidad/observabilidad
1. **Dual-write híbrido no estricto traga fallos parciales sin loguear** → SQLite/Postgres divergen
   en silencio (`strict_dual_write=False` por defecto). [02]
2. **`_read_with_fallback` confunde "vacío válido" con "fallo"** → datos potencialmente desfasados. [02]
3. **Amplificación de escritura por episodio**: SMG (≥3 eventos) + EventBus (fichero+DB por evento)
   + court_runtime (snapshot/window/flow/N risk/failure) + materialize_artifact, sobre SQLite **sin
   WAL/busy_timeout** → "database is locked" bajo concurrencia/multiproceso. [02][06][10][03]
4. **Sandbox de auto-modificación no-op sin `apply_fn`** (simula identidad → puede "aceptar" sin
   simular el cambio). [06]
5. **Benchmark heterogéneo resetea la cadena de belief/trayectoria** (runner nuevo por paso). [05]

### ⚫ [MUERTO] — código muerto / no-op
- Imports muertos masivos en `aeon_types`. [01]
- `runs` (tabla huérfana en schema PG). [02]
- `trajectory_window` siempre `None` (`if False`). [04]
- Memoria recuperada **no influye** en la intervención (no-op en ambos runners). [04]
- Protocolos de apagado homeostático = solo logging. [08]
- `EvolutionaryRehabilitationCenter` sin importadores. [09]
- `archive/` entero (14.5K, por política). [16]

### 🔵 [DISEÑO] — observaciones estructurales (selección)
- **Familias de razonamiento core (ABD/ANA/CAU/CTF/PROB) son stubs**; solo DED (Z3), el razonador
  externo (LLM gated) y HEUR/DIA_ADV/EML_SR computan. El "cierre triádico" es mayormente ceremonial. [07]
- **Duplicación de tipos**: ≥3 `HealthStatus`, 2 `EpistemeMeter`, 2 `CombinedModel`, 2 `EventBus`,
  2 cargadores de config, 2 kernels de viabilidad, ≥3 cómputos de "continuidad". [03][08][10]
- **Sesgo térmico** en continuidad/causalidad (`reality/continuity`, `certification/continuity_guard`
  hardcodean "temperature") → degenera en escenarios no térmicos. [05][09]
- **Heurísticos con números mágicos sin calibrar** en belief/risk/IoC/policy (p. ej. IoC techo 0.90). [05][06][07][09]
- **Naming "cuántico" cosmético** (episteme, loss_elite, meta_optimizer). [03][09]
- **Dialectos JSON-Schema mixtos** (draft 2020-12 vs draft-07). [01]

### 📄 [DOC] — contradicción doc↔código (confirma "docs desfasados")
- `alignment.py` dice "algoritmo húngaro" pero usa greedy. [04]
- Contratos formales (17 JSON schemas) **por detrás** del modelo de records (faltan ~9 schemas). [01][02]
- Los `.md` de fase (PHASE*/EXPERIMENT*) describen un sistema más completo del que el código realiza
  (familias stub, homeostasis stub).

## 4. Temas transversales

1. **El valor cognitivo real es estrecho.** Fuera de DED (Z3) y el razonador externo (lab-only,
   gated), el "razonamiento" es andamiaje. Interpretar los benchmarks de "ganancia cognitiva" con
   esto presente: miden orquestación/gating/cierre, no inferencia de las familias core.
2. **La nueva arquitectura es seria; la legacy es lastre.** `organism/reasoning/reality/world/msrc/
   certification/storage` están bien diseñados. `core` (Orchestrator), `evolution`, `homeostasis` y
   `archive` son deuda en distintos grados.
3. **Storage es el cuello de botella de fiabilidad.** El bug de `list_events`, la divergencia
   silenciosa del dual-write y la amplificación de escritura sobre SQLite sin WAL convergen en el
   mismo punto crítico (25 importadores).
4. **El razonador externo es la pieza estrella.** Triple gate, parsing defensivo, telemetría rica —
   calidad de producción. Es el foco correcto de la rama actual.

## 5. Recomendaciones priorizadas

| Prioridad | Acción |
|---|---|
| P0 | Arreglar `list_events` (filtrar run_id en SQL) y el upsert parcial PG de transfer_assessments. |
| P0 | Loguear divergencias de dual-write; activar WAL+busy_timeout en SQLite. |
| P1 | Decidir destino de las familias core stub: implementarlas o documentar explícitamente que el cierre es estructural. |
| P1 | Reparar/retirar el sandbox de auto-modificación no-op (seguridad). |
| P1 | Unificar HealthStatus/EpistemeMeter/continuidad duplicados; quitar el sesgo térmico. |
| P2 | Retirar o aislar formalmente la capa legacy (core Orchestrator, evolution, homeostasis stub, fenix_agent/trainer_fenix/data_loader rotos). |
| P2 | Completar los JSON schemas faltantes; unificar dialecto. |
| P3 | Considerar eliminar `archive/` del árbol (ya está en git); revisar la DB de ~890MB versionada. |

---

## Cobertura del análisis
16 módulos analizados, documentados en `docs/analysis/01..16`. Código vivo del proyecto leído
íntegro; terceros (H-Net, Mamba) caracterizados y atribuidos; legacy/archive caracterizado y
confirmado aislado. Índice y estado en [00_INDEX.md](00_INDEX.md).
