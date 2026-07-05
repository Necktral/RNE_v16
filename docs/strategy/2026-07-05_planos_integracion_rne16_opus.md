---
title: PLANOS_INTEGRACION_RNE16_PARA_OPUS
status: experimental
version: 1.0.0
date: 2026-07-05
owner: Wis
depends_on:
  - CANON_RNFE_v3_2_rc1.md
  - RUNTIME_SSOT_v1.md
  - SSOT_RAZONAMIENTOS_RNFE_v1.md
  - 2026-06-17_self_sustaining_cognitive_gain.md
  - docs/analysis/19_operational_conjunction.md
supersedes: []
notes:
  - Documento de dirección NO-NORMATIVO (Capa D con elementos de Capa C). No redefine el canon; toda pieza nueva entra como `experimental` con ruta de promoción.
  - Es el plano maestro de integración para el implementador (Opus). Las orientaciones son arquitectónicas; no contienen código.
---

# Planos de integración RNE v16 — instrucciones para Opus

Este documento convierte el estado real del repositorio (auditado línea por línea en `docs/analysis/`, mapeado de nuevo el 2026-07-05) en una secuencia de construcción gobernada que cubre once frentes: **cimientos de fiabilidad**, y los diez ejes pedidos — eficiencia en hardware normal, causalidad, memoria/RAG, razonamiento, agentes gobernados, compensaciones, validación, trazabilidad, degradación inteligente, autonomía ilimitada por política — más el **ecosistema de redes neuronales especializadas** que se integra sobre esos ejes.

---

## 0. Reglas de juego para Opus (mandato de implementación)

Estas reglas son vinculantes para todo lo que sigue. Si una mejora de este documento entra en conflicto con ellas, ganan las reglas.

1. **Precedencia del canon.** `canon/normative/CANON_RNFE_v3_2_rc1.md` es SSOT. Nada de este plan puede violar los axiomas A1–A9 ni los invariantes del núcleo (§3). En particular: A9 (economía de razón — nunca activar por defecto el razonamiento más caro), 10.1.3 (no declarar causalidad real sin C-GWM), 10.1.5 (no absorber conocimiento externo directo al tronco), 10.1.7 (multiagente no sustituye al núcleo cognitivo).
2. **Disciplina sombra-primero** (patrón `ADR_BUCLE_A_ACTIVATION`): toda capacidad nueva nace detrás de un flag `RNFE_*` off-por-defecto con conducta nominal **byte-idéntica**. Secuencia obligatoria: sombra → evidencia (campaña con verdict) → activación gated → default (solo con reaparición cross-seed, A7).
3. **Cero dependencias nuevas en el camino vivo.** `runtime/` sigue siendo Python puro + stdlib (más `z3` y `psycopg` ya admitidos). Dependencias pesadas (torch, dowhy, causal-learn, GGUF nuevos) viven en `scripts/`, `lab/` o `engines/`, con frontera verificada por test AST (patrón `tests/contracts/test_boundary_rules.py`).
4. **Todo cambio estructural exige ADR** en `governance/adr/` (un solo lugar: migrar `docs/adr/` allí), con hipótesis falsable, costo estimado en el hardware objetivo y plan de rollback (canon §5, §11).
5. **Contratos primero.** Toda pieza nueva define: dataclass frozen JSON-friendly (patrón `runtime/conjunction/contracts.py`), schema en `contracts/` cuando cruza frontera, y eventos append-only vía `StorageFacade` con el sobre de correlación (§9.M1). Los contratos que nazcan en código se retro-formalizan en `contracts/` en la misma PR.
6. **Regla del advisor.** Ningún modelo neuronal (local o externo) **actúa**: propone, puntúa o explica como `EvidenceItem` auditado. La decisión queda siempre en supervisor + conjunción + política. Esto ya es el contrato del razonador externo y de `maybe_llm_augment`; se generaliza a todo el ecosistema neuronal (§13).
7. **Regla de incertidumbre.** Ningún modelo neuronal entra ni en sombra sin: (a) estimación de incertidumbre en su salida, (b) fallback determinista definido, (c) presupuesto de cómputo asignado, (d) experimento falsable pre-registrado con criterio de refutación.
8. **Intocables sin ADR + campaña propia:** el orden `CORE_SEQUENCE` (abd→ana→cau→ctf→ded→prob, PROB último), el contrato de `ScenarioEpisodeRunner` (invariante `ADR_MSRC`), las 3 métricas de continuidad (`ADR_CONTINUITY_TRES_METRICAS` prohíbe fusionarlas), la frontera `runtime ↛ exocortex ↛ archive`.
9. **Evidencia estándar.** Cada campaña produce `manifest.json` + `REPORT.md` + `verdict.json` en `data/reports/<campaña>/` con seeds, commit y flags — el formato ya usado por `bucle_a_activation`. Los dictámenes negativos se publican sin adornos (cultura vigente del repo).
10. **Hardware objetivo.** Todo debe ser útil en una PC común: CPU multinúcleo, 8–16 GB RAM, GPU de ≤8 GB VRAM **opcional**. El criterio de promoción del canon (§5.1.4) es explícito al respecto. Nada del camino vivo puede requerir GPU.

---

## 1. Diagnóstico consolidado (2026-07-05)

Lo que es **real y funciona**: el bucle vital (`runtime/life/LifeKernel` + `scripts/life_kernel.py`), el episodio cognitivo de 14 pasos (`runtime/world/scenario_runner.py`), las 6 familias core con inferencia simbólica real + DED con Z3, la certificación por episodio con CVaR calibrado contra 480 runs, la memoria multiescala mínima (`mfm_lite`), el storage triple (SQLite/Postgres/híbrido), la capa de conjunción operacional (gate por decisión con evidencia, validadores, compensaciones — 10/10 tests), la autoevolución de knobs con sandbox constitucional, MSRC, y una maquinaria de campañas con dictámenes honestos.

Las **siete brechas transversales** que este plan cierra:

| # | Brecha | Evidencia |
|---|---|---|
| 1 | Los lazos metabólicos están en sombra: reward-guided, λV, override, enforcement de riesgo, todo off por defecto | roadmap 2026-06-17; A2 gated |
| 2 | El ruteo de cómputo por tiers es **decorativo**: se persiste, nada ejecuta distinto | `runtime/conjunction/router.py` sin consumidores aguas abajo |
| 3 | El organismo no siente su cuerpo: `NullVRAMSampler`, `resource_pressure≈0`, rama `sleep` muerta | `runtime/life/kernel.py:543`, `vitals.py` |
| 4 | Los IDs no se propagan entre capas: decisión→episodio→traza→certificado se reconstruye por adyacencia temporal | `life.step.completed` sin `episode_id` |
| 5 | La gobernanza es código, no política: umbrales hardcodeados, `human_approval` sin canal, kill-switch huérfano, `runtime/agents` vacío | `supervisor.py`, `service.py:152-233` |
| 6 | Conocimiento sin ciclo de vida: TTL letra muerta, memoria scoped a `run_id` (arranque en frío perpetuo), sin retracción tras rollback | `mfm_lite`, brecha Bucle B |
| 7 | Los veredictos avanzados no gobiernan: S-I-E/IoC*/reality-gate corren en sombra o fuera del bucle vivo | `certificate_builder.py`, `LifeKernel` nunca invoca `RealityValidationService` |

Trabajo en curso del owner que este plan **integra, no duplica**: `runtime/reasoning/scheduler_meta/governance.py` (envolvente de gobernanza META), `runtime/reasoning/scheduler_meta/degradation.py` (plan de degradación por evidencia), `runtime/world/causal_attestation.py` + tests (atestación factual/contrafactual). Son embriones de §7 (razonamiento), §11 (degradación) y §4 (causalidad): Opus debe consolidarlos — contratos tipados, eventos con correlación, tests de paridad — antes de construir piezas paralelas.

---

## 2. Fase 0 — Cimientos (bloqueante para todo lo demás)

Nada de los ejes se activa en vivo sin esta fase. Orden interno recomendado tal como está listado.

1. **Merge de `codex/operational-conjunction` a `main`.** Toda la obra se ancla en `runtime/conjunction` + su integración en `LifeKernel`; no puede vivir en una rama.
2. **Fiabilidad de storage (P0 de la auditoría, parcialmente hecho):** WAL + `busy_timeout` en SQLite; el híbrido deja de tragar fallos parciales (evento `storage.write_divergence` en el backend vivo); fix del upsert parcial de `transfer_assessments` en Postgres; `_read_with_fallback` distingue "vacío válido" de "fallo".
3. **Sobre de correlación causal** (§9.M1): `CausalContext` (run_id, step_index, decision_id, episode_id, parent_event_id) propagado decisión→episodio→traza→certificado→memoria. Es prerequisito de la atribución causal, la auditoría y las sagas.
4. **Paridad e índices del ledger** (§9.M2): columnas/índices `event_id`, `run_id`, `episode_id`, `decision_id` en SQLite; `find_events` indexado en la facade; eliminar el escaneo lineal de 500 eventos del cierre triádico.
5. **Sensado real de recursos** (§3.M1/§11.M1): `HostResourceSampler` (CPU/RAM/swap/VRAM-si-hay, stdlib con psutil opcional) + `ResourceProfile` al génesis persistido en el checkpoint. Reemplaza `NullVRAMSampler` y alimenta `resource_pressure` real en vitals, conjunción, MSRC y B_safe.
6. **Gate de realidad dentro del bucle vivo** (§8.M2): `reality_assessment` por episodio vivo + chequeo ventaneado cada N ciclos contra `GATE_PROFILES['ci']`, señal `reality_gate_ok` en vitals.
7. **Consolidación del WIP del owner:** la envolvente de gobernanza META y la atestación causal quedan con contrato tipado, eventos propios y tests de paridad nominal.
8. **Actualizar `RUNTIME_SSOT_v1.md`** para reconocer `LifeKernel` y la capa de conjunción como componentes normativos (hoy declara núcleo al runner legacy) — sin esto, ningún contrato nuevo puede encadenar `depends_on` correctamente (invariante de gobierno 3.2.1).

**Criterio de salida de Fase 0:** una corrida seedeada de `scripts/life_kernel.py` produce cadenas decisión→episodio→certificado reconstruibles al 100 % por IDs; `resource_pressure > 0` variando con carga real; reality gate emitiendo; suite completa verde en CPU sin GPU/Postgres; conducta nominal byte-idéntica a HEAD salvo eventos aditivos.

---

## 3. Eje 1 — Eficiencia en hardware normal

**Diagnóstico.** El camino vivo ya corre en CPU en milisegundos; el problema no es correr sino **saber cuánto cómputo se tiene y regular la conducta**. El ruteo de tiers no ejecuta nada distinto; no hay presupuesto por ciclo; hay hot-paths O(n) que degradan corridas largas; los engines GPU explotan al importar; el razonador externo paga la carga del modelo en cada llamada (60–96 s).

**Mejoras en orden:**

| # | Mejora | Esencia | Ancla |
|---|---|---|---|
| M1 | HostResourceSampler | Sensado real (CPU/RAM/VRAM) con caché TTL, cableado en kernel, vitals, conjunción, MSRC, B_safe | `runtime/control/msrc/vram_sampler.py`, `runtime/life/vitals.py` |
| M2 | Presupuesto por ciclo + tiers ejecutables | `cycle_budget_ms` medido con `perf_counter`; tier→directivas reales: tier_0 ⇒ `baseline_fixed` + budget mínimo del META + retrieval reducido; tier_2 ⇒ overlays completos; tier_3 ⇒ externo | `router.py`, `_apply_operational_gate`, `policy.trim_to_budget` |
| M3 | Hot-paths O(n) → SQL dirigido | cierre de episodio por consulta indexada; filtro de escenario de memoria en SQL; promoción macro agregada en SQL; TTL efectivo | `evaluator.py`, `sqlite_store.py`, `mfm_lite` |
| M4 | Perfil de hardware al génesis | `HardwareProfile` en el checkpoint; defaults derivados (sin GPU ⇒ tier máx. 2, `allow_external=False`); evento `life.hardware_profile.changed` al migrar de host | `kernel.py`, `checkpoints.py`, ADR que actualiza el supuesto WSL2 del canon |
| M5 | Razonador externo económico | backend `llama-server` residente opcional (arranque perezoso, shutdown por inactividad), caché de respuestas por hash de prompt validada por schema, presupuesto adaptativo | `llama_cpp_client.py`, `gating.py` |
| M6 | Cuarentena eficiente de engines GPU | puentes `hnet/`/`mamba_ssm/` 100 % lazy (PEP 562), `requirements-gpu.txt` segregado, ADR sobre el destino de H-Net/Mamba | puentes raíz, `engines/` |

**Regla de oro (A9):** el organismo activa el razonamiento más barato suficiente; pagar más requiere señal (conflicto, insuficiencia epistémica, régimen validado) y presupuesto disponible.

---

## 4. Eje 2 — Causalidad

**Diagnóstico.** Hay un embrión causal serio pero declarativo: firmas causales por escenario escritas a mano, nunca estimadas contra datos; CAU solo compara direcciones; el contrafactual es un oráculo del mundo sintético; la atribución de impacto por familia es un proxy multiplicativo, no causal; dowhy/causal-learn/pgmpy son dependencias fantasma con cero imports. La atestación causal del owner (`causal_attestation.py`) ya empieza a puntuar soporte factual/contrafactual — es la semilla de M2.

**Mejoras en orden (M1→M2→M4→M3→M5):**

- **M1 — `runtime/causal/` con SCM de primera clase.** Python puro: nodos desde la firma del escenario, ecuaciones lineales sembradas por polaridad/strength, ruido exógeno por nodo, operador `do()` explícito. Persistido como artifact content-addressable (`kind='causal_model'`). Evento `causal.intervention.applied` por episodio distinguiendo intervención de observación. *Esto es el camino a C-GWM-min: sin él, el canon prohíbe declarar causalidad real (10.1.3).*
- **M2 — Calibración empírica de la firma.** `CausalCalibrator` estima por intervención el efecto real (media + cota Agresti-Coull, reutilizando el estimador del risk_engine) desde el ledger; produce `CausalEdgeCalibration`; detecta deriva (`causal.model.drift`) y alimenta a la conjunción soporte causal **cuantitativo** (hoy el validador causal opera sobre supuestos declarados). Integrar aquí la atestación del owner.
- **M4 — Atribución causal decisión→resultado.** Reemplazar el proxy por contraste intervencional: re-ejecución sandbox seedeada del episodio con/sin familia (el episodio es determinista por seed ⇒ contrafactual computable exacto); brazo de control para automodificaciones (el ΔIoC pareado decide commit/revert, no el antes/después crudo). Prerequisito conceptual de la activación A2 honesta.
- **M3 — Contrafactual de modelo (Pearl 3 pasos).** Abducción de ruido → sustitución de acción → predicción, con `counterfactual_fidelity` medida contra el oráculo mientras exista; `ctf_source='model'` como fallback cuando el mundo no ofrezca oráculo. *Este módulo ES el world-model v0 del ecosistema neuronal (§13.H).*
- **M5 — Lab offline de descubrimiento causal.** `scripts/causal_discovery_lab.py` (dowhy/causal-learn SOLO en scripts): discovery + estimación + refutación sobre el ledger exportado; compara grafo descubierto vs declarado; promoción de topología únicamente vía ADR. Cero imports de esas libs bajo `runtime/` (test AST).

---

## 5. Eje 3 — Memoria / RAG

**Diagnóstico.** `mfm_lite` funciona pero es léxico (Jaccard), scoped a `run_id` (el organismo no recuerda entre corridas), con TTL fantasma, condenser acoplado al escenario térmico, y la búsqueda de evidencia de la conjunción escanea 100 eventos por substring.

**Mejoras en orden:**

- **M1 — Metadata canónica + memoria cross-run.** Escribir SIEMPRE `scenario_name/version/config_hash/compatibility_class` (lo exige `MEMORY_COMPATIBILITY_POLICY_v1 §3`); filtro de escenario en SQL; `organism_id` estable en el checkpoint para que la memoria sobreviva reinicios (`RNFE_MEMORY_CROSS_RUN`, sombra primero). *Inicio real del Bucle B.*
- **M2 — Consolidación episódica→semántica y sueño funcional.** `pattern_key` derivado de la firma causal (no de `context['formula']`); `ConsolidationService` ejecutado por la acción `sleep` del supervisor (hoy semihueca): compactar micros, agregar mesos, promover macros, purgar vencidos. El sueño pasa a tener función biológica real.
- **M3 — Olvido gobernado.** TTL efectivo con archivado auditable (`memory_tombstone` content-addressable + evento `memory.expired`); macro nunca expira por tiempo; nada referenciado por certificado en ventana de retención se borra físicamente.
- **M4 — Recuperación semántica híbrida (sombra).** `EmbeddingProvider` con dos implementaciones: `HashedNgramEmbedder` stdlib-puro (default, determinista, <1 ms) y `LlamaCppEmbedder` opcional (GGUF de embeddings, reutiliza la infra del externo). Tabla `memory_embeddings` en ambos backends; score híbrido léxico+coseno; modo shadow mide divergencia de ranking sin alterar hits.
- **M5 — RAG gobernado sobre el ledger y el canon.** `KnowledgeIndex` con dos corpus: eventos operacionales, y canon/ADRs chunked por sección (solo lectura, con cita doc+sección+hash). Reemplaza `_exact_event_search`; la compensación `expand_retrieval` pasa a recuperar de verdad. Un chunk del canon llega a la conjunción como `EvidenceItem canonical=True`. *Nota 10.1.5: el RAG recupera y cita; no escribe conocimiento externo en la memoria viva — la memoria solo nace de episodios certificados.*

**Integración obligatoria:** el índice vectorial y el KnowledgeIndex **respetan la retracción** (§8 compensaciones M3): `status=retracted` se excluye en SQL y dispara re-index.

---

## 6. Eje 4 — Razonamiento

**Diagnóstico.** META es maduro (features→budget→scoring→secuencia validada→ejecución→traza→recompensa), pero: el externo admitido por ADR no puede correr en runtime (`ValueError` en `policy.select_sequence`; guard y perfil viven en el benchmark); no hay metacognición (uncertainty=0.25 fijo); los críticos HEUR/DIA_ADV/FAL_GUARD son flags de 2 features; el contrato de motor SSOT §8 no se implementa en input; la selección guiada por recompensa está apagada y su pericia muere con el proceso.

**Mejoras en orden:**

- **M1 — Portar el conflict resolver a runtime gobernado.** guard → `runtime/reasoning/external_models/guard.py`; perfil `core_plus_external_reasoner_gated_v1` registrado en PROFILES; el `ValueError` se reemplaza por verificación contra `validate_external_reasoner_admission` + régimen validado; `maybe_llm_augment` pasa a validar schema y guard (hoy viola la regla del ADR). La evidencia dura: core solo 0.000 vs gated 0.875–1.000 en el régimen de conflicto — es el único recurso que resuelve ese régimen.
- **M2 — Metacognición operativa (`EpistemicAssessor`).** Determinista, con señales existentes: ancho Agresti-Coull de PROB, conflicto CAU≠CTF, contradicciones SMG, hit-rate de memoria, divergencia posterior→factual. Veredicto `confident/uncertain/ignorant` por episodio (evento `reasoning.epistemic`), alimenta `uncertainty` real de context_features, la acción `consult_external` del supervisor y el tier_3 de la conjunción. *Es la versión sin-red-neuronal del EDL (§13.G): primero la señal determinista, después la cabeza aprendida.*
- **M3 — Presupuestos reales (contrato SSOT §8).** `EngineInput` frozen (context_id, goal, budget, risk_budget, trace_policy) + `BudgetLedger` — **el mismo ledger** que usa la conjunción y el ciclo vital (ver §16, decisión I-1). Episodio sin presupuesto ⇒ skip auditable del externo.
- **M4 — Escalera de resolución de conflictos.** `ConflictDetector` tipificado sobre el blackboard (CAU≠CTF, DED-refuta-ABD vs PROB alto, memoria vs inferencia, regla heredada vs firma) + escalera determinista de coste creciente que culmina en el externo vía gate+guard.
- **M5 — Críticos reales.** HEUR = triage de presupuesto con las 14 features + historial; DIA_ADV = contraejemplos ejecutables vía `simulate_counterfactual` (contrato no mutante garantizado); FAL_GUARD = patrones de falacia sobre el blackboard con Z3/LOT-F.
- **M6 — Activación A2 + pericia cross-run.** Persistir `(regime, family)→Δr̄` como artifact/eventos; el selector se siembra del histórico al arrancar (Bucle B del razonamiento); modulación acotada del score estático sin tocar núcleo ni floors; campaña A2 con el formato estándar de evidencia, **usando la atribución por ablación de §4.M4**, gateada por R1.

---

## 7. Eje 5 — Agentes gobernados

**Diagnóstico.** El tejido (conjunción + `AgentPolicy`) existe y es real; los agentes no: `runtime/agents` está vacío, el único rol es `life_kernel` autoasignado con budget hardcodeado 1.0, `human_approval_required_actions` siempre vacío, y el tier no cambia la ejecución.

**Mejoras en orden (M1→M3→M2→M4→M5→M6):**

- **M1 — Contrato canónico de agente + registro.** `GovernedAgentManifest` (mandato textual, ciclo de vida proposed→sandboxed→active→suspended→retired, supervisor jerárquico, documento canónico que lo autoriza) + `AgentRegistry` persistido con eventos. El `AgentPolicy` del `life_kernel` pasa a venir del registro (byte-idéntico campo a campo). Nueva política normativa `AGENT_GOVERNANCE_POLICY_v1`.
- **M3 — Presupuesto real por agente.** `BudgetLedger` único (§16 I-1) debitando coste de ruta + duración medida + coste de razonamiento (los mismos λE ya computados); saldo en el checkpoint; `RNFE_AGENT_BUDGET_ENFORCEMENT` habilita degradar a tier_0 y dormir por escasez.
- **M2 — Ruteo con efecto real.** Ya descrito en §3.M2; aquí el énfasis es el transporte tipado (`OperationalDirectives` en vez de dict) respetando el invariante ADR_MSRC de no tocar el contrato del runner.
- **M4 — Mandato unificado de automodificación.** El `self_modify` del supervisor se vuelve `ModificationMandate` tipado; `AutoEvolutionController` solo aplica con mandato vigente; su detección autónoma se convierte en **propuesta ascendente**; la evidencia `validated_plan` referencia el artifact real del sandbox (nunca el placeholder de confianza 0.55). Cierra el doble lazo de automodificación.
- **M5 — Canal real de aprobación humana.** Cola sobre el ledger: `operational.approval.requested` (con hash de contexto y TTL) + CLI `scripts/organism_control.py approve/deny/pending`; grants de un solo uso, atados a operación y run. La acción se degrada mientras espera, nunca bloquea en seco.
- **M6 — Supervisión de segundo orden.** Fin del fail-open silencioso: contadores de fallos de infraestructura como señal `conjunction_health` en vitals; fail-closed selectivo (acciones críticas se degradan cuando la infra falla; `act` sobrevive); escalamiento del supervisor ante patrones (3 bloqueos consecutivos ⇒ modo conservador).

---

## 8. Eje 6 — Compensaciones (patrón saga)

**Diagnóstico.** Existe el 50 % del patrón: compensaciones PRE-ejecución (matriz de 16 códigos con lazo de dos pasadas) y tres islas POST no correlacionadas (revert de knobs en RAM — un crash a mitad de ventana deja la mutación aplicada sin vigilancia —, restore de checkpoint que no retracta nada derivado, y override T5 que reescribe el certificado sin tocar la memoria que causó).

**Mejoras en orden:**

- **M1 — Saga journal de primera clase.** `EffectRecord` (effect_class ∈ reversible/compensable/irreversible; status pending/committed/compensated/orphaned) declarado ANTES de mutar y asentado después, en los cuatro sitios reales de mutación (knobs, checkpoint/restore, promoción de memoria, transición de escala). Escaneo de huérfanos al arrancar.
- **M2 — Rollback certificado del kernel.** Verificación de hash del artifact al restaurar; cómputo del **span descartado** (episodios, certificados, memorias entre checkpoint y presente); `identity_continuity ≥ 0.60` post-restore o cuarentena; evento `life.rollback.certified` validando contra `contracts/rollback.schema.json` (hoy huérfano — este es su productor).
- **M3 — Retracción de memoria derivada.** `retract_memory_records` (marca `retracted`, nunca borra — ledger append-only): la invoca el rollback (span descartado), el override T5 (memorias del certificado degradado) y el TTL. El conocimiento de episodios invalidados deja de contaminar el futuro.
- **M4 — Saga de automodificación crash-safe.** El estado de la saga (proposal_id, knob_backup, ventana restante, deltas) viaja en el checkpoint de vida; al restaurar, la ventana se reanuda o se ejecuta revert conservador. Nunca queda una mutación aplicada sin saga.
- **M5 — Clasificación de reversibilidad + autorización.** Mapa acción→clase en `OperationalConstraints`; irreversible sin autorización de política = fail duro; `shutdown` y `consult_external` (efecto externo) son irreversibles; el canal de §7.M5 es la vía de autorización.
- **M6 — Compensación de divergencia dual-write.** Todo fallo parcial asienta `storage.write_divergence`; `scripts/reconcile_storage.py` re-aplica idempotente; implementa por fin la `POLICY_BACKUP_RECONCILIATION` ya escrita.

---

## 9. Eje 7 — Trazabilidad

(Sus M1 y M2 son parte de Fase 0; se listan aquí completos.)

- **M1 — `CausalContext` propagado** por todo el camino vivo: un `trace_id` por episodio (hoy uuid4 por paso), `decision_id`/`step_index` en `episode.closed`, certificado enlazando la traza. Criterio: cadenas reconstruibles por query, cero inferencia por timestamp.
- **M2 — Paridad del ledger SQLite + `find_events` indexado.** `event_id`, `payload_hash`, `run_id`, `episode_id`, `decision_id` como columnas/índices; `seq` como orden total; migración idempotente sobre DBs legacy.
- **M3 — Cadena de identidad y linaje.** Checkpoints encadenados (`parent_checkpoint_artifact_id`, decision_id que lo motivó); `proposal_id` único propagado por TODOS los eventos de un ciclo de mutación y por `LineageEntry`; procedencia de restauración. El linaje μₜ se vuelve grafo auditable hasta el génesis.
- **M4 — Auditoría reproducible.** `runtime/audit/` (puro, cero deps): `CausalChainReconstructor` que materializa un `audit_report` content-addressable con la cadena completa + verificación de hashes + veredicto de completitud; CLI `scripts/audit_run.py`. Detecta corrupción y huecos con el ID exacto.
- **M5 — Validación gated de payloads en el borde del ledger** (`RNFE_EVENT_VALIDATION=off/warn/strict`, validador stdlib con subset de JSON Schema).

---

## 10. Eje 8 — Validación

- **M1 — Validador de contratos en runtime** (stdlib, subset JSON Schema) enganchado a los puntos de escritura de la facade, con modos off/shadow/enforce; endurecer schemas (enums, additionalProperties:false donde aplique); crear los 4 schemas faltantes (`session_bridge` productores, `episode_export`, `telemetry_bridge`, `safety_policy`); test de paridad dataclass↔schema que detiene el drift.
- **M2 — Gate de realidad en el bucle vivo** (ya en Fase 0).
- **M3 — Promoción de veredictos sombra al certificado.** `RNFE_CERT_STRICT=off/compare/enforce`: en compare se persisten las divergencias (`certification.verdict.divergence`) entre el verdict de umbrales fijos y el compuesto (S-I-E: RECHAZAR⇒rejected, BUFFER⇒sin promoción; IoC* = ioc − λΩ·Ω); en enforce gobierna el compuesto. Invariantes constitucionales volcados al certificado.
- **M4 — Harness de certificación obligatorio pre-herencia.** `scripts/certification_gate.py`: contratos+fronteras → subset rápido de pytest (~21 s) → corrida seedeada del kernel evaluada contra `GATE_PROFILES['ci']` → verificación byte-idéntica con flags default. Verdict como artifact por commit; `check_inheritance_eligibility` lo consulta. Reemplaza y archiva el `lab/validation` roto. <10 min en CPU.
- **M5 — Registro de veredictos de benchmark.** `benchmark_verdict.schema.json` + `evidence_registry`: las campañas persisten dictámenes consultables por (campaign_id, commit); la admisión del externo y las activaciones gated consumen el registro en vez de constantes hardcodeadas duplicadas.

---

## 11. Eje 9 — Degradación inteligente

**Diagnóstico.** Vocabulario sin mecanismo: `allow/degrade/block` existe, pero los umbrales nunca disparan (presión≈0), Postgres caído mata el arranque (con el `.env` real del repo), y ningún modo degradado está certificado. El `degradation.py` del owner (plan de degradación por evidencia en META) es la semilla del nivel táctico de este eje.

**Mejoras en orden:**

- **M1 — `ResourceProfile` al arranque** (compartido con Fase 0.5): probes de storage/GPU/externo/artifact_root con razones; evento `resource.profile.detected`; diff al restaurar en otro host.
- **M2 — Storage resiliente.** Postgres caído ⇒ degradación gobernada a SQLite (evento `storage.degraded`, ventana de divergencia registrada) y reconciliación al volver (`storage.recovered`, idempotente). *Es el único fallo de recurso que hoy es muerte en vez de degradación.*
- **M3 — Niveles de servicio explícitos.** `ServiceLevel` ∈ {FULL, DEGRADED_LOCAL, MINIMAL, SURVIVAL} calculado determinísticamente desde perfil+vitals, mapeado a knobs existentes (closure_profile, tier máximo, msrc, externo, retrieval, EML); en SURVIVAL solo act/sleep/quarantine. Política declarativa `DEGRADATION_POLICY_v1`. Integrar aquí el plan de degradación del owner como fuente táctica por episodio.
- **M4 — `resource_pressure` real y gates dormidos activados** (sleep efectivo — el kernel realmente no corre episodio y consolida memoria §5.M2 —, límites del router, B_safe con telemetría real en sombra).
- **M5 — Externo como capacidad degradable.** Disponibilidad declarada al arranque, tier_3 condicionado, `expected_failure_mode=conflict_unresolved_no_external` en certificados del régimen de conflicto sin recurso, revalidación automática al volver.
- **M6 — Certificación de modos degradados.** Campaña niveles×escenarios con seeds y bootstrap: cada nivel queda CERTIFICADO/NO_CERTIFICADO con métricas de cierre/IoC/continuidad; la política rechaza transicionar a un nivel no certificado. *Un modo degradado solo es confiable si se midió.*

---

## 12. Eje 10 — Autonomía ilimitada por política

**Principio.** La autonomía del organismo no se limita por hardcodes sino exclusivamente por políticas declarativas auditadas: ampliar autonomía = cambiar política por el pipeline gobernado, jamás editar código. La contención es la política de máxima precedencia.

**Mejoras en orden (M1→M2→M3→M5→M4→M6):**

- **M1 — Motor de políticas declarativo.** `runtime/governance/PolicyEngine` carga `governance/policies/AUTONOMY_POLICY_v1.yaml` (validado contra `safety_policy.schema.json`, hash materializado como artifact, evento `governance.policy.loaded`): umbrales del supervisor, acciones permitidas/críticas/prohibidas, tier máximo, listas de aprobación humana, TTLs. Sin documento ⇒ defaults actuales byte-idénticos. El hash de la política vigente viaja en el checkpoint (la identidad incluye bajo qué ley vivió).
- **M2 — Kill-switch y contención con precedencia 0.** `ControlChannel` (órdenes `shutdown/quarantine/freeze` por ledger + archivo centinela `KILL` sin DB) + CLI `scripts/organism_control.py` + manejo de SIGTERM con checkpoint final. **Ninguna** compensación, validador o política puede degradar o revertir una orden de contención externa; las cláusulas de contención son inmutables en el propio documento de política.
- **M3 — Aprobación humana solo cuando la política lo exige** (canal de §7.M5, poblado desde la política, no desde código).
- **M5 — Capacidades auditadas.** Los flags `RNFE_*` dejan de ser política de facto dispersa: el PolicyEngine resuelve capability = documento ⊕ override de entorno y persiste el mapa {declared, effective, source} en `governance.policy.loaded` y el checkpoint; overrides contradictorios emiten `governance.capability.override`.
- **M4 — Ampliación de autonomía por cambio de política auditado.** `PolicyChangeProposal` (con los campos del roadmap §5B) → validación contra meta-política inmutable (contención, invariantes duros, deltas máximos) → sandbox → gate → apply → monitor → commit/revert — el mismo lazo ρₜ ya probado, aplicado a la política. Ejecutable por humano o por el organismo; toda ampliación queda como artifact + evento con ambos hashes.
- **M6 — Cerrar el bypass.** La autoevolución del organismo queda subordinada al mismo gate: sin directiva del supervisor y sin `operational.conjunction.evaluated(task_type=self_modification)` previo, el controlador propone pero no aplica. (Coordinar con §7.M4 — son la misma obra vista de dos ejes.)

---

## 13. El ecosistema neuronal (plan del owner, mejorado y anclado)

El stack propuesto (MoE frontier, LLM local, SSM/Mamba, LTC, GNN, KAN, PINO/world model, EDL, SAE, NCA) es direccionalmente correcto. Las mejoras de este plano son cuatro:

1. **Anclar cada red a un órgano existente** — el repo ya tiene el lugar exacto para 9 de las 10; ninguna necesita un sistema paralelo.
2. **Reordenar por dependencia real:** ninguna red entra antes de que los tiers sean ejecutables y el sensado de recursos exista (Olas 0–1); un modelo enchufado a un router decorativo es teatro.
3. **Someterlas al canon:** todas entran por Capa D (experimental) con hipótesis falsable, en sombra, advisorias (regla 0.6), con incertidumbre y fallback (regla 0.7), y presupuesto A9.
4. **Reconocer lo ya construido:** el repo YA tiene el LLM local gobernado (OpenThinker3-7B + triple gate), el extractor de leyes (EML-SR), el world-model v0 (SCM+contrafactual §4) y la semilla del EDL (EpistemicAssessor §6.M2). Varias "redes nuevas" son evoluciones de órganos existentes, no trasplantes.

### Tabla maestra

| Red | Función cognitiva | Órgano de anclaje real | Tier | Ola | Hipótesis falsable |
|---|---|---|---|---|---|
| **B. LLM local 7B** (existe) | razonador caro auditado | `external_models` + llama.cpp | 3 | 1 | residencia: latencia 2ª llamada <20 % de la 1ª |
| **G0. Assessor epistémico** (determinista) | saber cuándo no sabe | `scheduler_meta` §6.M2 | 0 | 1 | transición a consult_external ≤3 episodios ante firma contradicha |
| **Embeddings hashed** (stdlib) | similaridad estructural | `mfm_lite` §5.M4 | 0 | 2 | divergencia de ranking medida en sombra, <1 ms |
| **Embedder GGUF** | recuperación semántica | `memory/embeddings` + RAG §5.M5 | 1 | 2 | recall de evidencia > búsqueda exacta en corpus sembrado |
| **F. KAN** | leyes interpretables de viabilidad/riesgo | `runtime/symbolic/eml` (compite con EML-SR) | 2 (offline) | 2–3 | predice `viability_next` mejor que EML-SR y lineal, con fórmula inspeccionable |
| **C. SSM tiny (Mamba-mini/GRU)** | memoria temporal, predicción de deriva | vitals + ledger → `VitalsForecaster` | 2 | 3 | predice margen/colapso a k pasos mejor que persistencia y reglas |
| **D. LTC/CfC** | homeostasis continua | sombra del `AutonomySupervisor` | 2 | 3 | detecta degradación antes que los umbrales, con menos cuarentenas falsas |
| **G1. Cabezas EDL** | incertidumbre de cada modelo | adjuntas a C/D/E/H | — | 3 | calibración: cobertura empírica ≈ nominal por régimen |
| **A. Frontier/MoE consultor** | corteza superior bajo política | tier_3 extendido + política + aprobación | 3 | 4 | resuelve conflictos que el 7B local no, a coste aceptado por política |
| **E. GNN/Graph Transformer** | corteza relacional/causal | SMG + `runtime/causal` calibrado | 2 | 4 | predice efecto de intervenciones no vistas mejor que la calibración tabular |
| **H. World model neural / PINO** | imaginación operacional | sucesor del SCM §4.M3 cuando el mundo crezca | 2 | 4–5 | `counterfactual_fidelity` ≥ SCM en escenarios sin oráculo |
| **I. SAE** | introspección de representaciones | sobre embeddings §5.M4, luego sobre C/D | — | 4 | features que anticipan contaminación/deriva de memoria |
| **J. NCA** | morfogénesis experimental | `lab/` puro (A6) | — | 5 | solo investigación; sin rol en el núcleo |

### Notas de diseño por red (lo que cambia respecto a tu propuesta)

**A. Frontier/MoE — dos niveles, no uno.** El "modelo frontier" del organismo ya existe y es local: OpenThinker3-7B con triple gate, schema y guard — mantenerlo como tier_3 por defecto. El consultor de frontera verdadero (API MoE/Claude) entra después y solo como **segundo escalón del tier_3**, gobernado por: política declarativa (§12.M1: presupuesto propio, `human_approval_required` inicialmente), salida validada por el mismo schema+guard, resultado en cuarentena como `EvidenceItem` advisorio (10.1.5: jamás escribe en la memoria viva), y presupuesto A9 debitado del `BudgetLedger`. Mejora barata previa: probar un **MoE cuantizado local** (GGUF) como swap del 7B denso — misma interfaz llama.cpp, cero cambio de contrato, experimento falsable de calidad/latencia por VRAM.

**B. LLM local — ya está; hacerlo económico.** Las tres mejoras son §3.M5 (residencia llama-server, caché validada por schema, presupuesto adaptativo) + §6.M1 (que por fin pueda correr dentro de META). No introducir un segundo modelo local hasta que el registro de evidencia (§10.M5) muestre un régimen donde el 7B falla.

**C. SSM/Mamba — no cablear los engines grandes; entrenar uno diminuto.** `engines/mamba_vendor` es GPU-only (imports CUDA al tope, bf16, 100M–2B params): inviable e innecesario como memoria del organismo. El uso correcto del principio SSM aquí es un **VitalsForecaster diminuto** (<5 M params; Mamba-mini con la referencia `ssd_minimal.py` en torch-CPU, o GRU como fallback aún más simple) entrenado **offline en scripts/** sobre las secuencias del ledger (vitals, decisiones, recompensas — dataset extraíble gracias a §9.M2), que en runtime se sirve como tabla/ONNX-CPU y emite una señal sombra `predicted_viability_next` + riesgo de colapso a k pasos en los vitals. Cierra además la brecha auditada "viabilidad retrospectiva, no prospectiva". Los engines H-Net/Mamba quedan en cuarentena lazy (§3.M6) hasta que exista un caso de uso con GPU real; la decisión de su destino es un ADR, no un default.

**D. LTC/Liquid — sombra del supervisor, nunca su reemplazo.** El supervisor por reglas es deliberadamente auditable; la política es ley (§12). El LTC/CfC (~10⁴ params, CPU trivial) corre en paralelo proponiendo modo (`normal/conservative/recovery/quarantine`) y su propuesta entra como **una señal más** en vitals (como `conjunction_health`), no como decisor. Se promueve solo si la campaña muestra detección más temprana con menos falsos positivos; incluso entonces, decide el supervisor bajo política.

**E. GNN — después del sustrato causal, no antes.** Sin `runtime/causal` calibrado (§4.M1–M2) y sin SMG consultable (hoy SMGMin solo agrega y snapshotea — darle retrieval es prerequisito), una GNN no tiene grafo del que aprender. En los mundos sintéticos actuales la calibración tabular probablemente capture todo el valor; la GNN se justifica cuando el grafo crezca (mundos ricos, transferencia cross-escenario) y se mide contra esa calibración como baseline. Score de salida = soporte causal cuantitativo para el validador de la conjunción, con incertidumbre (G1).

**F. KAN — competir con el matemático interno que ya existe.** El repo ya tiene extractor de leyes: EML-SR (regresión simbólica acotada, en sombra tras doble flag, con scoring composite fit/estabilidad/dominio). La KAN entra en **el mismo harness** con la misma tarea (aprender `viability_next = f(resource_pressure, memory_purity, risk_score, recovery_debt)` y superficies de riesgo) y gana su lugar solo si supera a EML-SR y a un baseline lineal en ajuste + parsimonia + estabilidad cross-seed. La fórmula ganadora se propone (vía ADR) como candidata a kernel de viabilidad **prospectivo** — hoy inexistente según la auditoría del canon f2.1–f2.4. Entrenamiento offline en scripts/; el runtime solo evalúa la fórmula extraída (Python puro).

**G. EDL — dos etapas.** G0 es determinista y va primero (§6.M2): intervalos Agresti-Coull, conflictos estructurales, contradicciones SMG — cero redes, valor inmediato. G1 son cabezas de incertidumbre sobre **cada** modelo neuronal que entre (C, D, E, H): regla 0.7 — sin cabeza de incertidumbre no hay sombra. El `confidence_state` resultante alimenta las compensaciones existentes de la conjunción (que ya modelan `uncertainty` y `missing_evidence`).

**H. World model / PINO — ya empezó, y no es neural todavía.** El modelo interno de transición del organismo es el SCM + contrafactual de Pearl (§4.M1/M3), interpretable y con `counterfactual_fidelity` medida contra el oráculo. Un world model **neural** (MLP pequeño / neural operator) se justifica solo cuando los mundos dejen de ser ODEs cerradas (grid ≥5×5 en serio, entrada sensorial real — ambas son brechas declaradas del núcleo); su métrica de admisión ya existirá (fidelity). PINO en particular queda condicionado a que aparezcan escenarios con física PDE real; no antes.

**I. SAE — cuando haya activaciones que inspeccionar.** Primer objetivo útil: el espacio de embeddings de memoria (§5.M4) — features dispersas que anticipen contaminación/deriva y alimenten `memory_purity`. Después, las activaciones de C/D. Lab puro, Ola 4.

**J. NCA — laboratorio, como vos mismo dijiste.** Ruta A6 (morfogénesis tipada) en `lab/`, sin rol en el núcleo. Se revisita solo si algún experimento produce una señal de aceptación clara (canon §15).

### Lo que NO hacer (confirmando y extendiendo tu lista)

- No entrenar desde cero nada grande; no fine-tuning antes de trazabilidad completa (Fase 0) y registro de evidencia (§10.M5).
- No cablear H-Net/Mamba grandes al organismo; no MoE propio entrenado.
- No agentes ejecutores nuevos antes del registro de agentes (§7.M1) — y nunca como sustituto del núcleo (10.1.7).
- No reemplazar el supervisor, el META ni el gate por modelos: los modelos proponen, la política decide.
- No APIs cloud por defecto: opt-in por política, con presupuesto y aprobación.
- No NCA/PINO en el núcleo; no SAE antes de que existan representaciones densas reales.

---

## 14. Arquitectura funcional integrada (pipeline objetivo)

El flujo que propusiste, mapeado a los módulos reales:

```text
Ciclo vital (LifeKernel.step)
  ├─ ControlChannel: órdenes externas (kill/quarantine/freeze)      [§12.M2 — precedencia 0]
  ├─ ResourceProfile + HostResourceSampler → vitals reales          [§11.M1, §3.M1]
  ├─ VitalsForecaster (SSM tiny) + LTC → señales sombra en vitals   [§13.C, §13.D]
  ├─ AutonomySupervisor decide bajo AUTONOMY_POLICY (PolicyEngine)   [§12.M1]
  └─ Conjunción operacional evalúa la decisión:
       evidencia (memoria RAG + KnowledgeIndex canon + checkpoints)  [§5.M4-M5]
       soporte causal cuantitativo (SCM calibrado, atestación)       [§4.M1-M2]
       ComputeRouter → tier EJECUTABLE según ServiceLevel + budget   [§3.M2, §11.M3, §16.I-1]
       validadores (schema, evidencia, causal, riesgo, agente,
                    reversibilidad, política)                        [§10.M1, §8.M5]
       compensaciones ejecutables → segunda pasada                   [existente]
       saga journal: efecto declarado antes de mutar                 [§8.M1]
  └─ Episodio cognitivo (ScenarioEpisodeRunner, contrato intocable):
       SMG → LOT-F → memoria (híbrida) → intervención → contrafactual (oráculo o SCM)
       → META con EngineInput/BudgetLedger + EpistemicAssessor       [§6.M2-M3]
         → familias core + críticos reales + escalera de conflictos  [§6.M4-M5]
         → tier_3: LLM local residente → (2º escalón) frontier/MoE   [§13.A-B]
       → certificación (S-I-E/IoC* en compare→enforce) + reality gate[§10.M2-M3]
       → memoria micro/meso/macro con metadata canónica + TTL        [§5.M1-M3]
       → atribución causal por ablación → reward ν → selector A2     [§4.M4, §6.M6]
  └─ Trazabilidad: CausalContext en todo evento; auditoría por IDs   [§9]
  └─ Checkpoint encadenado con policy_hash, saga, perfil de hardware [§9.M3, §12.M1]
```

---

## 15. Secuencia maestra por olas (con gates de salida)

| Ola | Contenido | Gate de salida (falsable) |
|---|---|---|
| **0. Cimientos** | §2 completo (merge, storage P0, CausalContext, ledger indexado, sensado, reality gate, WIP consolidado, SSOT actualizado) | criterio de salida de §2 |
| **1. Gobernanza ejecutable** | tiers ejecutables (§3.M2), PolicyEngine + kill-switch + aprobación (§12.M1–M3), BudgetLedger único (I-1), niveles de servicio (§11.M3), saga journal + rollback certificado (§8.M1–M2), validador de contratos sombra (§10.M1), registro de agentes (§7.M1) | `certification_gate.py` verde; tier_0 vs tier_2 producen conducta observablemente distinta; kill-switch ≤1 ciclo; nominal byte-idéntico |
| **2. Ciclo cognitivo completo** | externo residente + portado a META (§3.M5, §6.M1), EpistemicAssessor (§6.M2), memoria cross-run + consolidación + TTL + retracción (§5.M1–M3, §8.M3), SCM + calibración (§4.M1–M2), atribución (§4.M4), storage resiliente (§11.M2) | campaña A2 ejecutada bajo R1 con verdict en el registro; conflicto resuelto en runtime vía gate; memoria sobrevive reinicio |
| **3. Primeras redes (sombra)** | embeddings + RAG (§5.M4–M5), KAN vs EML-SR (§13.F), VitalsForecaster (§13.C), LTC sombra (§13.D), cabezas EDL (§13.G1), contrafactual de modelo (§4.M3), modos degradados certificados (§11.M6) | cada modelo con REPORT + verdict vs su baseline determinista; cero cambios de conducta nominal |
| **4. Frontera** | GNN causal (§13.E), frontier/MoE consultor (§13.A), SAE (§13.I), world model neural si el mundo creció (§13.H), adapter layer de exocortex (O2 del ADR_OPENCLAW) si el producto lo exige | promociones individuales por ADR con evidencia del registro |
| **5. Investigación** | NCA, PINO, evolución neuronal (rescates de `runtime/evolution` legacy vía PROMOTE/REWRITE) | solo `lab/`, cláusula F |

Regla de avance: una ola no abre hasta que la anterior pasa su gate **y** el `certification_gate` del commit está verde. Dentro de una ola, las mejoras marcadas de ejes distintos pueden paralelizarse (no comparten archivos calientes salvo lo anotado en §16).

---

## 16. Decisiones de integración transversales (crítica de coherencia)

Estas decisiones resuelven los choques detectados entre ejes; Opus debe respetarlas:

- **I-1. Un solo `BudgetLedger`.** Tres ejes proponen contabilidad (§3.M2 ciclo, §6.M3 razonamiento, §7.M3 agente). Se implementa **una** pieza en `runtime/agents/budget.py` con tres vistas: débito por ciclo, por episodio/familia y por agente. Prohibido duplicar contadores.
- **I-2. Tiers ≠ MSRC.** El tier (conjunción) es la palanca de *cuánto razonar*; MSRC es *a qué escala representar*. No se fusionan; ambos leen el mismo `ResourceProfile`/sampler. (Espejo de la decisión ADR_CONTINUITY de no fusionar métricas.)
- **I-3. Retracción propaga a índices.** `retract_memory_records` (§8.M3) excluye en SQL y dispara re-index del vectorial y el KnowledgeIndex (§5.M4–M5).
- **I-4. La activación A2 usa atribución real.** La campaña de §6.M6 mide con la ablación de §4.M4, no con el proxy multiplicativo.
- **I-5. §7.M4 y §12.M6 son la misma obra** (mandato de automodificación): un solo diseño, dos criterios de aceptación.
- **I-6. El plan de degradación del owner** (`degradation.py`) es el nivel táctico (por episodio, dentro de META) del `ServiceLevel` estratégico (§11.M3): el segundo acota al primero; misma taxonomía de niveles.
- **I-7. Exocortex sigue pospuesto** (ADR_OPENCLAW §12): primero núcleo, luego adapter layer por contratos, recién entonces shell multicanal. Los dashboards existentes se actualizan al vocabulario de eventos vivo cuando se toquen, no antes.
- **I-8. Entorno físico.** La normativa asume WSL2 + `/mnt/d`; el entorno real es Linux nativo. El `HardwareProfile` (§3.M4) + un ADR corto actualizan `POLICY_ARTIFACT_PLANE` y el supuesto del canon §5.1.4 sin cambiar el espíritu (hardware modesto).
- **I-9. `aeon_event_log.db` sale de la raíz versionable:** DB por corrida bajo `RNFE_ARTIFACT_ROOT`, default portátil, `.gitignore` ya lo cubre — evita contaminación entre campañas (hallazgo repetido del mapa).

---

## 17. Protocolo de trabajo y definición de terminado

Para cada mejora, Opus entrega en una PR:

1. **ADR** si es estructural (plantilla: contexto → decisión → hipótesis falsable → costo en hardware objetivo → rollback).
2. **Contratos** (dataclass + schema + eventos con CausalContext) antes que lógica.
3. **Tests**: unitarios + el test de paridad nominal byte-idéntica (patrón `tests/comparison`) + regresión del subsistema tocado. La suite completa debe seguir verde en CPU pura.
4. **Verificación empírica**: si la mejora cambia conducta bajo flag, una corrida seedeada documentada (smoke o campaña según alcance) con artefactos en `data/reports/`.
5. **Actualización documental mínima**: el doc de análisis del módulo tocado (`docs/analysis/NN_*.md`) anota el cambio; el índice no se deja desincronizado.

Definición de terminado de una mejora: criterios de aceptación cumplidos y verificados, flag off por defecto, `certification_gate` verde, evidencia archivada, y ninguna regla de §0 violada.

---

*Este documento es Capa D: puede tensionar capas superiores solo vía ADR y ruta de promoción explícita (canon §1.2). La regla final del canon aplica también aquí: proteger lo constitutivo, flexibilizar lo arquitectónico, y exigir falsación a toda novedad de frontera.*
