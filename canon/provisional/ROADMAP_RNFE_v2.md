---
title: ROADMAP_RNFE_v2
status: provisional
version: 3.0.0
date: 2026-07-10
owner: Wis
depends_on:
  - RNFE_canon_matematico_f2_4_v3_0.md
  - CANON_RNFE_v3_2_rc1.md
  - SSOT_RAZONAMIENTOS_RNFE_v1.md
supersedes:
  - RNFE_manual_construccion_por_etapas_v1.md (orden fino)
  - RNFE_matriz_etapas_dependencias_v1.csv (una vez migrada)
reissue:
  - "v3.0.0 (2026-07-10, A20): re-emisión in-place que fusiona TRES planes en UNO solo activo
    -- reparación P0-P30, Fase 0/WS0->WS7 (docs/strategy/2026-07-05_orientaciones_fase0_para_codex.md)
    y campaña neural N0->N6 (Codex). Filename preservado para no romper referrers (RUNTIME_SSOT_v1.md).
    Ley de fondo: cúspide RNFE_canon_matematico_f2_4_v3_0.md (tag canon-apex-v3.0)."
  - "A19: PMV oficial declarado = thermal_homeostasis (2A). A17 fija los escenarios/morfismos reales
    en SCENARIO_CONTRACTS_v1.md. Fase vigente = PENDIENTE-HUMANO (0.2)."
---

# ROADMAP RNFE v2

## Principio rector — un organismo integral y simbiótico

RNFE es UN organismo. Este ROADMAP fusiona tres planes (reparación P0–P30, Fase 0/WS0→WS7, campaña neural N0→N6) **no como pistas paralelas con muros, sino como funciones simbióticas de un mismo cuerpo**, integradas con sinergia. La partición por zona (§18.2) es división del trabajo dentro del organismo; las dependencias "sustrato antes que órgano" (§18.3) **son la simbiosis**: el sustrato nutre al órgano y el órgano le da propósito al sustrato. Es un **principio compartido** — la misma orientación gobierna a Codex (campaña neural) y al orquestador de reparación. Fundamento axiomático: **A-M0** de `RNFE_canon_matematico_f2_4_v3_0.md`. Regla derivada: **ninguna optimización local de una pista es válida si rompe la sinergia del todo.**

## 0. Propósito

Este roadmap convierte el canon vigente en secuencia de construcción. No enumera deseos; define el orden exacto de ejecución, los gates técnicos, criterios de salida, riesgos y dependencias duras para construir un ser cibernético de próxima generación en condiciones físicas restringidas.

## 0.1 Re-emisión v3.0 — un solo plan activo (fusión de tres)

> **Esta re-emisión (v3.0.0, 2026-07-10) fusiona TRES planes en UN solo plan activo**, sin derogar ninguno:
>
> - **(a) Reparación P0-P30** — el backlog de reparación (`adjudicacion.md` A1-A20 + B1-B48), ordenado por dependencia y radio-SCC.
> - **(b) Fase 0 / WS0->WS7** — los cimientos ejecutables (`docs/strategy/2026-07-05_orientaciones_fase0_para_codex.md`), capa D/laboratorio.
> - **(c) Campaña neural N0->N6** — el track de crecimiento del cómputo neuronal, dueño de `runtime/neural/` (Codex).

El corazón de la fusión es la **tabla de correspondencia de tres columnas `WS <-> paquete-P <-> paquete-N`** (§17) y la **disciplina del track de crecimiento neural** (§18). Ningún plan reemplaza a otro: los tres se leen contra la cúspide ratificada `RNFE_canon_matematico_f2_4_v3_0.md` (tag `canon-apex-v3.0`), que es la ley. El orden canónico de cada pista sigue viviendo en su fuente (P0->P30 en el backlog; WS0->WS7 en las orientaciones; N0->N6 en la campaña); esta tabla solo fija los cruces entre pistas, no reordena ninguna.

## 0.2 FASE VIGENTE (campo de gobierno — lo fija el humano)

**`FASE_VIGENTE: 1`** — Fase 1 (Infraestructura basal y observabilidad, §4). **CONFIRMADO por el humano (2026-07-10).** Criterio aplicado: la fase más alta cuyos prerequisitos ya están reparados, no la que se aspira. Fase 0 (gobernanza/congelación) queda cumplida con `canon-apex-v3.0` (canon congelado, SSOT/contratos/aliases establecidos); la reparación foundacional en curso es infraestructura basal. Lo central: mantiene viva la prohibición de autoevolución de A5 ("no antes de cerrar Fase 5"), que se **ganará con P29/A5** (gate por veredicto + sandbox que simula), no por decreto.

Esta declaración es la que **desbloquea la adjudicación limpia de A5/J5**: la prohibición de autoevolución legítima está indexada a fase (§13, "no antes de cerrar Fase 5"); sin fase vigente declarada, esa prohibición no tiene contra qué adjudicarse y A5 queda cojo (A20). Fijar el número aquí es prerequisito doctrinal de la corrección de A5.

## 1. Restricciones reales de diseño

### 1.1 Restricciones físicas iniciales

- Windows 11 + WSL2
- GPU base objetivo: 8 GB VRAM
- trabajo individual
- necesidad de trazabilidad, reproducibilidad y rollback

### 1.2 Restricciones estratégicas

- no sacrificar núcleo por superficie de producto;
- no sacrificar medición por retórica;
- no abrir autoevolución sin certificados;
- no inflar multiagente antes de nacimiento cognitivo mínimo.

## 2. Arquitectura de entrega por fases

El roadmap se divide en cinco macrofases:

1. **Gobernanza y congelación**
2. **Nacimiento cognitivo mínimo**
3. **Estabilidad y continuidad**
4. **Ecología de razón y evolución controlada**
5. **Exocorteza operativa y producto**

## 3. Fase 0 — Congelación del proyecto

### Objetivo

Detener deriva documental y dejar el proyecto en estado gobernable.

### Entregables obligatorios

- `CANON_RNFE_v3_1.md`
- `SSOT_RAZONAMIENTOS_RNFE_v1.md`
- `ADR_OPENCLAW_ACOPLAMIENTO.md`
- estructura oficial de carpetas
- contratos base: episodio, propuesta, certificado, rollback, telemetry snapshot
- tabla de aliases históricos

### Criterio de salida

- existe una sola fuente por cada decisión estructural;
- ningún archivo histórico puede alterar el tronco por ambigüedad;
- los alias `RAZ`, `CSTR/OPT`, `EVO`, `HNet/Hctrl` ya están normalizados.

### Riesgo

Pseudoavance documental sin capacidad ejecutable.

## 4. Fase 1 — Infraestructura basal y observabilidad

### Objetivo

Construir el esqueleto técnico del organismo.

### Subetapas

#### 1A. Entorno reproducible

- lock de entorno;
- CLI base;
- runner;
- estructura de logs;
- directorio de artefactos por experimento.

#### 1B. Telemetría primaria

- VRAM;
- temperatura;
- latencia;
- espectro/condición;
- estabilidad numérica;
- señal de borde;
- carga por módulo.

#### 1C. Barreras y conjunto seguro

- projector QP;
- guardas duras;
- thresholds p95/p99;
- kill-switch de emergencia.

### Criterio de salida

- cada corrida deja artefacto auditable;
- todas las señales críticas están medidas;
- toda acción efectiva pasa por guardas de seguridad.

## 5. Fase 2 — Nacimiento cognitivo mínimo

Esta es la frontera entre teoría y existencia.

### 2A. PMV oficial

**PMV oficial declarado (A19): `thermal_homeostasis`.**

El ROADMAP prometía elegir el PMV entre dos candidatos; el PMV realmente construido y vivo es `thermal_homeostasis` (`runtime/world/thermal_scenario.py`), que no es ninguno de los dos. Esta re-emisión salda la inconsistencia declarándolo PMV oficial: variable principal `temperature`, lógica `TEMP_HIGH -> ACTIVATE_COOLING`, semántica de mejora "menor temperatura es mejor". Su contrato formal vive en `SCENARIO_CONTRACTS_v1.md §6.1`; el registro de los otros tres escenarios reales (`resource_management` canónico; `grid_thermal_5x5` y `deferred_load_trap` extra-canon) está en ese mismo documento (A17).

Registro histórico (candidatos originales, ninguno construido):
1. mini-mundo semiótico fractal
2. F1-IND / boxworld lógico-causal

### 2B. SMG mínimo

Objetivo: signos internos persistentes, relaciones de soporte/contradicción y anti-deriva.

### 2C. LOT-F mínimo

Objetivo: gramática mínima, parser, tipos, reglas, checker y traducción desde signos.

### 2D. C-GWM mínimo

Objetivo: factual vs contrafactual, intervenciones y world model causal mínimo.

### 2E. Cierre F–M–S

Objetivo: cerrar por primera vez el ciclo `observación → signo → formalización → acción/intervención → actualización de signo`.

### 2F. IoC proxy + certificado ampliado mínimo

Antes de herencia fuerte, debe existir una primera implementación de:

- `IoC*` proxy;
- continuidad identitaria;
- obstrucción global mínima;
- certificado ampliado mínimo;
- criterios de aceptación de episodio.

### Criterio de salida de la fase 2

RNFE solo sale de fase 2 si demuestra:

1. signos estables y útiles;
2. formalización trazable;
3. causalidad mínima operativa;
4. cierre triádico medible;
5. certificado episódico válido.

## 6. Fase 3 — Memoria, continuidad y homeostasis

### 3A. OMG + memoria episódica

- snapshots;
- hashes;
- árboles de episodios;
- versionado de signos y world states.

### 3B. MFM/VFD productivos

- memoria micro/meso/macro;
- no-interferencia;
- TTL/fallback;
- histéresis anti-flapping;
- ruteo bajo congestión.

### 3C. Hctrl/MRO + Edge

- regímenes crucero/análisis/emergencia;
- dwell-time;
- control de budgets;
- control de disipación;
- mantenimiento de viabilidad.

### Criterio de salida

- continuidad restaurable;
- rollback atómico probado;
- estabilidad bajo ruido, carga y limitación física.

## 7. Fase 4 — Ecología de razón

La antigua “Etapa 8” se redefine aquí en tres bloques.

### 4A. SSOT de familias de razonamiento

Debe existir mapping completo entre ontología, runtime y metagobierno.

### 4B. Scheduler con economía de razón

El scheduler decide:

- familia;
- presupuesto;
- secuencia;
- acceptance tests;
- fallback.

No elige por sofisticación nominal, sino por valor esperado de cierre ajustado por costo y riesgo.

### 4C. Engines operativos

Motores mínimos a integrar:

- DED
- IND
- ABD
- ANA
- CAU
- CTF
- PROB
- PLAN
- OPT
- EVO/SEARCH
- NESY
- DIA/ADV
- HEUR
- FAL-GUARD

### 4D. Acceptance suite

Cada familia debe tener:

- contrato de entrada/salida;
- trazabilidad;
- costo máximo;
- criterios de éxito;
- pruebas OOD;
- cross-check adversarial.

### Criterio de salida

- el organismo selecciona la familia correcta según contexto;
- supera a un razonador único en el PMV;
- puede explicar qué familia activó y por qué.

## 8. Fase 5 — Agentes y evolución controlada

### 5A. Agentes mínimos legitimados

1. agente de hiperparámetros
2. agente de rigidez
3. agente de imaginación
4. evaluación cruzada
5. ADC-PRIME

### 5B. Régimen de propuesta

Todo cambio se formaliza como propuesta con:

- hipótesis;
- costo estimado;
- riesgo estimado;
- experimento de sombra;
- criterio de aceptación;
- plan de rollback.

### 5C. S-I-E fuerte

Debe implementar:

- shadow mode;
- no-regresión;
- CVaR;
- tests metamórficos;
- commit/rollback;
- quarantine/lab;
- kill-switch.

### 5D. Herencia certificada

Solo heredan mutaciones que:

- mantienen viabilidad;
- preservan continuidad identitaria;
- mejoran `IoC*` o mejoran costo sin degradar cierre;
- pasan tests metamórficos y adversariales.

### Criterio de salida

- el organismo propone, prueba, rechaza y hereda cambios sin intervención manual microgestionada;
- existe bitácora completa de propuestas aprobadas y rechazadas.

## 9. Fase 6 — Ingesta externa certificada

### Objetivo

Permitir absorción de conocimiento externo sin contaminación catastrófica.

### Reglas

- toda fuente externa entra como propuesta;
- nunca se injerta directo en el núcleo;
- toda absorción pasa por shadow mode;
- se exige evidencia de ganancia útil, no de novedad retórica.

### Criterio de salida

- al menos una mejora externa fue integrada sin regresión y con trazabilidad total.

## 10. Fase 7 — Linajes y meta-aprendizaje

### Objetivo

Pasar de aprendizaje de primer orden a selección estable de variantes cognitivas.

### Capacidades requeridas

- variantes pequeñas y controladas;
- fitness por linaje;
- selección con riesgo acotado;
- promoción solo de tipos estables;
- control de redundancia entre linajes.

### Criterio de salida

- el sistema demuestra mejora de segundo orden, no solo tuning puntual;
- existe una medida de linajes con trazabilidad histórica.

## 11. Fase 8 — Exocorteza operativa y producto

### Objetivo

Conectar el núcleo vivo a una superficie operacional sin contaminar el organismo.

### Componentes admisibles

- gateway;
- nodos de dispositivo;
- canales;
- skills;
- plugins;
- apps;
- sandboxing;
- UI de control;
- ejecución de vertical productivo.

### Restricción suprema

La exocorteza no puede redefinir la ontología del organismo.

## 12. Backlog priorizado P0 / P1 / P2

### P0 — bloqueo inmediato

1. congelar canon y SSOT;
2. elegir PMV oficial;
3. crear contratos base;
4. runner + telemetry + barriers;
5. `SMG_min`;
6. `LOTF_min`;
7. `CGWM_min`;
8. `OMG/certificados_min`.

### P1 — habilitadores mayores

1. MFM/VFD productivos;
2. Hctrl/Edge/QP;
3. scheduler con economía de razón;
4. suite mínima de reasoning engines;
5. shadow mode y S-I-E.

### P2 — expansión controlada

1. enjambre multiagente real;
2. linajes avanzados;
3. ingestión externa intensiva;
4. verticales múltiples;
5. despliegue multicanal completo.

## 13. Matriz de prohibiciones temporales

No se permite antes de cerrar Fase 2:

- swarm multiagente pleno;
- autopoiesis estructural abierta;
- ingestión externa fuerte;
- monetización seria;
- dependencia operacional de shells externos.

No se permite antes de cerrar Fase 4:

- declarar razonamiento general;
- declarar meta-razonamiento real;
- declarar inteligencia general funcional.

No se permite antes de cerrar Fase 5:

- declarar autoevolución legítima;
- permitir cambios persistentes sin certificado.

## 14. Vertical económico: regla de entrada

La elección del vertical no ocurre por intuición ni por moda. Solo se habilita cuando:

1. existe nacimiento cognitivo mínimo validado;
2. existe estabilidad suficiente;
3. existe trazabilidad y rollback;
4. puede medirse una ventaja cognitiva específica.

## 15. Definición de “nueva era” dentro de RNFE

El proyecto entra en su primera fase de “nueva era” no cuando tenga más documentos, sino cuando logre simultáneamente:

- identidad operativa;
- cierre triádico real;
- memoria viva con continuidad;
- razón plural gobernada;
- herencia certificada;
- una superficie de valor real.

## 16. Cierre ejecutivo

El orden correcto de construcción queda fijado así:

**primero organismo mínimo → luego organismo estable → luego organismo que razona mejor → luego organismo que puede heredarse → luego organismo que puede operar y monetizar.**

---

# Anexo de fusión (re-emisión v3.0)

## 17. Tabla de correspondencia `WS <-> paquete-P <-> paquete-N`

Esta tabla es el núcleo operativo de la fusión. Cruza las tres pistas; **no** reordena ninguna (cada pista conserva su orden canónico en su fuente). Zonas según el `Protocolo de coordinación campaña neural`: **reparación** dueña de kernel/gate, storage, experience, contracts, canon; **campaña N** dueña de `runtime/neural/` + adaptadores; **compartidas** `scheduler_meta` y `world` (rebase + baseline obligatorios antes de despachar).

### 17.1 Filas ancladas en la Fase 0 (WS0->WS7, el tejido conectivo)

| WS (Fase 0) | Paquete-P (reparación) | Paquete-N (campaña neural) | Zona | Nota de correspondencia |
|---|---|---|---|---|
| **WS0** — Keystone: productor de recursos reales (`HostResourceSampler`, `RNFE_HOST_SENSING`) | sin P directo (capacidad nueva de Fase 0); adyacente a **P25/B12** (telemetría al camino vivo) y a **A20** (telemetría espectral recalendarizada) | prerequisito de sensado para futuros adaptadores N (el órgano necesita sentir el cuerpo) | world/life (compartida) | Sustrato de presión de recursos; alimenta la capa consumidora ya construida. Sin P numerado. |
| **WS1** — Endurecimiento SQLite (WAL + busy_timeout) | **P3 / B38** (sub-ítem WAL/busy_timeout, C12) | — | storage (reparación) | **ANCLA:** WS1 ≈ storage B38. |
| **WS2** — Divergencia de dual-write observable + paridad `transfer_assessments` | **P6 / B3** (dual-write silencioso) + **B46** (paridad `created_at`) | — | storage (reparación) | **ANCLA:** WS2 ≈ storage B3. |
| **WS3** — Sobre de correlación causal (`CausalContext`) = **CADENA-CAUSAL** | **P-CADENA-CAUSAL** (paquete-P propio, profundo) — construye `CausalContext` + la cadena decisión->episodio->traza->certificado por IDs; **incluye B41** (`organism_id` != `run_id`: identidad estable del sujeto de la cadena, movido desde P23); coordina con **P20/B11** (ledger) y **P28/B4** (memoria->intervención) | prerequisito de **todo N**: trazabilidad por IDs para atribución causal del órgano | world (compartida) | **ANCLA:** WS3 ≈ **P-CADENA-CAUSAL** (paquete-P profundo). **B41 vive dentro de este paquete.** |
| **WS4** — Paridad e índices del ledger + `find_events` | **P20 / B11** (migración ledger core<->storage) | — | storage (reparación) | `find_events` elimina el escaneo O(n) del cierre triádico. |
| **WS5** — Reality gate dentro del bucle vivo (shadow) | coordinación con **P11** (B26/B22 regímenes+edge) y **P25/B12** (telemetría) | shadow-gate reutilizable por N (evaluación de contacto con la realidad del órgano) | world/reality (compartida) | Sin P exacto; coordinación. Shadow por defecto. |
| **WS6** — Consolidar el WIP (contratos, CLI, contexto) | **P7 / B17-B20** (contratos activos) + **P2 / B15-B16** (huérfanos) | — | contracts/reasoning (reparación) | Cierra cabos del working set de gobernanza. |
| **WS7** — Actualizar `RUNTIME_SSOT_v1` (LifeKernel + conjunción como núcleo) | **P17 / A10** (gobernanza vinculante de life+conjunction) + **P16 / A11** (constitución T5) | — | canon/normative (reparación) | WS7 y A10 coinciden en sujeto: el núcleo vivo soberano. |

### 17.2 Filas ancladas en la campaña neural (N0->N6, el track de crecimiento)

| Tramo N | Sustrato (paquete-P, reparación) | Órgano (paquete-N) | Zona | Nota de correspondencia |
|---|---|---|---|---|
| **pre-N1** (sustrato de gate) | **P23 / B39 + B48** (identidad causal del gate + invariante total de bloqueo) | **N1** — primer órgano | kernel/gate (reparación) -> `runtime/neural/` (N) | **ANCLA DURA:** B48 + B39 ⟶ prerequisito de N1 (sustrato antes que órgano). Se adelanta en la fila. |
| **pre-N3** (sustrato identidad+experiencia) | **P-CADENA-CAUSAL / B41** (`organism_id` != `run_id`) + **P21 / B42-B45** (flujo experiencia/maestro) | **N3** | experience/identity (reparación) -> `runtime/neural/` (N) | **ANCLA DURA:** B41 + B42-B45 ⟶ prerequisito de N3. Se adelanta en la fila. |
| **N3 / N5 = A9 plena** | **P30 -> coordinación** (A9 enchufe mínimo tras guard + B29 bugs del vendor) | **N3 / N5** — engines neuronales plenos (build `csrc/`+flash_attn, adopción) | reasoning (compartida) + `runtime/neural/` (N) | **ANCLA:** N3/N5 ≈ A9-engines-plena ⇒ **P30 NO ejecuta la variante plena de A9**; solo coordina el enchufe-guard y las correcciones B29. La variante plena la aterriza la campaña N. |
| **N6-EVO** | **P29 / A5** (gate por `verdict=="certified"` + sandbox que simula con `apply_fn`) | **N6-EVO** — evolución del órgano | organism (reparación) + `runtime/neural/` (N) | **ANCLA:** N6-EVO converge con P29/A5; **comparten el mecanismo del sandbox-con-`apply_fn`** (el sandbox debe simular el cambio, no evaluar el estado). |
| **N0, N2, N4** | — | **N0** (bootstrap `runtime/neural/`), **N2**, **N4** | `runtime/neural/` (N) | **PENDIENTE-CODEX** — no existe spec N0->N6 en el repo (ver §18.4); scope y correspondencia los fija Codex. |

### 17.3 Leyenda de pendientes

- **`PENDIENTE-CODEX`**: correspondencia que depende del spec de la campaña neural, que **no existe en el repo** al momento de esta re-emisión (verificado: sin spec en `docs/`, `governance/`, `canon/`; `runtime/neural/` aún no creado). N0/N2/N4 y el scope fino de N1/N3/N5/N6 quedan a cargo de Codex.
- **`PENDIENTE-HUMANO`**: la fase vigente (§0.2). Ningún automatismo la fija.
- **"sin P directo / sin P numerado exacto"**: el WS construye capacidad de Fase 0 que la reparación no cubre con un paquete numerado (la reparación no construye capacidad nueva); se anota la adyacencia temática, no una equivalencia.

## 18. La campaña neural N como TRACK DE CRECIMIENTO OFICIAL

La campaña neural N0->N6 (Codex) entra al ROADMAP como el **track de crecimiento oficial** del cómputo neuronal del organismo. No es reparación (que restaura lo prometido a lo real): es crecimiento gobernado, y por eso tiene **disciplina de promoción propia**, obligatoria y documentada aquí.

### 18.1 Disciplina de promoción por órgano

Todo órgano de la campaña N recorre, sin saltos, la escalera de estatus:

**`experimental` -> `shadow` -> `provisional`**

- **experimental**: el órgano vive en `runtime/neural/` tras flag `RNFE_*` off por defecto; con el flag off, corrida seedeada byte-idéntica a HEAD. No toca decisiones.
- **shadow**: computa y persiste en paralelo (eventos aditivos), pero **no gobierna** ninguna decisión del organismo. Se recoge evidencia comparativa.
- **provisional**: solo tras benchmark reproducible que demuestre ganancia útil (no novedad retórica) y sin regresión; recién ahí el órgano puede influir conducta bajo flag on.

Requisitos duros de cada promoción:

1. **Benchmarks reproducibles obligatorios** — artefactos versionados (`manifest.json` / `REPORT.md` / `verdict.json`), corrida seedeada, comparación contra baseline. Sin benchmark reproducible no hay promoción.
2. **Un ADR por órgano** — cada órgano de la campaña N nace con su propio ADR en `governance/adr/` (contexto -> decisión -> hipótesis falsable -> costo en hardware objetivo -> plan de rollback). No hay órgano sin ADR.
3. **Cero regresión y byte-identidad nominal** — con el flag off, el organismo decide exactamente igual que antes del órgano.

### 18.2 Partición por zona (división simbiótica, no frontera adversarial)

Bajo el Principio rector (A-M0 de la cúspide), esta partición **no es un muro entre dos proyectos**: es la división del trabajo dentro de un mismo organismo. Cada zona es un órgano con su función; la frontera existe para que cada parte haga bien lo suyo y el todo mantenga sinergia, no para aislar. La "no-intrusión" protege la especialización simbiótica; no levanta un enemigo.

- **Reparación (dueña):** kernel/gate, storage, experience/memoria-sustrato, contracts, canon.
- **Campaña N (dueña):** `runtime/neural/` y los adaptadores de órganos.
- **Zonas compartidas** (`scheduler_meta`, `world`): chequeo obligatorio antes de despachar cualquier paquete que las toque (`git log origin/main`); si Codex mergeó algo ahí, **rebase de la rama de reparación + re-correr la suite** antes de seguir.
- **No-intrusión:** nada de reparar dentro de `runtime/neural/`. Si un paquete de reparación encuentra un problema ahí, **se reporta a Codex, no se toca**.
- **Cadencia de sync:** tras cada PR de Codex mergeado a `main` -> rebase + baseline, registrado en `externa/sync_campana_neural.md`.

### 18.3 Dependencias duras — sustrato antes que órgano

El órgano no aterriza sin su sustrato. Estas dependencias son **duras** (bloqueantes) y se adelantan en la fila de reparación:

- **B48 + B39 (gate)** ⟶ prerequisito de aterrizaje de **N1**. Empaquetadas en P23. B39 = identidad causal (`decision_id`/`created_at` propagados en las ramas transformadoras); B48 = invariante total de bloqueo + separación `validation_tier`/`execution_tier`. Tocan `runtime/life/kernel.py:455-513`.
- **B41 + B42-B45 (identidad y experiencia)** ⟶ prerequisito de aterrizaje de **N3**. B41 (**P-CADENA-CAUSAL**, WS3) desacopla `organism_id` de `run_id`; B42-B45 (P21) sanean el flujo experiencia/maestro. Ortogonales pero acoplados por datos (namespace `organism_id`): vigilar la coordinación P21<->P-CADENA-CAUSAL.

### 18.4 Estado del spec de la campaña N

Al momento de esta re-emisión **no existe spec N0->N6 en el repo** (verificado por búsqueda en `docs/`, `governance/`, `canon/`; `runtime/neural/` aún no creado). La columna N de §17 se construyó **exclusivamente** con las correspondencias que fija este encargo y el `Protocolo de coordinación campaña neural`:

- N1 <- sustrato B48+B39 (pre-N1); N3 <- sustrato B41+B42-B45 (pre-N3); N3/N5 ≈ A9-engines-plena; N6-EVO ≈ P29/A5.
- **N0, N2, N4 quedan `PENDIENTE-CODEX`**, junto con el scope fino de N1/N3/N5/N6. Cuando Codex publique el spec, esta tabla se actualiza contra él.

## 19. A20 — promesas provisionales recalendarizadas (fuera del alcance de reparación)

El ROADMAP arrastraba promesas provisionales sin ningún código. La directiva de reparación **no construye capacidad nueva**; por eso estas promesas **no** se ejecutan en la fase de reparación: se marcan explícitamente fuera de alcance y se recalendarizan al track/fase que corresponda. Re-emitirlas aquí es acto doctrinal (reacomodar el canon a la realidad), no "diferir".

| Promesa provisional | Estado | Recalendarización |
|---|---|---|
| Agentes de Fase 5 (`runtime/agents` = docstring) | fuera de alcance de reparación | Fase 5 (agentes y evolución controlada), tras nacimiento cognitivo y estabilidad |
| Telemetría espectral + projector QP + guardas p95/p99 | fuera de alcance de reparación | Fase 1 (observabilidad); WS0 sienta el sustrato de sensado real |
| Histéresis anti-flapping / ruteo de memoria bajo congestión | fuera de alcance de reparación | Fase 3 (memoria/homeostasis) |

Nada de esto se archiva ni se descarta: queda como intención recalendarizada, gobernada por la fase vigente (§0.2) una vez el humano la declare.

## 20. Cierre de la re-emisión

Esta v3.0 deja el ROADMAP como **un solo plan activo** que cualquier ejecutor (reparación u órgano neural) lee contra la misma ley (`canon-apex-v3.0`). Los dos actos de gobierno que quedan pendientes de mano humana son: **(1)** fijar `FASE_VIGENTE` (§0.2), que desbloquea A5/J5; y **(2)** ratificar esta re-emisión. Hasta ambos, la fila neural N0/N2/N4 permanece `PENDIENTE-CODEX` y la adjudicación de A5 permanece a la espera de la fase.
