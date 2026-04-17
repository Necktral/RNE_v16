---
title: RUNTIME_SSOT_v1
status: normative
version: 1.0.0
date: 2026-04-17
owner: Wis
depends_on:
  - CANON_RNFE_v3_2_rc1.md
  - ROADMAP_RNFE_v2.md
  - HARDENING_ROADMAP.md
supersedes: []
---

# RUNTIME SSOT v1

## 0. Propósito

Este documento fija la **fuente única de verdad del runtime vivo** de RNFE en su estado actual.
No reemplaza el canon matemático ni el roadmap general; define qué componentes gobiernan el organismo ejecutable hoy, cuáles son baseline histórico, cuáles son experimentales y qué gates regulan aceptación, continuidad y herencia.

Su objetivo es impedir deriva entre:
- código fusionado,
- PRs recientes,
- roadmap de hardening,
- y lectura oral del estado del proyecto.

---

## 1. Estado operativo actual del organismo

### 1.1 Tesis de estado
RNFE ya no está en fase de puro astillero.
El sistema dispone de:

- infraestructura desacoplada de runtime,
- nacimiento cognitivo mínimo,
- validación de realidad,
- scheduler adaptativo,
- certificación y memoria multiescala,
- y soporte inicial para escenarios parametrizables.

### 1.2 Capacidades activas del runtime
El runtime actual soporta, al menos:

1. ejecución de episodios cognitivos cerrados;
2. persistencia de eventos, trazas y artifacts;
3. validación de realidad invocable y ejecutable en shutdown;
4. perfiles de cierre diferenciados;
5. runners legacy y multi-escenario coexistentes;
6. memoria/certificación operativas;
7. familias adaptativas controladas;
8. engine EML en modo shadow.

---

## 2. Componentes normativos del runtime

### 2.1 Núcleo de ejecución
Declarar como normativos:

- `RuntimeRunner`
- `OrchestratorLifecycle`
- `RealityValidationService`
- `RealityValidationHook`
- `MetaScheduler`
- `PromotionGate`
- `SMGMin`
- `LOTFMin`

### 2.2 Gates normativos activos
Los siguientes gates gobiernan aceptación operacional:

- gate de cierre triádico;
- gate de continuidad;
- detector de colapso;
- gate de certificación;
- gate de promoción;
- gate de realidad en perfil `ci`;
- perfil de cierre `baseline_fixed`.

### 2.3 Artefactos normativos
Todo episodio o benchmark válido debe poder dejar:

- evento persistido,
- artifact materializado,
- trazabilidad de razonamiento,
- assessment / certificate / decision según aplique.

---

## 3. Componentes baseline / legacy

### 3.1 Definición
Se consideran baseline/legacy los componentes que siguen vigentes como referencia histórica, smoke baseline o comparador de regresión, aunque no sean el camino evolutivo principal.

### 3.2 Baseline actual
- `MinimalCognitiveEpisodeRunner`
- escenario térmico mínimo legacy-equivalente
- secuencia exacta `baseline_fixed = [ABD, ANA, CAU, CTF, DED, PROB]`

### 3.3 Regla de protección
El baseline:
- no se degrada por conveniencia adaptativa;
- no se elimina sin prueba de paridad o superioridad estable;
- permanece como comparador histórico.

---

## 4. Componentes de trayectoria oficial futura

### 4.1 Camino oficial
Se considera camino operativo futuro del organismo:

- `ScenarioEpisodeRunner`
- `CognitiveScenario`
- `ThermalScenario`
- `ResourceScenario`
- perfiles de cierre multi-modo
- benchmark multi-escenario

### 4.2 Regla de transición
La trayectoria oficial futura no desplaza al baseline por mera existencia.
Su promoción exige:

1. equivalencia o superioridad en cierre;
2. continuidad no inferior a umbral definido;
3. compatibilidad de artifacts y trazas;
4. no contaminación de memoria;
5. benchmark heterogéneo exitoso.

---

## 5. Componentes experimentales

### 5.1 Experimental controlado
Se consideran experimentales:

- `adaptive_min`
- familias opcionales (`DIA_ADV`, `HEUR`, `FAL_GUARD`, `EML_SR`)
- EML shadow
- modos cross-scenario analogical
- benchmark extendido

### 5.2 Regla de experimentalidad
Todo componente experimental:
- no redefine el baseline;
- no cambia contratos normativos sin ADR;
- no altera certificación viva por defecto;
- debe poder apagarse por flag/config.

---

## 6. Contratos vigentes del runtime

### 6.1 Contrato de episodio
Un episodio válido debe contener:
- `episode_id`
- `run_id`
- `timestamp`
- `context`
- `result`
- `trace`
- `scenario` o `scenario_metadata`
- artifact asociado
- evento `episode.closed`

### 6.2 Contrato de cierre
El cierre exige:
- observación presente,
- signos mínimos,
- fórmula válida,
- factual/contrafactual coherentes,
- evento de cierre persistido,
- secuencia de razonamiento válida según perfil.

### 6.3 Contrato de reality validation
Toda corrida de benchmark de realidad válida debe producir:
- `bench_run`
- `assessments`
- `summary`
- evento `reality.validation.completed`
- artifact `reality_report`

---

## 7. Perfiles normativos de razonamiento

### 7.1 baseline_fixed
Perfil canónico, exacto, histórico e inmutable.

### 7.2 adaptive_min
Perfil ecológico controlado:
- mantiene orden parcial obligatorio;
- permite intercalación de familias opcionales;
- no deroga el baseline.

### 7.3 Regla de conflicto
Cuando baseline y adaptive discrepen:
- baseline gobierna comparabilidad histórica;
- adaptive gobierna exploración controlada;
- ninguna conclusión experimental puede borrar baseline sin ADR.

---

## 8. Política de aceptación de cambios al runtime

Un cambio al runtime solo es aceptable si:

1. no rompe el baseline sin justificación formal;
2. mantiene artifacts y trazas auditables;
3. preserva portabilidad;
4. pasa tests relevantes;
5. declara si toca baseline, trayectoria futura o experimental;
6. deja evidencia de benchmark si altera cierre, continuidad o memoria.

---

## 9. Métricas oficiales de esta etapa

### 9.1 Métricas mínimas
- closure_rate
- continuity_mean
- collapse_count
- trace_integrity
- promotion verdict
- scenario_metrics

### 9.2 Métricas aún no suficientes por sí solas
- cantidad de módulos
- cantidad de familias
- complejidad simbólica
- número de escenarios

---

## 10. Decisiones congeladas

Listar aquí decisiones duras, por ejemplo:
- `baseline_fixed` permanece exacto
- el hook de realidad forma parte del ciclo del organismo
- el runner multi-escenario no sustituye aún al legacy
- el benchmark heterogéneo es gate de siguiente fase

---

## 11. Backlog inmediato ligado al SSOT

1. introducir `scenario_metadata` en toda la cadena;
2. filtrar memoria por compatibilidad de escenario;
3. documentar transición de runners;
4. formalizar benchmark heterogéneo.

---

## 12. Definiciones operativas

- baseline
- legacy
- trayectoria oficial futura
- experimental
- gate
- colapso
- continuidad
- contaminación de memoria
- compatibilidad de escenario
