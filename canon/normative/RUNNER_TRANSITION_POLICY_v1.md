---
title: RUNNER_TRANSITION_POLICY_v1
status: normative
version: 1.0.0
date: 2026-04-17
owner: Wis
depends_on:
  - RUNTIME_SSOT_v1.md
  - SCENARIO_CONTRACTS_v1.md
  - HARDENING_ROADMAP.md
---

# Runner Transition Policy v1

## 0. Propósito

Este documento regula la coexistencia, comparación y eventual transición entre:

- `MinimalCognitiveEpisodeRunner`
- `ScenarioEpisodeRunner`

Su objetivo es evitar una migración prematura que destruya comparabilidad histórica o introduzca regresiones ocultas.

---

## 1. Estado actual

### 1.1 MinimalCognitiveEpisodeRunner
Rol:
- baseline histórico
- smoke runner canónico
- comparador duro de regresión
- referencia térmica mínima

### 1.2 ScenarioEpisodeRunner
Rol:
- sucesor en incubación
- camino oficial futuro
- soporte multi-escenario
- punto de integración con metadata y benchmark heterogéneo

---

## 2. Regla de coexistencia

Ambos runners coexisten.
La existencia del runner nuevo no implica obsolescencia del viejo.

---

## 3. Criterios de comparación

Todo cambio relevante debe poder evaluarse en ambos, cuando aplique:

- cierre
- continuidad
- artifacts
- razonamiento
- certificación
- portabilidad
- persistencia

---

## 4. Tests de paridad obligatorios

### 4.1 Paridad térmica
El runner por escenarios con `thermal_homeostasis` debe producir comportamiento equivalente al runner legacy en el baseline mínimo.

### 4.2 Paridad de trazabilidad
Ambos deben materializar:
- evento `episode.closed`
- artifact equivalente
- secuencia de razonamiento comparable

### 4.3 Paridad de gates
La validación de cierre y certificación no deben divergir sin explicación documental.

---

## 5. Divergencia permitida

Se permite divergencia solo si:
- está documentada;
- mejora métrica relevante;
- no rompe baseline histórico;
- no degrada trazabilidad;
- pasa benchmark específico.

---

## 6. Ruta de transición

### Fase A
Coexistencia pura
Legacy manda en baseline; scenario runner en incubación.

### Fase B
Paridad controlada
Tests de equivalencia por escenario térmico.

### Fase C
Pre-promoción
Scenario runner pasa benchmark heterogéneo y no degrada baseline.

### Fase D
Promoción
Scenario runner pasa a camino oficial.
Legacy queda como adapter o smoke baseline.

---

## 7. Criterios de promoción de ScenarioEpisodeRunner

1. paridad térmica satisfactoria;
2. metadata de escenario completa;
3. benchmark heterogéneo exitoso;
4. memoria filtrada correctamente;
5. no contaminación de certificación;
6. artifacts consistentes;
7. no regresión de portabilidad.

---

## 8. Criterios para mantener el legacy indefinidamente

El legacy puede permanecer si:
- es más simple y más estable;
- sirve como baseline ultra confiable;
- sigue aportando valor como smoke runner;
- evita que el proyecto pierda una referencia mínima.

---

## 9. Criterios de rechazo de transición

Se rechaza promoción si:
- el scenario runner solo "hace más cosas" pero no conserva baseline;
- la transición rompe comparabilidad;
- la memoria o continuidad se degradan;
- aparecen bugs de contaminación o colapso.

---

## 10. Test suite sugerida

- `test_runner_parity_thermal.py`
- `test_runner_trace_equivalence.py`
- `test_runner_artifact_equivalence.py`
- `test_runner_certification_parity.py`
