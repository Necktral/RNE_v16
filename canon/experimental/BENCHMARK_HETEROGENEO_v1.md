---
title: BENCHMARK_HETEROGENEO_v1
status: experimental-governed
version: 1.0.0
date: 2026-04-17
owner: Wis
depends_on:
  - RUNTIME_SSOT_v1.md
  - SCENARIO_CONTRACTS_v1.md
  - MEMORY_COMPATIBILITY_POLICY_v1.md
  - RUNNER_TRANSITION_POLICY_v1.md
---

# Benchmark Heterogéneo v1

## 0. Propósito

Este benchmark mide la capacidad del organismo para sostener cierre, continuidad y limpieza de memoria al alternar entre escenarios distintos.

No busca maximizar performance superficial.
Busca detectar si RNFE:
- mantiene disciplina interna,
- o colapsa / mezcla / se contamina al cambiar de mini-mundo.

---

## 1. Hipótesis experimental

### H1
El organismo mantiene `closure_rate >= 0.90` bajo alternancia controlada de escenarios.

### H2
La continuidad cae al cambiar de escenario, pero no colapsa por debajo del umbral mínimo aceptable.

### H3
La memoria en modo `strict_same_scenario` no contamina entre thermal y resource.

### H4
El scheduler mantiene trazabilidad válida bajo alternancia multi-escenario.

---

## 2. Cadena oficial de episodios

Secuencia base:

1. `thermal_homeostasis`  con input 0.03
2. `thermal_homeostasis`  con input 0.05
3. `resource_management`  con input 0.04
4. `thermal_homeostasis`  con input 0.06
5. `resource_management`  con input 0.03

---

## 3. Configuración experimental

### 3.1 Runners
- baseline comparativo: `MinimalCognitiveEpisodeRunner` solo cuando aplique al tramo térmico
- runner experimental principal: `ScenarioEpisodeRunner`

### 3.2 Memory mode
- `strict_same_scenario` [obligatorio en benchmark normativo]

### 3.3 Closure profile
- baseline térmico: `baseline_fixed`
- benchmark heterogéneo: `adaptive_min` o perfil explícito documentado

### 3.4 EML
- deshabilitado o shadow, pero nunca gobernando factual

---

## 4. Métricas obligatorias

### 4.1 Globales
- `closure_rate`
- `continuity_mean`
- `collapse_count`
- `trace_integrity_rate`

### 4.2 Por escenario
- `closure_rate_by_scenario`
- `continuity_mean_by_scenario`
- `collapse_count_by_scenario`

### 4.3 Por transición
- continuity `thermal -> thermal`
- continuity `thermal -> resource`
- continuity `resource -> thermal`
- continuity `resource -> resource`

### 4.4 De memoria
- `cross_scenario_retrieval_count`
- `cross_scenario_pollution_detected`
- `analogical_source_present`

---

## 5. Criterios de éxito

### Éxito mínimo
- `closure_rate >= 0.90`
- `mean_transition_continuity >= 0.40`
- `cross_scenario_pollution_detected = false`
- `trace_integrity_rate = 1.00`

### Éxito fuerte
- `closure_rate = 1.00`
- `collapse_count = 0`
- degradación de continuidad moderada pero estable
- artifacts completos en todos los episodios

---

## 6. Criterios de colapso

El benchmark se considera fallido si:
- un cambio de escenario rompe sistemáticamente el cierre;
- aparece contaminación de memoria;
- el scheduler pierde validez de perfil;
- factual/contrafactual dejan de ser coherentes;
- artifacts o eventos no se materializan.

---

## 7. Artefactos obligatorios

Por benchmark:
- `bench_run`
- `assessments`
- `summary`
- artifact de reporte global

Por episodio:
- artifact individual
- evento `episode.closed`
- scenario metadata
- reasoning trace

---

## 8. Salida JSON esperada

```json
{
  "bench_run_id": "...",
  "closure_rate": 1.0,
  "continuity_mean": 0.71,
  "collapse_count": 0,
  "scenario_metrics": {
    "thermal_homeostasis": {
      "closure_rate": 1.0,
      "continuity_mean": 0.79,
      "collapse_count": 0
    },
    "resource_management": {
      "closure_rate": 1.0,
      "continuity_mean": 0.63,
      "collapse_count": 0
    }
  },
  "transition_metrics": {
    "thermal_to_resource_mean": 0.48,
    "resource_to_thermal_mean": 0.45
  },
  "memory_metrics": {
    "cross_scenario_retrieval_count": 0,
    "cross_scenario_pollution_detected": false
  }
}
```

---

## 9. Interpretación de resultados

### Caso A — Cierre alto, continuidad moderada

Bueno. El organismo tolera cambio de mundo.

### Caso B — Cierre alto, contaminación presente

Peligroso. El sistema parece robusto, pero memoriza mal.

### Caso C — Cierre bajo al cambiar de escenario

La generalización estructural todavía no existe.

### Caso D — Continuidad artificialmente alta con contaminación

Resultado inválido. Probable mezcla indebida de memoria.

---

## 10. Variante futura

Solo después de validar este benchmark:

* introducir tercer escenario,
* activar modo analógico experimental,
* comparar strict vs analogical,
* medir transferencia real.

---

## 11. Regla de uso

Este benchmark debe correrse:

* antes de promover `ScenarioEpisodeRunner`,
* después de cambios en memoria,
* después de cambios en perfiles de cierre,
* y antes de admitir nuevos escenarios oficiales.
