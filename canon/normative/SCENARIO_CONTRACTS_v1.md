---
title: SCENARIO_CONTRACTS_v1
status: normative
version: 1.1.0
date: 2026-07-10
owner: Wis
depends_on:
  - RNFE_canon_matematico_f2_4_v3_0.md
  - RUNTIME_SSOT_v1.md
  - HARDENING_ROADMAP.md
reissue:
  - "v1.1.0 (2026-07-10, A17): SCENARIO_CONTRACTS reconocía 2 escenarios; el registry vivo tiene 4.
    Se incorporan los 4 escenarios reales (thermal_homeostasis y resource_management canónicos;
    grid_thermal_5x5 y deferred_load_trap extra-canon/provisionales) y la taxonomía real de morfismos
    de 5 clases (el canon fijaba 4; la 5.a, 'adversarial', se incorpora). Nombres tomados de
    runtime/world/registry.py. Fija el criterio contra el cual se repara B26."
---

# Scenario Contracts v1

## 0. Propósito

Este documento fija el contrato formal de los escenarios cognitivos mínimos de RNFE.
Su función es impedir que el soporte multi-escenario derive en mundos incompatibles, no comparables o semánticamente ambiguos.

---

## 1. Definición de escenario cognitivo

Un escenario cognitivo es una instancia mínima de mundo operativo que define:

- estado observable;
- variable principal;
- condición de alarma;
- intervenciones válidas;
- transición factual;
- simulación contrafactual no mutante;
- proposiciones semánticas;
- fórmula LOTF mínima;
- criterio de evaluación factual vs contrafactual.

---

## 2. Campos obligatorios de identidad

Todo escenario debe declarar:

- `scenario_name`
- `scenario_version`
- `scenario_config_hash`
- `description`
- `main_variable`
- `alarm_threshold`
- `interventions`
- `formula_template`
- `type_context`

### 2.1 Regla
Dos escenarios con el mismo `scenario_name` y distinto comportamiento causal deben cambiar `scenario_version` y `scenario_config_hash`.

---

## 3. Contrato de interfaz

Todo escenario debe implementar:

- `observe()`
- `factual_transition(...)`
- `simulate_counterfactual(...)`
- `get_formula(...)`
- `select_intervention(...)`
- `get_main_proposition(...)`
- `get_intervention_proposition(...)`

Y opcionalmente:
- `evaluate_relation_kind(...)` si la semántica "mejor" no coincide con la default.

---

## 4. Invariantes obligatorios

### 4.1 No mutación contrafactual
`simulate_counterfactual()` no puede alterar el estado factual vivo.

### 4.2 Determinismo mínimo
Bajo mismo input, misma configuración y mismo estado inicial, el escenario debe producir la misma salida.

### 4.3 Coherencia LOTF
La fórmula generada debe ser chequeable por el contexto de tipos declarado.

### 4.4 Legibilidad causal mínima
Debe existir relación clara entre:
- observación,
- intervención,
- cambio en variable principal,
- comparación factual vs contrafactual.

---

## 5. Metadatos obligatorios en el episodio

Todo episodio generado desde un escenario debe persistir:

```json
{
  "scenario_metadata": {
    "scenario_name": "thermal_homeostasis",
    "scenario_version": "v1.0.0",
    "scenario_config_hash": "sha256:...",
    "main_variable": "temperature",
    "alarm_threshold": 0.85
  }
}
```

---

## 6. Escenarios oficiales actuales

Los cuatro escenarios registrados en el runtime vivo (`runtime/world/registry.py::SCENARIO_REGISTRY`). Los dos primeros son **canónicos**; los dos últimos son **extra-canon / provisionales** (nacidos como banco de prueba de capacidades específicas) y se reconocen aquí para que el soporte multi-escenario no derive en silencio. El nombre y la config de cada uno son literales del registry (no inventados).

### 6.1 thermal_homeostasis  *(canónico — PMV oficial, A19)*

* clase: `runtime/world/thermal_scenario.py::ThermalScenario`
* variable principal: `temperature`
* lógica: `TEMP_HIGH -> ACTIVATE_COOLING`
* intervenciones: `activate_cooling`, `deactivate_cooling`
* dirección de optimización: `minimize` (`lower_is_better`); alarma `threshold_above`
* topología de control: `threshold_single_loop`
* política contrafactual: `opposite_intervention`
* semántica de mejora: menor temperatura es mejor

### 6.2 resource_management  *(canónico)*

* clase: `runtime/world/resource_scenario.py::ResourceScenario`
* variable principal: `stock_level`
* lógica: `STOCK_LOW -> START_PRODUCTION`
* intervenciones: `start_production`, `stop_production`
* dirección de optimización: `maximize` (`higher_is_better`); alarma `threshold_below`
* topología de control: `threshold_recovery_loop`
* política contrafactual: `opposite_intervention`
* semántica de mejora: mayor stock es mejor

### 6.3 grid_thermal_5x5  *(extra-canon — provisional)*

* clase: `runtime/world/grid_thermal_scenario.py::GridThermalScenario`
* variable principal: `global_temp_mean` (media sobre grid 5x5)
* lógica: `TEMP_HIGH -> ACTIVATE_COOLING`
* intervenciones: `activate_cooling`, `deactivate_cooling`
* dirección de optimización: `minimize` (`lower_is_better`)
* topología de control: `threshold_single_loop_spatial` (extensión espacial del thermal)
* política contrafactual: `opposite_intervention`
* razón de ser: variante espacial del PMV para probar control sobre estado distribuido.

### 6.4 deferred_load_trap  *(extra-canon — provisional)*

* clase: `runtime/world/deferred_load_scenario.py::DeferredLoadScenario`
* variable principal: `load`
* lógica: `LOAD_HIGH -> SHED_LOAD`
* intervenciones: `boost_throughput` (trampa: mayor Δ inmediato, rebota vía deuda diferida), `shed_load` (previsor: sostenible)
* dirección de optimización: `minimize` (`lower_is_better`); alarma `threshold_above`
* topología de control: `threshold_single_loop` con **consecuencia diferida** (deuda oculta)
* política contrafactual: `opposite_intervention`
* razón de ser: trampa temporal para foresight — sólo un lector multi-paso (imaginación A11) evita elegir `boost_throughput`; terreno donde la previsión paga de forma medible.

> **Criterio de reparación de B26.** El registro de regímenes cubre hoy 2 de estos 4 escenarios y omite en silencio la renormalización en los cruces con los extra-canon. Con los 4 escenarios contractados aquí, B26 se repara extendiendo el mapeo a los 4 y **haciendo explícita** (no silenciosa) toda omisión de renormalización. Este documento fija el criterio contra el que B26 se mide.

---

## 7. Clasificación de comparabilidad entre escenarios

### 7.1 Equivalente

Mismo escenario, misma versión, mismo hash.

### 7.2 Compatible

Escenarios distintos, pero misma topología de control.

### 7.3 Analógico

Escenarios distintos con transferencia tentativa, no certificada por defecto.

### 7.4 Incompatible

No compartir memoria, benchmark comparativo ni claims de continuidad fuerte.

### 7.5 Taxonomía de morfismos dirigidos (5 clases) — A17

Las clases §7.1-§7.4 describen **comparabilidad simétrica**. El motor de morfismos vivo (`runtime/world/morphism_engine.py::MorphismClass`) computa además transformaciones **dirigidas** (source -> target, posiblemente asimétricas) y las clasifica en **cinco** clases. El canon fijaba cuatro; esta re-emisión (A17) incorpora la quinta (`adversarial`), que existe en el runtime y no tenía doctrina.

Correspondencia con la comparabilidad simétrica de §7 y semántica normativa:

| Clase de morfismo (dirigido) | Criterio del motor | Comparabilidad §7 | Regla normativa |
|---|---|---|---|
| **isomorphic** | `overall_score >= 0.95` y mismo `scenario_name`+`scenario_version` | §7.1 Equivalente | Transferencia total; memoria y benchmark compartibles. |
| **homomorphic** | `overall_score >= 0.75` | §7.2 Compatible | `is_transfer_safe_prior = True`; misma topología de control, diferencias menores. |
| **analogical** | `0.40 <= overall_score < 0.75` | §7.3 Analógico | Transferencia tentativa, no certificada por defecto; sólo como hint. |
| **adversarial** *(5.a, incorporada)* | `directionality_penalty >= 0.25` **y** `overall_score < 0.50` (inversión de dirección de optimización: `minimize <-> maximize`) | — (sin equivalente simétrico) | **Prohibida como transferencia positiva**: la interpretación de "mejora" se invierte entre escenarios. Es peligrosa, no meramente inútil. Todo cruce adversarial debe registrarse explícito y bloquear claims de continuidad/transferencia. |
| **incompatible** | `overall_score < 0.25` | §7.4 Incompatible | Sin alineamiento útil; no compartir memoria, benchmark ni continuidad fuerte. |

**Por qué la quinta clase importa.** `adversarial` no es un caso de "sin alineamiento" (eso es `incompatible`): es un alineamiento con **dirección de optimización invertida** — p.ej. `thermal_homeostasis` (`minimize`) frente a `resource_management` (`maximize`). Transferir memoria o política entre ellos como si fueran compatibles inyecta la mejora al revés. Por eso merece clase propia y regla de bloqueo, y por eso el registro de regímenes (B26) debe hacer explícita la omisión de renormalización en estos cruces en vez de callarla.

---

## 8. Reglas para agregar un nuevo escenario

Un nuevo escenario solo puede entrar si aporta:

1. semántica causal distinta;
2. no trivial duplicación de uno existente;
3. tests de factual y contrafactual;
4. metadata completa;
5. protocolo claro de comparabilidad;
6. benchmark mínimo dedicado.

---

## 9. Tests obligatorios por escenario

* observación válida
* transición factual actualiza estado
* contrafactual no muta
* fórmula parsea y chequea
* criterio de relación funciona
* runner produce episodio persistido
* artifact se materializa

---

## 10. Criterio de rechazo de escenario

Se rechaza un escenario si:

* oculta su semántica causal;
* depende de hacks del runner;
* no puede compararse consigo mismo de forma estable;
* no preserva no mutación contrafactual;
* rompe metadata o artifacts.

---

## 11. Plantilla de alta de escenario

### Nombre

### Versión

### Descripción

### Variable principal

### Umbral

### Intervenciones

### Fórmula

### Contexto de tipos

### Semántica de mejora

### Criterio factual vs contrafactual

### Tests

### Compatibilidad con escenarios existentes
