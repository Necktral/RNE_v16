---
title: SCENARIO_CONTRACTS_v1
status: normative
version: 1.0.0
date: 2026-04-17
owner: Wis
depends_on:
  - RUNTIME_SSOT_v1.md
  - HARDENING_ROADMAP.md
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

### 6.1 thermal_homeostasis

* variable principal: `temperature`
* lógica: HIGH -> ACTIVATE_COOLING
* semántica de mejora: menor temperatura es mejor

### 6.2 resource_management

* variable principal: `stock_level`
* lógica: LOW -> START_PRODUCTION
* semántica de mejora: mayor stock es mejor

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
