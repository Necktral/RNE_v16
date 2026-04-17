---
title: MEMORY_COMPATIBILITY_POLICY_v1
status: normative
version: 1.0.0
date: 2026-04-17
owner: Wis
depends_on:
  - RUNTIME_SSOT_v1.md
  - SCENARIO_CONTRACTS_v1.md
---

# Memory Compatibility Policy v1

## 0. Propósito

Este documento define la política de compatibilidad de memoria en RNFE para evitar contaminación entre episodios de distintos escenarios.

La memoria viva no debe mezclar experiencia solo por similitud superficial.
Debe respetar identidad de escenario, régimen causal y modo de recuperación.

---

## 1. Principio rector

**La memoria útil debe preservar continuidad sin introducir contaminación causal.**

Por defecto, la recuperación de memoria es conservadora.

---

## 2. Modos oficiales de compatibilidad

### 2.1 `strict_same_scenario` [DEFAULT]
Solo recupera memorias del mismo:
- `scenario_name`
- y preferiblemente misma `scenario_version`

Este modo gobierna:
- runtime vivo,
- certificación,
- benchmark normativo,
- promotion gate.

### 2.2 `cross_scenario_analogical` [EXPERIMENTAL]
Permite recuperación de otros escenarios con penalización explícita de score y marcado de procedencia analógica.

Este modo no gobierna por defecto:
- ni certificación,
- ni cierre baseline,
- ni benchmark normativo principal.

---

## 3. Metadata mínima obligatoria por memoria

Toda memoria persistida debe incluir:

```json
{
  "scenario_name": "...",
  "scenario_version": "...",
  "scenario_config_hash": "...",
  "compatibility_class": "equivalent|compatible|analogical|incompatible"
}
```

---

## 4. Política de recuperación

### 4.1 En modo estricto

Se filtran completamente memorias con escenario distinto.

### 4.2 En modo analógico

Se permite recuperar memorias cross-scenario, pero:

* con penalización de score;
* con bandera explícita `analogical_source=true`;
* sin promoción automática a evidencia fuerte.

---

## 5. Política de scoring sugerida

### 5.1 strict

```text
si scenario_name != query.scenario_name -> descartar
```

### 5.2 analogical

```text
si scenario_name != query.scenario_name:
    score = score * penalty_cross_scenario
```

Valores iniciales sugeridos:

* `penalty_cross_scenario = 0.5`
* `penalty_cross_version = 0.8`

---

## 6. Contaminación de memoria

Se define contaminación cuando una memoria:

* de otro escenario,
* o de configuración incompatible,
  influye selección, evaluación o certificación como si fuera evidencia equivalente.

### 6.1 Contaminación fuerte

Afecta intervención factual o promotion gate.

### 6.2 Contaminación moderada

Afecta continuidad o scoring comparativo.

### 6.3 Contaminación débil

Solo aparece como hint no vinculante.

---

## 7. Regla de uso por subsistema

### Runtime factual

`strict_same_scenario`

### Certificación

`strict_same_scenario`

### Benchmark heterogéneo normativo

`strict_same_scenario`

### Exploración de analogía

`cross_scenario_analogical`

### EML / motores simbólicos shadow

permitido experimentalmente, pero marcado

---

## 8. Reglas de promoción de memoria

Una memoria cross-scenario:

* no puede ascender a memoria macro como si fuera equivalente;
* no puede justificar sola una promoción;
* no puede mejorar artificialmente continuidad.

---

## 9. Tests obligatorios

* strict filtra memorias de otro escenario
* analogical permite recuperación con penalización
* certificación ignora analogical como evidencia fuerte
* benchmark detecta contaminación si aparece mezcla indebida

---

## 10. Señales a registrar en artifacts y logs

* `scenario_filter_mode`
* `retrieved_same_scenario_count`
* `retrieved_cross_scenario_count`
* `cross_scenario_penalty_applied`
* `analogical_source_present`

---

## 11. Criterio de éxito de esta policy

La policy es exitosa si:

* la memoria ayuda intra-escenario,
* no contamina benchmark heterogéneo,
* y el modo analógico queda totalmente distinguible del modo estricto.
