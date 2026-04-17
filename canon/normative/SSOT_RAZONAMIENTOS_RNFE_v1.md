---
title: SSOT_RAZONAMIENTOS_RNFE_v1
status: normative
version: 1.0.0
date: 2026-03-17
owner: Wis
depends_on:
  - CANON_RNFE_v3_1.md
primary_sources:
  - Tipos de Razonamiento Humano y su Replicación en IA.pdf
  - Mejoras en taxonomía de razonamientos.txt
  - RNFE_manual_construccion_por_etapas_v1.md
  - RNFE f2.2.txt
  - RNFE f2.3.txt
---

# SSOT de Razonamientos RNFE v1

## 0. Propósito

Este documento cierra de manera oficial la ontología, la operacionalización y la gobernanza de la capa de razonamientos RNFE. Su función es eliminar ambigüedad histórica, fijar alias canónicos, separar niveles y dejar contratos mínimos para implementación.

## 1. Principio rector

RNFE no usa “un razonamiento”, sino una **ecología de razón**. La razón del organismo se divide en tres estratos:

1. **familias inferenciales primarias**
2. **familias operativas de runtime**
3. **familias de gobierno y crítica**

La violación histórica que este documento corrige es mezclar esos tres estratos como si fueran uno solo.

## 2. Estrato I — Familias inferenciales primarias

Estas representan estilos inferenciales del organismo.

### 2.1 DED — Deductivo

**Definición**
Inferencia desde reglas, restricciones o axiomas hacia conclusiones necesarias.

**Rol en RNFE**
Verificación formal, chequeo de consistencia, validación de propuestas y constraints.

**Capa**
LOT-F / verificador / checker.

**Alias históricos**
`deductive`, `DED`.

**Motores plausibles**
rule engine, SMT, checker lógico, type checker.

**Acceptance**
La cadena inferencial debe ser trazable y reproducible.

### 2.2 IND — Inductivo

**Definición**
Generalización desde episodios, regularidades o patrones observados.

**Rol en RNFE**
Descubrimiento de regularidades del mini-mundo, detección de invariantes empíricas y compresión de episodios.

**Capa**
episodios / análisis de patrones / aprendizaje de estructura.

**Motores plausibles**
pattern mining, estimadores estadísticos, learners ligeros.

**Acceptance**
Debe mejorar predicción o compresión sin sobreajuste evidente.

### 2.3 ABD — Abductivo

**Definición**
Generación de hipótesis explicativas plausibles para observaciones parciales.

**Rol en RNFE**
Arranque de hipótesis causales, proposals internas, candidate explanations.

**Capa**
generador de hipótesis / imaginación controlada.

**Motores plausibles**
beam search explicativo, plantillas causales, LLM estructurado, generadores simbólicos.

**Acceptance**
Toda hipótesis abductiva debe pasar por pruebas causales, contrafactuales y crítica adversarial.

### 2.4 ANA — Analógico

**Definición**
Transferencia estructural desde episodios o grafos similares.

**Rol en RNFE**
Reutilización de patrones, recuperación por similitud estructural, compresión cognitiva multiescala.

**Capa**
SMG + memoria + retrieval estructural.

**Motores plausibles**
graph matching, retrieval estructural, embeddings de relaciones, VSA.

**Acceptance**
La analogía debe preservar estructura relevante, no solo similitud superficial.

### 2.5 CAU — Causal

**Definición**
Inferencia sobre relaciones de intervención y mecanismos generativos.

**Rol en RNFE**
Construcción del mundo, selección de acción, identificación de mecanismos.

**Capa**
C-GWM / causal graph / intervention model.

**Motores plausibles**
SCM, grafos causales, estimadores de intervención.

**Acceptance**
Debe diferenciar correlación de intervención útil.

### 2.6 CTF — Contrafactual

**Definición**
Simulación de mundos alternativos condicionados sobre eventos observados.

**Rol en RNFE**
Comparar propuestas, invalidar hipótesis y seleccionar acciones robustas.

**Capa**
world simulation / evaluation.

**Motores plausibles**
SCM contrafactual, simuladores internos, rollouts con constraints.

**Acceptance**
Debe producir comparación factual vs alternativo con trazabilidad causal.

### 2.7 PROB — Probabilístico

**Definición**
Razonamiento bajo incertidumbre, riesgo y evidencia parcial.

**Rol en RNFE**
Estimación de confianza, scoring, riesgo, posteriorización y decisión bajo incertidumbre.

**Capa**
scoring / uncertainty / selection.

**Motores plausibles**
Bayes, calibration, ensembles, Monte Carlo ligero.

**Acceptance**
Debe calibrar incertidumbre mejor que heurística nominal.

## 3. Estrato II — Familias operativas de runtime

Estas no equivalen exactamente a “tipos humanos” de razonamiento. Son modos operacionalizados que el runtime puede invocar.

### 3.1 PLAN — Planificación

**Definición**
Construcción de secuencias de acción hacia un objetivo dado el estado actual.

**Rol en RNFE**
Policy synthesis, scheduling interno, secuencias de intervención y pruebas.

**Alias históricos**
`PLAN`.

### 3.2 OPT — Optimización

**Definición**
Selección de configuraciones, rutas o decisiones bajo restricciones y costos.

**Rol en RNFE**
Control de presupuesto, tuning seguro, selección de acciones y trayectorias.

**Alias históricos**
parte de `CSTR/OPT`.

### 3.3 EVO/SEARCH — Exploración evolutiva y búsqueda

**Definición**
Búsqueda estructural o paramétrica de variantes útiles.

**Rol en RNFE**
Proponer mutaciones, explorar candidates, comparar poblaciones pequeñas.

**Alias históricos**
`EVO`.

### 3.4 NESY — Neuro-simbólico

**Definición**
Combinación explícita entre representaciones neuronales y operaciones simbólicas.

**Rol en RNFE**
Puente entre percepción comprimida, LOT-F, memoria y world model.

**Estado**
prioridad de laboratorio con camino explícito a integración.

## 4. Estrato III — Gobierno y crítica

### 4.1 META — Scheduler metacognitivo

**Definición**
Gobierno de la razón: selecciona familia, presupuesto, secuencia, profundidad, acceptance tests y fallback.

**Dictamen canónico**
`RAZ` deja de existir como familia primaria y se reinterpreta como `META`.

**Rol**
No “razona” sobre el mundo como DED o ABD; gobierna el uso de las demás familias.

### 4.2 DIA/ADV — Dialéctico / adversarial

**Definición**
Confrontación de hipótesis, destrucción de argumentos débiles, generación de contraejemplos.

**Rol**
ADC-PRIME, crítica interna, stress epistemológico.

### 4.3 HEUR — Heurístico

**Definición**
Atajos de bajo costo cuando el valor esperado de profundidad adicional no justifica el gasto.

**Rol**
triage, pruning, selección rápida, aproximaciones iniciales.

### 4.4 FAL-GUARD / CRIT — Guardián de falacias y criticidad epistémica

**Definición**
Subsistema permanente de higiene lógica y epistémica.

**Rol**
detectar contradicciones, falacias, razonamiento circular, abuso de analogía, confianza espuria.

## 5. Tabla de equivalencias históricas

| Alias histórico | Canónico actual | Observación |
|---|---|---|
| `RAZ` | `META` | no es familia inferencial primaria |
| `CSTR/OPT` | `DED-constraints + OPT` | constraints formales se absorben en DED; selección/costo en OPT |
| `EVO` | `EVO/SEARCH` | se explicita que es búsqueda/evolución guiada |
| `dialéctico` | `DIA/ADV` | se vuelve familia crítica explícita |
| `falacias` | `FAL-GUARD/CRIT` | subsistema permanente de higiene epistémica |
| `H-Net razona` | `META gobierna; engines razonan` | separación obligatoria |

## 6. Secuencia recomendada de uso

No existe una única secuencia universal, pero el baseline oficial de razonamiento compuesto es:

1. `ABD` genera hipótesis;
2. `ANA` recupera estructura similar;
3. `CAU` modela mecanismo;
4. `CTF` prueba alternativas;
5. `DED` verifica restricciones y cierre formal;
6. `PROB` calibra incertidumbre y riesgo;
7. `DIA/ADV` intenta destruir la hipótesis;
8. `FAL-GUARD` limpia errores de forma y sesgos lógicos;
9. `META` decide aceptación, fallback o escalamiento;
10. `S-I-E` decide herencia o cuarentena.

## 7. Política de selección por costo-beneficio

META debe decidir con al menos estas señales:

- costo estimado;
- ganancia esperada de cierre;
- riesgo;
- confianza;
- estado del borde;
- urgencia temporal;
- saturación de memoria/canal;
- historial de éxito por contexto.

## 8. Contrato mínimo para cualquier motor de razonamiento

Todo motor debe aceptar una estructura equivalente a:

- `context_id`
- `goal`
- `input_state`
- `evidence`
- `constraints`
- `budget`
- `risk_budget`
- `trace_policy`

Y debe devolver:

- `result`
- `confidence`
- `trace`
- `cost_used`
- `failure_mode`
- `recommended_next_family`
- `artifacts`

## 9. Acceptance tests por familia

### DED
Consistencia, reproducibilidad, ausencia de saltos no justificados.

### IND
Mejora predictiva o descriptiva fuera de muestra.

### ABD
Capacidad explicativa más evidencia de contraste.

### ANA
Conservación de estructura relevante.

### CAU
Mejora en intervención o explicación mecanística.

### CTF
Utilidad comparativa en alternativas y decisión.

### PROB
Calibración, cobertura y manejo de incertidumbre.

### PLAN/OPT
Cumplimiento de restricciones y mejor ruta/costo.

### EVO/SEARCH
Ganancia útil sin violar seguridad ni continuidad.

### NESY
Mejora en composicionalidad o trazabilidad sin costo desproporcionado.

### DIA/ADV
Capacidad de romper hipótesis débiles sin destruir rendimiento sano.

### HEUR
Ventaja costo/beneficio real frente a métodos más pesados.

### FAL-GUARD
Reducción de errores epistémicos y contradicciones.

## 10. Política de integración con el resto del organismo

- SMG aporta estructura semántica al razonamiento.
- LOT-F aporta formalización y chequeo.
- C-GWM aporta mundo e intervención.
- MFM/VFD aporta memoria multiescala y contexto.
- Hctrl/Edge limita el gasto cognitivo.
- S-I-E decide herencia.

## 11. Prohibiciones explícitas

1. usar `RAZ` como categoría canónica viva;
2. introducir una familia sin mapping a ontología, runtime y acceptance;
3. tratar heurística como sustituto permanente de causalidad o formalización;
4. aceptar un motor solo porque resuelve benchmarks externos sin integrabilidad real;
5. meter razonamientos nuevos sin tabla de alias y precedencia.

## 12. Cierre

La capa de razonamientos queda cerrada así:

- **primarias:** DED, IND, ABD, ANA, CAU, CTF, PROB
- **operativas:** PLAN, OPT, EVO/SEARCH, NESY
- **gobierno/crítica:** META, DIA/ADV, HEUR, FAL-GUARD

Con esto, la razón RNFE deja de ser una lista ambigua y pasa a ser una arquitectura gobernable.
