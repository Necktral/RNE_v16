---
title: ROADMAP_RNFE_v2
status: executive
version: 2.0.0
date: 2026-03-17
owner: Wis
depends_on:
  - CANON_RNFE_v3_1.md
  - SSOT_RAZONAMIENTOS_RNFE_v1.md
supersedes:
  - RNFE_manual_construccion_por_etapas_v1.md (orden fino)
  - RNFE_matriz_etapas_dependencias_v1.csv (una vez migrada)
---

# ROADMAP RNFE v2

## 0. Propósito

Este roadmap convierte el canon vigente en secuencia de construcción. No enumera deseos; define el orden exacto de ejecución, los gates técnicos, criterios de salida, riesgos y dependencias duras para construir un ser cibernético de próxima generación en condiciones físicas restringidas.

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

Seleccionar un único mini-mundo oficial para arranque. Debe ser suficientemente pequeño para iterar rápido y suficientemente rico para exigir signos, formalización y causalidad.

**Candidatos priorizados:**
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
