---
title: CANON_RNFE_v3_2_rc1
status: normative
version: 3.2.0-rc1
date: 2026-03-24
owner: Wis
depends_on:
  - RNFE_canon_matematico_f2_4_v3_0.md
  - SSOT_RAZONAMIENTOS_RNFE_v1.md
supersedes:
  - CANON_RNFE_v3_1.md
notes:
  - Esta versión refactoriza el canon para separar axiomas duros, normativa viva, arquitectura provisional y laboratorio falsable.
  - El objetivo es preservar identidad y disciplina sin congelar prematuramente hipótesis de frontera.
---

# CANON RNFE v3.2-rc1

## 0. Propósito

Este documento redefine el régimen normativo de RNFE para una etapa de frontera controlada.

Su objetivo no es volver blando el proyecto, sino hacer una distinción rigurosa entre:

1. lo que define la identidad del organismo y no debe licuarse;
2. lo que hoy gobierna el runtime real;
3. lo que es arquitectura vigente pero revisable;
4. lo que pertenece al laboratorio y debe vivir bajo falsación, no bajo autoridad doctrinal.

RNFE queda definido como un organismo cibernético digital autoevolutivo, no humanoide, orientado a inteligencia general adaptable, con cierre triádico, continuidad identitaria, viabilidad dinámica, memoria viva multiescala, ecología de razón gobernada y herencia certificada.

La tesis rectora de esta versión es:

**un proyecto de frontera necesita axiomas duros muy pocos y muy firmes, pero necesita arquitectura provisional explícita y laboratorio falsable amplio.**

---

## 1. Régimen normativo por capas

### 1.1 Capas oficiales

La autoridad documental queda organizada en cinco capas:

#### Capa A — Axiomática dura
Contiene únicamente invariantes constitutivos del organismo.
No regula táctica fina ni orden detallado de construcción.
Su modificación requiere contradicción empírica fuerte o sustituto formalmente superior.

#### Capa B — Normativa viva
Contiene contratos, gates y reglas que gobiernan el runtime real y el repositorio productivo actual.

#### Capa C — Arquitectura provisional
Contiene la mejor configuración vigente de módulos, acoplamientos, orden recomendado y decisiones estructurales revisables.

#### Capa D — Laboratorio falsable
Contiene blueprints, hipótesis, engines candidatos, papers internos, variantes de arquitectura, benchmarks y exploraciones de frontera.

#### Capa E — Histórico / deprecated
Contiene material archivado, absorbido, sustituido o no gobernante.

### 1.2 Regla de conflicto

La precedencia queda fijada así:

**Axiomática > Normativa viva > Arquitectura provisional > Laboratorio falsable > Histórico**

Una decisión de capa inferior no puede contradecir una superior.
Una idea del laboratorio puede tensionar una capa superior, pero solo mediante ADR y ruta explícita de promoción.

### 1.3 Regla de desviación legítima

Se permite desviación temporal del orden vigente solo si:

1. no viola axiomas duros;
2. no contamina el tronco vivo sin promoción;
3. deja ADR formal;
4. define experimento falsable;
5. respeta el presupuesto físico del proyecto.

---

## 2. Axiomática dura del organismo

Los siguientes axiomas son constitutivos y no se rebajan a preferencia arquitectónica.

### A1 — Primacía de inteligencia útil con cierre
RNFE optimiza primero capacidad cognitiva con cierre y después coste.
Ningún ahorro de coste justifica una degradación estructural del cierre.

### A2 — Viabilidad antes que seguridad instantánea
Un estado seguro no basta.
El organismo solo cuenta como vivo si además puede sostener continuidad, cierre y control bajo restricciones reales.

### A3 — Cierre triádico obligatorio
Ningún módulo se considera cognitivo si no contribuye al ciclo:

**Significado → Forma → Mundo → Significado**

o a su estabilización certificada.

### A4 — Continuidad identitaria como gate duro
Toda mejora local que rompa continuidad identitaria cuenta como reemplazo, colapso o degradación, no como evolución válida.

### A5 — Herencia solo bajo certificación fuerte
Toda mutación paramétrica o estructural entra como propuesta.
Nada asciende por entusiasmo, intuición o presión heurística.

### A6 — Morfogénesis tipada
La auto-reescritura solo es legítima si preserva tipos, contratos, viabilidad, trazabilidad y posibilidad de rollback.

### A7 — Estabilidad solo si reaparece
Ningún patrón cognitivo asciende por una corrida aislada.
Debe reaparecer con dispersión controlada entre semillas, contextos o condiciones.

### A8 — Observabilidad primaria
Las decisiones de ascenso, rechazo o herencia se toman con telemetría, artefactos y pruebas, no con retórica, analogía o sofisticación verbal.

### A9 — Economía de razón
El organismo no activa por defecto el razonamiento más profundo ni el más costoso, sino el que maximiza ganancia de cierre ajustada por riesgo, disipación y costo.

---

## 3. Invariantes obligatorios del núcleo vivo

Estos invariantes son duros porque separan existencia, simulación y autoengaño.

### 3.1 Invariantes cognitivos mínimos

1. **SMG** debe existir antes de declarar identidad cognitiva.
2. **LOT-F** debe existir antes de declarar deducción formal trazable.
3. **C-GWM** debe existir antes de declarar causalidad o contrafactualidad reales.
4. **OMG + certificados** deben existir antes de herencia fuerte.
5. **Edge/Hctrl** deben gobernar toda exploración no trivial.
6. **Rollback** debe ser atómico, auditable y probado.

### 3.2 Invariantes de gobierno

1. Todo documento activo declara `status`, `version`, `date`, `owner`, `depends_on`, `supersedes`.
2. Todo cambio estructural del tronco vivo requiere ADR.
3. Todo alias histórico debe mapear a nombre canónico.
4. Ningún documento de laboratorio modifica el tronco sin promoción explícita.
5. Ninguna ruta de producto redefine la ontología del organismo.

---

## 4. Estados documentales oficiales

Los estados válidos pasan a ser:

- `axiomatic`
- `normative`
- `provisional`
- `experimental`
- `historical`
- `deprecated`

### 4.1 Significado operativo

#### axiomatic
Define identidad constitutiva del organismo.

#### normative
Gobierna el runtime, contratos y gates actualmente vigentes.

#### provisional
Representa la mejor arquitectura vigente, pero puede cambiar sin crisis ontológica.

#### experimental
Representa hipótesis, blueprints, engines, módulos o teorías en fase de falsación.

#### historical
Material archivado sin poder gobernante.

#### deprecated
Material sustituido formalmente con migración definida.

---

## 5. Reglas de promoción

### 5.1 experimental → provisional

Requiere simultáneamente:

1. definición matemático-contractual suficiente;
2. hipótesis falsable explícita;
3. ubicación arquitectónica tentativa;
4. costo estimado compatible con Windows 11 + WSL2 + 8 GB VRAM y trabajo individual;
5. experimento inicial diseñado;
6. ADR de evaluación si impacta módulos del tronco.

### 5.2 provisional → normative

Requiere:

1. experimento o benchmark reproducible;
2. contrato de entrada/salida definido;
3. impacto arquitectónico confirmado;
4. costo/beneficio favorable en hardware objetivo;
5. integración sin contradicción con axiomas;
6. ADR de adopción;
7. plan de rollback.

### 5.3 normative → axiomatic

Solo procede si el elemento:

1. reaparece como necesidad estructural en múltiples módulos o fases;
2. su eliminación destruye identidad del organismo;
3. no es una preferencia táctica sino una condición de existencia;
4. está respaldado por experiencia de runtime y no solo por elegancia teórica.

### 5.4 normative/provisional → deprecated

Requiere:

1. sustituto explícito;
2. tabla de equivalencias;
3. plan de migración;
4. no regresión de continuidad documental.

---

## 6. Qué queda en la capa normativa viva

Permanece como normativa viva:

1. contratos obligatorios del repositorio;
2. gates de episodio, propuesta, certificado y rollback;
3. reglas de gobernanza documental;
4. criterios de aceptación del núcleo vivo;
5. restricciones de seguridad, observabilidad y trazabilidad;
6. prohibición de herencia sin certificación;
7. separación fuerte entre tronco y laboratorio.

También permanecen normativos:

- continuidad identitaria;
- certificado ampliado;
- no-regresión;
- shadow mode;
- CVaR o equivalente de cola;
- rollback auditable;
- kill-switch;
- cuarentena/lab para cambios dudosos.

---

## 7. Qué baja a arquitectura provisional

Desde esta versión dejan de ser constitucionales y pasan a capa `provisional`:

1. el orden fino de implementación;
2. la granularidad exacta de módulos;
3. el layout exacto de familias y subfamilias;
4. el orden concreto de activación de engines;
5. la forma concreta del scheduler;
6. la estrategia exacta de agentes;
7. la táctica de acoplamiento entre shell, exocorteza y núcleo;
8. la secuencia detallada de fases salvo los gates mínimos de existencia.

La razón es simple:
son decisiones de ingeniería revisables, no leyes de identidad del organismo.

---

## 8. Regla de frontera para exploración estructurada

### Cláusula F — Derecho de exploración estructurada

En zonas donde aún no existe evidencia suficiente para fijar arquitectura definitiva, RNFE autoriza múltiples diseños provisionales en paralelo, siempre que:

1. no violen axiomas duros;
2. no alteren el tronco vivo sin promoción;
3. definan contrato tentativo;
4. generen experimento falsable;
5. produzcan artefactos auditables;
6. no colapsen el presupuesto físico del proyecto.

Esta cláusula no autoriza caos.
Autoriza exploración disciplinada.

---

## 9. Integración del evaluador crítico formal

Se reconoce oficialmente una nueva clase de artefacto:

### `reasoning/critique/critical_evaluator`

Su función es evaluar hipótesis candidatas bajo evidencia cambiante, supuestos auditables, restricciones lógicas, soporte observacional, soporte intervencional, consistencia contrafactual, robustez, refutación adversarial, viabilidad e incertidumbre residual.

### 9.1 Estado canónico inicial
El evaluador crítico formal entra al corpus como:

**`experimental` con ruta prioritaria a `provisional`**

### 9.2 Condiciones para ascenso a `provisional`

Debe presentar:

1. contrato de entradas/salidas RNFE;
2. mapeo a SMG, LOT-F, C-GWM y scheduler;
3. costo estimado en mini-mundo oficial;
4. suite mínima de tests;
5. política de fallback a indeterminación racional;
6. criterios de rechazo por refutación decisiva o inviabilidad.

### 9.3 Regla de prudencia
Ningún evaluador crítico, por elegante que sea, puede gobernar el tronco vivo antes de que existan al menos:

- `SMG_min`
- `LOTF_min`
- `CGWM_min`
- contratos de episodio/propuesta/certificado
- telemetría y rollback

---

## 10. Prohibiciones reformuladas

Esta versión endurece menos la exploración, pero endurece mejor las fronteras.

### 10.1 Sigue prohibido

1. declarar identidad cognitiva sin SMG;
2. declarar deducción formal real sin LOT-F;
3. declarar causalidad/contrafactualidad real sin C-GWM;
4. abrir autoevolución sin certificados, rollback y cuarentena;
5. absorber conocimiento externo directo al tronco vivo;
6. confundir sofisticación textual con runtime falsable;
7. usar multiagente como sustituto del núcleo cognitivo;
8. romper nomenclatura sin tabla de equivalencias;
9. mezclar shell operativo con ontología del organismo;
10. monetizar antes de nacimiento cognitivo mínimo validado.

### 10.2 Deja de estar prohibido en laboratorio

Ya no queda prohibido, siempre que permanezca en `experimental` o `provisional`:

1. diseñar engines avanzados antes del cierre total del núcleo;
2. formalizar módulos de crítica, imaginación o evaluación;
3. explorar variantes de scheduler;
4. ensayar arquitecturas de razonamiento en paralelo;
5. producir teoría nueva si viene acompañada de contrato, experimento o ruta de integración.

---

## 11. Regla nueva para teoría admisible

La teoría nueva solo se considera admisible si responde de forma explícita al menos a tres de estas cuatro preguntas:

1. ¿qué módulo toca?
2. ¿qué contrato cambia o crea?
3. ¿qué métrica o gate pretende mejorar?
4. ¿qué experimento podría refutarla?

Si no responde eso, no sube del archivo de ideas.

---

## 12. Relación con el roadmap

El roadmap deja de operar como constitución de segundo nivel y pasa a ser:

**instrumento ejecutivo dependiente del canon, no fuente de axiomas.**

Se mantiene la secuencia macroscópica:

1. gobernanza y congelación;
2. nacimiento cognitivo mínimo;
3. estabilidad y continuidad;
4. ecología de razón y evolución controlada;
5. exocorteza operativa y producto.

Pero el orden fino, la táctica y la granularidad de subpasos pueden variar por ADR mientras no violen axiomas, invariantes ni gates de existencia.

---

## 13. Contratos mínimos obligatorios del repositorio

El repositorio vivo debe conservar como mínimo:

- `canon/`
- `roadmap/`
- `governance/`
- `contracts/`
- `core/`
- `smg/`
- `lotf/`
- `world/`
- `memory/`
- `control/`
- `reasoning/`
- `agents/`
- `evolution/`
- `ingest/`
- `lab/`
- `archive/`
- `telemetry/`

Y debe existir como mínimo contrato activo para:

- episodio
- propuesta
- certificado
- rollback
- telemetry snapshot

---

## 14. Criterio oficial de avance real

RNFE no avanza por aumentar corpus, nombres o taxonomías.
RNFE avanza si y solo si mejora simultáneamente al menos una de estas dimensiones sin degradar las demás de forma estructural:

1. cierre triádico efectivo;
2. continuidad identitaria;
3. viabilidad bajo restricciones reales;
4. capacidad de razonar mejor de forma medible;
5. calidad de herencia certificada;
6. trazabilidad y reversibilidad.

---

## 15. Definición de frontera legítima

Una exploración cuenta como frontera legítima si cumple estas cinco condiciones:

1. ataca una limitación real del organismo;
2. no contradice axiomas duros;
3. puede implementarse o simularse en mini-mundo o banco de prueba;
4. produce una señal de aceptación o rechazo;
5. deja rastro auditable.

---

## 16. Cierre normativo

Esta versión establece una frontera más precisa que la anterior:

- la identidad del organismo sigue siendo dura;
- la disciplina de herencia, continuidad, viabilidad y observabilidad sigue siendo dura;
- la arquitectura concreta deja de fingirse definitiva;
- el laboratorio deja de ser sospechoso por existir y pasa a ser obligatorio, pero bajo falsación;
- los blueprints de alto nivel pueden entrar al proyecto sin invadir el tronco antes de tiempo.

La regla final queda así:

**proteger lo constitutivo, flexibilizar lo arquitectónico, y exigir falsación a toda novedad de frontera.**