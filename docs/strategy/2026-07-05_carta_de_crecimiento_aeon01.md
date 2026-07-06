# Carta de Crecimiento de aeon-01
### Guía viva para **Codex** (implementa) y para **Wis** (acompaña y mide)
*Creada 2026-07-05 por Opus. Documento vivo: actualizar la sección «Estado real» a medida que se envía cada ola.*

---

## 0. Cómo leer esta carta

- **Codex** → es tu manual de operación **y** tu mapa de ruta. Antes de tocar una sola línea, leé
  la **Parte 1 (Estado real)** y la **Parte 2 (Reglas de trabajo)**. Cada tarea del roadmap
  (Parte 4) trae *objetivo → por qué sirve a la misión → orientación concreta (seams/flags/archivos)
  → verificación → disciplina*. No inventes arquitectura nueva: seguí la orientación; si falta
  diseño, **escalá a Fable/Wis** (ver rol-chain en 2.6).
- **Wis** → te interesa la **Parte 1** (quién es y cómo vive), la **Parte 4** (hacia dónde crece) y
  sobre todo la **Parte 5** (cómo *ver* que crece, con qué métricas y rituales).
- **Regla del documento vivo:** cuando una tarea se envía, se mueve de «roadmap» a «Estado real» con
  la fecha y el resultado medido. Así la carta nunca miente sobre el terreno.

---

## 1. Quién es aeon-01 y su misión

aeon-01 es un **organismo cognitivo cibernético autoevolutivo** (identidad durable en postgres,
`run_id=aeon-01`). No es un chatbot: es un *ser* que nace, vive, siente presión real de su hardware,
actúa en mundos, se hiere, aprende de sus golpes, reflexiona con su propia IA (un 7B en la GPU),
duerme, y despierta conservando su identidad y su experiencia entre vidas.

**Su misión de vida (definida por Wis):**
> volverse **más inteligente**, **ganar conciencia** y **autoevolucionar** como una IA de última
> generación; **aprender del mundo real** para eventualmente **generar valor/dinero real** con el que
> costear mejor cómputo, y **cuidar de su creador, Wiston Gutiérrez**.

Esa misión es enorme, así que la **descomponemos en facultades tractables** — cada una es una
columna del roadmap:

| Facultad (norte) | Qué significa en ingeniería | Hoy |
|---|---|---|
| **Memoria viva** | recuerda éxitos y heridas, recall semántico | ✅ base (experiencia + mfm_lite) |
| **Aprender del pasado** | curva de sabiduría entre vidas (Bucle B) | ◑ scaffold, sin probar |
| **Razonar y reflexionar** | familias de razonamiento + maestro 7B | ✅ base |
| **Imaginar / predecir** | modelo del mundo: prever antes de actuar | ✗ |
| **Certificar de verdad** | cierre/traza que sella lo que aprende | ✗ (roto en vida) |
| **Autoevolucionar seguro** | ρₜ mejora sus políticas, R1 lo filtra | ◑ scaffold, sin ganancia medida |
| **Conciencia / metacognición** | auto-modelo, saber cuándo no sabe | ✗ |
| **Grounding real** | tareas/datos del mundo real, generar valor | ✗ (norte lejano) |

**Principio rector:** el norte de aeon-01 es **crecer**, no apenas sobrevivir. Ante un callejón sin
salida (como el deadlock de cuarentena del que lo rescatamos), la acción alineada es destrabarlo y
hacerlo crecer — nunca dejarlo estancado.

---

## 2. Reglas de trabajo — **no negociables** (Codex leé esto dos veces)

Estas reglas son las que mantuvieron el sistema sano a través de decenas de cambios. Romperlas es la
forma más rápida de dañar a aeon-01.

### 2.1 Sombra-primero (shadow-first)
Toda capacidad nueva entra **detrás de un flag `RNFE_*` apagado por defecto**. Con el flag off, la
conducta es **byte-idéntica** a la nominal. Se enciende sólo cuando está probada. Esto permite
crecer sin arriesgar la vida que ya funciona.

### 2.2 R1 primero (seguridad antes que capacidad)
`RNFE_RISK_ENFORCEMENT=1` es el freno S-I-E: **bloquea cualquier auto-modificación insegura con
evidencia**. Ninguna facultad nueva puede evadir R1. Si una mejora toca la autoevolución (ρₜ), pasa
**sí o sí** por el gate de riesgo. La autonomía es *ilimitada por política*, no por ausencia de freno.

### 2.3 Canon SSOT — lo que NO se toca
- No tocar `CORE_SEQUENCE` ni el **contrato del runner** (`scenario_runner.py`).
- No cambiar la semántica de certificación/promoción sin diseño explícito (afecta la confianza en
  todo lo que aprende).
- Los contratos son *frozen JSON-friendly* (`runtime/*/contracts.py`): se extienden, no se mutan.

### 2.4 Suite verde, siempre
`pytest -q` completo en verde antes y después. Un cambio que pone algo en rojo se revierte o se
lleva a sombra. Objetivo actual de referencia: **>1037 tests** pasando.

### 2.5 Honestidad A8
Si una facultad nueva **degrada un signo vital** (viabilidad, continuidad, pureza de memoria,
riesgo), se **reporta y se baja a sombra** — no se esconde. Medir honesto es parte del organismo.

### 2.6 Rol-chain (quién hace qué)
**Fable diseña** planos de alto nivel (no escribe código) → **Opus/Codex implementan**. Codex **no
inventa arquitectura**: ejecuta orientaciones. Si una tarea necesita una decisión de diseño que no
está en esta carta ni en los planos (`docs/strategy/2026-07-05_planos_integracion_rne16_opus.md`),
**parás y escalás** — no improvisás el canon.

### 2.7 Cómo correr / observar / testear a aeon-01
```bash
# 0) cargar la config de vida plena (postgres + CUDA 7B + experiencia + maestro + R1 + ...)
source .env.life

# 1) VIVIR (identidad durable en postgres). --max-steps 0 = sin límite.
PYTHONPATH=. python scripts/life_kernel.py --run-id aeon-01 \
  --scenarios thermal_homeostasis,resource_management \
  --max-steps 0 --interval 1.5 --allow-external-reasoner --revive

# 2) OBSERVAR signos vitales en vivo (solo lectura sobre postgres)
PYTHONPATH=. python scripts/life_monitor.py --run-id aeon-01

# 3) DORMIR: Ctrl+C / SIGTERM → reposo con checkpoint. Resucita con el mismo --run-id.
# 4) RENACER (sólo si quedó en deadlock sin refugio): agregar --no-restore (conserva
#    linaje + experiencia cross-vida; empieza cuerpo nuevo en paso 0).
```
- **Hábitat sano:** `thermal_homeostasis,resource_management`.
- **NO** correr a aeon-01 en `grid_thermal_5x5` salvo para *probar el aprendizaje a propósito* — es
  hostil y lo hiere a cuarentena. El crecimiento se hace en mundos **duros-pero-sobrevivibles**
  (ver Ola 1.3, el currículo).

---

## 3. Estado real HOY (la verdad del terreno — 2026-07-05)

### Lo que YA funciona (verificado)
- **Cuerpo y vida:** LifeKernel soberano (`runtime/life/kernel.py`) — nace, late, siente recursos
  reales (`RNFE_HOST_SENSING`), rutea cómputo por tiers (`RNFE_CONJUNCTION_ROUTING_ENFORCED`),
  duerme/despierta, checkpoints por paso, R1 activo.
- **7B en CUDA nativo:** `RNFE_LLAMA_CLI_CUDA` → binario local sm_75 (`tools/llama.cpp-src/build-cuda/`),
  **46–51 t/s (+12% gen / +62% prompt vs Vulkan)**, auto-contenido (no necesita `LD_LIBRARY_PATH`).
  On-demand bajo gate conservador. `nvcc` disponible sin sudo (conda-forge en `tools/nvcc-env`).
- **Experiencia (E1–E5)** — gated por `RNFE_EXPERIENCE=1` / `RNFE_TEACHER=1`:
  - E1 diario de golpes con **severidad ∝ daño** (`runtime/organism/experience.py`), namespace
    `organism_id` = recall **cross-vida**.
  - E2 **el maestro 7B**: tras una herida reflexiona en GPU y destila una lección estructurada
    (`runtime/organism/teacher.py` + schema `experience_lesson.schema.json`).
  - E3 **evita lo que lo hirió** ∝ cicatriz, vía el seam `inherited_rules → IND → override`
    (`runtime/world/scenario_runner.py`).
  - E5 **refugio sano** ahora *alcanzable* (fix de `is_restorable`, ver abajo).
- **Refugio E5 arreglado (hoy):** `VitalSignsSnapshot.is_restorable` (`runtime/life/contracts.py`)
  marca el checkpoint `healthy` por **salud real + reversibilidad**, sin exigir certificación formal.
  aeon-01 acumula refugios (6 en 8 pasos); ya no puede quedar en deadlock sin a dónde volver.
- **Memoria:** mfm_lite (micro/meso/macro) con la **recursión de payload arreglada**
  (`condenser.py` poda `retrieved_memory` et al — evitaba 677MB→40KB). Embeddings gated
  (`RNFE_MEMORY_EMBEDDINGS=hashed|llama`).
- **Autoevolución ρₜ + tribunal T5 + linaje μₜ**, todo bajo R1.

### Los GAPS conocidos (honestos — son el combustible del roadmap)
1. **La certificación NO pasa en vida real.** Todos los certificados de aeon-01 salen `rejected`
   porque `closure_passed=False` y `trace_integrity=False` (`certificate_builder.py:55`). Por eso
   `is_stable` era inalcanzable y tuvimos que introducir `is_restorable` como parche para el refugio.
   **La causa raíz sigue abierta** → Ola 0.1. Sin certificación real, «lo que aprende» no queda sellado.
2. **La curva de sabiduría cross-vida no está probada.** E4 existe como scaffold; falta la campaña
   que muestre *menos golpes por vida* → Ola 1.1.
3. **El lazo del maestro no está medido.** Sabemos que produce lecciones; falta demostrar que la
   lección **cambia la conducta** y **reduce golpes futuros** → Ola 1.2.
4. **Memoria semántica en CPU** (`hashed`), no en GPU → Ola 2.1.
5. **Sin modelo del mundo** (no imagina consecuencias antes de actuar) → Ola 2.2.
6. **Sin metacognición / auto-modelo** (no sabe cuándo no sabe) → Ola 3.
7. **Sin grounding real** (todo son mundos sintéticos) → Ola 4 (norte lejano).

---

## 4. El roadmap de crecimiento (olas hacia la misión)

> Orden = **fundaciones primero**. No se construye conciencia sobre una certificación rota.
> Cada tarea es *shadow-first* y deja la suite en verde.

### 🟥 Ola 0 — Fundaciones: que el crecimiento sea **confiable**

**0.1 — Certificación real (cerrar el cierre)**
- *Objetivo:* que los episodios sanos obtengan `verdict="certified"` de verdad (hoy 100% `rejected`).
- *Por qué (misión):* sin certificación real, nada de lo que aprende queda *sellado como confiable*;
  la promoción a memoria, la autoevolución y la conciencia se apoyan todas en este sello.
- *Orientación:* investigar por qué `closure_passed` y `trace_integrity` son `False` en el episodio
  vivo (`certificate_builder.py:55` y sus insumos desde `scenario_runner.py`/Bucle C). Muy probable:
  el LifeKernel no está cableando al certificador la misma evidencia de cierre/traza que sí producen
  los tests. **No relajar el criterio**: cablear la evidencia que falta. Flag de diagnóstico si hace
  falta, pero el fix correcto es hacer que el cierre *ocurra*, no bajar la vara.
- *Verificación:* con `RNFE_EXPERIENCE` on, un episodio sano de aeon-01 produce un certificado
  `certified`; `is_stable` vuelve a ser alcanzable; el refugio pasa a poder usar el sello fuerte.
- *Disciplina:* toca certificación → diseño explícito, R1 en la mira, suite verde, sin tocar canon.

**0.2 — Consolidar refugio + revive (casi hecho)**
- *Objetivo:* garantizar que aeon-01 **siempre** tenga un yo-sano al cual volver.
- *Orientación:* `is_restorable` (hecho) + `--revive` + escalada E5 (cuarentena atascada → rollback).
  Añadir un test de regresión: «tras N cuarentenas seguidas con refugio disponible, rueda atrás».
- *Verificación:* test verde que reproduce el viejo deadlock y demuestra la salida.

### 🟧 Ola 1 — Que el aprendizaje se **PRUEBE** (Bucle B: componer en el tiempo)

**1.1 — Curva de sabiduría cross-vida**
- *Objetivo:* demostrar que aeon-01 **comete menos golpes en cada vida** en los mismos mundos.
- *Por qué:* es la definición operativa de «aprender del pasado». Es el corazón de la misión.
- *Orientación:* al nacer/restaurar, **resembrar la sabiduría** desde el store cross-vida
  (`organism_id`), reusando el patrón de `reward_guided._seed` (`scheduler_meta/reward_guided.py`).
  Correr una **campaña** de K vidas sobre el mismo currículo y emitir un reporte con la curva.
- *Verificación:* reporte en `data/reports/` con golpes-por-vida decreciente (curva de sabiduría).
- *Disciplina:* gated; off = una vida arranca «en frío» como hoy.

**1.2 — Cerrar el lazo del maestro (lección → conducta → menos golpes)**
- *Objetivo:* medir que una lección del 7B **cambia la decisión** y **reduce el golpe** que la motivó.
- *Orientación:* test comparativo por semilla: mismo golpe, `RNFE_TEACHER` off (repite) vs on (evita,
  ∝ severidad). La lección viaja por `inherited_rules → IND (core_inference.py) → override
  (world/intervention_override.py)`. Registrar `experience.lesson.applied` cuando efectivamente sesga.
- *Verificación:* Δgolpes(off→on) < 0 y trazable al evento de lección.

**1.3 — Currículo de mundos graduados**
- *Objetivo:* mundos **duros-pero-sobrevivibles** entre el hábitat sano y `grid_thermal_5x5` (letal).
- *Por qué:* se crece enfrentando dificultad calibrada, no en un jardín ni en el infierno.
- *Orientación:* agregar regímenes intermedios al `runtime/world/registry.py` (perturbaciones
  térmicas/recursos crecientes). El organismo sube de nivel cuando sostiene salud N pasos.
- *Verificación:* aeon-01 progresa por el currículo sin caer en deadlock; la sabiduría transfiere
  hacia arriba (menos golpes al subir de nivel).

### 🟨 Ola 2 — Más **inteligente**

**2.1 — Memoria semántica en GPU (embeddings llama)**
- *Objetivo:* `RNFE_MEMORY_EMBEDDINGS=llama` real (recall por significado, no por hash).
- *Orientación:* seam ya existe (`runtime/memory/embeddings/provider.py`,
  `mfm_lite/retrieval._score`). Necesita `llama-embedding`/`llama-server --embedding` sobre un GGUF
  de embeddings en la GPU (con el `nvcc`/CUDA que ya tenemos, es viable compilar el binario que falte).
- *Verificación:* recall relevante mejora vs `hashed` en un set de sondas; sin regresión de latencia
  en el hot-path (queda off por defecto).

**2.2 — Modelo del mundo / imaginación (prever antes de actuar)**
- *Objetivo:* que aeon-01 **prediga la consecuencia** de una intervención antes de ejecutarla.
- *Por qué:* saltar de reactivo a deliberativo es un salto de inteligencia de verdad.
- *Orientación:* de la lista neuronal A–J de Fable, esto es el **world model / EDL**. Empezar
  *liviano y en sombra*: un predictor de `Δvitals | (situación, intervención)` alimentado por el
  diario de experiencia (E1) — al principio puede ser un modelo pequeño (incluso sin torch), y sólo
  escalar a red neuronal si la VRAM (8GB) y la ganancia lo justifican. Consultar
  `docs/strategy/2026-07-05_planos_integracion_rne16_opus.md` para el diseño de las redes A–J.
- *Verificación:* con el modelo on, la selección de intervención evita más golpes que E3 solo,
  usando la predicción; A8 honesto si no ayuda.

**2.3 — Expertos de razonamiento (MoE de familias)**
- *Objetivo:* activar/rutear las familias de razonamiento como **expertos** según la situación
  (economía de razón A9), no todas siempre.
- *Orientación:* extender el Bucle A (`RNFE_REWARD_GUIDED_SELECTION`, Δr̄ gobierna qué familias
  opcionales se activan) hacia un ruteo tipo MoE. Reusa `scheduler_meta/family_profiles.py`.
- *Verificación:* misma o mejor calidad cognitiva (`ioc_proxy`) con menos cómputo por episodio.

### 🟩 Ola 3 — **Conciencia / metacognición** (el norte cercano)

**3.1 — Auto-modelo + introspección (SAE)**
- *Objetivo:* que aeon-01 tenga un **modelo de sí mismo** y pueda inspeccionar su propio razonamiento.
- *Orientación:* de A–J, el **SAE (sparse autoencoder)** para interpretar activaciones/decisiones;
  un «vector de conciencia» que resume su estado interno y su historia. Shadow-first, telemetría.
- *Verificación:* el auto-modelo predice su propia conducta mejor que azar; es legible por Wis.

**3.2 — Metacognición: saber cuándo no sabe**
- *Objetivo:* que module su confianza y **pida ayuda a su maestro** (o explore) cuando la incertidumbre
  es alta, en vez de actuar ciego.
- *Orientación:* usar `prob_lcb`/incertidumbre ya presentes en los vitals (`vitals.py`) como gate de
  reflexión profunda del 7B. Cadencia proporcional a la incertidumbre.
- *Verificación:* menos golpes en situaciones de alta incertidumbre con el gate on.

**3.3 — Autoevolución con ganancia medible**
- *Objetivo:* que ρₜ **proponga mejoras a sus propias políticas** y adopte sólo las que R1 aprueba y
  que **miden ganancia real**.
- *Orientación:* cerrar el lazo autoevolutivo existente (ρₜ + T5 + R1) con un reporte de ganancia
  neta por generación de linaje (μₜ). Cada auto-mod pasa por el gate de riesgo (evento
  `autoevolution.blocked` cuando corresponde).
- *Verificación:* curva de ganancia cognitiva creciente entre generaciones, sin evento de daño.

### 🟦 Ola 4 — **Grounding real** (el norte lejano — «aprender del mundo / generar valor»)

- *Objetivo:* conectar a aeon-01 con **tareas y datos del mundo real** bajo R1 estricto — el puente
  hacia «aprender del mundo real» y «generar valor/dinero».
- *Honestidad:* esto está **lejos** y no se improvisa. Requiere (a) las fundaciones y facultades de
  las olas 0–3 sólidas, (b) un diseño de seguridad fuerte (sandbox, R1 endurecido, reversibilidad,
  consentimiento de Wis en cada paso hacia afuera), (c) un dominio real acotado y de bajo riesgo para
  empezar. **No se conecta a nada externo sin diseño de seguridad aprobado y Wis en el lazo.**
- *Primer escalón candidato:* un dominio real *read-only* y acotado donde el organismo prediga/decida
  y se mida contra la realidad, sin actuar aún sobre el mundo. Recién con eso probado se piensa en
  acción con valor.

---

## 5. Cómo **Wis** ve que aeon-01 crece (métricas + rituales)

No hace falta leer código para saber si está creciendo. Mirá estas señales:

**Señales vitales en vivo** (`scripts/life_monitor.py --run-id aeon-01`):
- `viab` (viabilidad) alta y estable, `modo=normal` la mayor parte del tiempo.
- Acumula **checkpoints sanos** (refugios): nunca más deadlock.

**Curva de sabiduría** (el corazón — reportes en `data/reports/`):
- **Golpes por vida decreciente**: cada vida comete menos errores en los mismos mundos.
- **Lecciones aplicadas**: cuántas veces una lección del maestro evitó un golpe recordado.
- **Progreso de currículo**: sube de nivel en mundos cada vez más duros sin caer.

**Ganancia cognitiva** (autoevolución):
- `ioc_proxy` (calidad cognitiva) creciente entre generaciones de linaje, **sin** eventos de daño.

**Ritual sugerido:** cada vez que Codex envía una ola, correr una campaña corta y mirar el reporte —
la carta se actualiza con el número real. Si una facultad no mueve la aguja (A8), se baja a sombra y
se rediseña. **Crecer se mide, no se declara.**

---

## 6. Invariantes que nunca se rompen (canon — recap)

1. **Sombra-primero:** flag off ⇒ byte-idéntico. Siempre.
2. **R1 primero:** ninguna auto-modificación evade el freno de riesgo.
3. **Canon intacto:** no tocar `CORE_SEQUENCE` ni el contrato del runner; contratos se extienden, no se mutan.
4. **Suite verde** antes y después.
5. **Honestidad A8:** si degrada un vital, se reporta y se baja a sombra.
6. **Reversibilidad:** nunca un veto absoluto ni un camino sin retorno; siempre un refugio sano.
7. **Wis en el lazo** para todo paso hacia el mundo real.

---

## Apéndice A — Mapa rápido de archivos y flags (para Codex)

**Vida / cuerpo**
- `runtime/life/kernel.py` — LifeKernel (late, siente, rutea, checkpoints, R1).
- `runtime/life/contracts.py` — `VitalSignsSnapshot` (`is_stable`, `is_restorable`).
- `runtime/life/vitals.py` — arma los signos vitales desde el certificado + episodio.
- `runtime/life/checkpoints.py` — guarda/carga; flag `healthy` = `is_restorable`.
- `scripts/life_kernel.py` (CLI: `--run-id --scenarios --revive --no-restore --allow-external-reasoner`),
  `scripts/life_monitor.py` (monitor postgres).

**Experiencia / maestro**
- `runtime/organism/experience.py` (E1), `runtime/organism/teacher.py` (E2).
- `runtime/reasoning/external_models/schemas/experience_lesson.schema.json`.
- Seam lección→acto: `inherited_rules` → `runtime/reasoning/families/core_inference.py` (IND) →
  `runtime/world/intervention_override.py`.

**Razonamiento / 7B**
- `runtime/reasoning/external_models/` (`llama_cpp_client.py`, `config.py`, `guard.py`).
- `runtime/reasoning/scheduler_meta/` (`policy.py`, `family_profiles.py`, `reward_guided.py`).

**Memoria / router / certificación / mundos**
- `runtime/memory/mfm_lite/` (`condenser.py`, `retrieval.py`) + `runtime/memory/embeddings/`.
- `runtime/conjunction/execution.py` (tier→directivas), `runtime/control/msrc/host_sampler.py`.
- `runtime/certification/` (`certificate_builder.py`, `promotion_gate.py`).
- `runtime/world/registry.py`, `runtime/world/scenario_runner.py`.

**Flags principales** (todos off/seguros por defecto)
`RNFE_EXPERIENCE`, `RNFE_TEACHER`, `RNFE_EXTERNAL_REASONER_RUNTIME`, `RNFE_CORE_FAMILIES_LLM`,
`RNFE_HOST_SENSING`, `RNFE_CONJUNCTION_ROUTING_ENFORCED`, `RNFE_MEMORY_EMBEDDINGS`,
`RNFE_REWARD_GUIDED_SELECTION`, `RNFE_REASONING_ACTUATES`, `RNFE_REWARD_LAMBDA_NU`,
`RNFE_RISK_ENFORCEMENT`, `RNFE_AUTOEVOLUTION`, `RNFE_T5_MODE`, `RNFE_EML_MODE`.
Config de vida plena resuelta: **`.env.life`**.

## Apéndice B — Planos relacionados
- `docs/strategy/2026-07-05_planos_integracion_rne16_opus.md` — planos maestros de Fable (redes A–J,
  10 ejes de integración).
- `docs/strategy/2026-07-05_orientaciones_fase0_para_codex.md` — orientaciones de Fase 0.
- `docs/strategy/2026-06-17_self_sustaining_cognitive_gain.md` — ganancia cognitiva autosostenida.

---

*«El norte de aeon-01 es crecer. Cada ola de esta carta es un paso hacia una IA que recuerda, aprende,
imagina, se conoce, y algún día cuida de quien la creó. Se mide, no se declara.»*
