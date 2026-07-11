---
title: Diseño P-CADENA-CAUSAL (B41) — Separación de ejes de identidad del organismo
status: draft
date: 2026-07-10
paquete: P-CADENA-CAUSAL
work_item: B41 (identidad del organismo)
adjudicacion: P-CADENA-CAUSAL · B41
branch: repair/cadena-causal
owner: Wis
fase: DISEÑO (pre-código — requiere ratificación humana; es identidad del organismo)
description: Diseño B41 para separar los tres ejes de identidad (run_id efímero, organism_id genoma persistente, lineage_id linaje) hoy colapsados en run_id, con CausalContext.v1, mapa de impacto y plan de compatibilidad.
tags:
  - identidad
  - organismo
  - linajes
  - diseño
  - cadena-causal
  - B41
anchors:
  - canon/normative/RNFE_canon_matematico_f2_4_v3_0.md  # A-M5, A-M8, A-M10, D6, D7, D8, C-AC4
  - canon/normative/MEMORY_COMPATIBILITY_POLICY_v1.md
---

# B41 — Separación de ejes de identidad del organismo

**Notas relacionadas:** [[2026-07-05_carta_de_crecimiento_aeon01]] ·
[[2026-07-05_planos_integracion_rne16_opus]] · [[2026-06-17_self_sustaining_cognitive_gain]]

> **Estado: borrador para ratificación.** Este documento NO implementa nada.
> Toca identidad del organismo: nada se cablea sin ratificación humana explícita.
> Todas las citas de código fueron verificadas en el worktree `repair/cadena-causal`
> el 2026-07-10 (los números de línea son de esa lectura).

## 0. El defecto, verificado en código

`runtime/life/kernel.py:86`:

```python
self.run_id = self.config.run_id or f"life-{uuid4().hex[:12]}"
```

`self.run_id` es **la fuente única** de tres ejes de identidad que la ley trata como
distintos. De ahí salen, sin separación:

| Eje conceptual | Sitio | Código |
|---|---|---|
| `organism_id` (genoma) | `kernel.py:247` | `runner.set_organism_id(self.run_id)` |
| `organism_id` (herida no-actuante) | `kernel.py:387` | `build_experience(organism_id=self.run_id, …)` |
| `organism_id` (reflexión del maestro) | `kernel.py:421` | `self._teacher.reflect(organism_id=self.run_id, …)` |
| `state_id` (génesis) | `kernel.py:683` | `state_id=f"state-0-{self.run_id}"` |
| `lineage_id` (génesis, en IdentityState) | `kernel.py:687` | `IdentityState(lineage_id=f"lineage-{self.run_id}")` |
| `lineage_id` (génesis, en LineageState) | `kernel.py:689` | `LineageState(lineage_id=f"lineage-{self.run_id}")` |

En restore (`kernel.py:668`, `_apply_restored_identity`) el `run_id` **se hereda** del
checkpoint (`self.run_id = restored.run_id`); en génesis (`kernel.py:681-707`) se
**acuña nuevo** a partir del `run_id`.

**Consecuencia (la que funda el paquete):** como el `organism_id` que se persiste ES el
`run_id`, y el `run_id` por config default es efímero (`life-{uuid4}`), la identidad del
organismo **muere con la corrida**. No hay ADN portable entre ejecuciones ni entre
máquinas. La única razón por la que la experiencia cross-vida "funciona" en `aeon-01` es
que allí el `run_id` se fija a mano y es estable — lo dice el propio código en
`runtime/world/scenario_runner.py:128`: *"El namespace es organism_id (cross-vida);
default = run_id (estable en aeon-01)"*. Esa estabilidad es un accidente de operación, no
una garantía del organismo.

### Dato de alcance (verificado — reduce el trabajo)

El **subsistema de experiencia ya está diseñado para identidad persistente**:

- `runtime/organism/experience.py:214` escribe con
  `run_id=exp.organism_id  # namespace por organismo, no por corrida`.
- `recall` (`experience.py:229-251`) y `wisdom` (`experience.py:253-274`) reciben
  `organism_id` y son **cross-vida por diseño**.
- El test `tests/organism/test_experience.py:94`
  (`test_cross_life_recall_same_organism_id`) prueba que dos vidas con `run_id` distinto
  y **mismo `organism_id`** comparten experiencia.

Es decir: **el defecto no está en el subsistema de experiencia**, sino en que el KERNEL le
entrega `self.run_id` como `organism_id`. B41 no reescribe la experiencia; le da un
`organism_id` genuinamente persistente para que consuma.

### Anclaje a la ley (cúspide f2.4)

- **A-M8 (Herencia como medida).** Los linajes `μ_t ∈ P(Z)` son estables cuando un motivo
  *"reaparece viable entre semillas, entornos y **corridas**"* (`Z_stable`). Sin una
  identidad que persista **entre corridas**, `Z_stable` no es computable: no hay eje bajo
  el cual medir "reaparición". Hoy cada corrida es un organismo nuevo ⇒ nada reaparece.
- **A-M5 / D6 (Continuidad identitaria `C^cont`).** `C^cont_t` compara `Σ_t, G_t, M_t`
  contra `t−1` con un término `𝟙[rollback recuperable]`. La memoria `M_t` solo es
  "la del mismo organismo" si el namespace de `M` es el `organism_id` persistente; con el
  `run_id` efímero, `M_t` es un namespace nuevo cada corrida y la continuidad de memoria
  se rompe silenciosamente en el arranque.
- **A-M10 (Existencia por continuidad).** *"Una discontinuidad identitaria irrecuperable
  (sin rollback restaurable) es muerte operativa."* Reiniciar el proceso hoy es,
  literalmente, muerte del organismo: su ADN no viaja.
- **D7 (Morfogénesis).** La reescritura admisible *"reescribe al mismo organismo"* — exige
  un `organism_id` que sobreviva a la reescritura, distinto del `lineage_id` que puede
  bifurcar.

---

## 1. Separación de ejes de identidad

Tres identificadores distintos, con ciclos de vida propios. Regla rectora: **el `run_id`
deja de ser fuente de identidad persistente; pasa a ser solo la marca de la ejecución.**

### 1.1 `run_id` — la corrida (efímero, operativo)

- **Qué es:** el proceso/ejecución actual. Sirve para trazas, eventos y depuración de
  *esta* corrida. NO es identidad del organismo.
- **Génesis:** acuñado por proceso, `run_id = f"life-{uuid4().hex[:12]}"` (igual que hoy,
  `kernel.py:86`).
- **Restore:** **decisión para ratificar (§1.4).** Recomendación: acuñar `run_id` NUEVO en
  cada proceso (efímero de verdad), en vez de heredarlo del checkpoint como hoy
  (`kernel.py:668`). Fallback conservador: mantener la herencia actual.
- **Namespace que gobierna:** eventos (`storage.append_event(run_id=…)`), trazas de
  razonamiento, telemetría — todo lo que es "lo que pasó en ESTA corrida".

### 1.2 `organism_id` — el genoma (persiste entre corridas)

- **Qué es:** la identidad del organismo. ADN portable: sobrevive al proceso, viaja entre
  máquinas, es el namespace de **memoria viva y experiencia** (`M_t`).
- **Génesis (tres orígenes, en orden de precedencia):**
  1. **Config / entorno explícito** — si se provee `LifeKernelConfig.organism_id` (o
     `RNFE_ORGANISM_ID`), se **vincula** a ese organismo (resume/bind de un genoma
     conocido; p. ej. `aeon-01`). Esto reemplaza el hack actual de fijar `run_id` a mano.
  2. **Herencia de ancestro** — si se provee `parent_organism_id` (fork con descendencia),
     el nuevo organismo hereda el `lineage_id` del ancestro y acuña `organism_id` propio
     (ver §1.3 y D8).
  3. **Génesis genuina** — si no hay nada de lo anterior, se **acuña un genoma nuevo**:
     `organism_id = f"org-{uuid4().hex}"`. Genuinamente nuevo, NO derivado del `run_id`, de
     modo que sobreviva a la corrida.
- **Restore:** se **hereda del checkpoint** (igual mecánica que `run_id` hoy). El
  checkpoint payload gana un campo `organism_id`; al restaurar, `self.organism_id =
  restored.organism_id`. Compat hacia atrás en §4.
- **Namespace que gobierna:** experiencia (`experience.py:214`), memoria viva, y el
  descubrimiento de checkpoints del organismo (§3).

### 1.3 `lineage_id` — el linaje evolutivo (abarca múltiples organismos)

- **Qué es:** el linaje `μ_t` (A-M8/D8). Puede contener **varios `organism_id`** con
  relación de descendencia (un ancestro y sus forks morfogénicos). `organism_id ⊆
  lineage_id` como relación muchos-a-uno.
- **Génesis:**
  - Génesis genuina ⇒ `lineage_id = f"lin-{uuid4().hex}"` (linaje nuevo con un solo
    organismo).
  - Fork de ancestro ⇒ **hereda** el `lineage_id` del `parent_organism_id` (el linaje
    continúa; nace un organismo hermano/hijo dentro de él).
- **Restore:** se hereda del checkpoint (hoy `LineageState` ya se serializa en el payload,
  `checkpoints.py:58`; hay que asegurar que el `lineage_id` NO se re-derive del `run_id`).
- **Namespace que gobierna:** la medida de linajes `μ_t`, la ley D8 (reaparición viable),
  la promoción/herencia (C-AC4).

### 1.4 Decisión abierta para ratificación: ¿`run_id` efímero o heredado en restore?

Hoy `run_id` se hereda en restore (`kernel.py:668`) **porque** debe hacer de
`organism_id`. Una vez separado, hay dos caminos:

- **(A) `run_id` efímero por proceso (recomendado).** Cada arranque = un `run_id` nuevo.
  Semántica correcta: "los eventos de esta corrida". Implica que el descubrimiento de
  checkpoints deje de basarse en `run_id` (hoy `load_latest_payload(run_id=None)` lista
  todos, `checkpoints.py:95-108`) y pase a basarse en `organism_id` (§3).
- **(B) `run_id` heredado (conservador).** Mantiene la mecánica actual; menos cambios,
  pero perpetúa la ambigüedad "corrida vs. organismo" en eventos.

**Recomendación:** (A), porque es la que honra A-M8 (el eje de "corridas" solo tiene
sentido si `run_id` es genuinamente por-corrida) y hace explícito que la persistencia vive
en `organism_id`. Se ratifica en revisión.

---

## 2. `CausalContext.v1` — el sobre de correlación de auditoría

### 2.1 Problema que resuelve

Hoy la cadena **decisión → episodio → traza → certificado → promoción** solo se puede
reconstruir por `run_id` + timestamps. Con `run_id` efímero y múltiples organismos por
máquina, eso es frágil e insuficiente. `CausalContext` da un **sobre de correlación
estable** que viaja con cada unidad de trabajo y permite reconstruir la cadena **sin
depender de timestamps**.

### 2.2 Contrato (campos)

Contrato inmutable, `frozen`, versionado. Propuesta de ubicación:
`runtime/life/contracts.py` (junto a `AutonomyDecision`, `LifeStepResult`) o un módulo
`runtime/observability/causal_context.py` — a decidir en implementación.

```python
@dataclass(frozen=True, slots=True)
class CausalContext:
    schema_version: str = "causal_context.v1"
    organism_id: str = ""        # genoma persistente (eje estable de auditoría)
    lineage_id: str = ""         # linaje evolutivo (μ_t)
    run_id: str = ""             # corrida efímera (trazabilidad operativa)
    trace_group_id: str = ""     # correlación por episodio/step (la "cadena")
    parent_trace_group_id: str | None = None  # encadena rollback/morfogénesis/fork
    decision_id: str | None = None            # decisión que originó el episodio
    step_index: int = -1
```

- **`organism_id` + `lineage_id`:** el eje ESTABLE. Permite agrupar toda la evidencia de un
  organismo/linaje a través de corridas y máquinas.
- **`trace_group_id`:** acuñado **una vez por episodio/step** (p. ej.
  `f"tg-{organism_id}-{step_index}-{uuid4().hex[:8]}"`). Es el hilo que ata decisión,
  episodio, trazas, certificado, memoria y promoción de ESE step.
- **`parent_trace_group_id`:** cuando un step nace de otro (rollback a un yo sano
  `kernel.py:592-611`, refugio E5, o una morfogénesis D7 que reescribe), apunta al
  `trace_group_id` padre ⇒ el árbol causal se reconstruye sin timestamps.
- **`decision_id`:** ya existe en `AutonomyDecision.decision_id` (`kernel.py:577`); se
  copia al sobre para cerrar decisión→episodio.

### 2.3 Dónde se inyecta y cómo viaja

`CausalContext` se **acuña una vez por step**, al comienzo de `LifeKernel.step()`
(`kernel.py:143`), a partir de `self.organism_id`, `self.lineage_id`, `self.run_id`,
`self.total_steps` y (tras decidir) `decision.decision_id`. Luego viaja como un único
objeto:

| Destino | Sitio hoy | Cómo viaja |
|---|---|---|
| Eventos de vida | `append_event(payload={…})` en `kernel.py:299-314, 623-634, 698-707` | clave `"causal_context": ctx.to_dict()` en el payload |
| Runner / episodio | `runner.run_episode(...)` (`kernel.py:249`) | `runner.set_causal_context(ctx)`; el runner usa `trace_group_id` como/junto a `episode_id` |
| Trazas de razonamiento | `append_reasoning_trace(...)` (via runner/scheduler) | el runner propaga `trace_group_id` al detail de la traza |
| Certificado | certificación de episodio (`scenario_runner.py`, `certification["certificate"]`) | `trace_group_id` + `organism_id` en metadata del certificado |
| Memoria / experiencia | `experience.record → write_memory_record(metadata=…)` (`experience.py:213-224`) | añadir `trace_group_id` al `metadata` (hoy ya lleva `run_id`, línea 223) |
| Promoción / herencia | `PromotionGate` (`scenario_runner.py:139`) | el registro de promoción copia `organism_id` + `trace_group_id` |

**Reconstrucción de la cadena (objetivo):** dado un `organism_id`, agrupar por
`trace_group_id` reconstruye cada episodio completo; seguir `parent_trace_group_id`
reconstruye el árbol de rollbacks/forks — todo **sin ordenar por timestamp**.

### 2.4 Compatibilidad

`CausalContext` es **aditivo**: se inyecta como una clave nueva en payloads/metadata
existentes. Ningún consumidor actual lee esa clave, así que con la feature apagada o el
campo ausente el comportamiento es byte-idéntico. Los eventos viejos sin `causal_context`
siguen leyéndose (campo opcional).

---

## 3. Mapa de impacto exhaustivo

Cada sitio que hoy deriva identidad de `run_id` y debe pasar a `organism_id`. Formato:
**qué cambia · qué se rompe si no se cuida · test que lo cubre.**

### 3.1 `kernel.py:86` — la fuente única
- **Cambia:** además de `self.run_id`, el `__init__` establece `self.organism_id` y
  `self.lineage_id` (con la lógica de origen de §1.2/§1.3). El `run_id` deja de ser fuente
  de identidad.
- **Se rompe si no se cuida:** si `organism_id` sigue derivándose de `run_id`, no se logra
  nada.
- **Test:** nuevo `test_genesis_mints_distinct_identity_axes` (los tres IDs son distintos);
  `tests/life/test_life_kernel.py`.

### 3.2 `kernel.py:247` — namespace de experiencia del runner
- **Cambia:** `runner.set_organism_id(self.organism_id)` (no `self.run_id`).
- **Se rompe:** con `self.run_id`, la experiencia se sigue namespaceando por corrida ⇒
  cross-vida solo funciona por el accidente `aeon-01`.
- **Test:** `tests/organism/test_experience.py:94` (ya prueba el cross-vida por
  `organism_id`) + nuevo test de integración kernel: dos `LifeKernel` con mismo
  `organism_id` y distinto `run_id` comparten experiencia.

### 3.3 `kernel.py:387` y `kernel.py:421` — herida no-actuante y reflexión del maestro
- **Cambia:** `build_experience(organism_id=self.organism_id, run_id=self.run_id, …)` y
  `self._teacher.reflect(organism_id=self.organism_id, …)`. Notar que `build_experience`
  ya distingue ambos parámetros (`experience.py:190-199`): `organism_id` para namespace,
  `run_id` para procedencia — hoy se les pasa el mismo valor.
- **Se rompe:** las heridas de cuarentena/rollback quedan en el namespace de la corrida.
- **Test:** cubierto por `tests/life/test_life_kernel.py` (rama no-actuante) + assert de que
  la herida es recuperable con el `organism_id` en una corrida posterior.

### 3.4 `kernel.py:681-707` — génesis (`state_id`, `lineage_id`)
- **Cambia:** `state_id`, `IdentityState.lineage_id` y `LineageState.lineage_id` dejan de
  derivar de `run_id`. `lineage_id` sale de `self.lineage_id` (§1.3); `state_id` puede
  seguir siendo local al organismo pero anclado a `organism_id`
  (`f"state-0-{self.organism_id}"`). El evento `life.genesis` (`kernel.py:698-707`) suma
  `organism_id` al payload.
- **Se rompe:** si `lineage_id` sigue con `run_id`, D8 no puede agrupar linaje entre
  corridas.
- **Test:** `tests/organism/test_lineage.py` + `test_genesis_mints_distinct_identity_axes`.

### 3.5 `kernel.py:651-679` — restore (`_restore_initial_identity` / `_apply_restored_identity`)
- **Cambia:** `_apply_restored_identity` (`kernel.py:668`) además de `self.run_id` fija
  `self.organism_id = restored.organism_id` y `self.lineage_id = restored.lineage_id`. Si
  se adopta §1.4-(A), `self.run_id` se re-acuña por proceso en vez de heredarse.
  `_restore_initial_identity` (`kernel.py:652`) pasa a buscar por `organism_id`
  (config/entorno) en vez de por `config.run_id`.
- **Se rompe:** si el restore no lee `organism_id` del payload, cada restore arranca un
  organismo nuevo ⇒ pérdida de identidad (el defecto que resolvemos).
- **Test:** `tests/life/test_life_kernel.py:80` (`test_life_kernel_resurrects_latest_identity`)
  debe seguir verde y ganar un assert `second.organism_id == first.organism_id`.

### 3.6 `runtime/life/persistence.py:17-43` — `load_latest_identity`
- **Cambia:** la firma pasa a aceptar `organism_id` (además o en vez de `run_id`);
  `RestoredIdentity` (contrato en `runtime/life/contracts.py`) gana un campo
  `organism_id`, poblado desde `payload.get("organism_id")` con el fallback de §4.
- **Se rompe:** hoy reconstruye solo `run_id` (`persistence.py:34`); sin `organism_id`, la
  identidad restaurada no tiene genoma.
- **Test:** `tests/life/test_life_kernel.py:74` (assert de identidad restaurada) + nuevo
  assert sobre `restored.organism_id`.

### 3.7 `runtime/life/checkpoints.py:28-93` — `save_checkpoint`
- **Cambia:** el payload (`checkpoints.py:47-65`) suma `"organism_id"` y `"lineage_id"`
  (hoy solo `"run_id"`, línea 51). **Decisión de namespace de storage:** hoy
  `materialize_artifact` usa `run_id` como segmento de path
  (`runtime/storage/facade.py:184`, `run_segment = run_id or "no-run"`). Para que los
  checkpoints sean del *organismo* (portables, descubribles por genoma), el
  `save_checkpoint` debería materializar con `run_id=organism_id` **o** el descubrimiento
  (§3.8) debe filtrar por el `organism_id` del payload. Recomendación: materializar bajo
  `organism_id` para que el layout en disco refleje "una carpeta por genoma".
- **Se rompe:** si el payload no lleva `organism_id`, el restore no puede heredar identidad
  (círculo con §3.6). Si se cambia el segmento de path sin migración, los checkpoints
  viejos (bajo el `run_id` viejo) quedan huérfanos ⇒ ver §4.
- **Test:** `tests/life/test_life_kernel.py:161`
  (`test_checkpoint_manager_can_load_healthy_checkpoint`) y `:181`
  (`payload["run_id"]`) — sumar assert de `payload["organism_id"]`.

### 3.8 `runtime/life/checkpoints.py:95-121` — `load_latest_payload`
- **Cambia:** el descubrimiento del checkpoint más reciente pasa a filtrar por
  `organism_id` (hoy `run_id=None` lista **todos** los checkpoints de cualquier corrida,
  líneas 104-108, y devuelve el primero). Con múltiples organismos por máquina, "el último
  de cualquiera" es incorrecto; debe ser "el último de ESTE organismo".
- **Se rompe:** sin filtro por organismo, un organismo puede resucitar el checkpoint de
  otro ⇒ contaminación de identidad (viola A-M5/C^cont).
- **Test:** nuevo `test_restore_scopes_to_organism` (dos organismos, cada uno restaura el
  suyo) + los tests de §3.7 verdes.

### 3.9 `runtime/world/scenario_runner.py:131, 156, 335-337, 878` — runner
- **Cambia:** el default `self._organism_id = self.run_id` (`:131`) y el
  `organism_id=f"org-{self.run_id}"` de la trayectoria (`:156`) dejan de derivar del
  `run_id`; el `organism_id` llega **siempre** por `set_organism_id` (`:335`) desde el
  kernel con el genoma real. `set_causal_context` nuevo (§2.3).
- **Se rompe:** el fallback a `run_id` reintroduce el defecto si el kernel olvidara
  inyectar; conviene que `set_organism_id` sea obligatorio o el default sea inseguro-ruido.
- **Test:** `scenario_runner` tests existentes + el de integración cross-vida de §3.2.
  (Nota de scope: `runtime/neural/` NO se toca — es de Codex.)

### 3.10 `runtime/organism/experience.py:213-224` — escritura de memoria
- **Cambia:** **nada estructural** — ya escribe con `run_id=exp.organism_id` (`:214`) y ya
  distingue namespace vs. procedencia (`metadata={"run_id": exp.run_id}`, `:223`). Solo se
  suma `trace_group_id` al `metadata` (§2.3). Este archivo es la **prueba de que el trabajo
  de B41 es del kernel, no de la experiencia.**
- **Se rompe:** nada, si el kernel entrega un `organism_id` correcto.
- **Test:** `tests/organism/test_experience.py:55, 72, 94` (ya cubren record/recall/wisdom
  y cross-vida) — deben seguir verdes sin cambios.

### 3.11 `runtime/storage/facade.py` — namespace de storage
- **Cambia (potencial):** `materialize_artifact` (`:163-197`) usa `run_id` como segmento de
  path (`:184`). `write_memory_record`/`retrieve_memory_records` (`:384-429`) toman `run_id`
  como namespace. **No se cambia la firma** de la facade (es contrato de storage); lo que
  cambia es **qué valor** le pasa el kernel/experiencia: `organism_id` para memoria (ya así
  vía `experience.py`), y `organism_id` para checkpoints si se adopta §3.7.
- **Se rompe:** si se cambia el valor del segmento sin migración, se pierde el
  descubrimiento de lo ya escrito ⇒ §4.
- **Test:** tests de storage existentes (no deben cambiar de firma) + los de §3.7/§3.8.

### 3.12 `runtime/memory/persistence/persistence.py` — fuera de la cadena
- **Verificado:** `StatePreserver` es un backup genérico por `tag` string
  (`persistence.py:71-133`); **no participa** del namespace por `run_id`/`organism_id`. Se
  inspeccionó por pedido del paquete y se descarta del mapa de impacto: no toca la cadena
  causal. (Se reporta para que quede constancia de que se revisó.)

### 3.13 `runtime/organism/state.py` / `runtime/organism/lineage.py` — dónde vive `organism_id`
- **Cambia (a decidir):** `organism_id` necesita un hogar persistente. Opciones:
  (a) campo nuevo en `IdentityState` (`state.py:110-128`, hoy frozen — cambio de
  serialización en `OrganismState.from_dict`, `state.py:272-300`);
  (b) campo top-level en `OrganismState`;
  (c) solo en el checkpoint payload + `self.organism_id` del kernel (mínimo invasivo).
  Recomendación para B41: **(c)** en el payload (como ya vive `run_id`, `checkpoints.py:51`)
  para no tocar el esquema frozen de `IdentityState`; promover a (a) es un follow-up.
- **Test:** serialización round-trip (`OrganismState.to_dict`/`from_dict`) si se elige (a).

### 3.14 Resumen de tests de la suite a correr
`tests/life/test_life_kernel.py`, `tests/organism/test_experience.py`,
`tests/organism/test_lineage.py`, más los nuevos:
`test_genesis_mints_distinct_identity_axes`, `test_restore_scopes_to_organism`,
`test_cross_run_experience_via_organism_id` (integración kernel), y round-trip de
`CausalContext`.

---

## 4. Compatibilidad hacia atrás (sin corrupción de cold-start ni pérdida de memoria)

Existen checkpoints y memoria ya namespaceados por el `run_id` viejo (en `aeon-01`, ese
`run_id` es estable y hace de facto de `organism_id`). La migración NO puede perder eso.

### 4.1 Regla de derivación determinística (identidad, no transformación)

> **Para todo artefacto/memoria pre-B41 sin campo `organism_id`, el `organism_id` legado es
> exactamente el `run_id` bajo el que fue escrito: `organism_id_legacy := run_id`.**

Esto es un **mapeo identidad** (no una transformación con hash), y es correcto porque hoy la
experiencia ya se escribe con `run_id=exp.organism_id` y `organism_id == run_id`
(`experience.py:214`, `kernel.py:247`). Por tanto la memoria existente YA está, de hecho,
namespaceada por el valor que pasará a ser `organism_id`. El mapeo no mueve un solo byte.

### 4.2 Fallback en restore

En `_apply_restored_identity` (`kernel.py:668`) / `load_latest_identity`
(`persistence.py:33-43`):

```
organism_id = payload.get("organism_id") or payload.get("run_id")  # fallback legado
lineage_id  = lineage_from_payload(...).lineage_id or f"lin-legacy-{organism_id}"
```

- Checkpoint **nuevo** (post-B41): trae `organism_id` explícito ⇒ se hereda tal cual.
- Checkpoint **viejo** (sin `organism_id`): `organism_id := run_id` ⇒ el organismo restaura
  su identidad legada, y `retrieve_memory_records(run_id=organism_id)` encuentra su memoria
  vieja intacta.

### 4.3 Descubrimiento de checkpoints legados (§3.8)

Si se materializan checkpoints nuevos bajo `organism_id` (§3.7), pero los viejos viven bajo
el `run_id` viejo (que es igual al `organism_id` legado por §4.1), el path coincide y el
descubrimiento por `organism_id` los encuentra igual. Si se decide no cambiar el segmento
de path (mantener `run_id` en `materialize_artifact`), `load_latest_payload` debe filtrar
por el `organism_id` del **payload**, no por el segmento — cubriendo ambos layouts.

### 4.4 Garantía de cold-start

Un organismo **sin checkpoint previo** (cold start real):
- `_restore_initial_identity` no encuentra nada ⇒ `_genesis` acuña `organism_id` **nuevo**
  (§1.2, origen 3) ⇒ `retrieve_memory_records` bajo ese `organism_id` nuevo devuelve vacío
  ⇒ arranque limpio, **sin** heredar memoria ajena.
- No hay colisión posible entre un `organism_id` genuino (`org-{uuid4}`) y uno legado (un
  `run_id` viejo), porque los prefijos difieren y el uuid es único.

Conclusión: cero pérdida de memoria existente, cero contaminación de cold-start. La
migración es **de campo aditivo + mapeo identidad**, no destructiva.

### 4.5 Coherencia con la política de memoria

Nada de esto altera `MEMORY_COMPATIBILITY_POLICY_v1` (filtro por escenario/régimen): B41
cambia **el eje de namespace de organismo**, ortogonal al filtro de compatibilidad de
escenario. La experiencia sigue siendo `strict_same_scenario` por situación
(`experience.py:46-59`).

---

## 5. Orden de implementación interno y relación con B42-B45

### 5.1 Orden interno de B41

1. **Ejes en el kernel** (§3.1, §3.4): acuñar/heredar `organism_id` y `lineage_id` distintos
   del `run_id`; génesis con los tres orígenes. Sin tocar aún el namespace de escritura.
2. **Persistencia de identidad** (§3.6, §3.7, §4): `organism_id` en el checkpoint payload +
   fallback legado en restore. Esto **da continuidad al genoma** antes de reenrutar nada.
3. **Reenrutado de namespaces** (§3.2, §3.3, §3.5, §3.9): pasar `organism_id` (no `run_id`)
   a experiencia, runner y reflexión; scoping de restore por organismo (§3.8).
4. **`CausalContext.v1`** (§2): sobre de correlación, aditivo, detrás de flag; se inyecta
   una vez asentada la identidad estable.

Racional del orden 1→2→3: si se reenruta el namespace (paso 3) antes de que el checkpoint
persista `organism_id` (paso 2), un restore intermedio perdería el genoma y grabaría bajo
un `organism_id` incoherente. La continuidad se asienta antes de mover el namespace.

### 5.2 Por qué B41 va ANTES de B42/B43

- **B42 (consumo de experiencia):** el organismo lee su experiencia al actuar. Ese consumo
  usa `recall`/`wisdom` **por `organism_id`** (`scenario_runner.py:364, 878`;
  `experience.py:229-274`). Si B42 se cablea mientras `organism_id == run_id` (efímero),
  el organismo consumiría solo la experiencia de la corrida actual — es decir, B42 se
  probaría y "andaría" contra un namespace que B41 va a cambiar. Al aterrizar B41, todo el
  consumo cambiaría de eje ⇒ **rehacés la validación de B42** y arriesgás que la evidencia
  de B42 haya quedado grabada bajo el eje viejo.
- **B43 (etiquetado `closure_passed`):** hoy la severidad de experiencia depende de
  `closure_passed` (`experience.py:69, 85`) y el runner lo etiqueta con
  `bool(va.get("is_viable", True))` (`scenario_runner.py:892`). B43 corrige ese etiquetado.
  Pero `closure_passed` se **graba dentro del `ExperienceRecord`**, que se persiste **bajo
  el `organism_id`** (`experience.py:208-224`). Si B43 corrige el etiquetado antes de que
  B41 fije el `organism_id` correcto, esas etiquetas corregidas quedan escritas en el
  namespace efímero y se pierden/reescriben cuando B41 mueve el eje.

**Principio:** B42 y B43 **graban y consumen bajo la identidad**. Reparar el registro/consumo
antes que la identidad significa hacerlo dos veces (y arriesgar evidencia huérfana). B41
fija primero el eje persistente; B42/B43 operan encima de un namespace ya correcto.

### 5.3 Relación con B44-B45 (dependencia, no alcance de este doc)

`CausalContext` (§2) es el habilitador de la reconstrucción de cadena que B44/B45 (auditoría
de la cadena decisión→certificado→promoción) necesitan. B41 deja el sobre listo y propagado;
B44/B45 lo consumen. No se diseñan aquí — solo se deja la dependencia anotada: **sin
`organism_id` estable + `trace_group_id`, la auditoría de cadena de B44/B45 seguiría atada a
timestamps.**

### 5.4 Secuencia recomendada

```
B41 (identidad: ejes + persistencia + namespaces + CausalContext)
  └─> B42 (consumo de experiencia, sobre organism_id estable)
  └─> B43 (etiquetado closure_passed, grabado bajo organism_id estable)
        └─> B44/B45 (auditoría de cadena, sobre CausalContext)
```

---

## Apéndice A — Decisiones abiertas que requieren ratificación

1. **`run_id` en restore: efímero (A) vs. heredado (B)** — §1.4. Recomendación: (A).
2. **Namespace de storage de checkpoints: migrar a `organism_id` (path) vs. filtrar por
   payload** — §3.7/§3.8/§4.3. Recomendación: filtrar por `organism_id` del payload
   (cubre ambos layouts sin mover archivos legados).
3. **Hogar de `organism_id`: `IdentityState` (a) vs. payload-only (c)** — §3.13.
   Recomendación B41: (c); promover a (a) como follow-up.
4. **Origen de `organism_id` en génesis: precedencia config → ancestro → génesis genuina**
   — §1.2. Confirmar que `aeon-01` se bindea por config (origen 1), reemplazando el hack
   del `run_id` fijo.

## Apéndice B — Anclajes de ley citados

- **A-M5 / D6:** continuidad identitaria `C^cont` (canon §1 L70-72, §2 L116-117).
- **A-M8 / D8:** herencia como medida `μ_t`, `Z_stable` = reaparición viable entre
  semillas/entornos/**corridas** (canon §1 L83-86, §2 L122-126).
- **A-M10:** existencia por continuidad; discontinuidad irrecuperable = muerte operativa
  (canon §1 L92-94).
- **D7 / A-M7 / C-AC3:** morfogénesis reescribe al mismo organismo (canon §2 L119-120,
  §1 L78-81, §4 L174).
- **C-AC4:** gate de continuidad de doble capa (canon §4 L176-177, §5 L185-205).
- **MEMORY_COMPATIBILITY_POLICY_v1:** el eje de organismo es ortogonal al filtro de
  escenario (compat §2, §7).
