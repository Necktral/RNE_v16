# 01 — `contracts/` (contratos formales)

Capa base: define los **tipos compartidos** (dataclasses Python) y los **17 JSON Schemas**
que formalizan los registros que circulan por el sistema.

```
contracts/
├── __init__.py                  # docstring vacío
├── types/
│   ├── __init__.py              # re-export: from .aeon_types import *
│   └── aeon_types.py            # 227 LOC — dataclasses del "organismo"
└── *.schema.json               # 17 esquemas
```

---

## 1. `types/aeon_types.py` (227 LOC)

Define el sistema de tipos del organismo: estado termodinámico, salud, estado cognitivo,
umbrales de homeostasis, firma neuronal, etc. Todas las dataclasses son
`@dataclass(frozen=True, slots=True)` (inmutables y compactas en memoria) — buena decisión.

### Inventario
- `ComponentID = NewType('ComponentID', str)` (l.20) — usado en `HealthStatus.critical_components`.
- **Enums:** `SystemMode` (l.22, valores int 1-7), `ComponentStatus` (l.31, valores str).
- **Dataclasses:** `Vector3D` (l.40), `ThermodynamicState` (l.53), `CognitiveState` (l.71),
  `HealthStatus` (l.87), `ActionLog` (l.133), `SystemSnapshot` (l.153), `OntologySpaces` (l.176),
  `HomeostasisThresholds` (l.191), `NeuralSignature` (l.212).

### Hallazgos

**[MUERTO] Imports muertos masivos (l.2-17).** Se importan pero nunca se usan:
- `astuple` (l.2) — no usado (`asdict` sí).
- De `typing` (l.3-6): `Union`, `Callable`, `ClassVar`, `TypeVar`, `Generic`, `Protocol`,
  `runtime_checkable` — ninguno se usa. Solo se usan `Dict, List, Tuple, Any, Optional, NewType`.
- `auto` (l.7, `from enum import Enum, auto`) — no usado; los enums usan valores explícitos.
- `StrEnum` + su fallback completo (l.8-12) — **nunca se usa**; `ComponentStatus` es un `Enum`
  plano con valores string, no `StrEnum`. Todo el bloque try/except es código muerto.
- `datetime` (l.15) — no usado.
- `defaultdict` (l.16) — no usado.
- `log` (l.17, `from math import sqrt, log`) — no usado (`sqrt` sí).

**[RIESGO] `Vector3D.normalized()` divide por cero (l.47-49).** No comprueba magnitud nula.
El propio default `ThermodynamicState.thermal_gradient = Vector3D(0,0,0)` (l.60) es un vector
cero; normalizarlo lanza `ZeroDivisionError`.

**[RIESGO] `NeuralSignature.verify()` (l.218-227) tiene dos fragilidades:**
- `min(a/b, b/a)` (l.219-224) divide por cero si cualquier firma de capa vale 0.
- `zip(...)` (l.220-223) trunca a la lista más corta, pero divide entre `len(self.layer_signatures)`
  (l.224). Si `reference` tiene menos capas, el `layer_match` queda **infravalorado** (suma de
  menos términos sobre el conteo de `self`). Resultado de verificación dependiente del orden de los args.

**[DISEÑO] `ThermodynamicState.to_dict()` es lossy (l.62-69).** Omite `heat_dissipation` y
`thermal_gradient`; serializa solo 5 de 7 campos. Si se usa para persistir/telemetría, se pierde info.

**[DISEÑO] `CognitiveState.stability_index` ignora campos (l.79-85).** La fórmula usa
`prediction_accuracy`, `uncertainty`, `memory_load`, pero `attention_focus`, `learning_rate` y
`complexity_level` no entran. Campos definidos sin efecto en el índice de estabilidad.

**[DISEÑO] `HealthStatus.to_dict()` reconstruye de más (l.106-113).** Llama a `asdict(self)`
(copia profunda recursiva de todos los anidados) y luego sobrescribe `thermal_state`,
`cognitive_state`, `system_mode`, `critical_components`. El filtro `if not k.startswith('_')`
(l.108) es un no-op (ningún campo empieza por `_`). Ineficiencia menor + filtro inútil.

**[DISEÑO] `OntologySpaces.total_dimensions()` (l.183-189).** "Dimensiones totales" = suma de
**magnitudes** de 4 `Vector3D` casteada a `int`. Semántica cuestionable (la magnitud de un vector
no es un conteo de dimensiones). `temporal_depth` no participa.

**[DISEÑO] `ComponentStatus` no es `StrEnum` (l.31-38).** Tiene valores string pero como `Enum`
plano: `ComponentStatus.OPTIMAL == "optimal"` es `False`. Si en algún punto se compara contra
strings crudos, fallará silenciosamente.

### Alcance real (live vs. vestigial)
`aeon_types` se importa **solo** desde:
- `runtime/control/homeostasis/{life_monitor, thermodynamic_governor, energy_sensors, homeo_controller, shutdown_logic}.py`
- `runtime/core/homeo_controller.py`

➡️ **Hallazgo arquitectónico:** este sistema de tipos termodinámico/homeostático está
**confinado al subsistema de homeostasis**. NO lo consume el pipeline de razonamiento, MSRC,
reality ni organism. Conviene tenerlo presente: cambios aquí solo afectan a `control/homeostasis`.

---

## 2. Los 17 JSON Schemas

Formalizan los registros del sistema. Mapean (a confirmar al analizar `runtime/storage/records.py`
y `control/msrc/`) con las tablas/eventos persistidos.

| Schema | required clave | Observa |
|---|---|---|
| `artifact_index` | artifact_id, sha256, size, paths | índice de artefactos en disco |
| `certificate` | smg/lotf/world_artifacts, continuity_score, ioc_proxy, risk_score, verdict | certificación de episodio |
| `eml_candidate` | expr, depth, fit/stability/composite_score | candidato de ley simbólica (EML) |
| `eml_run` | top_candidates → `$ref eml_candidate` | corrida EML |
| `episode` | episode_id, context, result | episodio |
| `event` | event_type, payload, timestamp | evento genérico |
| `memory_record` | scale∈{micro,meso,macro}, no_interference, ioc_proxy | registro de memoria multi-escala |
| `msrc_scale_decision` | action.action_type∈{keep/upgrade/downgrade/fork_probe/...}, estimate.* (vram_*) | decisión MSRC |
| `msrc_transition_event` | costes estimados vs reales, ioc_delta, viability_delta, rollback_applied | transición MSRC |
| `proposal` | proposal_id, origin, change, risk | propuesta de cambio |
| `reality_assessment` | closure_passed, continuity_score, trace_integrity, collapse_detected | evaluación de realidad |
| `reasoning_trace` | trace_id, step_index, family, status | traza de razonamiento |
| `rollback` | rollback_id, target, reason | rollback |
| `session_bridge` | session_id, episode_id, channel | puente de sesión |
| `telemetry_snapshot` | snapshot_id, metrics | snapshot de telemetría |
| `tool_request` / `tool_result` | request_id, tool/status, input/output | E/S de herramientas |

### Hallazgos

**[DISEÑO] Dialectos mixtos de JSON Schema.** 15 esquemas usan **draft 2020-12**
(`https://json-schema.org/draft/2020-12/schema`), pero `msrc_scale_decision` y
`msrc_transition_event` usan **draft-07** (`http://json-schema.org/draft-07/schema#`).
Inconsistencia: un validador configurado para un dialecto puede tratar `$ref`/keywords distinto.

**[RIESGO] `eml_run` referencia relativa (l. de `top_candidates`).** `"$ref": "eml_candidate.schema.json"`
es una ruta **relativa de fichero**; solo resuelve si el validador recibe un `base_uri`/store
con esa ruta registrada. Si se valida el esquema aislado, el `$ref` falla.

**[DISEÑO] `additionalProperties: true` en los 17.** Contratos permisivos: aceptan campos extra.
Útil para evolución, pero no detectan typos en nombres de campos (p. ej. `continuity_score`
mal escrito pasa como propiedad adicional válida).

**[DISEÑO] Acoplamiento MSRC.** Las dos `action_type` enum (en `msrc_scale_decision` y
`msrc_transition_event`) están **duplicadas** literalmente. Si se añade una acción, hay que
editar ambos esquemas (y el motor en `control/msrc/scale_policy_engine.py`). Riesgo de drift.

### Consumo
Los esquemas se ejercitan desde `tests/contracts/test_contract_schemas.py` y se referencian en
múltiples scripts/tests (`benchmark_external_reasoner_*`, `validate_core_existence`,
`runtime/reality/{cli,evaluator,msrc_policy_benchmark}.py`, etc.). El cableado exacto del
validador se confirmará al analizar storage/reality.

---

## Resumen del módulo
- Capa de contratos **sana en intención** (dataclasses inmutables + esquemas formales).
- Deuda concreta: imports muertos en `aeon_types.py`, 2 divisiones por cero latentes
  (`Vector3D.normalized`, `NeuralSignature.verify`), `to_dict` lossy, dialectos JSON-Schema mixtos,
  enum `action_type` duplicado entre 2 esquemas.
- Tipos termodinámicos **aislados** al subsistema homeostasis (no son el "contrato central" del
  pipeline; los contratos centrales reales son los JSON schemas de storage/MSRC/reality).
