# 07 — `runtime/reasoning/` (meta-scheduler + familias + razonador externo)

4390 LOC. El cerebro del pipeline: un **meta-scheduler** que selecciona y ejecuta una secuencia de
**familias de razonamiento** (cierre ABD→ANA→CAU→CTF→DED→PROB + overlays), con una política
adaptativa explicable y un **razonador externo LLM** (OpenThinker via llama.cpp) fuertemente gated.

```
reasoning/
├── context.py                 # build_reasoning_context, resolve_reasoning_mode
├── contracts/family_result.py # FamilyResult + normalize_family_result
├── scheduler_meta/
│   ├── meta_scheduler.py       # MetaScheduler.run (importa familias y ejecuta)
│   ├── policy.py (936)         # select_sequence: régimen, scoring, overlays, validación, fallback
│   ├── family_profiles.py      # perfiles + FamilyAdmissionRecord (gate del ext reasoner)
│   ├── family_metrics.py, context_features.py, budgeting.py, fallbacks.py, metacognition_tracker.py
├── families/                   # abd, ana, cau, ctf, ded, prob, heur, dia_adv, eml_sr, ind, ...
│   ├── ded/engine.py (396)     # motor deductivo REAL con Z3
│   └── ext_open_thinker.py (534) # familia razonador externo LLM
└── external_models/            # config, gating, llama_cpp_client (subprocess llama-cli)
```

---

## Hallazgo principal

### [DISEÑO/DOC] Las familias *core* son stubs no-op; solo unas pocas razonan de verdad
Las familias del cierre canónico **ABD, ANA, CAU, CTF, PROB** (y `ind/nesy/opt/plan/evo_search`)
son **stubs** que devuelven un `state_delta` fijo sin computar nada:
```python
# families/abd/__init__.py
def execute(state):
    return {"family": "ABD", "status": "ok", "state_delta": {"abd_hypothesis": True},
            "confidence": 0.6, "cost": 1.0}
# families/prob/__init__.py → {"prob_calibrated": True}, confidence 0.66
```
El cómputo **real** vive solo en:
- **DED** (`families/ded/engine.py`): SAT booleano genuino con **Z3** (`Solver`, `sat/unsat`,
  `unsat_core`, modelo, literales entailment) sobre la fórmula LOT-F.
- **EXT_OPEN_THINKER**: razonador externo LLM (ver abajo).
- **HEUR**: lee `features` (edge_pressure, uncertainty) y produce un triage.
- **DIA_ADV** (`cognitive_self_challenge`) y **EML_SR** (símbolico): trabajo parcial.

➡️ Implicación: el "cierre triádico" y la "secuencia de razonamiento" que validan
`reality/evaluator` y la constitución son, para las familias core, **andamiaje ceremonial** que
registra que una familia "se ejecutó" sin razonamiento subyacente. El valor cognitivo real proviene
de DED (lógica) y, en laboratorio, del razonador externo. Esto debe tenerse muy presente al
interpretar los benchmarks de "ganancia cognitiva".

---

## Razonador externo (foco de la rama actual) — el código mejor construido del repo

`ext_open_thinker.py` + `external_models/{config,gating,llama_cpp_client}.py`:
- **Parsing ultra-defensivo** (ext_open_thinker.py): `_balanced_json_objects` (matcher de llaves
  con estados string/escape), `_strip_think_blocks` (quita `<think>…</think>`),
  `_looks_like_echoed_prompt` (rechaza eco del prompt), y `_validate_payload_shape` (campos
  requeridos/permitidos, longitudes máximas, ítems de lista, rango de confianza, intervención ∈
  permitidas). Errores con código estructurado (`ExternalReasonerParseError`).
- **Telemetría rica**: latency_s, prompt_tps, generation_tps, prompt_bytes, structured_output_mode,
  grammar/json_schema_used. Degrada a `status="skip"` si el modelo no está configurado.
- **`llama_cpp_client.py`**: envuelve `llama-cli` por **subprocess** (CUDA/CPU), `build_command`,
  `generate` con reintento/fallback de backend, `timeout`, parseo de timings, extracción de texto.
- **`config.py`**: `ExternalReasonerConfig.from_env` + `validation_error` (chequea model/cli/schema
  existen) + `subprocess_env` (LD_LIBRARY_PATH para CUDA).
- **Doble gobernanza del gate**:
  1. `policy.select_sequence` **lanza ValueError** si `ext_open_thinker` aparece en un perfil
     nominal (policy.py:821) → el razonador externo **no puede entrar** en secuencias normales,
     solo vía el perfil de laboratorio `core_plus_external_reasoner_gated_v1`.
  2. `external_models/gating.ExternalReasonerGate`: decide *llamar o no* al modelo según conflicto
     causal/contrafactual, core-risky o historia de correcciones.
  3. `family_profiles.validate_external_reasoner_admission`: exige gate+schema+guard+fallback.

  Esta es una pieza de ingeniería de seguridad notablemente cuidada.

---

## Otros hallazgos

### [DISEÑO] `MetaScheduler` importa familias dinámicamente sin validar `execute` (meta_scheduler.py:192)
`import_module(f"runtime.reasoning.families.{family}")` + `module.execute(state)`. Los nombres
vienen de la política (controlados), pero no hay verificación de que el módulo exponga `execute`
(→ AttributeError si una familia futura no lo define). El `_persist_trace` tiene un buen fallback
para dos firmas de `append_reasoning_trace` (try kwargs / except TypeError → record).

### [DISEÑO] `policy.py` con tres rutas de construcción de secuencia
`adaptive_family_ecology_v2` (la nueva), `_legacy_adaptive_sequence` (legacy) y la no-adaptativa.
Hay solapamiento entre v2 y legacy; mucha lógica de overlays/anclas/validación/fallback (sólida
pero compleja: 936 LOC). Decenas de umbrales mágicos (0.55, 0.45, 0.70…) en scoring/boosts/
activación — explicables pero sin calibración formal.

### [DISEÑO] `config.subprocess_env` hardcodea `/usr/lib/wsl/lib` (config.py:169)
Asume entorno WSL para CUDA (se **añade** a LD_LIBRARY_PATH, así que no rompe fuera de WSL, pero es
específico del entorno del autor).

### [DISEÑO] `family_profiles` ata el gate del ext-reasoner a rutas de benchmark concretas
`EXT_OPEN_THINKER_ADMISSION` referencia
`data/benchmarks/external_reasoner_gain/conflict-repeatability-gated-v1-4x4/…` como evidencia de
admisión. La admisión depende de artefactos en disco (acoplamiento código↔datos de benchmark).

---

## Aspectos positivos
- `MetaScheduler` minimal, totalmente trazable; contrato `FamilyResult` normalizado y robusto.
- `policy.py`: política **explicable** con régimen, floors obligatorios por régimen, validación de
  orden parcial (backbone + anclas de overlay), autocorrección y fallback a perfil seguro.
- **DED real con Z3** (sat/unsat/unsat_core) — el único motor lógico genuino del cierre.
- Razonador externo: la pieza más defensiva y mejor instrumentada del repositorio; triple gate.
- `context.py`/`family_result.py` limpios.

## Veredicto
Arquitectura de scheduling **sofisticada y bien gobernada**, con un razonador externo de calidad
de producción. Pero el **núcleo de familias es mayormente simbólico** (stubs): el sistema orquesta
y certifica un "cierre" cuyo contenido real, fuera de DED y el LLM gated, es nominal. Es el punto
más importante a comunicar sobre la validez de los resultados cognitivos.
