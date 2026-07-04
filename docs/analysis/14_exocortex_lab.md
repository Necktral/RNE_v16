# 14 — `exocortex/` (canales/tools/adapters) + `lab/` (validación)

`exocortex/` ~709 LOC, `lab/` ~471 LOC. Mayormente **tooling y validación legacy** alrededor del
Orchestrator antiguo y de H-Net.

## `exocortex/`
- **`adapters/runtime_bridge.py`** (16): solo `load_contract(name)` — lee `contracts/*.schema.json`.
  Bridge minimalista "exocortex↔runtime solo por contratos".
- **`channels/cli/aeon_main_loop.py`** (88): **la entrada CLI legacy** (la raíz `aeon_main_loop.py`
  reexporta aquí). Construye el `Orchestrator` legacy con TensorBoard (`runs/aeon_run0`) y llama
  `run_forever()`. Imprime diagnóstico CUDA/VRAM. Lee `health.temp/vram/entropy`
  (la `HealthStatus` de `core/homeo_controller`). Es el punto de arranque del **AGI loop legacy**
  (entrena sobre datos aleatorios; ver [03_core.md](03_core.md)).
- **`tools/`**: `generate_hnet.py` (206, generación con H-Net — la entrada `generate.py`),
  `event_dashboard_sqlite.py` (116), `aeon_dashboard.py` (89), `plot_core_metrics.py` (85),
  `plot_sensor_data.py` (63), `test_cuda.py` (34). Diagnóstico/visualización.

## `lab/validation/`
- **`fase0_cert.py`** (60): batería "Fase 0" que corre `pytest` y luego `python run_aeon.py …` 5
  veces con distintos flags.
- **`validate_core_existence.py`** (411): validación "de existencia del core" Fase-0, llena de
  **mocks/shims** (`MockBase`, shim logger, `_try_import`). Importa `src.evolution.meta_optimizer`,
  `src.core.epistemic_drift_predictor`; smoke test del core legacy con mucho mock.

## Hallazgos
### [BUG] `fase0_cert.py` invoca `run_aeon.py` inexistente (fase0_cert.py:38-53) + `shell=True`
Llama `python run_aeon.py --cycles … ` cinco veces, pero **no existe `run_aeon.py`** en el repo
(las entradas reales son `aeon_main_loop.py`/`generate.py`). Los pasos 2-5 fallarían. Además es el
**único** sitio con `subprocess.run(..., shell=True)` del repo (riesgo de shell, aquí con cadenas
fijas, pero mala práctica). Script roto/desactualizado.

### [DISEÑO] `aeon_main_loop` arranca el AGI loop legacy
La entrada CLI histórica levanta el `Orchestrator` (prototipo que entrena sobre ruido). No es el
pipeline de benchmarks vivo (ese se lanza vía `scripts/` y `runtime.reality.service`).

### [DISEÑO] `validate_core_existence` es validación mock-pesada del core legacy
Valida importabilidad/ejecución del core antiguo con clases mock; mide "existencia", no corrección
real del pipeline cognitivo actual.

### exocortex/tools mezcla vivo y legacy
`generate_hnet` (H-Net, terceros), dashboards SQLite (leen el ledger), plotting de métricas/sensores
(del loop legacy), `test_cuda` (diagnóstico). Útiles como utilidades, sin lógica de organismo.

## Veredicto
Capa de **interfaz/tooling**: un bridge de contratos minimalista, la entrada CLI del AGI legacy,
herramientas de diagnóstico/visualización y validadores Fase-0. El defecto concreto es
`fase0_cert.py` roto (referencia `run_aeon.py` + `shell=True`). El resto es legacy/diagnóstico, no
crítico para el pipeline vivo.
