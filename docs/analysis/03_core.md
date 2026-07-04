# 03 — `runtime/core/` (capa legacy "AEON FENIX-Δ")

2913 LOC, 31 archivos. Es la **capa más antigua** del proyecto: el orquestador "AGI" original
(AEON FENIX-Δ) con física "cuántica/termodinámica" decorativa. Conviene separar tres grupos:

1. **Utilidades vivas y reutilizadas** por el resto del runtime:
   `event_bus.py`, `event_log_sqlite.py`, `metrics.py`, `infrastructure.py`
   (`Event`/`EventBus`/`ConfigLoader`/`WorkerPool`), `episteme.py` (`EpistemeMeter`,
   vía `runtime/telemetry/episteme`), `epistemic_drift_predictor.py`, `orchestration/lifecycle.py`.
2. **Stack orquestador prototipo** (alcanzable solo desde el CLI histórico
   `exocortex/channels/cli/aeon_main_loop.py`): `module_orchestrator.Orchestrator`,
   `training/training_loop.py`, `train.QuantumDistributedTrainer`, `probabilistic_models.py`,
   `planner.py`, `loss.py`, `loss_elite.py`, `model.py`.
3. **Roto / muerto**: `data/loader.py`, `training/trainer_fenix.py`.

> El `EpistemeMeter` se importa con `from src.core.event_bus import ...` (vía shim) en vez de
> import relativo: `runtime.core.episteme` → `src.core.event_bus` → alias a
> `runtime.core.event_bus`. Funciona pero es un import circular cosmético.

---

## Hallazgos por severidad

### [BUG] `CompositeLoss.forward` rompe el backprop (loss.py:312-313)
```python
if any([t.requires_grad for t in [pred, target, kl] if isinstance(t, torch.Tensor)]):
    loss = loss.clone().detach().requires_grad_(True)
```
`.detach()` desconecta la pérdida del grafo y crea una hoja nueva. `loss.backward()` **no
propagará gradientes a los parámetros del modelo**. Parece un hack para satisfacer un test que
comprueba `loss.requires_grad`, pero invalida el entrenamiento real con esta clase de compat.

### [BUG] `data/loader.py` no importa (data/loader.py:1)
`from aeon_fenix_delta.data.loader import get_loader, AEONDataset` — `aeon_fenix_delta` (nombre
antiguo del proyecto) **no es un paquete del repo** → `ModuleNotFoundError` al importar. Además el
archivo contiene funciones `test_*` (tests colocados fuera de `tests/`). Módulo muerto/roto.

### [BUG] `trainer_fenix.run_fenix_training` está roto contra el stub actual (training/trainer_fenix.py:20-21)
Hace `DummyEnv(seq_len=…, input_dim=…, batch_size=…)` y luego `.reset().to(device)` + `.step(z)`,
pero el `DummyEnv` actual ([src/aeon_fenix/envs/env_dummy.py](../../src/aeon_fenix/envs/env_dummy.py))
es un stub que **no acepta argumentos**, `reset()` devuelve `{}` (sin `.to()`) y **no tiene
`.step()`**. La función falla en cuanto se ejecuta. Demo no funcional.

### [BUG] `RuntimeRunner` lee un atributo inexistente (orchestration/runner.py:53)
`run_id = getattr(orch, "run_id", None)` — el `Orchestrator` define `self.current_run_id`
(module_orchestrator.py:273), no `run_id`. El `reality_hook` siempre recibe `run_id=None`.

### [RIESGO] `EventBus.emit` hace I/O pesado por cada evento (event_bus.py:24-55)
Cada `emit()` (a) abre/escribe/cierra un fichero JSONL, **y** (b) escribe en la capa de storage
(`get_storage().append_event`), **y** (c) invoca listeners. `EpistemeMeter._check_homeostasis`
emite varios eventos por ciclo → ráfagas de open+write de fichero y escrituras a DB. Errores se
tragan con `print()` (no logging). Usa `datetime.utcnow()` (naive, deprecado).

### [RIESGO] Dos `EventBus` distintos coexisten (event_bus.py vs infrastructure.py:64)
- `event_bus.EventBus` (síncrono, singleton global, persiste a fichero+DB).
- `infrastructure.EventBus` (asíncrono, colas por severidad, sin persistencia).
El `Orchestrator` usa **ambos**: `self.bus` (infrastructure) y `self.event_bus` (global). El
`training_loop` publica eventos por las dos vías (training_loop.py:153-156). Diseño confuso.

### [RIESGO] `WorkerPool.__init__` crea tasks asyncio sin loop garantizado (infrastructure.py:30)
`asyncio.create_task(...)` en el constructor requiere un event loop corriendo. Si se instancia
`WorkerPool()` fuera de contexto async (el `Orchestrator` lo hace en `__init__` síncrono),
lanza `RuntimeError`.

### [RIESGO] El `Orchestrator.__init__` hace efectos colaterales y entrena sobre ruido
- Entrena con datos **aleatorios**: `raw_data = torch.randn(12000, 64)` (module_orchestrator.py:168).
- Escribe `config/data_stats.json` **dentro del constructor** (l.171).
- `training_loop` guarda `torch.save(..., "checkpoints/aeon_{cycle}.pt")` sin crear el dir
  (training_loop.py:173) → `FileNotFoundError` si `checkpoints/` no existe.
- `ConfigLoader` exige `config/config.yaml` existente, sin manejar `FileNotFoundError`
  (infrastructure.py:131).

### [RIESGO] `print()` de depuración en hot paths
- `probabilistic_models._debug_dtype_tree` imprime en **cada** forward (4×/forward).
- `train._distributed_train_step` imprime shapes y GradNorm en **cada** paso, hace `isfinite`
  sobre todos los params por paso, y traga excepciones devolviendo `None`.

### [RIESGO] `hook_manager.timeout` usa SIGALRM (hook_manager.py:29-44)
SIGALRM solo funciona en el hilo principal. Si los hooks corren en un worker thread (WorkerPool),
`signal.signal` lanza `ValueError`. `resolve_execution_order` no detecta ciclos de dependencias
(→ `RecursionError`). El 2º elemento de `actions: List[Tuple[str,int]]` (prioridad) **se ignora**.

### [DISEÑO] Física "cuántica/termodinámica" decorativa
- `episteme._stochastic_entropy_production` = `delta - beta·w + randn()*amp` (ruido gaussiano).
- `episteme._quantum_pid` añade `sin(time()*freq)*amp` (oscilador cosmético).
- `episteme._compute_lyapunov` NO es un exponente de Lyapunov: es el mayor autovalor de `J·Jᵀ`
  del jacobiano de la KL (mal etiquetado).
- `loss_elite.loss_elite_v1_2`: el término "adversarial" es `torch.rand_like(...)·0.01` (ruido);
  `beta` del PID puede ser **negativo** (sin clamp) y la pérdida resta `spec_reg`/`efe` → puede
  ir negativa; `torch.logdet` puede dar NaN si la covarianza no es definida positiva.
- `train.QuantumDistributedTrainer`: ni cuántico ni distribuido (paso MSE single-GPU).

### [DISEÑO] Duplicación de tipos/funciones
- `CombinedModel` definido **dos veces**: inline en module_orchestrator.py:45 y en model.py:5
  (el orquestador no importa el de model.py).
- **Tres** `HealthStatus`: `contracts/types/aeon_types`, `core/homeo_controller.py:5`,
  y el del subsistema homeostasis (a confirmar). Colisión de nombres.
- `eval_loop` duplicado: `core/eval_loop.py` (async) y `Orchestrator.eval_loop` (sync).
- Dos cargadores de config para el mismo `config/config.yaml`: `infrastructure.ConfigLoader`
  (yaml+pydantic) y `utils.load_config` (OmegaConf).

### [DISEÑO] `planner.AEONPlanner` "Q-learning" sin asignación de crédito (planner.py:107-117)
`update_policy(reward)` aplica el **mismo** reward a todas las estrategias y usa `Q` a ambos lados
de la actualización (sin estado siguiente). Todas las `Q` convergen igual → el RL es decorativo.

### [DISEÑO] Constantes de hardware hardcodeadas (metrics.py:5-8)
`MAX_VRAM_GB=8` ("RTX 2070 Super"), `THERMAL_THRESHOLD=85`, `CRITICAL_TEMP=100`. Acopla las
métricas normalizadas a una GPU concreta.

### [DISEÑO] `model.BaseModel` mock en módulo de producción (model.py:18-35)
"Mock avanzado para validación extrema" con `structure_hash` basada en `time.time()`. Código de
test incrustado en un módulo de runtime.

### [DISEÑO] `quarantine_manager`: `release`/`terminate` ponen `status` y luego `del` (quarantine_manager.py:96-104, 123-130)
El cambio de estado es inútil porque el ítem se elimina inmediatamente. Registro solo en memoria
(sin persistencia).

---

## Aspectos positivos
- `orchestration/lifecycle.py`: FSM con tabla de transiciones válidas — limpio y bien diseñado.
- `epistemic_drift_predictor`: lógica de detección de estancamiento (baja varianza) razonable y
  autónoma (aunque usa `print` y `force_mutation` muta `.weights` numpy directamente).
- `data/data_normalizer.py`, `rssm_lite2.py`, `scheduler.py`, `sparsity_logger.py`: utilidades
  pequeñas y correctas.
- `event_log_sqlite.py`: migración legacy `eve_type→event_type` idempotente.

## Veredicto del módulo
`runtime/core/` es **deuda técnica legacy**: el grafo de razonamiento/MSRC/reality (el sistema
"vivo" de los benchmarks) **no depende** del `Orchestrator` ni del training loop de aquí; solo
reutiliza utilidades sueltas (event_bus, metrics, episteme, drift_predictor, infrastructure).
Candidatos a limpieza: `data/loader.py` (roto), `trainer_fenix` (roto), el `BaseModel` mock,
y la unificación de los dos EventBus / tres HealthStatus / dos config loaders.
