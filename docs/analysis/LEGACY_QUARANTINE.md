# Cuarentena del legacy "AEON FENIX-Δ"

Fecha: 2026-06-10. Parte de la **unificación de las dos arquitecturas** (consolidar
sobre el organismo vivo). El legacy **no se borra** — se **aísla** (cuarentena
reversible) y se demarca, de modo que el organismo vivo (RTCME) sea la **única
arquitectura del camino vivo**.

## Qué es el legacy
El stack del orquestador AGI original "AEON FENIX-Δ": física "cuántica/termodinámica"
decorativa, entrenamiento sobre datos aleatorios, partes rotas y stubs. Solo es
alcanzable desde el **CLI histórico** `exocortex/channels/cli/aeon_main_loop.py`.

## Garantía de aislamiento (verificada)
- Los paquetes vivos (`runtime/{world, reality, organism, reasoning, control/msrc,
  certification, storage, memory, lotf, smg, symbolic}`) **no referencian** el stack
  orquestador legacy (grep vacío sobre `module_orchestrator|training_loop|
  QuantumDistributedTrainer|probabilistic_models|loss_elite|adaptation_controller|
  crisis_router`).
- El **único** acoplamiento vivo→legacy era `runtime/control/__init__.py`, que
  importaba *eager* `AdaptationController`/`CrisisRouter` (→ `core/infrastructure` →
  `pydantic`). **Resuelto**: ahora MSRC se importa eager y esos dos nombres se exponen
  por `__getattr__` perezoso. `import runtime.control` ya **no** carga la cadena legacy
  ni `pydantic` (verificado: `sys.modules` queda limpio).
- `pydantic` se importa en **un único fichero**: `runtime/core/infrastructure.py`
  (legacy). El camino vivo es, por tanto, pydantic-free.

## Inventario de módulos en cuarentena (marcados con cabecera `[LEGACY …]`)

### Orquestador / entrenamiento (prototipo, alcanzable solo vía `aeon_main_loop`)
- `runtime/core/module_orchestrator.py` (`Orchestrator`, `CombinedModel` inline)
- `runtime/core/training/training_loop.py`
- `runtime/core/train.py` (`QuantumDistributedTrainer` — ni cuántico ni distribuido)
- `runtime/core/probabilistic_models.py`, `runtime/core/planner.py`
- `runtime/core/loss.py` (backprop roto, ver abajo), `runtime/core/loss_elite.py`
- `runtime/core/model.py` (`CombinedModel`, `BaseModel` mock)

### Roto / muerto — **eliminado en la reorg 2026-07-01** (recuperable vía git history)
- `runtime/core/data/loader.py` — importaba `aeon_fenix_delta` inexistente (ImportError). **Borrado.**
- `runtime/core/training/trainer_fenix.py` — `DummyEnv` stub incompatible. **Borrado.**
- `runtime/agents/fenix_agent.py` — importaba `.rssm_lite` inexistente (ModuleNotFoundError). **Borrado.**
- `runtime/core/model.py` — `CombinedModel`/`BaseModel` mock, 0 importadores estáticos. **Borrado.**
- `runtime/core/utils.py` — solo `load_config()` (OmegaConf), 0 importadores. **Borrado.**
- `lab/validation/fase0_cert.py` + stub raíz `fase0_cert.py` — invocaban `run_aeon.py` inexistente. **Borrados.**
- `src/aeon_fenix/` — forwarders/stubs solo usados por `trainer_fenix.py`. **Borrado con la capa de shims.**

### Subsistemas legacy
- `runtime/evolution/` (neurogénesis/poda/NAS — algoritmos reales pero acoplados al
  Orchestrator; `EvolutionaryRehabilitationCenter` huérfano).
- `runtime/control/homeostasis/` (termodinámica real por ODE; protocolos de apagado =
  solo logging).
- `runtime/control/adaptation_controller.py`, `runtime/control/crisis_router.py`
  (soporte del training loop legacy; cablean los dos EventBus).

> Nota: `runtime/core/` es **mixto**. Permanecen *vivas y en su sitio* las utilidades
> reutilizadas por el organismo: `event_bus.py`, `event_log_sqlite.py`, `metrics.py`,
> `epistemic_drift_predictor.py`, `orchestration/lifecycle.py`,
> `rssm_lite2.py`, `data/data_normalizer.py`, `scheduler.py`, `sparsity_logger.py`.
> (`episteme.py` resultó legacy-only por grafo de imports — su único importador es
> `loss.py` — y se movió a `runtime/legacy/` en la reorg 2026-07-01.)

## Tipos duplicados → resueltos por la cuarentena
- **HealthStatus**: unificado a un canónico (ver A4 / `contracts/types/aeon_types.py`).
- **EventBus**: el vivo es `core/event_bus.EventBus` (sync, usado por el flujo de
  eventos/storage); `infrastructure.EventBus` (async) queda en la zona legacy.
- **EpistemeMeter**: el vivo es `telemetry/episteme/episteme_meter` (numpy); el de
  `core/episteme.py` (torch) queda legacy.
- **CombinedModel** (×2, `module_orchestrator` + `model.py`): legacy.

## Tests
Los tests que ejercitan el soporte legacy (`tests/regression/test_crisis_router.py`,
`test_adaptation_controller.py`, `test_runner_integration.py`) importan la cadena
legacy → `pydantic`. Se les añadió `pytest.importorskip("pydantic")` al inicio del
módulo, de modo que **saltan con gracia** en entornos sin `pydantic` (en vez de romper
la colección), igual que el patrón `requires_torch/postgres` del `conftest.py`. Se usa
`importorskip` (no un marcador de `pytest_runtest_setup`) porque el fallo es en la
**importación del módulo** (colección), antes de que un marcador pueda saltarlo.

## Reversibilidad
Actualización 2026-07-01 (reorg estructural): los módulos de la sección "Roto / muerto"
se **eliminaron** del árbol (recuperables vía git history); el clúster del orquestador se
**relocalizó físicamente** a `runtime/legacy/` (ver README de ese directorio); la capa de
shims (`src/` + paquetes raíz de forwarding) se **colapsó** a imports directos
`runtime.*`/`contracts.*`. Para "des-cuarentenar" el orquestador basta mover los módulos
de vuelta y revertir los imports del CLI histórico.
