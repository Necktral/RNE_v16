# 08 — `runtime/control/` (MSRC + homeostasis + crisis/adaptación)

4270 LOC. Dos subsistemas muy distintos en madurez:

- **`msrc/` (Multi-Scale Resolution Controller) — VIVO y bien diseñado.** Decide la "escala"
  representacional (p. ej. grid 1x1↔5x5) según demanda cognitiva y presión de VRAM.
- **`homeostasis/` — LEGACY (AEON FENIX), parcialmente stub.** Termodinámica/apagado por crisis;
  único consumidor de `contracts/types/aeon_types` (subsistema aislado).
- **`adaptation_controller.py` / `crisis_router.py` — soporte del Orchestrator legacy.**

---

## MSRC (`runtime/control/msrc/`, ~2.7K LOC) — el subsistema vivo

`MSRCController.step` orquesta: `vram_sampler.sample` → `estimator.estimate` →
`policy_engine.decide` → `transition_manager.execute_action` → `audit_logger`. Contratos limpios
(`contracts.py`): `ScaleSpec`, `ScaleEstimate`, `ProbeResult`, `ScaleAction`, `ScalePolicyState`
(con hysteresis, anti-oscilación, regret), `ScaleDecisionRecord`/`ScaleTransitionRecord` (mapean a
los JSON schemas `msrc_scale_decision`/`msrc_transition_event` — draft-07).

`ScalePolicyEngine.decide` (scale_policy_engine.py:200): step++, decrementa cooldown/lock; si
`regime_v3` usa `RegimeClassifier`; gestiona probes (`fork_probe`/`commit`/`discard`); si no, evalúa
candidatos y hace **selección lexicográfica** (cognitive_gain → viability → meta_cost_penalty) con
upgrade/downgrade gated por evidencia, cooldown y lock anti-oscilación. Tres variantes:
`baseline`/`aggressive`/`regime_v3`.

**Calidad:** sólido, auditable, con estado de hysteresis y registro de regret/oscilación. Es el
subsistema de control real que ejercitan los benchmarks `data/benchmarks/msrc/`.

Observaciones menores:
- **[DISEÑO]** `scale_policy_engine.py` (1256 LOC) duplica buena parte de la lógica entre la ruta
  clásica (`_decide_upgrade`/`_decide_downgrade`) y la `regime_v3` (`_decide_*_regime_v3`).
- **[DISEÑO]** Muchos umbrales/pesos mágicos en estimador y profiles de régimen (sin calibración formal).

---

## Homeostasis (`runtime/control/homeostasis/`, ~1.2K LOC) — legacy, parcialmente no-funcional

### [MUERTO/DISEÑO] Los protocolos de apagado son solo logging (shutdown_logic.py:201-243)
`PhasedShutdown` define protocolos reversibles (OPTIMIZATION/CRITICAL/EMERGENCY) con acciones como
`_reduce_cognitive_load`, `_compress_memory`, `_prune_knowledge_base`, `_freeze_learning`,
`_safe_power_down`. **Todas son stubs que solo hacen `logger.info(...)`** — no reducen carga, no
comprimen memoria, no podan nada. El "gestor de apagado por fases con reversibilidad" no ejecuta
ninguna acción real; es andamiaje. `_validate_dependencies` "verifica" dependencias con un
`logger.info(... OK)` dentro de un `try/except ImportError` que nunca puede dispararse.

### [BUG] Incompatibilidad de tipos `HealthStatus` entre governor y shutdown_logic
- `thermodynamic_governor.HealthStatus` (thermodynamic_governor.py:27-37) define campos
  `memory, energy, entropy, temperature, stability, cognitive_load, temp_forecast, thermal_gradient`.
- `shutdown_logic.evaluate_crisis` (shutdown_logic.py:87) **lee** `health.vram_usage`,
  `health.temperature`, `health.entropy_rate`, `health.power_consumption` — campos de la
  `HealthStatus` de `aeon_types` (importada en shutdown_logic.py:8), **no** los del governor.
Si se pasa `ThermodynamicGovernor.assess_health()` (que devuelve su propia HealthStatus) a
`PhasedShutdown.evaluate_crisis`, falla con `AttributeError` (no existe `vram_usage`/`entropy_rate`/
`power_consumption`). Son dos `HealthStatus` **incompatibles** con el mismo nombre. (Es ya la
**3ª–4ª** definición de `HealthStatus` del repo: aeon_types, core/homeo_controller, este governor.)

### [DISEÑO] `thermodynamic_governor`: física real mezclada con stubs
Modelo térmico **genuino** con `scipy.integrate.solve_ivp` (ley de enfriamiento de Newton:
`dT = (P - (T-T_amb)/R) / C`), entropía desde historial VFE, predicción de trayectoria térmica,
potencia vía pynvml. PERO: `get_thermal_metrics` devuelve ceros, `reset/initiate_cooling/
inject_noise` son stubs de logging, `emergency_shutdown` hace `raise SystemExit`. `pynvml.nvmlInit()`
en import (efecto colateral) y dependencia pesada de `scipy`. Contrasta con la física "cuántica"
cosmética de `core/episteme` — aquí la ODE térmica sí es real.

---

## Adaptación / Crisis (soporte del Orchestrator legacy)
- `adaptation_controller.AdaptationController`: construye contexto (delta_epist, thermal_risk,
  vram_usage normalizados con las constantes de `core/metrics`) y aplica adaptaciones vía
  `auto_mutator.step` + `trainer.apply_adaptation`. Parte del training loop legacy.
- `crisis_router.CrisisRouter`: cablea handlers sobre **los dos** event buses
  (`self.bus`=infrastructure async, `self.global_event_bus`=global sync — ver [03_core.md](03_core.md)),
  y `monitor_vitals` hace polling cada 5s publicando VRAMUsageHigh/ThermalAlert/EntropyMax/
  StabilityLoss. Handlers solo loguean (`_on_crisis_event` etc. son logging).

---

## Veredicto
**MSRC = subsistema serio y vivo** (el control de recursos real, auditable, con anti-oscilación).
**Homeostasis = legacy degradado**: termodinámica real (ODE) pero con protocolos de apagado
puramente decorativos (logging) y un **bug de incompatibilidad de `HealthStatus`** que rompería el
flujo governor→shutdown si se cablease. Está aislado del pipeline vivo, así que el bug no se dispara
hoy, pero el subsistema no haría nada útil en una crisis real. Crisis/adaptación pertenecen al
Orchestrator legacy.
